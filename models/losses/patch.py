"""Single struct-loss patcher shared by every loss term.

``apply_struct_loss`` monkey-patches ``model.compute_struct_loss`` exactly once,
regardless of which term is used.  It owns everything that is identical across
terms: building ``scores``/``labs``, optional label smoothing, optional
per-field class weights, the original negative-span masking, and the
``span_mask`` reduction.

Crucially it does **not** replace ``model.forward`` (unlike the old wrapper).
Field-name routing — needed only when ``field_weights`` is active — is done via
a lightweight ``forward_pre_hook`` plus a single patched ``_compute_sample_loss``
that mirrors the base implementation and injects the per-schema field names.
Because bi-encoder / two-tower archs override only ``_encode_batch`` (not
``forward`` / ``_compute_sample_loss``), this works uniformly for uni / bi /
bi-encoder-2-encoder.
"""
from __future__ import annotations

import types
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from .terms import TERMS_FULL_GRID, TERMS_NEEDING_ENIOU, compute_eniou_map
from .weights import extract_entity_field_names


# ---------------------------------------------------------------------------
# Shared scores/labels builder (identical to the original struct loss)
# ---------------------------------------------------------------------------

def _set_label(labs: torch.Tensor, i: int, k: int, span, scores: torch.Tensor) -> None:
    start, end = span
    width = end - start
    if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
        labs[i, k, start, width] = 1


def build_scores_and_labels(model, span_rep, schema_emb, structure, token_embs=None):
    """Return ``(scores, labs, gold_count)`` exactly as the original loss does.

    Scoring goes through ``model._score_spans`` (not a hardcoded dot-product
    einsum) so that a configured non-dot scorer (biaffine/cosine via
    ``--affine-exp`` / ``model.configure_scorer``) still receives gradient when
    any loss-toolkit modification (boundary smoothing, focal, class weights,
    ...) is also active. ``_score_spans`` defaults to the same
    ``einsum("lkd,bpd->bplk")`` for ``score_mode="dot"``, so this is a no-op
    change for the common case.

    ``token_embs`` is forwarded (as a kwarg, only when present) for
    ``boundary_token`` scorers; passing positionally would break 2-arg
    ``_score_spans`` implementations (test stubs, older archs).
    """
    gold_count = min(structure[0], 19)
    struct_proj = model.count_embed(schema_emb[1:], gold_count)
    if token_embs is not None:
        scores = model._score_spans(span_rep, struct_proj, token_embs=token_embs)
    else:
        scores = model._score_spans(span_rep, struct_proj)

    labs = torch.zeros_like(scores)
    for i in range(gold_count):
        gold_spans = structure[1][i]
        for k, span in enumerate(gold_spans):
            if span is None or span == (-1, -1):
                continue
            if isinstance(span, tuple):
                _set_label(labs, i, k, span, scores)
            elif isinstance(span, list):
                for sub in span:
                    if sub is None or sub == (-1, -1):
                        continue
                    _set_label(labs, i, k, sub, scores)
    return scores, labs, gold_count


# ---------------------------------------------------------------------------
# Boundary smoothing (Zhu & Li, ACL 2022) — adapted to per-field BCE
# ---------------------------------------------------------------------------

def compute_boundary_smoothed_labels(
    labs: torch.Tensor, *, size: int, epsilon: float
) -> torch.Tensor:
    """Boundary-smoothed soft target for GLiNER2's sigmoid/BCE struct loss.

    Faithful to Zhu & Li (ACL 2022) on the **(start, end)** span plane, ported to
    GLiNER2's per-field BCE.  ``labs`` is indexed ``[gold, field, start, width]``
    with ``end = start + width``.  The gold span keeps ``1 - epsilon``; candidate
    spans at Manhattan distance ``d in [1, size]`` on the (start, end) endpoints
    share ``epsilon / size`` per ring, split equally among the valid members of
    that ring.  Unlike the softmax original there is **no** "non-entity"
    redistribution (channels are independent); cells touched by several gold spans
    take the max soft value, and gold cells are never overwritten.
    """
    if epsilon <= 0.0 or size <= 0:
        return labs

    gold_count, num_fields, T, W = labs.shape
    smoothed = labs * (1.0 - epsilon)  # gold cells -> 1-ε, the rest stay 0

    for i in range(gold_count):
        for k in range(num_fields):
            golds = (labs[i, k] > 0.5).nonzero(as_tuple=False)  # (N, 2): (start, width)
            if golds.numel() == 0:
                continue
            for pos in golds:
                s = int(pos[0]); w = int(pos[1]); e = s + w
                for d in range(1, size + 1):
                    ring = []  # valid cells at Manhattan distance exactly d on (start, end)
                    for ds in range(-d, d + 1):
                        de_abs = d - abs(ds)
                        for de in ((-de_abs, de_abs) if de_abs > 0 else (0,)):
                            ns, ne = s + ds, e + de
                            nw = ne - ns
                            if ns < 0 or nw < 0 or ns >= T or nw >= W:
                                continue
                            if labs[i, k, ns, nw] > 0.5:  # never overwrite a gold span
                                continue
                            ring.append((ns, nw))
                    if not ring:
                        continue
                    share = (epsilon / size) / len(ring)
                    for (ns, nw) in ring:
                        if share > float(smoothed[i, k, ns, nw]):
                            smoothed[i, k, ns, nw] = share
    return smoothed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_struct_loss(
    model,
    term: Callable,
    *,
    term_kwargs: Optional[Dict[str, Any]] = None,
    field_weights: Optional[Dict[str, float]] = None,
    label_smoothing: float = 0.0,
    boundary_smoothing: float = 0.0,
    boundary_smoothing_size: int = 1,
    dynamic_field_weight: bool = False,
    max_class_weight: float = 10.0,
    full_grid: Optional[bool] = None,
) -> None:
    """Patch ``model.compute_struct_loss`` to use ``term``.

    Args:
        term: A loss term from :mod:`models.losses.terms` (or compatible).
        term_kwargs: Extra kwargs forwarded to ``term`` (e.g. ``gamma``, ``alpha``,
            ``pos_weight``).
        field_weights: Static per-label weight dict.  When non-empty, a field-name
            router is installed so each field is multiplied by its label weight.
        label_smoothing: ε for ``y' = y(1-ε) + ε/2`` (applied to the soft target;
            terms that need hard labels still get them via ``labs_hard``).
        boundary_smoothing: ε for boundary smoothing (Zhu & Li, 2022); spreads
            probability to spans near the gold boundary instead of uniformly.
            Mutually exclusive with ``label_smoothing``.
        boundary_smoothing_size: D, the Manhattan radius for boundary smoothing.
        dynamic_field_weight: Legacy per-sample inverse-frequency field weighting,
            used only when ``field_weights`` is empty.  Kept for backward compat.
        max_class_weight: Clamp for the legacy dynamic weighting.
    """
    term_kwargs = dict(term_kwargs or {})
    fw = field_weights or {}
    use_static_fw = len(fw) > 0
    needs_eniou = term in TERMS_NEEDING_ENIOU
    # `full_grid=True` forces the base over every cell even for a term the
    # registry does not list -- the control condition "full-grid BCE" needs
    # exactly the set-consistent sampling policy with none of its extra terms.
    full_grid = (term in TERMS_FULL_GRID) if full_grid is None else bool(full_grid)
    _ls = float(label_smoothing)
    _bs = float(boundary_smoothing)
    _bs_size = int(boundary_smoothing_size)
    if _ls > 0.0 and _bs > 0.0:
        raise ValueError(
            "label_smoothing and boundary_smoothing are mutually exclusive; set only one."
        )

    model._loss_current_field_names = None

    def _patched_compute_struct_loss(
        self,
        span_rep: torch.Tensor,
        schema_emb: torch.Tensor,
        structure: List[Any],
        span_mask: torch.Tensor,
        masking_rate: float = 0.5,
        token_embs: torch.Tensor = None,
    ) -> torch.Tensor:
        scores, labs, _ = build_scores_and_labels(
            self, span_rep, schema_emb, structure, token_embs=token_embs
        )

        # Soft (possibly smoothed) target for the term; hard target preserved.
        labs_soft = labs
        if _bs > 0:
            labs_soft = compute_boundary_smoothed_labels(labs, size=_bs_size, epsilon=_bs)
        elif _ls > 0:
            labs_soft = labs * (1.0 - _ls) + _ls * 0.5

        kw = dict(term_kwargs)
        kw["labs_hard"] = labs
        # Valid-span mask (T, W).  Terms that aggregate over a *set* of cells
        # (set-consistent FP mining) must drop padding spans themselves: they
        # place the aggregate at one anchor cell, which the per-cell span_mask
        # reduction at the end of this function can no longer reach.
        n_cells = scores.shape[2] * scores.shape[3]
        if span_mask is not None and span_mask.numel() >= n_cells:
            kw["valid_mask"] = (~span_mask.reshape(-1)[:n_cells]).view(
                scores.shape[2], scores.shape[3]
            )
        if needs_eniou:
            kw["eniou_map"] = compute_eniou_map(labs, scores.shape[2], scores.shape[3])

        loss = term(scores, labs_soft, **kw)

        # ── Per-field class weight (static names, else legacy dynamic) ──────
        num_fields = scores.shape[1]
        if use_static_fw and self._loss_current_field_names is not None:
            names = self._loss_current_field_names
            w_list = [
                fw.get(names[k], 1.0) if k < len(names) else 1.0
                for k in range(num_fields)
            ]
            field_w = torch.tensor(w_list, device=scores.device, dtype=scores.dtype)
            loss = loss * field_w[None, :, None, None]
        elif dynamic_field_weight:
            pos_per_field = (labs > 0.5).float().sum(dim=[0, 2, 3])
            has_pos = pos_per_field > 0
            total_pos = pos_per_field.sum()
            if has_pos.any() and total_pos > 0:
                n_active = has_pos.float().sum()
                field_w = torch.where(
                    has_pos,
                    torch.clamp(total_pos / (pos_per_field * n_active), min=0.5, max=max_class_weight),
                    torch.ones_like(pos_per_field),
                )
                loss = loss * field_w[None, :, None, None]

        # ── Negative-span masking (identical to the original) ──────────────
        # ... except for a full-grid term, where the base is *defined* over
        # every cell (see TERMS_FULL_GRID): masking it would optimise a
        # different functional from the one the appendix bounds.
        if full_grid:
            masking_rate = 0.0
        if masking_rate > 0.0 and self.training:
            negative = labs < 0.5
            to_mask = negative & (torch.rand_like(scores) < masking_rate)
            loss_mask = (~to_mask).float()
        else:
            loss_mask = torch.ones_like(scores)

        loss = loss * loss_mask
        loss = loss.view(loss.shape[0], loss.shape[1], -1) * (~span_mask[0]).float()
        return loss.sum()

    model.compute_struct_loss = types.MethodType(_patched_compute_struct_loss, model)

    if use_static_fw:
        _install_field_name_router(model)


# ---------------------------------------------------------------------------
# Field-name router (forward pre-hook + patched _compute_sample_loss)
# ---------------------------------------------------------------------------

def _install_field_name_router(model) -> None:
    """Route per-schema field names into ``compute_struct_loss`` WITHOUT
    replacing ``model.forward``.

    A forward pre-hook stashes ``batch.schema_tokens_list`` and resets a sample
    counter; the patched ``_compute_sample_loss`` (mirroring the base impl)
    reads the current sample's schema tokens by order and sets
    ``_loss_current_field_names`` around each ``compute_struct_loss`` call.
    """
    # Remove a previously installed hook so re-patching does not stack.
    if getattr(model, "_lw_prehook_handle", None) is not None:
        try:
            model._lw_prehook_handle.remove()
        except Exception:
            pass

    model._lw_schema_tokens = None
    model._lw_sample_idx = 0

    def _pre_hook(module, args):
        batch = args[0] if args else None
        module._lw_schema_tokens = getattr(batch, "schema_tokens_list", None)
        module._lw_sample_idx = 0

    model._lw_prehook_handle = model.register_forward_pre_hook(_pre_hook)

    def _patched_compute_sample_loss(
        self,
        token_embeddings,
        embs_per_schema,
        task_types,
        structure_labels,
        device,
        span_info=None,
    ):
        # Schema tokens for THIS sample (base forward calls us once per sample
        # in order, so the counter aligns 1:1 with batch.schema_tokens_list).
        sample_tokens = None
        if self._lw_schema_tokens is not None and self._lw_sample_idx < len(self._lw_schema_tokens):
            sample_tokens = self._lw_schema_tokens[self._lw_sample_idx]
        self._lw_sample_idx += 1

        cls_loss = torch.tensor(0.0, device=device)
        struct_loss = torch.tensor(0.0, device=device)
        count_loss = torch.tensor(0.0, device=device)

        if span_info is None:
            has_span_task = any(t != "classifications" for t in task_types)
            if has_span_task and token_embeddings.numel() > 0:
                span_info = self.compute_span_rep(token_embeddings)

        all_counts = []
        all_p_embs = []

        for i, task_type in enumerate(task_types):
            if not embs_per_schema[i]:
                continue
            schema_emb = torch.stack(embs_per_schema[i])

            if task_type == "classifications":
                cls_embeds = schema_emb[1:]
                logits = self.classifier(cls_embeds).squeeze(-1)
                labels = torch.tensor(structure_labels[i], dtype=torch.float, device=device)
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum"
                )
            else:
                structure = structure_labels[i]
                if structure[0] == 0:
                    continue
                if span_info is not None:
                    if sample_tokens is not None and i < len(sample_tokens):
                        self._loss_current_field_names = extract_entity_field_names(sample_tokens[i])
                    else:
                        self._loss_current_field_names = None
                    struct_loss = struct_loss + self.compute_struct_loss(
                        span_info["span_rep"], schema_emb,
                        structure, span_info["span_mask"],
                        token_embs=span_info.get("token_embs"),
                    )
                    self._loss_current_field_names = None
                if task_type != "entities":
                    all_counts.append(min(structure[0], 19))
                    all_p_embs.append(schema_emb[0])

        if all_counts and all_p_embs:
            counts = torch.tensor(all_counts, dtype=torch.long, device=device)
            p_embs = torch.stack(all_p_embs)
            count_loss = F.cross_entropy(self.count_pred(p_embs), counts, reduction="sum")

        return {"classification": cls_loss, "structure": struct_loss, "count": count_loss}

    model._compute_sample_loss = types.MethodType(_patched_compute_sample_loss, model)
