"""Pluggable per-element loss *terms* for the GLiNER2 structure loss.

A *term* is a pure function ``f(scores, labs, **kw) -> tensor`` that maps the
span logits ``scores`` and the gold target ``labs`` (both of shape
``(gold_count, num_fields, text_len, max_width)``) to a per-element loss tensor
of the **same shape**.  It does NOT reduce, mask, or apply field weights — that
is the job of :mod:`models.losses.patch`, which is shared by every term.

This separation is what makes the loss axis composable for ablation: adding a
new loss = writing one term here + one entry in :data:`LOSS_TERMS`.  No patching
boilerplate, no ``forward`` override, works with any ``--arch`` / ``--backbone``.

Terms accept ``labs`` as the (optionally label-smoothed) soft target and may
read ``labs_hard`` (the un-smoothed 0/1 target) via kwargs when they need a hard
positive/negative switch (Entity-IoU).  Unknown kwargs are ignored (``**_``) so
the dispatcher can pass extras uniformly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss terms
# ---------------------------------------------------------------------------

def bce_term(scores: torch.Tensor, labs: torch.Tensor, *, pos_weight=None, **_):
    """Standard binary cross-entropy (the GLiNER2 default).

    ``pos_weight`` (scalar) up-weights positive spans, pushing the model toward
    higher recall.  ``None`` recovers vanilla BCE.
    """
    pw = None
    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), device=scores.device, dtype=scores.dtype)
    return F.binary_cross_entropy_with_logits(
        scores, labs, reduction="none", pos_weight=pw
    )


def focal_term(scores: torch.Tensor, labs: torch.Tensor, *,
               gamma: float = 2.0, alpha=0.25, **_):
    """Focal loss ``FL = -α_t·(1-p_t)^γ·log(p_t)`` (Lin et al., 2017).

    Down-weights easy (well-classified) spans so training focuses on hard
    positives/negatives.  ``gamma=0`` recovers BCE; ``alpha=None`` disables the
    positive/negative balance factor.  Computed BCE-first for fp16/bf16 safety.
    """
    bce = F.binary_cross_entropy_with_logits(scores, labs, reduction="none")
    probs = torch.sigmoid(scores)
    p_t = labs * probs + (1.0 - labs) * (1.0 - probs)
    loss = (1.0 - p_t) ** gamma * bce
    if alpha is not None:
        a_t = labs * alpha + (1.0 - labs) * (1.0 - alpha)
        loss = a_t * loss
    return loss


def eiou_term(scores: torch.Tensor, labs: torch.Tensor, *,
              alpha: float = 0.25, eniou_map=None, labs_hard=None, **_):
    """Entity IoU loss (Liu et al., Neurocomputing 2022).

    ``positive: -(1-α)·log(p)`` and ``negative: -α·(1+ENIoU)·log(1-p)``.
    Hard negatives (spans overlapping gold entities, high ENIoU) get more
    weight.  Uses ``F.logsigmoid`` (avoids sigmoid→log NaN under half
    precision).  The positive/negative switch uses the **hard** labels.
    """
    if labs_hard is None:
        labs_hard = labs
    if eniou_map is None:
        eniou_map = torch.zeros_like(scores)
    log_p = F.logsigmoid(scores)        # log(p)   ≤ 0
    log_1mp = F.logsigmoid(-scores)     # log(1-p) ≤ 0
    pos_loss = -(1.0 - alpha) * log_p
    neg_loss = -alpha * (1.0 + eniou_map) * log_1mp
    is_positive = (labs_hard > 0.5).float()
    return is_positive * pos_loss + (1.0 - is_positive) * neg_loss


def margin_bce_term(scores: torch.Tensor, labs: torch.Tensor, *,
                    margin: float = 0.15, labs_hard=None, **_):
    """Additive-margin BCE (CosFace/ArcFace-style margin, adapted to sigmoid
    multi-label span scoring instead of the original softmax/cosine setting).

    Subtracts ``margin`` from gold-span logits before the sigmoid, forcing the
    model to push gold scores a margin above the decision boundary rather than
    barely positive. Targets semantically-close label pairs (e.g. OntoNotes
    DATE/TIME, GPE/LOC) that a plain dot/biaffine score leaves under-separated.
    No added parameters. Uses ``labs_hard`` (un-smoothed) as the gold indicator
    so it still composes with label/boundary smoothing on the BCE target.
    """
    if labs_hard is None:
        labs_hard = labs
    adjusted = scores - margin * labs_hard
    return F.binary_cross_entropy_with_logits(adjusted, labs, reduction="none")


def _gold_iou_grid(s: int, w: int, T: int, W: int, device) -> torch.Tensor:
    """IoU of the gold interval ``(s, s+w)`` (end-inclusive, width 0 = 1 word)
    with every ``(start, width)`` cell of the lattice. Same interval convention
    as :func:`compute_eniou_map`."""
    gs, ge = float(s), float(s + w)
    starts = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1).expand(T, W)
    widths = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).expand(T, W)
    ends = starts + widths
    inter = (torch.minimum(ends, ends.new_tensor(ge))
             - torch.maximum(starts, starts.new_tensor(gs)) + 1).clamp(min=0)
    union = (torch.maximum(ends, ends.new_tensor(ge))
             - torch.minimum(starts, starts.new_tensor(gs)) + 1).clamp(min=1)
    return inter / union


def set_consistent_term(scores: torch.Tensor, labs: torch.Tensor, *,
                        margin_set: float = 0.5, lambda_set: float = 1.0,
                        lambda_fp: float = 0.25,
                        margin_label: float = 0.5, lambda_label: float = 0.25,
                        iou_match: float = 0.5, set_topk: int = 0,
                        base_margin: float = 0.0, pos_weight=None,
                        labs_hard=None, valid_mask=None, **_):
    """Set-Consistent loss — train against the evaluator's set-matching view.

    The evaluation protocol (``HungarianEvaluator``) matches the *set* of
    emitted spans against the gold set with a lenient IoU >= 0.5 criterion, and
    the decode pipeline (``resolve_span_conflicts``) emits the highest-scoring
    span of each overlapping cluster. Per-cell BCE never sees either structure;
    this term closes the train/eval gap.

    Per field, the competition acts on the **above-threshold pool**

        T = {c : s_c > 0}                                          (paper Eq. 5)

    i.e. exactly the cells decoding would be willing to emit — no cardinality
    parameter. Each candidate is assigned to the gold whose lenient region owns
    it (max-IoU ownership, ``iou_match`` = the evaluator's tau; many-to-one, so
    multi-gold conflicts resolve to one owner), giving

        O(g) = {c in T \\ G : own(c) = g, IoU_max(c) >= tau}        (Eq. 8)
        F    = {c in T \\ G : IoU_max(c) <  tau}                    (Eq. 9)

    and then:

    * **ownership hinge** — the exact gold cell must outscore the best non-gold
      candidate its region owns by ``margin_set``. That is precisely the
      condition for conflict-resolution decoding to emit the exact boundary
      instead of a lenient near-miss (the measured lenient-strict gap of 4-6
      F1). Uses the evaluator's true lenient region rather than a fixed
      Manhattan radius, on the model's actual above-threshold competitors.
      (A one-to-one Hungarian assignment is deliberately NOT used here: with a
      geometry cost the exact gold cell always wins its own match and the hinge
      would never fire — the decode-consistent signal is region-argmax.)
    * **set-level FP mining** — pooled candidates outside every lenient region
      are precisely the evaluator's false positives, and get an extra
      ``lambda_fp * softplus(logit)`` push below the boundary.
    * **label axis** — at each gold cell,
      the best competing field must trail by ``margin_label``.

    Defining the pool by the threshold rather than by top-K is what makes the
    error decomposition of the paper (Lemma 1) valid: a rival that can displace
    ``g`` scores at least as high as ``s_g > 0``, hence is itself above the
    boundary and therefore *in* the pool. Under a top-K pool that implication
    fails (the rival may fall outside the K highest cells) and the bound with
    it. ``set_topk > 0`` restores the legacy top-K cap as an **ablation knob**
    (it is not part of the objective; use it to reproduce pre-Eq.5 runs).

    ``pos_weight`` up-weights gold cells in the BCE base — the recall knob for
    rare labels (FIN's MISC has 7 training entities against PER's 650).  It acts
    on the base only; the hinges are already relative comparisons.

    ``valid_mask`` (T, W) marks decodable span positions. The FP sum aggregates
    the whole pool into a single anchor cell, so the patcher's per-cell
    ``span_mask`` reduction can no longer reach it — without this mask padding
    spans past the sentence end would be mined as false positives, which with
    an unbounded pool is most of the grid.

    Designed to run on **null-adjusted logits** (scorer ``null_score=True``):
    logit 0 is then the decode boundary "beats the label's null", so golds are
    pushed above their null and FPs below it — threshold-free decoding, trained
    end-to-end. (Works on raw logits too; 0 then means sigmoid 0.5.) Hinge/FP
    contributions are placed at gold cells in the returned map so the patcher's
    random negative masking never drops them; gradients flow to the referenced
    competitor cells regardless of placement. With all lambdas 0 this is
    exactly BCE (or exactly margin-BCE when ``base_margin`` > 0).

    ``base_margin`` swaps the plain-BCE base for the additive-margin BCE of
    :func:`margin_bce_term` (gold logits shifted by ``-base_margin`` before the
    sigmoid) — combining the two best-measured components: the uniform margin
    that won MIT Restaurant and the set-level competition + null anchor that
    won FIN. The hinges still operate on the UNSHIFTED logits (the shift is a
    base-loss construct, not a change to the score surface).
    """
    if labs_hard is None:
        labs_hard = labs
    base_in = scores - base_margin * labs_hard if base_margin > 0 else scores
    pw = (torch.tensor(float(pos_weight), device=scores.device, dtype=scores.dtype)
          if pos_weight is not None else None)
    base = F.binary_cross_entropy_with_logits(base_in, labs, reduction="none",
                                              pos_weight=pw)
    gold = labs_hard > 0.5
    if not bool(gold.any()) or (lambda_set <= 0 and lambda_fp <= 0 and lambda_label <= 0):
        return base

    B, P, T, W = scores.shape
    cap = int(set_topk)
    hinge = torch.zeros_like(scores)
    for b in range(B):
        for p in range(P):
            gcells = gold[b, p].nonzero(as_tuple=False)          # (G, 2)
            if gcells.numel() == 0:
                continue
            x = scores[b, p]

            # ── Pool T = {c : s_c > 0} (Eq. 5) ─────────────────────────────
            # Membership is detached: locally constant in theta (Prop. 1), and
            # the boundary crossings it induces are the downward jumps of
            # Remark 1.  Padding spans are excluded so they cannot be mined.
            pool = x.detach() > 0
            if valid_mask is not None:
                pool = pool & valid_mask
            if cap > 0 and int(pool.sum()) > cap:                # legacy ablation
                flat = x.detach().flatten().masked_fill(~pool.flatten(), float("-inf"))
                keep = torch.zeros(T * W, dtype=torch.bool, device=scores.device)
                keep[flat.topk(cap).indices] = True
                pool = keep.view(T, W)
            idx = pool.nonzero(as_tuple=False)                   # (|T|, 2)

            anchor_s, anchor_w = gcells[0].tolist()              # placement: never masked
            contrib = scores.new_zeros(())

            if idx.numel() > 0 and (lambda_set > 0 or lambda_fp > 0):
                cs, cw = idx[:, 0], idx[:, 1]
                # IoU (G, |T|) between each gold and each pooled candidate.
                iou = torch.zeros(len(gcells), len(cs), device=scores.device)
                for gi, (gs_, gw_) in enumerate(gcells.tolist()):
                    grid = _gold_iou_grid(gs_, gw_, T, W, scores.device)
                    iou[gi] = grid[cs, cw]
                iou_max, owner = iou.max(dim=0)                  # Eq. (7)
                in_region = iou_max >= iou_match                 # lenient criterion
                is_gold_cell = labs_hard[b, p, cs, cw] > 0.5     # T \ G

                if lambda_set > 0:                               # O(g), Eq. (8)
                    for gi, (gs_, gw_) in enumerate(gcells.tolist()):
                        owned = in_region & (owner == gi) & ~is_gold_cell
                        if bool(owned.any()):
                            usurper = x[cs[owned], cw[owned]].max()
                            contrib = contrib + lambda_set * F.relu(
                                margin_set + usurper - x[gs_, gw_]
                            )
                if lambda_fp > 0:                                # F, Eq. (9)
                    fp_mask = (~in_region) & (~is_gold_cell)
                    if bool(fp_mask.any()):
                        contrib = contrib + lambda_fp * F.softplus(
                            x[cs[fp_mask], cw[fp_mask]]
                        ).sum()

            if lambda_label > 0 and P > 1:
                for gs_, gw_ in gcells.tolist():
                    col = scores[b, :, gs_, gw_]
                    competitor = labs_hard[b, :, gs_, gw_] < 0.5
                    if bool(competitor.any()):
                        comp = col.masked_fill(~competitor, float("-inf")).max()
                        contrib = contrib + lambda_label * F.relu(
                            margin_label + comp - scores[b, p, gs_, gw_]
                        )
            hinge[b, p, anchor_s, anchor_w] = contrib
    return base + hinge


def focal_eiou_term(scores: torch.Tensor, labs: torch.Tensor, *,
                    mix: float = 0.5, gamma: float = 2.0,
                    focal_alpha=0.25, eiou_alpha: float = 0.25,
                    eniou_map=None, labs_hard=None, **_):
    """Convex combination ``mix·focal + (1-mix)·eiou`` (per element).

    ``mix`` (the combination weight, NOT the per-term ``alpha``): ``mix=1`` is pure
    focal, ``mix=0`` is pure Entity-IoU.  ``gamma``/``focal_alpha`` tune the focal
    half; ``eiou_alpha``/``eniou_map``/``labs_hard`` tune the eiou half.  Listed in
    :data:`TERMS_NEEDING_ENIOU` so the patcher feeds the ENIoU map (without it the
    eiou half silently degrades to plain log-loss).

    Note the two halves live on different magnitudes, so ``mix=0.5`` is not
    "equal contribution" — treat ``mix`` as a tunable.
    """
    f = focal_term(scores, labs, gamma=gamma, alpha=focal_alpha)
    e = eiou_term(scores, labs, alpha=eiou_alpha, eniou_map=eniou_map, labs_hard=labs_hard)
    return mix * f + (1.0 - mix) * e


# ---------------------------------------------------------------------------
# ENIoU map (hard-negative weighting for eiou_term)
# ---------------------------------------------------------------------------

def compute_eniou_map(labs: torch.Tensor, text_len: int, max_width: int) -> torch.Tensor:
    """Max IoU between every candidate span position and the gold entity spans.

    Returns a tensor shaped like ``labs`` where each entry is the ENIoU of that
    ``(start, width)`` candidate with the nearest gold span of the same field
    (0 where the field has no gold entities).  Vectorised per field.
    """
    gold_count, num_fields = labs.shape[0], labs.shape[1]
    eniou = torch.zeros_like(labs)

    for k in range(num_fields):
        gold_intervals = []  # (start, end) with end = start + width
        for i in range(gold_count):
            positions = (labs[i, k] > 0.5).nonzero(as_tuple=False)  # (N, 2)
            for pos in positions:
                s, w = pos[0].item(), pos[1].item()
                gold_intervals.append((s, s + w))

        if not gold_intervals:
            continue

        gold_starts = torch.tensor(
            [iv[0] for iv in gold_intervals], device=labs.device, dtype=labs.dtype
        )
        gold_ends = torch.tensor(
            [iv[1] for iv in gold_intervals], device=labs.device, dtype=labs.dtype
        )

        start_grid = torch.arange(text_len, device=labs.device, dtype=labs.dtype).unsqueeze(1).expand(text_len, max_width)
        width_grid = torch.arange(max_width, device=labs.device, dtype=labs.dtype).unsqueeze(0).expand(text_len, max_width)
        end_grid = start_grid + width_grid

        cand_starts = start_grid.unsqueeze(-1)  # (T, W, 1)
        cand_ends = end_grid.unsqueeze(-1)
        g_starts = gold_starts.view(1, 1, -1)   # (1, 1, G)
        g_ends = gold_ends.view(1, 1, -1)

        inter_start = torch.max(cand_starts, g_starts)
        inter_end = torch.min(cand_ends, g_ends)
        intersection = (inter_end - inter_start + 1).clamp(min=0)

        union_start = torch.min(cand_starts, g_starts)
        union_end = torch.max(cand_ends, g_ends)
        union = (union_end - union_start + 1).clamp(min=1)

        iou = intersection / union          # (T, W, G)
        max_iou, _ = iou.max(dim=-1)         # (T, W)

        for i in range(gold_count):
            eniou[i, k] = max_iou

    return eniou


# ---------------------------------------------------------------------------
# Registry — the single source of truth for the loss ablation axis
# ---------------------------------------------------------------------------

LOSS_TERMS = {
    "bce": bce_term,
    "focal": focal_term,
    "eiou": eiou_term,
    "focal_eiou": focal_eiou_term,
    "margin": margin_bce_term,
    "set": set_consistent_term,
}

# Terms that consume the ENIoU map → the patcher must compute it for them.
TERMS_NEEDING_ENIOU = {eiou_term, focal_eiou_term}

# Terms whose base must see EVERY cell → the patcher must not apply random
# negative masking to them.  `set_consistent_term` is written as a sum over the
# whole grid (paper Eq. "set-consistent"), and the bound of the appendix is
# stated for that quantity: `models/losses/bounds.py` recomputes `L_set` with a
# full base, so masking here would make the loss that is *minimised* smaller
# than the loss the bound is *evaluated on*, and the two would drift apart with
# no way to see it in the logs.  It also changes what the coverage lemma
# measures: with the base masked for both objectives the gap collapses to
# exactly 2*lambda_own, whereas the full grid adds the sampler's own
# (1 - pi)*sigma(s_c) on top.
TERMS_FULL_GRID = {set_consistent_term}
