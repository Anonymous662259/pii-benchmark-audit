"""Composable structure-loss toolkit for GLiNER2 ablations.

Design
------
* ``terms.py``   — pure per-element loss functions + the ``LOSS_TERMS`` registry.
* ``patch.py``   — a single ``apply_struct_loss`` that owns masking / field
  weights / smoothing and patches ``compute_struct_loss`` once (no ``forward``
  override → works with any ``--arch`` / ``--backbone``).
* ``weights.py`` — global class weights + schema field-name extraction.

The ``apply_weighted_loss`` / ``apply_focal_loss`` / ``apply_entity_iou_loss``
functions below are thin presets over ``apply_struct_loss`` and preserve the
original public API (so ``training/train.py`` and notebooks keep working).
"""
from __future__ import annotations

from typing import Dict, Optional

from .patch import (
    apply_struct_loss,
    build_scores_and_labels,
    compute_boundary_smoothed_labels,
)
from .terms import (
    LOSS_TERMS,
    bce_term,
    compute_eniou_map,
    eiou_term,
    focal_eiou_term,
    focal_term,
    margin_bce_term,
    set_consistent_term,
)
from .weights import (
    compute_global_class_weights,
    extract_entity_field_names,
)


def apply_loss_by_name(
    model,
    name: str,
    *,
    term_kwargs: Optional[dict] = None,
    field_weights: Optional[Dict[str, float]] = None,
    label_smoothing: float = 0.0,
) -> None:
    """Look up ``name`` in :data:`LOSS_TERMS` and apply it.  The orthogonal entry
    point for the loss ablation axis (``--loss bce|focal|eiou``)."""
    if name not in LOSS_TERMS:
        raise ValueError(f"unknown loss '{name}'; choices: {sorted(LOSS_TERMS)}")
    apply_struct_loss(
        model, LOSS_TERMS[name],
        term_kwargs=term_kwargs, field_weights=field_weights,
        label_smoothing=label_smoothing,
    )


def apply_loss_from_spec(model, loss_spec: dict, train_examples=None) -> None:
    """Apply a :class:`config.spec.RunSpec` ``loss`` sub-spec to ``model``.

    ``loss_spec`` keys: ``name`` (bce|focal|eiou), ``gamma``/``alpha`` (focal/eiou),
    ``pos_weight``, ``class_weight`` (bool → global inverse-freq weights computed
    from ``train_examples``), ``max_class_weight``, ``label_weight_overrides``,
    ``label_smoothing``.  A plain ``{"name": "bce"}`` with no modifiers is a no-op
    (the GLiNER2 default loss).
    """
    spec = dict(loss_spec or {})
    name = spec.get("name", "bce")
    ls = float(spec.get("label_smoothing", 0.0) or 0.0)

    field_weights = None
    if spec.get("class_weight"):
        field_weights = compute_global_class_weights(
            train_examples or [],
            max_weight=float(spec.get("max_class_weight", 10.0)),
            label_weight_overrides=spec.get("label_weight_overrides"),
        )

    if name == "focal":
        apply_focal_loss(
            model, gamma=float(spec.get("gamma", 2.0)),
            alpha=spec.get("alpha", 0.25), label_smoothing=ls,
            field_weights=field_weights,
        )
    elif name == "margin":
        apply_margin_loss(
            model, margin=float(spec.get("margin", 0.15)),
            label_smoothing=ls, field_weights=field_weights,
        )
    elif name == "set":
        apply_set_consistent_loss(
            model,
            margin_set=float(spec.get("margin_set", 0.5)),
            lambda_set=float(spec.get("lambda_set", 1.0)),
            lambda_fp=float(spec.get("lambda_fp", 0.25)),
            lambda_label=float(spec.get("lambda_label", 0.25)),
            set_topk=int(spec.get("set_topk", 0)),
            base_margin=float(spec.get("base_margin", 0.0)),
            pos_weight=spec.get("pos_weight"),
            label_smoothing=ls, field_weights=field_weights,
        )
    elif name == "eiou":
        apply_entity_iou_loss(
            model, alpha=float(spec.get("alpha", 0.25)),
            label_smoothing=ls, field_weights=field_weights,
        )
    else:  # bce (+ optional pos_weight / class_weight / smoothing)
        apply_weighted_loss(
            model, pos_weight=spec.get("pos_weight"),
            field_weights=field_weights, label_smoothing=ls,
        )


# ---------------------------------------------------------------------------
# Backwards-compatible presets (original public API)
# ---------------------------------------------------------------------------

def apply_weighted_loss(
    model,
    *,
    pos_weight: Optional[float] = None,
    field_weights: Optional[Dict[str, float]] = None,
    label_smoothing: float = 0.0,
    boundary_smoothing: float = 0.0,
    boundary_smoothing_size: int = 1,
    class_weight: bool = False,
    max_class_weight: float = 10.0,
    full_grid: bool = False,
) -> None:
    """Weighted-BCE preset: pos_weight + (global or legacy dynamic) class weight
    + label/boundary smoothing.  No-op when nothing is requested.

    ``full_grid=True`` switches off GLiNER2's random negative-span masking, so the
    BCE base sees every cell -- as it always does under ``set``.  It is what makes
    ``bce_bundle`` differ from ``set`` by the competition terms alone."""
    has_field_weights = field_weights is not None and len(field_weights) > 0
    if (pos_weight is None and not has_field_weights and not class_weight
            and label_smoothing == 0 and boundary_smoothing == 0 and not full_grid):
        return  # nothing to patch

    apply_struct_loss(
        model, bce_term,
        term_kwargs={"pos_weight": pos_weight} if pos_weight is not None else None,
        field_weights=field_weights,
        label_smoothing=label_smoothing,
        boundary_smoothing=boundary_smoothing,
        boundary_smoothing_size=boundary_smoothing_size,
        dynamic_field_weight=bool(class_weight) and not has_field_weights,
        max_class_weight=max_class_weight,
        full_grid=True if full_grid else None,
    )
    _log("bce", pos_weight=pos_weight, field_weights=field_weights,
         label_smoothing=label_smoothing, boundary_smoothing=boundary_smoothing,
         legacy_cw=bool(class_weight) and not has_field_weights,
         full_grid=bool(full_grid))


def apply_focal_loss(
    model,
    *,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    label_smoothing: float = 0.0,
    field_weights: Optional[Dict[str, float]] = None,
) -> None:
    """Focal-loss preset (replaces BCE)."""
    apply_struct_loss(
        model, focal_term,
        term_kwargs={"gamma": gamma, "alpha": alpha},
        field_weights=field_weights,
        label_smoothing=label_smoothing,
    )
    _log(f"focal(γ={gamma})", alpha=alpha, field_weights=field_weights,
         label_smoothing=label_smoothing)


def apply_entity_iou_loss(
    model,
    *,
    alpha: float = 0.25,
    label_smoothing: float = 0.0,
    field_weights: Optional[Dict[str, float]] = None,
) -> None:
    """Entity-IoU-loss preset (replaces BCE)."""
    apply_struct_loss(
        model, eiou_term,
        term_kwargs={"alpha": alpha},
        field_weights=field_weights,
        label_smoothing=label_smoothing,
    )
    _log(f"entity_iou(α={alpha})", field_weights=field_weights,
         label_smoothing=label_smoothing)


def apply_focal_eiou_loss(
    model,
    *,
    mix: float = 0.5,
    gamma: float = 2.0,
    focal_alpha=0.25,
    eiou_alpha: float = 0.25,
    label_smoothing: float = 0.0,
    boundary_smoothing: float = 0.0,
    boundary_smoothing_size: int = 1,
    field_weights: Optional[Dict[str, float]] = None,
) -> None:
    """Combined ``mix·focal + (1-mix)·eiou`` preset (replaces BCE).

    ``mix`` is the combination weight (NOT the per-term ``alpha``); per-term alphas
    stay at the paper default 0.25 unless overridden.
    ``boundary_smoothing`` (ε) spreads probability to spans near gold boundaries
    before the term is applied (compatible with focal_eiou; exclusive with
    ``label_smoothing``)."""
    apply_struct_loss(
        model, focal_eiou_term,
        term_kwargs={"mix": mix, "gamma": gamma,
                     "focal_alpha": focal_alpha, "eiou_alpha": eiou_alpha},
        field_weights=field_weights,
        label_smoothing=label_smoothing,
        boundary_smoothing=boundary_smoothing,
        boundary_smoothing_size=boundary_smoothing_size,
    )
    _log(f"focal_eiou(mix={mix},γ={gamma})", field_weights=field_weights,
         label_smoothing=label_smoothing, boundary_smoothing=boundary_smoothing)


def apply_margin_loss(
    model,
    *,
    margin: float = 0.15,
    label_smoothing: float = 0.0,
    boundary_smoothing: float = 0.0,
    boundary_smoothing_size: int = 1,
    field_weights: Optional[Dict[str, float]] = None,
) -> None:
    """Additive-margin BCE preset (replaces BCE) — see :func:`margin_bce_term`.

    No added parameters (unlike the scorer-side lams/biaffine axis): pushes gold
    logits down by ``margin`` before the sigmoid, so the loss itself demands a
    separation margin between semantically-close labels.

    ``boundary_smoothing`` composes cleanly: the margin shift keys off
    ``labs_hard`` while the BCE target uses the smoothed labels.
    """
    apply_struct_loss(
        model, margin_bce_term,
        term_kwargs={"margin": margin},
        field_weights=field_weights,
        label_smoothing=label_smoothing,
        boundary_smoothing=boundary_smoothing,
        boundary_smoothing_size=boundary_smoothing_size,
    )
    _log(f"margin(m={margin})", field_weights=field_weights,
         label_smoothing=label_smoothing, boundary_smoothing=boundary_smoothing)


def apply_set_consistent_loss(
    model,
    *,
    margin_set: float = 0.5,
    lambda_set: float = 1.0,
    lambda_fp: float = 0.25,
    margin_label: float = 0.5,
    lambda_label: float = 0.25,
    set_topk: int = 0,
    iou_match: float = 0.5,
    pos_weight: Optional[float] = None,
    base_margin: float = 0.0,
    label_smoothing: float = 0.0,
    boundary_smoothing: float = 0.0,
    boundary_smoothing_size: int = 1,
    field_weights: Optional[Dict[str, float]] = None,
) -> None:
    """Set-Consistent preset — see :func:`set_consistent_term`.

    Trains against the evaluator's set-matching view: per field, the cells above
    the decision boundary (``s > 0``) are assigned to golds by max-IoU ownership
    (the evaluator's lenient criterion); each gold is hinged above the best
    non-gold cell its region owns (``margin_set``/``lambda_set``), pooled cells
    outside every region are mined as set-level FPs (``lambda_fp``), plus the
    cross-label hinge (``lambda_label``).  ``set_topk=0`` (default) is the
    published objective; ``set_topk>0`` caps the pool at the K highest cells, a
    legacy ablation knob that invalidates the paper's error decomposition.

    Pair with the scorer's ``null_score=True`` (train.py does this for
    ``--loss set``): logit 0 becomes the learned "beats the label's null"
    decode boundary — evaluate those checkpoints with ``--threshold 0.5``.
    """
    apply_struct_loss(
        model, set_consistent_term,
        term_kwargs={
            "margin_set": margin_set, "lambda_set": lambda_set,
            "lambda_fp": lambda_fp,
            "margin_label": margin_label, "lambda_label": lambda_label,
            "set_topk": set_topk, "iou_match": iou_match,
            "base_margin": base_margin, "pos_weight": pos_weight,
        },
        field_weights=field_weights,
        label_smoothing=label_smoothing,
        boundary_smoothing=boundary_smoothing,
        boundary_smoothing_size=boundary_smoothing_size,
    )
    _log(
        f"set(m={margin_set},λ_set={lambda_set},λ_fp={lambda_fp},"
        f"λ_lab={lambda_label},pool={'s>0' if set_topk <= 0 else f'top{set_topk}'},"
        f"base_m={base_margin})",
        pos_weight=pos_weight,
        field_weights=field_weights, label_smoothing=label_smoothing,
        boundary_smoothing=boundary_smoothing,
    )


def _log(name, *, pos_weight=None, alpha=None, field_weights=None,
         label_smoothing=0.0, boundary_smoothing=0.0, legacy_cw=False,
         full_grid=False) -> None:
    parts = [name]
    if full_grid:
        parts.append("full_grid(no negative masking)")
    if pos_weight is not None:
        parts.append(f"pos_weight={pos_weight}")
    if alpha is not None:
        parts.append(f"α={alpha}")
    if field_weights:
        parts.append(f"field_weights({len(field_weights)} labels)")
    if legacy_cw:
        parts.append("legacy_class_weight")
    if label_smoothing:
        parts.append(f"label_smoothing={label_smoothing}")
    if boundary_smoothing:
        parts.append(f"boundary_smoothing={boundary_smoothing}")
    print(f"[LossWrapper] Patched compute_struct_loss: {', '.join(parts)}")


__all__ = [
    "apply_struct_loss", "build_scores_and_labels", "compute_boundary_smoothed_labels",
    "LOSS_TERMS", "bce_term", "focal_term", "eiou_term", "focal_eiou_term", "margin_bce_term",
    "set_consistent_term",
    "compute_eniou_map",
    "compute_global_class_weights", "extract_entity_field_names",
    "apply_loss_by_name", "apply_loss_from_spec",
    "apply_weighted_loss", "apply_focal_loss", "apply_entity_iou_loss",
    "apply_focal_eiou_loss", "apply_margin_loss",
    "apply_set_consistent_loss",
]
