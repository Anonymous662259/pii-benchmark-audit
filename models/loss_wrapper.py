"""Backwards-compatible shim — the loss toolkit now lives in ``models.losses``.

Historically this module held three ~400-line ``apply_*`` functions that each
re-implemented ``forward`` / ``_compute_sample_loss``.  That duplication and the
``forward`` override (which clashed with the bi-encoder / two-tower archs) have
been replaced by a single composable patcher.  The public names below are kept
so ``training/train.py`` and existing notebooks import unchanged.

New code should prefer::

    from models.losses import apply_loss_by_name, apply_struct_loss, LOSS_TERMS
"""
from __future__ import annotations

from models.losses import (
    apply_entity_iou_loss,
    apply_focal_eiou_loss,
    apply_focal_loss,
    apply_set_consistent_loss,
    apply_loss_by_name,
    apply_margin_loss,
    apply_struct_loss,
    apply_weighted_loss,
    compute_global_class_weights,
)
from models.losses.terms import LOSS_TERMS, compute_eniou_map
from models.losses.weights import (
    extract_entity_field_names,
    _extract_entity_field_names,  # legacy private name
)

# Legacy private alias kept for any external callers / tests.
_compute_eniou_map = compute_eniou_map

__all__ = [
    "apply_weighted_loss",
    "apply_focal_loss",
    "apply_entity_iou_loss",
    "apply_focal_eiou_loss",
    "apply_margin_loss",
    "apply_set_consistent_loss",
    "apply_struct_loss",
    "apply_loss_by_name",
    "compute_global_class_weights",
    "LOSS_TERMS",
    "compute_eniou_map",
    "extract_entity_field_names",
]
