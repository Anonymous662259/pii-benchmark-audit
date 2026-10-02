"""Global per-label class weights + schema field-name extraction.

These helpers feed the optional ``field_weights`` multiplier shared by every
loss term (see :mod:`models.losses.patch`).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional


def compute_global_class_weights(
    examples,
    *,
    max_weight: float = 10.0,
    min_weight: float = 0.1,
    label_weight_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Per-label inverse-frequency weights over the full training set.

    ``weight[label] = mean_count / count[label]`` clamped to
    ``[min_weight, max_weight]``; manual ``label_weight_overrides`` are applied
    on top (NOT clamped).  Labels absent from the data implicitly get 1.0.
    """
    freq: Counter = Counter()
    for ex in examples:
        for label, values in (getattr(ex, "entities", None) or {}).items():
            if values:
                freq[label] += len(values)

    if not freq:
        return {}

    mean_count = sum(freq.values()) / len(freq)

    weights: Dict[str, float] = {}
    for label, count in freq.items():
        w = mean_count / count
        w = max(min_weight, min(w, max_weight))
        weights[label] = w

    if label_weight_overrides:
        for label, override_w in label_weight_overrides.items():
            weights[label] = override_w

    return weights


_E_TOKEN = "[E]"


def extract_entity_field_names(schema_tokens: List[str]) -> List[str]:
    """Ordered entity field names from a schema token list.

    Schema tokens look like ``["(", "[P]", ..., "[E]", "PERSON", "[E]", "EMAIL", ...]``;
    each ``[E]`` is followed by its field name.  Returns ``["PERSON", "EMAIL", ...]``.
    """
    fields = []
    i = 0
    while i < len(schema_tokens):
        if schema_tokens[i] == _E_TOKEN and i + 1 < len(schema_tokens):
            fields.append(schema_tokens[i + 1])
            i += 2
        else:
            i += 1
    return fields


# Backwards-compatible private alias (old name inside loss_wrapper).
_extract_entity_field_names = extract_entity_field_names
