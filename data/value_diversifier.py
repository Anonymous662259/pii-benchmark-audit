"""Value-frequency diversification to reduce entity value memorization.

Problem
-------
Some labels have very few unique values repeated many times:
    PREFIX:  60 unique / 1,658 total  →  3.6% diversity
    GENDER:  26 unique /   473 total  →  5.5% diversity
    MARITAL: 22 unique /   257 total  →  8.6% diversity

The model memorizes "Mr." = PREFIX instead of learning contextual cues.

Three complementary strategies
------------------------------
1. **Value-frequency-capped replacement** (``diversify_values``):
   For each (label, value) pair repeated > ``max_value_repeats`` times,
   replace the excess occurrences with realistic alternatives.  Only the
   over-represented values are touched; documents with rare values are left
   untouched.

2. **Context-only masking** (``mask_entity_values``):
   With probability ``mask_ratio``, replace entity surface text with a
   generic placeholder (e.g. ``[ENTITY]``).  Forces the model to rely on
   surrounding context rather than value memorization.  Applied *only*
   to low-diversity labels so high-diversity labels (PERSON, EMAIL) keep
   their value signal.

3. **Per-value occurrence cap** (``cap_value_occurrences``):
   Hard-drop documents where the only contribution is yet another copy of an
   already-well-represented value.  Simple and aggressive — use as last resort.

Usage
-----
    from data.value_diversifier import diversify_values, detect_low_diversity_labels
    low_div = detect_low_diversity_labels(examples, threshold=0.15)
    examples, stats = diversify_values(examples, low_div, max_value_repeats=5)
"""
from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from gliner2.training.data import InputExample

from data.entity_augmentation import (
    _example_to_spans,
    _make_generators,
    _replace_in_text_with_spans,
    _spans_to_example,
)


# ---------------------------------------------------------------------------
# Detection: which labels suffer from low value diversity?
# ---------------------------------------------------------------------------

def compute_value_stats(
    examples: List[InputExample],
) -> Dict[str, Dict[str, Any]]:
    """Return per-label statistics: total count, unique count, diversity ratio.

    Returns:
        {label: {"total": int, "unique": int, "ratio": float,
                 "value_counts": Counter}}
    """
    label_values: Dict[str, list] = defaultdict(list)
    for ex in examples:
        for label, values in (ex.entities or {}).items():
            for v in (values or []):
                if v:
                    label_values[label].append(v)

    stats: Dict[str, Dict[str, Any]] = {}
    for label, vals in label_values.items():
        vc = Counter(vals)
        total = len(vals)
        unique = len(vc)
        stats[label] = {
            "total": total,
            "unique": unique,
            "ratio": unique / total if total else 1.0,
            "value_counts": vc,
        }
    return stats


def detect_low_diversity_labels(
    examples: List[InputExample],
    threshold: float = 0.15,
    min_total: int = 50,
) -> FrozenSet[str]:
    """Return labels where unique/total ratio < ``threshold``.

    Args:
        examples: Training examples.
        threshold: Diversity ratio cutoff.  0.15 means labels where < 15%
            of occurrences are unique values.
        min_total: Minimum total occurrences to consider (skip tiny labels).
    """
    stats = compute_value_stats(examples)
    low_div = set()
    for label, s in stats.items():
        if s["total"] >= min_total and s["ratio"] < threshold:
            low_div.add(label)
    return frozenset(low_div)


# ---------------------------------------------------------------------------
# Strategy 1: Value-frequency-capped replacement
# ---------------------------------------------------------------------------

def diversify_values(
    examples: List[InputExample],
    target_labels: Optional[Set[str]] = None,
    max_value_repeats: int = 5,
    seed: int = 42,
) -> Tuple[List[InputExample], Dict[str, Any]]:
    """Replace over-repeated entity values with realistic alternatives.

    For each (label, value) pair that appears > ``max_value_repeats`` times
    across the training set, randomly select excess occurrences and replace
    the entity value in-place (fixing span offsets).

    Only labels with registered generators in ``entity_augmentation`` are
    eligible for replacement.

    Args:
        examples: Training examples (modified copies returned).
        target_labels: Labels to diversify.  If None, auto-detect via
            ``detect_low_diversity_labels``.
        max_value_repeats: Maximum allowed occurrences of any single value
            per label.
        seed: Reproducibility seed.

    Returns:
        (diversified_examples, stats_dict)
    """
    rng = random.Random(seed)
    gen_table = _make_generators(rng)

    if target_labels is None:
        target_labels = detect_low_diversity_labels(examples)

    eligible_labels = frozenset(target_labels) & frozenset(gen_table.keys())

    if not eligible_labels:
        return list(examples), {"skipped": "no eligible labels"}

    # Phase 1: count (label, value) → list of (example_idx)
    value_locations: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idx, ex in enumerate(examples):
        for label, values in (ex.entities or {}).items():
            if label not in eligible_labels:
                continue
            for v in (values or []):
                if v:
                    value_locations[label][v].append(idx)

    # Phase 2: identify excess occurrences to replace
    # For each (label, value), if count > max_value_repeats, select excess
    # example indices for replacement.
    to_replace: Dict[int, Dict[str, str]] = defaultdict(dict)
    # to_replace[example_idx][old_value] = label (we'll replace it)

    replace_stats: Dict[str, int] = Counter()
    for label in eligible_labels:
        for value, indices in value_locations[label].items():
            excess = len(indices) - max_value_repeats
            if excess <= 0:
                continue
            # Randomly pick which occurrences to replace
            chosen = set(rng.sample(indices, min(excess, len(indices))))
            for idx in chosen:
                to_replace[idx][value] = label
            replace_stats[label] += len(chosen)

    if not to_replace:
        return list(examples), {
            "target_labels": sorted(eligible_labels),
            "replacements": 0,
            "message": "all values within cap",
        }

    # Phase 3: apply replacements
    result: List[InputExample] = []
    n_modified = 0

    for idx, ex in enumerate(examples):
        if idx not in to_replace:
            result.append(ex)
            continue

        values_to_replace = to_replace[idx]
        spans = _example_to_spans(ex)

        # Filter spans: only replace spans whose (label, text) match
        target_spans = [
            s for s in spans
            if s["text"] in values_to_replace
            and values_to_replace[s["text"]] == s["label"]
        ]

        if not target_spans:
            result.append(ex)
            continue

        # Build a label_generators that only replaces the targeted values
        label_gens = {
            label: gen_table[label]
            for label in set(s["label"] for s in target_spans)
            if label in gen_table
        }

        new_text, new_spans = _replace_in_text_with_spans(
            ex.text, spans, label_gens, rng,
        )

        if new_text == ex.text:
            result.append(ex)
            continue

        aug_ex = _spans_to_example(new_text, new_spans, ex)
        result.append(aug_ex)
        n_modified += 1

    stats: Dict[str, Any] = {
        "target_labels": sorted(eligible_labels),
        "max_value_repeats": max_value_repeats,
        "examples_modified": n_modified,
        "replacements_per_label": dict(sorted(replace_stats.items())),
        "original_size": len(examples),
        "final_size": len(result),
    }
    return result, stats


# ---------------------------------------------------------------------------
# Strategy 2: Context-only masking
# ---------------------------------------------------------------------------

_MASK_TOKEN = "[ENTITY]"


def mask_entity_values(
    examples: List[InputExample],
    target_labels: Optional[Set[str]] = None,
    mask_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[List[InputExample], Dict[str, Any]]:
    """Create copies where low-diversity entity values are replaced with a
    generic ``[ENTITY]`` token, forcing context-based learning.

    For each document, with probability ``mask_ratio``, create a **masked copy**
    (appended, original kept) where target-label values become ``[ENTITY]``.

    Args:
        examples: Training examples.
        target_labels: Labels to mask (None = auto-detect low-diversity).
        mask_ratio: Fraction of eligible examples to create masked copies of.
        seed: Reproducibility seed.

    Returns:
        (examples_with_masked_copies, stats_dict)
    """
    rng = random.Random(seed)

    if target_labels is None:
        target_labels = detect_low_diversity_labels(examples)

    if not target_labels:
        return list(examples), {"skipped": "no target labels"}

    eligible_idx = [
        i for i, ex in enumerate(examples)
        if any(
            label in target_labels and values
            for label, values in (ex.entities or {}).items()
        )
    ]

    n_to_mask = max(1, int(round(len(eligible_idx) * mask_ratio)))
    selected = set(rng.sample(eligible_idx, min(n_to_mask, len(eligible_idx))))

    result = list(examples)
    n_added = 0

    for idx in selected:
        ex = examples[idx]
        spans = _example_to_spans(ex)

        # Filter: only mask spans of target labels
        mask_spans = [s for s in spans if s["label"] in target_labels]
        if not mask_spans:
            continue

        # Apply masking right-to-left
        new_text = ex.text
        sorted_spans = sorted(mask_spans, key=lambda s: s["start"], reverse=True)
        offset_adjustments: List[Tuple[int, int]] = []

        for span in sorted_spans:
            start, end = span["start"], span["end"]
            old_val = span["text"]
            if new_text[start:end] != old_val:
                continue
            new_text = new_text[:start] + _MASK_TOKEN + new_text[end:]
            delta = len(_MASK_TOKEN) - len(old_val)
            offset_adjustments.append((start, delta))

        # Rebuild all spans with corrected offsets
        all_new_spans = []
        for s in spans:
            ns = dict(s)
            if s["label"] in target_labels:
                # Masked spans → the value is now [ENTITY]
                # Find the new position
                shift = sum(d for pos, d in offset_adjustments if pos < s["start"])
                ns["start"] = s["start"] + shift
                ns["end"] = ns["start"] + len(_MASK_TOKEN)
                ns["text"] = _MASK_TOKEN
            else:
                shift = sum(d for pos, d in offset_adjustments if pos < s["start"])
                ns["start"] = s["start"] + shift
                end_shift = sum(d for pos, d in offset_adjustments if pos < s["end"])
                ns["end"] = s["end"] + end_shift
                ns["text"] = new_text[ns["start"]:ns["end"]]
            all_new_spans.append(ns)

        masked_ex = _spans_to_example(new_text, all_new_spans, ex)
        result.append(masked_ex)
        n_added += 1

    stats: Dict[str, Any] = {
        "target_labels": sorted(target_labels),
        "mask_ratio": mask_ratio,
        "eligible_examples": len(eligible_idx),
        "masked_copies_added": n_added,
        "original_size": len(examples),
        "final_size": len(result),
    }
    return result, stats


# ---------------------------------------------------------------------------
# Strategy 3: Per-value occurrence cap (document dropping)
# ---------------------------------------------------------------------------

def cap_value_occurrences(
    examples: List[InputExample],
    target_labels: Optional[Set[str]] = None,
    max_value_repeats: int = 8,
    seed: int = 42,
) -> Tuple[List[InputExample], Dict[str, Any]]:
    """Drop documents that only contribute over-represented values.

    For each (label, value) pair, once ``max_value_repeats`` documents
    containing that value have been kept, subsequent documents are dropped
    **only if** all their target-label values are already over-represented.
    Documents with at least one under-represented value are always kept.

    Args:
        examples: Training examples.
        target_labels: Labels to cap (None = auto-detect low-diversity).
        max_value_repeats: Cap per (label, value).
        seed: Reproducibility seed.

    Returns:
        (filtered_examples, stats_dict)
    """
    rng = random.Random(seed)

    if target_labels is None:
        target_labels = detect_low_diversity_labels(examples)

    if not target_labels:
        return list(examples), {"skipped": "no target labels"}

    # Shuffle to randomize which copies are kept vs dropped
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    value_seen: Dict[str, Counter] = defaultdict(Counter)
    kept_indices: Set[int] = set()
    dropped = 0

    for idx in indices:
        ex = examples[idx]
        target_values = []
        has_under_represented = False

        for label, values in (ex.entities or {}).items():
            if label not in target_labels:
                continue
            for v in (values or []):
                if v:
                    target_values.append((label, v))
                    if value_seen[label][v] < max_value_repeats:
                        has_under_represented = True

        # If no target-label values at all, keep the document
        if not target_values:
            kept_indices.add(idx)
            continue

        # If at least one value is under-represented, keep
        if has_under_represented:
            kept_indices.add(idx)
            for label, v in target_values:
                value_seen[label][v] += 1
        else:
            dropped += 1

    # Reconstruct in original order
    result = [examples[i] for i in sorted(kept_indices)]

    stats: Dict[str, Any] = {
        "target_labels": sorted(target_labels),
        "max_value_repeats": max_value_repeats,
        "original_size": len(examples),
        "dropped": dropped,
        "final_size": len(result),
    }
    return result, stats


# ---------------------------------------------------------------------------
# Convenience: print stats
# ---------------------------------------------------------------------------

def print_diversity_stats(stats: Dict[str, Any], strategy: str = "") -> None:
    """Pretty-print diversification stats."""
    prefix = f"[Diversity:{strategy}] " if strategy else "[Diversity] "
    for k, v in stats.items():
        print(f"{prefix}{k}: {v}")


def print_value_report(examples: List[InputExample], top_n: int = 10) -> None:
    """Print a report of the most repeated values per label."""
    stats = compute_value_stats(examples)
    print(f"\n{'LABEL':<25s} {'TOTAL':>7s} {'UNIQUE':>7s} {'RATIO':>8s}  TOP REPEATED VALUES")
    print("-" * 90)
    for label in sorted(stats, key=lambda l: stats[l]["ratio"]):
        s = stats[label]
        top_vals = s["value_counts"].most_common(3)
        top_str = ", ".join(f'"{v}"×{c}' for v, c in top_vals)
        print(f"{label:<25s} {s['total']:>7,d} {s['unique']:>7,d} {s['ratio']:>7.1%}  {top_str}")
