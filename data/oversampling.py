"""Document-level oversampling for rare-label balancing.

Strategy
--------
1. Compute per-label frequency across the full training set.
2. Any label with frequency < ``rare_threshold`` (fraction of mean count) is
   considered *rare*.
3. Each document is scored by how many **unique rare labels** it contains.
4. Documents with rare-label score > 0 are duplicated proportionally up to
   ``max_oversample_factor`` × original count.
5. The final list is shuffled deterministically.

This is purely document-level duplication — no text modification — so span
offsets remain valid.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

from gliner2.training.data import InputExample


# ---------------------------------------------------------------------------
# Rare-label detection
# ---------------------------------------------------------------------------

def compute_label_frequencies(examples: List[InputExample]) -> Counter:
    """Return {label: total_occurrence_count} across all examples."""
    freq: Counter = Counter()
    for ex in examples:
        for label, values in (ex.entities or {}).items():
            if values:
                freq[label] += len(values)
    return freq


def detect_rare_labels(
    freq: Counter,
    rare_threshold: float = 0.25,
) -> frozenset:
    """Return the set of labels whose frequency is below ``rare_threshold``
    times the mean label frequency.

    Args:
        freq: label → count mapping from ``compute_label_frequencies``.
        rare_threshold: multiplier on the mean count; labels below this are
            considered rare.  Default 0.25 → labels with < 25% of mean freq.
    """
    if not freq:
        return frozenset()
    mean_count = sum(freq.values()) / len(freq)
    cutoff = rare_threshold * mean_count
    return frozenset(label for label, cnt in freq.items() if cnt < cutoff)


def _rare_label_score(ex: InputExample, rare_labels: frozenset) -> int:
    """Count how many unique rare labels appear in ``ex``."""
    score = 0
    for label, values in (ex.entities or {}).items():
        if label in rare_labels and values:
            score += 1
    return score


# ---------------------------------------------------------------------------
# Oversampling
# ---------------------------------------------------------------------------

def oversample_rare_labels(
    examples: List[InputExample],
    rare_threshold: float = 0.25,
    max_oversample_factor: float = 4.0,
    seed: int = 42,
) -> Tuple[List[InputExample], Dict[str, Any]]:
    """Oversample documents containing rare labels.

    Args:
        examples: Original training examples.
        rare_threshold: Fraction of mean frequency below which a label is rare.
        max_oversample_factor: Maximum repeat multiplier for the highest-scored
            documents (documents with only 1 rare label get a smaller boost).
        seed: Reproducibility seed.

    Returns:
        (oversampled_examples, stats_dict)
    """
    freq = compute_label_frequencies(examples)
    rare_labels = detect_rare_labels(freq, rare_threshold)

    # Score each document
    scores = [_rare_label_score(ex, rare_labels) for ex in examples]
    max_score = max(scores) if scores else 0

    if max_score == 0 or not rare_labels:
        stats = {
            "original_size": len(examples),
            "final_size": len(examples),
            "rare_labels": [],
            "added_examples": 0,
        }
        return list(examples), stats

    result: List[InputExample] = list(examples)

    for ex, score in zip(examples, scores):
        if score == 0:
            continue
        # Linear scale: score / max_score → [0, 1] → [1, max_factor]
        factor = 1.0 + (score / max_score) * (max_oversample_factor - 1.0)
        # How many extra copies to add (fractional → probabilistic)
        extra_float = factor - 1.0
        extra_floor = math.floor(extra_float)
        frac = extra_float - extra_floor
        rng = random.Random(seed + hash(ex.text) % (2**31))
        extra_copies = extra_floor + (1 if rng.random() < frac else 0)
        for _ in range(extra_copies):
            result.append(ex)

    rng = random.Random(seed)
    rng.shuffle(result)

    # After-balancing frequencies
    after_freq = compute_label_frequencies(result)

    stats: Dict[str, Any] = {
        "original_size": len(examples),
        "final_size": len(result),
        "added_examples": len(result) - len(examples),
        "rare_threshold": rare_threshold,
        "max_oversample_factor": max_oversample_factor,
        "rare_labels": sorted(rare_labels),
        "before_freq": dict(sorted(freq.items(), key=lambda x: x[1])),
        "after_freq":  dict(sorted(after_freq.items(), key=lambda x: x[1])),
    }
    return result, stats


def print_oversampling_stats(stats: Dict[str, Any]) -> None:
    """Pretty-print oversampling statistics."""
    print(
        f"\nOversampling stats:\n"
        f"  Original examples : {stats['original_size']:,}\n"
        f"  Final examples    : {stats['final_size']:,}\n"
        f"  Added copies      : {stats['added_examples']:,}\n"
        f"  Rare labels ({len(stats['rare_labels'])}): {', '.join(stats['rare_labels']) or 'none'}\n"
    )
    if stats.get("before_freq"):
        print("  Label freq before → after (rare labels):")
        for lbl in stats["rare_labels"]:
            b = stats["before_freq"].get(lbl, 0)
            a = stats["after_freq"].get(lbl, 0)
            print(f"    {lbl:<20} {b:>6} → {a:>6}  (+{a-b})")
