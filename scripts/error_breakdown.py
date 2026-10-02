#!/usr/bin/env python3
"""Decompose strict-match errors into the four categories SCL's terms target.

The objective adds a span-ranking term, a type-ranking term and an active
false-positive term, but the paper so far reports only aggregate F1, so it cannot
say whether the terms reduce the error types they were designed for.  Each of the
four categories below is the prediction-level counterpart of one of them:

    type      gold span recovered with the exact offsets but the wrong label
              -> the event L_type penalises
    boundary  gold span recovered with the right label but wrong offsets
              -> the event L_span penalises
    missed    no prediction overlaps the gold span at all
              -> the event the BCE base term penalises
    spurious  emitted span that overlaps no gold span
              -> the event L_fp penalises

A gold span is assigned to the first category that applies (type, then boundary,
then boundary+type, then missed) and the prediction that explains it is consumed,
so no prediction is counted twice.  Everything left over is spurious.

This reads the stored ``predictions/all.jsonl`` dumps only; no model, no GPU.

    python scripts/error_breakdown.py --suite pii
    python scripts/error_breakdown.py --suite fin
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

CATEGORIES = ("correct", "type", "boundary", "boundary_type", "missed", "spurious")

PII_ROOT = Path("out_pii_baselines")
PII_BCE, PII_SET = "v1-bce-baseline-50k", "v1-set-consistent-50k"
PII_CORPORA = ["privy", "pii_500k", "kiji", "gretel"]
PII_SEEDS = [42, 1337]


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def classify_sentence(gt: Sequence[dict], preds: Sequence[dict],
                      rare: set = None) -> Counter:
    """One sentence -> counts over CATEGORIES.

    When `rare` is given, every gold-side category is additionally counted under a
    "rare/" or "freq/" prefix according to the gold label, so the same taxonomy can
    be read separately for the low-support half of the schema -- which is where the
    paper's claim lives. Spurious predictions are split by the predicted label.
    """
    counts: Counter = Counter()
    used_pred = set()
    matched_gold = set()

    pred_key = [(p["start"], p["end"], p["label"]) for p in preds]
    gold_key = [(g["start"], g["end"], g["tag"]) for g in gt]

    for gi, gk in enumerate(gold_key):
        for pi, pk in enumerate(pred_key):
            if pi in used_pred:
                continue
            if gk == pk:
                used_pred.add(pi)
                matched_gold.add(gi)
                counts["correct"] += 1
                if rare is not None:
                    counts[("rare/" if gk[2] in rare else "freq/") + "correct"] += 1
                break

    for gi, g in enumerate(gt):
        if gi in matched_gold:
            continue
        gs, ge, gl = g["start"], g["end"], g["tag"]

        exact_offsets = [pi for pi, p in enumerate(preds)
                         if pi not in used_pred and p["start"] == gs and p["end"] == ge]
        if exact_offsets:
            used_pred.add(exact_offsets[0])
            counts["type"] += 1
            if rare is not None:
                counts[("rare/" if gl in rare else "freq/") + "type"] += 1
            continue

        same_label = [pi for pi, p in enumerate(preds)
                      if pi not in used_pred and p["label"] == gl
                      and _overlaps(gs, ge, p["start"], p["end"])]
        if same_label:
            used_pred.add(same_label[0])
            counts["boundary"] += 1
            if rare is not None:
                counts[("rare/" if gl in rare else "freq/") + "boundary"] += 1
            continue

        any_overlap = [pi for pi, p in enumerate(preds)
                       if pi not in used_pred and _overlaps(gs, ge, p["start"], p["end"])]
        if any_overlap:
            used_pred.add(any_overlap[0])
            counts["boundary_type"] += 1
            if rare is not None:
                counts[("rare/" if gl in rare else "freq/") + "boundary_type"] += 1
            continue

        counts["missed"] += 1
        if rare is not None:
            counts[("rare/" if gl in rare else "freq/") + "missed"] += 1

    for pi, p in enumerate(preds):
        if pi not in used_pred:
            counts["spurious"] += 1
            if rare is not None:
                counts[("rare/" if p["label"] in rare else "freq/") + "spurious"] += 1
    counts["n_gold"] += len(gt)
    counts["n_pred"] += len(preds)
    return counts


def breakdown(path: Path, rare: set = None) -> Counter:
    total: Counter = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total.update(classify_sentence(row.get("gt") or [],
                                           row.get("predictions") or [], rare))
    return total


def rare_labels(path: Path) -> set:
    """Lower-support half of the labels, by gold support in this test set."""
    support: Counter = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            for g in json.loads(line).get("gt") or []:
                support[g["tag"]] += 1
    ordered = sorted(support, key=lambda l: support[l])
    import math
    return set(ordered[:max(1, math.ceil(len(ordered) / 2))])


def mean_counts(paths: Sequence[Path], rare: set = None) -> Dict[str, float]:
    per_run = [breakdown(p, rare) for p in paths]
    keys = set(list(CATEGORIES) + ["n_gold", "n_pred"])
    for c in per_run:
        keys |= set(c)
    return {k: statistics.mean(c.get(k, 0) for c in per_run) for k in keys}


def report(name: str, bce: Dict[str, float], scl: Dict[str, float]) -> Dict[str, float]:
    gold = bce["n_gold"]
    print(f"\n{name}  ({gold:.0f} gold spans, mean over seeds)")
    print(f"  {'category':14s} {'BCE':>9s} {'SCL':>9s} {'delta':>9s} {'rel.':>8s}")
    deltas = {}
    for cat in CATEGORIES:
        b, s = bce[cat], scl[cat]
        rel = (s - b) / b * 100 if b else float("nan")
        deltas[cat] = s - b
        print(f"  {cat:14s} {b:9.1f} {s:9.1f} {s - b:+9.1f} {rel:+7.1f}%")
    return deltas


GOLD_SIDE = ("correct", "type", "boundary", "boundary_type", "missed")


def pii_suite() -> None:
    rows = {}
    for corpus in PII_CORPORA:
        bce_paths = [PII_ROOT/PII_BCE/"gliner2"/corpus/f"seed-{s}"/"test"/"predictions"/"all.jsonl"
                     for s in PII_SEEDS]
        scl_paths = [PII_ROOT/PII_SET/"gliner2"/corpus/f"seed-{s}"/"test"/"predictions"/"all.jsonl"
                     for s in PII_SEEDS]
        rare = rare_labels(bce_paths[0])
        rows[corpus] = (mean_counts(bce_paths, rare), mean_counts(scl_paths, rare))

    for bucket, title in (("rare/", "LOW-SUPPORT half"), ("freq/", "HIGH-SUPPORT half")):
        print("\n" + "=" * 78)
        print(f"{title} of each schema -- change in counts, set-consistent minus BCE")
        print("  (gold-side categories are assigned by GOLD label; spurious by PREDICTED label)")
        header = f"  {'corpus':10s} {'gold':>7s} " + " ".join(f"{c:>13s}" for c in GOLD_SIDE)                  + f" {'spurious':>13s} {'balance':>8s}"
        print(header)
        totals = Counter()
        gold_total = 0.0
        for corpus, (bce, scl) in rows.items():
            gold = sum(bce.get(bucket + c, 0) for c in GOLD_SIDE)
            gold_total += gold
            cells, balance = [], 0.0
            for cat in GOLD_SIDE + ("spurious",):
                d = scl.get(bucket + cat, 0) - bce.get(bucket + cat, 0)
                totals[cat] += d
                if cat in GOLD_SIDE:
                    balance += d
                pct = d / gold * 100 if gold else 0.0
                cells.append(f"{d:+7.1f} ({pct:+.2f}%)".rjust(13))
            print(f"  {corpus:10s} {gold:7.0f} " + " ".join(cells) + f" {balance:+8.1f}")
        cells = []
        balance = 0.0
        for cat in GOLD_SIDE + ("spurious",):
            d = totals[cat]
            if cat in GOLD_SIDE:
                balance += d
            pct = d / gold_total * 100 if gold_total else 0.0
            cells.append(f"{d:+7.1f} ({pct:+.2f}%)".rjust(13))
        print(f"  {'TOTAL':10s} {gold_total:7.0f} " + " ".join(cells) + f" {balance:+8.1f}")
        print("  balance = sum of the five gold-side changes; it must be 0 because the gold")
        print("  count is identical in both arms. A non-zero value means a miscount.")


def fin_suite() -> None:
    bce_dirs = sorted(Path("out_eval/ner_loss/fin").glob("bce-seed*"))
    scl_dirs = sorted(Path("out_eval/cwpw2_10seed").glob("fin__s*"))
    bce_paths = [d/"test"/"predictions"/"all.jsonl" for d in bce_dirs]
    bce_paths = [p for p in bce_paths if p.is_file()]
    scl_paths = [d/"test"/"predictions"/"all.jsonl" for d in scl_dirs]
    scl_paths = [p for p in scl_paths if p.is_file()]
    if not bce_paths or not scl_paths:
        print("fin: prediction dumps not found")
        return
    print(f"fin: {len(bce_paths)} BCE run(s), {len(scl_paths)} SCL run(s)")
    report("fin", mean_counts(bce_paths), mean_counts(scl_paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="pii", choices=["pii", "fin"])
    args = parser.parse_args()
    (pii_suite if args.suite == "pii" else fin_suite)()


if __name__ == "__main__":
    sys.exit(main())
