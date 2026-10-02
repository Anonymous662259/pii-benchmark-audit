#!/usr/bin/env python3
"""Paired sentence-level bootstrap for the PII objective comparison.

``scripts/significance_battery.py`` covers macro and micro F1 across four sampling units,
but not **rare-half macro** -- and on these corpora that is where the effect lives, so the
headline number would otherwise go unTested.  This adds it, on the same paired design:
the two arms saw identical data and identical seeds, so a resample draws one set of
sentence indices and scores both arms on it.

Macro is recomputed here as well.  It is not redundant: it has to reproduce the battery's
macro p-values, which is the check that this implementation is counting the same way.

    python scripts/pii_significance.py                      # all four corpora
    python scripts/pii_significance.py --datasets gretel --resamples 20000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import HungarianEvaluator  # noqa: E402

STRICT = HungarianEvaluator(iou_threshold=1.0, fuzzy_max_dist=None)
ROOT = Path("out_pii_baselines")
BCE, SET = "v1-bce-baseline-50k", "v1-set-consistent-50k"
DATASETS = ["privy", "kiji", "pii_500k", "gretel"]
SEEDS = [42, 1337]


def per_sentence_counts(path: Path, labels: List[str]) -> np.ndarray:
    """(n_sentences, n_labels, 4) of tp/fp/fn/support, strict exact match."""
    index = {label: i for i, label in enumerate(labels)}
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            tp, fp, fn, sup = STRICT.count_example(record.get("gt") or [],
                                                   record.get("predictions") or [])
            cell = np.zeros((len(labels), 4))
            for source, col in ((tp, 0), (fp, 1), (fn, 2), (sup, 3)):
                for label, value in source.items():
                    if label in index:
                        cell[index[label], col] += value
            rows.append(cell)
    return np.asarray(rows)


def label_union(paths: Sequence[Path]) -> List[str]:
    labels = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for span in (record.get("gt") or []):
                    labels.add(span["tag"])
                for span in (record.get("predictions") or []):
                    labels.add(span["label"])
    return sorted(labels)


def metrics_from(totals: np.ndarray, rare: np.ndarray) -> Tuple[float, float, float]:
    """totals: (n_labels, 4) summed over the resampled sentences -> macro, rare, micro."""
    tp, fp, fn, sup = totals[:, 0], totals[:, 1], totals[:, 2], totals[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    seen = sup > 0
    macro = 100 * f1[seen].mean() if seen.any() else 0.0
    rare_seen = rare & seen
    rare_macro = 100 * f1[rare_seen].mean() if rare_seen.any() else 0.0
    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro = 100 * (2 * p * r / (p + r) if p + r else 0.0)
    return macro, rare_macro, micro


def summarise(d: np.ndarray, observed: float) -> Dict[str, float]:
    """Percentile CI + two-sided bootstrap p for one metric's draw vector."""
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": round(float(observed), 3),
            "ci_lo": round(float(np.percentile(d, 2.5)), 3),
            "ci_hi": round(float(np.percentile(d, 97.5)), 3),
            "p": round(float(min(1.0, p)), 4)}


def bootstrap(dataset: str, resamples: int, rng_seed: int, family: str = "gliner2") -> Dict:
    paths_a = [ROOT/BCE/family/dataset/f"seed-{s}"/"test"/"predictions"/"all.jsonl" for s in SEEDS]
    paths_b = [ROOT/SET/family/dataset/f"seed-{s}"/"test"/"predictions"/"all.jsonl" for s in SEEDS]
    missing = [p for p in paths_a + paths_b if not p.is_file()]
    if missing:
        return {"dataset": dataset, "error": f"missing {len(missing)} dump(s)"}

    labels = label_union(paths_a + paths_b)
    A = np.stack([per_sentence_counts(p, labels) for p in paths_a])   # (seeds, sents, L, 4)
    B = np.stack([per_sentence_counts(p, labels) for p in paths_b])
    if A.shape[1] != B.shape[1]:
        return {"dataset": dataset, "error": "arms scored different sentence counts"}
    n_sent = A.shape[1]

    # "Rare half" is fixed once, from the gold support of the whole test set, so both arms
    # and every resample are judged on the same set of labels. Recomputing it per resample
    # would let the two arms be scored on different label sets.
    support = A[0].sum(axis=0)[:, 3]
    order = np.argsort(support)
    rare = np.zeros(len(labels), dtype=bool)
    rare[order[:max(1, math.ceil((support > 0).sum() / 2))]] = True

    def observed(T):
        return np.array([metrics_from(T[i].sum(axis=0), rare) for i in range(T.shape[0])]).mean(axis=0)

    obs_a, obs_b = observed(A), observed(B)
    diff = obs_b - obs_a

    # Resampling sentences with replacement == drawing multinomial counts over sentence
    # slots, so a whole block of resamples is one matmul against the per-sentence counts
    # instead of a Python loop with fancy indexing. Same estimator, ~100x the throughput,
    # which is what makes 5k resamples on four corpora practical.
    flat_a = A.reshape(A.shape[0], n_sent, -1)         # (seeds, sents, L*4)
    flat_b = B.reshape(B.shape[0], n_sent, -1)
    rng = np.random.default_rng(rng_seed)
    draws = np.empty((resamples, 3))
    block = 250
    done = 0
    while done < resamples:
        size = min(block, resamples - done)
        weights = rng.multinomial(n_sent, np.full(n_sent, 1.0 / n_sent), size=size).astype(np.float64)
        tot_a = np.einsum("bn,snf->bsf", weights, flat_a)   # (block, seeds, L*4)
        tot_b = np.einsum("bn,snf->bsf", weights, flat_b)
        for k in range(size):
            da = np.array([metrics_from(tot_a[k, i].reshape(-1, 4), rare)
                           for i in range(A.shape[0])]).mean(axis=0)
            db = np.array([metrics_from(tot_b[k, i].reshape(-1, 4), rare)
                           for i in range(B.shape[0])]).mean(axis=0)
            draws[done + k] = db - da
        done += size

    out = {"dataset": dataset, "family": family, "n_sentences": n_sent, "n_labels": len(labels),
           "n_rare": int(rare.sum()), "resamples": resamples, "metrics": {},
           "_draws": draws, "_obs": diff}
    for j, name in enumerate(("macro", "rare_macro", "micro")):
        # Two-sided bootstrap p floors at 2/resamples, so "0.0000" means "below that
        # floor", never exactly zero.
        out["metrics"][name] = {"bce": round(float(obs_a[j]), 3),
                                "set": round(float(obs_b[j]), 3),
                                **summarise(draws[:, j], diff[j])}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--family", default="gliner2", choices=["gliner2", "gliner"],
                        help="Model family whose two arms are compared.")
    parser.add_argument("--out", default="out_pii_baselines/significance.json")
    args = parser.parse_args()

    results = []
    for i, dataset in enumerate(args.datasets):
        res = bootstrap(dataset, args.resamples, args.seed + 101 * i, args.family)
        results.append(res)
        if "error" in res:
            print(f"{dataset}: {res['error']}")
            continue
        print(f"\n{dataset}  ({res['n_sentences']} sentences, {res['n_labels']} labels, "
              f"{res['n_rare']} in the rare half, {res['resamples']} resamples)")
        for name, m in res["metrics"].items():
            star = "***" if m["p"] < .001 else "**" if m["p"] < .01 else "*" if m["p"] < .05 else ""
            print(f"  {name:11s} {m['bce']:7.2f} -> {m['set']:7.2f}  "
                  f"diff {m['diff']:+.2f}  95% CI [{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]  "
                  f"p={m['p']:.4f} {star}")

    # Across-corpus average: the estimand is the unweighted mean of the per-corpus
    # deltas -- the "average" row of a results table. Each draw resamples sentences
    # inside every corpus (paired within corpus, independent across them) and averages
    # the four deltas, so the interval carries the sentence-level noise of all four at
    # once. It does NOT carry seed noise: with two seeds there is none to estimate.
    usable = [r for r in results if "error" not in r]
    if len(usable) > 1:
        draws = np.stack([r["_draws"] for r in usable])        # (corpora, resamples, 3)
        mean_draws = draws.mean(axis=0)
        mean_obs = np.stack([r["_obs"] for r in usable]).mean(axis=0)
        agg = {"dataset": f"AVERAGE of {len(usable)}", "corpora": [r["dataset"] for r in usable],
               "resamples": usable[0]["resamples"], "metrics": {}}
        print(f"\nAVERAGE over {len(usable)} corpora (unweighted)")
        for j, name in enumerate(("macro", "rare_macro", "micro")):
            m = summarise(mean_draws[:, j], mean_obs[j])
            agg["metrics"][name] = m
            star = "***" if m["p"] < .001 else "**" if m["p"] < .01 else "*" if m["p"] < .05 else ""
            print(f"  {name:11s} {'':7s}    {'':7s}  diff {m['diff']:+.2f}  "
                  f"95% CI [{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]  p={m['p']:.4f} {star}")
        results.append(agg)

    for r in results:
        r.pop("_draws", None); r.pop("_obs", None)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
