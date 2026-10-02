# -*- coding: utf-8 -*-
"""PROTOCOL.md section 9.5: do the arm comparisons survive the benchmark flaws of the audit?

Re-scores the existing test dumps of (a)/(b)/(c) on the four protocol corpora under
variants that each remove one flaw found by scripts/pii_audit.py.  Thresholds stay at each
run's tuned value from the main table (tuned on validation, not re-tuned), so a change
can only come from the scoring.

  V0         original scoring -- must reproduce rare_pii_tables.py
  V1-k       drop labels with < k test mentions (k = 10, 20, 50); rare half re-drawn
  V2         novel values only: drop gold mentions whose (label, value) occurs in train,
             and every prediction overlapping a dropped gold span
  V3         merge confusable labels: A~B if sum_v min(n_A(v), n_B(v)) >= 5% of the smaller
             label's train mentions; connected components; rare half re-drawn
  V4         drop predictions overlapping rule-validated PII with no gold (U) or an
             unfilled generator slot (G)

    python scripts/pii_sensitivity.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pii_audit as A  # noqa: E402
import rare_pii_tables as T  # noqa: E402
from evaluation.rule_detector import detect_by_rules  # noqa: E402
from pii_significance import metrics_from  # noqa: E402

OUT = Path("out_pii_baselines/pii_sensitivity.json")
MD = Path("out_pii_baselines/pii_sensitivity.md")
SUPPORT_FLOORS = (10, 20, 50)
MERGE_SHARE = 0.05
PAIRS = (("a", "b"), ("b", "c"), ("a", "c"))


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


@lru_cache(maxsize=None)
def texts(path: Path) -> Tuple[str, ...]:
    with path.open(encoding="utf-8") as handle:
        return tuple(json.loads(line)["text"] for line in handle if line.strip())


@lru_cache(maxsize=None)
def train_info(ds: str):
    """(seen (label, value) pairs, label mention counts, value -> Counter(label)) from train."""
    seen, n_label, by_value = set(), Counter(), defaultdict(Counter)
    for _, spans in A.load_split(ds, "train"):
        for _, _, lab, v in spans:
            n_label[lab] += 1
            key = v.strip().lower()
            if key:
                seen.add((lab, key))
                by_value[key][lab] += 1
    return seen, n_label, by_value


@lru_cache(maxsize=None)
def merge_map(ds: str) -> Dict[str, str]:
    """V3: label -> group name ("A+B+..."), merging by the 5% shared-value rule."""
    _, n_label, by_value = train_info(ds)
    shared = Counter()
    for c in by_value.values():
        labs = sorted(c)
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                shared[(labs[i], labs[j])] += min(c[labs[i]], c[labs[j]])
    parent = {l: l for l in n_label}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (x, y), n in shared.items():
        if n >= MERGE_SHARE * min(n_label[x], n_label[y]):
            parent[find(x)] = find(y)
    groups = defaultdict(list)
    for l in n_label:
        groups[find(l)].append(l)
    return {l: "+".join(sorted(g)) for g in groups.values() for l in g}


@lru_cache(maxsize=None)
def flagged_regions(ds: str) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """V4: per test sentence, spans of uncovered rule-validated PII and unfilled slots."""
    ref = T.dump("a", ds, T.seeds_of("a")[0], "test")
    out = []
    for (gt, _), text in zip(T.rows(ref), texts(ref)):
        gold = [(dict(g)["start"], dict(g)["end"]) for g in gt]
        regions = [(m["start"], m["end"]) for m in detect_by_rules(text, A.RULE_TYPES)
                   if not any(_overlap(m["start"], m["end"], s, e) for s, e in gold)]
        regions += [(m.start(), m.end()) for m in A._slots(text)]
        out.append(tuple(regions))
    return tuple(out)


def transform(variant: str, ds: str, i: int, gt: List[dict], preds: List[dict], text: str, stats: Counter):
    if variant == "V2":
        seen = train_info(ds)[0]
        drop = [g for g in gt if (g["tag"], (g.get("value") or text[g["start"]:g["end"]]).strip().lower()) in seen]
        if drop:
            stats["gold_dropped"] += len(drop)
            gt = [g for g in gt if g not in drop]
            keep = [p for p in preds if not any(_overlap(p["start"], p["end"], g["start"], g["end"]) for g in drop)]
            stats["preds_dropped"] += len(preds) - len(keep)
            preds = keep
        stats["gold_kept"] += len(gt)
    elif variant == "V3":
        m = merge_map(ds)
        gt = [{**g, "tag": m.get(g["tag"], g["tag"])} for g in gt]
        preds = [{**p, "label": m.get(p["label"], p["label"])} for p in preds]
    elif variant == "V4":
        regions = flagged_regions(ds)[i]
        if regions:
            keep = [p for p in preds if not any(_overlap(p["start"], p["end"], s, e) for s, e in regions)]
            stats["preds_dropped"] += len(preds) - len(keep)
            preds = keep
    return gt, preds


@lru_cache(maxsize=None)
def totals(variant: str, arm: str, ds: str, seed: int):
    """(labels, (n_labels, 4) tp/fp/fn/support summed over test) at the run's tuned threshold."""
    path = T.dump(arm, ds, seed, "test")
    base = T.labels_of(ds, "test")
    labels = tuple(sorted({merge_map(ds).get(l, l) for l in base})) if variant == "V3" else base
    index = {l: k for k, l in enumerate(labels)}
    tot = np.zeros((len(labels), 4))
    stats = Counter()
    for i, ((gt, preds), text) in enumerate(zip(T.filtered(path, T.tuned_t(arm, ds, seed)), texts(path))):
        gt, preds = transform(variant, ds, i, gt, preds, text, stats)
        for src, col in zip(T.STRICT.count_example(gt, preds)[:4], range(4)):
            for l, v in src.items():
                tot[index[l], col] += v
    return labels, tot, dict(stats)


def rare_half(support: np.ndarray, keep: np.ndarray) -> np.ndarray:
    present = [i for i in range(len(support)) if support[i] > 0 and keep[i]]
    order = sorted(present, key=lambda i: support[i])
    mask = np.zeros(len(support), dtype=bool)
    mask[order[:max(1, math.ceil(len(present) / 2))]] = True
    return mask


def metrics(variant: str, arm: str, ds: str, seed: int) -> Tuple[float, float, float]:
    base_variant = "V0" if variant.startswith("V1") else variant
    labels, tot, _ = totals(base_variant, arm, ds, seed)
    ref_labels, ref_tot, _ = totals(base_variant, "a", ds, T.seeds_of("a")[0])   # gold is the same for every run
    assert ref_labels == labels
    support = ref_tot[:, 3]
    if variant == "V0" or variant in ("V2", "V4"):
        mask = T.rare_mask(ds)            # the main table's rare half
        return metrics_from(tot, mask)
    if variant == "V3":
        return metrics_from(tot, rare_half(support, np.ones(len(labels), dtype=bool)))
    k = int(variant.split("-")[1])        # V1-k
    keep = support >= k
    tot = tot.copy()
    tot[~keep] = 0.0                      # labels below the floor are not scored at all
    return metrics_from(tot, rare_half(support, keep))


def contrast(variant: str, x: str, y: str) -> Dict:
    seeds = [s for s in T.seeds_of(x) if s in T.seeds_of(y)]
    d = {m: [] for m in ("macro", "rare", "micro")}
    for ds in T.DATASETS:
        for s in seeds:
            mx, my = metrics(variant, x, ds, s), metrics(variant, y, ds, s)
            for j, m in enumerate(d):
                d[m].append(my[j] - mx[j])
    return {m: {"diff": round(float(np.mean(v)), 3), "p_sign_flip": round(T.sign_flip_p(v), 4),
                "n_positive": int(sum(dd > 0 for dd in v)), "n_pairs": len(v)} for m, v in d.items()} | {"seeds": seeds}


def arm_avg(variant: str, arm: str) -> Dict[str, float]:
    per = [np.mean([metrics(variant, arm, ds, s) for s in T.seeds_of(arm)], axis=0) for ds in T.DATASETS]
    return {m: round(float(np.mean([p[j] for p in per])), 3) for j, m in enumerate(("macro", "rare", "micro"))}


def describe(variant: str) -> Dict:
    """What the variant removed, per corpus (from arm (a), first seed; gold-side is arm-free)."""
    out = {}
    for ds in T.DATASETS:
        s0 = T.seeds_of("a")[0]
        if variant.startswith("V1"):
            k = int(variant.split("-")[1])
            _, tot, _ = totals("V0", "a", ds, s0)
            sup = tot[:, 3]
            out[ds] = {"labels_kept": int(((sup >= k) & (sup > 0)).sum()), "labels_total": int((sup > 0).sum())}
        elif variant == "V3":
            groups = sorted({g for g in merge_map(ds).values() if "+" in g})
            out[ds] = {"merged_groups": groups}
        elif variant in ("V2", "V4"):
            out[ds] = {arm: totals(variant, arm, ds, T.seeds_of(arm)[0])[2] for arm in T.ARMS}
    return out


def main() -> None:
    variants = ["V0"] + [f"V1-{k}" for k in SUPPORT_FLOORS] + ["V2", "V3", "V4"]
    result = {"seeds": {arm: list(T.seeds_of(arm)) for arm in T.ARMS}, "variants": {}}
    lines = ["| variant | (a) rare | (b) rare | (c) rare | (b)-(a) rare [p, runs>0] | (c)-(b) rare [p, runs>0] "
             "| (b)-(a) holds | (c)-(b) not decisive |", "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        print(f"== {v}", flush=True)
        rec = {"arms": {arm: arm_avg(v, arm) for arm in T.ARMS},
               "contrasts": {f"{y}-{x}": contrast(v, x, y) for x, y in PAIRS},
               "removed": describe(v)}
        ba, cb = rec["contrasts"]["b-a"]["rare"], rec["contrasts"]["c-b"]["rare"]
        rec["b_over_a_holds"] = ba["diff"] > 0 and ba["p_sign_flip"] < 0.05
        rec["c_over_b_not_decisive"] = cb["p_sign_flip"] >= 0.05 or cb["diff"] <= 0
        result["variants"][v] = rec
        cell = lambda c: f"{c['diff']:+.2f} [{c['p_sign_flip']:.3f}, {c['n_positive']}/{c['n_pairs']}]"
        lines.append(f"| {v} | {rec['arms']['a']['rare']:.2f} | {rec['arms']['b']['rare']:.2f} | "
                     f"{rec['arms']['c']['rare']:.2f} | {cell(ba)} | {cell(cb)} | "
                     f"{'yes' if rec['b_over_a_holds'] else 'NO'} | {'yes' if rec['c_over_b_not_decisive'] else 'NO'} |")
        print("   " + lines[-1], flush=True)
        print("   removed:", json.dumps(rec["removed"], ensure_ascii=False)[:600], flush=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("wrote", OUT, "and", MD)


if __name__ == "__main__":
    main()
