# -*- coding: utf-8 -*-
"""PROTOCOL.md 9.7: the audit's consequences on all ten corpora, from zero-shot GLiNER2.

Reads out_pii_baselines/v1-zeroshot/gliner2/<ds>/seed-0/test/predictions/all.jsonl (decoded
at 0.5 by scripts/modal_pii_baselines.py, recipe `zeroshot`) and computes the three
analyses fixed in 9.7 before the runs:

  1. recall on test mentions whose (type, value) occurs in the converted train split vs
     on novel ones -- the model never saw train, so a gap is value difficulty, not recall of
     training data; next to it the same gap for the trained objective (b) where it exists
  2. precision with and without predictions that overlap a flagged region (rule-validated
     PII with no gold, or an unfilled generator slot)
  3. rare-half macro-F1 as scored, and with types under 10 test mentions left out

    python scripts/pii_zeroshot_analysis.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pii_audit as A  # noqa: E402
from evaluation.metrics import HungarianEvaluator  # noqa: E402
from evaluation.rule_detector import detect_by_rules  # noqa: E402
from pii_significance import metrics_from  # noqa: E402

ROOT = Path("out_pii_baselines")
OUT = ROOT / "pii_zeroshot_analysis.json"
STRICT = HungarianEvaluator(iou_threshold=1.0, fuzzy_max_dist=None)
# registry key -> key used by pii_audit.json / the paper
CORPORA = {"privy": "privy", "pii_500k": "pii_500k", "kiji": "kiji", "gretel": "gretel",
           "nemotron_pii": "nemotron", "ai4p_300k": "ai4privacy_300k", "ai4p_400k": "ai4privacy_400k",
           "gretel_finance": "gretel_finance", "tab": "tab", "mapa": "mapa"}
TRAINED_B = ROOT / "v1-bce-bundle-50k" / "gliner2"


def dump(ds: str, split: str = "test") -> Path:
    return ROOT / "v1-zeroshot" / "gliner2" / ds / "seed-0" / split / "predictions" / "all.jsonl"


FAILED = Counter()   # dump path -> documents whose inference raised (excluded, and reported)


def rows(path: Path, t: float = 0.5) -> List[dict]:
    out = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("error"):          # e.g. one CUDA OOM document in gretel_finance
                FAILED[str(path)] += 1
                continue
            r["predictions"] = [p for p in (r.get("predictions") or []) if p.get("score", 1.0) >= t]
            out.append(r)
    return out


def train_values(ds: str) -> set:
    """(type, lower value) pairs of the converted train split this model family trains on."""
    from datasets import load_dataset
    from config.ner_datasets import NER_DATASETS
    repo = NER_DATASETS[ds]["repo"]
    seen = set()
    try:
        data = load_dataset(repo, split="train", token=A._token() or None)
    except Exception:
        return seen                      # test-only release (MAPA)
    for r in data:
        for m in r["privacy_mask"] or []:
            v = (m.get("value") or "").strip().lower()
            if v:
                seen.add((m["label"], v))
    return seen


def recall_split(data: List[dict], seen: set) -> Dict[str, float]:
    hit = Counter()
    for r in data:
        pset = {(p["start"], p["end"], p["label"]) for p in r["predictions"]}
        for g in r.get("gt") or []:
            k = "seen" if (g["tag"], (g.get("value") or "").strip().lower()) in seen else "novel"
            hit[k + "_n"] += 1
            hit[k + "_tp"] += (g["start"], g["end"], g["tag"]) in pset
    out = {k: round(100 * hit[f"{k}_tp"] / hit[f"{k}_n"], 2) if hit[f"{k}_n"] else None for k in ("seen", "novel")}
    out["share_seen"] = round(100 * hit["seen_n"] / max(1, hit["seen_n"] + hit["novel_n"]), 2)
    return out


def per_type_recall(data: List[dict], seen: set) -> Dict[str, Counter]:
    """type -> Counter(seen_n, seen_tp, novel_n, novel_tp)."""
    out: Dict[str, Counter] = {}
    for r in data:
        pset = {(p["start"], p["end"], p["label"]) for p in r["predictions"]}
        for g in r.get("gt") or []:
            k = "seen" if (g["tag"], (g.get("value") or "").strip().lower()) in seen else "novel"
            c = out.setdefault(g["tag"], Counter())
            c[k + "_n"] += 1
            c[k + "_tp"] += (g["start"], g["end"], g["tag"]) in pset
    return out


def stratified_did(zs: Dict[str, Counter], tr: Dict[str, Counter], min_n: int = 20) -> Dict:
    """Post-hoc robustness check of analysis 1 (PROTOCOL.md 9.7): the trained-minus-zero-shot
    difference of the seen-minus-novel recall gap, computed within each type (types with at
    least `min_n` seen and novel mentions) and averaged with equal and with mention weights."""
    rows = []
    for t in zs:
        z, b = zs[t], tr.get(t)
        if not b or min(z["seen_n"], z["novel_n"]) < min_n:
            continue
        gap = lambda c: 100 * (c["seen_tp"] / c["seen_n"] - c["novel_tp"] / c["novel_n"])
        rows.append((t, gap(b) - gap(z), z["seen_n"] + z["novel_n"]))
    if not rows:
        return {"types": 0}
    d = np.array([r[1] for r in rows]); w = np.array([r[2] for r in rows], dtype=float)
    return {"types": len(rows), "equal_weight": round(float(d.mean()), 2),
            "mention_weight": round(float((d * w).sum() / w.sum()), 2),
            "types_positive": int((d > 0).sum()), "min_n": min_n}


def flagged(text: str, gold: List[dict]) -> List[tuple]:
    spans = [(g["start"], g["end"]) for g in gold]
    regions = [(m["start"], m["end"]) for m in detect_by_rules(text, A.RULE_TYPES)
               if not any(m["start"] < e and s < m["end"] for s, e in spans)]
    return regions + [(m.start(), m.end()) for m in A._slots(text)]


def precision_split(data: List[dict]) -> Dict[str, float]:
    tp = fp = fp_flag = 0
    for r in data:
        t, f, _, _ = STRICT.count_example(r.get("gt") or [], r["predictions"])
        tp += sum(t.values()); fp += sum(f.values())
        regions = flagged(r["text"], r.get("gt") or [])
        if regions:
            gold = {(g["start"], g["end"], g["tag"]) for g in r.get("gt") or []}
            fp_flag += sum(1 for p in r["predictions"] if (p["start"], p["end"], p["label"]) not in gold
                           and any(p["start"] < e and s < p["end"] for s, e in regions))
    p_all = 100 * tp / max(1, tp + fp)
    p_clean = 100 * tp / max(1, tp + fp - fp_flag)
    return {"precision": round(p_all, 2), "precision_without_flagged": round(p_clean, 2),
            "fp": fp, "fp_in_flagged": fp_flag, "fp_in_flagged_pct": round(100 * fp_flag / max(1, fp), 2)}


def rare_half(data: List[dict], floor: int = 0) -> Dict[str, float]:
    labels = sorted({g["tag"] for r in data for g in r.get("gt") or []} |
                    {p["label"] for r in data for p in r["predictions"]})
    index = {l: i for i, l in enumerate(labels)}
    tot = np.zeros((len(labels), 4))
    for r in data:
        for src, col in zip(STRICT.count_example(r.get("gt") or [], r["predictions"])[:4], range(4)):
            for l, v in src.items():
                tot[index[l], col] += v
    support = tot[:, 3]
    keep = support >= max(1, floor)
    tot[~keep] = 0.0
    present = [i for i in range(len(labels)) if keep[i]]
    order = sorted(present, key=lambda i: support[i])
    mask = np.zeros(len(labels), dtype=bool)
    mask[order[:max(1, math.ceil(len(present) / 2))]] = True
    macro, rare, micro = metrics_from(tot, mask)
    return {"macro": round(macro, 2), "rare": round(rare, 2), "micro": round(micro, 2),
            "types_scored": len(present)}


def main() -> None:
    result = {}
    for ds, audit_key in CORPORA.items():
        path = dump(ds)
        if not path.is_file():
            print(f"{ds:15s} no zero-shot dump yet")
            continue
        data = rows(path)
        seen = train_values(ds)
        rec = {"audit_key": audit_key, "docs": len(data), "docs_failed_excluded": FAILED[str(path)],
               "recall": recall_split(data, seen) if seen else None,
               "precision": precision_split(data),
               "rare_half": rare_half(data), "rare_half_floor10": rare_half(data, 10)}
        seeds = sorted(TRAINED_B.glob(f"{ds}/seed-*/test/predictions/all.jsonl"))
        if seeds and seen:
            # trained (b) at p0 = 0.5, every seed pooled (gold is identical across seeds, so
            # pooled counts = the seed-average recall); same seen-value set as the zero-shot run
            trained = [r for path_b in seeds for r in rows(path_b)]
            rec["trained_b_seeds"] = [p.parts[-4] for p in seeds]
            rec["recall_trained_b"] = recall_split(trained, seen)
            rec["did_by_type"] = stratified_did(per_type_recall(data, seen), per_type_recall(trained, seen))
        result[ds] = rec
        rc, pr = rec["recall"] or {}, rec["precision"]
        print(f"{ds:15s} recall seen/novel {rc.get('seen')}/{rc.get('novel')} "
              f"(trained b: {rec.get('recall_trained_b', {}).get('seen')}/{rec.get('recall_trained_b', {}).get('novel')}) | "
              f"precision {pr['precision']} -> {pr['precision_without_flagged']} "
              f"({pr['fp_in_flagged_pct']}% of FP in flagged regions) | rare {rec['rare_half']['rare']} -> "
              f"{rec['rare_half_floor10']['rare']} with floor 10", flush=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
