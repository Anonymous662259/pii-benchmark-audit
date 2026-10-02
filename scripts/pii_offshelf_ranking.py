# -*- coding: utf-8 -*-
"""PROTOCOL.md 9.10: do system rankings on the PII benchmarks survive the audit's variants?

Systems (zero-shot, threshold 0.5, no training): GLiNER2 Multi-v1 and four released PII
checkpoints.  For every corpus, rank the systems by rare-half macro-F1 under
  V0 original | V1-10 types with >= 10 test mentions | V2 novel values only |
  V3 confusable types merged | V4 predictions in flagged regions removed
(V2/V3 use the converted train split, so they do not apply to MAPA), and report Kendall's
tau against V0 and whether the top system changes.  Secondary, fixed in advance: a model
trained on a corpus (nvidia -> Nemotron-PII, gretel -> Gretel-PII) ranks higher there than
its median rank on the other corpora.

    python scripts/pii_offshelf_ranking.py
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
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pii_audit as A  # noqa: E402
from evaluation.metrics import HungarianEvaluator  # noqa: E402
from evaluation.rule_detector import detect_by_rules  # noqa: E402
from pii_significance import metrics_from  # noqa: E402

ROOT = Path("out_pii_baselines")
OUT = ROOT / "pii_offshelf_ranking.json"
STRICT = HungarianEvaluator(iou_threshold=1.0, fuzzy_max_dist=None)
CORPORA = ["privy", "pii_500k", "kiji", "gretel", "nemotron_pii", "ai4p_300k", "ai4p_400k",
           "gretel_finance", "tab", "mapa"]
# system name -> (result tag, family)
SYSTEMS = {
    "GLiNER2 Multi-v1": ("v1-zeroshot", "gliner2"),
    "GLiNER2-PII": ("v1-zeroshot-gliner2-privacy-filter-pii-multi", "gliner2"),
    "GLiNER multi-PII": ("v1-zeroshot-gliner_multi_pii-v1", "gliner"),
    "NVIDIA GLiNER-PII": ("v1-zeroshot-gliner-pii", "gliner"),
    "Gretel GLiNER": ("v1-zeroshot-gretel-gliner-bi-large-v1.0", "gliner"),
}
TRAINED_ON = {"NVIDIA GLiNER-PII": "nemotron_pii", "Gretel GLiNER": "gretel"}
VARIANTS = ["V0", "V1-10", "V2", "V3", "V4"]
MERGE_SHARE = 0.05


def dump(system: str, ds: str) -> Path:
    tag, family = SYSTEMS[system]
    return ROOT / tag / family / ds / "seed-0" / "test" / "predictions" / "all.jsonl"


@lru_cache(maxsize=None)
def rows(path: Path) -> Tuple[Tuple[dict, ...], int]:
    """(documents at p0 = 0.5, documents whose inference failed -- excluded and reported)."""
    out, failed = [], 0
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("error"):
                failed += 1
                continue
            r["predictions"] = [p for p in r.get("predictions") or [] if p.get("score", 1.0) >= 0.5]
            out.append(r)
    return tuple(out), failed


@lru_cache(maxsize=None)
def train_split(ds: str):
    """(seen (type, value) pairs, type counts, value -> Counter(type)) of the converted train."""
    from datasets import load_dataset
    from config.ner_datasets import NER_DATASETS
    seen, n_label, by_value = set(), Counter(), defaultdict(Counter)
    try:
        data = load_dataset(NER_DATASETS[ds]["repo"], split="train", token=A._token() or None)
    except Exception:
        return None                                   # test-only release (MAPA)
    for r in data:
        for m in r["privacy_mask"] or []:
            n_label[m["label"]] += 1
            v = (m.get("value") or "").strip().lower()
            if v:
                seen.add((m["label"], v))
                by_value[v][m["label"]] += 1
    return seen, n_label, by_value


@lru_cache(maxsize=None)
def merge_map(ds: str) -> Dict[str, str]:
    info = train_split(ds)
    if info is None:
        return {}
    _, n_label, by_value = info
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


def flagged(r: dict) -> List[Tuple[int, int]]:
    gold = [(g["start"], g["end"]) for g in r.get("gt") or []]
    regions = [(m["start"], m["end"]) for m in detect_by_rules(r["text"], A.RULE_TYPES)
               if not any(m["start"] < e and s < m["end"] for s, e in gold)]
    return regions + [(m.start(), m.end()) for m in A._slots(r["text"])]


def _ov(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def rare_half(docs, ds: str, variant: str) -> Optional[float]:
    if variant in ("V2", "V3") and train_split(ds) is None:
        return None
    seen = train_split(ds)[0] if variant == "V2" else None
    mm = merge_map(ds) if variant == "V3" else {}
    counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4))
    for r in docs:
        gt, preds = list(r.get("gt") or []), list(r["predictions"])
        if variant == "V2":
            drop = [g for g in gt if (g["tag"], (g.get("value") or "").strip().lower()) in seen]
            gt = [g for g in gt if g not in drop]
            preds = [p for p in preds if not any(_ov(p["start"], p["end"], g["start"], g["end"]) for g in drop)]
        elif variant == "V3":
            gt = [{**g, "tag": mm.get(g["tag"], g["tag"])} for g in gt]
            preds = [{**p, "label": mm.get(p["label"], p["label"])} for p in preds]
        elif variant == "V4":
            regions = flagged(r)
            preds = [p for p in preds if not any(_ov(p["start"], p["end"], s, e) for s, e in regions)]
        for src, col in zip(STRICT.count_example(gt, preds)[:4], range(4)):
            for l, v in src.items():
                counts[l][col] += v
    labels = sorted(counts)
    tot = np.array([counts[l] for l in labels])
    return tot, labels


def score(system: str, ds: str, variant: str, support: Dict[str, float]) -> Optional[float]:
    docs, _ = rows(dump(system, ds))
    got = rare_half(docs, ds, variant)
    if got is None:
        return None
    tot, labels = got
    sup = np.array([support.get(l, 0.0) for l in labels])     # rare half from the gold, not the system
    keep = sup >= (10 if variant == "V1-10" else 1)
    tot = tot.copy()
    tot[~keep] = 0.0
    present = [i for i in range(len(labels)) if keep[i]]
    order = sorted(present, key=lambda i: sup[i])
    mask = np.zeros(len(labels), dtype=bool)
    mask[order[:max(1, math.ceil(len(present) / 2))]] = True
    return float(metrics_from(tot, mask)[1])


def gold_support(ds: str, variant: str, any_system: str) -> Dict[str, float]:
    docs, _ = rows(dump(any_system, ds))
    tot, labels = rare_half(docs, ds, variant)
    return {l: float(tot[i, 3]) for i, l in enumerate(labels)}


def ranking(scores: Dict[str, float]) -> List[str]:
    return sorted(scores, key=lambda s: -scores[s])


def main() -> None:
    result = {"systems": list(SYSTEMS), "corpora": {}}
    for ds in CORPORA:
        present = [s for s in SYSTEMS if dump(s, ds).is_file()]
        if len(present) < 2:
            print(f"{ds:15s} only {len(present)} system(s) available")
            continue
        rec = {"failed_documents": {s: rows(dump(s, ds))[1] for s in present}, "variants": {}}
        for v in VARIANTS:
            if v in ("V2", "V3") and train_split(ds) is None:
                continue
            support = gold_support(ds, v, present[0])
            sc = {s: score(s, ds, v, support) for s in present}
            rec["variants"][v] = {"rare_half": {s: round(x, 2) for s, x in sc.items()}, "ranking": ranking(sc)}
        r0 = rec["variants"]["V0"]["ranking"]
        for v, vr in rec["variants"].items():
            if v == "V0":
                continue
            pos0 = [r0.index(s) for s in present]
            posv = [vr["ranking"].index(s) for s in present]
            tau = kendalltau(pos0, posv).statistic
            vr["kendall_tau_vs_V0"] = round(float(tau), 3)
            vr["top_changed"] = vr["ranking"][0] != r0[0]
        result["corpora"][ds] = rec
        print(f"{ds:15s} V0 ranking: {r0}")
        for v, vr in rec["variants"].items():
            if v != "V0":
                print(f"   {v:6s} tau {vr['kendall_tau_vs_V0']:+.2f} top changed: {vr['top_changed']}  {vr['ranking'][:2]}")
    # summary: corpora with a top-system change per variant
    summary = {}
    for v in VARIANTS[1:]:
        cs = [c for c, r in result["corpora"].items() if v in r["variants"]]
        summary[v] = {"corpora": len(cs),
                      "top_changed": sum(result["corpora"][c]["variants"][v]["top_changed"] for c in cs),
                      "median_tau": round(statistics.median(result["corpora"][c]["variants"][v]["kendall_tau_vs_V0"]
                                                            for c in cs), 3) if cs else None}
    result["summary"] = summary
    # secondary: rank on the corpus the model was trained on vs elsewhere (V0)
    home = {}
    for system, own in TRAINED_ON.items():
        ranks = {c: r["variants"]["V0"]["ranking"].index(system) + 1
                 for c, r in result["corpora"].items() if system in r["variants"]["V0"]["ranking"]}
        if own in ranks:
            others = [k for c, k in ranks.items() if c != own]
            home[system] = {"own_corpus": own, "rank_own": ranks[own],
                            "median_rank_elsewhere": statistics.median(others) if others else None,
                            "n_systems": len(result["corpora"][own]["variants"]["V0"]["ranking"])}
    result["trained_on_own_corpus"] = home
    print("summary", json.dumps(summary))
    print("own-corpus ranks", json.dumps(home))
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
