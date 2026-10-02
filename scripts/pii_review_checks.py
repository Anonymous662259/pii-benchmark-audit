# -*- coding: utf-8 -*-
"""Checks asked for by the NAACL-style review (PROTOCOL.md 9.17), all from existing dumps.

W2 (D1, D2)  why do single-corpus models fall below the untrained model elsewhere?
             D1: rare-half target types a model never predicts (type collapse), trained
                 vs untrained, and the rare-half difference on the types both predict.
             D2: recall on target types whose name exists in the home schema vs not.
W3           power of "rankings do not change": margins of adjacent released systems
             and whether any adjacent pair flips under V1-V4; objective contrasts with
             the corpus (not the run) as unit.
W8 (D3)      rankings of Tables 6/7 under micro-F1 next to rare-half macro-F1.

    python scripts/pii_review_checks.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pii_offshelf_ranking as R  # noqa: E402
import pii_sensitivity as S  # noqa: E402
import pii_transfer_matrix as TM  # noqa: E402
import rare_pii_tables as T  # noqa: E402
from config.ner_datasets import NER_DATASETS  # noqa: E402
from pii_significance import metrics_from  # noqa: E402

OUT = Path("out_pii_baselines/pii_review_checks.json")
BASE = "GLiNER2 Multi-v1"
RELEASED = list(R.SYSTEMS)
R.SYSTEMS.update(TM.TRAINED)


def table(system: str, ds: str):
    """(labels, totals[tp, fp, fn, support]) for one system on one corpus, V0."""
    docs, _ = R.rows(R.dump(system, ds))
    tot, labels = R.rare_half(docs, ds, "V0")
    return labels, tot


def rare_labels(ds: str) -> list:
    labels, tot = table(BASE, ds)
    sup = {l: tot[i, 3] for i, l in enumerate(labels)}
    present = sorted((l for l in labels if sup[l] > 0), key=lambda l: sup[l])
    return present[:max(1, math.ceil(len(present) / 2))]


def f1_of(row) -> float:
    tp, fp, fn = row[0], row[1], row[2]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 100 * (2 * p * r / (p + r) if p + r else 0.0)


def rare_macro(system: str, ds: str, only=None) -> float:
    labels, tot = table(system, ds)
    idx = {l: i for i, l in enumerate(labels)}
    rl = [l for l in rare_labels(ds) if only is None or l in only]
    return float(np.mean([f1_of(tot[idx[l]]) if l in idx else 0.0 for l in rl])) if rl else float("nan")


def micro(system: str, ds: str) -> float:
    labels, tot = table(system, ds)
    return float(metrics_from(tot, np.zeros(len(labels), dtype=bool))[2])


def home_of(name: str) -> str:
    return name.split(" ", 1)[1]


def predicted(system: str, ds: str) -> set:
    labels, tot = table(system, ds)
    return {l for i, l in enumerate(labels) if tot[i, 0] + tot[i, 1] > 0}


def w2(systems=None) -> dict:
    trained = [t for t in (systems or TM.TRAINED) if R.dump(t, "privy").is_file()]
    pairs, rec = [], {}
    for t in trained:
        h = home_of(t)
        home_schema = {k.upper() for k in NER_DATASETS[h]["label_descriptions"]}
        for ds in R.CORPORA:
            if ds == h or not R.dump(t, ds).is_file():
                continue
            rl = rare_labels(ds)
            pt, pb = predicted(t, ds), predicted(BASE, ds)
            zero_t = [l for l in rl if l not in pt]
            zero_b = [l for l in rl if l not in pb]
            both = set(rl) & pt & pb
            d_all = rare_macro(t, ds) - rare_macro(BASE, ds)
            d_both = rare_macro(t, ds, both) - rare_macro(BASE, ds, both) if both else float("nan")
            # D2: recall by whether the target type's name exists in the home schema
            lt, tt = table(t, ds)
            lb, tb = table(BASE, ds)
            it, ib = {l: i for i, l in enumerate(lt)}, {l: i for i, l in enumerate(lb)}
            grp = {}
            for name, members in (("shared", [l for l in lt if l.upper() in home_schema]),
                                  ("unseen", [l for l in lt if l.upper() not in home_schema])):
                def rec_of(tab, ix):
                    tp = sum(tab[ix[l], 0] for l in members if l in ix)
                    fn = sum(tab[ix[l], 2] for l in members if l in ix)
                    return 100 * tp / (tp + fn) if tp + fn else float("nan")
                grp[name] = {"types": len(members), "recall_trained": round(rec_of(tt, it), 1),
                             "recall_untrained": round(rec_of(tb, ib), 1)}
            row = {"home": h, "target": ds, "rare_types": len(rl), "zero_trained": len(zero_t),
                   "zero_untrained": len(zero_b), "delta_rare": round(d_all, 2),
                   "delta_rare_on_types_both_predict": round(d_both, 2) if d_both == d_both else None,
                   "types_both_predict": len(both), "by_name": grp}
            pairs.append(row)
    below = [p for p in pairs if p["delta_rare"] < 0]
    rec["pairs"] = pairs
    rec["summary"] = {
        "pairs": len(pairs), "pairs_below_untrained": len(below),
        "below_with_more_zero_types": sum(p["zero_trained"] > p["zero_untrained"] for p in below),
        "mean_zero_trained": round(statistics.mean(p["zero_trained"] for p in pairs), 2),
        "mean_zero_untrained": round(statistics.mean(p["zero_untrained"] for p in pairs), 2),
        "median_delta_rare": round(statistics.median(p["delta_rare"] for p in pairs), 2),
        "median_delta_on_both": round(statistics.median(p["delta_rare_on_types_both_predict"]
                                                        for p in pairs
                                                        if p["delta_rare_on_types_both_predict"] is not None), 2),
        "unseen_recall_trained": round(statistics.median(p["by_name"]["unseen"]["recall_trained"] for p in pairs
                                                         if p["by_name"]["unseen"]["types"]), 1),
        "unseen_recall_untrained": round(statistics.median(p["by_name"]["unseen"]["recall_untrained"] for p in pairs
                                                           if p["by_name"]["unseen"]["types"]), 1),
        "shared_recall_trained": round(statistics.median(p["by_name"]["shared"]["recall_trained"] for p in pairs
                                                         if p["by_name"]["shared"]["types"]
                                                         and p["by_name"]["shared"]["recall_trained"] == p["by_name"]["shared"]["recall_trained"]), 1),
        "shared_recall_untrained": round(statistics.median(p["by_name"]["shared"]["recall_untrained"] for p in pairs
                                                           if p["by_name"]["shared"]["types"]
                                                           and p["by_name"]["shared"]["recall_untrained"] == p["by_name"]["shared"]["recall_untrained"]), 1),
    }
    return rec


def w3() -> dict:
    off = json.loads(R.OUT.read_text(encoding="utf-8"))
    adj, flips = [], []
    for ds, rec in off["corpora"].items():
        v0 = rec["variants"]["V0"]
        order = v0["ranking"]
        for a, b in zip(order, order[1:]):
            m = v0["rare_half"][a] - v0["rare_half"][b]
            flipped = [v for v, vr in rec["variants"].items() if v != "V0"
                       and vr["ranking"].index(a) > vr["ranking"].index(b)]
            adj.append({"corpus": ds, "upper": a, "lower": b, "margin": round(m, 2), "flips_under": flipped})
    close = [p for p in adj if p["margin"] < 3]
    # objectives: one unit per corpus (seed-averaged), not per run
    obj = {}
    for v in ("V0", "V1-10", "V2", "V3", "V4"):
        for x, y in (("a", "b"), ("b", "c")):
            seeds = [s for s in T.seeds_of(x) if s in T.seeds_of(y)]
            per = [float(np.mean([S.metrics(v, y, ds, s)[1] - S.metrics(v, x, ds, s)[1] for s in seeds]))
                   for ds in T.DATASETS]
            obj[f"{v}:{y}-{x}"] = {"per_corpus": [round(d, 2) for d in per], "positive": sum(d > 0 for d in per),
                                   "p_sign_flip_corpus": round(T.sign_flip_p(per), 4),
                                   "min_attainable_p": round(2 / 2 ** len(per), 4)}
    return {"adjacent": adj, "n_adjacent": len(adj), "n_close_lt3": len(close),
            "close_flipped": sum(bool(p["flips_under"]) for p in close),
            "any_flipped": sum(bool(p["flips_under"]) for p in adj),
            "min_margin": min(p["margin"] for p in adj), "objectives_by_corpus": obj}


def w8() -> dict:
    out = {}
    systems = [s for s in R.SYSTEMS]
    for ds in R.CORPORA:
        present = [s for s in systems if R.dump(s, ds).is_file()]
        rh = {s: rare_macro(s, ds) for s in present}
        mi = {s: micro(s, ds) for s in present}
        rel = [s for s in present if s in RELEASED]
        tau_all = kendalltau([rh[s] for s in present], [mi[s] for s in present]).correlation
        tau_rel = kendalltau([rh[s] for s in rel], [mi[s] for s in rel]).correlation if len(rel) > 2 else None
        top = lambda d, ss: max(ss, key=lambda s: d[s])
        out[ds] = {"rare_half": {s: round(v, 2) for s, v in rh.items()}, "micro": {s: round(v, 2) for s, v in mi.items()},
                   "tau_all": round(tau_all, 3), "tau_released": round(tau_rel, 3) if tau_rel is not None else None,
                   "top_rare_all": top(rh, present), "top_micro_all": top(mi, present),
                   "top_rare_released": top(rh, rel), "top_micro_released": top(mi, rel)}
    homes = [home_of(t) for t in TM.TRAINED if R.dump(t, "privy").is_file()]
    first_micro = sum(out[h]["top_micro_all"] == f"(b) {h}" for h in homes if h in out)
    return {"corpora": out, "median_tau_all": round(statistics.median(v["tau_all"] for v in out.values()), 3),
            "median_tau_released": round(statistics.median(v["tau_released"] for v in out.values()
                                                           if v["tau_released"] is not None), 3),
            "top_released_same": sum(v["top_rare_released"] == v["top_micro_released"] for v in out.values()),
            "n_corpora": len(out), "home_first_under_micro": first_micro, "homes": len(homes)}


def micro_variant(system: str, ds: str, variant: str):
    """Micro-F1 of one system on one corpus under a variant (V1-k: types under k mentions unscored)."""
    docs, _ = R.rows(R.dump(system, ds))
    base = "V0" if variant.startswith("V1") else variant
    got = R.rare_half(docs, ds, base)
    if got is None:
        return None
    tot, labels = got
    if variant.startswith("V1"):
        k = int(variant.split("-")[1])
        tot = tot.copy()
        tot[tot[:, 3] < k] = 0.0
    return float(metrics_from(tot, np.zeros(len(labels), dtype=bool))[2])


def w5() -> dict:
    """Review W5: the adjacent-pair and ranking analysis of W3, repeated under micro-F1."""
    variants = ["V0", "V1-10", "V2", "V3", "V4"]
    adj, taus, top_changed, n_corp = [], {v: [] for v in variants[1:]}, {v: 0 for v in variants[1:]}, {v: 0 for v in variants[1:]}
    per = {}
    for ds in R.CORPORA:
        present = [s for s in RELEASED if R.dump(s, ds).is_file()]
        sc = {v: {s: micro_variant(s, ds, v) for s in present} for v in variants}
        sc = {v: d for v, d in sc.items() if all(x is not None for x in d.values())}
        per[ds] = {v: {s: round(x, 2) for s, x in d.items()} for v, d in sc.items()}
        order = sorted(present, key=lambda s: -sc["V0"][s])
        for v in sc:
            if v == "V0":
                continue
            taus[v].append(kendalltau([sc["V0"][s] for s in present], [sc[v][s] for s in present]).correlation)
            n_corp[v] += 1
            top_changed[v] += max(present, key=lambda s: sc[v][s]) != order[0]
        for a, b in zip(order, order[1:]):
            flipped = [v for v in sc if v != "V0" and sc[v][a] < sc[v][b]]
            adj.append({"corpus": ds, "upper": a, "lower": b,
                        "margin": round(sc["V0"][a] - sc["V0"][b], 2), "flips_under": flipped})
    close = [p for p in adj if p["margin"] < 3]
    flipped = [p["margin"] for p in adj if p["flips_under"]]
    # objectives (a)/(b)/(c) under micro-F1, corpus as the unit
    obj = {}
    for v in variants:
        for x, y in (("a", "b"), ("b", "c")):
            seeds = [s for s in T.seeds_of(x) if s in T.seeds_of(y)]
            d = [float(np.mean([S.metrics(v, y, ds, s)[2] - S.metrics(v, x, ds, s)[2] for s in seeds]))
                 for ds in T.DATASETS]
            obj[f"{v}:{y}-{x}"] = {"per_corpus": [round(z, 2) for z in d], "mean": round(float(np.mean(d)), 3),
                                   "positive": sum(z > 0 for z in d)}
    return {"per_corpus": per, "adjacent": adj, "n_adjacent": len(adj), "n_close_lt3": len(close),
            "close_flipped": sum(bool(p["flips_under"]) for p in close),
            "any_flipped": len(flipped), "max_flipped_margin": max(flipped) if flipped else 0.0,
            "median_tau": {v: round(statistics.median(t), 3) for v, t in taus.items() if t},
            "top_changed": {v: f"{top_changed[v]}/{n_corp[v]}" for v in top_changed},
            "objectives_micro": obj}


def main() -> None:
    res = {"W2": w2(), "W3": w3(), "W8": w8(), "W5": w5()}
    w5r = res["W5"]
    print("W5 micro:", {k: w5r[k] for k in ("n_adjacent", "n_close_lt3", "close_flipped", "any_flipped",
                                            "max_flipped_margin", "median_tau", "top_changed")})
    for k, v in w5r["objectives_micro"].items():
        print("  micro", k, v["per_corpus"], "mean", v["mean"], "pos", v["positive"])
    # 9.17 human-annotated control, decomposed the same way (kept apart from the 8 synthetic homes)
    R.SYSTEMS.update(TM.TAB_CONTROL)
    res["W2_tab"] = w2(list(TM.TAB_CONTROL))
    print("TAB control:", json.dumps(res["W2_tab"]["summary"]))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["W2"]["summary"], indent=1))
    w3r = res["W3"]
    print({k: w3r[k] for k in ("n_adjacent", "n_close_lt3", "close_flipped", "any_flipped", "min_margin")})
    for k, v in w3r["objectives_by_corpus"].items():
        print(k, v["per_corpus"], v["positive"], v["p_sign_flip_corpus"])
    w8r = res["W8"]
    print({k: w8r[k] for k in ("median_tau_all", "median_tau_released", "top_released_same", "n_corpora",
                               "home_first_under_micro", "homes")})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
