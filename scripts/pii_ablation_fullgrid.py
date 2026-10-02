# -*- coding: utf-8 -*-
"""PROTOCOL.md 9.9 / 9.13: how much of (b)-(a) does switching off negative masking alone explain?

Arm (a') = `--loss bce --full-grid` on the four case-study corpora.  Rare-half macro-F1 on
test at each run's validation-tuned threshold (as in the main table), over the seeds that
(a'), (a) and (b) all have (42 in 9.9; 42 and 1337 once 9.13 lands), paired by seed.
Reported per corpus and pooled: (a')-(a), (b)-(a), and their ratio.  Reading rule (fixed
before the runs): "masking explains most of the gain" if the pooled (a')-(a) >= 0.5 x
(b)-(a).  Descriptive only.

    python scripts/pii_ablation_fullgrid.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rare_pii_tables as T  # noqa: E402

T.ARMS["f"] = "v1-bce-fullgrid-50k"      # (a') before any cached call
OUT = Path("out_pii_baselines/pii_ablation_fullgrid.json")


def main() -> None:
    seeds = [s for s in T.seeds_of("f") if s in T.seeds_of("a") and s in T.seeds_of("b")]
    result = {"seeds": seeds, "corpora": {}}
    for ds in T.DATASETS:
        per_seed = {}
        for s in seeds:
            m = {arm: T.score(arm, ds, s, "test", T.tuned_t(arm, ds, s)) for arm in ("a", "f", "b")}
            per_seed[s] = {"a": float(m["a"][1]), "f": float(m["f"][1]), "b": float(m["b"][1])}
        fa = statistics.mean(v["f"] - v["a"] for v in per_seed.values())
        ba = statistics.mean(v["b"] - v["a"] for v in per_seed.values())
        rec = {"per_seed": {str(s): {k: round(x, 3) for k, x in v.items()} for s, v in per_seed.items()},
               "f_minus_a": round(fa, 3), "b_minus_a": round(ba, 3),
               "share": round(fa / ba, 3) if abs(ba) > 1e-9 else None}
        result["corpora"][ds] = rec
        print(f"{ds:9s} a'-a {fa:+.2f}  b-a {ba:+.2f}  (seeds {seeds})")
    fa = float(statistics.mean(r["f_minus_a"] for r in result["corpora"].values()))
    ba = float(statistics.mean(r["b_minus_a"] for r in result["corpora"].values()))
    runs = [(v["f"] - v["a"]) for r in result["corpora"].values() for v in r["per_seed"].values()]
    result["pooled"] = {"f_minus_a": round(fa, 3), "b_minus_a": round(ba, 3),
                        "share": round(fa / ba, 3) if abs(ba) > 1e-9 else None,
                        "f_minus_a_runs_positive": f"{sum(d > 0 for d in runs)}/{len(runs)}",
                        "masking_explains_most": bool(fa >= 0.5 * ba)}
    print("pooled", result["pooled"])
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
