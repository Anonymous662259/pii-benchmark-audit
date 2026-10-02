# -*- coding: utf-8 -*-
"""PROTOCOL.md 9.8: the 9.5 re-scoring variants on the two heavy-artifact corpora.

(a) and (b) trained on AI4Privacy-300k and -400k (seeds 42, 1337).  Same variants and
thresholds as scripts/pii_sensitivity.py; the seen-value set and the merge map come from
the converted train split the runs trained on.  Reading rule (fixed before the runs):
four run differences, so no significance claim -- report the mean (b)-(a) rare-half delta
and the runs above zero per variant; "(b) > (a) replicates" if >= 3/4 runs are positive
under V0; "the ranking holds" if the sign of the mean does not change across variants.

    python scripts/pii_sensitivity_ai4p.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rare_pii_tables as T  # noqa: E402

# before any cached call: these corpora, arms (a) and (b) only
T.DATASETS[:] = ["ai4p_300k", "ai4p_400k"]   # 300k rerun on A100 without skipped batches (PROTOCOL.md 9.12)
T.ARMS.pop("c", None)

import pii_sensitivity as S  # noqa: E402
import pii_offshelf_ranking as R  # noqa: E402

S.train_info = R.train_split          # converted train: (seen, n_label, by_value)
S.merge_map.cache_clear()

OUT = Path("out_pii_baselines/pii_sensitivity_ai4p.json")
MD = Path("out_pii_baselines/pii_sensitivity_ai4p.md")


def main() -> None:
    variants = ["V0", "V1-10", "V1-20", "V1-50", "V2", "V3", "V4"]
    result = {"seeds": {arm: list(T.seeds_of(arm)) for arm in T.ARMS}, "variants": {}}
    lines = ["| variant | (a) rare | (b) rare | (b)-(a) rare | runs > 0 | per run |", "|---|---|---|---|---|---|"]
    for v in variants:
        seeds = [s for s in T.seeds_of("a") if s in T.seeds_of("b")]
        diffs = [S.metrics(v, "b", ds, s)[1] - S.metrics(v, "a", ds, s)[1] for ds in T.DATASETS for s in seeds]
        rec = {"arms": {arm: S.arm_avg(v, arm) for arm in T.ARMS},
               "b-a": {"diff": round(float(np.mean(diffs)), 3), "n_positive": int(sum(d > 0 for d in diffs)),
                       "n": len(diffs), "per_run": [round(d, 2) for d in diffs]},
               "removed": S.describe(v)}
        result["variants"][v] = rec
        lines.append(f"| {v} | {rec['arms']['a']['rare']:.2f} | {rec['arms']['b']['rare']:.2f} | "
                     f"{rec['b-a']['diff']:+.2f} | {rec['b-a']['n_positive']}/{rec['b-a']['n']} | {rec['b-a']['per_run']} |")
        print(lines[-1], flush=True)
    v0 = result["variants"]["V0"]["b-a"]
    signs = {np.sign(r["b-a"]["diff"]) for r in result["variants"].values()}
    result["replicates"] = v0["n_positive"] >= 3   # >= 3/4 runs (PROTOCOL.md 9.12)
    result["ranking_holds"] = len(signs) == 1
    result["v4_change"] = round(result["variants"]["V4"]["b-a"]["diff"] - v0["diff"], 3)
    print(f"(b)>(a) replicates: {result['replicates']} | ranking holds across variants: {result['ranking_holds']} "
          f"| V4 changes the delta by {result['v4_change']:+.3f}")
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
