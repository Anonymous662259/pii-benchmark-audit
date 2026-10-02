# -*- coding: utf-8 -*-
"""PROTOCOL.md 9.11: controlled transfer matrix of single-corpus (b) models.

Each (b) model (seed 42) was trained on one corpus with the same recipe; it is evaluated
on the test split of all ten corpora with that corpus's schema (recipe `zeroshot`,
--model anonymous-org/pii-ckpt-bundle-<ds>-s42).  Ranked per corpus together with GLiNER2
Multi-v1 and the four released systems of 9.10 by rare-half macro-F1 (V0).
  H1  each (b) model ranks first on the corpus it was trained on
  H2  its median rank elsewhere is worse than its rank at home
  H3  (reported, no direction) its score elsewhere minus GLiNER2 Multi-v1's
Sanity check: on its home corpus the re-evaluated checkpoint must reproduce the original
run's test dump (same checkpoint, same evaluate.py call).

    python scripts/pii_transfer_matrix.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pii_offshelf_ranking as R  # noqa: E402

OUT = Path("out_pii_baselines/pii_transfer_matrix.json")
HOMES = ["privy", "pii_500k", "kiji", "gretel", "ai4p_400k", "ai4p_300k", "nemotron_pii", "gretel_finance"]
TRAINED = {f"(b) {h}": (f"v1-zeroshot-pii-ckpt-bundle-{h}-s42", "gliner2") for h in HOMES}
ORIGINAL = {"privy": "v1-bce-bundle-50k", "pii_500k": "v1-bce-bundle-50k", "kiji": "v1-bce-bundle-50k",
            "gretel": "v1-bce-bundle-50k", "ai4p_400k": "v3-bce-bundle-50k", "ai4p_300k": "v1-bce-bundle-50k",
            "nemotron_pii": "v1-bce-bundle-50k", "gretel_finance": "v1-bce-bundle-50k",
            "tab": "tabhome-bce-bundle-50k"}
# PROTOCOL.md 9.17: the human-annotated control, kept out of the 9.11 statistics (which stay on
# the 8 synthetic homes and 13 systems) and ranked separately among all 14 systems
TAB_CONTROL = {"(b) tab": ("v1-zeroshot-pii-ckpt-bundle-tab-s42", "gliner2")}


def main() -> None:
    R.SYSTEMS.update(TRAINED)                    # dump() and score() now see the (b) models
    trained = [t for t in TRAINED if any(R.dump(t, ds).is_file() for ds in R.CORPORA)]
    result = {"trained": trained, "corpora": {}, "home": {}}
    for ds in R.CORPORA:
        present = [s for s in R.SYSTEMS if R.dump(s, ds).is_file()]
        if not present:
            continue
        support = R.gold_support(ds, "V0", present[0])
        sc = {s: R.score(s, ds, "V0", support) for s in present}
        rank = R.ranking(sc)
        result["corpora"][ds] = {"rare_half": {s: round(x, 2) for s, x in sc.items()}, "ranking": rank,
                                 "failed_documents": {s: R.rows(R.dump(s, ds))[1] for s in present}}
        print(f"{ds:15s} best: {rank[0]:22s} | " + "  ".join(f"{s.replace('(b) ', 'b:')}={sc[s]:.1f}"
                                                         for s in rank[:4]))
    for t in trained:
        result["home"][t] = home_record(t, result["corpora"])
    h1 = [t for t, r in result["home"].items() if r["rank_home"] == 1]
    result["H1_first_at_home"] = f"{len(h1)}/{len(result['home'])}"
    print("H1 first at home:", result["H1_first_at_home"])
    # 9.17 control: add the TAB model and rank it among all systems
    R.SYSTEMS.update(TAB_CONTROL)
    if any(R.dump("(b) tab", ds).is_file() for ds in R.CORPORA):
        corpora = {}
        for ds in R.CORPORA:
            present = [x for x in R.SYSTEMS if R.dump(x, ds).is_file()]
            support = R.gold_support(ds, "V0", present[0])
            sc = {x: R.score(x, ds, "V0", support) for x in present}
            corpora[ds] = {"rare_half": {x: round(v, 2) for x, v in sc.items()}, "ranking": R.ranking(sc)}
        rec = home_record("(b) tab", corpora)
        rec["rare_half"] = {ds: c["rare_half"].get("(b) tab") for ds, c in corpora.items()}
        rec["below_base_elsewhere"] = sum(g < 0 for g in rec["minus_base_elsewhere"].values())
        result["tab_control"] = rec
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


def home_record(t: str, corpora: dict) -> dict:
    """Rank at home, median rank elsewhere, gain over the untrained model, and the re-eval check."""
    if True:
        home = t.replace("(b) ", "")
        result = {"corpora": corpora}
        ranks = {ds: r["ranking"].index(t) + 1 for ds, r in result["corpora"].items() if t in r["ranking"]}
        base = "GLiNER2 Multi-v1"
        gains = {ds: round(r["rare_half"][t] - r["rare_half"][base], 2)
                 for ds, r in result["corpora"].items() if t in r["rare_half"] and base in r["rare_half"] and ds != home}
        rec = {"rank_home": ranks.get(home), "n_systems_home": len(result["corpora"].get(home, {}).get("ranking", [])),
               "median_rank_elsewhere": statistics.median([k for ds, k in ranks.items() if ds != home]) if len(ranks) > 1 else None,
               "minus_base_elsewhere": gains,
               "median_minus_base_elsewhere": statistics.median(gains.values()) if gains else None}
        # sanity: the home-corpus re-evaluation reproduces the original run's dump
        orig = R.ROOT / ORIGINAL[home] / "gliner2" / home / "seed-42" / "test" / "predictions" / "all.jsonl"
        if orig.is_file() and R.dump(t, home).is_file():
            support = R.gold_support(home, "V0", t)
            rec["home_reeval"] = round(R.score(t, home, "V0", support), 2)
            # the original dump sits under seed-42; score it through the same function
            R.SYSTEMS["__original__"] = (ORIGINAL[home], "gliner2")
            original_dump = R.dump
            R.dump = lambda s, ds, _d=original_dump: orig if s == "__original__" else _d(s, ds)
            rec["home_original"] = round(R.score("__original__", home, "V0", support), 2)
            R.dump = original_dump
            R.SYSTEMS.pop("__original__", None)
        print(f"{t:18s} rank at home {rec['rank_home']}/{rec['n_systems_home']}, median rank elsewhere "
              f"{rec['median_rank_elsewhere']}, median minus GLiNER2 elsewhere {rec['median_minus_base_elsewhere']}"
              f" | home re-eval {rec.get('home_reeval')} vs original {rec.get('home_original')}")
        return rec


if __name__ == "__main__":
    main()
