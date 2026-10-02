# -*- coding: utf-8 -*-
"""PROTOCOL.md 9.19: type names for zero-shot use, chosen on validation, scored on test.

Strategies: S0 original registry names, S1 humanized ("ACCOUNT_NUMBER" -> "account number"),
S2 the type's description as its name (descriptions always kept). Models: GLiNER2 Multi-v1
without training, and the released GLiNER2-PII.

    python scripts/pii_naming.py            # select on validation, list the test runs still needed
    python scripts/pii_naming.py --report   # test gains of the validation-selected names
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pii_offshelf_ranking as R  # noqa: E402
import rare_pii_tables as T  # noqa: E402
from pii_significance import metrics_from  # noqa: E402

OUT = Path("out_pii_baselines/pii_naming.json")
CORPORA = ["privy", "pii_500k", "kiji", "gretel", "nemotron_pii", "ai4p_300k", "ai4p_400k",
           "gretel_finance", "tab"]
MODELS = {"untrained": "zeroshot", "GLiNER2-PII": "zeroshot-gliner2-privacy-filter-pii-multi"}
STRATS = {"S0": "", "S1": "-labelhuman", "S2": "-labelpara"}
FLAG = {"S1": "humanized", "S2": "description"}


def path(model: str, strat: str, ds: str, split: str) -> Path:
    round_ = "val" if split == "validation" else "v1"
    tag = f"{round_}-{MODELS[model]}{STRATS[strat]}"
    return R.ROOT / tag / "gliner2" / ds / "seed-0" / split / "predictions" / "all.jsonl"


def score(p: Path):
    """(rare-half macro, micro); rare half = lower half of the types by this split's gold support."""
    docs, failed = R.rows(p)
    tot, labels = R.rare_half(docs, None, "V0")
    sup = tot[:, 3]
    present = [i for i in range(len(labels)) if sup[i] > 0]
    order = sorted(present, key=lambda i: sup[i])
    mask = np.zeros(len(labels), dtype=bool)
    mask[order[:max(1, math.ceil(len(present) / 2))]] = True
    _, rare, mic = metrics_from(tot, mask)
    return float(rare), float(mic), failed


def select() -> dict:
    sel, missing_val = {}, []
    for m in MODELS:
        for ds in CORPORA:
            got = {}
            for s in STRATS:
                p = path(m, s, ds, "validation")
                if p.is_file():
                    got[s] = score(p)[0]
                else:
                    missing_val.append((m, s, ds))
            if len(got) == len(STRATS):
                best = max(STRATS, key=lambda s: (got[s], s == "S0"))     # ties keep S0
                sel[f"{m}|{ds}"] = {"val": {s: round(v, 2) for s, v in got.items()}, "chosen": best}
    # global rule: per model, the strategy with the highest median validation gain over S0
    glob = {}
    for m in MODELS:
        rows = [v for k, v in sel.items() if k.startswith(m + "|")]
        if len(rows) == len(CORPORA):
            med = {s: statistics.median(r["val"][s] - r["val"]["S0"] for r in rows) for s in STRATS}
            glob[m] = {"median_val_gain": {s: round(x, 2) for s, x in med.items()},
                       "chosen": max(STRATS, key=lambda s: (med[s], s == "S0"))}
    need = []
    for k, v in sel.items():
        m, ds = k.split("|")
        for s in {v["chosen"], glob.get(m, {}).get("chosen", "S0")}:
            if s != "S0" and not path(m, s, ds, "test").is_file():
                need.append((m, s, ds))
    return {"selection": sel, "global": glob, "missing_validation": missing_val, "test_runs_needed": need}


def report(res: dict) -> dict:
    out = {}
    for m in MODELS:
        rows = []
        g = res["global"].get(m, {}).get("chosen", "S0")
        for ds in CORPORA:
            v = res["selection"].get(f"{m}|{ds}")
            if not v:
                continue
            c = v["chosen"]
            base = score(path(m, "S0", ds, "test"))
            ch = score(path(m, c, ds, "test"))
            gl = score(path(m, g, ds, "test"))
            rows.append({"corpus": ds, "chosen": c, "S0": round(base[0], 2), "selected": round(ch[0], 2),
                         "gain": round(ch[0] - base[0], 2), "gain_micro": round(ch[1] - base[1], 2),
                         "global": g, "gain_global": round(gl[0] - base[0], 2)})
        gains = [r["gain"] for r in rows]
        out[m] = {"rows": rows, "n": len(rows), "improved": sum(x > 0 for x in gains),
                  "unchanged": sum(x == 0 for x in gains),
                  "median_gain": round(statistics.median(gains), 2) if gains else None,
                  "p_sign_flip": round(T.sign_flip_p([x for x in gains if x != 0]), 4) if any(gains) else None,
                  "global_strategy": g,
                  "global_improved": sum(r["gain_global"] > 0 for r in rows),
                  "median_global_gain": round(statistics.median(r["gain_global"] for r in rows), 2) if rows else None,
                  "chosen_counts": {s: sum(r["chosen"] == s for r in rows) for s in STRATS}}
    out["rule_holds"] = all(out[m]["improved"] >= 6 for m in MODELS if out[m]["n"] == len(CORPORA))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    res = select()
    if args.report:
        res["test"] = report(res)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("missing validation:", res["missing_validation"])
    for k, v in res["selection"].items():
        print(f"{k:28s} chosen {v['chosen']}  val {v['val']}")
    print("global:", res["global"])
    print("test runs needed:", res["test_runs_needed"])
    if args.report:
        for m in MODELS:
            t = res["test"][m]
            print(m, {k: t[k] for k in ("n", "improved", "unchanged", "median_gain", "p_sign_flip",
                                         "global_strategy", "global_improved", "median_global_gain",
                                         "chosen_counts")})
        print("rule holds:", res["test"]["rule_holds"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
