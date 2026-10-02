# -*- coding: utf-8 -*-
"""PROTOCOL.md 9.18: does a single-corpus model depend on the type NAMES it was trained with?

At evaluation, each type's name is replaced by its description (descriptions unchanged;
predictions mapped back before scoring). For every home corpus h:
  d_trained   = F1(model trained on h, renamed) - F1(model trained on h, original names)
  d_untrained = the same for GLiNER2 Multi-v1 without training, on h
  DiD         = d_trained - d_untrained
on rare-half macro-F1 (primary) and micro-F1 (secondary); exact sign-flip test over homes.
P1 (fixed before the runs): DiD < 0 on at least 7 of 9 homes.

    python scripts/pii_labelpara.py
"""
from __future__ import annotations

import json
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

OUT = Path("out_pii_baselines/pii_labelpara.json")
HOMES = ["privy", "pii_500k", "kiji", "gretel", "nemotron_pii", "ai4p_300k", "ai4p_400k",
         "gretel_finance", "tab"]
SYS = {"untrained": "v1-zeroshot", "untrained_para": "v1-zeroshot-labelpara"}
for h in HOMES:
    SYS[f"trained_{h}"] = f"v1-zeroshot-pii-ckpt-bundle-{h}-s42"
    SYS[f"trained_{h}_para"] = f"v1-zeroshot-pii-ckpt-bundle-{h}-s42-labelpara"
R.SYSTEMS.update({k: (v, "gliner2") for k, v in SYS.items()})


def scores(system: str, ds: str):
    """(rare-half macro, micro) with the rare half fixed by the gold of the untrained run."""
    support = R.gold_support(ds, "V0", "untrained")
    rare = R.score(system, ds, "V0", support)
    docs, _ = R.rows(R.dump(system, ds))
    tot, labels = R.rare_half(docs, ds, "V0")
    micro = float(metrics_from(tot, np.zeros(len(labels), dtype=bool))[2])
    return rare, micro


def main() -> None:
    rows, missing = [], []
    for h in HOMES:
        need = ["untrained", "untrained_para", f"trained_{h}", f"trained_{h}_para"]
        absent = [s for s in need if not R.dump(s, h).is_file()]
        if absent:
            missing.append((h, absent))
            continue
        ur, um = scores("untrained", h)
        upr, upm = scores("untrained_para", h)
        tr, tm = scores(f"trained_{h}", h)
        tpr, tpm = scores(f"trained_{h}_para", h)
        failed = {s: R.rows(R.dump(s, h))[1] for s in need}
        rows.append({"home": h, "trained": round(tr, 2), "trained_para": round(tpr, 2),
                     "untrained": round(ur, 2), "untrained_para": round(upr, 2),
                     "d_trained": round(tpr - tr, 2), "d_untrained": round(upr - ur, 2),
                     "did": round((tpr - tr) - (upr - ur), 2),
                     "micro": {"trained": round(tm, 2), "trained_para": round(tpm, 2),
                               "untrained": round(um, 2), "untrained_para": round(upm, 2),
                               "did": round((tpm - tm) - (upm - um), 2)},
                     "failed_documents": failed})
    res = {"rows": rows, "missing": missing}
    if rows:
        did = [r["did"] for r in rows]
        didm = [r["micro"]["did"] for r in rows]
        res["summary"] = {
            "homes": len(rows), "did_negative": sum(d < 0 for d in did),
            "median_did": round(statistics.median(did), 2),
            "median_d_trained": round(statistics.median(r["d_trained"] for r in rows), 2),
            "median_d_untrained": round(statistics.median(r["d_untrained"] for r in rows), 2),
            "p_sign_flip": round(T.sign_flip_p(did), 4),
            "micro_did_negative": sum(d < 0 for d in didm),
            "median_micro_did": round(statistics.median(didm), 2),
            "micro_p_sign_flip": round(T.sign_flip_p(didm), 4),
            "P1_holds": sum(d < 0 for d in did) >= 7 and len(rows) == 9,
        }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for r in rows:
        print(f"{r['home']:15s} trained {r['trained']:6.2f} -> {r['trained_para']:6.2f} ({r['d_trained']:+.2f}) | "
              f"untrained {r['untrained']:6.2f} -> {r['untrained_para']:6.2f} ({r['d_untrained']:+.2f}) | "
              f"DiD {r['did']:+.2f} | micro DiD {r['micro']['did']:+.2f}")
    if missing:
        print("missing:", missing)
    print(json.dumps(res.get("summary"), indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
