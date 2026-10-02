#!/usr/bin/env python3
"""Share of gold spans lost to a span that the ownership rule gives to ANOTHER gold.

Appendix A splits the cells that keep a gold span g from being emitted into four
classes.  Three of them are charged by a term of the objective:

    own    the blocking cell is owned by g itself (IoU(c,g) >= tau)  -> L_span
    mined  the blocking cell is owned by no gold at tau              -> L_fp
    miss   nothing overlaps g at all                                 -> the BCE base

The fourth is charged by nothing:

    cross  the blocking cell is owned by a DIFFERENT gold h (IoU(c,h) >= tau)

That last class is the correction term X of the bound, and it is what this script
measures.  The training-time X needs the full score grid, hence a checkpoint; the
runs were cleaned up, so this measures the realised, decode-time counterpart on the
stored prediction dumps: a gold span not recovered exactly, which an EMITTED span
owned by another gold overlaps.

Assumption (D) of the theorem decodes per field, so per-field competition is the
default; --decoder cross_type reproduces what resolve_span_conflicts actually does
(a single pool over every label), which can only be looser.

    python scripts/cross_gold_coverage.py
    python scripts/cross_gold_coverage.py --decoder cross_type --arm scl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

TAU = 0.5

PII_ROOT = Path("out_pii_baselines")
PII_BCE, PII_SET = "v1-bce-baseline-50k", "v1-set-consistent-50k"


def _pii(key: str, seeds=(42, 1337)) -> Tuple[List[Path], List[Path]]:
    bce = [PII_ROOT / PII_BCE / "gliner2" / key / f"seed-{s}" / "test" / "predictions" / "all.jsonl"
           for s in seeds]
    scl = [PII_ROOT / PII_SET / "gliner2" / key / f"seed-{s}" / "test" / "predictions" / "all.jsonl"
           for s in seeds]
    return bce, scl


def _std(key: str, scl_root: str) -> Tuple[List[Path], List[Path]]:
    bce = sorted((Path("out_eval/ner_loss") / key).glob("bce-seed*/test/predictions/all.jsonl"))
    scl = sorted(Path(scl_root).glob(key + "__s*/test/predictions/all.jsonl"))
    return list(bce), list(scl)


DATASETS = [
    ("CoNLL-2003", lambda: _std("conll2003", "out_eval/cwpw2_10seed")),
    ("OntoNotes 5.0", lambda: _std("ontonotes5", "out_eval/cwpw2_onto")),
    ("FIN", lambda: _std("fin", "out_eval/cwpw2_10seed")),
    ("MITRestaurant", lambda: _std("mit_restaurant", "out_eval/cwpw2_10seed")),
    ("Privy", lambda: _pii("privy")),
    ("OpenPII", lambda: _pii("pii_500k")),
    ("Kiji", lambda: _pii("kiji")),
    ("Gretel", lambda: _pii("gretel")),
]

CLASSES = ("cross_any", "cross_only", "own", "mined", "miss")


def _iou(a, b) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _overlaps(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def sentence(gt: Sequence[dict], preds: Sequence[dict], per_field: bool) -> Dict[str, int]:
    gold = [(g["start"], g["end"], g["tag"]) for g in gt]
    pred = [(p["start"], p["end"], p["label"]) for p in preds]

    used, matched = set(), set()
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if pi not in used and p == g:
                used.add(pi)
                matched.add(gi)
                break

    out = {k: 0 for k in CLASSES}
    out["n_gold"] = len(gold)
    out["n_lost"] = 0

    for gi, g in enumerate(gold):
        if gi in matched:
            continue
        out["n_lost"] += 1
        saw_cross = saw_own = saw_mined = False
        for p in pred:
            if per_field and p[2] != g[2]:
                continue
            if not _overlaps(p, g):
                continue
            # Owner of this emitted cell: the gold span it has the largest IoU with,
            # restricted to the same field when the decoder competes per field.
            best_i, best = None, 0.0
            for hi, h in enumerate(gold):
                if per_field and h[2] != p[2]:
                    continue
                v = _iou(p, h)
                if v > best:
                    best, best_i = v, hi
            if best >= TAU and best_i is not None and best_i != gi:
                saw_cross = True
            elif best >= TAU:
                saw_own = True
            else:
                saw_mined = True
        if saw_cross:
            out["cross_any"] += 1
            if not (saw_own or saw_mined):
                out["cross_only"] += 1
        elif saw_own:
            out["own"] += 1
        elif saw_mined:
            out["mined"] += 1
        else:
            out["miss"] += 1
    return out


def run(path: Path, per_field: bool) -> Dict[str, int]:
    total = {k: 0 for k in list(CLASSES) + ["n_gold", "n_lost"]}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts = sentence(row.get("gt") or [], row.get("predictions") or [], per_field)
            for key, value in counts.items():
                total[key] += value
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decoder", default="per_field", choices=["per_field", "cross_type"])
    parser.add_argument("--arm", default="both", choices=["bce", "scl", "both"])
    parser.add_argument("--out", default="out_pii_baselines/cross_gold_coverage.json")
    args = parser.parse_args()
    per_field = args.decoder == "per_field"

    print("decoder=" + args.decoder + "  tau=" + str(TAU)
          + "  (every percentage is of ALL gold spans)")
    head = ("{:15s} {:4s} {:>4s} {:>8s} {:>7s} {:>11s} {:>12s} {:>7s} {:>7s} {:>7s}"
            .format("dataset", "arm", "runs", "gold", "lost%", "cross_any%",
                    "cross_only%", "own%", "mined%", "miss%"))
    print(head)
    print("-" * len(head))

    report = {}
    for name, resolve in DATASETS:
        bce_paths, scl_paths = resolve()
        entry = {}
        for arm, paths in (("bce", bce_paths), ("scl", scl_paths)):
            if args.arm != "both" and args.arm != arm:
                continue
            paths = [p for p in paths if p.is_file()]
            if not paths:
                print("{:15s} {:4s}  -- no dumps".format(name, arm))
                continue
            per_run = [run(p, per_field) for p in paths]
            gold = statistics.mean(r["n_gold"] for r in per_run)
            pct = {}
            for key in ["n_lost"] + list(CLASSES):
                mean = statistics.mean(r[key] for r in per_run)
                pct[key] = 100 * mean / gold if gold else 0.0
            spread = [100 * r["cross_any"] / r["n_gold"] for r in per_run if r["n_gold"]]
            sd = statistics.stdev(spread) if len(spread) > 1 else 0.0
            entry[arm] = dict({"n_runs": len(per_run), "n_gold": round(gold, 1),
                               "cross_any_sd": round(sd, 3)},
                              **{k + "_pct": round(v, 3) for k, v in pct.items()})
            print("{:15s} {:4s} {:4d} {:8.0f} {:7.2f} {:11.3f} {:12.3f} {:7.2f} {:7.2f} {:7.2f}"
                  .format(name, arm, len(per_run), gold, pct["n_lost"], pct["cross_any"],
                          pct["cross_only"], pct["own"], pct["mined"], pct["miss"]))
        report[name] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"decoder": args.decoder, "tau": TAU,
                                          "data": report}, indent=2), encoding="utf-8")
    print()
    print("cross_any  = lost gold with at least one blocking span owned by another gold")
    print("cross_only = lost gold whose blockers are ALL of that kind")
    print("own / mined / miss are charged by L_span / L_fp / the BCE base respectively")
    print("wrote " + args.out)


if __name__ == "__main__":
    main()
