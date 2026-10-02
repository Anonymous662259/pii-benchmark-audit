# -*- coding: utf-8 -*-
"""Hand-check sample for the audit probes (PROTOCOL.md 9.4): how many flags are real problems?

The automatic probes of scripts/pii_audit.py are upper bounds: C also counts legitimate
ambiguity ("United Kingdom" as ORG and as LOC), U counts regex false alarms and PII the
schema never defined, G counts bracketed text that is not a generator slot.  This script
draws a fixed random sample of flags per (corpus, probe) for two annotators, and scores the
filled sheet: share of flags that are real problems (Wilson 95% CI) and Cohen's kappa.

    python scripts/pii_audit_sample.py                  # write audit_validation/sample.csv
    python scripts/pii_audit_sample.py --per-cell 10    # smaller sample
    python scripts/pii_audit_sample.py --score          # after both annotator columns are filled

Answers (one letter per annotator column):
  O  H = harmful (the offsets cut a wrong span)   B = benign (case/whitespace only)   ? = unclear
  C  E = the gold label here is wrong             L = legitimate in this context      ? = unclear
  U  M = real PII missed by gold, type in schema  S = real PII, type not in schema
     N = not PII (regex false alarm)              ? = unclear
  G  A = generator artifact (unfilled slot / null value annotated as PII)
     T = legitimate text                          ? = unclear
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pii_audit as A  # noqa: E402
from evaluation.rule_detector import detect_by_rules  # noqa: E402

CORPORA = A.TABLE_ORDER
OUT_DIR = Path("out_pii_baselines/audit_validation")
SHEET = OUT_DIR / "sample.csv"
SEED = 2026
CONTEXT = 90
PROBLEM = {"O": {"H"}, "C": {"E"}, "U": {"M"}, "G": {"A"}}    # answers that count as a real problem
QUESTION = {
    "O": "Offsets and value disagree. Does the span cut by the offsets differ in content? H/B/?",
    "C": "This string carries a minority label in this corpus. Is the gold label wrong HERE? E/L/?",
    "U": "Regex-validated PII with no gold span. Missed in-schema (M), out-of-schema (S), not PII (N)?",
    "G": "Is the marked text a generator artifact (unfilled slot, or a null token labelled as PII)? A/T/?",
}


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")   # NUL etc. occur in some corpora and break CSV readers


def clean(x: str) -> str:
    return _CONTROL.sub(" ", x)


def ctx(text: str, s: int, e: int) -> str:
    left, right = text[max(0, s - CONTEXT):s], text[e:e + CONTEXT]
    return clean(left + " [[" + text[s:e] + "]] " + right)


def flags(ds: str) -> Dict[str, List[dict]]:
    train, test = A.load_split(ds, "train"), A.load_split(ds, "test")
    out = defaultdict(list)
    schema = {l for _, sp in train + test for _, _, l, _ in sp}
    has = {t: any(re.search(rx, l) for l in schema) for t, rx in A.RULE_SCHEMA.items()}
    # C: majority label per value over train + test, as in probe_consistency
    by_val = defaultdict(Counter)
    for _, sp in train + test:
        for _, _, lab, v in sp:
            if v.strip():
                by_val[v.strip().lower()][lab] += 1
    for text, spans in test:
        for s, e, lab, v in spans:
            if ds not in A.NO_VALUE_FIELD and v and text[s:e] != v:
                out["O"].append({"labels": lab, "flagged": f"value={v[:80]!r} | at offsets={text[s:e][:80]!r}",
                                 "context": ctx(text, s, e)})
            key = v.strip().lower()
            c = by_val.get(key)
            if c and sum(c.values()) >= 3 and len(key) >= 2 and len(c) > 1 and lab != c.most_common(1)[0][0]:
                out["C"].append({"labels": f"here={lab}; corpus={dict(c.most_common(4))}",
                                 "flagged": text[s:e], "context": ctx(text, s, e)})
            if (v or text[s:e]).strip().lower() in A.NULL_VALUES:
                out["G"].append({"labels": f"null value labelled {lab}", "flagged": text[s:e],
                                 "context": ctx(text, s, e)})
        for m in detect_by_rules(text, A.RULE_TYPES):
            if not any(m["start"] < e and s < m["end"] for s, e, _, _ in spans):
                out["U"].append({"labels": f"rule={m['label']}; schema_has_type={has[m['label']]}",
                                 "flagged": m["text"], "context": ctx(text, m["start"], m["end"])})
        for m in A._slots(text):
            out["G"].append({"labels": "slot in text", "flagged": m.group(0), "context": ctx(text, m.start(), m.end())})
    return out


def write_sample(per_cell: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, sizes = [], {}
    for ds in CORPORA:
        print(f"== {ds}", flush=True)
        for probe, items in sorted(flags(ds).items()):
            rng = random.Random(f"{SEED}-{ds}-{probe}")
            pick = rng.sample(items, min(per_cell, len(items)))
            sizes[f"{ds}/{probe}"] = {"flags": len(items), "sampled": len(pick)}
            print(f"   {probe}: {len(items)} flags, sampled {len(pick)}", flush=True)
            for k, it in enumerate(pick):
                rows.append({"id": f"{ds}-{probe}-{k:02d}", "corpus": ds, "probe": probe,
                             "question": QUESTION[probe], **{f: clean(v) for f, v in it.items()},
                             "annotator_1": "", "annotator_2": "", "note": ""})
    # shuffle so annotators do not see one corpus/probe block at a time; ids keep the cell
    random.Random(SEED).shuffle(rows)
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:     # BOM: opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "sizes.json").write_text(__import__("json").dumps(sizes, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} items to {SHEET}")


def extend_sample(per_cell: int) -> None:
    """Top every (corpus, probe) cell up to `per_cell` items, keeping the existing sheet.

    Items already on the sheet (and their answers) are kept as they are; the new ones are
    drawn from the remaining flags of the cell with a separate seed and written both to the
    sheet and to sample_ext.csv, which is what the next labelling passes see.
    """
    import json
    rows = list(csv.DictReader(SHEET.open(encoding="utf-8-sig")))
    sizes = json.loads((OUT_DIR / "sizes.json").read_text(encoding="utf-8"))
    have = defaultdict(Counter)
    for r in rows:
        have[(r["corpus"], r["probe"])][(r["labels"], r["flagged"], r["context"])] += 1
    new = []
    for ds in CORPORA:
        print(f"== {ds}", flush=True)
        for probe, items in sorted(flags(ds).items()):
            seen = Counter(have[(ds, probe)])
            rest = []
            for it in items:
                key = tuple(clean(it[f]) for f in ("labels", "flagged", "context"))
                if seen[key] > 0:
                    seen[key] -= 1
                else:
                    rest.append(it)
            n_old = sum(have[(ds, probe)].values())
            if len(items) != sizes.get(f"{ds}/{probe}", {}).get("flags", len(items)):
                raise SystemExit(f"{ds}/{probe}: {len(items)} flags now vs {sizes[f'{ds}/{probe}']['flags']} "
                                 "when the sheet was drawn -- probes changed, do not extend")
            if sum(seen.values()):
                raise SystemExit(f"{ds}/{probe}: {sum(seen.values())} sheet items not found among the flags")
            take = random.Random(f"{SEED}-{ds}-{probe}-ext").sample(rest, max(0, min(per_cell - n_old, len(rest))))
            sizes[f"{ds}/{probe}"] = {"flags": len(items), "sampled": n_old + len(take)}
            print(f"   {probe}: {len(items)} flags, {n_old} on sheet, +{len(take)}", flush=True)
            for k, it in enumerate(take, start=n_old):
                new.append({"id": f"{ds}-{probe}-{k:02d}", "corpus": ds, "probe": probe,
                            "question": QUESTION[probe], **{f: clean(v) for f, v in it.items()},
                            "annotator_1": "", "annotator_2": "", "note": ""})
    random.Random(f"{SEED}-ext").shuffle(new)
    for path, out in ((SHEET, rows + new), (OUT_DIR / "sample_ext.csv", new)):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(out)
    (OUT_DIR / "sizes.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")
    print(f"added {len(new)} items; sheet now {len(rows) + len(new)}")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))


def kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score() -> None:
    import json
    rows = list(csv.DictReader(SHEET.open(encoding="utf-8-sig")))
    done = [r for r in rows if r["annotator_1"].strip() and r["annotator_2"].strip()]
    print(f"{len(done)}/{len(rows)} items have both answers")
    cells = defaultdict(list)
    for r in done:
        cells[(r["corpus"], r["probe"])].append(r)
    sizes = json.loads((OUT_DIR / "sizes.json").read_text(encoding="utf-8"))
    out = {}
    print("| corpus | probe | n | agree % | kappa | problem % (both agree) [95% CI] | flags in corpus | est. real problems |")
    print("|---|---|---|---|---|---|---|---|")
    for (ds, probe), rs in sorted(cells.items()):
        a = [r["annotator_1"].strip().upper()[:1] for r in rs]
        b = [r["annotator_2"].strip().upper()[:1] for r in rs]
        agreed = [x for x, y in zip(a, b) if x == y]
        k = sum(x in PROBLEM[probe] for x in agreed)
        lo, hi = wilson(k, len(agreed))
        share = 100 * k / max(1, len(agreed))
        n_flags = sizes.get(f"{ds}/{probe}", {}).get("flags", 0)
        out[f"{ds}/{probe}"] = {"n": len(rs), "agree_pct": round(100 * len(agreed) / len(rs), 1),
                               "kappa": round(kappa(a, b), 3), "problem_pct": round(share, 1),
                               "ci": [round(lo, 1), round(hi, 1)], "flags": n_flags,
                               "est_real": round(n_flags * share / 100)}
        o = out[f"{ds}/{probe}"]
        print(f"| {ds} | {probe} | {o['n']} | {o['agree_pct']} | {o['kappa']} | {o['problem_pct']} "
              f"[{o['ci'][0]}, {o['ci'][1]}] | {n_flags} | {o['est_real']} |")
    (OUT_DIR / "validation_scores.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def fleiss(table: List[List[str]]) -> float:
    """Fleiss' kappa for items x raters (every item rated by the same number of raters)."""
    n_items, m = len(table), len(table[0]) if table else 0
    if n_items == 0 or m < 2:
        return float("nan")
    cats = sorted({x for row in table for x in row})
    p_i = [(sum(c * c for c in Counter(row).values()) - m) / (m * (m - 1)) for row in table]
    p_j = [sum(row.count(c) for row in table) / (n_items * m) for c in cats]
    pbar, pe = sum(p_i) / n_items, sum(p * p for p in p_j)
    # one category only: kappa is undefined (chance agreement is already 1)
    return (pbar - pe) / (1 - pe) if pe < 1 else float("nan")


def score_passes(files: List[str]) -> None:
    """Score independent model passes (PROTOCOL.md 9.16): majority label + agreement.

    Each file is an `id,answer,note` CSV from one pass over the blind, re-ordered sheet.
    Agreement between passes of one model measures its self-consistency, NOT agreement
    between human annotators, and is reported under that name.
    """
    import json
    from itertools import combinations
    rows = {r["id"]: r for r in csv.DictReader(SHEET.open(encoding="utf-8-sig"))}
    passes = []
    for f in files:
        got = {r["id"]: r["answer"].strip().upper()[:1] or "?" for r in csv.DictReader(open(f, encoding="utf-8-sig"))}
        missing = set(rows) - set(got)
        if missing or set(got) - set(rows):
            raise SystemExit(f"{f}: {len(missing)} ids missing, {len(set(got) - set(rows))} unknown")
        for i, a in got.items():
            allowed = {"O": "HB?", "C": "EL?", "U": "MSN?", "G": "AT?"}[rows[i]["probe"]]
            if a not in allowed:
                raise SystemExit(f"{f}: id {i} answer {a!r} not in {allowed}")
        passes.append(got)
    sizes = json.loads((OUT_DIR / "sizes.json").read_text(encoding="utf-8"))
    cells = defaultdict(list)
    for i, r in rows.items():
        cells[(r["corpus"], r["probe"])].append(i)
    out: Dict[str, dict] = {}

    def summarise(ids: List[str], probe: str, n_flags: int = 0) -> dict:
        table = [[p[i] for p in passes] for i in ids]
        maj = []
        for row in table:
            top, cnt = Counter(row).most_common(1)[0]
            maj.append(top if cnt * 2 > len(row) else None)
        decided = [x for x in maj if x is not None and x != "?"]
        k = sum(x in PROBLEM[probe] for x in decided)
        lo, hi = wilson(k, len(decided))
        share = 100 * k / max(1, len(decided))
        pair = [kappa([p[i] for i in ids], [q[i] for i in ids]) for p, q in combinations(passes, 2)]
        extra = {}
        if probe == "U":    # secondary, descriptive: real PII whether or not the schema has the type
            kp = sum(x in {"M", "S"} for x in decided)
            plo, phi = wilson(kp, len(decided))
            extra = {"k_pii": kp, "pii_pct": round(100 * kp / max(1, len(decided)), 1),
                     "pii_ci": [round(plo, 1), round(phi, 1)]}
        return {**extra, "k": k, "n": len(ids), "n_majority": len(decided),
                "unanimous_pct": round(100 * sum(len(set(r)) == 1 for r in table) / len(table), 1),
                "fleiss_kappa": round(fleiss(table), 3),
                "pairwise_kappa": [round(x, 3) for x in pair],
                "problem_pct": round(share, 1), "ci": [round(lo, 1), round(hi, 1)],
                "flags": n_flags, "est_real": round(n_flags * share / 100)}

    print("| corpus | probe | n | unanimous % | Fleiss k | problem % (majority) [95% CI] | flags | est. real |")
    print("|---|---|---|---|---|---|---|---|")
    for (ds, probe), ids in sorted(cells.items()):
        o = out[f"{ds}/{probe}"] = summarise(ids, probe, sizes.get(f"{ds}/{probe}", {}).get("flags", 0))
        print(f"| {ds} | {probe} | {o['n']} | {o['unanimous_pct']} | {o['fleiss_kappa']} | "
              f"{o['problem_pct']} [{o['ci'][0]}, {o['ci'][1]}] | {o['flags']} | {o['est_real']} |")
    for probe in "OCUG":
        ids = [i for i, r in rows.items() if r["probe"] == probe]
        o = out[f"all/{probe}"] = summarise(ids, probe)
        print(f"| all | {probe} | {o['n']} | {o['unanimous_pct']} | {o['fleiss_kappa']} | "
              f"{o['problem_pct']} [{o['ci'][0]}, {o['ci'][1]}] | | |")
    # Stratified estimate per probe (review W1): the sample is capped per (corpus, probe)
    # cell, so the pooled sample share is not a share of all flags. Weight each cell by its
    # number of flags; variance with finite-population correction, and a +0.5 adjustment
    # inside the variance only, so that 0% / 100% cells do not claim zero uncertainty.
    for probe in "OCUG":
        for field, kkey in (("strat", "k"), ("strat_pii", "k_pii")):
            cells_p = [v for key, v in out.items() if key.endswith(f"/{probe}") and not key.startswith("all/")
                       and kkey in v and v["n_majority"] > 0]
            if not cells_p:
                continue
            total = sum(v["flags"] for v in cells_p)
            est = var = 0.0
            for v in cells_p:
                w, n, big_n = v["flags"] / total, v["n_majority"], v["flags"]
                est += w * v[kkey] / n
                pt = (v[kkey] + 0.5) / (n + 1)
                fpc = max(0.0, 1 - n / big_n) if big_n else 0.0
                var += w * w * pt * (1 - pt) / n * fpc
            half = 1.96 * math.sqrt(var)
            out[f"all/{probe}"][f"{field}_pct"] = round(100 * est, 1)
            out[f"all/{probe}"][f"{field}_ci"] = [round(100 * max(0.0, est - half), 1),
                                                   round(100 * min(1.0, est + half), 1)]
            out[f"all/{probe}"][f"{field}_flags"] = total
        o = out[f"all/{probe}"]
        print(f"stratified {probe}: {o.get('strat_pct')} {o.get('strat_ci')}"
              + (f" | any PII {o.get('strat_pii_pct')} {o.get('strat_pii_ci')}" if probe == "U" else ""))
    out["_meta"] = {"raters": "independent passes of one LLM (Claude), blind to each other, "
                              "corpus name withheld, item order shuffled per pass",
                    "files": [Path(f).name for f in files],
                    "agreement": "self-consistency of one model, not inter-annotator agreement"}
    (OUT_DIR / "validation_scores.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'validation_scores.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--passes", nargs="+", default=None,
                    help="score independent model passes (id,answer,note CSVs) instead of the sheet")
    ap.add_argument("--extend", type=int, default=None, metavar="N",
                    help="top every cell up to N items, keeping the existing sheet and its answers")
    args = ap.parse_args()
    if args.extend:
        extend_sample(args.extend)
    elif args.passes:
        score_passes(args.passes)
    else:
        score() if args.score else write_sample(args.per_cell)


if __name__ == "__main__":
    main()
