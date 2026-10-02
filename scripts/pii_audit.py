# -*- coding: utf-8 -*-
"""Audit probes for the four converted PII benchmarks: do they measure what we think?

CPU only, no model.  Four measurements and one consequence test:

  S  support      how many types have too few test mentions for per-type F1 to mean anything
  L  leakage      test documents whose *template* (text with every gold span replaced by its
                  type) also occurs in train, and test mentions whose (type, value) was seen
                  in train -- synthetic corpora are generated from templates and value lists
  C  consistency  one surface string annotated with different types across the corpus
                  (e.g. "Ms." as FIRSTNAME)
  U  unannotated  checksum/regex-validated PII (email, URL, IP, IBAN, card) that no gold
                  span covers -- missing annotation, or a type outside the schema
  X  consequence  F1 of the existing (a)/(b)/(c) predictions on leaked vs unseen templates,
                  and recall on seen vs unseen values: does the flaw change the numbers?

    python scripts/pii_audit.py                          # the four converted corpora
    python scripts/pii_audit.py --datasets nemotron      # adds Nemotron-PII (nvidia/Nemotron-PII)
    python scripts/pii_audit.py --datasets ai4privacy_300k,ai4privacy_400k,gretel_finance,tab,mapa

Public releases in SPECS are audited as released (test = the split the release calls
validation/test).  Two extra probes: B (whitespace at span edges, overlapping spans) on every
corpus, and IAA (pairwise agreement between TAB's annotators of the same document).
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.rule_detector import detect_by_rules  # noqa: E402
from evaluation.metrics import HungarianEvaluator  # noqa: E402

DATASETS = ["privy", "pii_500k", "kiji", "gretel"]
ROOT = Path("out_pii_baselines")
OUT = ROOT / "pii_audit.json"
# Corpora that were not converted by prepare_pii_datasets.py: their own repo, and whatever
# predictions exist for the consequence test (no (a)/(b)/(c) runs on them).
REPO_OVERRIDE = {"nemotron": "nvidia/Nemotron-PII",          # the official release
                 "nemotron_anonymous-org_full": "anonymous-org/nemotron-full"}   # an earlier re-upload
_JULY = {"GLiNER2 uni+affine (July run on the anonymous-org re-upload, thr 0.7)":
         Path("archive/eval_output/nemotron-uni-affine/predictions/all.jsonl")}
EXTRA_PREDICTIONS = {"nemotron": _JULY, "nemotron_anonymous-org_full": _JULY}
RULE_TYPES = {"EMAIL", "URL", "IP", "IBAN", "CARD_NUMBER"}   # validated, high precision
ARMS = {"a": "v1-bce-baseline-50k", "b": "v1-bce-bundle-50k", "c": "v1-set-consistent-50k"}
STRICT = HungarianEvaluator(iou_threshold=1.0, fuzzy_max_dist=None)
Doc = Tuple[str, List[Tuple[int, int, str, str]]]


def _token() -> str:
    tok = os.environ.get("HF_TOKEN", "").strip()
    p = Path(".hf_token")
    return tok or (p.read_text(encoding="utf-8").strip() if p.is_file() else "")


def _parquet_rows(repo: str, split: str):
    """Rows of data/<split>-*.parquet via resumable downloads, or None if the repo has none.

    Streaming a 150 MB split over this connection has been reset mid-file; hf_hub_download
    resumes, and the file then sits in the HF cache for the next run.
    """
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd
    try:
        files = [f for f in HfApi(token=_token() or None).list_repo_files(repo, repo_type="dataset")
                 if f.startswith(f"data/{split}-") and f.endswith(".parquet")]
    except Exception:
        return None
    if not files:
        return None
    frames = []
    for f in sorted(files):
        for attempt in range(5):
            try:
                path = hf_hub_download(repo, f, repo_type="dataset", token=_token() or None)
                break
            except Exception as exc:  # connection reset: retry, the download resumes
                print(f"   retry {attempt + 1} for {f}: {type(exc).__name__}", flush=True)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames).to_dict("records")


# Public corpora audited as released (no conversion of ours).  role -> file selector.
SPECS = {
    "ai4privacy_300k": {"repo": "ai4privacy/pii-masking-300k", "kind": "jsonl",
                        "train": "data/train/", "test": "data/validation/"},
    "ai4privacy_400k": {"repo": "ai4privacy/pii-masking-400k", "kind": "jsonl",
                        "train": "data/train/", "test": "data/validation/"},
    "gretel_finance": {"repo": "gretelai/synthetic_pii_finance_multilingual", "kind": "parquet",
                       "train": "data/train-", "test": "data/test-"},   # default config; per-language and combined_* files repeat the same documents
    "tab": {"repo": "ildpil/text-anonymization-benchmark", "kind": "tab",
            "train": "echr_train.zip", "test": "echr_test.zip"},
    # MAPA is released as one test-only file (21 EU languages, 5 types): no leakage probe
    "mapa": {"repo": "piimb/mapa-eur-lex-pii", "kind": "mapa", "train": None,
             "test": "data/mapa-eur-lex.jsonl"},
}
NO_VALUE_FIELD = {"gretel_finance", "mapa"}
TAB_ALL: Dict[str, List[dict]] = {}      # role -> every annotator's row, for the IAA probe


def _download(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download
    for attempt in range(6):
        try:
            return hf_hub_download(repo, path, repo_type="dataset", token=_token() or None)
        except Exception as exc:   # connection resets on this link: retry, downloads resume
            print(f"   retry {attempt + 1} for {path}: {type(exc).__name__}", flush=True)
    raise RuntimeError(f"could not download {repo}/{path}")


def _repo_files(repo: str) -> List[str]:
    from huggingface_hub import HfApi
    return HfApi(token=_token() or None).list_repo_files(repo, repo_type="dataset")


def _parse_spans(raw, text: str, keys=("start", "end", "label")) -> List[Tuple[int, int, str, str]]:
    raw = (json.loads(raw) if raw.strip().startswith("[{\"") else ast.literal_eval(raw)) if isinstance(raw, str) else (raw or [])
    s_k, e_k, l_k = keys
    out = []
    for m in raw:
        s, e = int(m[s_k]), int(m[e_k])
        v = m.get("value", m.get("text", m.get("span_text")))
        out.append((s, e, str(m[l_k]).upper(), text[s:e] if v is None else str(v)))
    return sorted(out)


def load_spec(ds: str, role: str) -> List[Doc]:
    spec = SPECS[ds]
    sel = spec[role]
    if sel is None:
        return []
    repo, kind = spec["repo"], spec["kind"]
    docs: List[Doc] = []
    if kind == "jsonl":
        for f in sorted(p for p in _repo_files(repo) if p.startswith(sel) and p.endswith(".jsonl")):
            for line in open(_download(repo, f), encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    docs.append((r["source_text"], _parse_spans(r["privacy_mask"], r["source_text"])))
    elif kind == "parquet":
        import pandas as pd
        for f in sorted(p for p in _repo_files(repo) if p.startswith(sel) and p.endswith(".parquet")):
            for r in pd.read_parquet(_download(repo, f)).to_dict("records"):
                docs.append((r["generated_text"], _parse_spans(r["pii_spans"], r["generated_text"])))
    elif kind == "tab":
        import zipfile
        z = zipfile.ZipFile(_download(repo, sel))
        rows = json.loads(z.read(z.namelist()[0]))
        TAB_ALL[role] = rows
        first = {}
        for r in sorted(rows, key=lambda r: (r["doc_id"], r["annotator_id"])):
            first.setdefault(r["doc_id"], r)       # one annotator per document for S/L/C/U/B
        for r in first.values():
            docs.append((r["text"], _parse_spans(r["entity_mentions"], r["text"],
                                                 ("start_offset", "end_offset", "entity_type"))))
    elif kind == "mapa":
        for line in open(_download(repo, sel), encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                docs.append((r["sentence"], _parse_spans(r["entities"], r["sentence"])))
    return docs


def probe_boundaries(docs: List[Doc]) -> Dict:
    """B: spans with leading/trailing whitespace, and spans overlapping another span (a flat
    decoder can emit only one of them)."""
    n = ws = overlap = 0
    for text, spans in docs:
        for i, (s, e, _, _) in enumerate(spans):
            n += 1
            seg = text[s:e]
            ws += bool(seg) and (seg[0].isspace() or seg[-1].isspace())
            overlap += any(j != i and s < e2 and s2 < e for j, (s2, e2, _, _) in enumerate(spans))
    return {"spans": n, "edge_whitespace_pct": round(100 * ws / max(1, n), 3),
            "overlapping_pct": round(100 * overlap / max(1, n), 3)}


def probe_iaa(rows: List[dict]) -> Dict:
    """IAA (TAB): pairwise agreement between annotators of the same document.
    exact = same offsets and type; span = same offsets, any type. F1 of one vs the other."""
    by_doc: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    exact, span = [], []
    for anns in by_doc.values():
        sets = [({(m["start_offset"], m["end_offset"], m["entity_type"]) for m in a["entity_mentions"]},
                 {(m["start_offset"], m["end_offset"]) for m in a["entity_mentions"]}) for a in anns]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                for store, k in ((exact, 0), (span, 1)):
                    a, b = sets[i][k], sets[j][k]
                    if a or b:
                        store.append(100 * 2 * len(a & b) / (len(a) + len(b)))
    return {"docs_multi": sum(len(v) >= 2 for v in by_doc.values()), "pairs": len(exact),
            "exact_f1_mean": round(statistics.mean(exact), 2) if exact else None,
            "span_only_f1_mean": round(statistics.mean(span), 2) if span else None}


def load_split(ds: str, split: str) -> List[Doc]:
    if ds in SPECS:
        return load_spec(ds, split)
    from datasets import load_dataset
    from config.ner_datasets import NER_DATASETS
    repo = REPO_OVERRIDE.get(ds) or NER_DATASETS[ds]["repo"]
    data = _parquet_rows(repo, split)
    if data is None:
        data = load_dataset(repo, split=split, token=_token() or None, streaming=True)
    docs = []
    for row in data:
        if "privacy_mask" in row:     # prepare_pii_datasets.py schema
            mask = row.get("privacy_mask")   # a numpy array when read from parquet
            spans = [(int(m["start"]), int(m["end"]), m["label"], m.get("value") or "")
                     for m in ([] if mask is None else list(mask))]
            text = row["source_text"]
        else:                         # nvidia/Nemotron-PII: `spans` is a Python-repr string
            raw = row.get("spans") or []
            raw = ast.literal_eval(raw) if isinstance(raw, str) else raw
            # some span `text` fields are ints (e.g. an age) -- keep them as strings
            spans = [(int(m["start"]), int(m["end"]), str(m["label"]).upper(),
                      "" if m.get("text") is None else str(m["text"])) for m in raw]
            text = row["text"]
        docs.append((text, sorted(spans)))
    return docs


def template(text: str, spans) -> str:
    out, last = [], 0
    for s, e, lab, _ in sorted(spans):
        if s < last:
            continue
        out.append(text[last:s]); out.append(f"<{lab}>"); last = e
    out.append(text[last:])
    t = re.sub(r"\d", "0", "".join(out).lower())
    return re.sub(r"\s+", " ", t).strip()


def probe_offsets(docs: List[Doc]) -> Dict:
    """O: spans whose offsets do not reproduce their annotated value."""
    n = bad = 0
    examples = []
    for text, spans in docs:
        for s, e, lab, v in spans:
            if not v:
                continue
            n += 1
            if text[s:e] != v:
                bad += 1
                if len(examples) < 5:
                    examples.append({"label": lab, "value": v[:40], "text_at_offsets": text[s:e][:40]})
    return {"spans_checked": n, "mismatch": bad, "mismatch_pct": round(100 * bad / max(1, n), 3),
            "examples": examples}


def probe_support(test: List[Doc]) -> Dict:
    sup = Counter(lab for _, spans in test for _, _, lab, _ in spans)
    order = sorted(sup, key=sup.get)
    rare = order[:max(1, math.ceil(len(order) / 2))]
    return {"n_types": len(sup), "support": dict(sorted(sup.items(), key=lambda kv: kv[1])),
            "types_lt_10": sum(v < 10 for v in sup.values()), "types_lt_20": sum(v < 20 for v in sup.values()),
            "types_lt_50": sum(v < 50 for v in sup.values()),
            "rare_half_min_support": min(sup[l] for l in rare), "rare_half_median_support": statistics.median(sup[l] for l in rare)}


def probe_leakage(train: List[Doc], test: List[Doc]) -> Tuple[Dict, set, set]:
    train_text = {re.sub(r"\s+", " ", t).strip() for t, _ in train}
    train_tpl = {template(t, s) for t, s in train}
    train_vals = {(lab, v.strip().lower()) for _, s in train for _, _, lab, v in s if v.strip()}
    leaked_tpl = {t for t, s in test if template(t, s) in train_tpl}
    n_ment = n_seen = 0
    for _, s in test:
        for _, _, lab, v in s:
            n_ment += 1
            n_seen += (lab, v.strip().lower()) in train_vals
    return ({"exact_text_dup_pct": round(100 * sum(re.sub(r"\s+", " ", t).strip() in train_text for t, _ in test) / len(test), 2),
             "template_leak_pct": round(100 * len([1 for t, s in test if template(t, s) in train_tpl]) / len(test), 2),
             "value_seen_pct": round(100 * n_seen / max(1, n_ment), 2),
             "n_train_templates": len(train_tpl), "n_train": len(train)},
            leaked_tpl, train_vals)


def probe_consistency(train: List[Doc], test: List[Doc]) -> Dict:
    by_val: Dict[str, Counter] = defaultdict(Counter)
    for _, s in train + test:
        for _, _, lab, v in s:
            if v.strip():
                by_val[v.strip().lower()][lab] += 1
    total = incons = 0
    worst = []
    for v, c in by_val.items():
        n = sum(c.values())
        if n < 3 or len(v) < 2:
            continue
        maj = c.most_common(1)[0][1]
        total += n
        incons += n - maj
        if len(c) > 1:
            worst.append((n - maj, v, dict(c.most_common(3))))
    worst.sort(reverse=True)
    return {"mentions_in_repeated_values": total, "non_majority_label_pct": round(100 * incons / max(1, total), 2),
            "top_inconsistent": [{"value": v, "labels": c} for _, v, c in worst[:8]]}


# Which corpus labels a rule type corresponds to.  An uncovered hit whose type has no such
# label is PII the schema does not define (e.g. URLs in ai4privacy-400k), not a missed annotation.
RULE_SCHEMA = {"URL": r"^URL$", "EMAIL": r"EMAIL", "IP": r"^(IP|IPV4|IPV6|IP_ADDRESS)$", "IBAN": r"^IBAN$",
               "CARD_NUMBER": r"^(CREDIT_CARD|CREDITCARDNUMBER|CREDIT_CARD_NUMBER|CREDIT_DEBIT_CARD)$"}


def probe_unannotated(test: List[Doc], schema: set) -> Dict:
    hits = unannot = in_schema = 0
    per_type, examples = Counter(), []
    has = {t: any(re.search(rx, l) for l in schema) for t, rx in RULE_SCHEMA.items()}
    for text, spans in test:
        for m in detect_by_rules(text, RULE_TYPES):
            hits += 1
            if not any(m["start"] < e and s < m["end"] for s, e, _, _ in spans):
                unannot += 1
                in_schema += has[m["label"]]
                per_type[m["label"]] += 1
                if len(examples) < 6:
                    examples.append({"type": m["label"], "value": m["text"][:60]})
    chars = max(1, sum(len(t) for t, _ in test))
    return {"rule_hits": hits, "uncovered_hits": unannot, "docs": len(test),
            "uncovered_per_1k_docs": round(1000 * unannot / len(test), 2),
            # documents range from one sentence (MAPA) to a whole judgment (TAB): also per text length
            "uncovered_per_1M_chars": round(1e6 * unannot / chars, 2),
            "schema_has": has,
            "uncovered_in_schema": in_schema, "uncovered_out_of_schema": unannot - in_schema,
            "in_schema_per_1k_docs": round(1000 * in_schema / len(test), 2),
            "out_of_schema_per_1k_docs": round(1000 * (unannot - in_schema) / len(test), 2),
            "uncovered_by_type": dict(per_type), "examples": examples}


NULL_VALUES = {"nan", "none", "null", "n/a", "undefined"}
# generator slot markup: [USERNAME], [GIVENNAME1], TIME_BG(..., {{name}}; >=4 chars so [EUR] is not one
SLOT_RX = re.compile(r"\[[A-Z][A-Z0-9_]{3,}\]|\b[A-Z][A-Z0-9]*_[A-Z]{2}\(|\{\{\s*\w+\s*\}\}")


# bracketed tokens that are real text, not slots: log levels ([ERROR] in a log line)
NOT_SLOTS = {"[ERROR]", "[INFO]", "[WARNING]", "[WARN]", "[DEBUG]", "[CRITICAL]", "[FATAL]", "[TRACE]", "[NOTICE]"}


def _slots(text: str):
    return [m for m in SLOT_RX.finditer(text) if m.group(0) not in NOT_SLOTS]


def probe_artifacts(test: List[Doc]) -> Dict:
    """G: generator artifacts in test -- gold values that are null tokens or carry slot markup,
    and documents whose text still contains an unfilled slot."""
    n = null = markup = 0
    docs_slot, ex = 0, []
    kinds = Counter()        # slot strings with digits collapsed, e.g. [IPV#_#]
    for text, spans in test:
        found = _slots(text)
        slot = found[0] if found else None
        kinds.update(re.sub(r"\d+", "#", m.group(0)) for m in found)
        if slot:
            docs_slot += 1
            if len(ex) < 4:
                ex.append(text[max(0, slot.start() - 40):slot.end() + 20].replace("\n", " "))
        for s, e, _, v in spans:
            n += 1
            val = v or text[s:e]
            null += val.strip().lower() in NULL_VALUES
            markup += bool(_slots(val))
    return {"spans": n, "null_value_pct": round(100 * null / max(1, n), 3), "null_values": null,
            "markup_value_pct": round(100 * markup / max(1, n), 3), "markup_values": markup,
            "docs_with_unfilled_slot_pct": round(100 * docs_slot / max(1, len(test)), 3),
            "docs_with_unfilled_slot": docs_slot, "slot_examples": ex,
            "top_slots": dict(kinds.most_common(8))}


def consequence(ds: str, test: List[Doc], leaked_tpl: set, train_vals: set) -> Dict:
    """Existing (a)/(b)/(c) predictions at p0=0.5, split by template leakage / value novelty."""
    leak_text = {re.sub(r"\s+", " ", t).strip() for t in leaked_tpl}
    sources = EXTRA_PREDICTIONS.get(ds) and {k: [v] for k, v in EXTRA_PREDICTIONS[ds].items()} or {
        arm: [ROOT / proto / "gliner2" / ds / f"seed-{s}" / "test" / "predictions" / "all.jsonl" for s in (42, 1337)]
        for arm, proto in ARMS.items()}
    out = {}
    for arm, paths in sources.items():
        per_seed = []
        for p in paths:
            if not p.is_file():
                continue
            agg = {k: Counter() for k in ("leaked", "unseen")}
            rec = {"seen": [0, 0], "unseen": [0, 0]}
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                gt, preds = r.get("gt") or [], [x for x in (r.get("predictions") or []) if x.get("score", 1) >= 0.5]
                key = "leaked" if re.sub(r"\s+", " ", r["text"]).strip() in leak_text else "unseen"
                tp, fp, fn, _ = STRICT.count_example(gt, preds)
                agg[key].update({"tp": sum(tp.values()), "fp": sum(fp.values()), "fn": sum(fn.values()), "docs": 1})
                pset = {(x["start"], x["end"], x["label"]) for x in preds}
                for g in gt:
                    k = "seen" if (g["tag"], (g.get("value") or "").strip().lower()) in train_vals else "unseen"
                    rec[k][0] += (g["start"], g["end"], g["tag"]) in pset
                    rec[k][1] += 1
            f1 = lambda c: 100 * 2 * c["tp"] / max(1, 2 * c["tp"] + c["fp"] + c["fn"])
            per_seed.append({"micro_leaked": f1(agg["leaked"]), "micro_unseen": f1(agg["unseen"]),
                             "docs_leaked": agg["leaked"]["docs"], "docs_unseen": agg["unseen"]["docs"],
                             "recall_seen_value": 100 * rec["seen"][0] / max(1, rec["seen"][1]),
                             "recall_unseen_value": 100 * rec["unseen"][0] / max(1, rec["unseen"][1])})
        if per_seed:
            out[arm] = {k: round(statistics.mean(s[k] for s in per_seed), 2) for k in per_seed[0]}
    return out


TABLE_ORDER = ["privy", "pii_500k", "kiji", "gretel", "nemotron", "ai4privacy_300k", "ai4privacy_400k",
               "gretel_finance", "tab", "mapa"]


def table(result: Dict) -> str:
    """One markdown row per audited corpus, from pii_audit.json."""
    def g(d, k):
        return "n/a" if not d or "na" in d or d.get(k) is None else str(d[k])
    head = ("| corpus | train / test docs | types | types <20 test | rare-half min supp. | O offset mism. % "
            "| template leak % | value seen % | C non-majority % | U in-schema /1k docs | U out-of-schema /1k docs "
            "| G null+markup values % | G unfilled-slot docs % | B overlap % | IAA exact F1 |")
    rows = [head, "|" + "---|" * (head.count("|") - 1)]
    for ds in [d for d in TABLE_ORDER if d in result]:   # nemotron_anonymous-org_full: appendix only
        r = result[ds]
        s, l, u = r["S_support"], r["L_leakage"], r["U_unannotated"]
        iaa = r.get("IAA", {}).get("test")
        G = r.get("G_artifacts")
        rows.append(f"| {ds} | {r.get('n_train', l.get('n_train', '?'))} / {r.get('n_test', u['docs'])} | {s['n_types']} "
                    f"| {s['types_lt_20']} | {s['rare_half_min_support']} | {g(r.get('O_offsets'), 'mismatch_pct')} "
                    f"| {g(l, 'template_leak_pct')} | {g(l, 'value_seen_pct')} "
                    f"| {r['C_consistency']['non_majority_label_pct']} | {g(u, 'in_schema_per_1k_docs')} "
                    f"| {g(u, 'out_of_schema_per_1k_docs')} "
                    f"| {round(G['null_value_pct'] + G['markup_value_pct'], 3) if G else 'n/a'} "
                    f"| {g(G, 'docs_with_unfilled_slot_pct')} | {g(r.get('B_boundaries'), 'overlapping_pct')} "
                    f"| {iaa['exact_f1_mean'] if iaa else 'n/a'} |")
    return "\n".join(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help="comma list; also 'nemotron' (anonymous-org/nemotron-full)")
    ap.add_argument("--table", action="store_true", help="print the combined table from pii_audit.json and exit")
    ap.add_argument("--only-g", action="store_true",
                    help="recompute only the G probe (test split) and keep every other field")
    args = ap.parse_args()
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}
    if args.table:
        (ROOT / "pii_audit_table.md").write_text(table(result) + "\n", encoding="utf-8")
        print(table(result))
        return
    if args.only_g:
        for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            result.setdefault(ds, {})["G_artifacts"] = gg = probe_artifacts(load_split(ds, "test"))
            print(f"{ds}: null {gg['null_values']}, markup {gg['markup_values']}, docs with unfilled slot "
                  f"{gg['docs_with_unfilled_slot']} ({gg['docs_with_unfilled_slot_pct']}%)", flush=True)
            OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"== {ds}: loading train/test ...", flush=True)
        train, test = load_split(ds, "train"), load_split(ds, "test")
        print(f"   {len(train)} train / {len(test)} test documents", flush=True)
        if train:
            leak, leaked_tpl, train_vals = probe_leakage(train, test)
        else:   # test-only release
            leak, leaked_tpl, train_vals = {"na": "no train split released"}, set(), set()
        result[ds] = {"n_train": len(train), "n_test": len(test),
                      "O_offsets": ({"na": "spans carry no value field"} if ds in NO_VALUE_FIELD
                                    else probe_offsets(train + test)),
                      "S_support": probe_support(test), "L_leakage": leak,
                      "C_consistency": probe_consistency(train, test),
                      "U_unannotated": probe_unannotated(test, {l for _, sp in train + test for _, _, l, _ in sp}),
                      "B_boundaries": probe_boundaries(test), "G_artifacts": probe_artifacts(test),
                      "X_consequence": consequence(ds, test, leaked_tpl, train_vals) if train else {}}
        if ds in TAB_ALL or ds == "tab":
            result[ds]["IAA"] = {role: probe_iaa(rows) for role, rows in TAB_ALL.items()}
        r = result[ds]
        o = r["O_offsets"]
        if "na" in o:
            print(f"   O: n/a ({o['na']})")
        else:
            print(f"   O: {o['mismatch']}/{o['spans_checked']} spans whose offsets do not give their value "
                  f"({o['mismatch_pct']}%) {o['examples'][:2]}")
        print(f"   S: {r['S_support']['n_types']} types, <20 test mentions: {r['S_support']['types_lt_20']}, "
              f"rare-half min/median support {r['S_support']['rare_half_min_support']}/{r['S_support']['rare_half_median_support']}")
        if "na" in leak:
            print(f"   L: n/a ({leak['na']})")
        else:
            print(f"   L: exact dup {leak['exact_text_dup_pct']}%, template leak {leak['template_leak_pct']}%, "
                  f"test mentions with value seen in train {leak['value_seen_pct']}%")
        print(f"   C: non-majority label {r['C_consistency']['non_majority_label_pct']}% of repeated-value mentions; "
              f"top: {[(x['value'], x['labels']) for x in r['C_consistency']['top_inconsistent'][:3]]}")
        u = r["U_unannotated"]
        print(f"   U: {u['uncovered_hits']}/{u['rule_hits']} validated PII hits not covered by gold "
              f"({u['uncovered_per_1k_docs']} per 1k docs; in schema {u['in_schema_per_1k_docs']}, "
              f"outside schema {u['out_of_schema_per_1k_docs']}) {u['uncovered_by_type']}")
        gg = r["G_artifacts"]
        print(f"   G: null-token values {gg['null_values']} ({gg['null_value_pct']}%), slot markup in values "
              f"{gg['markup_values']} ({gg['markup_value_pct']}%), docs with unfilled slot "
              f"{gg['docs_with_unfilled_slot']} ({gg['docs_with_unfilled_slot_pct']}%) {gg['slot_examples'][:2]}")
        b = r["B_boundaries"]
        print(f"   B: {b['edge_whitespace_pct']}% of test spans start/end on whitespace, "
              f"{b['overlapping_pct']}% overlap another span")
        for role, v in r.get("IAA", {}).items():
            print(f"   IAA ({role}): {v['pairs']} annotator pairs on {v['docs_multi']} docs, "
                  f"exact F1 {v['exact_f1_mean']}, span-only F1 {v['span_only_f1_mean']}")
        for arm, v in r["X_consequence"].items():
            print(f"   X ({arm}): micro leaked {v['micro_leaked']} ({v['docs_leaked']} docs) vs unseen {v['micro_unseen']} "
                  f"({v['docs_unseen']}) | recall seen-value {v['recall_seen_value']} vs unseen-value {v['recall_unseen_value']}")
        OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")   # after each corpus
    print("wrote", OUT)


if __name__ == "__main__":
    main()
