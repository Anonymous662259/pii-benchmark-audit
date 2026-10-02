#!/usr/bin/env python3
"""Convert four public PII corpora into this repo's NER schema and push them to the Hub.

Target schema (what ``data/normalizer.normalize_record_generic`` reads):

    {"source_text": str,
     "privacy_mask": [{"start": int, "end": int, "label": str, "value": str}, ...],
     "language": str}

Sources and what had to be repaired for each:

    privy     piimb/privy (config ``privy-small``) - char offsets are exact, but the
              corpus labels every non-Presidio span ``O``; those are dropped.  ``full_text``
              is a *bytes repr* (``b'...'``); it is kept verbatim because the offsets are
              measured on that string and unescaping it would move every span.
    kiji      DataikuNLP/kiji-pii-training-data - already start/end/label/value, only the
              column name differs.  Six languages.
    gretel    gretelai/gretel-pii-masking-en-v1 - has NO offsets, only a stringified list of
              ``{entity, types}``.  Offsets are recovered by exact search; measured on 5k
              rows, every entity string occurs exactly once (0 missing, 0 ambiguous), so the
              recovery is unique rather than a heuristic.  Spans where that fails are dropped.
    pii_500k  ai4privacy/open-pii-masking-500k-ai4privacy - already in the target schema.
              Has no test split, so ``test`` and ``validation`` are carved disjointly out of
              the upstream ``validation`` split.

Every span is verified with ``text[start:end] == value`` before it is kept, and its edges are
pulled off whitespace (see ``_trim_span``: upstream gold contains values like ``'07180 00760 '``
that GLiNER's tokenizer cannot align to, and fixing that per-family would give the two model
families different targets).  The per-dataset report prints how many were dropped and trimmed.  Label sets are measured from the (capped) train split
and labels below ``--min-label-count`` are dropped from every split - the tail of these
corpora is annotation typos (kiji has ``BUILDNUM``/``COLONY``/``NHSNUMBER`` with 1-2
occurrences) which would otherwise sit in the schema as permanently-zero classes.

Usage:
    python scripts/prepare_pii_datasets.py --datasets privy --max-train 200
    python scripts/prepare_pii_datasets.py --push --hf-user anonymous-org
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Upstream language names -> ISO codes, so `language` means the same thing in every repo.
_LANG_CODE = {
    "English": "en", "French": "fr", "German": "de", "Dutch": "nl",
    "Spanish": "es", "Danish": "da", "Italian": "it", "Swedish": "sv",
    "France": "fr",   # gretel_finance spells its French subset this way
}

DEFAULT_MAX_TRAIN = 50_000
DEFAULT_MAX_VAL = 2_500
DEFAULT_MAX_TEST = 5_000
DEFAULT_MIN_LABEL_COUNT = 20
# Train-split length cap, in characters. privy carries dense XML/SQL rows -- 4,699 chars in
# only 264 words -- which explode into thousands of subword tokens and made both trainers
# log OOM and *skip the batch*, silently dropping exactly the longest documents at a rate
# that depends on the GPU. Capping the train split makes that loss explicit, identical for
# both model families, and stable for the set-consistent runs that reuse this data later.
# gretel/kiji/pii_500k max out at 1,750/1,149/963 chars, so only privy is affected.
DEFAULT_MAX_TRAIN_CHARS = 2000
SEED = 42


# -----------------------------------------------------------------------------
# Per-source row converters: raw HF example -> target schema (or None to drop)
# -----------------------------------------------------------------------------

def _convert_privy(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = row.get("full_text") or ""
    mask = []
    for span in row.get("spans") or []:
        label = (span.get("entity_type") or "").strip()
        if not label or label == "O":
            continue
        mask.append({
            "start": int(span["start_position"]),
            "end": int(span["end_position"]),
            "label": label,
            "value": span.get("entity_value") or "",
        })
    return {"source_text": text, "privacy_mask": mask, "language": "en"}


def _convert_kiji(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = row.get("text") or ""
    mask = [
        {"start": int(m["start"]), "end": int(m["end"]),
         "label": (m.get("label") or "").strip(), "value": m.get("value") or ""}
        for m in row.get("privacy_mask") or []
        if (m.get("label") or "").strip()
    ]
    lang = row.get("language") or ""
    return {"source_text": text, "privacy_mask": mask,
            "language": _LANG_CODE.get(lang, lang.lower())}


def _convert_gretel(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Recover char offsets by exact search; drop a span when it is not unique."""
    text = row.get("text") or ""
    raw = row.get("entities")
    if isinstance(raw, str):
        try:
            entities = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    else:
        entities = raw or []

    spans: List[Dict[str, Any]] = []
    for entity in entities:
        value = str(entity.get("entity") or "").strip()
        types = entity.get("types") or []
        if not value or not types:
            continue
        label = str(types[0]).upper()
        start = text.find(value)
        if start < 0 or text.find(value, start + 1) >= 0:
            # missing, or ambiguous (>1 occurrence) -> no defensible offset
            continue
        spans.append({"start": start, "end": start + len(value),
                      "label": label, "value": value})

    # `name` and `first_name` can both be annotated over the same characters; keeping
    # both would make one of them an unavoidable FP. Longest span wins.
    spans.sort(key=lambda s: (s["start"] - s["end"], s["start"]))
    kept: List[Dict[str, Any]] = []
    for span in spans:
        if any(span["start"] < k["end"] and k["start"] < span["end"] for k in kept):
            continue
        kept.append(span)
    kept.sort(key=lambda s: (s["start"], s["end"]))
    return {"source_text": text, "privacy_mask": kept, "language": "en"}


def _convert_500k(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = row.get("source_text") or ""
    mask = [
        {"start": int(m["start"]), "end": int(m["end"]),
         "label": (m.get("label") or "").strip(), "value": m.get("value") or ""}
        for m in row.get("privacy_mask") or []
        if (m.get("label") or "").strip()
    ]
    return {"source_text": text, "privacy_mask": mask, "language": row.get("language") or ""}


# --- the six corpora audited in PROTOCOL.md 9.4 (acl2023_pii_audit.tex) ------------------

def _mask_from(raw, text: str, s_key="start", e_key="end", l_key="label",
               v_key: Optional[str] = "value") -> List[Dict[str, Any]]:
    """Normalise one span list; a missing value is read from the offsets (then verified)."""
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.lstrip().startswith("[{\"") else ast.literal_eval(raw)
    mask = []
    for m in raw or []:
        label = str(m.get(l_key) or "").strip().upper()
        if not label:
            continue
        start, end = int(m[s_key]), int(m[e_key])
        value = m.get(v_key) if v_key else None
        value = text[start:end] if value is None else str(value)
        mask.append({"start": start, "end": end, "label": label, "value": value})
    return mask


def _convert_ai4p(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """ai4privacy pii-masking-300k/400k: already source_text + privacy_mask.  300k has values
    that still carry generator markup (TIME_BG(...); their offsets fail the check and drop."""
    text = row.get("source_text") or ""
    lang = row.get("language") or ""
    return {"source_text": text, "privacy_mask": _mask_from(row.get("privacy_mask"), text),
            "language": _LANG_CODE.get(lang, lang.lower())}


def _convert_gretel_finance(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """pii_spans is a JSON string of {start, end, label} without values."""
    text = row.get("generated_text") or ""
    lang = row.get("language") or ""
    return {"source_text": text, "privacy_mask": _mask_from(row.get("pii_spans"), text, v_key=None),
            "language": _LANG_CODE.get(lang, lang.lower())}


def _convert_nemotron(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """nvidia/Nemotron-PII: `spans` is a Python repr; 0.77% of values differ from the text
    only in letter case (measured), so a case-only difference takes the text's spelling."""
    text = row.get("text") or ""
    mask = _mask_from(row.get("spans"), text, v_key="text")
    for span in mask:
        seg = text[span["start"]:span["end"]]
        if seg != span["value"] and seg.lower() == span["value"].lower():
            span["value"] = seg
    return {"source_text": text, "privacy_mask": mask, "language": "en"}


def _convert_tab(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = row.get("text") or ""
    return {"source_text": text,
            "privacy_mask": _mask_from(row.get("entity_mentions"), text, "start_offset", "end_offset",
                                       "entity_type", "span_text"),
            "language": "en"}


def _convert_mapa(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = row.get("sentence") or ""
    return {"source_text": text, "privacy_mask": _mask_from(row.get("entities"), text, v_key=None),
            "language": row.get("language") or ""}


def _hub_file(repo: str, path: str) -> str:
    """hf_hub_download with retries (this link resets long downloads); files are cached."""
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN") or None
    for attempt in range(6):
        try:
            return hf_hub_download(repo, path, repo_type="dataset", token=token)
        except Exception as exc:
            print(f"    retry {attempt + 1} for {path}: {type(exc).__name__}")
    raise RuntimeError(f"could not download {repo}/{path}")


def _hub_list(repo: str) -> List[str]:
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(repo, repo_type="dataset")


def _files_loader(repo: str, pattern: str, kind: str):
    """split -> datasets.Dataset from the release's own files (pattern gets {split})."""
    def load(split: str):
        import pandas as pd
        from datasets import Dataset
        prefix = pattern.format(split=split)
        files = sorted(f for f in _hub_list(repo) if f.startswith(prefix) and f.endswith(kind))
        if not files:
            raise FileNotFoundError(f"{repo}: no {kind} files under {prefix}")
        if kind == ".jsonl":
            rows = [json.loads(line) for f in files
                    for line in open(_hub_file(repo, f), encoding="utf-8") if line.strip()]
            return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)
        frame = pd.concat([pd.read_parquet(_hub_file(repo, f)) for f in files])
        return Dataset.from_pandas(frame, preserve_index=False)
    return load


def _tab_loader(split: str):
    """TAB ships one row per (document, annotator); keep the first annotator by id, as the
    audit does (PROTOCOL.md 9.4) -- the other annotations are for the agreement probe."""
    import zipfile
    from datasets import Dataset
    z = zipfile.ZipFile(_hub_file("ildpil/text-anonymization-benchmark", f"echr_{split}.zip"))
    rows = json.loads(z.read(z.namelist()[0]))
    first: Dict[str, Any] = {}
    for r in sorted(rows, key=lambda r: (r["doc_id"], r["annotator_id"])):
        first.setdefault(r["doc_id"], r)
    return Dataset.from_list([{"text": r["text"], "entity_mentions": json.dumps(r["entity_mentions"])}
                              for r in first.values()])


# -----------------------------------------------------------------------------
# Source registry
# -----------------------------------------------------------------------------
# `splits`: target split -> upstream split taken from the front of the shuffle.
# `carve` : target split -> upstream split it must share, taken AFTER the rows the
#           other target split already consumed, so the two never overlap.

SOURCES: Dict[str, Dict[str, Any]] = {
    "privy": {
        "repo": "piimb/privy",
        "config": "privy-small",
        "convert": _convert_privy,
        "splits": {"train": "train", "validation": "dev", "test": "test"},
        "carve": {},
        "hub_name": "pii-privy",
    },
    "kiji": {
        "repo": "DataikuNLP/kiji-pii-training-data",
        "config": None,
        "convert": _convert_kiji,
        "splits": {"train": "train", "test": "test"},
        "carve": {"validation": "train"},
        "hub_name": "pii-kiji",
    },
    "gretel": {
        "repo": "gretelai/gretel-pii-masking-en-v1",
        "config": None,
        "convert": _convert_gretel,
        "splits": {"train": "train", "validation": "validation", "test": "test"},
        "carve": {},
        "hub_name": "pii-gretel",
    },
    "pii_500k": {
        "repo": "ai4privacy/open-pii-masking-500k-ai4privacy",
        "config": None,
        "convert": _convert_500k,
        "splits": {"train": "train", "test": "validation"},
        "carve": {"validation": "validation"},
        "hub_name": "pii-500k",
    },
    # ---- audited corpora (PROTOCOL.md 9.4); the release's test/validation is our test ----
    "ai4p_300k": {
        "repo": "ai4privacy/pii-masking-300k", "config": None, "convert": _convert_ai4p,
        "loader": _files_loader("ai4privacy/pii-masking-300k", "data/{split}/", ".jsonl"),
        "splits": {"train": "train", "test": "validation"}, "carve": {"validation": "train"},
        "hub_name": "pii-ai4p-300k",
    },
    "ai4p_400k": {
        "repo": "ai4privacy/pii-masking-400k", "config": None, "convert": _convert_ai4p,
        "loader": _files_loader("ai4privacy/pii-masking-400k", "data/{split}/", ".jsonl"),
        "splits": {"train": "train", "test": "validation"}, "carve": {"validation": "train"},
        "hub_name": "pii-ai4p-400k",
    },
    "gretel_finance": {
        # default configuration only: the per-language and combined_* files repeat its rows
        "repo": "gretelai/synthetic_pii_finance_multilingual", "config": None,
        "convert": _convert_gretel_finance,
        "loader": _files_loader("gretelai/synthetic_pii_finance_multilingual", "data/{split}-", ".parquet"),
        "splits": {"train": "train", "test": "test"}, "carve": {"validation": "train"},
        "hub_name": "pii-gretel-finance",
    },
    "nemotron_pii": {
        "repo": "nvidia/Nemotron-PII", "config": None, "convert": _convert_nemotron,
        "loader": _files_loader("nvidia/Nemotron-PII", "data/{split}-", ".parquet"),
        "splits": {"train": "train", "test": "test"}, "carve": {"validation": "train"},
        "hub_name": "pii-nemotron",
    },
    "tab": {
        # whole ECHR judgments (thousands of characters): the train length cap would drop
        # every document, so it is off; TAB is used for evaluation (the evaluator chunks)
        "repo": "ildpil/text-anonymization-benchmark", "config": None, "convert": _convert_tab,
        "loader": _tab_loader, "max_train_chars": 0,
        "splits": {"train": "train", "validation": "dev", "test": "test"}, "carve": {},
        "hub_name": "pii-tab",
    },
    "mapa": {
        # released as a test set only: no train/validation, labels are taken from test
        "repo": "piimb/mapa-eur-lex-pii", "config": None, "convert": _convert_mapa,
        "loader": lambda split: _mapa_rows(),
        "splits": {"test": "test"}, "carve": {},
        "hub_name": "pii-mapa",
    },
}


def _mapa_rows():
    from datasets import Dataset
    path = _hub_file("piimb/mapa-eur-lex-pii", "data/mapa-eur-lex.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return Dataset.from_list([{"sentence": r["sentence"], "entities": json.dumps(r["entities"]),
                               "language": r["language"]} for r in rows])


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def _load(repo: str, config: Optional[str], split: str, loader=None):
    if loader is not None:            # release files read directly (see SOURCES)
        return loader(split)
    from datasets import load_dataset
    return load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)


_WHITESPACE = re.compile(r"\s")


def _trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """Pull span edges off whitespace.

    GLiNER's converter tokenizes on ``\\S+`` and cuts those tokens at gold boundaries, so a
    boundary sitting inside whitespace has no token to align to and
    ``data/gliner_converter.to_gliner_example`` raises.  Upstream gold has such spans:
    ai4privacy ``'07180 00760 '``, privy ``'... International '``, kiji ``'Belgi\\x1e'``
    (Python's ``\\s`` counts U+001E as whitespace).  Trimming here, in the shared data, keeps
    GLiNER and GLiNER2 on byte-identical gold -- fixing it only in the GLiNER converter would
    give the two families different targets.
    """
    while start < end and _WHITESPACE.match(text[start]):
        start += 1
    while end > start and _WHITESPACE.match(text[end - 1]):
        end -= 1
    return start, end


def _convert_split(raw, convert, cap: int, offset: int = 0
                   ) -> Tuple[List[Dict[str, Any]], int, int]:
    """Shuffle deterministically, take `cap` rows starting at `offset`, convert, verify."""
    from tqdm import tqdm

    raw = raw.shuffle(seed=SEED)
    if offset >= len(raw):
        return [], 0, 0
    stop = min(len(raw), offset + cap)
    raw = raw.select(range(offset, stop))

    rows: List[Dict[str, Any]] = []
    dropped_spans = kept_spans = trimmed_spans = 0
    for example in tqdm(raw, desc=f"  convert[{offset}:{stop}]", leave=False):
        row = convert(example)
        if row is None or not row["source_text"].strip():
            continue
        text = row["source_text"]
        verified = []
        seen = set()
        for span in row["privacy_mask"]:
            start, end, value = span["start"], span["end"], span["value"]
            if not (0 <= start < end <= len(text) and text[start:end] == value):
                dropped_spans += 1
                continue
            new_start, new_end = _trim_span(text, start, end)
            if new_start >= new_end:            # span was whitespace only
                dropped_spans += 1
                continue
            if (new_start, new_end) != (start, end):
                trimmed_spans += 1
                span = {**span, "start": new_start, "end": new_end,
                        "value": text[new_start:new_end]}
            key = (span["start"], span["end"], span["label"])
            if key in seen:                     # trimming can collapse two spans into one
                continue
            seen.add(key)
            verified.append(span)
        kept_spans += len(verified)
        row["privacy_mask"] = verified
        rows.append(row)
    if trimmed_spans:
        print(f"    trimmed {trimmed_spans} whitespace-edged span(s)")
    return rows, kept_spans, dropped_spans


def _label_counts(rows: Iterable[Dict[str, Any]]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for row in rows:
        for span in row["privacy_mask"]:
            counter[span["label"]] += 1
    return counter


def _filter_labels(rows: List[Dict[str, Any]], allowed: set) -> None:
    for row in rows:
        row["privacy_mask"] = [s for s in row["privacy_mask"] if s["label"] in allowed]


def build(key: str, args: argparse.Namespace) -> Dict[str, Any]:
    source = SOURCES[key]
    caps = {"train": args.max_train, "validation": args.max_val, "test": args.max_test}
    label = f"{source['repo']}" + (f" [{source['config']}]" if source["config"] else "")
    print(f"\n{'=' * 70}\n{key}  <-  {label}\n{'=' * 70}")

    # A carved split takes the TAIL of its upstream split and the primary split is
    # shortened to match. Taking the head for both would silently return 0 rows
    # whenever the upstream split is smaller than the train cap (kiji: 46,345 < 50,000).
    reserved: Dict[str, int] = {}
    for target, upstream in source.get("carve", {}).items():
        reserved[upstream] = reserved.get(upstream, 0) + caps[target]

    splits: Dict[str, List[Dict[str, Any]]] = {}
    stats: Dict[str, Any] = {}
    sizes: Dict[str, int] = {}
    long_rows = long_spans = 0
    for target, upstream in source["splits"].items():
        raw = _load(source["repo"], source["config"], upstream, source.get("loader"))
        sizes[upstream] = len(raw)
        cap = min(caps[target], len(raw) - reserved.get(upstream, 0))
        rows, kept, dropped = _convert_split(raw, source["convert"], max(cap, 0))
        splits[target] = rows
        stats[target] = {"upstream_split": f"{upstream}[:{max(cap, 0)}]",
                         "rows": len(rows), "spans": kept, "spans_dropped": dropped}
        print(f"  {target:<11s} <- {upstream}[:{max(cap, 0)}]  {len(rows):>6d} rows  "
              f"{kept:>7d} spans  ({dropped} failed offset check)")

    for target, upstream in source.get("carve", {}).items():
        raw = _load(source["repo"], source["config"], upstream, source.get("loader"))
        offset = max(len(raw) - caps[target], 0)
        rows, kept, dropped = _convert_split(raw, source["convert"], caps[target], offset=offset)
        splits[target] = rows
        stats[target] = {"upstream_split": f"{upstream}[{offset}:]", "rows": len(rows),
                         "spans": kept, "spans_dropped": dropped}
        print(f"  {target:<11s} <- {upstream}[{offset}:]  {len(rows):>6d} rows  "
              f"{kept:>7d} spans  ({dropped} failed offset check)")

    # Length cap: train split only. Validation and test stay representative -- the cap
    # exists to bound training memory, not to make the benchmark easier.
    max_chars = source.get("max_train_chars", args.max_train_chars)
    if max_chars and "train" in splits:
        keep = []
        for row in splits["train"]:
            if len(row["source_text"]) > max_chars:
                long_rows += 1
                long_spans += len(row["privacy_mask"])
            else:
                keep.append(row)
        if long_rows:
            print(f"  dropped {long_rows} train row(s) over {max_chars} chars "
                  f"({long_spans} spans)")
        splits["train"] = keep
        stats["train"]["rows"] = len(keep)
        stats["train"]["dropped_long_rows"] = long_rows
        stats["train"]["dropped_long_spans"] = long_spans
        stats["max_train_chars"] = max_chars

    # A test-only release (MAPA) has no train split to measure label frequency on: keep
    # every label that occurs in test.
    ref = splits.get("train") or splits["test"]
    counts = _label_counts(ref)
    floor = args.min_label_count if "train" in splits else 1
    allowed = {name for name, n in counts.items() if n >= floor}
    dropped_labels = {name: n for name, n in counts.items() if name not in allowed}
    if dropped_labels:
        print(f"  dropped {len(dropped_labels)} rare label(s) "
              f"(<{args.min_label_count} in train): {dropped_labels}")
    for rows in splits.values():
        _filter_labels(rows, allowed)

    lengths = sorted(len(r["source_text"]) for r in ref)

    def pct(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))] if lengths else 0

    langs = collections.Counter(r["language"] for r in ref)
    print(f"  labels kept : {len(allowed)}")
    print(f"  text chars  : p50={pct(.5)} p90={pct(.9)} p99={pct(.99)} "
          f"max={lengths[-1] if lengths else 0}")
    print(f"  languages   : {dict(langs.most_common())}")

    stats["labels"] = {name: counts[name] for name in sorted(allowed, key=lambda l: -counts[l])}
    stats["dropped_labels"] = dropped_labels
    stats["text_chars"] = {"p50": pct(.5), "p90": pct(.9), "p99": pct(.99),
                           "max": lengths[-1] if lengths else 0}
    stats["languages"] = dict(langs.most_common())
    return {"splits": splits, "stats": stats}


def push(key: str, built: Dict[str, Any], hf_user: str, private: bool, token: str) -> str:
    from datasets import Dataset, DatasetDict

    ddict = DatasetDict({
        name: Dataset.from_list(rows)
        for name, rows in built["splits"].items()
        if rows
    })
    repo_id = f"{hf_user}/{SOURCES[key]['hub_name']}"
    print(f"  pushing -> {repo_id} ({'private' if private else 'public'})")
    ddict.push_to_hub(repo_id, private=private, token=token)
    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    parser.add_argument("--max-val", type=int, default=DEFAULT_MAX_VAL)
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--min-label-count", type=int, default=DEFAULT_MIN_LABEL_COUNT)
    parser.add_argument("--max-train-chars", type=int, default=DEFAULT_MAX_TRAIN_CHARS,
                        help="Drop train rows longer than this (0 disables). See the constant.")
    parser.add_argument("--hf-user", default="anonymous-org")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true", help="Without this, nothing is uploaded.")
    parser.add_argument("--stats-out", default="config/pii_dataset_stats.json")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        candidate = Path(__file__).resolve().parents[1] / ".hf_token"
        if candidate.is_file():
            token = candidate.read_text(encoding="utf-8").strip()
    if args.push and not token:
        raise SystemExit("--push needs HF_TOKEN (env) or a .hf_token file at the repo root")

    all_stats = {}
    for key in args.datasets:
        built = build(key, args)
        if args.push:
            built["stats"]["repo"] = push(key, built, args.hf_user, args.private, token)
        all_stats[key] = built["stats"]

    out = Path(args.stats_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    existing.update(all_stats)
    out.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nstats -> {out}")


if __name__ == "__main__":
    main()
