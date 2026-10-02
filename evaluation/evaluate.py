"""CLI evaluation script for GLiNER2 PII models.

Usage:
    python evaluation/evaluate.py --model anonymous-org/gliner2-multi-v1-3e-20260514

    # With mmBERT backbone
    python evaluation/evaluate.py --model anonymous-org/gliner2-mmbert-base-rosenxt-6e-20260515 --mmbert --chunk-tokens 2048

    # Push results to HF (defaults to anonymous-org/eval-<model-name>)
    python evaluation/evaluate.py --model anonymous-org/gliner2-multi-v1-3e-20260514 --push

    # Quantized backends (needs CUDA GPU + onnxruntime-gpu). Baseline is
    # --backend pytorch_fp32 (default). ONNX backends export the encoder once.
    python evaluation/evaluate.py --model anonymous-org/gliner2-multi-v1-3e-20260514 \
        --backend onnx_cuda_fp16 --export-onnx
    # To compare all backends side-by-side, use scripts/run_quant_bench.py.

Output structure:
    <output-dir>/
    ├── run_info.json          (config, timestamps, device)
    ├── timing.json            (mean/median/p95/p99/samples_per_sec)
    ├── metrics/
    │   ├── overall.csv        (P/R/F1/TP/FP/FN per label, all languages)
    │   ├── <lang>.csv         (per-language breakdown)
    │   └── summary.json       (micro + macro + macro-weighted aggregates per lang + overall)
    └── predictions/
        ├── all.jsonl          (one JSON per line: text, gt, predictions, time_ms)
        └── <lang>.jsonl       (same, filtered by language)
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datasets import load_dataset

from config.ner_datasets import load_desc_override as _load_desc_file
from tqdm import tqdm

from config.hyperparams import (
    BENCH_COLLECTION_SLUG,
    BENCHMARK_THRESHOLD,
    CHUNK_SAFETY_MARGIN,
    EVAL_DATASET_ID,
    EVAL_SPLIT,
    EXTRA_LANGS_FROM_FT,
    FINETUNE_VAL_DATASET,
    FINETUNE_VAL_MAX,
    FINETUNE_VAL_SPLIT,
    FUZZY_MAX_DIST,
    FUZZY_MAX_DIST_STRICT,
    HF_ORG,
    HF_TOKEN,
    IOU_THRESH,
    IOU_THRESH_STRICT,
    LANGUAGES,
    MAX_CHARS_PER_CHUNK,
    MAX_TOKENS_PER_CHUNK,
    MAX_WIDTH,
    PARALLEL_BATCH_SIZE,
    PER_LABEL_THRESHOLDS,
    THRESHOLD,
    TIMING_PROBE_SAMPLES,
    USE_EXTRA_FT_VAL,
)
from config.labels import PII_LABEL_DESCRIPTIONS
from components.wordChunker import build_word_chunker
from evaluation.inference import (
    apply_advanced_postprocessing,
    clean_text_with_mapping,
    normalize_predictions,
    resolve_span_conflicts,
)
from evaluation.metrics import HungarianEvaluator, compute_metrics_df
from evaluation.rule_detector import detect_by_rules
from models.loader import load_gliner2_model
from models.onnx_backend import ALL_BACKENDS, ONNX_BACKENDS, build_backend_model
from models.tokenizer_patch import apply_tokenizer_patch

# PROTOCOL.md 9.18: eval-time label-name intervention (--label-names). When set, the model is
# queried with renamed types and every prediction is mapped back to the original type before
# conflict resolution, post-processing and scoring, so only the string shown to the model changes.
_LABEL_RENAME: Dict[str, str] = {}      # ORIGINAL (upper) -> name shown to the model
_LABEL_INVERSE: Dict[str, str] = {}     # NAME SHOWN (upper) -> original type (upper)


def _unrename(preds: List[Dict]) -> List[Dict]:
    if _LABEL_INVERSE:
        for p in preds:
            p["label"] = _LABEL_INVERSE.get(str(p["label"]).upper().strip(), p["label"])
    return preds


def build_gt(row: Dict, label_descriptions=None) -> Dict:
    """Convert a dataset row to ground-truth format.

    label_descriptions: when provided (NER mode), GT labels are checked against
    this dict's keys (case-insensitive / uppercased) instead of PII_LABEL_DESCRIPTIONS.
    """
    import ast, json as _json
    text = str(row.get("source_text", "") or "")
    masks = row.get("privacy_mask", [])
    if isinstance(masks, str):
        try:
            masks = _json.loads(masks)
        except Exception:
            try:
                masks = ast.literal_eval(masks)
            except Exception:
                masks = []

    # For NER mode, build an uppercase lookup set so predictions (always uppercase)
    # match GT tags regardless of how the dataset stores label case.
    allowed_upper = {k.upper() for k in label_descriptions} if label_descriptions is not None else None

    ents = []
    for m in masks:
        try:
            s, e = int(m["start"]), int(m["end"])
        except Exception:
            continue
        if s < 0 or e <= s or e > len(text):
            continue
        key = str(m.get("label") or "").upper().strip()
        if allowed_upper is not None:
            if key not in allowed_upper:
                continue
        else:
            if key not in PII_LABEL_DESCRIPTIONS:
                continue
        ents.append({"start": s, "end": e, "tag": key,
                     "value": m.get("value") or text[s:e], "text": text[s:e]})
    result = {"text": text, "language": str(row.get("language", "")), "gt": ents}
    if row.get("sample_id"):
        result["sample_id"] = str(row["sample_id"])
    return result


def _build_schema(model, use_desc: bool = False, label_descriptions=None,
                  schema_case: str = "upper"):
    """Build GLiNER2 extraction schema.

    When label_descriptions is provided (NER mode), use those labels with no
    per-label threshold overrides.  Otherwise use all 44 PII labels with their
    configured PER_LABEL_THRESHOLDS.
    use_desc=False (default) matches notebook benchmark config — NOTE this is a
    TRAIN/EVAL MISMATCH for --ner-dataset checkpoints: training always carries
    entity_descriptions (to_input_example_generic default), so keys-only eval
    strips text the model saw during training.
    schema_case: "upper" (legacy default) uppercases label keys; "original"
    keeps the registry's casing — the exact strings used at TRAINING time
    (matters for mixed-case sets like mit_restaurant's "Amenity"). Downstream
    matching is case-safe either way: normalize_predictions and build_gt both
    uppercase before scoring.
    """
    schema = model.create_schema()

    if label_descriptions is not None:
        if schema_case == "original":
            ld = dict(label_descriptions)          # match training exactly
        else:
            ld = {k.upper(): v for k, v in label_descriptions.items()}
        if _LABEL_RENAME:
            ld = {_LABEL_RENAME.get(k.upper(), k): v for k, v in ld.items()}
        if use_desc:
            schema.entities(ld)
        else:
            schema.entities(list(ld.keys()))
        return schema

    # PII mode: split labels into those with and without per-label threshold overrides.
    simple = {k: v for k, v in PII_LABEL_DESCRIPTIONS.items() if k not in PER_LABEL_THRESHOLDS}
    with_thresh = {k: v for k, v in PII_LABEL_DESCRIPTIONS.items() if k in PER_LABEL_THRESHOLDS}

    # Add simple labels (no threshold override)
    if use_desc:
        schema.entities(simple)
    else:
        schema.entities(list(simple.keys()))

    # Add labels with per-entity threshold (always include description so GLiNER2
    # can build the full config dict with the threshold field)
    if with_thresh:
        schema.entities({
            label: {
                "description": desc,
                "threshold": PER_LABEL_THRESHOLDS[label],
            }
            for label, desc in with_thresh.items()
        })

    return schema


def _merge_rule_predictions(text: str, all_preds: List[Dict]) -> List[Dict]:
    """Gap-fill model predictions with the regex rule detector (PII-only).

    Rulebase is a SAFETY-NET gap-filler: the model layer owns recall, so a
    rule must never override a model span (overriding with a wrong/ambiguous
    label would turn a model TP into an FN). Resolve the model spans exactly
    as in the no-rules path, then add only the rule spans that land in gaps
    the model left empty. This makes it structurally impossible for the
    rulebase to create an FN — it can only recover misses the model made.
    """
    cleaned, char_map = clean_text_with_mapping(text)
    orig_to_clean: Dict[int, int] = {}
    for ci, op in enumerate(char_map):
        orig_to_clean.setdefault(op, ci)
    model_preds_clean = []
    for p in all_preds:
        cs = orig_to_clean.get(p["start"])
        ce = orig_to_clean.get(p["end"])
        if ce is None:
            # p["end"] may land on a stripped char (e.g. trailing space).
            # Scan forward to the next mapped original position.
            for k in range(p["end"] + 1, len(text) + 2):
                ce = orig_to_clean.get(k)
                if ce is not None:
                    break
        if cs is None or ce is None or ce <= cs:
            continue
        model_preds_clean.append({
            "label": p["label"], "text": cleaned[cs:ce],
            "start": cs, "end": ce, "score": p.get("score", 0.0),
        })
    rule_preds = [
        {**e, "score": 1.0}
        for e in detect_by_rules(cleaned, list(PII_LABEL_DESCRIPTIONS.keys()))
    ]
    model_resolved = resolve_span_conflicts(model_preds_clean)
    occupied = [(m["start"], m["end"]) for m in model_resolved]
    rule_gap = [
        r for r in rule_preds
        if not any(r["start"] < e and r["end"] > s for s, e in occupied)
    ]
    combined = model_resolved + resolve_span_conflicts(rule_gap, strategy="rule_first")
    result = []
    for r in combined:
        os_ = char_map[r["start"]] if r["start"] < len(char_map) else None
        oe_ = char_map[r["end"] - 1] + 1 if (r["end"] - 1) < len(char_map) else None
        if os_ is None or oe_ is None or oe_ <= os_:
            continue
        result.append({
            "label": r["label"], "start": os_, "end": oe_,
            "text": text[os_:oe_], "score": float(r.get("score", 1.0)),
        })
    return result


def _chunk_text(text: str, chunk_tokens: int, chunker=None) -> List[Tuple[str, int]]:
    if chunker is not None:
        raw_chunks = chunker.chunk(text)
        return [(c.text, c.start_index) for c in raw_chunks]
    # No chunker (--no-chunk): treat the whole text as a single chunk.
    return [(text, 0)]


def run_inference(
    model, text: str, chunk_tokens: int, threshold: float, use_rules: bool,
    chunker=None, use_desc: bool = False, emit_duplicates: bool = False,
    label_descriptions=None, schema_case: str = "upper",
) -> List[Dict]:
    """Chunk text, run model.extract() with schema, merge predictions.

    emit_duplicates=False → model_spans mode (notebook's primary benchmark mode).
    emit_duplicates=True  → value_broadcast mode (find all occurrences of values).
    label_descriptions: when set (NER mode), use these labels instead of PII labels;
      rule-based detection is skipped automatically.
    """
    schema = _build_schema(model, use_desc=use_desc, label_descriptions=label_descriptions,
                           schema_case=schema_case)
    chunks = _chunk_text(text, chunk_tokens, chunker)
    inc_spans = not emit_duplicates
    inc_conf  = not emit_duplicates
    all_preds = []
    for chunk_text, chunk_start in chunks:
        # model_spans mode: format_results=False bypasses GLiNER2's value-level
        # dedup (which collapses repeated values like "Mr."/"William" to a single
        # span), so every position the model predicted survives. We unwrap the
        # raw structure in normalize_predictions. value_broadcast keeps the
        # formatted output (deduped values) and re-broadcasts via string search.
        raw = model.extract(
            chunk_text,
            schema,
            threshold=threshold,
            include_spans=inc_spans,
            include_confidence=inc_conf,
            format_results=emit_duplicates,
        )
        preds = _unrename(normalize_predictions(raw, chunk_text, emit_duplicates=emit_duplicates))
        for p in preds:
            p["start"] += chunk_start
            p["end"] += chunk_start
        all_preds.extend(preds)

    # Rule-based detection is PII-specific; skip in NER mode.
    if use_rules and label_descriptions is None:
        all_preds = _merge_rule_predictions(text, all_preds)
    else:
        all_preds = resolve_span_conflicts(all_preds)

    return apply_advanced_postprocessing(all_preds, text)


def run_inference_batch(
    model, texts: List[str], chunk_tokens: int, threshold: float, use_rules: bool,
    chunker=None, use_desc: bool = False, emit_duplicates: bool = False,
    label_descriptions=None, schema_case: str = "upper",
    batch_size: int = PARALLEL_BATCH_SIZE,
) -> List[List[Dict]]:
    """Batched equivalent of run_inference for throughput on large eval sets.

    Chunks every text (same chunker/logic as run_inference), sends ALL chunks
    across ALL texts through a single model.batch_extract() call (one padded
    forward pass instead of one Python-level call per text), then re-applies
    the exact same per-text rule-merge / conflict-resolution / postprocessing.
    Only the model forward pass is batched — predictions should match
    run_inference's output for the same schema/threshold/chunking.
    """
    if not texts:
        return []
    schema = _build_schema(model, use_desc=use_desc, label_descriptions=label_descriptions,
                           schema_case=schema_case)

    chunk_texts: List[str] = []
    chunk_owner: List[int] = []
    chunk_starts: List[int] = []
    for ti, text in enumerate(texts):
        for chunk_text, chunk_start in _chunk_text(text, chunk_tokens, chunker):
            chunk_texts.append(chunk_text)
            chunk_owner.append(ti)
            chunk_starts.append(chunk_start)

    inc_spans = not emit_duplicates
    inc_conf  = not emit_duplicates
    raw_results = model.batch_extract(
        chunk_texts, schema, batch_size=batch_size, threshold=threshold,
        include_spans=inc_spans, include_confidence=inc_conf,
        format_results=emit_duplicates,
    ) if chunk_texts else []

    per_text_preds: List[List[Dict]] = [[] for _ in texts]
    for raw, chunk_text, chunk_start, ti in zip(raw_results, chunk_texts, chunk_starts, chunk_owner):
        preds = _unrename(normalize_predictions(raw, chunk_text, emit_duplicates=emit_duplicates))
        for p in preds:
            p["start"] += chunk_start
            p["end"] += chunk_start
        per_text_preds[ti].extend(preds)

    results: List[List[Dict]] = []
    for ti, text in enumerate(texts):
        all_preds = per_text_preds[ti]
        if use_rules and label_descriptions is None:
            all_preds = _merge_rule_predictions(text, all_preds)
        else:
            all_preds = resolve_span_conflicts(all_preds)
        results.append(apply_advanced_postprocessing(all_preds, text))
    return results


def run_inference_debug(
    model,
    text: str,
    chunk_tokens: int,
    threshold: float,
    use_rules: bool,
    chunker=None,
    use_desc: bool = False,
    emit_duplicates: bool = False,
) -> Dict:
    """Single-text debug run: return chunks, per-chunk raw outputs, and final predictions."""
    schema = _build_schema(model, use_desc=use_desc)
    if chunker is not None:
        raw_chunks = chunker.chunk(text)
        chunks = [(c.text, c.start_index) for c in raw_chunks]
    else:
        chunks = [(text, 0)]

    inc_spans = not emit_duplicates
    inc_conf = not emit_duplicates
    all_preds = []
    chunk_logs = []

    for idx, (chunk_text, chunk_start) in enumerate(chunks, start=1):
        raw = model.extract(
            chunk_text,
            schema,
            threshold=threshold,
            include_spans=inc_spans,
            include_confidence=inc_conf,
            format_results=emit_duplicates,
        )
        preds = _unrename(normalize_predictions(raw, chunk_text, emit_duplicates=emit_duplicates))
        for p in preds:
            p["start"] += chunk_start
            p["end"] += chunk_start
        all_preds.extend(preds)
        chunk_logs.append({
            "chunk_index": idx,
            "start": chunk_start,
            "end": chunk_start + len(chunk_text),
            "text": chunk_text,
            "raw_output": raw,
            "normalized_predictions": preds,
        })

    if use_rules:
        cleaned, char_map = clean_text_with_mapping(text)
        orig_to_clean: Dict[int, int] = {}
        for ci, op in enumerate(char_map):
            orig_to_clean.setdefault(op, ci)
        model_preds_clean = []
        for p in all_preds:
            cs = orig_to_clean.get(p["start"])
            ce = orig_to_clean.get(p["end"])
            if ce is None:
                # p["end"] may land on a stripped char (e.g. trailing space).
                # Scan forward to the next mapped original position.
                for k in range(p["end"] + 1, len(text) + 2):
                    ce = orig_to_clean.get(k)
                    if ce is not None:
                        break
            if cs is None or ce is None or ce <= cs:
                continue
            model_preds_clean.append({
                "label": p["label"], "text": cleaned[cs:ce],
                "start": cs, "end": ce, "score": p.get("score", 0.0),
            })
        rule_preds = [
            {**e, "score": 1.0}
            for e in detect_by_rules(cleaned, list(PII_LABEL_DESCRIPTIONS.keys()))
        ]
        combined = resolve_span_conflicts(rule_preds + model_preds_clean, strategy="rule_first")
        result = []
        for r in combined:
            os_ = char_map[r["start"]] if r["start"] < len(char_map) else None
            oe_ = char_map[r["end"] - 1] + 1 if (r["end"] - 1) < len(char_map) else None
            if os_ is None or oe_ is None or oe_ <= os_:
                continue
            result.append({
                "label": r["label"], "start": os_, "end": oe_,
                "text": text[os_:oe_], "score": float(r.get("score", 1.0)),
            })
        all_preds = result
    else:
        all_preds = resolve_span_conflicts(all_preds)

    final_preds = apply_advanced_postprocessing(all_preds, text)
    return {
        "chunks": chunk_logs,
        "final_predictions": final_preds,
    }


def _compute_timing_stats(timings_ms: List[float]) -> Dict:
    """Descriptive timing stats from per-sample inference times (in ms)."""
    if not timings_ms:
        return {
            "total_s": 0.0, "n_samples": 0, "samples_per_sec": 0.0,
            "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0,
            "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0,
        }
    n = len(timings_ms)
    sorted_t = sorted(timings_ms)
    total_s = sum(timings_ms) / 1000.0
    return {
        "total_s":        round(total_s, 3),
        "n_samples":      n,
        "samples_per_sec": round(n / total_s, 2) if total_s > 0 else 0.0,
        "mean_ms":        round(statistics.mean(timings_ms), 2),
        "median_ms":      round(statistics.median(timings_ms), 2),
        "p95_ms":         round(sorted_t[min(int(0.95 * n), n - 1)], 2),
        "p99_ms":         round(sorted_t[min(int(0.99 * n), n - 1)], 2),
        "min_ms":         round(sorted_t[0], 2),
        "max_ms":         round(sorted_t[-1], 2),
    }


def _compute_aggregate(df: pd.DataFrame) -> Dict:
    """Micro + macro + macro-weighted P/R/F1 from a per-label metrics DataFrame."""
    if df.empty:
        return {
            "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
            "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
            "macro_weighted_precision": 0.0, "macro_weighted_recall": 0.0, "macro_weighted_f1": 0.0,
            "weighted_precision": 0.0, "weighted_recall": 0.0, "weighted_f1": 0.0,
            "support": 0, "n_labels": 0,
        }
    tp = int(df["tp"].sum())
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    sup = int(df["support"].sum())
    w = df["support"] / sup if sup else pd.Series(0.0, index=df.index)
    macro_weighted_p = float((df["precision"] * w).sum())
    macro_weighted_r = float((df["recall"] * w).sum())
    macro_weighted_f1 = float((df["f1"] * w).sum())
    return {
        "micro_precision":    round(float(micro_p), 4),
        "micro_recall":       round(float(micro_r), 4),
        "micro_f1":           round(float(micro_f1), 4),
        "macro_precision":    round(float(df["precision"].mean()), 4),
        "macro_recall":       round(float(df["recall"].mean()), 4),
        "macro_f1":           round(float(df["f1"].mean()), 4),
        "macro_weighted_precision": round(macro_weighted_p, 4),
        "macro_weighted_recall":    round(macro_weighted_r, 4),
        "macro_weighted_f1":        round(macro_weighted_f1, 4),
        "weighted_precision": round(macro_weighted_p, 4),
        "weighted_recall":    round(macro_weighted_r, 4),
        "weighted_f1":        round(macro_weighted_f1, 4),
        "support":            sup,
        "n_labels":           int(len(df)),
    }


class _ScoreAggregator:
    """Accumulates TP/FP/FN/support (overall + per-language) for one HungarianEvaluator config."""

    def __init__(self, evaluator: HungarianEvaluator):
        self.evaluator = evaluator
        self.total_tp  = defaultdict(int)
        self.total_fp  = defaultdict(int)
        self.total_fn  = defaultdict(int)
        self.total_sup = defaultdict(int)
        self.per_lang: Dict[str, tuple] = defaultdict(
            lambda: (defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int))
        )

    def add(self, gt_ents: List[Dict], pred_ents: List[Dict], language: str) -> Tuple[int, int, int]:
        """Score one sample against this config; return (tp_n, fp_n, fn_n)."""
        tp, fp, fn, sup = self.evaluator.count_example(gt_ents, pred_ents)
        for d_total, d_item in [(self.total_tp, tp), (self.total_fp, fp),
                                 (self.total_fn, fn), (self.total_sup, sup)]:
            for k, v in d_item.items():
                d_total[k] += v
        l_tp, l_fp, l_fn, l_sup = self.per_lang[language]
        for d_total, d_item in [(l_tp, tp), (l_fp, fp), (l_fn, fn), (l_sup, sup)]:
            for k, v in d_item.items():
                d_total[k] += v
        return int(sum(tp.values())), int(sum(fp.values())), int(sum(fn.values()))

    def finalize(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        overall_df = compute_metrics_df(self.total_tp, self.total_fp, self.total_fn, self.total_sup)
        lang_dfs = {lang: compute_metrics_df(*vals) for lang, vals in self.per_lang.items()}
        return overall_df, lang_dfs


def _build_score_aggregators() -> Dict[str, "_ScoreAggregator"]:
    return {
        "lenient": _ScoreAggregator(HungarianEvaluator(iou_threshold=IOU_THRESH, fuzzy_max_dist=FUZZY_MAX_DIST)),
        "strict":  _ScoreAggregator(HungarianEvaluator(iou_threshold=IOU_THRESH_STRICT, fuzzy_max_dist=FUZZY_MAX_DIST_STRICT)),
    }


def evaluate_model(
    model,
    test_data: List[Dict],
    chunk_tokens: int,
    use_rules: bool,
    threshold: float = BENCHMARK_THRESHOLD,
    chunker=None,
    use_desc: bool = False,
    emit_duplicates: bool = False,
    log_each_sample: bool = False,
    log_text_chars: int = 120,
    label_descriptions=None,
    schema_case: str = "upper",
    timing_probe_samples: int = TIMING_PROBE_SAMPLES,
    infer_batch_size: int = PARALLEL_BATCH_SIZE,
) -> Tuple:
    """Evaluate model; return (overall_dfs, lang_dfs, sample_records, timing_stats).

    Every sample is scored under two HungarianEvaluator configs at once (same
    predictions, no extra inference cost):
      - "lenient": IoU >= IOU_THRESH (0.5) OR fuzzy match (start within FUZZY_MAX_DIST chars) — the
        historical default used across existing results.
      - "strict":  IoU >= IOU_THRESH_STRICT (1.0), fuzzy disabled — exact span-boundary match.
    overall_dfs / lang_dfs are ``{"lenient": ..., "strict": ...}`` dicts.

    Throughput: only the first ``timing_probe_samples`` samples are run
    sequentially (one at a time, wall-clock timed) to measure realistic
    single-request samples/s. The remaining samples are scored via
    run_inference_batch() (one batched model.batch_extract() call) purely for
    speed; their time is excluded from samples_per_sec. Pass
    infer_batch_size=0 to disable batching and time every sample.

    sample_records: list of per-sample dicts with text, gt, predictions, timing.
    timing_stats: descriptive stats over the timing-probe inference times.
    """
    aggregators = _build_score_aggregators()
    sample_records: List[Dict] = []
    timings_ms: List[float] = []

    def _record(i: int, item: Dict, preds: List[Dict], elapsed_ms, error_msg) -> None:
        sample_id = item.get("sample_id", str(i))
        language = item.get("language", "unknown")
        gt_n = len(item["gt"])
        pred_n = len(preds)

        counts = {mode: agg.add(item["gt"], preds, language) for mode, agg in aggregators.items()}
        tp_n, fp_n, fn_n = counts["lenient"]
        status = "ERROR" if error_msg else ("MISMATCH" if (fp_n or fn_n) else "OK")

        if log_each_sample:
            text_preview = " ".join(str(item["text"]).split())[:max(1, log_text_chars)]
            time_str = "n/a (batched)" if elapsed_ms is None else f"{elapsed_ms:.2f}"
            tqdm.write(
                f"[sample {i + 1}/{len(test_data)}] "
                f"id={sample_id} lang={language} status={status} "
                f"time_ms={time_str} gt={gt_n} pred={pred_n} "
                f"tp={tp_n} fp={fp_n} fn={fn_n} (lenient) text='{text_preview}'"
            )
            if error_msg:
                tqdm.write(f"  error={error_msg}")

        sample_records.append({
            "sample_id":          sample_id,
            "language":           language,
            "text":               item["text"],
            "gt":                 item["gt"],
            "predictions":        preds,
            "inference_time_ms":  round(elapsed_ms, 2) if elapsed_ms is not None else None,
            "status":             status,
            "tp_lenient": counts["lenient"][0], "fp_lenient": counts["lenient"][1], "fn_lenient": counts["lenient"][2],
            "tp_strict":  counts["strict"][0],  "fp_strict":  counts["strict"][1],  "fn_strict":  counts["strict"][2],
            "error":              error_msg,
        })

    probe_n = min(timing_probe_samples, len(test_data)) if infer_batch_size else len(test_data)
    probe_data = test_data[:probe_n]
    batch_data = test_data[probe_n:]

    for i, item in enumerate(tqdm(probe_data, desc="eval (sequential, timing probe)")):
        t0 = time.perf_counter()
        error_msg = None
        try:
            preds = run_inference(model, item["text"], chunk_tokens, threshold, use_rules,
                                  chunker=chunker, use_desc=use_desc, schema_case=schema_case,
                                  emit_duplicates=emit_duplicates,
                                  label_descriptions=label_descriptions)
        except Exception as ex:
            preds = []
            error_msg = f"{type(ex).__name__}: {ex}"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings_ms.append(elapsed_ms)
        _record(i, item, preds, elapsed_ms, error_msg)

    if batch_data:
        try:
            batch_preds = run_inference_batch(
                model, [r["text"] for r in batch_data], chunk_tokens, threshold, use_rules,
                chunker=chunker, use_desc=use_desc, schema_case=schema_case,
                emit_duplicates=emit_duplicates,
                label_descriptions=label_descriptions, batch_size=infer_batch_size,
            )
            batch_errors = [None] * len(batch_data)
        except Exception as ex:
            # Mirrors the per-sample try/except in the sequential path: one bad
            # batch shouldn't crash the whole eval run, just mark it ERROR.
            batch_preds = [[] for _ in batch_data]
            batch_errors = [f"{type(ex).__name__}: {ex}"] * len(batch_data)

        for j, (item, preds, error_msg) in enumerate(
            tqdm(list(zip(batch_data, batch_preds, batch_errors)), desc="eval (batched)")
        ):
            _record(probe_n + j, item, preds, None, error_msg)

    overall_dfs: Dict[str, pd.DataFrame] = {}
    lang_dfs_by_mode: Dict[str, Dict[str, pd.DataFrame]] = {}
    for mode, agg in aggregators.items():
        overall_dfs[mode], lang_dfs_by_mode[mode] = agg.finalize()

    timing_stats = _compute_timing_stats(timings_ms)
    timing_stats["n_samples_total"] = len(test_data)
    timing_stats["probe_samples"] = probe_n
    timing_stats["batch_samples"] = len(batch_data)

    return overall_dfs, lang_dfs_by_mode, sample_records, timing_stats


def save_results(
    output_dir: str,
    overall_dfs: Dict[str, pd.DataFrame],
    lang_dfs_by_mode: Dict[str, Dict[str, pd.DataFrame]],
    sample_records: List[Dict],
    timing_stats: Dict,
    run_info: Dict,
) -> None:
    """Write all evaluation artifacts to output_dir.

    overall_dfs / lang_dfs_by_mode: {"lenient": ..., "strict": ...} (see evaluate_model).

    Structure:
        <output_dir>/
        ├── run_info.json
        ├── timing.json
        ├── metrics/
        │   ├── overall_<mode>.csv        (mode = lenient | strict)
        │   ├── <lang>_<mode>.csv
        │   └── summary.json              ({"lenient": {...}, "strict": {...}})
        └── predictions/
            ├── all.jsonl                 (includes tp/fp/fn for both modes per sample)
            └── <lang>.jsonl
    """
    metrics_dir = os.path.join(output_dir, "metrics")
    preds_dir   = os.path.join(output_dir, "predictions")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    # ── run metadata ──────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    # ── timing ────────────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "timing.json"), "w") as f:
        json.dump(timing_stats, f, indent=2)

    # ── per-label CSVs + summary, one block per scoring mode ─────────────────
    summary: Dict[str, Dict] = {}
    for mode, overall_df in overall_dfs.items():
        lang_dfs = lang_dfs_by_mode[mode]
        overall_df.to_csv(os.path.join(metrics_dir, f"overall_{mode}.csv"))
        for lang, df in lang_dfs.items():
            df.to_csv(os.path.join(metrics_dir, f"{lang}_{mode}.csv"))
        summary[mode] = {
            "overall":     _compute_aggregate(overall_df),
            "by_language": {lang: _compute_aggregate(df) for lang, df in lang_dfs.items()},
        }
    with open(os.path.join(metrics_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── per-sample predictions ────────────────────────────────────────────────
    by_lang: Dict[str, List] = defaultdict(list)
    with open(os.path.join(preds_dir, "all.jsonl"), "w", encoding="utf-8") as f:
        for rec in sample_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            by_lang[rec["language"]].append(rec)

    for lang, recs in by_lang.items():
        with open(os.path.join(preds_dir, f"{lang}.jsonl"), "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _push_results(
    output_dir: str,
    repo_id: str,
    model_name: str,
    *,
    repo_type: str = "dataset",
    path_in_repo: Optional[str] = None,
) -> None:
    """Push eval results either to their own HF **dataset** repo (default) or as a
    subfolder inside the **model** repo itself (``repo_type="model"``,
    ``path_in_repo`` e.g. ``eval/test``) — see ``--push-eval-into-model``."""
    from huggingface_hub import add_collection_item, create_repo, upload_folder
    create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=f"eval results for {model_name}"
        + (f" ({path_in_repo})" if path_in_repo else ""),
    )
    if repo_type == "dataset":
        print(f"Results → https://huggingface.co/datasets/{repo_id}")
        if BENCH_COLLECTION_SLUG:
            add_collection_item(BENCH_COLLECTION_SLUG, repo_id, item_type="dataset", exists_ok=True)
            print(f"Added to collection SecurePII-Bench")
    else:
        print(f"Results → https://huggingface.co/{repo_id}/tree/main/{path_in_repo or ''}")


def _print_summary(
    overall_dfs: Dict[str, pd.DataFrame],
    lang_dfs_by_mode: Dict[str, Dict[str, pd.DataFrame]],
    timing_stats: Dict,
) -> None:
    n_total = timing_stats.get("n_samples_total", timing_stats["n_samples"])
    print(f"\n=== {n_total} samples total "
          f"({timing_stats['probe_samples']} timed sequentially, "
          f"{timing_stats['batch_samples']} scored via batch inference) ===")

    for mode, overall_df in overall_dfs.items():
        lang_dfs = lang_dfs_by_mode[mode]
        agg = _compute_aggregate(overall_df)
        label = "lenient (IoU>=0.5 or fuzzy)" if mode == "lenient" else "strict (exact match, IoU>=1.0)"
        print(f"\n--- Scoring: {mode} — {label} ---")
        print(f"  micro   P={agg['micro_precision']:.4f}  R={agg['micro_recall']:.4f}  F1={agg['micro_f1']:.4f}")
        print(f"  macro-weighted P={agg['macro_weighted_precision']:.4f}  R={agg['macro_weighted_recall']:.4f}  F1={agg['macro_weighted_f1']:.4f}")
        if not overall_df.empty:
            print(overall_df[["precision", "recall", "f1", "support"]].to_string())

        for lang, df in sorted(lang_dfs.items()):
            agg_l = _compute_aggregate(df)
            print(f"  [{lang}] micro F1={agg_l['micro_f1']:.4f}  macro-weighted F1={agg_l['macro_weighted_f1']:.4f}  support={agg_l['support']}")

    print(f"\n=== Timing (sequential probe, {timing_stats['probe_samples']} samples) ===")
    print(f"  {timing_stats['samples_per_sec']:.1f} samples/s over {timing_stats['total_s']:.1f}s  "
          f"mean={timing_stats['mean_ms']:.1f}ms  "
          f"median={timing_stats['median_ms']:.1f}ms  "
          f"p95={timing_stats['p95_ms']:.1f}ms  "
          f"p99={timing_stats['p99_ms']:.1f}ms")


def load_eval_data(args, label_descriptions=None) -> List[Dict]:
    """Load + build ground-truth eval samples from the configured dataset(s).

    Extracted from main() so the quant-bench orchestrator can reuse the exact
    same data (and merge the rosenxt en/de val set when USE_EXTRA_FT_VAL).

    label_descriptions: when provided (NER mode), GT labels are filtered by these
    keys and the language filter is skipped (NER datasets may lack a language column).
    """
    ds = load_dataset(args.dataset, split=args.split)
    # Language filter: skip in NER mode (datasets may have no language column)
    if args.languages and label_descriptions is None:
        ds = ds.filter(lambda x: x.get("language") in args.languages)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    test_data = [build_gt(row, label_descriptions=label_descriptions)
                 for row in tqdm(ds, desc="build gt")]
    test_data = [r for r in test_data if r["text"].strip()]

    # Merge rosenxt en/de val set (matches notebook USE_EXTRA_FT_VAL=True)
    # Skipped in NER mode (no_extra_val implied by label_descriptions being set)
    if USE_EXTRA_FT_VAL and not getattr(args, "no_extra_val", False) and label_descriptions is None:
        ft_ds = load_dataset(FINETUNE_VAL_DATASET, split=FINETUNE_VAL_SPLIT)
        ft_ds = ft_ds.filter(lambda x: x.get("language") in args.languages)
        ft_ds = ft_ds.select(range(min(FINETUNE_VAL_MAX, len(ft_ds))))
        extra_langs = set(EXTRA_LANGS_FROM_FT)
        ft_ds = ft_ds.filter(lambda x: x.get("language") in extra_langs)
        extra = [build_gt(r) for r in tqdm(ft_ds, desc="build gt extra")]
        extra = [r for r in extra if r["text"].strip()]
        test_data.extend(extra)
        lang_counts = ", ".join(
            f"{l}:{sum(1 for r in test_data if r['language'] == l)}"
            for l in args.languages
        )
        print(f"Merged eval set: {len(test_data)} samples ({lang_counts})")
    return test_data


def resolve_onnx_path(args) -> str:
    """Conventional ONNX path for a model: <onnx_dir>/<model_slug>_encoder.onnx."""
    if getattr(args, "onnx_path", None):
        return args.onnx_path
    slug = args.model.rstrip("/").split("/")[-1]
    return os.path.join(args.onnx_dir, f"{slug}_encoder.onnx")


def apply_backend(model, args):
    """Turn a loaded (FP32, on-device) model into the requested backend variant.

    For ONNX backends, exports the encoder ONNX first if missing (or --export-onnx).
    Returns the (possibly mutated) model. No-op for pytorch_fp32.
    """
    if args.backend == "pytorch_fp32":
        return model

    onnx_path = None
    if args.backend in ONNX_BACKENDS:
        onnx_path = resolve_onnx_path(args)
        if getattr(args, "export_onnx", False) or not os.path.exists(onnx_path):
            from models.onnx_export import export_encoder_onnx
            print(f"[backend] exporting encoder ONNX -> {onnx_path}")
            export_encoder_onnx(model, onnx_path, fp16=True)
        else:
            print(f"[backend] using cached ONNX -> {onnx_path}")

    return build_backend_model(model, args.backend, onnx_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GLiNER2 PII model")
    parser.add_argument("--model",        required=True, help="HF repo or local path")
    parser.add_argument("--mmbert",       action="store_true", help="Use CustomExtractor for mmBERT")
    parser.add_argument("--backbone",     default=None,
                        help="Uni-encoder backbone key (config/backbones.py). Only sets the "
                             "default --chunk-tokens to match how the variant was trained "
                             "(DeBERTa=1024, ModernBERT/mmBERT=2048); the checkpoint itself "
                             "is still loaded from --model.")
    parser.add_argument("--subfolder",    default=None,
                        help="Optional subfolder inside HF repo (e.g. 'best')")
    parser.add_argument("--max-width",    type=int, default=MAX_WIDTH,
                        help="Span max width (default: 30; pass None-like to keep config default)")
    parser.add_argument("--keep-config-max-width", action="store_true",
                        help="Don't override max_width, use model config default (for quy models)")
    parser.add_argument("--no-chunk",     action="store_true",
                        help="Disable chunking entirely (single extract call per sample)")
    parser.add_argument("--dataset",      default=EVAL_DATASET_ID)
    parser.add_argument("--split",        default=EVAL_SPLIT, help="Dataset split to evaluate on (e.g. train, test)")
    parser.add_argument("--test-text",    type=str, default=None,
                        help="Run quick single-sentence test mode with this raw text (skip dataset loading)")
    parser.add_argument("--test-language", type=str, default="manual",
                        help="Language tag used with --test-text (default: manual)")
    parser.add_argument("--languages",    nargs="+", default=LANGUAGES)
    parser.add_argument("--max-samples",  type=int, default=None)
    parser.add_argument("--chunk-tokens", type=int, default=None,
                        help="Token budget per chunk. Default: from --backbone (DeBERTa=1024, "
                             f"ModernBERT/mmBERT=2048), else {MAX_TOKENS_PER_CHUNK}.")
    parser.add_argument("--threshold",    type=float, default=BENCHMARK_THRESHOLD,
                        help="Model confidence threshold (default: 0.7, matches notebook benchmark)")
    parser.add_argument("--use-desc",     action="store_true",
                        help="Use label descriptions in schema (default: False, keys only). "
                             "NOTE: --ner-dataset checkpoints were TRAINED with descriptions "
                             "(to_input_example_generic default), so keys-only eval is a "
                             "train/eval mismatch — pass this flag to match training.")
    parser.add_argument("--schema-case", default="upper", choices=["upper", "original"],
                        help="Label-key casing in the schema. 'upper' = legacy eval behavior; "
                             "'original' = registry casing, the exact strings seen at TRAINING "
                             "time (matters for mixed-case sets like mit_restaurant's 'Amenity'). "
                             "Scoring is case-safe either way.")
    parser.add_argument("--label-names", default="original", choices=["original", "description", "humanized"],
                        help="PROTOCOL.md 9.18: 'description' shows the model each type's "
                             "description as its NAME (descriptions unchanged) and maps "
                             "predictions back to the original types before scoring.")
    parser.add_argument("--desc-file", default=None,
                        help="JSON file overriding label descriptions for the schema "
                             "({label: description}). Keys must match the --ner-dataset "
                             "registry labels (case-insensitive). Enables description A/B "
                             "tests (contrastive / exemplar-in-schema) without retraining. "
                             "Implies --use-desc.")
    parser.add_argument("--emit-duplicates", action="store_true",
                        help="value_broadcast mode: find all occurrences of each entity value "
                             "(default: model_spans — use model's predicted spans directly)")
    parser.add_argument("--log-each-sample", action="store_true",
                        help="Print per-sample processing logs (status, counts, labels, errors)")
    parser.add_argument("--log-text-chars", type=int, default=120,
                        help="Max characters of text preview in each per-sample log line")
    parser.add_argument("--no-extra-val", action="store_true",
                        help="Skip merging rosenxt en/de into eval set (default: merge, matches notebook)")
    parser.add_argument(
        "--ner-dataset", default=None, metavar="KEY",
        help="NER benchmark dataset key (e.g. conll2003, bc5cdr, ontonotes5). "
             "Switches GT building and schema to dataset-native labels (not PII). "
             "Language filter is skipped. Available: "
             "conll2003 bc5cdr fin ontonotes5 mit_restaurant bionlp2004 wnut2017 multinerd.",
    )
    parser.add_argument("--use-rules",    action="store_true", help="Ensemble with regex rule detector")
    parser.add_argument("--timing-probe-samples", type=int, default=TIMING_PROBE_SAMPLES,
                        help=f"Samples run sequentially (timed one-by-one) to measure samples/s "
                             f"(default: {TIMING_PROBE_SAMPLES}). The rest are scored via batch "
                             f"inference for speed and excluded from the samples/s stat.")
    parser.add_argument("--infer-batch-size", type=int, default=PARALLEL_BATCH_SIZE,
                        help=f"model.batch_extract() batch size used for samples beyond "
                             f"--timing-probe-samples (default: {PARALLEL_BATCH_SIZE}). "
                             f"Pass 0 to disable batching and time every sample sequentially.")
    parser.add_argument("--backend",      default="pytorch_fp32", choices=list(ALL_BACKENDS),
                        help="Inference backend (default: pytorch_fp32 = baseline run_inference). "
                             "onnx_cuda_fp16 / onnx_trt_fp16 swap the encoder for an ONNX-Runtime "
                             "session (needs CUDA GPU + onnxruntime-gpu).")
    parser.add_argument("--onnx-path",    default=None,
                        help="Explicit encoder ONNX path (default: <onnx-dir>/<model>_encoder.onnx)")
    parser.add_argument("--onnx-dir",     default="artifacts/onnx",
                        help="Directory for exported encoder ONNX files")
    parser.add_argument("--export-onnx",  action="store_true",
                        help="(Re-)export the encoder ONNX even if a cached file exists")
    parser.add_argument("--output-dir",   default="./eval_output")
    parser.add_argument("--output-repo",  default=None,
                        help="HF dataset repo to push results (defaults to anonymous-org/eval-<model-name>). "
                             "With --push-eval-into-model, this is instead the MODEL repo id "
                             "(defaults to --model).")
    parser.add_argument("--push",         action="store_true", help="Push results to HF Hub")
    parser.add_argument("--push-eval-into-model", action="store_true",
                        help="Push eval results as a subfolder inside the MODEL repo "
                             "(repo_type=model) instead of a separate HF dataset repo. "
                             "Subfolder path is --eval-path-in-repo (default eval/<split>).")
    parser.add_argument("--eval-path-in-repo", default=None,
                        help="Subfolder path inside the model repo for --push-eval-into-model "
                             "(default: eval/<split>)")
    args = parser.parse_args()

    # Resolve chunk_tokens: explicit --chunk-tokens > per-backbone default > global.
    if args.chunk_tokens is None:
        if args.backbone:
            from config.backbones import get_backbone
            args.chunk_tokens = get_backbone(args.backbone).chunk_tokens
        else:
            args.chunk_tokens = MAX_TOKENS_PER_CHUNK
    print(f"[eval] chunk_tokens={args.chunk_tokens}"
          + (f" (from backbone {args.backbone})" if args.backbone else ""))

    from huggingface_hub import login
    if HF_TOKEN:
        login(token=HF_TOKEN)

    apply_tokenizer_patch()

    # Resolve NER label descriptions (when --ner-dataset is given)
    label_descriptions = None
    if args.ner_dataset:
        from config.ner_datasets import NER_DATASETS
        if args.ner_dataset not in NER_DATASETS:
            parser.error(f"Unknown --ner-dataset '{args.ner_dataset}'. "
                         f"Available: {', '.join(sorted(NER_DATASETS.keys()))}")
        ner_cfg = NER_DATASETS[args.ner_dataset]
        label_descriptions = ner_cfg["label_descriptions"]
        print(f"[NER] Dataset key : {args.ner_dataset}")
        print(f"[NER] Labels      : {list(label_descriptions.keys())}")
        if args.desc_file:
            label_descriptions = _load_desc_file(
                args.desc_file, label_descriptions
            )
            args.use_desc = True   # overriding descriptions implies using them
            print(f"[NER] Descriptions overridden from {args.desc_file} (use_desc forced on)")
    elif args.desc_file:
        parser.error("--desc-file requires --ner-dataset")
    if args.label_names != "original":
        if label_descriptions is None:
            parser.error("--label-names requires --ner-dataset")
        if not args.use_desc:
            parser.error(f"--label-names {args.label_names} requires --use-desc")
        if args.label_names == "description":
            shown = {k.upper(): str(v).strip() for k, v in label_descriptions.items()}
        else:                                   # humanized (9.19): "ACCOUNT_NUMBER" -> "account number"
            shown = {k.upper(): k.replace("_", " ").lower().strip() for k in label_descriptions}
        names = [v.upper() for v in shown.values()]
        # every shown name must map back to exactly one type; a shown name equal to its OWN type
        # (case aside) is harmless, equal to ANOTHER type's name is not
        if len(set(names)) != len(names) or any(v.upper() in shown and v.upper() != k for k, v in shown.items()):
            parser.error(f"--label-names {args.label_names}: shown names collide")
        _LABEL_RENAME.update(shown)
        _LABEL_INVERSE.update({v.upper(): k for k, v in shown.items()})
        print(f"[NER] Label names shown as '{args.label_names}' for {len(shown)} types")

    test_data = load_eval_data(args, label_descriptions=label_descriptions)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_gliner2_model(
        args.model,
        max_width=args.max_width,
        use_mmbert=args.mmbert,
        subfolder=args.subfolder,
        override_max_width=not args.keep_config_max_width,
    )
    model = model.to(device)
    model.eval()

    # Swap the encoder for ONNX/TensorRT (or .half()) if a non-baseline backend
    # is requested. pytorch_fp32 is the baseline and leaves the model untouched.
    if args.backend != "pytorch_fp32":
        print(f"Backend: {args.backend}")
        model = apply_backend(model, args)

    # Use NER label descriptions for chunker budget calculation when in NER mode.
    _chunker_labels = label_descriptions if label_descriptions is not None else PII_LABEL_DESCRIPTIONS
    # Uppercase NER labels (to match schema keys passed to model)
    if label_descriptions is not None:
        _chunker_labels = {k.upper(): v for k, v in label_descriptions.items()}
    if args.no_chunk:
        chunker = None
        print("Chunker: DISABLED (--no-chunk)")
    else:
        chunker = build_word_chunker(
            args.chunk_tokens, _chunker_labels,
            CHUNK_SAFETY_MARGIN, MAX_CHARS_PER_CHUNK,
        )
        print(f"Chunker: budget_words={chunker.budget}, max_chars={chunker.max_chars}")

    if args.test_text is not None:
        t0 = time.perf_counter()
        debug_result = run_inference_debug(
            model,
            str(args.test_text),
            args.chunk_tokens,
            args.threshold,
            args.use_rules,
            chunker=chunker,
            use_desc=args.use_desc,
            emit_duplicates=args.emit_duplicates,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        _print_single_text_debug(debug_result)
        print(f"\nSingle-text inference time: {elapsed_ms:.2f} ms")

        os.makedirs(args.output_dir, exist_ok=True)
        debug_out = {
            "text": str(args.test_text),
            "language": args.test_language,
            "inference_time_ms": round(elapsed_ms, 2),
            "threshold": args.threshold,
            "use_rules": args.use_rules,
            "inference_mode": "value_broadcast" if args.emit_duplicates else "model_spans",
            **debug_result,
        }
        debug_path = os.path.join(args.output_dir, "single_text_debug.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_out, f, ensure_ascii=False, indent=2)
        print(f"Saved single-text debug to: {os.path.abspath(debug_path)}")
        return

    overall_dfs, lang_dfs_by_mode, sample_records, timing_stats = evaluate_model(
        model, test_data, args.chunk_tokens, args.use_rules,
        threshold=args.threshold, chunker=chunker, use_desc=args.use_desc,
        schema_case=args.schema_case,
        emit_duplicates=args.emit_duplicates,
        log_each_sample=args.log_each_sample,
        log_text_chars=args.log_text_chars,
        label_descriptions=label_descriptions,
        timing_probe_samples=args.timing_probe_samples,
        infer_batch_size=args.infer_batch_size,
    )

    run_info = {
        "model":        args.model,
        "dataset":      args.dataset if args.test_text is None else None,
        "eval_split":   args.split if args.test_text is None else None,
        "languages":    args.languages if args.test_text is None else [args.test_language],
        "n_samples":    len(test_data),
        "chunk_tokens": args.chunk_tokens,
        "threshold":    args.threshold,
        "iou_threshold_lenient": IOU_THRESH,
        "fuzzy_max_dist_lenient": FUZZY_MAX_DIST,
        "iou_threshold_strict": IOU_THRESH_STRICT,
        "fuzzy_max_dist_strict": FUZZY_MAX_DIST_STRICT,
        "timing_probe_samples": args.timing_probe_samples,
        "infer_batch_size": args.infer_batch_size,
        "use_rules":    args.use_rules,
        "use_desc":     args.use_desc,
        "schema_case":  args.schema_case,
        "desc_file":    args.desc_file,
        "label_names":  args.label_names,
        "inference_mode": "value_broadcast" if args.emit_duplicates else "model_spans",
        "extra_ft_val": (USE_EXTRA_FT_VAL and not args.no_extra_val) if args.test_text is None else False,
        "single_text_mode": args.test_text is not None,
        "mmbert":       args.mmbert,
        "backend":      args.backend,
        "device":       device,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    save_results(args.output_dir, overall_dfs, lang_dfs_by_mode, sample_records, timing_stats, run_info)
    _print_summary(overall_dfs, lang_dfs_by_mode, timing_stats)
    print(f"\nResults saved to: {os.path.abspath(args.output_dir)}")

    if args.push or args.output_repo or args.push_eval_into_model:
        if args.push_eval_into_model:
            repo_id = args.output_repo or args.model
            path_in_repo = args.eval_path_in_repo or f"eval/{args.split}"
            _push_results(args.output_dir, repo_id, args.model,
                          repo_type="model", path_in_repo=path_in_repo)
        else:
            model_slug = args.model.split("/")[-1]
            repo_id = args.output_repo or f"{HF_ORG}/eval-{model_slug}"
            _push_results(args.output_dir, repo_id, args.model)


if __name__ == "__main__":
    main()
