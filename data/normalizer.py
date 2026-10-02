import ast
import json
from typing import Any, Dict, List, Optional

from config.labels import LABEL_MAPPINGS, PII_LABEL_DESCRIPTIONS


def parse_mask(mask_data) -> List[Dict]:
    """Parse privacy_mask field from JSON string, literal string, or list."""
    if isinstance(mask_data, str):
        try:
            return json.loads(mask_data)
        except Exception:
            try:
                return ast.literal_eval(mask_data)
            except Exception:
                return []
    return mask_data if isinstance(mask_data, list) else []


def normalize_label(label: str) -> str:
    """Uppercase + LABEL_MAPPINGS lookup."""
    label = str(label or "").upper().strip()
    return LABEL_MAPPINGS.get(label, label)


def dedupe_spans(spans: List[Dict]) -> List[Dict]:
    """Remove duplicate spans, keeping stable order."""
    seen, out = set(), []
    for item in sorted(spans, key=lambda x: (x["start"], x["end"], x["label"], x["text"])):
        key = (item["start"], item["end"], item["label"], item["text"])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def normalize_record(example: Dict[str, Any]) -> Dict[str, Any]:
    """Full normalization pipeline: raw HF dataset row → clean record dict.

    Returns {"text": str, "language": str, "spans": List[Dict]}.
    Drops spans with invalid offsets or labels not in PII_LABEL_DESCRIPTIONS.
    """
    text = str(example.get("source_text", "") or "")
    raw_masks = parse_mask(example.get("privacy_mask", []))
    spans = []
    for m in raw_masks:
        try:
            start = int(m.get("start"))
            end = int(m.get("end"))
        except Exception:
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        key = normalize_label(m.get("label"))
        if not key or key not in PII_LABEL_DESCRIPTIONS:
            continue
        span_value = m.get("value") or text[start:end]
        if not span_value or span_value not in text:
            continue
        spans.append({
            "start": start,
            "end": end,
            "label": key,
            "text": span_value,
        })
    return {
        "text": text,
        "language": str(example.get("language", "") or ""),
        "spans": dedupe_spans(spans),
    }


def normalize_record_generic(
    example: Dict[str, Any],
    allowed_labels: Optional[set] = None,
) -> Dict[str, Any]:
    """Like normalize_record but without PII-schema label filtering.

    Labels are kept as-is (no uppercase, no LABEL_MAPPINGS translation).
    If ``allowed_labels`` is provided (set of strings), only spans whose label
    is in that set are kept.  Pass None to keep all non-empty labels.

    Returns {"text": str, "language": str, "spans": List[Dict]}.
    """
    text = str(example.get("source_text", "") or "")
    raw_masks = parse_mask(example.get("privacy_mask", []))
    spans = []
    for m in raw_masks:
        try:
            start = int(m.get("start"))
            end = int(m.get("end"))
        except Exception:
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        key = str(m.get("label") or "").strip()
        if not key:
            continue
        if allowed_labels is not None and key not in allowed_labels:
            continue
        span_value = m.get("value") or text[start:end]
        if not span_value or span_value not in text:
            continue
        spans.append({
            "start": start,
            "end": end,
            "label": key,
            "text": span_value,
        })
    return {
        "text": text,
        "language": str(example.get("language", "") or ""),
        "spans": dedupe_spans(spans),
    }
