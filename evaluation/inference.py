"""Inference utilities: prediction normalization, post-processing.

Text chunking lives in :mod:`components.wordChunker` (RecursiveChunkerWrapper).
"""
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# ── Span priority for conflict resolution (native schema) ────────────────────
# Specific/structured labels win overlaps against generic ones (e.g. an IBAN or
# CARD_NUMBER should beat a generic BANK_ACCOUNT on the same span).
_LABEL_PRIORITY: Dict[str, int] = {
    "WALLET": 10,
    "IBAN": 10,
    "CARD_NUMBER": 10,
    "NATIONAL_ID": 10,
    "IP": 10,
    "URL": 10,
    "EMAIL": 10,
    "MONEY": 10,
    "AGE": 10,
}

_STRIP_CHARS = frozenset(["­", "​", "‌", "‍", "⁠", "﻿", "�"])
_C1_RE = re.compile(r"[\x80-\x9f]")
_C0_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[^\S\n]+")


def clean_text_with_mapping(text: str) -> Tuple[str, List[int]]:
    """Return (cleaned_text, char_map). Matches notebook implementation exactly.

    char_map[i] = original position of cleaned_text[i].
    A sentinel char_map[len(cleaned)] = len(text) is appended for end-exclusive lookups.
    """
    if not text:
        return text, []

    text = unicodedata.normalize("NFC", text)

    cleaned_chars: List[str] = []
    char_map: List[int] = []
    orig_pos = 0
    for ch in text:
        if ch in _STRIP_CHARS:
            orig_pos += 1
            continue
        if _C0_RE.match(ch):
            orig_pos += 1
            continue
        if _C1_RE.match(ch):
            cleaned_chars.append(" ")
            char_map.append(orig_pos)
            orig_pos += 1
            continue
        cleaned_chars.append(ch)
        char_map.append(orig_pos)
        orig_pos += 1

    cleaned = "".join(cleaned_chars)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()

    final_map: List[int] = []
    i = 0
    for ch in cleaned:
        while i < len(char_map) and (text[char_map[i]] != ch and not (ch == " " and text[char_map[i]].isspace())):
            i += 1
        final_map.append(char_map[i] if i < len(char_map) else len(text))
        i += 1
    final_map.append(len(text))

    return cleaned, final_map


def _find_all_occurrences(text: str, value: str) -> List[Tuple[int, int]]:
    """Find ALL (start, end) positions of value in text. Used by value_broadcast mode."""
    spans = []
    if not value:
        return spans
    start = 0
    while True:
        idx = text.find(value, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(value)))
        start = idx + max(1, len(value))
    return spans


def normalize_predictions(raw_preds, text: str, emit_duplicates: bool = False) -> List[Dict]:
    """Normalize GLiNER2 model.extract() output to a flat list of span dicts.

    Args:
        raw_preds: model.extract output (dict or list).
        text: chunk text used for value broadcasting.
        emit_duplicates: if True, broadcast each entity value to ALL its occurrences
                         in the text (value_broadcast mode). If False, use model spans.

    Deduplicates by (label, start, end) keeping highest score.
    """
    spans = []
    if isinstance(raw_preds, dict):
        entities = raw_preds.get("entities", raw_preds)
        # model.extract(..., format_results=False) returns entities as a single-
        # element list wrapping the {LABEL: [span dicts]} mapping (one dict per
        # sample). This raw form keeps every predicted position — it skips the
        # engine's value-level dedup — so model_spans no longer drops repeated
        # values. format_results=True returns the mapping directly. Handle both.
        if isinstance(entities, list):
            entities = entities[0] if (entities and isinstance(entities[0], dict)) else {}
        if not isinstance(entities, dict):
            entities = {}
        for label, items in entities.items():
            if not isinstance(items, list):
                continue
            label = str(label).upper().strip()
            for item in items:
                if isinstance(item, dict):
                    s = item.get("start")
                    e = item.get("end")
                    txt = item.get("text") or item.get("value") or ""
                    score = float(item.get("score", item.get("confidence", 1.0)) or 0.0)
                    has_span = s is not None and e is not None and int(e) > int(s)
                    if emit_duplicates and txt:
                        for cs, ce in _find_all_occurrences(text, str(txt)):
                            spans.append({"label": label, "start": cs, "end": ce,
                                          "text": str(txt), "score": score})
                    elif has_span:
                        spans.append({
                            "label": label, "start": int(s), "end": int(e),
                            "text": item.get("text", text[int(s):int(e)]),
                            "score": score,
                        })
                    elif txt:
                        for cs, ce in _find_all_occurrences(text, str(txt)):
                            spans.append({"label": label, "start": cs, "end": ce,
                                          "text": str(txt), "score": score})
                elif isinstance(item, str):
                    for cs, ce in _find_all_occurrences(text, item):
                        spans.append({"label": label, "start": cs, "end": ce,
                                      "text": item, "score": 0.0})
    elif isinstance(raw_preds, list):
        spans = list(raw_preds)

    # Dedup keeping highest score
    best: Dict[tuple, Dict] = {}
    for s in spans:
        key = (s.get("label"), s.get("start"), s.get("end"))
        if key not in best or s.get("score", 0) > best[key].get("score", 0):
            best[key] = s
    return list(best.values())


def resolve_span_conflicts(
    entities: List[Dict],
    strategy: str = "score_first",
    rule_labels: Optional[set] = None,
) -> List[Dict]:
    """Drop overlapping spans. Matches notebook implementation exactly.

    Strategies:
      - "score_first":   is_rule > score > priority > length
      - "longest_first": is_rule > length > score > priority
      - "rule_first":    is_rule > priority > score > length
    """
    if not entities:
        return []

    from evaluation.rule_detector import RULE_BASED_LABELS as _RULE_LABELS
    _rule_labels = rule_labels if rule_labels is not None else _RULE_LABELS

    def _sort_key(e: Dict):
        is_rule = 1 if e.get("label") in _rule_labels else 0
        score = 1.0 if is_rule else float(e.get("score") or 0.0)
        length = e["end"] - e["start"]
        prio = _LABEL_PRIORITY.get(e.get("label", ""), 0)
        if strategy == "score_first":
            return (is_rule, score, prio, length)
        elif strategy == "longest_first":
            return (is_rule, length, score, prio)
        else:  # rule_first
            return (is_rule, prio, score, length)

    ranked = sorted(entities, key=_sort_key, reverse=True)
    kept: List[Dict] = []
    occupied: List[Tuple[int, int]] = []

    def _overlaps(s: int, e: int) -> bool:
        for (ks, ke) in occupied:
            if s < ke and e > ks:
                return True
        return False

    for ent in ranked:
        s, e = ent["start"], ent["end"]
        if not _overlaps(s, e):
            kept.append(ent)
            occupied.append((s, e))
    return sorted(kept, key=lambda x: x["start"])


def apply_advanced_postprocessing(entities: List[Dict], original_text: str) -> List[Dict]:
    """Apply notebook post-processing rules exactly.

    1. Drop DATE/TIME if immediately wrapped by [ ... ] (single char on both sides).
    2. Keep PREFIX only if next entity is PERSON within 3 chars.
    3. Drop any entity whose value contains ** / *** / **** (masking placeholders).
    4. Drop numeric-type entities (CARD_NUMBER, PHONE, …) whose value contains XX+.
    """
    if not entities:
        return []

    from evaluation.validators import apply_span_validators
    entities = apply_span_validators(entities)

    sorted_ents = sorted(entities, key=lambda x: x["start"])
    final_entities: List[Dict] = []
    for i, ent in enumerate(sorted_ents):
        label = ent.get("label")
        start = ent["start"]
        end = ent["end"]

        if label in ("TIME", "DATE"):
            prev_char = original_text[start - 1] if start > 0 else ""
            next_char = original_text[end] if end < len(original_text) else ""
            if prev_char == "[" and next_char == "]":
                continue

        if label == "PREFIX":
            has_person_after = False
            if i + 1 < len(sorted_ents):
                next_ent = sorted_ents[i + 1]
                if next_ent.get("label") in {"PERSON", "PERSONAL_NAME"}:
                    gap = next_ent["start"] - end
                    if 0 <= gap <= 3:
                        has_person_after = True
            if not has_person_after:
                continue

        final_entities.append(ent)
    return final_entities
