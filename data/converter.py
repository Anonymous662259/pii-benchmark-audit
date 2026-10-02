from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from gliner2.training.data import InputExample

from config.labels import PII_LABEL_DESCRIPTIONS


def spans_to_entities(spans: List[Dict]) -> Dict[str, List[str]]:
    """Convert span list → {label: [values]} dict."""
    entities: Dict[str, List[str]] = defaultdict(list)
    for s in spans:
        entities[s["label"]].append(s["text"])
    return dict(entities)


def spans_to_char_spans(
    spans: List[Dict],
) -> Optional[Dict[str, List[Tuple[str, int, int]]]]:
    """Convert span list → {label: [(text, start, end)]} preserving character offsets.

    Only spans that carry both 'start' and 'end' keys are included.
    Returns None when no span in the list has positional information.
    """
    result: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for s in spans:
        start = s.get("start")
        end = s.get("end")
        if start is None or end is None:
            continue
        result[s["label"]].append((s["text"], int(start), int(end)))
    return dict(result) if result else None


def to_input_example(record: Dict[str, Any], use_label_descriptions: bool = True) -> InputExample:
    """Convert normalized record → GLiNER2 InputExample.

    All 44 labels are included (empty list for missing ones) so that
    no-entity samples are still kept and the model learns negative examples.
    """
    text = record["text"]
    spans = record["spans"]
    entities = spans_to_entities(spans)
    full_entities = {label: entities.get(label, []) for label in PII_LABEL_DESCRIPTIONS}
    kwargs: Dict[str, Any] = {"text": text, "entities": full_entities}
    if use_label_descriptions:
        kwargs["entity_descriptions"] = PII_LABEL_DESCRIPTIONS
    char_spans = spans_to_char_spans(spans)
    if char_spans is not None:
        kwargs["entity_char_spans"] = char_spans
    return InputExample(**kwargs)


def to_input_example_generic(
    record: Dict[str, Any], 
    label_descriptions: Dict[str, str],
    use_label_descriptions: bool = True
) -> InputExample:
    """Convert normalized record → GLiNER2 InputExample using custom labels.

    Generic version that accepts arbitrary label_descriptions instead of
    always using PII_LABEL_DESCRIPTIONS. Useful for custom NER datasets.
    
    Args:
        record: Normalized record with 'text' and 'spans' keys
        label_descriptions: Dict mapping label → description
        use_label_descriptions: Whether to include entity_descriptions in output
    
    Returns:
        InputExample with all labels from label_descriptions included
    """
    text = record["text"]
    spans = record["spans"]
    entities = spans_to_entities(spans)
    full_entities = {label: entities.get(label, []) for label in label_descriptions}
    kwargs: Dict[str, Any] = {"text": text, "entities": full_entities}
    if use_label_descriptions:
        kwargs["entity_descriptions"] = label_descriptions
    char_spans = spans_to_char_spans(spans)
    if char_spans is not None:
        kwargs["entity_char_spans"] = char_spans
    return InputExample(**kwargs)


def is_valid_example(ex: InputExample) -> bool:
    """Reject examples with missing/empty text or entity values absent from text."""
    text = getattr(ex, "text", None)
    if not isinstance(text, str) or not text.strip():
        return False
    entities = getattr(ex, "entities", {}) or {}
    for values in entities.values():
        for v in (values or []):
            if v and v not in text:
                return False
    return True
