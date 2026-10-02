"""Span-level validators for GLiNER2 post-processing.

Mirrors the GLiNER2 RegexValidator(pattern, exclude=True) concept but operates
on already-extracted span dicts (list of {label, start, end, text, score}).

Usage:
    from evaluation.validators import MASKING_VALIDATOR, NUMERIC_MASK_VALIDATOR
    spans = MASKING_VALIDATOR.filter(spans)
    spans = NUMERIC_MASK_VALIDATOR.filter(spans)

    # Or combined convenience call:
    spans = apply_span_validators(spans)
"""
import re
from typing import Dict, List, Optional, Set


class SpanRegexValidator:
    """Exclude (or keep) span dicts whose ``text`` field matches a regex pattern.

    Mirrors ``RegexValidator(pattern, mode, exclude, flags)`` from the GLiNER2
    schema API, but applied to a flat list of span dicts rather than a schema
    field pipeline.

    Args:
        pattern: Regex pattern string or compiled ``re.Pattern``.
        mode: ``"full"`` (re.fullmatch) or ``"partial"`` (re.search).
        exclude: When ``True`` (default), spans that *match* are removed.
                 When ``False``, spans that do *not* match are removed.
        flags: ``re.RegexFlag`` value used when *pattern* is a string.
        labels: If given, validator only applies to spans whose ``label``
                is in this set.  Spans with other labels pass through unchanged.
    """

    def __init__(
        self,
        pattern: str,
        mode: str = "partial",
        exclude: bool = True,
        flags: int = 0,
        labels: Optional[Set[str]] = None,
    ):
        if isinstance(pattern, str):
            self._re = re.compile(pattern, flags)
        else:
            self._re = pattern
        self._mode = mode
        self._exclude = exclude
        self._labels: Optional[Set[str]] = (
            {lbl.upper() for lbl in labels} if labels is not None else None
        )

    def _matches(self, text: str) -> bool:
        if self._mode == "full":
            return bool(self._re.fullmatch(text))
        return bool(self._re.search(text))

    def validate(self, span: Dict) -> bool:
        """Return ``True`` if the span should be *kept*."""
        label = str(span.get("label", "")).upper()
        if self._labels is not None and label not in self._labels:
            return True  # not in scope → keep unconditionally
        text = str(span.get("text") or span.get("value") or "")
        matched = self._matches(text)
        return not matched if self._exclude else matched

    def filter(self, spans: List[Dict]) -> List[Dict]:
        """Return only spans that pass this validator."""
        return [s for s in spans if self.validate(s)]


# ---------------------------------------------------------------------------
# Numeric / identifier labels that may be masked with "XX…" placeholders
# ---------------------------------------------------------------------------
_NUMERIC_LABELS: Set[str] = {
    "CARD_NUMBER",
    "PHONE",
    "PIN",
    "CVV",
    "TIN",
    "BANK_ACCOUNT",
    "IBAN",
    "SWIFT",
    "WALLET",
    "IP",
    "ZIP_CODE",
}

# ---------------------------------------------------------------------------
# Pre-built validators
# ---------------------------------------------------------------------------

# Rule 1: Any entity whose value contains two or more consecutive asterisks (**).
# Pattern excludes "**", "***", "****", etc.
MASKING_VALIDATOR = SpanRegexValidator(
    pattern=r"\*{2,}",
    mode="partial",
    exclude=True,
    labels=None,  # applies to ALL labels
)

# Rule 2: Numeric/identifier labels whose value contains two or more consecutive
# uppercase X's (XX, XXX, XXXX …) — masked placeholders like "XXXX-XXXX".
NUMERIC_MASK_VALIDATOR = SpanRegexValidator(
    pattern=r"X{2,}",
    mode="partial",
    exclude=True,
    flags=0,       # case-sensitive: only uppercase X triggers the rule
    labels=_NUMERIC_LABELS,
)

# Ordered list applied by apply_span_validators()
_DEFAULT_VALIDATORS: List[SpanRegexValidator] = [
    MASKING_VALIDATOR,
    NUMERIC_MASK_VALIDATOR,
]


def apply_span_validators(
    spans: List[Dict],
    validators: Optional[List[SpanRegexValidator]] = None,
) -> List[Dict]:
    """Run all validators over *spans* in order, returning only passing spans.

    Args:
        spans: Flat list of span dicts ({label, start, end, text, score}).
        validators: Override the default validator list. ``None`` → use the
                    module-level ``_DEFAULT_VALIDATORS``.
    """
    active = _DEFAULT_VALIDATORS if validators is None else validators
    result = spans
    for v in active:
        result = v.filter(result)
    return result
