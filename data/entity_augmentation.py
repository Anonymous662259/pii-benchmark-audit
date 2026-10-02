"""Entity-aware augmentation: replace entity values with realistic alternatives.

Strategy
--------
For each *rare* label we maintain pools of realistic replacement values.
At augmentation time a selected document has its entity values replaced
text-in-place while span offsets are recomputed.

Supported labels (with realistic replacement pools):
    PERSON, ORGANIZATION, BANK_ACCOUNT, NATIONAL_ID, PASSPORT,
    TIN, PLATE, IBAN, SWIFT, CARD_NUMBER, PHONE, EMAIL, IP,
    WALLET, API_KEY, CVV, PIN, PASSWORD, ZIP_CODE, EMPLOYEE_ID,
    INSURANCE_ID, TICKET_ID, ACCOUNT_ID, LICENSE, RELIGION,
    ETHNICITY, MARITAL, TRADE_UNION, MEDICAL_INFO, NATIONALITY,
    GENDER, PREFIX

Offset preservation
-------------------
When a replacement value has a different byte-length than the original, all
subsequent span offsets in the same text are shifted accordingly.  The output
example has corrected (start, end) tuples.

Usage
-----
    from data.entity_augmentation import entity_aware_augment
    aug_examples, stats = entity_aware_augment(train_examples, rare_labels)
"""
from __future__ import annotations

import copy
import random
import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from gliner2.training.data import InputExample

# ---------------------------------------------------------------------------
# Replacement value pools — realistic synthetic values per label
# ---------------------------------------------------------------------------

_PERSON_NAMES = [
    "Alice Johnson", "Bob Martinez", "Chloe Nguyen", "David Kim",
    "Emma Wilson", "Frank Müller", "Grace Tanaka", "Hans Becker",
    "Ivan Petrov", "Julia Santos", "Kevin O'Brien", "Li Wei",
    "Maria Garcia", "Nguyen Van An", "Olga Ivanova", "Pierre Dubois",
    "Ravi Sharma", "Sarah Chen", "Thomas Weber", "Uma Patel",
    "Victor Hugo", "Wendy Clarke", "Xiao Ming", "Yuki Yamamoto",
    "Zara Ahmed", "Aiden Murphy", "Beatriz Ferreira", "Carlos López",
    "Diana Novak", "Edward Park",
]

_ORGANIZATIONS = [
    "Acme Financial Group", "Blue Ridge Technologies", "CitiCore Systems",
    "DataFlow Analytics", "Eagle Capital Partners", "FrontLine Consulting",
    "Global Nexus Corp", "Horizon Ventures", "Innotech Solutions",
    "JetStream Logistics", "Kingfisher Industries", "LunarByte Ltd",
    "MapleSoft Inc", "NovaTech GmbH", "OmniDirect AG", "Pinnacle Services",
    "Quantum Dynamics", "Riverstone Group", "Summit Advisors",
    "TerraVox Networks", "UniCore Financials", "Vortex Systems",
    "Wavefront Innovations", "Xenith Holdings", "Yellowstone Partners",
]

_RELIGIONS = [
    "Christianity", "Islam", "Hinduism", "Buddhism", "Judaism",
    "Sikhism", "Jainism", "Shinto", "Zoroastrianism", "Taoism",
    "Atheism", "Agnosticism",
]

_ETHNICITIES = [
    "Asian", "Black or African American", "Hispanic or Latino",
    "Native American", "Pacific Islander", "White or Caucasian",
    "Middle Eastern", "Multiracial", "South Asian", "Southeast Asian",
]

_MARITAL_STATUSES = [
    # English
    "Single", "Married", "Divorced", "Widowed", "Separated",
    "Domestic Partnership", "Civil Union",
    "single", "married", "divorced", "widowed",
    # German
    "Ledig", "Verheiratet", "Geschieden", "Verwitwet", "Getrennt",
    "ledig", "verheiratet", "geschieden",
    # Vietnamese
    "Độc thân", "Đã kết hôn", "Ly hôn", "Góa", "Ly thân",
]

_TRADE_UNIONS = [
    "United Auto Workers", "Teamsters Union", "Service Employees International",
    "American Federation of Teachers", "Communications Workers of America",
    "International Brotherhood of Electrical Workers", "Steel Workers Union",
]

_MEDICAL_INFO = [
    "Type 2 diabetes", "Hypertension", "Asthma", "Hyperlipidemia",
    "Chronic back pain", "Major depressive disorder", "Anxiety disorder",
    "Osteoarthritis", "Hypothyroidism", "GERD",
]

_NATIONALITIES = [
    "American", "British", "German", "French", "Chinese", "Japanese",
    "Indian", "Brazilian", "Vietnamese", "Korean", "Australian", "Canadian",
    "Dutch", "Spanish", "Italian", "Polish", "Russian", "Turkish",
    "Mexican", "Thai", "Indonesian", "Malaysian", "Filipino", "Swedish",
    "Norwegian", "Danish", "Finnish", "Swiss", "Austrian", "Belgian",
    "Portuguese", "Greek", "Czech", "Romanian", "Hungarian", "Irish",
    "South African", "Egyptian", "Nigerian", "Singaporean", "New Zealander",
]

_GENDERS = [
    # English
    "Male", "Female", "Non-binary", "Prefer not to say",
    "male", "female", "non-binary",
    # German
    "Männlich", "Weiblich", "Divers", "männlich", "weiblich", "divers",
    # Vietnamese
    "Nam", "Nữ", "Khác", "nam", "nữ",
    # Abbreviations
    "M", "F", "m", "f",
]

_PREFIXES = [
    # English
    "Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Sir", "Dame", "Rev.", "Capt.", "Lt.",
    "Mr", "Ms", "Mrs", "Dr", "Prof",
    # German
    "Herr", "Frau", "Dr.", "Prof. Dr.", "Dipl.-Ing.",
    # Vietnamese
    "Ông", "Bà", "Anh", "Chị", "Em", "Cô", "Chú", "Bác",
    "TS.", "PGS.", "GS.", "ThS.", "KS.",
]


def _random_digits(n: int, rng: random.Random) -> str:
    return "".join(rng.choices(string.digits, k=n))


def _luhn_complete(partial: str, rng: random.Random) -> str:
    """Generate a 16-digit Luhn-valid card number with a given 6-digit IIN prefix."""
    digits = list(partial[:15] + _random_digits(15 - len(partial[:15]), rng))
    # Compute Luhn check digit
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    check = (10 - (total % 10)) % 10
    return "".join(digits) + str(check)


def _gen_iban(rng: random.Random) -> str:
    country = rng.choice(["DE", "GB", "FR", "NL", "AT", "CH", "SE", "NO"])
    bban = _random_digits(rng.randint(18, 22), rng)
    raw = bban + country + "00"
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in raw)
    check = 98 - (int(numeric) % 97)
    return f"{country}{check:02d}{bban}"


def _gen_swift(rng: random.Random) -> str:
    bank = "".join(rng.choices(string.ascii_uppercase, k=4))
    country = rng.choice(["DE", "GB", "FR", "US", "CH", "NL", "AU", "JP"])
    loc = "".join(rng.choices(string.ascii_uppercase + string.digits, k=2))
    branch = "".join(rng.choices(string.ascii_uppercase + string.digits, k=3))
    return f"{bank}{country}{loc}{branch}"


def _gen_bank_account(rng: random.Random) -> str:
    return _random_digits(rng.randint(8, 12), rng)


def _gen_national_id(rng: random.Random) -> str:
    # Generic 9–12 digit national ID
    return _random_digits(rng.randint(9, 12), rng)


def _gen_passport(rng: random.Random) -> str:
    letters = "".join(rng.choices(string.ascii_uppercase, k=2))
    nums = _random_digits(7, rng)
    return letters + nums


def _gen_tin(rng: random.Random) -> str:
    return _random_digits(9, rng)


def _gen_plate(rng: random.Random) -> str:
    letters = "".join(rng.choices(string.ascii_uppercase, k=rng.randint(2, 3)))
    nums = _random_digits(rng.randint(2, 4), rng)
    return f"{letters}-{nums}"


def _gen_card_number(rng: random.Random) -> str:
    # Visa: 4xxxxx… or Mastercard: 5xxxxx…
    prefix = rng.choice(["4532", "5412", "4916", "5200", "4111"])
    return _luhn_complete(prefix, rng)


def _gen_phone(rng: random.Random) -> str:
    country = rng.choice(["+1", "+44", "+49", "+33", "+84", "+82"])
    body = _random_digits(9, rng)
    return f"{country} {body[:3]} {body[3:6]} {body[6:]}"


def _gen_email(rng: random.Random) -> str:
    domains = ["example.com", "mail.test", "secure.dev", "testmail.org"]
    user = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(5, 9)))
    return f"{user}@{rng.choice(domains)}"


def _gen_ip(rng: random.Random) -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def _gen_wallet(rng: random.Random) -> str:
    # Ethereum-like address
    hex_chars = string.hexdigits[:16]
    return "0x" + "".join(rng.choices(hex_chars, k=40))


def _gen_api_key(rng: random.Random) -> str:
    chars = string.ascii_letters + string.digits
    return "sk-" + "".join(rng.choices(chars, k=40))


def _gen_cvv(rng: random.Random) -> str:
    return _random_digits(rng.randint(3, 4), rng)


def _gen_pin(rng: random.Random) -> str:
    return _random_digits(rng.randint(4, 6), rng)


def _gen_password(rng: random.Random) -> str:
    chars = string.ascii_letters + string.digits + "!@#$%"
    return "".join(rng.choices(chars, k=rng.randint(10, 16)))


def _gen_zip(rng: random.Random) -> str:
    return _random_digits(5, rng)


def _gen_id(rng: random.Random, prefix: str = "EMP") -> str:
    return f"{prefix}-{_random_digits(6, rng)}"


# ---------------------------------------------------------------------------
# Dispatch table: label → generator(rng) → str
# ---------------------------------------------------------------------------

def _make_generators(rng: random.Random) -> Dict[str, Any]:
    return {
        "PERSON":       lambda r: r.choice(_PERSON_NAMES),
        "ORGANIZATION": lambda r: r.choice(_ORGANIZATIONS),
        "BANK_ACCOUNT": _gen_bank_account,
        "NATIONAL_ID":  _gen_national_id,
        "PASSPORT":     _gen_passport,
        "TIN":          _gen_tin,
        "PLATE":        _gen_plate,
        "IBAN":         _gen_iban,
        "SWIFT":        _gen_swift,
        "CARD_NUMBER":  _gen_card_number,
        "PHONE":        _gen_phone,
        "EMAIL":        _gen_email,
        "IP":           _gen_ip,
        "WALLET":       _gen_wallet,
        "API_KEY":      _gen_api_key,
        "CVV":          _gen_cvv,
        "PIN":          _gen_pin,
        "PASSWORD":     _gen_password,
        "ZIP_CODE":     _gen_zip,
        "EMPLOYEE_ID":  lambda r: _gen_id(r, "EMP"),
        "INSURANCE_ID": lambda r: _gen_id(r, "INS"),
        "TICKET_ID":    lambda r: _gen_id(r, "TKT"),
        "ACCOUNT_ID":   lambda r: _gen_id(r, "ACC"),
        "LICENSE":      lambda r: r.choice(string.ascii_uppercase) + _random_digits(7, r),
        "RELIGION":     lambda r: r.choice(_RELIGIONS),
        "ETHNICITY":    lambda r: r.choice(_ETHNICITIES),
        "MARITAL":      lambda r: r.choice(_MARITAL_STATUSES),
        "TRADE_UNION":  lambda r: r.choice(_TRADE_UNIONS),
        "MEDICAL_INFO": lambda r: r.choice(_MEDICAL_INFO),
        "NATIONALITY":  lambda r: r.choice(_NATIONALITIES),
        "GENDER":       lambda r: r.choice(_GENDERS),
        "PREFIX":       lambda r: r.choice(_PREFIXES),
    }


# ---------------------------------------------------------------------------
# Span reconstruction after text-replacement
# ---------------------------------------------------------------------------

def _replace_in_text_with_spans(
    text: str,
    spans: List[Dict[str, Any]],
    label_generators: Dict[str, Any],
    rng: random.Random,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace entity values in ``text`` and fix span offsets.

    Only labels that have a registered generator are replaced.
    Replacements are applied right-to-left (highest start first) so earlier
    offsets remain valid.

    Args:
        text: Original text.
        spans: List of {"label", "start", "end", "text"} dicts.
        label_generators: {label: generator_fn(rng)} dict.
        rng: Random number generator.

    Returns:
        (new_text, new_spans) with updated offsets.
    """
    if not spans:
        return text, spans

    # Sort spans by start descending for right-to-left replacement
    sortable = sorted(enumerate(spans), key=lambda x: x[1]["start"], reverse=True)

    new_text = text
    offset_adjustments: List[Tuple[int, int]] = []  # (position, delta)
    replaced_positions: Dict[int, int] = {}  # start → new_length (for self-span fix)

    for orig_idx, span in sortable:
        label = span.get("label", "")
        gen = label_generators.get(label)
        if gen is None:
            continue
        old_val = span.get("text", "")
        start = span["start"]
        end = span["end"]
        # Verify the span aligns with text (skip if stale)
        if new_text[start:end] != old_val:
            continue
        new_val = gen(rng)
        # Avoid replacing with same length-0 value
        if not new_val:
            continue
        new_text = new_text[:start] + new_val + new_text[end:]
        delta = len(new_val) - len(old_val)
        offset_adjustments.append((start, delta))
        replaced_positions[start] = len(new_val)

    # Rebuild spans with shifted offsets
    if not offset_adjustments:
        return new_text, spans

    new_spans = [dict(s) for s in spans]
    for i, span in enumerate(new_spans):
        orig_start = span["start"]
        shift = sum(delta for pos, delta in offset_adjustments if pos < orig_start)
        new_spans[i]["start"] = orig_start + shift
        # If this span was itself replaced, use the new value length
        if orig_start in replaced_positions:
            new_spans[i]["end"] = new_spans[i]["start"] + replaced_positions[orig_start]
        else:
            new_spans[i]["end"] = span["end"] + shift
        # Update cached text value
        new_spans[i]["text"] = new_text[new_spans[i]["start"]:new_spans[i]["end"]]

    return new_text, new_spans


# ---------------------------------------------------------------------------
# InputExample → record → augmented InputExample
# ---------------------------------------------------------------------------

def _example_to_spans(ex: InputExample) -> List[Dict[str, Any]]:
    """Reconstruct span list with (start, end) from the example.

    When entity_char_spans is available, uses exact ground-truth offsets directly
    (no text search). Falls back to blind text.find only when entity_char_spans is
    absent (e.g. hand-built InputExample in tests/scripts).
    """
    ecs = getattr(ex, "entity_char_spans", None)
    if ecs is not None:
        spans = [
            {"label": label, "start": start, "end": end, "text": text}
            for label, triples in ecs.items()
            for (text, start, end) in triples
        ]
        return sorted(spans, key=lambda s: s["start"])

    # Fallback: blind text search (ambiguous when multiple same-value occurrences exist)
    spans = []
    text = ex.text or ""
    for label, values in (ex.entities or {}).items():
        for val in (values or []):
            if not val:
                continue
            idx = 0
            while True:
                pos = text.find(val, idx)
                if pos == -1:
                    break
                spans.append({"label": label, "start": pos, "end": pos + len(val), "text": val})
                idx = pos + 1
    return sorted(spans, key=lambda s: s["start"])


def _spans_to_example(
    text: str,
    spans: List[Dict[str, Any]],
    original_ex: InputExample,
) -> InputExample:
    """Rebuild an InputExample from (text, spans), preserving label set."""
    from collections import defaultdict as _defaultdict
    from data.converter import spans_to_char_spans

    entities: Dict[str, List[str]] = _defaultdict(list)
    # Keep all labels from original (empty → empty)
    for label in (original_ex.entities or {}):
        entities[label] = []
    for span in spans:
        entities[span["label"]].append(span["text"])
    # Deduplicate values while preserving order
    deduped = {k: list(dict.fromkeys(v)) for k, v in entities.items()}

    kwargs: Dict[str, Any] = {"text": text, "entities": deduped}
    for attr in ("entity_descriptions", "classifications", "structures", "relations"):
        v = getattr(original_ex, attr, None)
        if v:
            kwargs[attr] = v
    # Rebuild exact char spans from the (possibly value-replaced) span list so
    # augmented copies carry correct positions through to the processor.
    char_spans = spans_to_char_spans(spans)
    if char_spans is not None:
        kwargs["entity_char_spans"] = char_spans
    return InputExample(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def entity_aware_augment(
    examples: List[InputExample],
    rare_labels: Optional[Set[str]] = None,
    aug_ratio: float = 0.4,
    aug_factor: int = 1,
    seed: int = 42,
) -> Tuple[List[InputExample], Dict[str, Any]]:
    """Augment examples containing rare entities by replacing entity values.

    Args:
        examples: Training examples.
        rare_labels: Set of rare label names.  If None, all labels with a
            registered generator are eligible.
        aug_ratio: Fraction of eligible examples to augment.
        aug_factor: Number of augmented copies per selected example.
        seed: Reproducibility seed.

    Returns:
        (augmented_examples, stats_dict)
    """
    rng = random.Random(seed)
    gen_table = _make_generators(rng)

    if rare_labels is None:
        eligible_labels = frozenset(gen_table.keys())
    else:
        eligible_labels = frozenset(rare_labels) & frozenset(gen_table.keys())

    # Identify examples that contain at least one augmentable entity
    eligible_idx = [
        i for i, ex in enumerate(examples)
        if any(
            label in eligible_labels and values
            for label, values in (ex.entities or {}).items()
        )
    ]

    n_to_aug = max(1, int(round(len(eligible_idx) * aug_ratio)))
    selected_idx = set(rng.sample(eligible_idx, min(n_to_aug, len(eligible_idx))))

    result: List[InputExample] = list(examples)
    n_added = 0
    label_aug_counts: Counter = defaultdict(int)

    for idx in selected_idx:
        ex = examples[idx]
        spans = _example_to_spans(ex)
        for _ in range(aug_factor):
            new_text, new_spans = _replace_in_text_with_spans(
                ex.text, spans, gen_table, rng
            )
            if new_text == ex.text:
                continue  # nothing was replaced (no generator matched)
            aug_ex = _spans_to_example(new_text, new_spans, ex)
            result.append(aug_ex)
            n_added += 1
            for span in new_spans:
                if span["label"] in eligible_labels:
                    label_aug_counts[span["label"]] += 1

    rng.shuffle(result)

    stats: Dict[str, Any] = {
        "original_size": len(examples),
        "final_size": len(result),
        "added_examples": n_added,
        "eligible_examples": len(eligible_idx),
        "selected_for_aug": len(selected_idx),
        "augmented_labels": dict(sorted(label_aug_counts.items())),
    }
    return result, stats


def print_entity_aug_stats(stats: Dict[str, Any]) -> None:
    """Pretty-print entity-augmentation statistics."""
    print(
        f"\nEntity-aware augmentation stats:\n"
        f"  Original examples : {stats['original_size']:,}\n"
        f"  Final examples    : {stats['final_size']:,}\n"
        f"  Added examples    : {stats['added_examples']:,}\n"
        f"  Eligible docs     : {stats['eligible_examples']:,}\n"
        f"  Selected for aug  : {stats['selected_for_aug']:,}\n"
    )
    if stats.get("augmented_labels"):
        print("  Augmented entity counts:")
        for lbl, cnt in sorted(stats["augmented_labels"].items(), key=lambda x: -x[1]):
            print(f"    {lbl:<20} {cnt:>6}")
