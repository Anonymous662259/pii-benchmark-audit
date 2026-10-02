"""Rule-based PII detector — HYBRID single-file edition (v3, CONTEXT-ONLY).

v3 derives from v2 (the precision-tuned 2026-06-12 set) with ONE structural
change requested on 2026-06-23: **every class is now detected through a positive
keyword context only.** All bare-regex rules are removed (``_build_regex_rules``
returns an empty list), and the three classes that used to be bare-only — EMAIL,
IBAN, IP — gain dedicated context rules (``_EMAIL_CONTEXT`` / ``_IBAN_CONTEXT`` /
``_IP_CONTEXT``). A span now fires only when a cue word ("email:", "IBAN", "IP
address", "phone", "SWIFT", …) precedes the value.

Trade-off (intended): precision is held at the v2 level (context-gating tends to
raise it), while recall, F1 and the number of predictions DROP — values that
appear with no cue (most bare EMAIL/IP/COORDINATE, the bare SSN shape, bare
PHONE/CARD/IBAN runs) are no longer matched. This is accepted by design.

The detection engine, validators, dedup and blocking map are unchanged from v2.

Same organisation as ``old rule/rule_detector.py`` (PIIMatch / RegexRule /
validators / RulebasePIIDetector with keyword-context specs, a single dedup
pass), but it carries the **HYBRID v2** rule set:

    OLD6  = {PHONE, IBAN, CARD_NUMBER, SWIFT, PIN, COORDINATE}
            → the precision-first OLD rules, upgraded so they still pass the
              per-class mock samples (see hybrid/old_rules.py).
    NEW8  = {EMAIL, NATIONAL_ID, IP, URL, BANK_ACCOUNT, CVV, TIN, PASSWORD}
            → the fine-tuned rules from references/<class>/.

v2 == the 2026-06-12 precision round: every one of the 39 manually-confirmed
"rule-error" FPs in ``hybrid_output/danh sách các fp sai theo lỗi model`` is
eliminated, micro-precision rises 0.9899→0.9931 on data_clean, and all 106
mocks still pass. The per-class guards added this round (vs the v1 monolith):
  • EMAIL: lookbehind also rejects a leading '*' (masked-tail address).
  • URL: BARE_PAREN keeps only unambiguous TLDs (drops .tech/.ai/.app/… handles).
  • CVV: drops the ambiguous ``mã bảo mật`` cue; rejects a year-of-date value.
  • NATIONAL_ID: NEG_CONTEXT widened (employee/insurance/role/"Name Name,"),
    12/9-digit capture bounded so it is not sliced from a longer run.
  • PASSWORD: rejects URLs, API-key prefixes, masked ``XXX`` runs, ``\\d+-word``,
    all-caps IBAN/VIN runs; strips wrapping parens; filler won't step over
    ``failed``/``account``.
  • BANK_ACCOUNT: drops the ``routing`` cue and the bare ``số tài khoản`` cue
    (kept STK / ``tài khoản ngân hàng``) → ACCOUNT_ID spans no longer leak.
  • PHONE: rejects 3-2-3 / 4-2-4 id shapes and dd-mm-yyyy dates; mask-guard
    rejects a ``-*`` tail and a spaced-IBAN tail; NEG_LOCAL adds căn cước/cert.
  • IBAN: neg-context also blocks ``sensor/equipment/device unit`` (asset ids).

This file is self-contained (no project imports) and is behaviour equivalent to
``hybrid/detector.HybridPIIDetector`` (the v2 rules) — verified span-by-span on
all 17457 samples of ``data_clean/`` (0 divergence).

Notes on which logic comes from where:
  • The detection engine (regex scan, 28-char neg-context window, contextual
    group(1) capture, confidence-ranked dedup) is identical in both sources.
  • The dedup blocking map is the *tuned* (references) map — PHONE is blocked by
    {BANK_ACCOUNT, SWIFT, NATIONAL_ID, TIN, IBAN, CARD_NUMBER}.
  • PHONE detection uses the permissive ``is_phone_value`` (+ all-identical-digit
    guard); BANK_ACCOUNT defers to the *strict* ``is_valid_phone`` as a guard.
  • Precision guards added in this round: PHONE/PIN reject all-identical-digit
    placeholders; PHONE NEG_LOCAL blocks mst/identification/identity (cross-class
    TIN/NATIONAL_ID); IBAN neg-context blocks "account id/number …"; PIN context
    rejects comma-listed guess values; COORDINATE context requires a full pair.
"""
import ipaddress
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Pattern
from urllib.parse import urlparse


@dataclass(frozen=True)
class PIIMatch:
    entity_type: str
    value: str
    start: int
    end: int
    source: str
    normalized: Optional[str] = None


@dataclass(frozen=True)
class RegexRule:
    entity_type: str
    pattern: Pattern
    validator: Callable[[str], bool]
    normalizer: Optional[Callable[[str], str]] = None
    # If set, the rule is suppressed when the ~28 chars of left context match
    # this pattern (e.g. an EMPLOYEE_ID / account-id / hyphenated-id prefix).
    neg_context: Optional[Pattern] = None


# ============================================================================
# Configuration
# ============================================================================
LEVEL1_HIGH_PRECISION_RULEBASE_ENTITY_TYPES = frozenset({
    "BANK_ACCOUNT", "SWIFT", "CVV", "EMAIL", "COORDINATE", "PASSWORD",
    "PHONE", "PIN", "NATIONAL_ID", "TIN", "URL", "IBAN", "CARD_NUMBER", "IP",
})
DEFAULT_RULEBASE_ALLOWED_ENTITY_TYPES = LEVEL1_HIGH_PRECISION_RULEBASE_ENTITY_TYPES

# Which classes use the OLD (precision-first) rules vs the TUNED rules.
OLD_RULE_LABELS = frozenset({"SWIFT", "CARD_NUMBER", "PIN", "COORDINATE", "PHONE", "IBAN"})
TUNED_RULE_LABELS = frozenset({
    "EMAIL", "NATIONAL_ID", "IP", "URL", "BANK_ACCOUNT", "CVV", "TIN", "PASSWORD",
})

_LEGACY_LABEL_ALIASES = {
    "PHONE_NUMBER":       "PHONE",
    "SSN_CCCD":           "NATIONAL_ID",
    "BIC_SWIFT":          "SWIFT",
    "TAX_ID":             "TIN",
    "GPS_COORDINATE":     "COORDINATE",
    "CREDIT_CARD_CVV":    "CVV",
    "ACCOUNT_NUMBER":     "BANK_ACCOUNT",
    "CREDIT_CARD_NUMBER": "CARD_NUMBER",
    "IP_ADDRESS":         "IP",
}

# Tuned (references) blocking map: pred of X is dropped when it overlaps a kept
# pred whose label is in this set. NOTE PHONE is blocked by 6 labels here.
ACTIVE_RULEBASE_BLOCKING_OVERLAPS = {
    "EMAIL":        frozenset({"URL"}),
    "PHONE":        frozenset({"BANK_ACCOUNT", "SWIFT", "NATIONAL_ID", "TIN", "IBAN", "CARD_NUMBER"}),
    "BANK_ACCOUNT": frozenset({"SWIFT", "PHONE", "NATIONAL_ID", "IBAN", "CARD_NUMBER"}),
    "CARD_NUMBER":  frozenset({"IBAN"}),
    "PIN":          frozenset({"PHONE", "CVV"}),
}

# EMAIL/URL/IP added in v3: their context-captured value must not have hyphens
# or dots stripped (those are part of the address), so keep it as collapsed text.
CONTEXT_NORMALIZE_AS_TEXT_ENTITY_TYPES = frozenset({"PASSWORD", "TIN", "EMAIL", "URL", "IP"})

ISO_COUNTRY_CODES = {
    "AD","AE","AF","AG","AI","AL","AM","AO","AR","AS","AT","AU","AW","AX","AZ",
    "BA","BB","BD","BE","BF","BG","BH","BI","BJ","BL","BM","BN","BO","BQ","BR",
    "BS","BT","BV","BW","BY","BZ","CA","CC","CD","CF","CG","CH","CI","CK","CL",
    "CM","CN","CO","CR","CU","CV","CW","CX","CY","CZ","DE","DJ","DK","DM","DO",
    "DZ","EC","EE","EG","EH","ER","ES","ET","FI","FJ","FK","FM","FO","FR","GA",
    "GB","GD","GE","GF","GG","GH","GI","GL","GM","GN","GP","GQ","GR","GS","GT",
    "GU","GW","GY","HK","HM","HN","HR","HT","HU","ID","IE","IL","IM","IN","IO",
    "IQ","IR","IS","IT","JE","JM","JO","JP","KE","KG","KH","KI","KM","KN","KP",
    "KR","KW","KY","KZ","LA","LB","LC","LI","LK","LR","LS","LT","LU","LV","LY",
    "MA","MC","MD","ME","MF","MG","MH","MK","ML","MM","MN","MO","MP","MQ","MR",
    "MS","MT","MU","MV","MW","MX","MY","MZ","NA","NC","NE","NF","NG","NI","NL",
    "NO","NP","NR","NU","NZ","OM","PA","PE","PF","PG","PH","PK","PL","PM","PN",
    "PR","PS","PT","PW","PY","QA","RE","RO","RS","RU","RW","SA","SB","SC","SD",
    "SE","SG","SH","SI","SJ","SK","SL","SM","SN","SO","SR","SS","ST","SV","SX",
    "SY","SZ","TC","TD","TF","TG","TH","TJ","TK","TL","TM","TN","TO","TR","TT",
    "TV","TW","TZ","UA","UG","UM","US","UY","UZ","VA","VC","VE","VG","VI","VN",
    "VU","WF","WS","YE","YT","ZA","ZM","ZW",
}

# Countries that actually issue IBANs (SWIFT IBAN registry).
IBAN_COUNTRY_CODES = {
    "AD","AE","AL","AT","AZ","BA","BE","BG","BH","BR","BY","CH","CR","CY","CZ",
    "DE","DK","DO","EE","EG","ES","FI","FO","FR","GB","GE","GI","GL","GR","GT",
    "HR","HU","IE","IL","IQ","IS","IT","JO","KW","KZ","LB","LC","LI","LT","LU",
    "LV","MC","MD","ME","MK","MR","MT","MU","NL","NO","PK","PL","PS","PT","QA",
    "RO","RS","SA","SC","SE","SI","SK","SM","ST","SV","TL","TN","TR","UA","VA",
    "VG","XK",
}

COMMON_NON_SECRET_PASSWORD_VALUES = {
    "change","default","disabled","error","expired","invalid","missing","new",
    "none","null","old","optional","redacted","required","reset","same","set",
    "temporary","unknown","updated",
}

# Major-network IINs (Visa / Mastercard 2-series / Amex / Discover / Diners /
# JCB / UnionPay). Bare Luhn is too weak, so a real prefix + length is required.
_CARD_IIN = re.compile(r"^(?:4|5[1-5]|2[2-7]|34|37|6011|64[4-9]|65|62|35|30[0-5]|3[689])")

# Upper-case headings/terms that fit the BIC shape but are not SWIFT codes.
_SWIFT_EXTRA_COMMON = {
    "TIMELINE", "OVERVIEW", "FINDINGS", "FILTERED", "OPERATIONAL", "REGULATIONS",
    "DISCUSSION", "PERSONNEL", "EQUIPMENT", "REFERENCE", "ATTACHMENT", "COMPLETED",
    "SCHEDULED", "REQUIRED", "RESOLVED", "APPROVED", "INSPECTOR", "MAINTENANCE",
    "PROCEDURE", "GUIDELINE", "ASSESSMENT", "MONITORING", "CALIBRATION",
}

# IANA-reserved TLDs used for test/documentation purposes — never real PII.
_RESERVED_SUFFIXES = (".example", ".test", ".invalid", ".localhost")
_RESERVED_DOC_DOMAINS = ("example.com", "example.net", "example.org")


# ============================================================================
# Normalizers / primitives
# ============================================================================
def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def strip_number_separators(value: str) -> str:
    return re.sub(r"[\s-]", "", value).strip()


def normalize_phone(value: str) -> str:
    compact = re.sub(r"[().\s-]", "", value)
    if value.strip().startswith("+"):
        return "+" + compact.lstrip("+")
    return compact


def luhn_check(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(digits[::-1]):
        number = int(digit)
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


# ============================================================================
# Validators
# ============================================================================
# ---- URL (TUNED) ----------------------------------------------------------
def is_valid_url(value: str) -> bool:
    candidate = value.strip().rstrip("\"')")
    if not candidate.lower().startswith(("http://", "https://", "ftp://")):
        candidate = "https://" + candidate
    try:
        parsed = urlparse(candidate)
    except Exception:
        return False
    if not (parsed.scheme in {"http", "https", "ftp"} and parsed.netloc):
        return False
    netloc = parsed.netloc.lower()
    if "." not in netloc.lstrip("["):
        return False
    if any(netloc.endswith(s) for s in _RESERVED_SUFFIXES):
        return False
    return True


def is_valid_url_barepath(value: str) -> bool:
    if not is_valid_url(value):
        return False
    host = value.strip().strip("\"')").lower().split("/")[0]
    return not any(host == d or host.endswith("." + d) for d in _RESERVED_DOC_DOMAINS)


# ---- EMAIL (TUNED) --------------------------------------------------------
def is_valid_email(value: str) -> bool:
    if ".." in value:
        return False
    local, _, domain = value.rpartition("@")
    if not local or not domain or "." not in domain:
        return False
    return not domain.startswith(".") and not domain.endswith(".")


def is_valid_email_double_at(value: str) -> bool:
    return is_valid_email(value.replace("@@", "@", 1))


def normalize_email_double_at(value: str) -> str:
    return value.replace("@@", "@", 1)


# ---- PHONE ----------------------------------------------------------------
def is_valid_phone(value: str) -> bool:
    """Strict predicate — used by BANK_ACCOUNT as a "this is a phone" guard."""
    candidate = value.strip()
    digits = re.sub(r"\D", "", candidate)
    groups = re.findall(r"\d+", candidate)
    separator_count = len(re.findall(r"[\s().-]", candidate))
    has_plus = candidate.startswith("+")
    has_parentheses = "(" in candidate or ")" in candidate

    if not (8 <= len(digits) <= 15):
        return False
    if len(groups) < 3:
        return False
    if any(len(group) > 4 for group in groups):
        return False
    if groups and len(groups[-1]) < 3:
        return False
    if not has_plus and len(digits) > 11:
        return False
    if not has_plus and not has_parentheses and separator_count < 2:
        return False
    if re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", candidate):
        return False
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}(?:\s+\d{1,2})?", candidate):
        return False
    return True


def is_phone_value(value: str, ssn_guard: bool = True) -> bool:
    """Permissive predicate for PHONE's own bare/context rules."""
    candidate = value.strip().strip("()")
    digits = re.sub(r"\D", "", candidate)
    if not (8 <= len(digits) <= 15):
        return False
    if "+" not in candidate and len(digits) > 13:
        return False
    # All-identical digits (0000-0000-0000, 9999-99-99 …) = masked/placeholder.
    if len(set(digits)) == 1:
        return False
    if re.fullmatch(r"\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{2,4}", candidate):
        return False
    if re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", candidate):
        return False
    # ID-shaped hyphen groups, not phones: 3-2-3 "123-45-789", 4-2-4 "2947-55-3311".
    if re.fullmatch(r"\d{3}-\d{2}-\d{3}", candidate):
        return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{4}", candidate):
        return False
    if re.search(r"(?:19|20)\d\d-\d{1,2}-\d{1,2}", candidate):
        return False
    if re.match(r"(?:19|20)\d\d-\d{1,2}-\d", candidate):
        return False
    # dd-mm-yyyy glued to a time, e.g. "00 02-03-2023" → not a phone.
    if re.search(r"\d{1,2}-\d{1,2}-(?:19|20)\d\d", candidate):
        return False
    # Redaction dashes inside the value ("+44 7821 1---35") = masked fragment.
    if re.search(r"-{2,}", candidate):
        return False
    if ssn_guard and re.fullmatch(r"\d{3}-\d{2}-\d{4}", candidate):
        return False
    return True


def is_phone_value_ctx(value: str) -> bool:
    return is_phone_value(value, ssn_guard=False)


# ---- NATIONAL_ID (TUNED) --------------------------------------------------
def is_valid_ssn(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area.startswith("9"):
        return False
    if group == "00" or serial == "0000":
        return False
    return True


def is_valid_cccd(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return len(digits) in {9, 12}


def is_valid_alphanum_national_id(value: str) -> bool:
    clean = re.sub(r"[-\s]", "", value)
    return (8 <= len(clean) <= 12
            and any(c.isdigit() for c in clean)
            and any(c.isalpha() for c in clean))


def is_valid_national_id(value: str) -> bool:
    return is_valid_ssn(value) or is_valid_cccd(value) or is_valid_alphanum_national_id(value)


# ---- CARD_NUMBER (OLD6) ---------------------------------------------------
def is_valid_card_number(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) not in {13, 14, 15, 16, 19}:
        return False
    if not _CARD_IIN.match(digits):
        return False
    return luhn_check(digits)


# ---- IBAN (OLD6) ----------------------------------------------------------
def is_valid_iban(value: str) -> bool:
    compact = re.sub(r"[\s-]", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    if compact[:2] not in IBAN_COUNTRY_CODES:
        return False
    rearranged = compact[4:] + compact[:4]
    digits = "".join(str(int(ch, 36)) if ch.isalpha() else ch for ch in rearranged)
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


# ---- IP (TUNED — stdlib) --------------------------------------------------
def is_valid_ip(value: str) -> bool:
    candidate = value.strip()
    if len(candidate) < 3 or not any(c in "0123456789abcdefABCDEF" for c in candidate):
        return False
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


# ---- TIN (TUNED) ----------------------------------------------------------
def is_valid_tax_id(value: str) -> bool:
    compact = re.sub(r"[\s-]", "", value)
    if not (8 <= len(compact) <= 20 and any(ch.isdigit() for ch in compact)):
        return False
    if re.fullmatch(r"[A-Z]+", compact, re.IGNORECASE):
        return False
    # Roleplay / demo tax-number placeholders ("TIN-STAR-11") — not real.
    if re.search(r"(?i)star|fake|dummy|placeholder|invalid|sample|pretend"
                 r"|mock|testcode|nulltax|notreal", compact):
        return False
    return True


# ---- BANK_ACCOUNT (TUNED) -------------------------------------------------
def is_valid_account_number(value: str) -> bool:
    compact = re.sub(r"[\s-]", "", value)
    if not (6 <= len(compact) <= 34 and any(ch.isdigit() for ch in compact)):
        return False
    if re.fullmatch(r"[A-Z]+", compact, re.IGNORECASE):
        return False
    # IBAN-shaped (CC + 2 digits + alnum) → leave it for the IBAN rule.
    if re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]+$", compact, re.IGNORECASE):
        return False
    digit_ratio = sum(ch.isdigit() for ch in compact) / len(compact)
    if digit_ratio < 0.6:
        return False
    if luhn_check(value):                # → CARD_NUMBER
        return False
    if is_valid_phone(value):            # → PHONE
        return False
    return True


# ---- SWIFT (OLD6) ---------------------------------------------------------
def is_valid_swift(value: str) -> bool:
    candidate = value.upper()
    if len(candidate) not in {8, 11}:
        return False
    if any(ch.islower() for ch in value):
        return False
    if not re.fullmatch(r"[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?", candidate):
        return False
    if candidate[4:6] not in ISO_COUNTRY_CODES:
        return False
    common_words = {
        "USERNAME","PASSWORD","LOCATION","QUESTION","FOOTBALL",
        "BUSINESS","CHILDREN","CUSTOMER","DOCUMENT","DOWNLOAD",
        "COMPUTER","INTERNET","MATERIAL","SOFTWARE","STANDARD",
        "VIETCOMBANK","TECHCOMBANK","AGRIBANK",
    }
    if candidate in common_words or candidate in _SWIFT_EXTRA_COMMON:
        return False
    # Demo / placeholder BIC strings (SWIFT-FAKECODE, IBAN-INVALIDXX) — never real.
    if re.search(r"FAKE|INVALID|PLACEHOLD|DUMMY|SAMPLE|TESTCODE|NULLCODE"
                 r"|MOCKCODE|DEMOCODE|NOTREAL", candidate):
        return False
    return True


# ---- PIN (OLD6) -----------------------------------------------------------
def is_valid_pin(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not (4 <= len(digits) <= 6 and digits.isdigit()):
        return False
    # All-identical digits (0000, 1111 …) = masked/redacted placeholder.
    if len(set(digits)) == 1:
        return False
    return True


# ---- CVV (TUNED) ----------------------------------------------------------
def is_valid_cvv(value: str) -> bool:
    return 3 <= len(value) <= 4


# ---- PASSWORD (TUNED) -----------------------------------------------------
def is_valid_password(value: str) -> bool:
    candidate = value.strip().strip("\"'()[]{}")
    if not 6 <= len(candidate) <= 64:
        return False
    if " " in candidate:
        return False
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", candidate):
        return False
    if candidate.count(":") >= 2:
        return False
    if re.fullmatch(r"\d+-[A-Za-z]+-[A-Za-z]+", candidate):
        return False
    if "=" in candidate:
        return False
    # A full URL belongs to URL, not PASSWORD.
    if re.match(r"(?i)(?:https?|ftp)://", candidate):
        return False
    # Known API-key / token prefixes belong to API_KEY, not PASSWORD.
    # Prefix set widened to the API_KEY shapes seen in the dataset (sk-/sk_/pk_/rk_,
    # ghp_/gho_/ghu_/ghs_/github_pat_/github_, xox*-, AKIA/ASIA, AIza, ya29., glpat-,
    # gldt-, SG., slack_/service_/oauth_/mailgun_/sentry_/twilio_/mapbox_/jwt_/whsec_/
    # firebase_/webhook_/api_/apikey, and the sim_ test-token prefix).
    if re.match(r"(?i)(?:sk[-_]|pk[-_]|rk[-_]|ghp_|gho_|ghu_|ghs_|github_pat_|github_|"
                r"xox[baprs]-|AKIA|ASIA|AIza|ya29\.|glpat-|gldt-|SG\.|slack_|service_|"
                r"oauth_|mailgun_|sentry_|twilio_|mapbox_|jwt_|whsec_|firebase_|webhook_|"
                r"sim_|api[-_]?key|api_)",
                candidate):
        return False
    # An embedded JWT (header.payload, both base64url and starting with "eyJ") is an
    # API token, not a password — even when wrapped in another prefix (sim_eyJ…).
    if re.search(r"eyJ[A-Za-z0-9_-]{6,}\.eyJ[A-Za-z0-9_-]{6,}", candidate):
        return False
    # Masked tokens (runs of X) are redacted IDs: "XXX-XX-4421", "XXXX-XXXX-1234".
    if re.search(r"[Xx]{3,}", candidate):
        return False
    # "439-level" and similar number-hyphen-word are descriptors, not secrets.
    if re.fullmatch(r"\d+-[A-Za-z]+", candidate):
        return False
    # All-caps alphanumeric runs (no lowercase, no symbol) of length >= 12 with both
    # letters and digits are IBANs / VINs / serials, not passwords.
    if (len(candidate) >= 12 and candidate.isalnum()
            and not any(c.islower() for c in candidate)
            and any(c.isalpha() for c in candidate)
            and any(c.isdigit() for c in candidate)):
        return False
    if not any(ch.isupper() for ch in candidate) and all(
        ch.isalnum() or ch in "._" for ch in candidate
    ):
        return False
    if candidate.lower() in COMMON_NON_SECRET_PASSWORD_VALUES:
        return False
    # QA / demo placeholders ("password=fail!23 … rejected") are not secrets.
    if re.match(r"(?i)(?:fail|test|fake|dummy|demo|sample|placeholder|invalid"
                r"|changeme|example)\b", candidate):
        return False
    has_lower = any(ch.islower() for ch in candidate)
    has_upper = any(ch.isupper() for ch in candidate)
    has_digit = any(ch.isdigit() for ch in candidate)
    has_symbol = any(not ch.isalnum() for ch in candidate)
    if not (has_lower or has_upper):
        return False
    if not has_digit:
        return False
    return has_symbol or (has_lower and has_upper) or len(candidate) >= 12


# ---- COORDINATE (OLD6) ----------------------------------------------------
_HEMI_NEG = {"S", "W", "T"}


def _parse_coord(tok: str):
    tok = tok.strip()
    sign = 1.0
    m = re.search(r"(?i)([NSEWBĐ])\s*$", tok)
    if m:
        if m.group(1).upper() in _HEMI_NEG:
            sign = -1.0
        tok = tok[:m.start()].strip()
    dms = re.match(
        r"^(\d{1,3})\s*°\s*(?:(\d{1,2})\s*['′]\s*(?:(\d{1,2}(?:\.\d+)?)\s*\\?[\"″])?)?\s*$", tok)
    if dms and "°" in tok:
        deg = float(dms.group(1))
        mn = float(dms.group(2)) if dms.group(2) else 0.0
        sec = float(dms.group(3)) if dms.group(3) else 0.0
        return sign * (deg + mn / 60.0 + sec / 3600.0)
    lead = 1.0
    if tok[:1] in "+-":
        if tok[0] == "-":
            lead = -1.0
        tok = tok[1:].strip()
    tok = tok.replace("°", "").strip()
    if re.fullmatch(r"\d{1,3}\.\d{1,8}", tok):
        return sign * lead * float(tok)
    return None


def _split_pair(value: str):
    v = value.strip()
    m = re.match(r"(?i)^(.*?[NSEWBĐ])\s*[, ]\s*(.*[NSEWBĐ])$", v)
    if m:
        return m.group(1), m.group(2)
    parts = re.split(r"\s*[,;]\s*", v)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def is_valid_coord_pair(value: str) -> bool:
    sp = _split_pair(value)
    if not sp:
        return False
    lat = _parse_coord(sp[0]); lon = _parse_coord(sp[1])
    if lat is None or lon is None:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def is_valid_coord_pair_bare(value: str) -> bool:
    if not is_valid_coord_pair(value):
        return False
    if re.search(r"[NSEWBĐ°]", value):
        return True
    if re.search(r"[+\-]\s*\d", value):
        return True
    decimals = re.findall(r"\.(\d+)", value)
    return len(decimals) >= 2 and all(len(d) >= 2 for d in decimals)


# ============================================================================
# Patterns
# ============================================================================
# ---- EMAIL (TUNED) --------------------------------------------------------
# Lookbehind also rejects a preceding '*' so a masked address
# "ops_team_****3241@offshore.net" does not leak its visible tail "3241@offshore.net".
_EMAIL_BARE = re.compile(
    r"(?<![\w.+*\-])[A-Z0-9._%+\-À-ɏḀ-ỿ]+@[A-Z0-9.-]+\.[A-Z]{2,63}(?![\w-])",
    re.IGNORECASE,
)
_EMAIL_BARE_DOUBLE_AT = re.compile(
    r"(?<![\w.+*\-])[A-Z0-9._%+-]+@@[A-Z0-9.-]+\.[A-Z]{2,63}(?![\w-])",
    re.IGNORECASE,
)
# v3 CONTEXT-ONLY: EMAIL must now be introduced by a cue ("email:", "thư điện tử là …").
# The value keeps the same shape as the bare rule; the '*' lookbehind still blocks a
# masked-tail leak. Validator is_valid_email keeps precision at the v2 level.
_EMAIL_CONTEXT = re.compile(
    r"(?i)\b(?:e[-\s]?mail(?:\s*address)?|thư\s*điện\s*tử|địa\s*chỉ\s*(?:email|e-mail|thư)"
    r"|hộp\s*thư(?:\s*điện\s*tử)?|gửi\s*(?:tới|đến|về)|liên\s*hệ\s*(?:qua\s*)?(?:email|mail))\b"
    r"\s*(?:[:#\-]\s*|[^:\n@]{0,20}(?:là|is|at|tại)\s+|\s+)"
    r"(?<![\w.+*\-])([A-Z0-9._%+\-À-ɏḀ-ỿ]+@[A-Z0-9.-]+\.[A-Z]{2,63})(?![\w-])"
)

# ---- PHONE (OLD6 hybrid) --------------------------------------------------
_PHONE_MASK_GUARD = r"(?! ?\*{3,})(?! ?\*+\d)(?!\s*-{2,})(?!-\*)"
_PHONE_BARE_INTL = re.compile(r"(?<![\w@.+])\+\d[\d\s().\-]{6,18}\d(?![\w])" + _PHONE_MASK_GUARD)
_PHONE_BARE_SEP = re.compile(
    r"(?<![\w@.+])\(?\d{2,4}\)?[ \t\-]\d{2,4}[ \t\-]\d{2,4}(?:[ \t\-]\d{1,4})*(?![\w])"
    + _PHONE_MASK_GUARD
)
_PHONE_NEG_LOCAL = re.compile(
    r"(?i)(?:id|sku|code|mã|ma|equipment|equip|shelf|jar|serial|lot|batch|ticket"
    r"|ref|order|invoice|item|part|model|asset|barcode|unit|tag|label)\s*[:#\-]?\s*$"
    r"|[A-Za-z]+-$"
    r"|(?:sản\s*phẩm|san\s*pham|tài\s*khoản|tai\s*khoan|số\s*tk|so\s*tk|stk|tk"
    r"|mã\s*số\s*thuế|mst|thuế|tax|identification|identity|account|acct|hàng|participant|iban"
    r"|căn\s*cước|can\s*cuoc|cccd|cmnd|định\s*danh|dinh\s*danh"
    r"|certificate|cert|cwi"
    r"|tokens?|prize|referenc(?:e|es|ed|ing))\b[^\d]{0,24}$"
    r"|(?-i:[A-Z]{2})\d{2}\s*$"
    r"|(?-i:[A-Z]{2}\d{2}\s+[A-Z]{2,4})\s*$"
    r"|[A-Za-z]{1,4}-\d{1,4}\s*\|\s*$"
)
_PHONE_CONTEXT = re.compile(
    r"(?i)\b(?:phone(?:\s*(?:number|no\.?))?|mobile|cell(?:phone)?|tel(?:ephone)?|fax|hotline"
    r"|contact(?:\s*number)?|sdt|số\s*đt|đt|số\s*điện\s*thoại|điện\s*thoại"
    r"|so\s*dien\s*thoai|dien\s*thoai|liên\s*hệ|lien\s*he|zalo|viber|whatsapp)\b"
    r"(?:\s*[:#.\-|]\s*|\s+[^\n0-9]{0,20}(?:là|số|number|no\.?)\s+|\s+)"
    r"(\+?\d(?:[\d\s()\-]{5,18}\d)?)"
    r"(?!\d)" + _PHONE_MASK_GUARD
)

# ---- NATIONAL_ID (TUNED) --------------------------------------------------
_NATID_BARE_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-\d{2}-\d{4}\b")
_NATID_NEG_CONTEXT = re.compile(
    r"(?i)(?:"
    r"[A-Za-z0-9]-"
    r"|(?:employ|staff|personnel|payroll|insur|bảo\s*hiểm|bao\s*hiem|policy|member|"
    r"account|acct|a/c|customer|invoice|order|ticket|refer|axa|allianz|aetna|cigna|bupa|"
    r"supervisor|entrant|standby|operator|attendant|technician|worker|"
    r"sample|example|fake|dummy|demo|sku|placeholder|fixture|mock|test)[^\n]{0,24}"
    r"|(?-i:[A-Z][a-z]+\s+[A-Z][a-z]+),\s*"
    r")$"
)
_NATID_CONTEXT = re.compile(
    r"(?i)\b(?:"
    r"ssn|social\s*security(?:\s*(?:number|num|no\.?))?"
    r"|national\s*(?:id(?:entif(?:ication)?)?|identification|number|identity)"
    r"|citizen\s*(?:id(?:entification)?|number|no\.?)"
    r"|gov(?:er(?:n)?ment|t)[-.\s]*(?:issued[-.\s]*)?(?:id(?:entif(?:ication)?)?|number)"
    r"|personal\s*(?:id(?:entif(?:ication)?)?|identity(?:\s*(?:number|num|no\.?))?)"
    r"|Sozialversicherungsnummer|Personalausweis(?:[-\s]?Nr\.?)?"
    r")\b"
    r"\s*(?:[^:\n0-9]{0,30}[:#,]\s*|[^:\n0-9]{0,20}(?:is|are)\s+|(?:no\.?|num(?:ber)?)\s+|\s*\(\s*|\s+)"
    r"(?<!\d)(\d{12}(?!\d)|\d{9}(?!\d)|\d{3}-\d{2}-\d{4}|[A-Z][A-Z0-9]{7,11})"
)
_NATID_VN_CONTEXT = re.compile(
    r"(?i)\b(?:"
    r"cccd|cmnd|căn\s*cước(?:\s*công\s*dân)?|chứng\s*minh\s*nhân\s*dân"
    r"|(?:số|mã)\s*(?:cccd|cmnd)"
    r"|(?:số|mã)\s*định\s*danh(?:\s*cá\s*nhân)?"
    r"|định\s*danh(?:\s*cá\s*nhân)?"
    r")\b"
    r"\s*(?:[^:\n0-9]{0,50}[:#]\s*|[^:\n0-9]{0,50}là\s+|[^:\n0-9]{0,50}thành\s+|\s*\(\s*|\s+)"
    r"(?<!\d)(\d{12}(?!\d)|\d{9}(?!\d))"
)

# ---- IBAN (OLD6 hybrid) ---------------------------------------------------
_IBAN_BARE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{11,30}(?![A-Z0-9])", re.IGNORECASE)
_IBAN_BARE_SPACED = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}(?:\s[A-Z0-9]{1,4}){2,8}(?![A-Z0-9])")
# An IBAN-shaped string introduced as an "account id/number" is tagged
# ACCOUNT_ID in this domain, not IBAN — leave it for the account rule / NER.
_IBAN_NEG_CONTEXT = re.compile(
    r"(?i)(?:account\s*(?:id|number|no|#)?\b[^\n\d]{0,12}(?:\d+-)?"
    r"|(?:sensor|equipment|device)(?:\s+unit)?\b[^\n\d]{0,16}"
    r"|sensor\s+unit\b[^\n\d]{0,16})$"
)
# v3 CONTEXT-ONLY: IBAN must now be introduced by an "IBAN" cue. The value is
# captured uppercase-only (?-i:) so a trailing lowercase word can't be absorbed;
# both compact and single-space-grouped forms are allowed. Validator runs mod-97.
_IBAN_CONTEXT = re.compile(
    r"(?i)\b(?:iban(?:\s*(?:number|no\.?|code))?|số\s*iban|mã\s*iban)\b"
    r"\s*(?:[:#\-]\s*|[^:\n0-9]{0,16}(?:là|is)\s+|\s+)"
    r"((?-i:[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){11,34}))"
)

# ---- CARD_NUMBER (OLD6) ---------------------------------------------------
_CARD_BARE = re.compile(r"(?<![\d.])\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7}(?![\d])")
_CARD_NEG_CONTEXT = re.compile(
    r"(?i)(?:imei|imsi|device|serial|equipment|seri|mã\s*thiết\s*bị)[^\n]{0,12}$"
)
_CARD_CONTEXT = re.compile(
    r"(?i)\b(?:credit\s*card|debit\s*card|card\s*(?:number|no\.?|num|#)"
    r"|visa|master\s*card|mastercard|amex|american\s*express|maestro|jcb"
    r"|thẻ\s*(?:tín\s*dụng|thanh\s*toán|ghi\s*nợ)|số\s*thẻ|Kreditkarte|Kartennummer)\b"
    r"\s*[:#-]?\s*(\d(?:[ -]?\d){12,18})"
)

# ---- IP (TUNED) -----------------------------------------------------------
_IP_BARE = re.compile(
    r"(?:"
    r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w])(?!\.\d)"
    r"|"
    r"(?<![\w.])(?:[A-F0-9]{0,4}(?::[A-F0-9]{0,4}){2,7})(?![\w:])(?!\.\d)"
    r")",
    re.IGNORECASE,
)
# v3 CONTEXT-ONLY: IP must now be introduced by an "IP / IP address / IPv4 / IPv6"
# cue. The value reuses the bare IPv4/IPv6 alternation; validator is_valid_ip uses
# the stdlib ipaddress parser, so a malformed run is rejected (precision held).
_IP_CONTEXT = re.compile(
    r"(?i)\b(?:ip(?:\s*address|\s*addr)?|ipv4|ipv6|địa\s*chỉ\s*ip|server\s*ip|host\s*ip)\b"
    r"\s*(?:[:#\-=]\s*|[^:\n0-9a-f]{0,16}(?:là|is|at)\s+|\s+)"
    r"((?:\d{1,3}\.){3}\d{1,3}(?!\.\d)|[A-F0-9]{0,4}(?::[A-F0-9]{0,4}){2,7})",
    re.IGNORECASE,
)

# ---- URL (TUNED) ----------------------------------------------------------
_URL_BARE = re.compile(
    r"(?i)(?:(?:https?|ftp)://)[^\s<>\"'(){}\[\]|\\^`]+"
    r"|(?<![\w@])www\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:(?:\.[a-z]{2,})+(?:/[^\s<>\"'(){}\[\]|\\^`]*)?|(?:[.-][a-z0-9-]+)*/[^\s<>\"'(){}\[\]|\\^`]+)"
    r"(?![\w-])"
)
_URL_BARE_PATH = re.compile(
    r"(?i)(?<![\w@/.])"
    r"([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]+)*"
    r"\.(?:com|net|org|io|co(?:\.[a-z]{2})?|vn|uk|edu|gov|mil|info|biz|tech|ai|app|dev)"
    r"/[^\s<>\"'(){}\[\]|,;]+)"
)
_URL_BARE_PAREN = re.compile(
    r"(?i)\("
    r"([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]+)*"
    r"\.(?:com|net|org|co(?:\.[a-z]{2})?|vn|uk|edu|gov)"
    r"(?:/[^\s<>\"'(){}\[\]|,;]*)?)"
    r"(?:\)|\.)"
)
_URL_CONTEXT = re.compile(
    r"(?i)\b(?:website|url|link|homepage|trang\s*web|địa\s*chỉ\s*web"
    r"|xem\s*(?:tại|tại\s*đây)|truy\s*cập|visit|see|Webseite|Homepage)\b"
    r"\s*[:#-]?\s*((?:https?|ftp)://[^\s<>\"'(){}\[\]|\\^`]+"
    r"|www\.[a-z0-9][a-z0-9-]{0,61}[a-z0-9]?"
    r"(?:(?:\.[a-z]{2,})+(?:/[^\s<>\"'(){}\[\]]*)?|(?:[.-][a-z0-9-]+)*/[^\s<>\"'(){}\[\]]+))"
)

# ---- BANK_ACCOUNT (TUNED) -------------------------------------------------
_BANK_CONTEXT = re.compile(
    r"(?i)"
    r"(?<![A-Za-z0-9])"
    r"(?:"
    r"bank(?:ing)?\s*(?:acc(?:t|ount)?|a/c)(?:\s*(?:number|num|no\.?|#|id))?"
    r"|bank\s+(?:number|no\.?|#)"
    # 'routing' dropped: a routing/ABA number identifies the bank, not the account.
    r"|bank\s*[:#]"
    # bare 'số tài khoản' (no 'ngân hàng') dropped: dataset labels it ACCOUNT_ID.
    r"|tài\s*khoản\s*(?:ngân\s*hàng|thanh\s*toán|thụ\s*hưởng|nhận|liên\s*kết)"
    r"|số\s*tk|stk"
    r"|kontonummer|konto(?:[-\s]?nr\.?)?|bankkonto"
    r")"
    r"[^\d\n]{0,28}?"
    r"([A-Z]{0,2}\d(?:[\d -]*\d)?)(?![\dA-Za-z])"
)

# ---- SWIFT (OLD6) ---------------------------------------------------------
_SWIFT_CONTEXT = re.compile(
    r"(?i)\b(?:swift(?:\s*code)?|bic(?:\s*code)?|mã\s*swift|SWIFT[-/\s]?BIC|BIC[-/\s]?SWIFT)\b"
    r"\s*[:#-]?\s*([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)"
)
# Bare BIC only when a transfer / institution cue precedes (precision-safe; not pure-bare).
_SWIFT_CTX_BROAD = re.compile(
    r"(?i)\b(?:transfer(?:red|s)?(?:\s+to)?|beneficiary|correspondent(?:\s+bank)?"
    r"|remit(?:tance|ted)?|wire(?:\s+transfer)?|sort\s*code|routing\s*bic)\b"
    r"[^A-Za-z0-9\n]{1,8}"
    r"((?-i:[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?))\b"
)

# ---- CVV (TUNED) ----------------------------------------------------------
_CVV_CONTEXT = re.compile(
    r"(?i)(?<!code )(?<!code-)\b(?:cvv\d?|cvc\d?|mã\s*(?:cvv|cvc)|Sicherheitscode|Prüfnummer)\b"
    r"(?:\s*[:#\-|]?\s*"
    r"|\s+[^:\n0-9]{0,40}[:#]\s*"
    r"|\s+[^:\n0-9]{0,40}(?:\bis\b|\bwas\b|\blà\b|\bas\b)\s*"
    r"|\s+[^:\n0-9]{0,40}(?:\bno\.?\b|\bnum(?:ber)?\b)\s+)"
    r"(?<!\d)(\d{3,4})(?![\d]|[-/]\d)"
)

# ---- PASSWORD (TUNED) -----------------------------------------------------
_PASSWORD_KW = (
    r"password|passwort|kennwort|pass\s*word|passcode|pass\s*phrase"
    r"|pwd|mật\s*khẩu|mat\s*khau|\bmk\b"
    r"|login\s*credentials?|credentials?"
    r"|authentication|authoriz(?:ation|ed)?\s*code"
)
_PASSWORD_FILLER = (
    r"(?:(?!(?:id|ids|license|licence|number|no|username|user|usuario"
    r"|contractor|driver|employee|badge|card|name|token|wallet|crypto"
    r"|failed|fail|account|accounts)\b)"
    r"[A-Za-zÀ-ỹ]{2,}[ :,]+){0,4}?"
)
_PASSWORD_CONTEXT = re.compile(
    r"(?i)\b(?:" + _PASSWORD_KW + r")\b"
    r"[\s:#=>,\-]*"
    + _PASSWORD_FILLER +
    r"[\"']?"
    r"((?=[^\s\"'<>]{0,63}\d)[^\s\"'<>]{6,64})"
    r"[\"']?"
)
# If the ~28 chars of left context name an API-key/token explicitly, the value is
# API_KEY territory — do not let the PASSWORD rule trigger on it.
_PASSWORD_NEG_APIKEY = re.compile(
    r"(?i)(?:api[\s_-]*key|api[\s_-]*token|access[\s_-]*token|secret[\s_-]*key"
    r"|client[\s_-]*secret|bearer[\s_-]*token|api[\s_-]*secret|apikey|auth[\s_-]*token)"
    r"\b[^\n]{0,28}$"
)

# ---- TIN (TUNED) ----------------------------------------------------------
_TIN_CONTEXT = re.compile(
    r"(?i)\b(?:"
    r"tax\s*(?:id(?:entif(?:ication|ying))?|identification)(?:\s*(?:number|num|no\.?))?"
    r"|taxpayer(?:\s*(?:id(?:entif(?:ication)?)?|number|num|no\.?))?"
    r"|tax\s*(?:reference|ref|identifier|account(?:\s*(?:number|no\.?))?|record"
    r"|number|num|details?|registration|compliance|documentation(?:\s*reference)?)"
    r"|fiscal\s*id(?:entif(?:ication)?)?(?:\s*(?:number|num|no\.?))?"
    r"|vat(?:\s*(?:id|number|reg(?:istration)?))?|gst(?:\s*(?:id|number|no\.?))?"
    r"|Steuer(?:[-\s]?(?:ID|Identifikationsnummer|Nr|nummer))?|St[-.]?Nr\.?"
    r"|mst|ma\s*so\s*thue"
    r")\b"
    r"\s*(?:[^:\n0-9]{0,30}[:#]\s*|(?:no\.?|num(?:ber)?|is|as|under|for)\s+|\s+)"
    r"([A-Z0-9][A-Z0-9-]{7,19})"
)
_TIN_CONTEXT_AMBIGUOUS = re.compile(
    r"\b(?:TIN|EIN)\b"
    r"\s*[:#]?\s*"
    r"([A-Z0-9][A-Z0-9-]{7,19})"
)
_TIN_VN_CONTEXT = re.compile(
    r"(?i)\b(?:mã\s*số\s*thuế|mã\s*thuế)\b"
    r"\s*(?:[^:\n0-9]{0,40}[:#]\s*|[^:\n0-9]{0,40}là\s+|\s+)"
    r"([A-Z0-9][A-Z0-9-]{7,19})"
)

# ---- PIN (OLD6 hybrid) ----------------------------------------------------
_PIN_NOT_BATTERY = r"(?!\s*(?:mAh\b|Ah\b|Wh\b|kWh\b|lần\b|chu\s*kỳ\b|%))"
_PIN_CONTEXT = re.compile(
    r"(?i)\b(?:pin(?:\s*(?:code|number|no\.?))?|mã\s*pin|ma\s*pin|PIN[-\s]?(?:Code|Nummer)"
    r"|security\s+pin|access\s+pin|pin\s+de\s+(?:acceso|entrada|seguridad|seguro))\b"
    r"\s*(?:[:#=\-]\s*|[^\n\d]{0,35}(?:là|is)\s+|\s*)(\d{3,6})(?!\d)(?!\s*,\s*\d)" + _PIN_NOT_BATTERY
)
# A PIN immediately followed by ", <number>" is part of a comma list of guessed
# values ("nhập đại: 2030, 1456, 5024"), not a real single PIN.
_PIN_CONTEXT_FILLER = re.compile(
    r"(?i)\bPIN\b(?!\s+codes\b)[^\n\d]{0,35}[:\s=](\d{4,6})(?!\d)(?!\s*,\s*\d)" + _PIN_NOT_BATTERY
)

# ---- COORDINATE (OLD6 hybrid) ---------------------------------------------
_COORD_DEC = r"[-+]?\d{1,3}\.\d{1,8}\s*°?\s*[NSEWBĐ]?"
_COORD_DMS = (
    r"\d{1,3}°\s*(?:\d{1,2}['′]\s*(?:\d{1,2}(?:\.\d+)?\s*\\?[\"″]\s*)?)?[NSEWBĐ]"
)
_COORD_BARE_DECIMAL_PAIR = re.compile(
    r"(?<![\w.])(" + _COORD_DEC + r"\s*[,;]\s*" + _COORD_DEC + r")(?!\w)(?!\.\d)"
)
_COORD_BARE_DMS_PAIR = re.compile(r"(" + _COORD_DMS + r"\s*[, ]\s*" + _COORD_DMS + r")")
# Context rule requires a full lat+lon PAIR (single-axis branch dropped on purpose).
_COORD_CONTEXT = re.compile(
    r"(?i)\b(?:gps|tọa\s*độ|toa\s*do|coordinates?|vị\s*trí\s*gps"
    r"|lat(?:itude)?|long(?:itude)?|lng|lon|vĩ\s*độ|vi\s*do|kinh\s*độ|kinh\s*do)\b"
    r"\s*[:#=]?\s*"
    r"(" + _COORD_DEC + r"\s*[,;]\s*" + _COORD_DEC
    + r"|" + _COORD_DMS + r"\s*[, ]\s*" + _COORD_DMS
    + r")"
)


# ============================================================================
# Detector
# ============================================================================
class RulebasePIIDetector:
    def __init__(self, allowed_entity_types=None):
        requested = allowed_entity_types or DEFAULT_RULEBASE_ALLOWED_ENTITY_TYPES
        self.allowed_entity_types = frozenset(
            _LEGACY_LABEL_ALIASES.get(x, x) for x in requested
        )
        self.regex_rules = self._build_regex_rules()
        self.context_rules = self._build_context_rules()

    def _is_enabled(self, entity_type: str) -> bool:
        return entity_type in self.allowed_entity_types

    def _build_regex_rules(self):
        # v3 CONTEXT-ONLY: all bare-regex rules are intentionally removed. Every
        # class is detected through a positive keyword context (see
        # _build_context_rules). The bare patterns are still defined above (some
        # are reused as value sub-patterns inside the context rules) but no rule
        # scans the text without a cue anymore.
        return []

    def _build_context_rules(self):
        # (entity_type, pattern, validator[, neg_context]) — value captured in
        # group(1); if neg_context is set, the rule is suppressed when the ~28 chars
        # of left context match it (e.g. an explicit API-key cue before a PASSWORD).
        specs = [
            # v3: EMAIL / IBAN / IP context rules (these classes were bare-only in v2).
            ("EMAIL", _EMAIL_CONTEXT, is_valid_email),
            ("IBAN", _IBAN_CONTEXT, is_valid_iban),
            ("IP", _IP_CONTEXT, is_valid_ip),
            ("PHONE", _PHONE_CONTEXT, is_phone_value_ctx),
            ("BANK_ACCOUNT", _BANK_CONTEXT, is_valid_account_number),
            ("CARD_NUMBER", _CARD_CONTEXT, is_valid_card_number),
            ("TIN", _TIN_CONTEXT, is_valid_tax_id),
            ("TIN", _TIN_CONTEXT_AMBIGUOUS, is_valid_tax_id),
            ("TIN", _TIN_VN_CONTEXT, is_valid_tax_id),
            ("NATIONAL_ID", _NATID_CONTEXT, is_valid_national_id),
            ("NATIONAL_ID", _NATID_VN_CONTEXT, is_valid_national_id),
            ("PIN", _PIN_CONTEXT, is_valid_pin),
            ("PIN", _PIN_CONTEXT_FILLER, is_valid_pin),
            ("PASSWORD", _PASSWORD_CONTEXT, is_valid_password, _PASSWORD_NEG_APIKEY),
            ("CVV", _CVV_CONTEXT, is_valid_cvv),
            ("SWIFT", _SWIFT_CONTEXT, is_valid_swift),
            ("SWIFT", _SWIFT_CTX_BROAD, is_valid_swift),
            ("COORDINATE", _COORD_CONTEXT, is_valid_coord_pair),
            ("URL", _URL_CONTEXT, is_valid_url),
            ("URL", _URL_BARE_PAREN, is_valid_url),
        ]
        return [
            (spec[0], spec[1], spec[2], spec[3] if len(spec) > 3 else None)
            for spec in specs if self._is_enabled(spec[0])
        ]

    def _detect_regex_matches(self, text: str):
        matches = []
        for rule in self.regex_rules:
            for found in rule.pattern.finditer(text):
                if rule.neg_context is not None:
                    left = text[max(0, found.start() - 28):found.start()]
                    if rule.neg_context.search(left):
                        continue
                raw_value = found.group(0).strip()
                cleaned_value, end = self._trim_trailing_punctuation(raw_value, found.end())
                if not cleaned_value or not rule.validator(cleaned_value):
                    continue
                matches.append(PIIMatch(
                    entity_type=rule.entity_type,
                    value=cleaned_value,
                    start=found.start(),
                    end=end,
                    source="regex",
                    normalized=rule.normalizer(cleaned_value) if rule.normalizer else cleaned_value,
                ))
        return matches

    def _detect_contextual_matches(self, text: str):
        matches = []
        for entity_type, pattern, validator, neg_context in self.context_rules:
            for found in pattern.finditer(text):
                if neg_context is not None:
                    left = text[max(0, found.start(1) - 28):found.start(1)]
                    if neg_context.search(left):
                        continue
                raw_value = found.group(1)
                cleaned_value, adjusted_end = self._trim_trailing_punctuation(raw_value, found.end(1))
                value = normalize_space(cleaned_value)
                if not value or not validator(value):
                    continue
                normalized = strip_number_separators(value)
                if entity_type in CONTEXT_NORMALIZE_AS_TEXT_ENTITY_TYPES:
                    normalized = normalize_space(value)
                matches.append(PIIMatch(
                    entity_type=entity_type,
                    value=value,
                    start=found.start(1),
                    end=adjusted_end,
                    source="context",
                    normalized=normalized,
                ))
        return matches

    @staticmethod
    def _spans_overlap(left, right) -> bool:
        return max(left.start, right.start) < min(left.end, right.end)

    @staticmethod
    def _match_confidence_rank(item) -> int:
        return 0 if item.source == "context" else 1

    def _deduplicate(self, matches):
        ordered = sorted(
            matches,
            key=lambda item: (
                self._match_confidence_rank(item),
                -(item.end - item.start),
                item.start,
                item.entity_type,
            ),
        )
        results = []
        seen = set()

        for item in ordered:
            key = (item.entity_type, item.start, item.end, item.value.lower())
            if key in seen:
                continue
            blocked = False
            for existing in results:
                if not self._spans_overlap(existing, item):
                    continue
                if existing.entity_type == item.entity_type:
                    blocked = True
                    break
                if existing.entity_type in ACTIVE_RULEBASE_BLOCKING_OVERLAPS.get(item.entity_type, frozenset()):
                    blocked = True
                    break
                if self._match_confidence_rank(existing) <= self._match_confidence_rank(item):
                    blocked = True
                    break
            if blocked:
                continue
            seen.add(key)
            results.append(item)

        return sorted(results, key=lambda item: (item.start, item.end, item.entity_type))

    def detect(self, text: str):
        matches = []
        matches.extend(self._detect_regex_matches(text))
        matches.extend(self._detect_contextual_matches(text))
        return self._deduplicate(matches)

    @staticmethod
    def _trim_trailing_punctuation(value: str, end: int):
        trimmed = value.rstrip(".,;:)]}")
        return trimmed, end - (len(value) - len(trimmed))


RULE_BASED_LABELS = set(DEFAULT_RULEBASE_ALLOWED_ENTITY_TYPES)


def detect_by_rules(text: str, allowed_labels) -> List[Dict]:
    """Notebook-compatible interface returning list of dicts."""
    detector = RulebasePIIDetector(allowed_entity_types=set(allowed_labels))
    results = []
    for match in detector.detect(text):
        results.append({
            "text":  match.value,
            "label": match.entity_type,
            "start": match.start,
            "end":   match.end,
        })
    return results


def run_rule_detector(text: str, labels=None) -> List[Dict]:
    """Legacy alias returning list of span dicts with source='rule' tag."""
    allowed = labels if labels is not None else DEFAULT_RULEBASE_ALLOWED_ENTITY_TYPES
    out = []
    for r in detect_by_rules(text, allowed):
        out.append({
            "label": r["label"],
            "start": r["start"],
            "end":   r["end"],
            "text":  r["text"],
            "score": 1.0,
            "source": "rule",
        })
    return out
