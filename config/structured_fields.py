from typing import List, Dict, Any

# Per-field config for structured entity variant in evaluation
STRUCT_FIELD_CONFIG: List[Dict[str, Any]] = [
    {
        "name": "url",
        "threshold": 0.5,
        "pattern": r"https?://\S+|http://\S+",
        "mode": "partial",
    },
    {
        "name": "mac_address",
        "threshold": 0.5,
        "pattern": r"^([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$",
        "mode": "full",
    },
    {
        "name": "http_cookie",
        "threshold": 0.25,
        "pattern": r"\w+=.+;",
        "mode": "partial",
    },
    {
        "name": "coordinate",
        "threshold": 0.30,
        "pattern": r"\d+\.\d+\s*[NSns],?\s*\d+\.\d+\s*[EWew]",
        "mode": "partial",
    },
    {
        "name": "unique_id",
        "threshold": 0.9,
        "pattern": (
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
            r"|[0-9a-fA-F]{32}|[0-9a-fA-F]{40,}"
        ),
        "mode": "full",
    },
]

# Maps structured field name → PII label tag
STRUCT_FIELD_TO_TAG: Dict[str, str] = {
    "url":         "URL",
    "mac_address": "MAC_ADDRESS",
    "http_cookie": "HTTP_COOKIE",
    "coordinate":  "COORDINATE",
    "unique_id":   "UNIQUE_ID",
}
