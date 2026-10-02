"""Monkey-patch WhitespaceTokenSplitter to atomic-tokenize PII patterns.

Import this module (or call apply_tokenizer_patch()) before any GLiNER2
inference or training. The patch is global for the process lifetime.

WARNING: Do not modify _IMPROVED_PATTERN without understanding the full
tokenization pipeline. This pattern prevents URLs, emails, IP addresses,
and JWT-like tokens from being split across word boundaries, which breaks
PII span offsets.
"""

import re

from gliner2.processor import WhitespaceTokenSplitter

_IMPROVED_PATTERN = re.compile(
    r"""
    # Scheme / www URL
    (?:(?:https?|ftp)://|www\.)[^\s<>"'`{}|\\^]*[^\s<>"'`{}|\\^.,;:!?()\[\]]
    
    # Email address
    |[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}
    
    # Social/user handle
    |@[a-z0-9_](?:[a-z0-9_.-]{0,28}[a-z0-9_])?
    
        # IPv4 address. Allows sentence punctuation after the IP, but avoids
        # matching inside a longer dotted numeric sequence.
    |(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}(?=$|[^\w])(?!\.\d)
    
        # IPv6 address, including compressed forms and optional zone IDs.
    |(?:
        (?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}
        |[0-9a-f]{1,4}:(?:(?::[0-9a-f]{1,4}){1,6})
        |(?:[0-9a-f]{1,4}:){1,2}(?::[0-9a-f]{1,4}){1,5}
        |(?:[0-9a-f]{1,4}:){1,3}(?::[0-9a-f]{1,4}){1,4}
        |(?:[0-9a-f]{1,4}:){1,4}(?::[0-9a-f]{1,4}){1,3}
        |(?:[0-9a-f]{1,4}:){1,5}(?::[0-9a-f]{1,4}){1,2}
        |(?:[0-9a-f]{1,4}:){1,6}:[0-9a-f]{1,4}
        |:(?:(?::[0-9a-f]{1,4}){1,7}|:)
        |(?:[0-9a-f]{1,4}:){1,7}:
    )(?:%[0-9a-z_.-]+)?(?![0-9a-f:])
    
        # Strict JWT-like token
    |[a-z0-9_-]{10,}\.[a-z0-9_-]{10,}\.[a-z0-9_-]{10,}
    
        # Shorter dot-separated token-like string, e.g. sk-abc.def.ghi
    |(?=[a-z0-9_.-]*[-_\d])[a-z0-9_-]{2,}(?:\.[a-z0-9_-]{2,}){2,}
    
        # Bare domain / URL-like candidate. Validation belongs downstream; this
        # tokenizer only keeps plausible URL spans from being split into pieces.
    |(?<![\w@])
        (?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+
        (?:[a-z]{2,63}|xn--[a-z0-9-]{2,59})
        (?=$|/|:|[^\w-])
        (?::\d{2,5})?
        (?:/(?:[^\s<>"'`{}|\\^]*[^\s<>"'`{}|\\^.,;:!?()\[\]])?)?
    
        # Letter-containing word token: must have at least one letter or
        # underscore (the [^\W\d] anchor). Only underscores join tokens;
        # hyphens always split (except digit-digit handled below).
    |(?:\d*[^\W\d]\w*)(?:_\w+)*
    
        # Pure digit sequence: hyphens/underscores only join other digit groups.
        # "555-1234" stays atomic; "13-tuổi" → "13" / "-" / "tuổi" so the
        # numeric entity (AGE, DATE, …) is not fused with the surrounding word.
    |\d+(?:[-_]\d+)*
    
        # Fallback: any single non-space character.
    |\S
    """,
    re.VERBOSE | re.IGNORECASE,
)


def apply_tokenizer_patch() -> None:
    WhitespaceTokenSplitter._PATTERN = _IMPROVED_PATTERN
    print("Splitter patched.")
