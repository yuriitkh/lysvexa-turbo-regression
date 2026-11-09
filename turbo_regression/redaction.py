"""Keep Worker credentials and enrollment tokens out of logs, tracebacks and printed errors."""
from __future__ import annotations

import logging
import re

REDACTED = "<redacted>"

_PATTERNS = (
    # Credentials and enrollment tokens are recognisable by the prefixes the registry issues.
    (re.compile(r"lysvexa-(worker|enroll)-[A-Za-z0-9_-]{8,}"), r"lysvexa-\1-" + REDACTED),
    (re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1" + REDACTED),
    # A secret that was copied into JSON or key=value text under its usual field name.
    (re.compile(r"""(?i)(["']?(?:credential|enrollment_token|authorization)["']?\s*[:=]\s*)(["']?)[^"',\s}]{4,}\2"""),
     r'\1"' + REDACTED + '"'),
)
_SECRET_FIELDS = frozenset({"credential", "enrollment_token", "authorization"})


def redact(text: object) -> str:
    value = str(text)
    for pattern, replacement in _PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def redact_mapping(value: dict) -> dict:
    """A copy that is safe to print: secret fields are replaced, everything else is kept."""
    return {
        key: REDACTED if isinstance(key, str) and key.lower() in _SECRET_FIELDS and item is not None
        else redact_mapping(item) if isinstance(item, dict) else item
        for key, item in value.items()
    }


class RedactingFormatter(logging.Formatter):
    """Redacts the final line, including formatted exception tracebacks and stack text."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))
