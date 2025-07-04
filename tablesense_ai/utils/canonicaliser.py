"""Canonicaliser for noisy QA answers — now returns *plain Python values*.

Numerical outputs are plain ``float``\ s or ``int``\ s instead of
``Decimal('1.20')``.  This keeps your logs tidy while still normalising all
formatting quirks (currency, commas, ranges, etc.).

The public signature is unchanged:
    >>> canonicalise("$1.20")
    ('number', 1.2)
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Tuple, Optional

__all__ = ["canonicalise"]

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(
    r"""^\s*
        (?P<hour>\d{1,2})
        :
        (?P<minute>\d{2})
        (?:\s*(?P<ampm>a\.?m\.?|p\.?m\.?|am|pm))?
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_COLON_DURATION_RE = re.compile(
    r"^\s*([+\-])?(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)\s*$"
)  # [+|-]H?:MM:SS(.sss)

_MINSEC_DURATION_RE = re.compile(r"^\s*([+\-])?(\d+):(\d{2}(?:\.\d+)?)\s*$")

_DURATION_TEXT_RE = re.compile(
    r"""^\s*
        (?:(?P<hours>\d+)\s*(?:h|hours?|hrs?)\s*)?
        (?:and\s+)?
        (?:(?P<minutes>\d+)\s*(?:m|minutes?|mins?)\s*)?
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PERCENT_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*%\s*$")

_NUMBER_UNIT_RE = re.compile(
    r"^\s*([-+]?)\s*([£$]?)(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?"  # main numeric part
    r"(?:\s*(thousand|million|billion|[kKmMbB]))?"                     # magnitude word / suffix
    r"[\s\w%°/().-]*$",                                               # trailing units we ignore
    re.IGNORECASE,
)

_HYPHEN_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

_BOOL_MAP = {
    "true": True,
    "false": False,
    "yes": True,
    "yes.": True,
    "y": True,
    "no": False,
    "no.": False,
    "n": False,
}

_MAGNITUDE = {
    None: 1,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "k": 1_000,
    "K": 1_000,
    "m": 1_000_000,
    "M": 1_000_000,
    "b": 1_000_000_000,
    "B": 1_000_000_000,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_quotes_punct(s: str) -> str:
    s = s.strip().strip("\"' “”")
    while s and s[-1] in ",." and (len(s) == 1 or not s[-2].isdigit()):
        s = s[:-1].rstrip()
    return s

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_time_of_day(s: str) -> Optional[int]:
    m = _TIME_RE.match(s)
    if not m:
        return None
    hour = int(m.group("hour"))
    minute = int(m.group("minute"))
    ampm = m.group("ampm")
    if ampm:
        ampm = ampm.lower().replace(".", "")
        if ampm.startswith("p") and hour < 12:
            hour += 12
        elif ampm.startswith("a") and hour == 12:
            hour = 0
    if 0 <= hour < 24 and 0 <= minute < 60:
        return hour * 60 + minute
    return None


def _parse_colon_duration(s: str) -> Optional[float]:
    """Parse +H:MM:SS(.s) or +M:SS(.s) into seconds."""
    # Try full H:MM:SS first
    m = _COLON_DURATION_RE.match(s)
    if m:
        sign, h, m_, sec = m.groups()
        total = (int(h) if h else 0) * 3600 + int(m_) * 60 + float(sec)
        if sign == "-":
            total = -total
        return total
    # Then try MM:SS
    m = _MINSEC_DURATION_RE.match(s)
    if m:
        sign, mins, sec = m.groups()
        total = int(mins) * 60 + float(sec)
        if sign == "-":
            total = -total
        return total
    return None


def _parse_text_duration(s: str) -> Optional[int]:
    m = _DURATION_TEXT_RE.match(s)
    if m:
        hours = int(m.group("hours")) if m.group("hours") else 0
        minutes = int(m.group("minutes")) if m.group("minutes") else 0
        if hours or minutes:
            return hours * 3600 + minutes * 60
    return None


def _parse_percentage(s: str) -> Optional[float]:
    m = _PERCENT_RE.match(s)
    if not m:
        return None
    return float(Decimal(m.group(1)) / 100)


def _parse_number_with_units(s: str) -> Optional[float]:
    m = _NUMBER_UNIT_RE.match(s)
    if not m:
        return None
    sign, cur, int_part, frac_part, mag_word = m.groups()
    num_str = int_part.replace(",", "")
    if frac_part:
        num_str += "." + frac_part
    try:
        val = float(Decimal(num_str))
    except InvalidOperation:
        return None
    val *= _MAGNITUDE.get(mag_word, 1)
    if sign == "-":
        val = -val
    return val

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def canonicalise(text: str | None) -> Tuple[str, Any]:
    """Canonicalise *text* into a (type, value) pair ready for exact‑match."""
    if text is None:
        return ("text", "")

    # Keep only right‑most side of equations the model might echo
    if "=" in text:
        text = text.split("=")[-1]

    s = _strip_quotes_punct(text)

    # Normalise simple numeric ranges ("1-9" → "1 to 9")
    if _HYPHEN_RANGE_RE.match(s):
        a, b = _HYPHEN_RANGE_RE.match(s).groups()
        s = f"{a} to {b}"

    low = s.lower()
    if low in _BOOL_MAP:
        return ("boolean", _BOOL_MAP[low])

    # Try %
    pct = _parse_percentage(s)
    if pct is not None:
        return ("percentage", pct)


    # Time of day  h:mm [am/pm]
    tod = _parse_time_of_day(s)
    if tod is not None:
        return ("time", tod)

    # Colon durations  HH:MM:SS or MM:SS
    dur = _parse_colon_duration(s)
    if dur is not None:
        return ("duration", dur)

    # Textual durations  X hours Y minutes
    dur2 = _parse_text_duration(s)
    if dur2 is not None:
        return ("duration", dur2)

    # Numbers (with £/$, magnitude words, trailing units)
    num = _parse_number_with_units(s)
    if num is not None:
        return ("number", num)

    # Fallback = plain text
    return ("text", low)
