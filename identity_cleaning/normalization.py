from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Iterable

import pandas as pd
from dateutil import parser


_SEPARATORS_RE = re.compile(r"[-_'`´’‘.,;:/\\|()\[\]{}]+")
_SPACES_RE = re.compile(r"\s+")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")


def is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""


def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(value: object) -> str:
    """Lowercase, remove accents, normalize punctuation and collapse spaces."""
    if is_missing(value):
        return ""
    text = strip_accents(str(value)).lower()
    text = _SEPARATORS_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return _SPACES_RE.sub(" ", text).strip()


def compact_text(value: object) -> str:
    return normalize_text(value).replace(" ", "")


def name_tokens(value: object) -> list[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []
    return sorted(set(normalized.split()))


def tokens_as_string(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


def parse_disappearance_date(value: object) -> str:
    """Return an ISO date string when parsing is possible, otherwise empty."""
    if is_missing(value):
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip()
    if not text:
        return ""

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=not bool(_ISO_DATE_RE.match(text)))
    if not pd.isna(parsed):
        return parsed.date().isoformat()

    try:
        return parser.parse(text, dayfirst=True, fuzzy=True).date().isoformat()
    except (ValueError, OverflowError, TypeError):
        return ""
