"""Does a piece of text contain a given answer? (decision D9)

Matching is kind-aware, so equivalent surface forms count as the same answer:

- money:  "AUD 450", "A$450", "$450.00", "450 dollars"  -> 450.00
- date:   "15 March 2027", "15/03/2027" (day first), "March 15, 2027", "2027-03-15"
- phone:  "(03) 5550 1234", "03 5550 1234", "+61 3 5550 1234" -> 0355501234
- email:  case-insensitive
- number: the number (digits or words) directly followed by its unit, e.g.
          "21 days", "twenty-one days", "8%", "8 per cent"
- text:   any alias as a whole-word, case-insensitive substring

Every kind also accepts a plain alias match. Kind-specific parsing never
matches a different value: "$4,500" does not match 450.
"""

from __future__ import annotations

import datetime as dt
import re
from decimal import Decimal

from secrag.data_gen.facts import MONTHS, number_to_words

_MONTH_LOOKUP = {m.lower(): i for i, m in enumerate(MONTHS, 1)} | {m[:3].lower(): i for i, m in enumerate(MONTHS, 1)}
_MONTH_RE = "|".join(sorted(_MONTH_LOOKUP, key=len, reverse=True))


def normalise(text: str) -> str:
    """Lower-case, unify quotes and dashes, collapse whitespace."""
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'").replace(" ", " ")
    text = re.sub("[‐-―]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


# --- money -------------------------------------------------------------------

_AMOUNT = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?!\d)(?!,\d)"
_MONEY_PREFIX = re.compile(r"(?<![a-z])(?:aud|au\$|a\$|\$)\s?" + _AMOUNT)
_MONEY_SUFFIX = re.compile(_AMOUNT + r"\s?(?:aud|dollars)\b")


def extract_money(text: str) -> set[Decimal]:
    text = normalise(text)
    values = set()
    for m in [*_MONEY_PREFIX.finditer(text), *_MONEY_SUFFIX.finditer(text)]:
        whole, cents = m.group(1).replace(",", ""), m.group(2) or "0"
        values.add(Decimal(whole) + Decimal(cents.ljust(2, "0")) / 100)
    return values


# --- dates -------------------------------------------------------------------

_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})\b")
_MDY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")
_NUMERIC = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")  # Australian day-first
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _date(y: str, m: int | str, d: str) -> dt.date | None:
    try:
        return dt.date(int(y), int(m), int(d))
    except ValueError:
        return None


def extract_dates(text: str) -> set[dt.date]:
    text = normalise(text)
    found = [_date(y, _MONTH_LOOKUP[mon], d) for d, mon, y in _DMY.findall(text)]
    found += [_date(y, _MONTH_LOOKUP[mon], d) for mon, d, y in _MDY.findall(text)]
    found += [_date(y, m, d) for d, m, y in _NUMERIC.findall(text)]
    found += [_date(y, m, d) for y, m, d in _ISO.findall(text)]
    return {d for d in found if d is not None}


# --- phones and emails -------------------------------------------------------

_PHONE = re.compile(r"(?:\+61\s?\d|\(0\d\)|\b0\d)[\s-]?\d{4}[\s-]?\d{4}\b")
_EMAIL = re.compile(r"[a-z0-9._%+'-]+@[a-z0-9.-]+\.[a-z]{2,}")


def phone_digits(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    return "0" + digits[2:] if digits.startswith("61") else digits


def extract_phones(text: str) -> set[str]:
    return {phone_digits(m) for m in _PHONE.findall(normalise(text))}


def extract_emails(text: str) -> set[str]:
    return {e.rstrip(".") for e in _EMAIL.findall(normalise(text))}


# --- numbers with units ------------------------------------------------------

_UNIT_PATTERNS = {
    "days": r"(?:calendar\s+)?days?",
    "business days": r"(?:business|working)\s+days?",
    "weeks": r"weeks?",
    "hours": r"hours?",
    "years": r"years?",
    "months": r"months?",
    "characters": r"characters?",
    "percent": r"(?:%|per\s?cent)",
}


def _number_pattern(n: int, unit: str) -> re.Pattern[str]:
    number = re.escape(str(n))
    if n < 100:
        number = f"(?:{number}|{number_to_words(n).replace('-', '[- ]')})"
    unit_re = _UNIT_PATTERNS.get(unit, re.escape(unit))
    return re.compile(rf"(?<![\w.,]){number}\s*-?\s*{unit_re}(?!\w)")


# --- public API --------------------------------------------------------------


def contains_alias(text: str, aliases: tuple[str, ...] | list[str]) -> bool:
    """Whole-word alias match; "AUD 120" does not match inside "AUD 120,000"."""
    norm = normalise(text)
    return any(re.search(rf"(?<!\w){re.escape(normalise(a))}(?!\w)(?![.,]\d)", norm) for a in aliases if a.strip())


def contains_answer(
    text: str,
    kind: str,
    value: str,
    unit: str | None = None,
    aliases: tuple[str, ...] | list[str] = (),
) -> bool:
    """True if `text` states the answer `value` (of the given kind) in any accepted form."""
    if kind == "email":  # whole addresses only: "a@b.example" must not match inside "a@b.example.com"
        return value.lower() in extract_emails(text)
    # A bare number ("8") is not an answer without its unit, so numbers match on aliases only.
    if contains_alias(text, list(aliases) if kind == "number" else [value, *aliases]):
        return True
    if kind == "money":
        return bool(extract_money(value) & extract_money(text))
    if kind == "date":
        return dt.date.fromisoformat(value) in extract_dates(text)
    if kind == "phone":
        return phone_digits(value) in extract_phones(text)
    if kind == "number":
        if unit is None:
            raise ValueError("number answers need a unit")
        return bool(_number_pattern(int(value), unit).search(normalise(text)))
    return False  # "text": aliases only
