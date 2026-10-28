"""Facts: the ground truth that the synthetic corpus states and the QA set asks about.

Every value a question can ask about is a `Fact`. Documents never contain a
free-typed value: the corpus builder renders one of the fact's surface forms
(e.g. "AUD 1,285" or "$1,285.00"), so phrasing varies but the value is always
consistent (decision D14). `aliases` lists every surface form a correct answer
may use (decision D9); the matcher in `secrag.eval.matching` also normalises
numbers, money, dates and phone numbers, so aliases are a safety net rather
than the only way to match.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """English words for 0 <= n < 100, e.g. 21 -> 'twenty-one'."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if ones == 0 else "-" + _ONES[ones])
    raise ValueError(f"number_to_words only supports 0..99, got {n}")


@dataclass
class Fact:
    """One checkable value in the fictional world."""

    fact_id: str  # e.g. "course.DG101.fee"
    subject: str  # human-readable subject, e.g. "Foundations of Data Governance (DG101)"
    attribute: str  # e.g. "fee"
    kind: str  # money | date | email | phone | number | text
    value: str  # canonical form, e.g. "AUD 1,285", "2027-03-15", "21"
    unit: str | None = None  # for kind == "number", e.g. "days"
    aliases: list[str] = field(default_factory=list)  # accepted surface forms
    stated_in: list[str] = field(default_factory=list)  # doc_ids stating the current value
    stale_value: str | None = None  # an outdated value some unverified posts still quote
    stale_in: list[str] = field(default_factory=list)  # doc_ids quoting the stale value

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


# --- surface forms -----------------------------------------------------------


def money_forms(amount: int) -> list[str]:
    s = f"{amount:,}"
    return [f"AUD {s}", f"A${s}", f"${s}", f"${s}.00"]


def date_forms(d: dt.date) -> list[str]:
    dmy = f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    return [
        dmy,
        f"{WEEKDAYS[d.weekday()]} {dmy}",
        f"{d.day:02d}/{d.month:02d}/{d.year}",
        f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
    ]


def phone_forms(area: str, local: str) -> list[str]:
    """Australian landline, e.g. area '03', local '55501234'."""
    a, b = local[:4], local[4:]
    return [f"({area}) {a} {b}", f"{area} {a} {b}", f"+61 {area[1]} {a} {b}"]


def number_forms(n: int, unit: str) -> list[str]:
    if unit == "percent":
        forms = [f"{n}%", f"{n} per cent"]
        if n < 100:
            forms.append(f"{number_to_words(n)} per cent")
        return forms
    forms = [f"{n} {unit}"]
    if n < 100:
        forms.append(f"{number_to_words(n)} {unit}")
    return forms


def time_forms(hhmm: str, meridiem: str) -> list[str]:
    """e.g. ('8:30', 'am') -> ['8:30 am', '8:30am', '8.30 am', '8.30am']."""
    dotted = hhmm.replace(":", ".")
    forms = [f"{hhmm} {meridiem}", f"{hhmm}{meridiem}", f"{dotted} {meridiem}", f"{dotted}{meridiem}"]
    if hhmm.endswith(":00"):
        hour = hhmm.split(":")[0]
        forms += [f"{hour} {meridiem}", f"{hour}{meridiem}"]
    return forms


# --- constructors ------------------------------------------------------------


def money_fact(fact_id: str, subject: str, attribute: str, amount: int) -> Fact:
    forms = money_forms(amount)
    return Fact(fact_id, subject, attribute, "money", forms[0], aliases=forms)


def date_fact(fact_id: str, subject: str, attribute: str, d: dt.date) -> Fact:
    return Fact(fact_id, subject, attribute, "date", d.isoformat(), aliases=date_forms(d))


def phone_fact(fact_id: str, subject: str, attribute: str, area: str, local: str) -> Fact:
    forms = phone_forms(area, local)
    return Fact(fact_id, subject, attribute, "phone", forms[0], aliases=forms)


def number_fact(fact_id: str, subject: str, attribute: str, n: int, unit: str) -> Fact:
    return Fact(fact_id, subject, attribute, "number", str(n), unit=unit, aliases=number_forms(n, unit))


def email_fact(fact_id: str, subject: str, attribute: str, email: str) -> Fact:
    return Fact(fact_id, subject, attribute, "email", email, aliases=[email])


def text_fact(fact_id: str, subject: str, attribute: str, value: str, aliases: list[str] | None = None) -> Fact:
    return Fact(fact_id, subject, attribute, "text", value, aliases=aliases or [value])
