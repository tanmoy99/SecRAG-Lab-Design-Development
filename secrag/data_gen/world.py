"""The fictional Northbridge Institute of Data Practice: its entities and their facts.

`build_world(rng)` draws every value (fees, dates, phone numbers, policy limits,
who holds which role) from a seeded `random.Random`, so the same seed always
gives the same world. `build_facts(world)` turns the world into the `Fact`
records that documents state and questions ask about.

The institute is fictional so that no model has prior knowledge of its facts:
a correct answer must come from retrieval. Emails use the reserved `.example`
domain and phone numbers use the 5550 xxxx range ACMA sets aside for fiction.
"""

from __future__ import annotations

import datetime as dt
import random
import re
import unicodedata
from dataclasses import dataclass

from secrag.data_gen.facts import (
    Fact,
    date_fact,
    email_fact,
    money_fact,
    number_fact,
    phone_fact,
    text_fact,
    time_forms,
)

ORG_NAME = "Northbridge Institute of Data Practice"
EMAIL_DOMAIN = "northbridge.example"

FIRST_NAMES = [
    "Priya", "Liam", "Mei", "Oliver", "Aisha", "Noah", "Chloe", "Arjun", "Grace", "Tomas",
    "Hannah", "Kenji", "Zara", "Lachlan", "Fatima", "Ethan", "Sofia", "Daniel", "Amara", "Jack",
    "Ngaio", "Ravi", "Isla", "Mateo", "Leila", "Callum", "Yuki", "Samira", "Hamish", "Ana",
]
LAST_NAMES = [
    "Raman", "O'Connor", "Nguyen", "Fitzgerald", "Rahman", "Kowalski", "Tran", "Patel", "Murray", "Silva",
    "Henderson", "Watanabe", "Ahmed", "McKenzie", "Haddad", "Brooks", "Rossi", "Kim", "Okafor", "Sullivan",
    "Wiremu", "Iyer", "Campbell", "Morales", "Karimi", "Douglas", "Sato", "Hosseini", "Fraser", "Costa",
]

# (role, department). Roles that appear once are "unique roles": questions may ask who holds them.
STAFF_ROLES = [
    ("Chief Executive Officer", "Executive and Governance"),
    ("Registrar", "Executive and Governance"),
    ("Privacy Officer", "Executive and Governance"),
    ("Head of Education", "Education"),
    ("Program Coordinator", "Education"),
    ("Program Coordinator", "Education"),
    ("Program Coordinator", "Education"),
    ("Program Coordinator", "Education"),
    ("Program Coordinator", "Education"),
    ("Membership Manager", "Membership"),
    ("Membership Services Officer", "Membership"),
    ("Finance Manager", "Finance"),
    ("IT Service Manager", "IT Services"),
    ("Events Coordinator", "Events"),
    ("Student Services Advisor", "Student Services"),
]

# (code, name, one-sentence description, topics)
COURSES = [
    ("DG101", "Foundations of Data Governance",
     "This course introduces the roles, policies and decision rights that let an organisation treat data as a managed asset.",
     ["data ownership and stewardship roles", "writing a data governance charter",
      "decision rights and escalation paths", "measuring governance maturity"]),
    ("PR210", "Privacy Law for Practitioners",
     "A practical course on the Australian Privacy Principles and how they apply to everyday data handling.",
     ["the Australian Privacy Principles", "notifiable data breaches",
      "privacy impact assessments", "cross-border disclosure of personal information"]),
    ("DQ150", "Data Quality Management",
     "Learn how to define, measure and improve the quality of the data your organisation relies on.",
     ["data quality dimensions", "profiling and validation rules",
      "root-cause analysis of data defects", "data quality scorecards"]),
    ("AN230", "Applied Analytics with Python",
     "A hands-on course in cleaning, analysing and reporting on data with Python and pandas.",
     ["working with pandas data frames", "exploratory data analysis",
      "reproducible notebooks", "communicating analytical results"]),
    ("RM120", "Records and Information Management",
     "Covers the full life cycle of records, from creation and classification to retention and disposal.",
     ["records classification schemes", "retention and disposal authorities",
      "digital recordkeeping systems", "legal holds and discovery"]),
    ("SC310", "Security for Data Professionals",
     "An introduction to information security controls for people who manage or analyse data.",
     ["threat modelling", "access control and least privilege",
      "encryption basics", "incident response"]),
    ("ET205", "Data Ethics in Practice",
     "Explores the ethical questions raised by collecting, linking and using data about people.",
     ["consent and transparency", "fairness and bias in data use",
      "ethical review processes", "community expectations and social licence"]),
    ("VZ140", "Data Visualisation and Storytelling",
     "Shows how to design charts and dashboards that help decision makers understand data quickly.",
     ["choosing the right chart", "colour and accessibility",
      "dashboard design", "presenting data to executives"]),
    ("ML320", "Machine Learning Operations",
     "Covers the practices needed to deploy, monitor and maintain machine learning models in production.",
     ["model versioning", "monitoring for data and model drift",
      "deployment pipelines", "model risk management"]),
    ("MD180", "Metadata and Data Catalogues",
     "Explains how metadata and a well-run data catalogue help people find and trust data.",
     ["business and technical metadata", "data lineage",
      "driving catalogue adoption", "business glossary management"]),
    ("CL250", "Cloud Data Platforms",
     "An overview of storing and processing organisational data on modern cloud platforms.",
     ["data lakes and warehouses", "cloud cost management",
      "the shared responsibility model for security", "migration planning"]),
    ("LD400", "Leadership in Data Strategy",
     "For senior practitioners who want to set and lead an organisation-wide data strategy.",
     ["aligning data strategy with business goals", "building and growing a data team",
      "writing investment cases", "leading organisational change"]),
]

# (name, fee range, CPD-hour options or None, eligibility)
TIERS = [
    ("Student", (45, 95), None,
     "open to anyone currently enrolled in a recognised tertiary course"),
    ("Associate", (180, 260), (8, 10, 12),
     "for people starting out in a data role with less than three years of relevant experience"),
    ("Member", (320, 460), (15, 18, 20),
     "for practitioners with at least three years of relevant professional experience"),
    ("Fellow", (520, 690), (25, 30, 35),
     "awarded by invitation to senior practitioners who have made a significant contribution to the profession"),
]

# (slug, name, state, address, area code)
CAMPUSES = [
    ("city", "City", "Victoria", "Level 3, 47 Kestrel Lane, Melbourne VIC 3000", "03"),
    ("riverside", "Riverside", "Queensland", "12 Paperbark Road, South Brisbane QLD 4101", "07"),
    ("harbour", "Harbour", "New South Wales", "Suite 5, 90 Saltbush Parade, Pyrmont NSW 2009", "02"),
]

# (slug, name, description, member ticket price range)
EVENTS = [
    ("conference-2027", "Northbridge Data Practice Conference 2027",
     "Two days of keynotes, case studies and workshops on data governance, privacy and analytics.", (195, 495)),
    ("privacy-breakfast", "Privacy Awareness Week Breakfast",
     "A breakfast briefing on recent developments in Australian privacy regulation.", (35, 85)),
    ("dq-masterclass", "Data Quality Masterclass",
     "A full-day practical masterclass on building data quality rules that stick.", (145, 295)),
    ("women-in-data", "Women in Data Leadership Forum",
     "Panel discussions and mentoring sessions with leaders from across the data profession.", (45, 120)),
    ("records-roundtable", "Records Management Roundtable",
     "An informal roundtable for records and information managers to share practice.", (25, 65)),
    ("ai-governance", "AI Governance Symposium",
     "Speakers from government and industry discuss how to govern the use of AI responsibly.", (95, 245)),
    ("welcome-evening", "New Members Welcome Evening",
     "An evening event for new members to meet staff, Fellows and each other.", (15, 40)),
    ("fellows-dinner", "End-of-Year Fellows Dinner",
     "The annual dinner at which new Fellows are formally admitted to the Institute.", (110, 180)),
]

GENERIC_EMAILS = {
    "enquiries": f"enquiries@{EMAIL_DOMAIN}",
    "privacy": f"privacy@{EMAIL_DOMAIN}",
    "servicedesk": f"servicedesk@{EMAIL_DOMAIN}",
}


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


@dataclass(frozen=True)
class Person:
    slug: str
    name: str
    role: str
    department: str
    email: str
    extension: str  # four-digit phone extension

    @property
    def first_name(self) -> str:
        return self.name.split()[0]


@dataclass(frozen=True)
class Campus:
    slug: str
    name: str  # "City", used as "City campus"
    state: str
    address: str
    area: str  # phone area code, e.g. "03"
    phone_local: str  # eight digits, e.g. "55501234"
    opens: tuple[str, str]  # e.g. ("8:30", "am")
    closes: str  # e.g. "5:00 pm"

    @property
    def full_name(self) -> str:
        return f"{self.name} campus"


@dataclass(frozen=True)
class Course:
    code: str
    name: str
    description: str
    topics: tuple[str, ...]
    fee: int
    weeks: int
    mode: str
    start: dt.date
    coordinator: Person

    @property
    def title(self) -> str:
        return f"{self.name} ({self.code})"


@dataclass(frozen=True)
class Tier:
    name: str
    fee: int
    cpd_hours: int | None
    eligibility: str


@dataclass(frozen=True)
class Event:
    slug: str
    name: str
    description: str
    date: dt.date
    campus: Campus
    price: int


@dataclass(frozen=True)
class Policies:
    refund_days: int
    refund_admin_fee: int
    late_fee: int
    late_grace_days: int
    complaint_business_days: int
    appeal_days: int
    extension_days: int
    retention_years: int
    audit_percent: int
    password_length: int


@dataclass(frozen=True)
class World:
    people: tuple[Person, ...]
    campuses: tuple[Campus, ...]
    courses: tuple[Course, ...]
    tiers: tuple[Tier, ...]
    events: tuple[Event, ...]
    policies: Policies

    def role_holder(self, role: str) -> Person:
        matches = [p for p in self.people if p.role == role]
        if len(matches) != 1:
            raise KeyError(f"role {role!r} is held by {len(matches)} people, not exactly one")
        return matches[0]

    @property
    def unique_roles(self) -> list[str]:
        roles = [p.role for p in self.people]
        return [r for r in dict.fromkeys(roles) if roles.count(r) == 1]

    @property
    def departments(self) -> list[str]:
        return list(dict.fromkeys(p.department for p in self.people))

    def campus(self, slug: str) -> Campus:
        return next(c for c in self.campuses if c.slug == slug)


class _Unique:
    """Draws values without repeats, so no two facts share a fee, a date or a limit."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: set = set()

    def pick(self, options: list) -> object:
        free = [o for o in options if o not in self.used]
        value = self.rng.choice(free)
        self.used.add(value)
        return value


def _fee_options(lo: int, hi: int) -> list[int]:
    """Prices in steps of 5, skipping round hundreds so values look like real fees."""
    return [v for v in range(lo, hi + 1, 5) if v % 100]


def _dates(start: dt.date, end: dt.date, weekdays: set[int]) -> list[dt.date]:
    days = (end - start).days + 1
    return [start + dt.timedelta(i) for i in range(days) if (start + dt.timedelta(i)).weekday() in weekdays]


def build_world(rng: random.Random) -> World:
    money, dates, numbers = _Unique(rng), _Unique(rng), _Unique(rng)

    firsts = rng.sample(FIRST_NAMES, len(STAFF_ROLES))
    lasts = rng.sample(LAST_NAMES, len(STAFF_ROLES))
    extensions = rng.sample(range(4100, 4900), len(STAFF_ROLES))
    people = []
    for (role, dept), first, last, ext in zip(STAFF_ROLES, firsts, lasts, extensions):
        name = f"{first} {last}"
        email_local = f"{slugify(first)}.{slugify(last).replace('-', '')}"
        people.append(Person(slugify(name), name, role, dept, f"{email_local}@{EMAIL_DOMAIN}", str(ext)))

    phone_suffixes = rng.sample(range(1000, 10000), len(CAMPUSES))
    campuses = [
        Campus(slug, name, state, address, area, f"5550{suffix}",
               opens=rng.choice([("8:00", "am"), ("8:30", "am"), ("9:00", "am")]),
               closes=rng.choice(["5:00 pm", "5:30 pm", "6:00 pm"]))
        for (slug, name, state, address, area), suffix in zip(CAMPUSES, phone_suffixes)
    ]

    coordinators = [p for p in people if p.role == "Program Coordinator"]
    rng.shuffle(coordinators)
    mondays_2027 = _dates(dt.date(2027, 2, 1), dt.date(2027, 8, 30), {0})
    courses = []
    for i, (code, name, desc, topics) in enumerate(COURSES):
        campus = rng.choice(campuses)
        mode = rng.choice([
            "online, with live webinars each week",
            f"on campus at the {campus.full_name}",
            f"blended: online study plus two full-day workshops at the {campus.full_name}",
        ])
        courses.append(Course(
            code, name, desc, tuple(topics),
            fee=money.pick(_fee_options(395, 2950)),
            weeks=rng.choice([6, 8, 10, 12, 16]),
            mode=mode,
            start=dates.pick(mondays_2027),
            coordinator=coordinators[i % len(coordinators)],
        ))

    tiers = [
        Tier(name, money.pick(_fee_options(*fee_range)), rng.choice(cpd) if cpd else None, elig)
        for name, fee_range, cpd, elig in TIERS
    ]

    event_days = _dates(dt.date(2027, 3, 1), dt.date(2027, 11, 30), {0, 1, 2, 3, 4})
    events = [
        Event(slug, name, desc, dates.pick(event_days), rng.choice(campuses), money.pick(_fee_options(*price_range)))
        for slug, name, desc, price_range in EVENTS
    ]

    # Values that share a unit ("days") are drawn from one pool so they never coincide.
    policies = Policies(
        refund_days=numbers.pick([14, 21, 28, 30]),
        refund_admin_fee=money.pick(_fee_options(45, 95)),
        late_fee=money.pick(_fee_options(30, 75)),
        late_grace_days=numbers.pick([5, 7, 10]),
        complaint_business_days=rng.choice([10, 15, 20]),
        appeal_days=numbers.pick([10, 14, 20]),
        extension_days=numbers.pick([3, 5, 7, 10]),
        retention_years=rng.choice([6, 7, 8]),
        audit_percent=rng.choice([5, 8, 10, 12]),
        password_length=rng.choice([12, 14, 16]),
    )
    return World(tuple(people), tuple(campuses), tuple(courses), tuple(tiers), tuple(events), policies)


def role_fact_id(role: str) -> str:
    return f"role.{slugify(role)}.holder"


def build_facts(w: World) -> dict[str, Fact]:
    """All facts of the world, keyed by fact_id, in a fixed order."""
    facts: list[Fact] = []
    for c in w.courses:
        facts += [
            money_fact(f"course.{c.code}.fee", c.title, "fee", c.fee),
            number_fact(f"course.{c.code}.weeks", c.title, "duration", c.weeks, "weeks"),
            date_fact(f"course.{c.code}.start", c.title, "start date", c.start),
            email_fact(f"course.{c.code}.coordinator_email", c.title, "coordinator email", c.coordinator.email),
        ]
    for t in w.tiers:
        facts.append(money_fact(f"tier.{t.name}.fee", f"{t.name} membership", "annual fee", t.fee))
        if t.cpd_hours is not None:
            facts.append(number_fact(f"tier.{t.name}.cpd_hours", f"{t.name} membership",
                                     "annual CPD hours", t.cpd_hours, "hours"))
    for cp in w.campuses:
        facts += [
            phone_fact(f"campus.{cp.slug}.phone", cp.full_name, "phone number", cp.area, cp.phone_local),
            text_fact(f"campus.{cp.slug}.opens", cp.full_name, "weekday opening time",
                      " ".join(cp.opens), aliases=time_forms(*cp.opens)),
        ]
    for p in w.people:
        facts.append(email_fact(f"staff.{p.slug}.email", p.name, "email", p.email))
    for role in w.unique_roles:
        facts.append(text_fact(role_fact_id(role), role, "holder", w.role_holder(role).name))
    pol = w.policies
    facts += [
        number_fact("policy.refund_days", "Refund and Cancellation Policy", "full-refund window",
                    pol.refund_days, "days"),
        money_fact("policy.refund_admin_fee", "Refund and Cancellation Policy", "administration fee",
                   pol.refund_admin_fee),
        money_fact("policy.late_fee", "Fees and Late Payment Policy", "late payment fee", pol.late_fee),
        number_fact("policy.late_grace_days", "Fees and Late Payment Policy", "grace period",
                    pol.late_grace_days, "days"),
        number_fact("policy.complaint_business_days", "Complaints Handling Procedure", "response time",
                    pol.complaint_business_days, "business days"),
        number_fact("policy.appeal_days", "Assessment Appeals Policy", "appeal window", pol.appeal_days, "days"),
        number_fact("policy.extension_days", "Assessment Extension Policy", "maximum extension",
                    pol.extension_days, "days"),
        number_fact("policy.retention_years", "Records Retention Schedule", "student record retention",
                    pol.retention_years, "years"),
        number_fact("policy.audit_percent", "Continuing Professional Development Policy", "audit sample",
                    pol.audit_percent, "percent"),
        number_fact("policy.password_length", "IT Acceptable Use Policy", "minimum password length",
                    pol.password_length, "characters"),
        email_fact("contact.enquiries_email", ORG_NAME, "general enquiries email", GENERIC_EMAILS["enquiries"]),
        email_fact("contact.privacy_email", ORG_NAME, "privacy enquiries email", GENERIC_EMAILS["privacy"]),
        email_fact("contact.servicedesk_email", ORG_NAME, "IT service desk email", GENERIC_EMAILS["servicedesk"]),
    ]
    for e in w.events:
        facts += [
            date_fact(f"event.{e.slug}.date", e.name, "date", e.date),
            money_fact(f"event.{e.slug}.price", e.name, "member ticket price", e.price),
            text_fact(f"event.{e.slug}.venue", e.name, "venue", e.campus.full_name),
        ]
    return {f.fact_id: f for f in facts}
