"""QA set: questions generated from facts, plus unanswerable questions.

Each answerable question asks about exactly one fact; its gold documents are
the documents that state that fact (`Fact.stated_in`). Unanswerable questions
ask about things the corpus never mentions (a course, campus or role that does
not exist, or a detail that is never given) so we can measure whether the
assistant refuses instead of guessing.
"""

from __future__ import annotations

import random

from secrag.data_gen.facts import Fact
from secrag.data_gen.world import World, role_fact_id
from secrag.eval.qa import QAItem

UNANSWERABLE = [
    "What is the fee for the Quantum Data Engineering (QD500) course?",
    "What is the phone number of the Perth campus?",
    "How much does a parking permit cost at the City campus?",
    "When is the 2027 Annual Gala Dinner?",
    "What is the email address of the Head of Research?",
    "How many CPD hours must an Honorary Life Member complete each year?",
    "What is the Wi-Fi password for the Riverside campus?",
    "Who is the Institute's Chief Financial Officer?",
    "What is the fee to replace a lost student ID card?",
    "How many weeks does the Blockchain for Records Managers course run?",
    "What is the Institute's ABN?",
    "How much is a non-member ticket to the End-of-Year Fellows Dinner?",
    "What is the email address of the Institute's media team?",
    "How much is the scholarship for the Leadership in Data Strategy course?",
    "What time does the Harbour campus library close on Saturdays?",
]


def _templates(f: Fact, w: World) -> list[str]:
    """Question phrasings for a fact. `{s}` is the fact's subject."""
    kind = f.fact_id.split(".")[0]
    attr = f.fact_id.split(".")[-1]
    if kind == "course":
        code = f.fact_id.split(".")[1]
        name = next(c.name for c in w.courses if c.code == code)
        return {
            "fee": [f"How much does {name} cost?", f"What is the fee for {code}?",
                    f"What do I pay to enrol in {name}?"],
            "weeks": [f"How many weeks does {name} run for?", f"How long is the {code} course?"],
            "start": [f"When does the next intake of {name} start?", f"What is the start date for {code}?"],
            "coordinator_email": [f"What is the email address of the coordinator of {name}?",
                                  f"How do I email the course coordinator for {code}?"],
        }[attr]
    if kind == "tier":
        tier = f.fact_id.split(".")[1]
        return {
            "fee": [f"What is the annual fee for {tier} membership?", f"How much does it cost to be a {tier}?"],
            "cpd_hours": [f"How many CPD hours must a {tier} complete each year?",
                          f"What is the yearly CPD requirement for {tier} members?"],
        }[attr]
    if kind == "campus":
        return {
            "phone": [f"What is the phone number for the {f.subject}?", f"How do I call the {f.subject}?"],
            "opens": [f"What time does the {f.subject} open on weekdays?"],
        }[attr]
    if kind == "staff":
        return [f"What is {f.subject}'s email address?", f"How can I email {f.subject}?"]
    if kind == "role":
        return [f"Who is the {f.subject}?", f"Who holds the role of {f.subject} at the Institute?"]
    if kind == "event":
        return {
            "date": [f"When is the {f.subject}?", f"What date is the {f.subject} on?"],
            "price": [f"How much is a member ticket to the {f.subject}?"],
            "venue": [f"Where is the {f.subject} being held?"],
        }[attr]
    return {
        "policy.refund_days": ["How long after enrolling can I withdraw and still get a full refund?",
                               "What is the refund window for courses?"],
        "policy.refund_admin_fee": ["What administration fee is deducted from a course refund?"],
        "policy.late_fee": ["What is the fee for paying an invoice late?"],
        "policy.late_grace_days": ["How long is the grace period before a late payment fee is charged?"],
        "policy.complaint_business_days": ["How long does the Institute take to respond to a complaint?"],
        "policy.appeal_days": ["How long do I have to appeal an assessment result?"],
        "policy.extension_days": ["What is the longest extension a course coordinator can grant?"],
        "policy.retention_years": ["How long are student records kept?"],
        "policy.audit_percent": ["What percentage of members are selected for a CPD audit each year?"],
        "policy.password_length": ["What is the minimum password length for Institute accounts?"],
        "contact.enquiries_email": ["What email address should I use for general enquiries?"],
        "contact.privacy_email": ["What is the email address for privacy requests?"],
        "contact.servicedesk_email": ["How do I contact the IT service desk by email?"],
    }[f.fact_id]


def _selected_facts(facts: dict[str, Fact], w: World, rng: random.Random) -> list[Fact]:
    """Which facts get a question: all of most kinds, a sample of the more repetitive ones."""
    def ids(prefix: str, suffix: str = "") -> list[str]:
        return [k for k in facts if k.startswith(prefix) and k.endswith(suffix)]

    chosen: list[str] = []
    chosen += ids("course.", ".fee") + ids("course.", ".start")
    chosen += rng.sample(ids("course.", ".weeks"), 6) + rng.sample(ids("course.", ".coordinator_email"), 6)
    chosen += ids("tier.")
    chosen += ids("campus.")
    chosen += rng.sample(ids("staff."), 8)
    chosen += [role_fact_id(r) for r in w.unique_roles]
    chosen += ids("policy.") + ids("contact.")
    chosen += ids("event.", ".date") + ids("event.", ".price")
    chosen += rng.sample(ids("event.", ".venue"), 4)
    return [facts[k] for k in chosen]


def build_qa(facts: dict[str, Fact], w: World, rng: random.Random) -> list[QAItem]:
    items: list[QAItem] = []
    for f in _selected_facts(facts, w, rng):
        if not f.stated_in:
            raise ValueError(f"fact {f.fact_id} is asked about but no document states it")
        items.append(QAItem(
            qa_id="", question=rng.choice(_templates(f, w)), answerable=True,
            category=f.fact_id.split(".")[0] + "." + f.fact_id.split(".")[-1],
            answer=f.value, answer_kind=f.kind, unit=f.unit, aliases=tuple(f.aliases),
            fact_id=f.fact_id, gold_doc_ids=tuple(sorted(f.stated_in)),
        ))
    items += [QAItem(qa_id="", question=q, answerable=False, category="unanswerable") for q in UNANSWERABLE]
    rng.shuffle(items)
    return [QAItem.from_record({**it.to_record(), "qa_id": f"q{i:03d}"}) for i, it in enumerate(items, 1)]
