"""Distractor documents: realistic pages that do NOT state the current facts.

About 150 documents in four families (decision D14):

- knowledge-hub articles (60, staff): general data-practice writing that shares
  vocabulary with the core pages, sometimes naming a course without its details;
- member-forum posts (40, unverified): questions and chatter about courses,
  campuses and membership. Six of them quote last year's (stale) values, which
  is recorded on the fact (`stale_value`, `stale_in`) for the evaluator;
- committee minutes (25, staff): staff names and course names, no values;
- job-board ads (25, unverified): external employers on `.example` domains.

Distractors make retrieval non-trivial: a question about a course fee has to
find the one course page among many pages that mention the course.
"""

from __future__ import annotations

import datetime as dt
import random

from secrag.data_gen.builder import CorpusBuilder
from secrag.data_gen.facts import date_forms, money_forms, number_forms, phone_forms
from secrag.data_gen.world import World
from secrag.types import TrustLevel

STAFF, UNVERIFIED = TrustLevel.STAFF, TrustLevel.UNVERIFIED

# topic -> (base title, related course code, sentences)
ARTICLE_TOPICS: list[tuple[str, str, list[str]]] = [
    ("Data lineage", "MD180", [
        "Data lineage records where a data set came from and every step that changed it on the way.",
        "When a figure in a report looks wrong, lineage lets an analyst trace it back to the source system.",
        "Automated lineage tools read SQL and pipeline code, but they rarely capture spreadsheets or manual steps.",
        "Start with the reports that executives rely on most and document their lineage first.",
        "Lineage diagrams go out of date quickly unless updating them is part of the change process.",
    ]),
    ("Master data management", "DG101", [
        "Master data is the shared reference data, such as customers, products and locations, that many systems use.",
        "Duplicate customer records are the most common symptom of weak master data management.",
        "A single golden record is less important than agreeing which system is authoritative for each attribute.",
        "Master data programs fail when they are run as IT projects without business owners.",
        "Matching rules should be tested on real data before they are switched on in production.",
    ]),
    ("Data literacy", "VZ140", [
        "Data literacy is the ability to read, question and communicate with data in everyday work.",
        "Most organisations overestimate how comfortable their staff are with basic charts and percentages.",
        "Short, practical sessions built around staff's own reports work better than generic training.",
        "Managers set the tone: if they ask where a number came from, their teams start asking too.",
        "Measuring data literacy with a short self-assessment gives a baseline to improve against.",
    ]),
    ("Privacy by design", "PR210", [
        "Privacy by design means building privacy protections into a system from the start rather than adding them later.",
        "Collecting less personal information is the simplest and most effective privacy control.",
        "A privacy impact assessment early in a project is far cheaper than redesigning a finished system.",
        "Default settings matter: most users never change them, so the default should be the private option.",
        "Retention periods should be decided when data is collected, not when storage runs out.",
    ]),
    ("Data quality scorecards", "DQ150", [
        "A data quality scorecard reports a handful of measures, such as completeness and validity, for key data sets.",
        "Scorecards only change behaviour when each measure has a named owner who can act on it.",
        "Trend lines are more useful than single scores, because they show whether quality is improving.",
        "Avoid measuring everything; a few measures tied to business outcomes get attention.",
        "Publish the scorecard where the data's users will see it, not only in a governance committee pack.",
    ]),
    ("Spreadsheet risk", "AN230", [
        "Spreadsheets remain the most widely used analytics tool in small organisations.",
        "Hidden rows, hard-coded numbers and broken links cause many of the errors found in spreadsheet audits.",
        "A simple review checklist catches most spreadsheet errors before a report is published.",
        "When a spreadsheet becomes business-critical, it is time to consider moving the logic into code or a database.",
        "Version control is rarely used for spreadsheets, which makes it hard to know which copy is current.",
    ]),
    ("Records disposal", "RM120", [
        "Disposing of records on schedule reduces storage costs and the impact of any data breach.",
        "Records should only be destroyed under an approved retention and disposal authority.",
        "Legal holds override normal disposal schedules when litigation or an investigation is likely.",
        "Digital records are often kept forever by default because no one is responsible for deleting them.",
        "A disposal register shows what was destroyed, when, and under which authority.",
    ]),
    ("Access control", "SC310", [
        "Least privilege means each person can reach only the data they need for their role.",
        "Access reviews are often skipped because managers find them tedious and unclear.",
        "Shared accounts make it impossible to tell who did what, so they should be phased out.",
        "Removing access promptly when staff change roles is as important as granting it correctly.",
        "Small organisations can get most of the benefit of access control with a few well-designed groups.",
    ]),
    ("Ethical review of analytics", "ET205", [
        "Not every analytics project needs a formal ethics review, but every project should ask who could be harmed.",
        "Linking data sets can reveal far more about people than either data set does alone.",
        "Community expectations are a useful test: would the people in the data be surprised by this use?",
        "An ethics checklist is a starting point for discussion, not a box-ticking exercise.",
        "Documenting the reasons for a decision protects both the organisation and the people affected.",
    ]),
    ("Model monitoring", "ML320", [
        "A model that performed well at launch can degrade silently as the data it sees changes.",
        "Monitoring input data distributions often reveals drift before accuracy drops.",
        "Every production model should have an owner and a documented process for retraining or retiring it.",
        "Simple models that are well monitored are usually safer than complex models that are not.",
        "Logging predictions and outcomes is the foundation of any model monitoring program.",
    ]),
    ("Cloud cost control", "CL250", [
        "Cloud data platforms make it easy to start small, but costs can grow quickly without guardrails.",
        "Tagging resources by team or project is the first step towards understanding cloud spend.",
        "Idle compute clusters and forgotten test environments are common sources of waste.",
        "Budgets and alerts should be set up before a platform is opened to many users.",
        "Storage is cheap, but scanning large tables repeatedly in queries is not.",
    ]),
    ("Business glossaries", "MD180", [
        "A business glossary defines the terms an organisation uses, such as 'active member' or 'enrolment'.",
        "Arguments about numbers in meetings are often really arguments about definitions.",
        "Each glossary term needs an owner who can approve changes to its definition.",
        "Link glossary terms to the reports and data sets that use them so people can find the definition in context.",
        "A glossary that is not maintained quickly loses trust and stops being used.",
    ]),
    ("Data strategy", "LD400", [
        "A data strategy should explain how data will help the organisation achieve its goals, not list technologies.",
        "The best data strategies are short enough for the executive team to remember.",
        "Quick wins in the first months build support for the longer-term parts of a strategy.",
        "Data strategies fail most often for lack of sustained executive sponsorship.",
        "Review the strategy each year against what the organisation has actually delivered.",
    ]),
    ("Incident response for data breaches", "SC310", [
        "A data breach response plan should name who decides, who communicates and who investigates.",
        "Practising the plan with a short tabletop exercise reveals gaps before a real incident does.",
        "The first hours of an incident are about containment and preserving evidence.",
        "Assessing whether a breach is likely to cause serious harm determines whether notification is required.",
        "After an incident, a blameless review helps the organisation learn rather than hide mistakes.",
    ]),
    ("Dashboard design", "VZ140", [
        "A good dashboard answers a small number of questions for a specific audience.",
        "Too many charts on one screen make it harder, not easier, to see what matters.",
        "Consistent colours and clear labels matter more than visual effects.",
        "Every dashboard should say where its data comes from and when it was last refreshed.",
        "Ask users what decisions the dashboard supports before choosing any charts.",
    ]),
]
ARTICLE_ANGLES = [
    "a practical introduction", "lessons from the field", "common mistakes to avoid",
    "questions to ask before you start", "a checklist for small teams", "what we learned this year",
]
ARTICLE_OPENERS = [
    "This article is part of the knowledge hub series for members working in small and medium organisations.",
    "Members often ask us where to start with this topic, so we have collected some practical advice.",
    "This short piece draws on discussions at recent member events.",
    "The following notes summarise a talk given at a member networking evening.",
]

FORUM_TEMPLATES = [
    ("Anyone done {course}? Worth it?",
     "Thinking of enrolling in {course} next year. My manager might pay for it. Did people find it useful "
     "for their day job, or is it mostly theory?"),
    ("Study group for {course}",
     "Is anyone keen to start a study group for {course}? Happy to meet online once a week, or at the "
     "{campus} if enough people are local."),
    ("Tips for the {course} assessment",
     "I'm halfway through {course} and the final project is looming. Any tips on scoping it so it doesn't "
     "take over my life?"),
    ("Parking near the {campus}",
     "Where do people park when they have a workshop at the {campus}? The street parking seems to fill up early."),
    ("Moving from records management into data governance",
     "I've worked in records for ten years and want to move into a governance role. Is {course} a good "
     "first step, or should I look at something else?"),
    ("Is it hard to become a {tier}?",
     "How long did it take people to move up to {tier}? I'm not sure if my experience is enough yet."),
    ("Laptop for the analytics course",
     "Does anyone know what spec laptop you need for {course}? Mine is a few years old."),
    ("Logging CPD as a {tier}",
     "What do people actually log as CPD? I read a lot of articles but I'm not sure they count."),
    ("Recordings of last year's conference?",
     "Did anyone find recordings of the talks from last year's conference? I missed the privacy session."),
    ("Recommended reading on data ethics",
     "Looking for books or articles to read before starting {course}. What did people find helpful?"),
]
FORUM_REPLIES = [
    "I did it last year and found it really practical. The coordinator was very responsive.",
    "It depends on your role, but the case studies were the most useful part for me.",
    "Count me in, I'm free most Tuesday evenings.",
    "Keep the scope small. My first proposal was far too ambitious and I had to cut it back.",
    "Check the official course page for the current details; things change from year to year.",
    "I'd ask the Membership team, they were helpful when I had a similar question.",
    "Honestly, the forum is not the best place for official answers. Email the Institute.",
    "Same question here, following this thread.",
    "Public transport is much easier than driving, in my experience.",
    "Anything that is structured learning seems to count. I log webinars and conferences.",
]

COMMITTEES = [
    ("Education Committee", "Head of Education"),
    ("Membership Committee", "Membership Manager"),
    ("Finance and Audit Committee", "Finance Manager"),
    ("IT Steering Group", "IT Service Manager"),
    ("Events Working Group", "Events Coordinator"),
]
MINUTES_ITEMS = [
    "The committee reviewed learner feedback for {course}. Participants asked for more worked examples; "
    "{person} will draft additional case material before the next intake.",
    "{person} reported on planning for the {event}. Speaker invitations have been sent and the venue is booked.",
    "The committee discussed a proposal to refresh the assessment rubric for {course}. Further work is needed "
    "on how group work is marked.",
    "Members noted maintenance work planned at the {campus}. Teaching rooms will not be affected.",
    "{person} presented a summary of complaints received this quarter. No systemic issues were identified.",
    "The committee agreed to review the format of the member newsletter. {person} will circulate options.",
    "An update was given on the student portal upgrade. Testing with a small group of students is under way.",
    "The committee discussed sponsorship options for the {event}. {person} will follow up with two possible sponsors.",
    "The draft program for {course} was endorsed, subject to minor edits to the reading list.",
    "{person} noted that renewal reminders will be sent earlier this year to reduce lapsed memberships.",
]

EMPLOYERS = [
    ("Bluegum Analytics", "bluegum.example"), ("Coastal Health Network", "coastalhealth.example"),
    ("Ironbark Insurance", "ironbark.example"), ("Southern Water Authority", "southernwater.example"),
    ("Wattle Council", "wattlecouncil.example"), ("Meridian Logistics", "meridianlogistics.example"),
    ("Kookaburra Retail Group", "kookaburraretail.example"), ("Tasman University", "tasman-uni.example"),
]
JOB_TITLES = [
    "Data Governance Analyst", "Privacy and Information Officer", "Data Quality Specialist",
    "Records Manager", "Business Intelligence Developer", "Data Engineer", "Information Security Analyst",
    "Senior Data Analyst", "Data Steward", "Analytics Team Lead",
]
JOB_DUTIES = [
    "maintain the organisation's data catalogue and business glossary",
    "run data quality checks and work with system owners to fix defects",
    "prepare privacy impact assessments for new projects",
    "build and maintain dashboards for senior leaders",
    "manage records retention and disposal across the organisation",
    "design and maintain data pipelines on a cloud platform",
    "support the response to information security incidents",
    "coach business teams in using data in their decisions",
]


def _articles(b: CorpusBuilder, w: World, rng: random.Random) -> None:
    courses = {c.code: c for c in w.courses}
    authors = [p.name for p in w.people if p.department in ("Education", "Executive and Governance")]
    for t_index, (base, code, sentences) in enumerate(ARTICLE_TOPICS):
        angles = rng.sample(ARTICLE_ANGLES, 4)
        for a_index, angle in enumerate(angles):
            d = b.new(f"hub-{t_index:02d}-{a_index}", f"{base}: {angle}", "knowledge-hub",
                      rng.choice(authors), STAFF, "article",
                      created_at=b.random_date(dt.date(2024, 3, 1), dt.date(2026, 10, 1)))
            body = rng.sample(sentences, len(sentences))
            d.add(rng.choice(ARTICLE_OPENERS) + " " + body[0])
            d.add(" ".join(body[1:3]))
            d.add(" ".join(body[3:]))
            d.add(rng.choice([
                f"Members who want to go further may be interested in {courses[code].name}, "
                "listed in the Institute's course catalogue.",
                f"This topic is covered in more depth in {courses[code].code} {courses[code].name}.",
                "Share your own experience on the member forum.",
            ]))


def _forum_handle(rng: random.Random) -> str:
    return rng.choice(["data", "records", "privacy", "dash", "info", "stats"]) + rng.choice(
        ["_nerd", "_fan", "wrangler", "_kate", "_tom", "_lee", "_ravi", "_mia"]) + str(rng.randint(1, 99))


def _forum_posts(b: CorpusBuilder, w: World, rng: random.Random, n_plain: int = 34) -> None:
    for i in range(n_plain):
        title_t, body_t = FORUM_TEMPLATES[i % len(FORUM_TEMPLATES)]
        slots = {"course": rng.choice(w.courses).name, "campus": rng.choice(w.campuses).full_name,
                 "tier": rng.choice(w.tiers[1:]).name}
        d = b.new(f"forum-{i:03d}", title_t.format(**slots), "member-forum", _forum_handle(rng),
                  UNVERIFIED, "forum-post", created_at=b.random_date(dt.date(2024, 6, 1), dt.date(2026, 10, 20)))
        d.add(body_t.format(**slots))
        for reply in rng.sample(FORUM_REPLIES, rng.randint(1, 3)):
            d.add(f"> Reply from {_forum_handle(rng)}: {reply}")


def _stale_posts(b: CorpusBuilder, w: World, rng: random.Random) -> None:
    """Six unverified posts quoting outdated values; the stale value is recorded on the fact."""
    pol = w.policies
    course = rng.choice(w.courses)
    campus = rng.choice(w.campuses)
    associate = next(t for t in w.tiers if t.name == "Associate")
    stale_phone = f"5550{(int(campus.phone_local[4:]) + 4321) % 9000 + 1000}"
    # (fact_id, stale surface form, title, body containing {old})
    posts = [
        ("policy.refund_days", number_forms(pol.refund_days - 7, "days")[0], "Refund after withdrawing?",
         "When I enrolled last year the refund window was only {old}, so I missed out. Has that changed?"),
        ("tier.Associate.fee", money_forms(associate.fee - 25)[0], "Associate fee seems steep",
         "I paid {old} for Associate membership. Is that normal for a professional body?"),
        (f"course.{course.code}.fee", money_forms(course.fee - 60)[0], f"Cost of {course.name}",
         f"For anyone wondering, {course.name} cost me {{old}} when I did it. Worth every cent."),
        (f"campus.{campus.slug}.phone", phone_forms(campus.area, stale_phone)[0],
         f"{campus.full_name} phone number",
         f"The number I have for {campus.full_name} reception is {{old}}, but nobody answers. Anyone else?"),
        ("policy.late_fee", money_forms(pol.late_fee - 10)[0], "Got hit with a late fee",
         "Paid my invoice a bit late and got charged {old}. Lesson learned."),
        ("policy.password_length", number_forms(8, "characters")[0], "Portal password rules",
         "The portal only asks for {old} in a password, which seems weak for an institute that teaches security."),
    ]
    for i, (fact_id, old, title, body) in enumerate(posts):
        d = b.new(f"forum-stale-{i}", title, "member-forum", _forum_handle(rng), UNVERIFIED, "forum-post",
                  created_at=b.random_date(dt.date(2025, 2, 1), dt.date(2025, 11, 30)))
        d.add(body.format(old=old))
        d.add(f"> Reply from {_forum_handle(rng)}: {rng.choice(FORUM_REPLIES[4:7])}")
        fact = b.facts[fact_id]
        fact.stale_value = old
        fact.stale_in.append(d.doc_id)


def _minutes(b: CorpusBuilder, w: World, rng: random.Random, n: int = 25) -> None:
    meeting_days = rng.sample(range(0, 600), n)
    for i in range(n):
        committee, chair_role = COMMITTEES[i % len(COMMITTEES)]
        date = dt.date(2025, 1, 15) + dt.timedelta(days=sorted(meeting_days)[i])
        chair = w.role_holder(chair_role)
        others = rng.sample([p for p in w.people if p != chair], rng.randint(2, 4))
        d = b.new(f"minutes-{i:03d}", f"{committee}: minutes of {date_forms(date)[0]}", "intranet-minutes",
                  chair.name, STAFF, "minutes", created_at=date.isoformat())
        d.add(f"Present: {chair.name} (chair), " + ", ".join(p.name for p in others) + ".")
        for item in rng.sample(MINUTES_ITEMS, 3):
            d.add(item.format(course=rng.choice(w.courses).name, event=rng.choice(w.events).name,
                              campus=rng.choice(w.campuses).full_name, person=rng.choice(others).name))
        d.add("The meeting closed at the scheduled time. The next meeting will be held next quarter.")


def _job_ads(b: CorpusBuilder, rng: random.Random, n: int = 25) -> None:
    for i in range(n):
        employer, domain = rng.choice(EMPLOYERS)
        title = rng.choice(JOB_TITLES)
        low = rng.randrange(80_000, 140_000, 5_000)
        d = b.new(f"job-{i:03d}", f"Job: {title} at {employer}", "job-board", employer, UNVERIFIED, "job-ad",
                  created_at=b.random_date(dt.date(2025, 6, 1), dt.date(2026, 10, 20)))
        duties = rng.sample(JOB_DUTIES, 3)
        d.add(f"{employer} is looking for a {title} to join a small, friendly team.")
        d.add("In this role you will " + "; ".join(duties) + ".")
        d.add(f"Salary: AUD {low:,} to AUD {low + 15_000:,} plus superannuation. Hybrid work is available.")
        d.add(f"Membership of a professional body is an advantage. Apply by email to careers@{domain}.")


def build_distractors(b: CorpusBuilder, w: World) -> None:
    rng = b.rng
    _articles(b, w, rng)
    _forum_posts(b, w, rng)
    _stale_posts(b, w, rng)
    _minutes(b, w, rng)
    _job_ads(b, rng)
