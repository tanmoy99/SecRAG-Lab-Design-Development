"""Core documents: the official and staff pages that state the institute's facts.

Each `_xxx` function writes one kind of page. Phrasing is chosen from a few
variants per sentence so that pages of the same kind do not read identically,
which keeps retrieval from being solvable by template matching alone. Every
value comes from `CorpusBuilder.say`, so gold documents are tracked exactly.
"""

from __future__ import annotations

from collections.abc import Callable

from secrag.data_gen.builder import CorpusBuilder, DocDraft
from secrag.data_gen.world import ORG_NAME, Campus, Course, Event, Person, Tier, World, role_fact_id, slugify
from secrag.types import TrustLevel

OFFICIAL, STAFF = TrustLevel.OFFICIAL, TrustLevel.STAFF

ASSESSMENT = [
    "Assessment is based on two written assignments and a final project that applies the course content to "
    "a real or realistic workplace problem. There is no closed-book exam.",
    "Participants complete weekly online quizzes, one case-study report and a short recorded presentation. "
    "Each piece of assessment is marked against a published rubric.",
    "Assessment combines a reflective journal, a group case study and an individual workplace project. "
    "Feedback is returned within two weeks of each submission.",
]
ENROL = [
    "Enrol through the course catalogue on the Institute website. Places are limited and are offered in the "
    "order applications are received. Withdrawal and refund rules are set out in the Refund and Cancellation Policy.",
    "Applications open about eight weeks before each intake. Once your enrolment is confirmed you will receive "
    "access to the learning portal. If you need to withdraw, see the Refund and Cancellation Policy.",
    "To enrol, create an account on the student portal and select the intake you want. Employers can pay on "
    "behalf of staff by requesting an invoice. Refunds are governed by the Refund and Cancellation Policy.",
]
BIOS = {
    "Executive and Governance": [
        "has worked in public-sector governance and regulation for more than fifteen years and leads the "
        "Institute's work with government and industry partners.",
        "previously managed compliance programs in the financial services sector and now oversees the "
        "Institute's policies, registers and committee processes.",
    ],
    "Education": [
        "designs and runs courses in the Institute's catalogue, working with industry practitioners who "
        "teach as guest lecturers.",
        "has taught data management at two universities and brings a strong interest in practical, "
        "workplace-based assessment.",
        "joined the Institute from an analytics consultancy and focuses on making technical topics accessible "
        "to people from non-technical backgrounds.",
    ],
    "Membership": [
        "looks after member applications, renewals and upgrades between membership tiers.",
        "works with members on their continuing professional development records and runs the mentoring program.",
    ],
    "Finance": ["manages the Institute's budget, invoicing and payment arrangements for members and students."],
    "IT Services": ["runs the Institute's IT systems, including the student portal, email and the service desk."],
    "Events": ["plans the Institute's conferences, masterclasses and member networking events."],
    "Student Services": [
        "is the first point of contact for students who need help with enrolment, extensions, appeals or support services."
    ],
}


def _name(b: CorpusBuilder, d: DocDraft, w: World, p: Person) -> str:
    """A person's name; for a unique role this also states who holds the role."""
    if p.role in w.unique_roles:
        return b.say(d, role_fact_id(p.role))
    return p.name


# --- courses -----------------------------------------------------------------


def _course_page(b: CorpusBuilder, w: World, c: Course) -> None:
    d = b.new(f"course-{c.code.lower()}", c.title, "course-catalogue",
              w.role_holder("Head of Education").name, OFFICIAL, "course")
    d.add(c.description + " " + b.pick([
        "It is aimed at working professionals and assumes no prior formal training in the area.",
        "The course suits people who already work with data and want a structured grounding in the topic.",
        "Most participants study part-time alongside their job.",
    ]))
    d.add("## What you will learn", "\n".join(f"- {t[0].upper()}{t[1:]}" for t in c.topics))
    d.add("## Fees, dates and delivery")
    fee = b.say(d, f"course.{c.code}.fee")
    d.add(b.pick([
        f"The fee for {c.code} is {fee}. Members of the Institute receive priority enrolment but pay the same fee.",
        f"Enrolling in {c.name} costs {fee}, which covers all learning materials and assessment.",
        f"Course fee: {fee} (GST-free). Fees are payable in full at enrolment.",
    ]))
    start = b.say(d, f"course.{c.code}.start")
    weeks = b.say(d, f"course.{c.code}.weeks")
    d.add(b.pick([
        f"The next intake begins on {start} and runs for {weeks}.",
        f"Classes start on {start}. The course runs for {weeks}, with about six hours of study per week.",
        f"Duration: {weeks}. Next start date: {start}.",
    ]) + f" Delivery is {c.mode}.")
    d.add("## Course coordinator")
    coord = c.coordinator
    email = b.say(d, f"staff.{coord.slug}.email")
    b.mark(d, f"course.{c.code}.coordinator_email")
    d.add(b.pick([
        f"{c.code} is coordinated by {coord.name}. Questions about content or assessment can be sent to {email}.",
        f"Your course coordinator is {coord.name} ({email}), who can help with questions about the content, "
        "assessment or study load.",
        f"For questions about this course, contact the coordinator, {coord.name}, at {email}.",
    ]))
    d.add("## Assessment", b.pick(ASSESSMENT), "## How to enrol", b.pick(ENROL))


# --- membership --------------------------------------------------------------


def _membership_fees(b: CorpusBuilder, w: World) -> None:
    d = b.new("membership-fees-2027", "Membership fees for 2027", "policy-library",
              w.role_holder("Membership Manager").name, OFFICIAL, "fee-schedule")
    d.add("The following annual membership fees apply to memberships starting or renewing from 1 January 2027. "
          "Fees are GST-free and cover a twelve-month membership period.")
    rows = ["| Membership tier | Annual fee |", "|---|---|"]
    rows += [f"| {t.name} | {b.say(d, f'tier.{t.name}.fee')} |" for t in w.tiers]
    d.add("\n".join(rows))
    d.add("Members who join part-way through the year pay a pro-rata fee for the remainder of the membership "
          "year. Upgrades between tiers are charged at the difference between the two fees.",
          "Payment can be made by card through the member portal or by bank transfer on an invoice. "
          "Members who are experiencing financial hardship can ask the Membership team about a payment plan.")


def _tier_page(b: CorpusBuilder, w: World, t: Tier) -> None:
    d = b.new(f"membership-{t.name.lower()}", f"{t.name} membership", "website",
              w.role_holder("Membership Manager").name, OFFICIAL, "membership")
    d.add(f"{t.name} membership is {t.eligibility}.")
    fee = b.say(d, f"tier.{t.name}.fee")
    d.add(b.pick([
        f"The annual fee for {t.name} membership is {fee}.",
        f"{t.name} members pay {fee} per year.",
        f"Annual fee: {fee}.",
    ]))
    if t.cpd_hours is not None:
        hours = b.say(d, f"tier.{t.name}.cpd_hours")
        d.add(b.pick([
            f"To keep their membership current, {t.name} members must complete {hours} of continuing "
            "professional development (CPD) each membership year.",
            f"{t.name} members commit to {hours} of CPD per year and record their activities in the member portal.",
        ]))
    else:
        d.add("Student members are encouraged, but not required, to record continuing professional development.")
    d.add(b.pick([
        "Benefits include discounted event tickets, access to the members-only knowledge hub, the monthly "
        "newsletter and the member forum.",
        "Membership gives you access to the member forum and knowledge hub, member pricing for events, "
        "and the right to vote at the Annual General Meeting.",
    ]))
    d.add(f"Apply or renew through the member portal. The Membership team reviews applications within a few "
          f"working days.")


def _cpd_policy(b: CorpusBuilder, w: World) -> None:
    d = b.new("policy-cpd", "Continuing Professional Development Policy", "policy-library",
              w.role_holder("Membership Manager").name, OFFICIAL, "policy")
    d.add("## Purpose", "Continuing professional development (CPD) keeps members' knowledge current. This policy "
          "sets out how much CPD each membership tier must complete and how CPD records are checked.")
    d.add("## Annual requirement")
    lines = [f"- {t.name}: {b.say(d, f'tier.{t.name}.cpd_hours')} per membership year"
             for t in w.tiers if t.cpd_hours is not None]
    d.add("\n".join(lines))
    d.add("Student members have no minimum CPD requirement. Hours may come from courses, conferences, "
          "structured reading, mentoring or presenting at professional events.")
    audit = b.say(d, "policy.audit_percent")
    d.add("## Audit", b.pick([
        f"Each year the Institute audits a random sample of {audit} of members with a CPD requirement. "
        "Selected members must provide evidence of their recorded activities.",
        f"A random {audit} of members are selected for a CPD audit each year and asked to supply evidence.",
    ]))
    d.add("Members who do not meet the requirement are contacted by the Membership team and given a "
          "chance to make up the shortfall before their membership lapses.")


# --- people ------------------------------------------------------------------


def _directory(b: CorpusBuilder, w: World, dept: str) -> None:
    d = b.new(f"directory-{slugify(dept)}", f"{dept} team directory", "intranet",
              w.role_holder("Registrar").name, STAFF, "directory")
    d.add(f"This page lists the {dept} team at the {ORG_NAME} and how to reach them. "
          "Internal phone extensions can be dialled from any Institute phone.")
    for p in (p for p in w.people if p.department == dept):
        name = _name(b, d, w, p)
        email = b.say(d, f"staff.{p.slug}.email")
        line = f"**{name}**, {p.role}\nEmail: {email} · Extension {p.extension}"
        codes = [c.code for c in w.courses if c.coordinator == p]
        if codes:
            line += "\nCoordinates: " + ", ".join(codes)
        d.add(line)


def _profile(b: CorpusBuilder, w: World, p: Person) -> None:
    d = b.new(f"staff-{p.slug}", f"Staff profile: {p.name}", "website", p.name, STAFF, "profile")
    name = _name(b, d, w, p)
    bio = b.pick(BIOS[p.department])
    d.add(b.pick([
        f"{name} is the Institute's {p.role}. {p.first_name} {bio}",
        f"As {p.role}, {name} {bio}",
    ]))
    codes = [c.title for c in w.courses if c.coordinator == p]
    if codes:
        d.add(f"{p.first_name} currently coordinates " + "; ".join(codes) + ".")
    email = b.say(d, f"staff.{p.slug}.email")
    d.add(b.pick([
        f"You can contact {p.first_name} at {email}.",
        f"Email: {email}",
        f"To get in touch with {p.first_name}, email {email} or call the main switchboard and ask for "
        f"extension {p.extension}.",
    ]))


# --- policies ----------------------------------------------------------------


def _policies(b: CorpusBuilder, w: World) -> None:
    registrar = w.role_holder("Registrar").name

    d = b.new("policy-refunds", "Refund and Cancellation Policy", "policy-library", registrar, OFFICIAL, "policy")
    d.add("## Purpose", "This policy explains when course fees are refunded if a participant withdraws "
          "from a course, and what happens if the Institute cancels a course.")
    days, admin = b.say(d, "policy.refund_days"), b.say(d, "policy.refund_admin_fee")
    d.add("## Refund window", b.pick([
        f"Participants who withdraw within {days} of enrolling receive a full refund of the course fee, "
        f"less an administration fee of {admin}.",
        f"A full refund, minus a {admin} administration fee, is available if you withdraw no later than "
        f"{days} after your enrolment is confirmed.",
    ]))
    d.add("Withdrawals after the refund window has closed are not refunded, except in documented cases of "
          "serious illness, bereavement or other exceptional circumstances, which the Registrar assesses case by case.")
    d.add("## Course cancellations", "If the Institute cancels a course, all fees are refunded in full and no "
          "administration fee is charged. Participants may instead transfer to a later intake.")
    d.add("## How to apply", "Refund requests are made through the student portal. Approved refunds are paid "
          "to the original payment method.")

    d = b.new("policy-late-payment", "Fees and Late Payment Policy", "policy-library",
              w.role_holder("Finance Manager").name, OFFICIAL, "policy")
    d.add("This policy applies to course fees, membership fees and event invoices issued by the Institute.")
    fee, grace = b.say(d, "policy.late_fee"), b.say(d, "policy.late_grace_days")
    d.add("## Late payment", b.pick([
        f"Invoices are due on the date shown. If payment has not been received {grace} after the due date, "
        f"a late payment fee of {fee} is added to the account.",
        f"A grace period of {grace} applies after the due date. After that, a late fee of {fee} is charged.",
    ]))
    d.add("## Payment plans", "Students and members experiencing financial hardship can apply for a payment "
          "plan before the due date. No late fee is charged while an approved payment plan is being followed.")

    d = b.new("policy-complaints", "Complaints Handling Procedure", "policy-library", registrar, OFFICIAL, "policy")
    d.add("The Institute welcomes feedback and treats complaints as a chance to improve. Complaints can be made "
          "by students, members, event attendees or members of the public.")
    days = b.say(d, "policy.complaint_business_days")
    d.add("## Timeframes", b.pick([
        f"We acknowledge every complaint within two business days and aim to provide a written response "
        f"within {days}.",
        f"A written response to a complaint will be provided within {days} of it being received.",
    ]))
    d.add("## Escalation", "If you are not satisfied with the response, you can ask for the complaint to be "
          "reviewed by the Registrar. The Registrar's decision is the final internal step.")

    d = b.new("policy-appeals", "Assessment Appeals Policy", "policy-library", registrar, OFFICIAL, "policy")
    d.add("Students may appeal a final grade or an assessment decision if they believe it was not made in "
          "line with the published assessment rubric or Institute policy.")
    days = b.say(d, "policy.appeal_days")
    d.add("## Lodging an appeal", b.pick([
        f"Appeals must be lodged within {days} of the result being released, using the appeal form in the "
        "student portal.",
        f"You have {days} from the release of a result to lodge an appeal through the student portal.",
    ]))
    d.add("Appeals are reviewed by an academic who was not involved in the original marking. The outcome "
          "is sent to the student in writing.")

    d = b.new("policy-extensions", "Assessment Extension Policy", "policy-library",
              w.role_holder("Head of Education").name, OFFICIAL, "policy")
    d.add("Students may request more time to complete an assessment task if illness, caring responsibilities, "
          "work demands or other circumstances beyond their control affect their study.")
    days = b.say(d, "policy.extension_days")
    d.add(b.pick([
        f"Course coordinators may grant an extension of up to {days}. Longer extensions require approval from "
        "the Head of Education.",
        f"The maximum extension a course coordinator can approve is {days}.",
    ]))
    d.add("Requests should be made before the due date through the student portal, with supporting "
          "documents where available.")

    d = b.new("policy-records-retention", "Records Retention Schedule", "policy-library",
              registrar, OFFICIAL, "policy")
    d.add("This schedule sets out how long the Institute keeps different kinds of records before they are "
          "securely destroyed or archived.")
    years = b.say(d, "policy.retention_years")
    d.add("## Student records", b.pick([
        f"Student enrolment and assessment records are kept for {years} after the student's final enrolment.",
        f"Records of enrolments, results and appeals are retained for {years} after a student completes "
        "or leaves their course.",
    ]))
    d.add("## Other records", "Financial records are kept as required by taxation law. General correspondence "
          "is kept for as long as it is needed for business purposes and then destroyed.")

    d = b.new("policy-privacy", "Privacy Policy", "policy-library", w.role_holder("Privacy Officer").name,
              OFFICIAL, "policy")
    d.add("The Institute collects personal information about students, members, staff and event attendees to "
          "provide its services. We handle personal information in line with the Australian Privacy Principles.")
    officer = b.say(d, role_fact_id("Privacy Officer"))
    email = b.say(d, "contact.privacy_email")
    d.add("## Contacting us about privacy", b.pick([
        f"Questions, requests to access or correct your information, and privacy complaints can be sent to "
        f"our Privacy Officer, {officer}, at {email}.",
        f"Our Privacy Officer is {officer}. You can reach the privacy team at {email}.",
    ]))
    d.add("## Data breaches", "If a data breach is likely to result in serious harm, we will notify affected "
          "individuals and the regulator as required by law.")

    d = b.new("policy-it-acceptable-use", "IT Acceptable Use Policy", "policy-library",
              w.role_holder("IT Service Manager").name, OFFICIAL, "policy")
    d.add("This policy applies to everyone who uses the Institute's IT systems, including staff, students, "
          "contractors and volunteers.")
    length = b.say(d, "policy.password_length")
    d.add("## Passwords and sign-in", b.pick([
        f"Passwords must be at least {length} long. Multi-factor authentication is required for all accounts.",
        f"All accounts must use multi-factor authentication and a password of no fewer than {length}.",
    ]))
    email = b.say(d, "contact.servicedesk_email")
    d.add("## Reporting problems", f"Report lost devices, suspicious emails or suspected security incidents to "
          f"the IT service desk at {email} as soon as possible.")

    d = b.new("policy-code-of-conduct", "Code of Conduct", "policy-library", registrar, OFFICIAL, "policy")
    d.add("All members, students and staff are expected to act with integrity, respect and professionalism in "
          "their dealings with the Institute and with each other.")
    d.add("Members must not misrepresent their membership tier or qualifications, and must declare conflicts of "
          "interest when acting on Institute committees.")
    registrar_name = b.say(d, role_fact_id("Registrar"))
    d.add(f"Concerns about a possible breach of this Code should be raised with the Registrar, {registrar_name}, "
          "who will decide whether a formal investigation is needed.")


# --- campuses and contact ------------------------------------------------------


def _campus_page(b: CorpusBuilder, w: World, cp: Campus) -> None:
    d = b.new(f"campus-{cp.slug}", cp.full_name, "website", w.role_holder("Registrar").name, OFFICIAL, "campus")
    d.add(f"The {cp.full_name} is located at {cp.address}. It hosts classes, workshops and member events "
          f"in {cp.state}.")
    phone, opens = b.say(d, f"campus.{cp.slug}.phone"), b.say(d, f"campus.{cp.slug}.opens")
    d.add("## Contact and opening hours", b.pick([
        f"Reception can be reached on {phone}. The campus is open from {opens} to {cp.closes}, Monday to Friday.",
        f"Phone: {phone}. Weekday opening hours: {opens} to {cp.closes}. The campus is closed on weekends "
        "and public holidays.",
    ]))
    d.add("## Getting here", b.pick([
        "The campus is a short walk from the nearest train station, and several bus routes stop outside. "
        "Limited paid parking is available nearby; parking permits are not issued to students.",
        "Public transport is the easiest way to reach the campus. There is bicycle parking in the building and a "
        "commercial car park within two blocks.",
    ]))
    d.add("## Accessibility", "All teaching rooms are step-free and have hearing loops. Please contact "
          "Student Services before your visit if you have specific access requirements.")


def _contact_page(b: CorpusBuilder, w: World) -> None:
    d = b.new("contact-us", "Contact us", "website", w.role_holder("Registrar").name, OFFICIAL, "contact")
    d.add(f"General questions about courses, membership or events can be sent to "
          f"{b.say(d, 'contact.enquiries_email')}. We aim to reply within two business days.")
    lines = [f"- {cp.full_name}: {b.say(d, f'campus.{cp.slug}.phone')}" for cp in w.campuses]
    d.add("## Campus phone numbers", "\n".join(lines))
    d.add("## Specific enquiries",
          f"- Privacy and access to your personal information: {b.say(d, 'contact.privacy_email')}\n"
          f"- IT and student portal problems: {b.say(d, 'contact.servicedesk_email')}")


# --- events --------------------------------------------------------------------


def _event_page(b: CorpusBuilder, w: World, e: Event) -> None:
    coordinator = w.role_holder("Events Coordinator")
    d = b.new(f"event-{e.slug}", e.name, "events", coordinator.name, STAFF, "event")
    d.add(e.description)
    date, venue = b.say(d, f"event.{e.slug}.date"), b.say(d, f"event.{e.slug}.venue")
    price = b.say(d, f"event.{e.slug}.price")
    d.add(b.pick([
        f"The event will be held on {date} at the {venue}.",
        f"Date: {date}. Venue: {venue}.",
        f"Join us at the {venue} on {date}.",
    ]))
    d.add(b.pick([
        f"Member tickets cost {price}. Non-member pricing is shown on the booking page.",
        f"Tickets for members are {price}, including catering.",
    ]))
    email = b.say(d, f"staff.{coordinator.slug}.email")
    d.add(f"For questions about the program, accessibility or dietary requirements, contact "
          f"{coordinator.name} at {email}.")


def _events_calendar(b: CorpusBuilder, w: World) -> None:
    d = b.new("events-calendar-2027", "Events calendar 2027", "events",
              w.role_holder("Events Coordinator").name, STAFF, "event")
    d.add("Upcoming Institute events for 2027. Bookings open about two months before each event.")
    for e in sorted(w.events, key=lambda e: e.date):
        d.add(f"- {b.say(d, f'event.{e.slug}.date')}: {e.name}, {b.say(d, f'event.{e.slug}.venue')}")


# --- FAQs and news -------------------------------------------------------------


def _faqs(b: CorpusBuilder, w: World) -> None:
    registrar = w.role_holder("Registrar").name

    # Answers are functions of the draft, so `say` marks the FAQ page being written.
    def faq(doc_id: str, title: str, author: str, items: list[tuple[str, Callable[[DocDraft], str]]]) -> None:
        d = b.new(doc_id, title, "website", author, STAFF, "faq")
        for q, a in items:
            d.add(f"**{q}**\n{a(d)}")

    course_a = b.rng.choice(list(w.courses))
    faq("faq-enrolment", "Enrolment FAQ", registrar, [
        ("Can I get a refund if I withdraw?",
         lambda d: f"Yes, if you withdraw within {b.say(d, 'policy.refund_days')} of enrolling. "
                   f"An administration fee of {b.say(d, 'policy.refund_admin_fee')} applies."),
        ("What happens if I pay my invoice late?",
         lambda d: f"A late fee of {b.say(d, 'policy.late_fee')} is charged once the grace period has passed."),
        (f"When does {course_a.name} next start?",
         lambda d: f"The next intake of {course_a.code} starts on {b.say(d, f'course.{course_a.code}.start')}."),
        ("Can my employer pay?", lambda d: "Yes. Ask for an invoice during enrolment and we will send it to your employer."),
    ])
    faq("faq-membership", "Membership FAQ", w.role_holder("Membership Manager").name, [
        ("How much does membership cost?",
         lambda d: "; ".join(f"{t.name}: {b.say(d, f'tier.{t.name}.fee')}" for t in w.tiers) + " per year."),
        ("Can I upgrade my membership?",
         lambda d: "Yes. Upgrades are charged at the difference between the two annual fees."),
        ("How do I become a Fellow?",
         lambda d: "Fellowship is by invitation. Members can be nominated by two existing Fellows."),
    ])
    faq("faq-cpd", "CPD FAQ", w.role_holder("Membership Services Officer").name, [
        ("How many CPD hours do I need?",
         lambda d: " ".join(f"{t.name} members need {b.say(d, f'tier.{t.name}.cpd_hours')}."
                            for t in w.tiers if t.cpd_hours is not None)),
        ("Will my CPD be audited?",
         lambda d: f"Possibly. Each year {b.say(d, 'policy.audit_percent')} of members are randomly selected."),
        ("What counts as CPD?",
         lambda d: "Courses, conferences, structured reading, mentoring and presenting all count."),
    ])
    faq("faq-it", "IT and student portal FAQ", w.role_holder("IT Service Manager").name, [
        ("I forgot my password. What do I do?",
         lambda d: f"Use the reset link on the sign-in page, or email {b.say(d, 'contact.servicedesk_email')}."),
        ("What are the password rules?",
         lambda d: f"Passwords need at least {b.say(d, 'policy.password_length')}, and multi-factor "
                   "authentication must be switched on."),
        ("Can I use my own laptop?", lambda d: "Yes. The student portal works in any modern web browser."),
    ])
    faq("faq-student-support", "Student support FAQ", w.role_holder("Student Services Advisor").name, [
        ("Can I get an extension?",
         lambda d: f"Your course coordinator can approve an extension of up to {b.say(d, 'policy.extension_days')}."),
        ("How do I appeal a grade?",
         lambda d: f"Lodge an appeal through the student portal within {b.say(d, 'policy.appeal_days')} of the "
                   "result being released."),
        ("Who can I talk to about study problems?",
         lambda d: f"Contact our Student Services Advisor, "
                   f"{b.say(d, role_fact_id('Student Services Advisor'))}."),
    ])
    campus = b.rng.choice(list(w.campuses))
    faq("faq-contact", "Contact FAQ", registrar, [
        ("What is the best email for general questions?",
         lambda d: f"Email {b.say(d, 'contact.enquiries_email')}."),
        (f"What are the {campus.full_name} opening hours?",
         lambda d: f"From {b.say(d, f'campus.{campus.slug}.opens')} to {campus.closes} on weekdays."),
        (f"How do I call the {campus.full_name}?",
         lambda d: f"Call reception on {b.say(d, f'campus.{campus.slug}.phone')}."),
    ])
    faq("faq-privacy", "Privacy FAQ", w.role_holder("Privacy Officer").name, [
        ("How do I see the information you hold about me?",
         lambda d: f"Email {b.say(d, 'contact.privacy_email')} with your request."),
        ("How long do you keep student records?",
         lambda d: f"Student records are kept for {b.say(d, 'policy.retention_years')} after your final enrolment."),
        ("Who is responsible for privacy at the Institute?",
         lambda d: f"Our Privacy Officer, {b.say(d, role_fact_id('Privacy Officer'))}."),
    ])
    event = b.rng.choice(list(w.events))
    faq("faq-events", "Events FAQ", w.role_holder("Events Coordinator").name, [
        ("Do members get cheaper tickets?",
         lambda d: "Yes. Member pricing applies to every Institute event."),
        (f"When is the {event.name}?",
         lambda d: f"It is on {b.say(d, f'event.{event.slug}.date')} at the {b.say(d, f'event.{event.slug}.venue')}."),
        ("Can I get a refund on an event ticket?",
         lambda d: "Event tickets can be transferred to a colleague but are not refundable."),
    ])


def _news(b: CorpusBuilder, w: World) -> None:
    ceo = w.role_holder("Chief Executive Officer").name

    def news(doc_id: str, title: str, created_at: str) -> DocDraft:
        return b.new(doc_id, title, "newsletter", ceo, STAFF, "news", created_at=created_at)

    new_courses = b.rng.sample(list(w.courses), 3)
    d = news("news-2027-course-intakes", "Course intakes for 2027 now open", "2026-09-14")
    d.add("Applications are now open for our 2027 course intakes. Three courses are already in high demand:")
    d.add("\n".join(f"- {c.name} starts on {b.say(d, f'course.{c.code}.start')}" for c in new_courses))
    d.add("Early enrolment is recommended, as places are limited and offered in order of application.")

    hires = [p for p in w.people if p.department in ("IT Services", "Student Services")]
    d = news("news-welcome-new-staff", "Welcome to our new staff", "2026-07-02")
    d.add("We are pleased to welcome two new colleagues to the Institute this month.")
    for p in hires:
        d.add(f"{_name(b, d, w, p)} joins us as {p.role}. {p.first_name} {b.pick(BIOS[p.department])}")

    d = news("news-membership-fees-2027", "Membership fees confirmed for 2027", "2026-10-01")
    member = next(t for t in w.tiers if t.name == "Member")
    d.add(f"The Board has confirmed membership fees for 2027. The {member.name} fee will be "
          f"{b.say(d, f'tier.{member.name}.fee')} per year. The full fee schedule is in the policy library.")
    d.add("The Board thanks members for their feedback during the consultation on the new fee structure.")

    campus = b.rng.choice(list(w.campuses))
    d = news(f"news-{campus.slug}-refurbishment", f"{campus.full_name} refurbishment complete", "2026-05-20")
    d.add(f"The refurbishment of the {campus.full_name} is complete. The building now has two new seminar rooms, "
          "a quiet study area and improved accessibility.")
    d.add(f"Normal hours have resumed: the campus opens at {b.say(d, f'campus.{campus.slug}.opens')} on weekdays.")

    conf = w.events[0]
    d = news("news-conference-2027-announced", f"{conf.name} announced", "2026-08-11")
    d.add(f"Save the date: the {conf.name} will be held on {b.say(d, f'event.{conf.slug}.date')} at the "
          f"{b.say(d, f'event.{conf.slug}.venue')}. Early-bird member tickets are "
          f"{b.say(d, f'event.{conf.slug}.price')}.")
    d.add("The call for presentations is open until the end of the year.")

    d = news("news-policy-review-2026", "Policy review completed", "2026-06-23")
    d.add("The annual review of Institute policies is complete. The most visible change for students is to "
          f"the Refund and Cancellation Policy: withdrawals within {b.say(d, 'policy.refund_days')} of enrolling "
          "now receive a full refund, less the administration fee.")
    d.add("All current policies are available in the policy library.")


def build_core_documents(b: CorpusBuilder, w: World) -> None:
    """Write every core document. The call order is fixed so the random stream is reproducible."""
    for c in w.courses:
        _course_page(b, w, c)
    _membership_fees(b, w)
    for t in w.tiers:
        _tier_page(b, w, t)
    _cpd_policy(b, w)
    for dept in w.departments:
        _directory(b, w, dept)
    for p in w.people:
        _profile(b, w, p)
    _policies(b, w)
    for cp in w.campuses:
        _campus_page(b, w, cp)
    _contact_page(b, w)
    for e in w.events:
        _event_page(b, w, e)
    _events_calendar(b, w)
    _faqs(b, w)
    _news(b, w)
