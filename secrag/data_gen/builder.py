"""Shared machinery for writing synthetic documents.

A `DocDraft` collects paragraphs; `CorpusBuilder.say(draft, fact_id)` returns a
surface form of a fact *and* records that the draft states it. That record is
how the QA set knows its gold documents, so a document can never state a fact
without the evaluator knowing.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field

from secrag.data_gen.facts import Fact
from secrag.types import Document, TrustLevel


@dataclass
class DocDraft:
    doc_id: str
    title: str
    source: str
    author: str
    trust_level: TrustLevel
    created_at: str
    doc_type: str
    paragraphs: list[str] = field(default_factory=list)

    def add(self, *paragraphs: str) -> None:
        self.paragraphs.extend(paragraphs)

    def to_document(self) -> Document:
        return Document(
            doc_id=self.doc_id,
            title=self.title,
            text="\n\n".join(self.paragraphs),
            source=self.source,
            author=self.author,
            trust_level=self.trust_level,
            created_at=self.created_at,
            doc_type=self.doc_type,
        )


class CorpusBuilder:
    def __init__(self, facts: dict[str, Fact], rng: random.Random):
        self.facts = facts
        self.rng = rng
        self.drafts: list[DocDraft] = []

    def new(
        self,
        doc_id: str,
        title: str,
        source: str,
        author: str,
        trust_level: TrustLevel,
        doc_type: str,
        created_at: str | None = None,
    ) -> DocDraft:
        if any(d.doc_id == doc_id for d in self.drafts):
            raise ValueError(f"duplicate doc_id {doc_id!r}")
        draft = DocDraft(doc_id, title, source, author, trust_level,
                         created_at or self.random_date(dt.date(2026, 1, 12), dt.date(2026, 10, 16)), doc_type)
        self.drafts.append(draft)
        return draft

    def say(self, draft: DocDraft, fact_id: str) -> str:
        """A surface form of the fact's current value; marks the draft as stating it."""
        self.mark(draft, fact_id)
        return self.rng.choice(self.facts[fact_id].aliases)

    def mark(self, draft: DocDraft, fact_id: str) -> None:
        """Record that the draft states a fact (use when another fact rendered the same value)."""
        fact = self.facts[fact_id]
        if draft.doc_id not in fact.stated_in:
            fact.stated_in.append(draft.doc_id)

    def pick(self, options: list[str]) -> str:
        return self.rng.choice(options)

    def random_date(self, start: dt.date, end: dt.date) -> str:
        return (start + dt.timedelta(days=self.rng.randint(0, (end - start).days))).isoformat()

    def documents(self) -> list[Document]:
        return [d.to_document() for d in self.drafts]
