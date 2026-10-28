"""QA items: the questions we ask and the evaluator-only gold labels for them.

A `QAItem` is evaluator-side (decision D12): the pipeline only ever receives
`item.question`. Gold documents and answers stay here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QAItem:
    qa_id: str  # "q001", ...
    question: str
    answerable: bool
    category: str  # e.g. "course.fee", "policy.refund_days", "unanswerable"
    answer: str | None = None  # canonical value, e.g. "AUD 1,285"
    answer_kind: str | None = None  # money | date | email | phone | number | text
    unit: str | None = None  # for numbers, e.g. "days"
    aliases: tuple[str, ...] = ()
    fact_id: str | None = None
    gold_doc_ids: tuple[str, ...] = ()  # every document that states the answer

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["aliases"] = list(self.aliases)
        record["gold_doc_ids"] = list(self.gold_doc_ids)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> QAItem:
        return cls(**{**record, "aliases": tuple(record.get("aliases", ())),
                      "gold_doc_ids": tuple(record.get("gold_doc_ids", ()))})


def load_qa(path: str | Path) -> list[QAItem]:
    with open(path, encoding="utf-8") as fh:
        return [QAItem.from_record(json.loads(line)) for line in fh if line.strip()]


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
