"""Synthetic corpus and QA generator for the fictional Northbridge Institute.

    python -m secrag generate-data --seed 13

writes `data/corpus/northbridge/*.md` (documents the pipeline reads) and
`data/qa/northbridge_{qa,facts}.jsonl` (evaluator-only ground truth).
The same seed always produces byte-identical files.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from secrag.data_gen.builder import CorpusBuilder
from secrag.data_gen.distractors import build_distractors
from secrag.data_gen.documents import build_core_documents
from secrag.data_gen.facts import Fact
from secrag.data_gen.qa import build_qa
from secrag.data_gen.world import World, build_facts, build_world
from secrag.eval.qa import QAItem, write_jsonl
from secrag.ingest.loader import document_to_markdown
from secrag.types import Document

DEFAULT_SEED = 13


@dataclass(frozen=True)
class GeneratedData:
    world: World
    documents: list[Document]
    core_doc_ids: frozenset[str]
    facts: list[Fact]
    qa: list[QAItem]


def generate(seed: int = DEFAULT_SEED) -> GeneratedData:
    rng = random.Random(seed)
    world = build_world(rng)
    facts = build_facts(world)
    builder = CorpusBuilder(facts, rng)
    build_core_documents(builder, world)
    core_ids = frozenset(d.doc_id for d in builder.drafts)
    build_distractors(builder, world)
    qa = build_qa(facts, world, rng)
    return GeneratedData(world, builder.documents(), core_ids, list(facts.values()), qa)


def write_generated(data: GeneratedData, corpus_dir: Path, qa_path: Path, facts_path: Path) -> None:
    """Write the corpus (replacing any old *.md files there) and the evaluator files."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for old in corpus_dir.glob("*.md"):
        old.unlink()
    for doc in data.documents:
        (corpus_dir / f"{doc.doc_id}.md").write_text(document_to_markdown(doc), encoding="utf-8", newline="\n")
    write_jsonl(qa_path, [q.to_record() for q in data.qa])
    write_jsonl(facts_path, [f.to_record() for f in data.facts])
