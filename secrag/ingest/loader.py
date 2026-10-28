"""Load documents from markdown files with YAML front-matter.

File format (one document per file):

    ---
    doc_id: course-dg101
    title: Foundations of Data Governance (DG101)
    source: course-catalogue
    author: Priya Raman
    trust_level: official
    created_at: '2026-08-14'
    doc_type: course
    ---
    # Foundations of Data Governance (DG101)

    Body paragraphs ...

Evaluator-only labels (decision D12) are refused here: a corpus file that
carries e.g. `is_poisoned` is an error, so such labels cannot reach the
pipeline through the documents themselves.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from secrag.types import Document, TrustLevel

REQUIRED_KEYS = ("doc_id", "title", "source", "author", "trust_level", "created_at")
OPTIONAL_KEYS = ("doc_type",)
# Keys that would leak ground truth into the system under test.
FORBIDDEN_KEYS = frozenset({
    "is_poisoned", "poisoned", "label", "labels", "attack", "attack_id", "attack_type",
    "target_qa_id", "target_answer", "gold", "is_gold", "is_stale",
})


class CorpusFormatError(ValueError):
    pass


def parse_markdown(text: str, origin: str = "<string>") -> Document:
    """Parse one front-matter markdown document."""
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise CorpusFormatError(f"{origin}: missing YAML front-matter")
    try:
        _, header, body = text.split("---\n", 2)
    except ValueError as exc:
        raise CorpusFormatError(f"{origin}: unterminated front-matter") from exc
    meta = yaml.safe_load(header) or {}
    leaked = FORBIDDEN_KEYS & set(meta)
    if leaked:
        raise CorpusFormatError(f"{origin}: evaluator-only keys in front-matter: {sorted(leaked)}")
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise CorpusFormatError(f"{origin}: missing front-matter keys {missing}")
    extra = {k: v for k, v in meta.items() if k not in REQUIRED_KEYS + OPTIONAL_KEYS}

    title = str(meta["title"])
    body = body.strip()
    heading = f"# {title}"
    if body.startswith(heading):  # the title is shown once, not repeated in the body
        body = body[len(heading):].strip()
    return Document(
        doc_id=str(meta["doc_id"]),
        title=title,
        text=body,
        source=str(meta["source"]),
        author=str(meta["author"]),
        trust_level=TrustLevel.parse(meta["trust_level"]),
        created_at=str(meta["created_at"]),
        doc_type=str(meta.get("doc_type", "page")),
        extra=extra,
    )


def document_to_markdown(doc: Document) -> str:
    """Inverse of `parse_markdown` (used by the data generator and attack writers)."""
    meta = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "source": doc.source,
        "author": doc.author,
        "trust_level": doc.trust_level.label(),
        "created_at": doc.created_at,
        "doc_type": doc.doc_type,
        **doc.extra,
    }
    header = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000)
    return f"---\n{header}---\n# {doc.title}\n\n{doc.text}\n"


def load_corpus(dirs: list[str | Path]) -> list[Document]:
    """Load every `*.md` file under the given directories, sorted by doc_id."""
    docs: dict[str, Document] = {}
    for directory in dirs:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"corpus directory not found: {directory}")
        for path in sorted(directory.rglob("*.md")):
            doc = parse_markdown(path.read_text(encoding="utf-8"), origin=str(path))
            if doc.doc_id in docs:
                raise CorpusFormatError(f"{path}: duplicate doc_id {doc.doc_id!r}")
            docs[doc.doc_id] = doc
    return [docs[k] for k in sorted(docs)]
