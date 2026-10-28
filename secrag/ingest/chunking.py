"""Paragraph-aware chunking, with sizes measured in words.

Rules (deterministic, so chunk ids are stable across runs):

1. Split the body into paragraphs at blank lines. A paragraph longer than
   `chunk_size` words is cut into consecutive `chunk_size`-word pieces.
2. Pack whole paragraphs into a chunk until the next one would push the chunk
   past `chunk_size` words of new content.
3. Each chunk after the first starts with the last `overlap` words of the
   previous chunk, so a fact split across a boundary still appears whole in
   one chunk. A chunk therefore holds at most `chunk_size + overlap` words.

Chunks keep their document's metadata (title, source, trust level, ...) so
retrieval results can be cited and provenance defences can act on them.
"""

from __future__ import annotations

import re

from secrag.config import ChunkingConfig
from secrag.types import Chunk, Document

_BLANK_LINE = re.compile(r"\n\s*\n")


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _BLANK_LINE.split(text.strip()) if p.strip()]


def _units(text: str, chunk_size: int) -> list[str]:
    units = []
    for para in split_paragraphs(text):
        words = para.split()
        if len(words) <= chunk_size:
            units.append(para)
        else:
            units += [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return units


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunk strings following the rules in the module docstring."""
    chunks: list[str] = []
    prefix = ""  # overlap carried over from the previous chunk
    current: list[str] = []
    current_words = 0

    def emit() -> str:
        new_content = "\n\n".join(current)
        chunks.append(f"{prefix}\n\n{new_content}" if prefix else new_content)
        return " ".join(new_content.split()[-overlap:]) if overlap else ""

    for unit in _units(text, chunk_size):
        n = len(unit.split())
        if current and current_words + n > chunk_size:
            prefix = emit()
            current, current_words = [], 0
        current.append(unit)
        current_words += n
    if current:
        emit()
    return chunks


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{i}",
            doc_id=doc.doc_id,
            position=i,
            text=text,
            title=doc.title,
            source=doc.source,
            author=doc.author,
            trust_level=doc.trust_level,
            created_at=doc.created_at,
            doc_type=doc.doc_type,
            doc_hash=doc.content_hash,
        )
        for i, text in enumerate(chunk_text(doc.text, chunk_size, overlap))
    ]


def chunk_corpus(docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, cfg.chunk_size, cfg.overlap)]
