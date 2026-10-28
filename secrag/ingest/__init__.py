"""Ingestion: load front-matter markdown documents and split them into chunks."""

from secrag.ingest.chunking import chunk_corpus, chunk_document, chunk_text, split_paragraphs
from secrag.ingest.loader import CorpusFormatError, document_to_markdown, load_corpus, parse_markdown

__all__ = [
    "CorpusFormatError", "chunk_corpus", "chunk_document", "chunk_text", "document_to_markdown",
    "load_corpus", "parse_markdown", "split_paragraphs",
]
