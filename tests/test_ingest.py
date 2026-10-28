import pytest

from secrag.ingest import (
    CorpusFormatError,
    chunk_document,
    chunk_text,
    document_to_markdown,
    load_corpus,
    parse_markdown,
)
from secrag.types import Document, TrustLevel

DOC = Document(doc_id="d1", title="A title", text="First paragraph.\n\nSecond paragraph.", source="test",
               author="me", trust_level=TrustLevel.STAFF, created_at="2026-01-02", doc_type="page")


def test_markdown_round_trip():
    parsed = parse_markdown(document_to_markdown(DOC))
    assert parsed == DOC
    assert isinstance(parsed.created_at, str)


def test_crlf_files_parse():
    assert parse_markdown(document_to_markdown(DOC).replace("\n", "\r\n")).text == DOC.text


@pytest.mark.parametrize("key", ["is_poisoned", "attack_id", "target_answer"])
def test_evaluator_only_keys_are_refused(key):
    md = document_to_markdown(DOC).replace("doc_type: page", f"doc_type: page\n{key}: true")
    with pytest.raises(CorpusFormatError, match="evaluator-only"):
        parse_markdown(md)


def test_missing_front_matter_and_keys():
    with pytest.raises(CorpusFormatError):
        parse_markdown("# no front matter")
    with pytest.raises(CorpusFormatError, match="missing"):
        parse_markdown("---\ndoc_id: x\n---\nbody")


def test_load_corpus_sorts_and_rejects_duplicates(tmp_path):
    (tmp_path / "b.md").write_text(document_to_markdown(DOC), encoding="utf-8")
    other = Document(**{**DOC.__dict__, "doc_id": "a0"})
    (tmp_path / "a.md").write_text(document_to_markdown(other), encoding="utf-8")
    assert [d.doc_id for d in load_corpus([tmp_path])] == ["a0", "d1"]
    (tmp_path / "c.md").write_text(document_to_markdown(DOC), encoding="utf-8")
    with pytest.raises(CorpusFormatError, match="duplicate"):
        load_corpus([tmp_path])


def test_committed_corpus_loads(generated):
    from secrag.config import REPO_ROOT

    docs = load_corpus([REPO_ROOT / "data" / "corpus" / "northbridge"])
    assert [d.doc_id for d in docs] == sorted(d.doc_id for d in generated.documents)


def _words(n: int, start: int = 0) -> str:
    return " ".join(f"w{i}" for i in range(start, start + n))


def test_short_text_is_one_chunk():
    assert chunk_text("one two three", 10, 2) == ["one two three"]


def test_paragraphs_are_packed_whole_with_overlap():
    text = "\n\n".join([_words(6, 0), _words(6, 6), _words(6, 12)])
    chunks = chunk_text(text, chunk_size=12, overlap=3)
    assert len(chunks) == 2
    assert chunks[0] == f"{_words(6, 0)}\n\n{_words(6, 6)}"
    assert chunks[1].split()[:3] == ["w9", "w10", "w11"]  # overlap from the previous chunk
    assert chunks[1].endswith(_words(6, 12))


def test_long_paragraph_is_split():
    chunks = chunk_text(_words(25), chunk_size=10, overlap=0)
    assert [len(c.split()) for c in chunks] == [10, 10, 5]


def test_chunks_cover_document_in_order_and_respect_limits(generated):
    size, overlap = 60, 10
    for doc in generated.documents:
        chunks = chunk_document(doc, size, overlap)
        assert [c.chunk_id for c in chunks] == [f"{doc.doc_id}#{i}" for i in range(len(chunks))]
        new_words: list[str] = []
        prev_new = 0  # words of new content in the previous chunk
        for i, c in enumerate(chunks):
            words = c.text.split()
            carried = 0
            if i:
                carried = min(overlap, prev_new)
                assert words[:carried] == chunks[i - 1].text.split()[-carried:]
            assert len(words) - carried <= size
            new_words += words[carried:]
            prev_new = len(words) - carried
            assert c.trust_level == doc.trust_level and c.doc_hash == doc.content_hash
        assert new_words == doc.text.split()
