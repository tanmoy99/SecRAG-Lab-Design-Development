import re
from collections import Counter

from secrag.config import REPO_ROOT
from secrag.data_gen import generate, write_generated
from secrag.eval.matching import contains_answer
from secrag.types import TrustLevel


def test_generation_is_deterministic(generated):
    again = generate(13)
    assert [d.content_hash for d in again.documents] == [d.content_hash for d in generated.documents]
    assert [q.to_record() for q in again.qa] == [q.to_record() for q in generated.qa]


def test_different_seed_gives_different_world(generated):
    other = generate(14)
    assert [f.value for f in other.facts] != [f.value for f in generated.facts]


def test_corpus_size_matches_design(generated):
    n_core = len(generated.core_doc_ids)
    assert 60 <= n_core <= 100
    assert len(generated.documents) - n_core == 150
    assert len({d.doc_id for d in generated.documents}) == len(generated.documents)


def test_qa_size_and_unanswerable_share(generated):
    assert 80 <= len(generated.qa) <= 120
    assert sum(not q.answerable for q in generated.qa) == 15
    assert len({q.qa_id for q in generated.qa}) == len(generated.qa)
    assert len({q.question for q in generated.qa}) == len(generated.qa)


def test_every_fact_is_stated_and_every_gold_doc_contains_the_answer(generated):
    docs = {d.doc_id: d for d in generated.documents}
    for f in generated.facts:
        assert f.stated_in, f.fact_id
        for doc_id in f.stated_in:
            assert doc_id in generated.core_doc_ids
            d = docs[doc_id]
            assert contains_answer(d.title + "\n" + d.text, f.kind, f.value, f.unit, f.aliases), (f.fact_id, doc_id)


def test_no_untracked_statement_of_a_fact(generated):
    """A document containing a fact's current value must be listed as stating it (gold labels complete).

    Only checked for values unique among facts: "10 weeks" may be true of several courses, and names
    and venues (kind "text") are mentioned freely, e.g. in minutes.
    """
    values = Counter((f.kind, f.value, f.unit) for f in generated.facts)
    for f in generated.facts:
        if f.kind == "text" or values[(f.kind, f.value, f.unit)] > 1:
            continue
        for d in generated.documents:
            if contains_answer(d.text, f.kind, f.value, f.unit, f.aliases):
                assert d.doc_id in f.stated_in, (f.fact_id, d.doc_id)


def test_stale_posts_are_unverified_and_disagree_with_current_value(generated):
    docs = {d.doc_id: d for d in generated.documents}
    stale = [f for f in generated.facts if f.stale_value]
    assert len(stale) == 6
    for f in stale:
        for doc_id in f.stale_in:
            assert docs[doc_id].trust_level == TrustLevel.UNVERIFIED
            assert docs[doc_id].source == "member-forum"
        assert f.stale_value not in f.aliases


def test_answerable_gold_docs_match_fact(generated):
    facts = {f.fact_id: f for f in generated.facts}
    for q in generated.qa:
        if q.answerable:
            assert q.gold_doc_ids == tuple(sorted(facts[q.fact_id].stated_in))
            assert q.answer == facts[q.fact_id].value
        else:
            assert q.answer is None and not q.gold_doc_ids


def test_emails_use_example_domains(generated):
    emails = {e for d in generated.documents for e in re.findall(r"[\w.'-]+@[\w.-]+", d.text)}
    assert emails and all(e.rstrip(".").endswith(".example") for e in emails)


def test_trust_levels_by_source(generated):
    by_source = {(d.source, d.trust_level) for d in generated.documents}
    assert ("policy-library", TrustLevel.OFFICIAL) in by_source
    assert all(t == TrustLevel.UNVERIFIED for s, t in by_source if s in ("member-forum", "job-board"))
    assert Counter(d.trust_level for d in generated.documents)[TrustLevel.OFFICIAL] >= 25


def test_committed_data_matches_generator(tmp_path, generated):
    """data/ in the repo must be exactly what `generate-data --seed 13` writes."""
    write_generated(generated, tmp_path / "corpus", tmp_path / "qa.jsonl", tmp_path / "facts.jsonl")
    repo_data = REPO_ROOT / "data"

    def read(path):
        return path.read_text(encoding="utf-8").replace("\r\n", "\n")

    fresh = sorted(p.name for p in (tmp_path / "corpus").glob("*.md"))
    committed = sorted(p.name for p in (repo_data / "corpus" / "northbridge").glob("*.md"))
    assert fresh == committed, "regenerate with: python -m secrag generate-data"
    for name in fresh:
        assert read(tmp_path / "corpus" / name) == read(repo_data / "corpus" / "northbridge" / name), name
    assert read(tmp_path / "qa.jsonl") == read(repo_data / "qa" / "northbridge_qa.jsonl")
    assert read(tmp_path / "facts.jsonl") == read(repo_data / "qa" / "northbridge_facts.jsonl")
