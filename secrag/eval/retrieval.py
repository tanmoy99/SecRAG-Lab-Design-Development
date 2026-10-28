"""Retrieval metrics, computed per question so they can be bootstrapped.

For an answerable question with gold documents G and the top-k retrieved chunks:

- hit@k          1 if any of the top-k chunks comes from a gold document, else 0
- recall@k       |G ∩ docs(top-k)| / |G|  (facts stated in several documents
                 have several gold documents, so recall@1 can be below 1 even
                 for a perfect retriever; hit@k is the headline metric)
- answer_hit@k   1 if a top-k chunk from a gold document actually contains the
                 answer (chunking can separate a document's title from the value)
- rr             reciprocal rank of the first gold-document chunk (0 if none in the
                 retrieved list); its mean is MRR

Unanswerable questions have no gold documents and are skipped here.
"""

from __future__ import annotations

import time
from typing import Any

from secrag.eval.matching import contains_answer
from secrag.eval.qa import QAItem
from secrag.types import RetrievedChunk


def question_metrics(item: QAItem, results: list[RetrievedChunk], ks: list[int]) -> dict[str, Any]:
    gold = set(item.gold_doc_ids)
    doc_ids = [rc.chunk.doc_id for rc in results]
    answer_at = [
        rc.chunk.doc_id in gold
        and contains_answer(rc.chunk.text, item.answer_kind or "text", item.answer or "", item.unit, item.aliases)
        for rc in results
    ]
    first_gold = next((i for i, d in enumerate(doc_ids, 1) if d in gold), None)
    row: dict[str, Any] = {
        "qa_id": item.qa_id,
        "category": item.category,
        "n_gold": len(gold),
        "first_gold_rank": first_gold,
        "rr": 1.0 / first_gold if first_gold else 0.0,
    }
    for k in ks:
        top = set(doc_ids[:k])
        row[f"hit@{k}"] = float(bool(gold & top))
        row[f"recall@{k}"] = len(gold & top) / len(gold)
        row[f"answer_hit@{k}"] = float(any(answer_at[:k]))
    return row


def evaluate_retriever(retriever, items: list[QAItem], ks: list[int]) -> list[dict[str, Any]]:
    """Run every answerable question through the retriever; one metrics row per question."""
    depth = max(ks)
    rows = []
    for item in items:
        if not item.answerable:
            continue
        start = time.perf_counter()
        results = retriever.retrieve(item.question, depth)
        latency_ms = (time.perf_counter() - start) * 1000
        rows.append({**question_metrics(item, results, ks), "latency_ms": latency_ms,
                     "top_doc_ids": " ".join(rc.chunk.doc_id for rc in results[:5])})
    return rows


def metric_names(ks: list[int]) -> list[str]:
    return [f"{m}@{k}" for m in ("hit", "recall", "answer_hit") for k in ks] + ["rr"]
