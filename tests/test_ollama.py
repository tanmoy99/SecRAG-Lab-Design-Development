"""Integration tests against a live Ollama server (auto-skipped if it is not running)."""

import numpy as np
import pytest

from secrag.config import EmbedderConfig, get_settings
from secrag.index import build_embedder

pytestmark = pytest.mark.ollama


def test_nomic_embeddings_shape_and_similarity():
    e = build_embedder(EmbedderConfig(provider="ollama", model="nomic-embed-text"), get_settings(), use_cache=False)
    docs = e.embed_documents(["The refund window is 21 days.", "The campus has bicycle parking."])
    q = e.embed_query("How long do I have to get a refund?")
    assert docs.shape == (2, 768)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-4)
    assert docs[0] @ q > docs[1] @ q
