"""Shared fixtures. Tests marked `ollama` are skipped when no Ollama server answers."""

from __future__ import annotations

import httpx
import pytest

from secrag.config import ChunkingConfig, get_settings
from secrag.data_gen import GeneratedData, generate
from secrag.ingest import chunk_corpus
from secrag.types import Chunk


def _ollama_up() -> bool:
    try:
        return httpx.get(f"{get_settings().ollama_host}/api/version", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def pytest_collection_modifyitems(config, items):
    if any("ollama" in item.keywords for item in items) and not _ollama_up():
        skip = pytest.mark.skip(reason="Ollama server not reachable")
        for item in items:
            if "ollama" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def generated() -> GeneratedData:
    return generate(13)


@pytest.fixture(scope="session")
def chunks(generated) -> list[Chunk]:
    return chunk_corpus(generated.documents, ChunkingConfig())
