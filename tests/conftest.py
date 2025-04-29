import chromadb
import pytest

from rag_store import RagStore
from utils import Config


@pytest.fixture
def config():
    yield Config(_env_file=".env.test")


@pytest.fixture
def chroma_client():
    yield chromadb.EphemeralClient(
        settings=chromadb.config.Settings(),
        tenant=chromadb.config.DEFAULT_TENANT,
        database=chromadb.config.DEFAULT_DATABASE,
    )


@pytest.fixture
def rag_store(config, chroma_client):
    RagStore.clear()
    yield RagStore(config, client=chroma_client)
