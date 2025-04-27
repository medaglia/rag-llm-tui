import glob
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import chromadb
import pytest
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.vectorstores import VectorStoreRetriever

import rag_store
from utils import Config

test_client = chromadb.EphemeralClient(
    settings=chromadb.config.Settings(),
    tenant=chromadb.config.DEFAULT_TENANT,
    database=chromadb.config.DEFAULT_DATABASE,
)

test_config = Config(_env_file=".env.test")


class TestRagStore:
    @patch("rag_store.get_client", return_value=test_client)
    def test_create_store(self, mock_get_client):
        store = rag_store.RagStore(test_config)
        assert store.get_count() == 0
        assert isinstance(store.retriever, VectorStoreRetriever)
        assert isinstance(store.store, Chroma)

    @patch("rag_store.get_client", return_value=test_client)
    def test_reset(self, mock_get_client):
        store = rag_store.RagStore(test_config)
        store.reset()
        assert store.get_count() == 0

    @patch("rag_store.get_client", return_value=test_client)
    def test_get_count(self, mock_get_client):
        store = rag_store.RagStore(test_config)
        assert store.get_count() == 0

    @patch("rag_store.get_client", return_value=test_client)
    @pytest.mark.asyncio
    async def test_load(self, mock_get_client, monkeypatch):
        store = rag_store.RagStore(test_config)

        file_paths = ["pdfs/doc1.pdf", "pdfs/doc2.pdf"]
        monkeypatch.setattr(glob, "glob", MagicMock(return_value=file_paths))

        mock_loader_instance = AsyncMock(PyMuPDFLoader)
        mock_loader_instance.file_path = "mock_file_path.pdf"
        mock_lazy_load = MagicMock()
        mock_lazy_load.__aiter__.return_value = [{"content": "page1"}]
        mock_loader_instance.alazy_load.return_value = mock_lazy_load

        mock_loader_class = MagicMock(return_value=mock_loader_instance)
        monkeypatch.setattr(rag_store, "PyMuPDFLoader", mock_loader_class)

        store.load_pages = AsyncMock()

        update_func = MagicMock()

        await store.load(update_func)

        glob.glob.assert_called_once_with(
            test_config.pdf_dir + "/**/*.pdf", recursive=True
        )
        assert mock_loader_class.call_count == len(file_paths)
        store.load_pages.assert_awaited_once_with(
            [{"content": "page1"}, {"content": "page1"}], update_func
        )
