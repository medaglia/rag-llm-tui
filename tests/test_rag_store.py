import glob
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import chromadb
import pytest
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters.base import TextSplitter

import rag_store
from utils import Config

test_client = chromadb.EphemeralClient(
    settings=chromadb.config.Settings(),
    tenant=chromadb.config.DEFAULT_TENANT,
    database=chromadb.config.DEFAULT_DATABASE,
)

test_config = Config(_env_file=".env.test")


@pytest.fixture
def test_store():
    rag_store.RagStore.clear()
    yield rag_store.RagStore(Config(_env_file=".env.test"), client=test_client)


def test_get_splitter():
    splitter = rag_store.get_splitter()
    assert isinstance(splitter, TextSplitter)


class TestRagStore:
    def test_create_store(self):
        store = rag_store.RagStore(Config(_env_file=".env.test"), client=test_client)
        assert store.get_count() == 0
        assert isinstance(store.retriever, VectorStoreRetriever)
        assert isinstance(store.store, Chroma)

    @pytest.mark.asyncio
    async def test_reset(self, test_store):
        await test_store.reset()
        assert test_store.get_count() == 0

    def test_get_count(self, test_store):
        assert test_store.get_count() == 0

    @pytest.mark.asyncio
    async def test_load(self, test_store, monkeypatch):
        file_paths = ["pdfs/doc1.pdf", "pdfs/doc2.pdf"]
        monkeypatch.setattr(glob, "glob", MagicMock(return_value=file_paths))

        mock_loader_instance = AsyncMock(PyMuPDFLoader)
        mock_loader_instance.file_path = "mock_file_path.pdf"
        mock_lazy_load = MagicMock()
        mock_lazy_load.__aiter__.return_value = [{"content": "page1"}]
        mock_loader_instance.alazy_load.return_value = mock_lazy_load

        mock_loader_class = MagicMock(return_value=mock_loader_instance)
        monkeypatch.setattr(rag_store, "PyMuPDFLoader", mock_loader_class)

        test_store.load_pages = AsyncMock()

        update_func = Mock()

        await test_store.load(update_func)

        glob.glob.assert_called_once_with(
            test_config.pdf_dir + "/**/*.pdf", recursive=True
        )
        assert mock_loader_class.call_count == len(file_paths)
        test_store.load_pages.assert_awaited_once_with(
            [{"content": "page1"}, {"content": "page1"}], update_func
        )

    @patch.object(Chroma, "aadd_documents")
    @patch("rag_store.get_splitter")
    @pytest.mark.asyncio
    async def test_load_pages(self, mock_get_splitter, mock_aadd_documents, test_store):
        update_func = Mock()
        documents = [
            Document(page_content="text_1"),
            Document(page_content="text_2"),
            Document(page_content="text_3"),
        ]
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = documents
        mock_get_splitter.return_value = mock_splitter
        await test_store.load_pages(documents, update_func)

        calls = update_func.call_args_list
        assert calls[0] == call(total=3)
        assert calls[1] == call(advance=3)
