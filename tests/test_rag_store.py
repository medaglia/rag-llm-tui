import glob
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters.base import TextSplitter

from rag_store import RagStore, get_splitter


def test_get_splitter():
    splitter = get_splitter()
    assert isinstance(splitter, TextSplitter)


class TestRagStore:
    def test_create_store(self, config, chroma_client):
        store = RagStore(config, client=chroma_client)
        assert store.get_count() == 0
        assert isinstance(store.retriever, VectorStoreRetriever)
        assert isinstance(store.store, Chroma)

    @pytest.mark.asyncio
    async def test_reset(self, rag_store):
        await rag_store.reset()
        assert rag_store.get_count() == 0

    def test_get_count(self, rag_store):
        assert rag_store.get_count() == 0

    @pytest.mark.asyncio
    @patch("rag_store.PyMuPDFLoader")
    async def test_load(self, mock_loader_class, rag_store, monkeypatch):
        file_paths = ["pdfs/doc1.pdf", "pdfs/doc2.pdf"]
        monkeypatch.setattr(glob, "glob", MagicMock(return_value=file_paths))

        mock_loader_instance = AsyncMock(PyMuPDFLoader)
        mock_loader_instance.file_path = "mock_file_path.pdf"
        mock_lazy_load = MagicMock()
        mock_lazy_load.__aiter__.return_value = [{"content": "page1"}]
        mock_loader_instance.alazy_load.return_value = mock_lazy_load

        mock_loader_class.return_value = mock_loader_instance

        rag_store.load_pages = AsyncMock()

        update_func = Mock()

        await rag_store.load(update_func)

        glob.glob.assert_called_once_with(
            rag_store.config.pdf_dir + "/**/*.pdf", recursive=True
        )
        assert mock_loader_class.call_count == len(file_paths)
        rag_store.load_pages.assert_awaited_once_with(
            [{"content": "page1"}, {"content": "page1"}], update_func
        )

    @patch.object(Chroma, "aadd_documents")
    @patch("rag_store.get_splitter")
    @pytest.mark.asyncio
    async def test_load_pages(self, mock_get_splitter, mock_aadd_documents, rag_store):
        update_func = Mock()
        documents = [
            Document(page_content="text_1"),
            Document(page_content="text_2"),
            Document(page_content="text_3"),
        ]
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = documents
        mock_get_splitter.return_value = mock_splitter
        await rag_store.load_pages(documents, update_func)

        calls = update_func.call_args_list
        assert calls[0] == call(total=3)
        assert calls[1] == call(advance=3)
