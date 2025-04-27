import glob
import itertools
from time import time

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import Config, Singleton, get_logger

logger = get_logger(__name__)

CHROMA_PATH = ".chroma"
TEXT_SPLITTER_BATCH_SIZE = 50  # Number of documents to split at a time


def get_embedder(config):
    """Get embeddings for PDFs"""
    return OllamaEmbeddings(model=config.embedding_model)


def get_splitter(embedding_function):
    """Get text splitter for PDFs"""
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def get_client() -> chromadb.Client:
    """Get client for Chroma."""
    # By default, Chroma stores data in a .chroma directory in the current directory
    return chromadb.PersistentClient(path=CHROMA_PATH)


class RagStore(metaclass=Singleton):
    """RAG store for PDFs"""

    def __init__(self, config: Config, *args, **kwargs):
        """Initialize the RAG store"""
        super().__init__(*args, **kwargs)

        self.config = config
        self.pdf_dir = config.pdf_dir
        self.create_store()

    def create_store(self):
        """Create a new store"""
        client: chromadb.Client = get_client()
        logger.debug(f"Creating store: {self.config.chroma_collection_name}")
        self.store = Chroma(
            collection_name=self.config.chroma_collection_name,
            embedding_function=get_embedder(self.config),
            client=client,
            collection_metadata={"source": "pdfs"},
        )
        self.retriever: VectorStoreRetriever = self.store.as_retriever()

    async def reset(self):
        """Reset the RAG store"""
        client = get_client()
        client.delete_collection(name=self.config.chroma_collection_name)
        self.create_store()

    def get_count(self) -> int:
        return self.store._collection.count()

    async def load(self, update_func):
        """Load PDFs to the store from a directory"""
        file_paths = glob.glob(self.pdf_dir + "/**/*.pdf", recursive=True)
        loaders = [PyMuPDFLoader(file_path, mode="page") for file_path in file_paths]

        pages = []
        for loader in loaders:
            logger.debug(f"Loading: {loader.file_path}")

            async for page in loader.alazy_load():
                pages.append(page)

        await self.load_pages(pages, update_func)

    async def load_pages(self, pages, update_func):
        """Load pages into the store"""

        text_splitter = get_splitter(get_embedder(self.config))
        doc_splits = text_splitter.split_documents(pages)

        start = time()
        total = len(doc_splits)
        completed = 0
        update_func(total=total)

        # process doc splits in batches
        for batch in itertools.batched(doc_splits, TEXT_SPLITTER_BATCH_SIZE):
            await self.store.aadd_documents(batch)
            completed += len(batch)
            logger.debug(
                f"{completed}/{total} pages loaded in {time() - start:.2f} seconds"
            )
            update_func(advance=len(batch))
