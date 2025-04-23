import os

from pydantic import BaseModel, Field


class Config(BaseModel):
    chat_model: str = Field(
        default=os.getenv("CHAT_MODEL"),
        description="Chat model",
        frozen=True,
        validate_default=True,
    )
    chat_provider: str = Field(
        default=os.getenv("CHAT_MODEL_PROVIDER"),
        description="Chat provider",
        frozen=True,
        validate_default=True,
    )
    chat_api_key: str = Field(
        default=os.getenv("OPENAI_API_KEY"),
        description="Chat API key",
        frozen=True,
        validate_default=True,
    )
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL"),
        description="Embedding model",
        frozen=True,
        validate_default=True,
    )
    pdf_dir: str = Field(
        default=os.getenv("PDF_DIR"),
        description="The directory containing PDF files for loading",
        frozen=True,
        validate_default=True,
    )
    chroma_collection_name: str = Field(
        default=os.getenv("CHROMA_COLLECTION_NAME"),
        description="Chroma collection name",
        frozen=True,
        validate_default=True,
    )
