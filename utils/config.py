from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    chat_model: str = Field(
        description="Chat model",
        frozen=True,
    )
    chat_provider: str = Field(
        description="Chat provider",
        frozen=True,
    )
    chat_api_key: str = Field(
        description="Chat API key",
        frozen=True,
    )
    embedding_model: str = Field(
        description="Embedding model",
        frozen=True,
    )
    pdf_dir: str = Field(
        description="The directory containing PDF files for loading",
        frozen=True,
    )
    chroma_collection_name: str = Field(
        description="Chroma collection name",
        frozen=True,
    )
