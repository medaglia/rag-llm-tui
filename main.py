import asyncio
import os

from cli import SCREEN_CHAT, SCREEN_MANAGE_STORE, CliApp
from config import Config
from rag_store import RagStore
from utils import get_logger
from workflows import LLM

logger = get_logger(__name__)

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "CHAT_MODEL_PROVIDER",
    "CHAT_MODEL",
    "PDF_DIR",
    "EMBEDDING_MODEL",
]


def validate_env() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise Exception(f"Missing environmental variable(s) {missing}")


config = Config()
store = RagStore(config)
llm_agent = LLM(config, store)


async def on_mount(_app) -> None:
    """Run on cli mount"""
    logger.debug("\n" + llm_agent.graph().draw_mermaid())


async def on_prompt(text: str, update_func) -> None:
    """Takes user input and streams the response using the update function"""
    await llm_agent.stream_response(text, update_func)


async def load_rag(update_func) -> None:
    """Create and load the RAG store. Update progress using the update function"""
    await store.load(update_func)


async def reset_rag() -> None:
    """Reset the RAG store"""
    await store.reset()


async def main() -> None:
    validate_env()
    await llm_agent.initialize_workflow()

    # check for rag initialization and prompt user
    initial_screen = SCREEN_MANAGE_STORE if store.get_count() == 0 else SCREEN_CHAT

    app = CliApp(
        on_prompt,
        on_load_rag=load_rag,
        on_reset_rag=reset_rag,
        initial_screen=initial_screen,
        mount_func=on_mount,
    )
    await app.run_async()


if __name__ == "__main__":
    asyncio.run(main())
