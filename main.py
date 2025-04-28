import asyncio

from pydantic import ValidationError

import rag_store
from cli import SCREEN_CHAT, SCREEN_MANAGE_STORE, CliApp
from utils import Config, get_logger
from workflows import LLM

logger = get_logger(__name__)

try:
    config = Config()
except ValidationError as e:
    print(f"Failed to load config. {e}")
    exit(1)

store = rag_store.RagStore(config, rag_store.get_client())
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
