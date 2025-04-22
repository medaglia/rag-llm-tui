from collections.abc import Awaitable, Callable
from functools import partial
from typing import Iterable

from textual import on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Markdown, ProgressBar, Static

from utils import History, get_logger

logger = get_logger(__name__)

SCREEN_CHAT = "chat"
SCREEN_MANAGE_STORE = "manage_store"


class ManageStore(Screen):
    """Manage store screen."""

    BINDINGS = [("escape,space,q", "app.pop_screen", "Close")]
    CSS_PATH = "stylesheets/manage_store.tcss"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(classes="screenBox"):
            with Vertical(id="load-view", classes="dialogBox"):
                yield Static("Manage your PDFs store.")
                with Horizontal():
                    yield Button.success("Load", id="load")
                    yield Button.success("Reset", id="reset")
                yield Container(id="statusBox")
        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = dict(
            load=self.load_rag,
            reset=self.reset_rag,
        )
        event.button.loading = True
        if event.button.id not in actions:
            return

        actions[event.button.id](event.button)

    @work
    async def load_rag(self, button):
        statusBox = self.query_one("#statusBox")
        progress = ProgressBar(show_eta=False, total=100)
        statusBox.remove_children()
        statusBox.mount(progress)

        await self.app.on_load_rag(progress.update)

        button.loading = False
        statusBox = self.query_one("#statusBox")
        statusBox.remove_children()
        statusBox.mount(Static("Store loaded."))

    @work
    async def reset_rag(self, button):
        await self.app.on_reset_rag()

        button.loading = False
        statusBox = self.query_one("#statusBox")
        statusBox.remove_children()
        statusBox.mount(Static("Store reset."))


class Prompt(Markdown):
    """Markdown for the user prompt."""


class Response(Markdown):
    """Markdown for the reply from the LLM."""

    BORDER_TITLE = "AI"


class Chat(Screen):
    """Chat screen."""

    AUTO_FOCUS = "Input"
    CSS_PATH = "stylesheets/chat.tcss"
    BINDINGS = [("up", "history_up", "Get last"), ("down", "history_down", "Get next")]

    history = History(100)

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield Response("How can I help?")
        yield Input(placeholder=">")
        yield Footer()

    @on(Input.Submitted)
    async def on_input(self, event: Input.Submitted) -> None:
        """When the user hits return."""
        chat_view = self.query_one("#chat-view")
        if not event.value.strip():
            return

        event.input.clear()
        await chat_view.mount(Prompt(event.value))
        await chat_view.mount(response := Response("..."))
        response.anchor()
        self.history.append(event.value)
        self.send_prompt(event.value, response)

    def action_history_up(self) -> None:
        """Get the last message from history and insert it into the input."""

        text = self.history.prev()
        if text is None:
            return

        input = self.query_one(Input)
        input.clear()
        input.insert(text, 0)

    def action_history_down(self) -> None:
        """Get the next message from history and insert it into the input."""

        text = self.history.next()

        # If we can't go forward any more, insert an empty string
        if text is None:
            text = ""

        input = self.query_one(Input)
        input.clear()
        input.insert(text, 0)

    @work(thread=True)
    def send_prompt(self, prompt: str, response: Response) -> None:
        """Call the parent's prompt function so that it can update the thread."""
        update_func = partial(self.app.call_from_thread, response.update)
        self.app.prompt_func(prompt, update_func)


class CliApp(App):
    """Simple app to demonstrate chatting to an LLM."""

    SCREENS = {
        SCREEN_CHAT: Chat,
        SCREEN_MANAGE_STORE: ManageStore,
    }

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        yield SystemCommand("Admin PDFs", "Manage PDF store", self.load_pdf_screen)

    def load_pdf_screen(self):
        self.push_screen(ManageStore())

    def __init__(
        self,
        prompt_func,
        on_load_rag: Callable[[], Awaitable[None]],
        on_reset_rag: Callable[[], Awaitable[None]],
        initial_screen: str = SCREEN_CHAT,
        mount_func: Callable[[App], Awaitable[None]] | None = None,
        *args,
        **kwargs,
    ):
        self.prompt_func = prompt_func
        self.on_load_rag = on_load_rag
        self.on_reset_rag = on_reset_rag
        self.initial_screen = initial_screen
        if mount_func and callable(mount_func):
            self.mount_func: Callable = mount_func

        super().__init__(*args, **kwargs)

    async def on_mount(self) -> None:
        """Run setup on mount"""
        if self.mount_func:
            await self.mount_func(self)

        self.title = "Agent"

        # Make the chat screen the base screen
        self.push_screen(SCREEN_CHAT)

        if self.initial_screen != SCREEN_CHAT:
            self.push_screen(self.initial_screen)
