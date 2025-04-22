from typing import Callable

from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import Tool
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import Config
from dice_tool import DICE_TOOL_NAME, DiceTool
from rag_store import RagStore
from utils import Singleton, get_logger

logger = get_logger(__name__)

TEMPERATURE = 0.0


class DiceMessage(BaseMessage):
    """Dice message"""

    name: str = "dice"
    type: str = "dice"


class LLM(metaclass=Singleton):
    """LLM for game rules lookup"""

    SYSTEM_MESSAGE = "You are a helpful assistant tasked with looking up game rules and rolling dice."
    RETRIEVER_MESSAGE = "Search and return information about the role playing game."

    def __init__(self, config: Config, store: RagStore, *args, **kwargs):
        """Initialize the LLM"""
        super().__init__(*args, **kwargs)
        self.config = config

        retriever_tool: Tool = create_retriever_tool(
            store.retriever, "retrieve_rules", self.RETRIEVER_MESSAGE
        )
        tools: list[Tool] = [retriever_tool, DiceTool()]

        self.model = init_chat_model(
            config.chat_model,
            model_provider=config.chat_provider,
            api_key=config.chat_api_key,
            streaming=True,
            temperature=TEMPERATURE,
        ).bind_tools(tools)

        # Create the agent
        self.agent = self.create_agent(tools).compile()

    def create_agent(self, tools: list[Tool]) -> StateGraph:
        """Create the agent"""
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("dice", self.dice_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END,
            },
        )
        # workflow.add_edge("tools", "generate")
        workflow.add_conditional_edges(
            "tools",
            self.tools_response_condition,
            {
                "generate": "generate",
                "dice": "dice",
            },
        )
        workflow.add_edge("generate", END)
        return workflow

    def generate_node(self, state: MessagesState) -> dict[str, list[dict[str, str]]]:
        """Generate a response based on the original prompt and the retrieved documents"""

        prompt = hub.pull("rlm/rag-prompt")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content

        model = init_chat_model(
            self.config.chat_model,
            model_provider=self.config.chat_provider,
            api_key=self.config.chat_api_key,
            streaming=True,
            temperature=TEMPERATURE,
        )

        rag_chain = prompt | model

        # TODO: make async
        response = rag_chain.invoke({"context": docs, "question": question})

        logger.debug(f"Response: {response}")
        return {"messages": [response]}

    def agent_node(self, state: MessagesState):
        """LLM decides whether to call a tool or not"""

        # TODO: make async
        return {
            "messages": [
                self.model.invoke(
                    [SystemMessage(content=self.SYSTEM_MESSAGE)] + state["messages"]
                )
            ]
        }

    def dice_node(self, state: MessagesState):
        """Create a dice message from the tool response"""
        dice_message = DiceMessage(content=state["messages"][-1].content)
        return {"messages": state["messages"] + [dice_message]}

    def tools_response_condition(self, state: MessagesState):
        """Route the tool response to the appropriate node"""
        logger.debug(f"--- Route tool response: {state} ---")
        last_message = state["messages"][-1]

        # Return the response from the dice tool directly
        if (
            isinstance(last_message, ToolMessage)
            and last_message.name == DICE_TOOL_NAME
        ):
            return "dice"

        return "generate"

    def graph(self) -> StateGraph:
        """Get the graph of the workflow"""
        return self.agent.get_graph()

    def stream_response(
        self, user_input: str, update_func: Callable[[str], None]
    ) -> None:
        """Takes user input and streams the response using the update function"""
        response = ""
        for event in self.agent.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            logger.debug(f"Event: {event}")
            for value in event.values():
                if "messages" not in value or not value["messages"]:
                    continue
                last_message = value["messages"][-1]

                # If the last message is an AIMessage, add it to the response
                if isinstance(last_message, AIMessage):
                    response += last_message.content
                    update_func(response)

                elif isinstance(last_message, DiceMessage):
                    response += last_message.content
                    update_func(response)
