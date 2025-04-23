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

from rag_store import RagStore
from tools.dice import DICE_TOOL_NAME, DiceTool
from utils import Config, Singleton, get_logger

logger = get_logger(__name__)

TEMPERATURE = 0.0

AGENT_NODE = "agent"
TOOLS_NODE = "tools"
GENERATOR_NODE = "generator"
DICE_NODE = "dice"


class DiceMessage(BaseMessage):
    """Dice message"""

    name: str = "dice"
    type: str = "dice"


class LLM(metaclass=Singleton):
    """LLM for game rules lookup"""

    SYSTEM_MESSAGE = "You are a helpful assistant tasked with looking up game rules and rolling dice."
    RETRIEVER_MESSAGE = "Search and return information about the role playing game."

    agent: StateGraph

    def __init__(self, config: Config, store: RagStore, *args, **kwargs):
        """Initialize the LLM"""
        super().__init__(*args, **kwargs)
        self.config = config

        retriever_tool: Tool = create_retriever_tool(
            store.retriever, "retrieve_rules", self.RETRIEVER_MESSAGE
        )
        self.tools: list[Tool] = [retriever_tool, DiceTool()]
        self.model = init_chat_model(
            config.chat_model,
            model_provider=config.chat_provider,
            api_key=config.chat_api_key,
            streaming=True,
            temperature=TEMPERATURE,
        ).bind_tools(self.tools)

    async def initialize_workflow(self) -> None:
        """Initialize the graph workflow"""
        workflow = StateGraph(MessagesState)
        workflow.add_node(AGENT_NODE, self.agent_node)
        workflow.add_node(TOOLS_NODE, ToolNode(self.tools))
        workflow.add_node(GENERATOR_NODE, self.generator_node)
        workflow.add_node(DICE_NODE, self.dice_node)

        workflow.add_edge(START, AGENT_NODE)
        workflow.add_conditional_edges(
            AGENT_NODE,
            tools_condition,
            {
                TOOLS_NODE: TOOLS_NODE,
                END: END,  # If no tools are called, end the workflow
            },
        )
        workflow.add_conditional_edges(
            TOOLS_NODE,
            self.tools_response_condition,
            {
                GENERATOR_NODE: GENERATOR_NODE,
                DICE_NODE: DICE_NODE,
            },
        )
        workflow.add_edge(GENERATOR_NODE, END)
        self.agent = workflow.compile()

    async def generator_node(
        self, state: MessagesState
    ) -> dict[str, list[dict[str, str]]]:
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

        response = await rag_chain.ainvoke({"context": docs, "question": question})

        logger.debug(f"Response: {response}")
        return {"messages": [response]}

    async def agent_node(self, state: MessagesState):
        """Decides whether to call a tool or not"""
        response = await self.model.ainvoke(
            [SystemMessage(content=self.SYSTEM_MESSAGE)] + state["messages"]
        )
        return {"messages": [response]}

    def tools_response_condition(self, state: MessagesState):
        """Route the tool response to the appropriate node"""
        logger.debug(f"--- Route tool response: {state} ---")
        last_message = state["messages"][-1]

        # Return the response from the dice tool directly
        if (
            isinstance(last_message, ToolMessage)
            and last_message.name == DICE_TOOL_NAME
        ):
            return DICE_NODE

        return GENERATOR_NODE

    async def dice_node(self, state: MessagesState):
        """Create a dice message from the tool response"""
        dice_message = DiceMessage(content=state["messages"][-1].content)
        return {"messages": state["messages"] + [dice_message]}

    def graph(self) -> StateGraph:
        """Get the graph of the workflow"""
        return self.agent.get_graph()

    async def stream_response(
        self, user_input: str, update_func: Callable[[str], None]
    ) -> None:
        """Takes user input and streams the response using the update function"""
        response = ""
        async for event in self.agent.astream_events(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if event["event"] != "on_chain_end" or event["name"] != "LangGraph":
                continue

            messages = event["data"]["output"]["messages"]

            logger.debug(f"Event Output Messages: {messages}")

            # Get the last message
            last_message = messages[-1]

            # If the last message is an AIMessage or DiceMessage, add it to the response
            if isinstance(last_message, (AIMessage, DiceMessage)):
                response += last_message.content
                update_func(response)
