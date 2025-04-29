from typing import Any, Callable, Dict, Optional, Sequence, Union
from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph

from tools.dice import DICE_TOOL_NAME
from workflows import DICE_NODE, GENERATOR_NODE, LLM

fake_responses = [
    "response_1",
    "response_2",
    "response_3",
]


class FakeModel(FakeListChatModel):
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return self


initial_state = MessagesState(messages=[HumanMessage(content="test")])


class TestLLM:
    @patch("workflows.init_chat_model")
    def test_init(self, init_chat_model, config, rag_store):
        init_chat_model.return_value = FakeModel(responses=fake_responses)
        llm = LLM(config, rag_store)
        assert llm.config == config

    @pytest.mark.asyncio
    @patch("workflows.init_chat_model")
    async def test_initialize_workflow(self, init_chat_model, config, rag_store):
        init_chat_model.return_value = FakeModel(responses=fake_responses)
        llm = LLM(config, rag_store)
        await llm.initialize_workflow()
        assert llm.agent is not None
        assert isinstance(llm.agent, CompiledStateGraph)

    @pytest.mark.asyncio
    @patch("workflows.init_chat_model")
    async def test_generator_node(self, init_chat_model, config, rag_store):
        init_chat_model.return_value = FakeModel(responses=fake_responses)
        llm = LLM(config, rag_store)
        await llm.initialize_workflow()
        state = initial_state
        response = await llm.generator_node(state)
        assert response["messages"][-1].content == fake_responses[0]

    @pytest.mark.asyncio
    @patch("workflows.init_chat_model")
    async def test_agent_node(self, init_chat_model, config, rag_store):
        init_chat_model.return_value = FakeModel(responses=fake_responses)
        llm = LLM(config, rag_store)
        await llm.initialize_workflow()
        state = initial_state
        response = await llm.agent_node(state)
        assert response["messages"][-1].content == fake_responses[0]

    @pytest.mark.asyncio
    @patch("workflows.init_chat_model")
    async def test_dice_node(self, init_chat_model, config, rag_store):
        init_chat_model.return_value = FakeModel(responses=fake_responses)
        llm = LLM(config, rag_store)
        await llm.initialize_workflow()
        state = initial_state
        response = await llm.dice_node(state)
        assert response["messages"][-1].content == initial_state["messages"][-1].content

    def test_tools_response_condition(self, config, rag_store):
        llm = LLM(config, rag_store)
        state = initial_state
        response = llm.tools_response_condition(state)
        assert response == GENERATOR_NODE

        state["messages"][-1] = ToolMessage(
            name=DICE_TOOL_NAME, content=["5"], tool_call_id="5"
        )
        response = llm.tools_response_condition(state)
        assert response == DICE_NODE

    # @pytest.mark.asyncio
    # @patch("workflows.init_chat_model")
    # async def test_stream_response(self, init_chat_model, config, rag_store):
    #     init_chat_model.return_value = FakeModel(responses=fake_responses)
    #     llm = LLM(config, rag_store)
    #     await llm.initialize_workflow()
    #     update_func = Mock()
    #     response = await llm.stream_response("test", update_func)
