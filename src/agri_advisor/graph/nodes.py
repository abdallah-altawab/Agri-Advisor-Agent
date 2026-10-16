"""Graph nodes."""

import logging
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage

from agri_advisor.graph.prompts import SYSTEM_PROMPT
from agri_advisor.graph.state import AgentState

log = logging.getLogger(__name__)

CHAT_NODE = "chat"


def make_chat_node(
    llm: BaseChatModel,
    system_prompt: str = SYSTEM_PROMPT,
) -> Callable[[AgentState], dict[str, list[AnyMessage]]]:
    """Create the LLM node.

    The system prompt is prepended at call time and is NOT stored in the state,
    so changing the prompt later does not require migrating saved conversations.
    """
    system = SystemMessage(content=system_prompt)

    def chat_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        history = state["messages"]
        log.debug("chat_node: %d messages in history", len(history))
        response = llm.invoke([system, *history])
        return {"messages": [response]}

    return chat_node