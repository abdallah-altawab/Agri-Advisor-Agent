"""Graph construction. LLM, tools and checkpointer are injected, which keeps it testable."""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agri_advisor.graph.nodes import CHAT_NODE, TOOLS_NODE, make_chat_node
from agri_advisor.graph.routing import route_after_chat
from agri_advisor.graph.state import AgentState

log = logging.getLogger(__name__)

# One tool round trip is 3 steps (chat -> tools -> chat), so 12 allows about 4 rounds.
DEFAULT_RECURSION_LIMIT = 12

TOOL_FAILURE_MESSAGE = (
    "The tool failed unexpectedly. Tell the user the tool is temporarily unavailable."
)


def _handle_tool_failure(exc: Exception) -> str:
    """Log the real error, but give the model (and the user) a safe, generic message."""
    log.error("Tool execution failed: %s: %s", type(exc).__name__, exc)
    return TOOL_FAILURE_MESSAGE


def run_config(thread_id: str, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> dict[str, Any]:
    """Per-run config: which conversation to use and how many steps are allowed."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


def build_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph:
    """Build the conversation graph.

    Without tools:  START -> chat -> END
    With tools:     START -> chat -> (tools -> chat)* -> END
    """
    builder = StateGraph(AgentState)
    if tools:
        tool_list = list(tools)
        builder.add_node(CHAT_NODE, make_chat_node(llm.bind_tools(tool_list)))
        builder.add_node(
            TOOLS_NODE, ToolNode(tool_list, handle_tool_errors=_handle_tool_failure)
        )
        builder.add_edge(START, CHAT_NODE)
        builder.add_conditional_edges(
            CHAT_NODE, route_after_chat, {TOOLS_NODE: TOOLS_NODE, END: END}
        )
        builder.add_edge(TOOLS_NODE, CHAT_NODE)
    else:
        builder.add_node(CHAT_NODE, make_chat_node(llm))
        builder.add_edge(START, CHAT_NODE)
        builder.add_edge(CHAT_NODE, END)
    return builder.compile(checkpointer=checkpointer)