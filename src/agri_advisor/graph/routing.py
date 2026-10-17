"""Routing logic of the conversation graph."""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from agri_advisor.graph.nodes import TOOLS_NODE
from agri_advisor.graph.state import AgentState


def route_after_chat(state: AgentState) -> str:
    """Go to the tools node if the model asked for tool calls, otherwise finish."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return TOOLS_NODE
    return END