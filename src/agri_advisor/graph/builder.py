"""Graph construction. The LLM and checkpointer are injected, which keeps it testable."""

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agri_advisor.graph.nodes import CHAT_NODE, make_chat_node
from agri_advisor.graph.state import AgentState


def build_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the conversation graph: START -> chat -> END."""
    builder = StateGraph(AgentState)
    builder.add_node(CHAT_NODE, make_chat_node(llm))
    builder.add_edge(START, CHAT_NODE)
    builder.add_edge(CHAT_NODE, END)
    return builder.compile(checkpointer=checkpointer)