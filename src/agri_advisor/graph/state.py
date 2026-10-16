"""Graph state definition."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Conversation state.

    `add_messages` is a reducer: node updates are appended to the history
    (and messages with the same id are replaced) instead of overwriting it.
    """

    messages: Annotated[list[AnyMessage], add_messages]