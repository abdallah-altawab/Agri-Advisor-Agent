import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END

from agri_advisor.graph.builder import build_graph, run_config
from agri_advisor.graph.routing import route_after_chat


class ToolCallingFake(GenericFakeChatModel):
    """Fake chat model that accepts bind_tools (it returns itself)."""

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo:{text}"


@tool
def broken() -> str:
    """Always fails."""
    raise RuntimeError("secret-detail")


def human(text):
    return {"messages": [HumanMessage(content=text)]}


def test_tool_loop_runs_tool_and_returns_final_answer():
    llm = ToolCallingFake(messages=iter([tool_call("echo", {"text": "hi"}, "call_1"), "done"]))
    graph = build_graph(llm, InMemorySaver(), tools=[echo])

    result = graph.invoke(human("go"), config=run_config("t1"))

    kinds = [m.type for m in result["messages"]]
    assert kinds == ["human", "ai", "tool", "ai"]
    assert result["messages"][2].content == "echo:hi"
    assert result["messages"][-1].content == "done"


def test_unexpected_tool_failure_does_not_leak_details():
    llm = ToolCallingFake(messages=iter([tool_call("broken", {}, "call_1"), "sorry"]))
    graph = build_graph(llm, InMemorySaver(), tools=[broken])

    result = graph.invoke(human("go"), config=run_config("t1"))

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "secret-detail" not in tool_messages[0].content
    assert "unavailable" in tool_messages[0].content


def test_endless_tool_calls_are_stopped_by_recursion_limit():
    def endless():
        n = 0
        while True:
            n += 1
            yield tool_call("echo", {"text": "x"}, f"call_{n}")

    graph = build_graph(ToolCallingFake(messages=endless()), InMemorySaver(), tools=[echo])

    with pytest.raises(GraphRecursionError):
        graph.invoke(human("go"), config=run_config("t1", recursion_limit=6))


def test_router_sends_tool_calls_to_tools_node():
    state = {"messages": [tool_call("echo", {"text": "x"}, "call_1")]}
    assert route_after_chat(state) == "tools"


def test_router_ends_on_plain_answer():
    state = {"messages": [AIMessage(content="hello")]}
    assert route_after_chat(state) == END