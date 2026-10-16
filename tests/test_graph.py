from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from agri_advisor.graph.builder import build_graph
from agri_advisor.graph.nodes import CHAT_NODE
from agri_advisor.messages import content_to_text
from agri_advisor.persistence import open_checkpointer


def make_graph(replies, saver=None):
    llm = GenericFakeChatModel(messages=iter(replies))
    return build_graph(llm, saver or InMemorySaver())


def human(text):
    return {"messages": [HumanMessage(content=text)]}


def texts(result):
    return [m.content for m in result["messages"]]


def test_reply_is_appended_to_history():
    graph = make_graph(["hello farmer"])
    result = graph.invoke(human("hi"), config={"configurable": {"thread_id": "t1"}})
    assert texts(result) == ["hi", "hello farmer"]


def test_memory_persists_within_a_thread():
    graph = make_graph(["first", "second"])
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(human("one"), config=config)
    result = graph.invoke(human("two"), config=config)
    assert texts(result) == ["one", "first", "two", "second"]


def test_threads_are_isolated():
    graph = make_graph(["a", "b"])
    graph.invoke(human("x"), config={"configurable": {"thread_id": "t1"}})
    result = graph.invoke(human("y"), config={"configurable": {"thread_id": "t2"}})
    assert texts(result) == ["y", "b"]


def test_system_prompt_is_sent_but_not_stored():
    class RecordingLLM:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content="ok")

    llm = RecordingLLM()
    graph = build_graph(llm, InMemorySaver())
    result = graph.invoke(human("hi"), config={"configurable": {"thread_id": "t1"}})

    assert isinstance(llm.calls[0][0], SystemMessage)
    assert not any(isinstance(m, SystemMessage) for m in result["messages"])


def test_checkpoint_survives_restart(tmp_path):
    db = tmp_path / "ckpt.db"
    config = {"configurable": {"thread_id": "t1"}}

    with open_checkpointer(db) as saver:
        make_graph(["one"], saver).invoke(human("hi"), config=config)

    with open_checkpointer(db) as saver:
        result = make_graph(["two"], saver).invoke(human("again"), config=config)

    assert texts(result) == ["hi", "one", "again", "two"]


def test_streaming_emits_tokens_from_chat_node():
    graph = make_graph(["hello there farmer"])
    config = {"configurable": {"thread_id": "s1"}}
    streamed = ""
    for chunk, metadata in graph.stream(human("hi"), config=config, stream_mode="messages"):
        if metadata.get("langgraph_node") == CHAT_NODE:
            streamed += content_to_text(chunk.content)
    assert streamed == "hello there farmer"