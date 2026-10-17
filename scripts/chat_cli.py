"""Interactive terminal chat with streaming and tools.

Usage: uv run python scripts/chat_cli.py [--thread NAME]
Re-using the same --thread name continues the same conversation, even after a restart.
"""

import argparse
import logging
import sys

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from agri_advisor.config import get_settings
from agri_advisor.graph.builder import build_graph, run_config
from agri_advisor.graph.nodes import CHAT_NODE, TOOLS_NODE
from agri_advisor.llm import build_llm
from agri_advisor.logging_setup import configure_logging
from agri_advisor.messages import content_to_text
from agri_advisor.persistence import open_checkpointer
from agri_advisor.services.http import make_client
from agri_advisor.tools.weather_tool import build_weather_tool

log = logging.getLogger("chat_cli")


def stream_reply(graph: CompiledStateGraph, user_text: str, thread_id: str) -> None:
    """Send one user message and print the reply token by token (and tool activity)."""
    inputs = {"messages": [HumanMessage(content=user_text)]}
    for chunk, metadata in graph.stream(
        inputs, config=run_config(thread_id), stream_mode="messages"
    ):
        node = metadata.get("langgraph_node")
        if node == TOOLS_NODE and isinstance(chunk, ToolMessage):
            print(f"\n[tool: {chunk.name}]", flush=True)
        elif node == CHAT_NODE and isinstance(chunk, AIMessageChunk):
            print(content_to_text(chunk.content), end="", flush=True)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with Agri Advisor in the terminal.")
    parser.add_argument("--thread", default="cli-default", help="Conversation id to continue.")
    args = parser.parse_args()

    try:
        settings = get_settings()
    except ValidationError as exc:
        missing = [".".join(map(str, e["loc"])) for e in exc.errors()]
        print(f"Invalid or missing settings: {missing}. Check your .env file.")
        return 2

    configure_logging(settings.log_level)
    llm = build_llm(settings)
    print(f"thread={args.thread} | model={settings.llm_model} | type 'exit' to quit")

    with make_client() as http_client, open_checkpointer(settings.checkpoint_db_path) as saver:
        graph = build_graph(llm, saver, tools=[build_weather_tool(http_client)])
        while True:
            try:
                user_text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                continue
            print("Bot: ", end="", flush=True)
            try:
                stream_reply(graph, user_text, args.thread)
            except Exception as exc:  # noqa: BLE001 - keep the chat alive, report the error
                print()
                log.error("Request failed: %s: %s", type(exc).__name__, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())