"""Interactive terminal chat with streaming, tools and image attachments.

Usage:
  uv run python scripts/chat_cli.py [--thread NAME] [--stub-classifier --stub-scenario ambiguous]
Attach a photo with:  /image "C:\\path\\to\\leaf.jpg" optional message
Re-using the same --thread name continues the same conversation, even after a restart.
"""

import argparse
import logging
import sys
from pathlib import Path

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from agri_advisor.chat_commands import parse_image_command, with_attachment
from agri_advisor.config import get_settings
from agri_advisor.graph.builder import build_graph, run_config
from agri_advisor.graph.nodes import CHAT_NODE, TOOLS_NODE
from agri_advisor.llm import build_llm
from agri_advisor.logging_setup import configure_logging
from agri_advisor.messages import content_to_text
from agri_advisor.persistence import open_checkpointer
from agri_advisor.services.errors import ServiceError
from agri_advisor.services.http import make_client
from agri_advisor.tools.classifier_tool import build_classify_tool
from agri_advisor.tools.weather_tool import build_weather_tool
from agri_advisor.vision.images import store_image
from agri_advisor.vision.service import DiagnosisService
from agri_advisor.vision.stub import SCENARIOS, StubClassifier

log = logging.getLogger("chat_cli")

STUB_NOTICE = "DEMO STUB MODEL: this result is fake test data, not a real diagnosis. Tell the user."


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


def prepare_message(line: str, images_enabled: bool, uploads_dir: Path) -> str | None:
    """Turn a typed line into the message to send; None means 'do not send'."""
    try:
        command = parse_image_command(line)
    except ValueError as exc:
        print(exc)
        return None
    if command is None:
        return line
    if not images_enabled:
        print("Image analysis is not enabled in this session.")
        return None
    try:
        image_id = store_image(Path(command.path), uploads_dir)
    except ServiceError as exc:
        print(f"Image rejected: {exc}")
        return None
    return with_attachment(command.text, image_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with Agri Advisor in the terminal.")
    parser.add_argument("--thread", default="cli-default", help="Conversation id to continue.")
    parser.add_argument(
        "--stub-classifier", action="store_true", help="Use a FAKE image classifier (demo only)."
    )
    parser.add_argument("--stub-scenario", choices=sorted(SCENARIOS), default="ambiguous")
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
        tools: list[BaseTool] = [build_weather_tool(http_client)]
        if args.stub_classifier:
            service = DiagnosisService(StubClassifier.from_scenario(args.stub_scenario))
            tools.append(build_classify_tool(service, settings.uploads_dir, notice=STUB_NOTICE))
            print(f"*** STUB CLASSIFIER ACTIVE (scenario={args.stub_scenario}): results are FAKE ***")
        graph = build_graph(llm, saver, tools=tools)

        while True:
            try:
                line = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.lower() in {"exit", "quit"}:
                break
            if not line:
                continue
            message = prepare_message(line, args.stub_classifier, settings.uploads_dir)
            if message is None:
                continue
            print("Bot: ", end="", flush=True)
            try:
                stream_reply(graph, message, args.thread)
            except Exception as exc:  # noqa: BLE001 - keep the chat alive, report the error
                print()
                log.error("Request failed: %s: %s", type(exc).__name__, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())