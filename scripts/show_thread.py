"""Print the saved messages of a conversation thread (no LLM request is made)."""

import argparse

from agri_advisor.config import get_settings
from agri_advisor.graph.builder import build_graph
from agri_advisor.llm import build_llm
from agri_advisor.messages import content_to_text
from agri_advisor.persistence import open_checkpointer


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a saved conversation thread.")
    parser.add_argument("--thread", required=True, help="Thread id to inspect.")
    parser.add_argument("--full", action="store_true", help="Print complete message content.")
    args = parser.parse_args()

    settings = get_settings()
    with open_checkpointer(settings.checkpoint_db_path) as saver:
        graph = build_graph(build_llm(settings), saver)
        snapshot = graph.get_state({"configurable": {"thread_id": args.thread}})

    messages = snapshot.values.get("messages", [])
    print(f"thread={args.thread} messages={len(messages)} next={snapshot.next}")
    for message in messages:
        calls = [call["name"] for call in getattr(message, "tool_calls", None) or []]
        suffix = f" tool_calls={calls}" if calls else ""
        text = content_to_text(message.content)
        if args.full:
            print(f"- {message.type}{suffix}\n{text}\n")
        else:
            print(f"- {message.type:>5}: {text[:80]!r}{suffix}")


if __name__ == "__main__":
    main()