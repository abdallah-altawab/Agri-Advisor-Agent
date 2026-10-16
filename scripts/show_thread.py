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
    args = parser.parse_args()

    settings = get_settings()
    with open_checkpointer(settings.checkpoint_db_path) as saver:
        graph = build_graph(build_llm(settings), saver)
        snapshot = graph.get_state({"configurable": {"thread_id": args.thread}})

    messages = snapshot.values.get("messages", [])
    print(f"thread={args.thread} messages={len(messages)} next={snapshot.next}")
    for message in messages:
        print(f"- {message.type:>5}: {content_to_text(message.content)[:80]!r}")


if __name__ == "__main__":
    main()