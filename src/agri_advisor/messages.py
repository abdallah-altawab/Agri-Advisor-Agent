"""Helpers for working with LangChain message content."""

from typing import Any


def content_to_text(content: str | list[Any] | None) -> str:
    """Flatten message content into plain text.

    Some providers return `content` as a list of blocks (dicts with a "type" key)
    instead of a string. Only text blocks are kept; other block types
    (and any extra metadata such as signatures) are ignored.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)