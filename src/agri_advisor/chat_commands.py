"""Parsing of terminal chat commands (separate from the script so it can be unit-tested)."""

from dataclasses import dataclass

DEFAULT_IMAGE_TEXT = "Please look at the attached photo."


@dataclass(frozen=True)
class ImageCommand:
    path: str
    text: str


def parse_image_command(line: str) -> ImageCommand | None:
    """Parse '/image <path> [message]'; the path may be wrapped in double quotes.

    Returns None if the line is not an image command; raises ValueError on bad usage.
    """
    head, _, rest = line.strip().partition(" ")
    if head.lower() != "/image":
        return None
    rest = rest.strip()
    if not rest:
        raise ValueError('Usage: /image <path> [message]   (quote paths that contain spaces)')
    if rest.startswith('"'):
        end = rest.find('"', 1)
        if end == -1:
            raise ValueError("Missing closing quote in the image path.")
        return ImageCommand(path=rest[1:end], text=rest[end + 1 :].strip())
    path, _, text = rest.partition(" ")
    return ImageCommand(path=path, text=text.strip())


def with_attachment(text: str, image_id: str) -> str:
    """Append the attachment note the agent looks for."""
    return f"{text.strip() or DEFAULT_IMAGE_TEXT}\n[image attached: {image_id}]"