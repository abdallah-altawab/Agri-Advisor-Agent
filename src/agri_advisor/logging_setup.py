"""Logging configuration shared by all entry points."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging and silence chatty libraries."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # HTTP client logs can include full request URLs; keep them out of our logs.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # google-genai logs an informational line on every call ("AFC is enabled ...").
    # The retry messages (google_genai._api_client) stay visible on purpose.
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)