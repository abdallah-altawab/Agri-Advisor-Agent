"""Logging configuration shared by all entry points."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging and silence chatty HTTP libraries."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # HTTP client logs can include full request URLs; keep them out of our logs.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)