"""Stage 1 smoke test: one tiny request to verify key, model name and timeouts."""

import logging
import sys
import time

from pydantic import ValidationError

from agri_advisor.config import get_settings
from agri_advisor.llm import build_llm
from agri_advisor.logging_setup import configure_logging
from agri_advisor.messages import content_to_text

log = logging.getLogger("smoke_test")


def main() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        missing = [".".join(map(str, e["loc"])) for e in exc.errors()]
        print(f"Invalid or missing settings: {missing}. Check your .env file.")
        return 2

    configure_logging(settings.log_level)
    log.info("Using model=%s", settings.llm_model)

    llm = build_llm(settings)
    start = time.perf_counter()
    try:
        response = llm.invoke("Reply with exactly one word: ready")
    except Exception as exc:  # noqa: BLE001 - top-level boundary, report and exit
        log.error("LLM call failed: %s: %s", type(exc).__name__, exc)
        return 1

    print("raw content type :", type(response.content).__name__)
    print("text             :", content_to_text(response.content))
    print("usage            :", response.usage_metadata)
    print(f"elapsed          : {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())