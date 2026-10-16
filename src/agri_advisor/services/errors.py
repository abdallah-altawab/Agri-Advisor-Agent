"""Errors raised by external data services."""


class ServiceError(Exception):
    """A failure in an external data service.

    The message is written to be safe to show to end users (and to an LLM):
    it never contains URLs, keys or internal details.
    """