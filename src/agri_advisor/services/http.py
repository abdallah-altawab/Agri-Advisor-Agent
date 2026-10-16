"""Small HTTP helper shared by the data services."""

import logging
from typing import Any

import httpx

from agri_advisor.services.errors import ServiceError

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10.0


def make_client(timeout_s: float = DEFAULT_TIMEOUT_S) -> httpx.Client:
    """Create an HTTP client with an explicit timeout."""
    return httpx.Client(timeout=timeout_s, headers={"User-Agent": "agri-advisor/0.1"})


def get_json(client: httpx.Client, url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET a JSON object. Raises ServiceError (with a safe message) on any failure."""
    try:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException as exc:
        raise ServiceError("The data service timed out. Please try again shortly.") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        log.warning("HTTP %s from %s: %s", status, exc.request.url.host, exc.response.text[:200])
        raise ServiceError(f"The data service returned an error (HTTP {status}).") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise ServiceError("Could not reach the data service.") from exc

    if not isinstance(data, dict):
        raise ServiceError("Unexpected response from the data service.")
    return data