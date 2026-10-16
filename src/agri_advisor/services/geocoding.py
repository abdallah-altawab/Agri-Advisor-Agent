"""Place name -> coordinates, using the Open-Meteo Geocoding API (no API key)."""

from dataclasses import dataclass

import httpx

from agri_advisor.services.errors import ServiceError
from agri_advisor.services.http import get_json

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
MIN_QUERY_LENGTH = 2


@dataclass(frozen=True)
class Location:
    """A resolved place."""

    name: str
    latitude: float
    longitude: float
    country: str | None = None
    admin1: str | None = None
    timezone: str | None = None

    @property
    def label(self) -> str:
        """Human-readable label without repeated parts, e.g. 'Berlin, Germany'."""
        parts: list[str] = []
        for part in (self.name, self.admin1, self.country):
            if part and part not in parts:
                parts.append(part)
        return ", ".join(parts)


def geocode(client: httpx.Client, query: str) -> Location:
    """Resolve a place name to its best match.

    The query may include a qualifier, e.g. "Springfield, Illinois".
    Raises ServiceError if nothing matches.
    """
    query = query.strip()
    if len(query) < MIN_QUERY_LENGTH:
        raise ServiceError("Please provide a place name (at least 2 characters).")

    data = get_json(
        client,
        GEOCODING_URL,
        {"name": query, "count": 1, "language": "en", "format": "json"},
    )
    results = data.get("results") or []  # the key is omitted when nothing matches
    if not results:
        raise ServiceError(f"No place found for '{query}'. Try a nearby city or add the country.")

    top = results[0]
    try:
        return Location(
            name=top["name"],
            latitude=float(top["latitude"]),
            longitude=float(top["longitude"]),
            country=top.get("country"),
            admin1=top.get("admin1"),
            timezone=top.get("timezone"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ServiceError("Unexpected response from the geocoding service.") from exc