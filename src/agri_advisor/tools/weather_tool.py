"""LangChain tool wrapping the weather data service."""

import httpx
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agri_advisor.services.errors import ServiceError
from agri_advisor.services.geocoding import geocode
from agri_advisor.services.weather import HIGH_HUMIDITY_PCT, WeatherSummary, fetch_recent_weather


class RecentWeatherArgs(BaseModel):
    """Arguments of the get_recent_weather tool."""

    place: str = Field(
        description="City or town name, optionally with a country, e.g. 'Cairo, Egypt'."
    )
    days: int = Field(
        default=7,
        description="How many complete past days to summarise (1 to 14).",
    )


def format_weather_summary(summary: WeatherSummary) -> str:
    """Render a summary as compact text for the LLM."""
    loc = summary.location
    first, last = summary.days[0].date, summary.days[-1].date
    lines = [
        f"Location: {loc.label} ({loc.latitude:.2f}, {loc.longitude:.2f})",
        "Source: Open-Meteo model data for the surrounding grid cell, "
        "not on-farm sensors. Today is excluded.",
        f"Period: {first} to {last} ({len(summary.days)} days with data)",
        f"date | Tmin C | Tmax C | mean RH % | rain mm | hours RH>={HIGH_HUMIDITY_PCT:.0f}%",
    ]
    for d in summary.days:
        lines.append(
            f"{d.date} | {d.temp_min_c} | {d.temp_max_c} | {d.humidity_mean_pct} "
            f"| {d.precipitation_mm} | {d.high_humidity_hours}"
        )
    lines.append(
        f"Totals: rain {summary.total_precipitation_mm} mm, {summary.rainy_days} rainy day(s), "
        f"{summary.total_high_humidity_hours} hour(s) with RH >= {HIGH_HUMIDITY_PCT:.0f}%."
    )
    return "\n".join(lines)


def build_weather_tool(client: httpx.Client) -> BaseTool:
    """Create the weather tool bound to an HTTP client (injected for testability)."""

    @tool("get_recent_weather", args_schema=RecentWeatherArgs)
    def get_recent_weather(place: str, days: int = 7) -> str:
        """Get a summary of the last few days of weather (temperature, humidity, rain) for a place.

        Use it when recent weather could help judge which plant disease is more plausible,
        or when the user asks about recent weather. A place name is required: ask the user
        if they have not given one. The data is regional model data for the area,
        not a reading from the user's own field or greenhouse.
        """
        try:
            location = geocode(client, place)
            summary = fetch_recent_weather(client, location, days)
        except ServiceError as exc:
            return f"Weather lookup failed: {exc}"
        return format_weather_summary(summary)

    return get_recent_weather