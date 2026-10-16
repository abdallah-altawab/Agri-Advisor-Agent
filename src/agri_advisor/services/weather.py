"""Recent weather summaries for plant-disease risk reasoning (Open-Meteo, no API key).

Values are model-based for a grid cell around the location,
NOT readings from sensors on the farm.
"""

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

import httpx

from agri_advisor.services.errors import ServiceError
from agri_advisor.services.geocoding import Location
from agri_advisor.services.http import get_json

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_DAYS = 14
HIGH_HUMIDITY_PCT = 90.0  # placeholder threshold; to be replaced by sourced values later
RAINY_DAY_MM = 1.0


@dataclass(frozen=True)
class DailyWeather:
    """Aggregates for one local calendar day."""

    date: str
    temp_mean_c: float
    temp_min_c: float
    temp_max_c: float
    humidity_mean_pct: float
    precipitation_mm: float
    high_humidity_hours: int


@dataclass(frozen=True)
class WeatherSummary:
    """Weather over the last N complete days for one location."""

    location: Location
    days: tuple[DailyWeather, ...]

    @property
    def total_precipitation_mm(self) -> float:
        return round(sum(d.precipitation_mm for d in self.days), 1)

    @property
    def rainy_days(self) -> int:
        return sum(1 for d in self.days if d.precipitation_mm >= RAINY_DAY_MM)

    @property
    def total_high_humidity_hours(self) -> int:
        return sum(d.high_humidity_hours for d in self.days)


def fetch_recent_weather(
    client: httpx.Client, location: Location, days: int = 7
) -> WeatherSummary:
    """Fetch hourly data and aggregate the last `days` complete local days."""
    if not 1 <= days <= MAX_DAYS:
        raise ServiceError(f"days must be between 1 and {MAX_DAYS}.")

    data = get_json(
        client,
        FORECAST_URL,
        {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation",
            "past_days": days,
            "forecast_days": 1,
            "timezone": "auto",
        },
    )
    return _summarise(location, data)


def _summarise(location: Location, data: dict) -> WeatherSummary:
    hourly = data.get("hourly")
    if not isinstance(hourly, dict):
        raise ServiceError("Unexpected response from the weather service.")
    try:
        times = hourly["time"]
        temps = hourly["temperature_2m"]
        hums = hourly["relative_humidity_2m"]
        rains = hourly["precipitation"]
    except KeyError as exc:
        raise ServiceError("Unexpected response from the weather service.") from exc
    if not (len(times) == len(temps) == len(hums) == len(rains)):
        raise ServiceError("Inconsistent response from the weather service.")

    by_day: dict[str, list[tuple]] = defaultdict(list)
    for stamp, temp, hum, rain in zip(times, temps, hums, rains):
        by_day[str(stamp)[:10]].append((temp, hum, rain))

    # The last date is "today" and is partly forecast: keep complete past days only.
    past_dates = sorted(by_day)[:-1]

    daily: list[DailyWeather] = []
    for date in past_dates:
        rows = by_day[date]
        t = [r[0] for r in rows if r[0] is not None]
        h = [r[1] for r in rows if r[1] is not None]
        p = [r[2] for r in rows if r[2] is not None]
        if not t or not h:
            continue  # skip days without usable data
        daily.append(
            DailyWeather(
                date=date,
                temp_mean_c=round(mean(t), 1),
                temp_min_c=min(t),
                temp_max_c=max(t),
                humidity_mean_pct=round(mean(h), 1),
                precipitation_mm=round(sum(p), 1),
                high_humidity_hours=sum(1 for value in h if value >= HIGH_HUMIDITY_PCT),
            )
        )

    if not daily:
        raise ServiceError("No recent weather data available for this location.")
    return WeatherSummary(location=location, days=tuple(daily))