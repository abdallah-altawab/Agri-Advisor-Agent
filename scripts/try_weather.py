"""Manual check against the real Open-Meteo API (uses no Gemini quota).

Usage: uv run python scripts/try_weather.py "Cairo" --days 5
"""

import argparse
import sys

from agri_advisor.services.errors import ServiceError
from agri_advisor.services.geocoding import geocode
from agri_advisor.services.http import make_client
from agri_advisor.services.weather import fetch_recent_weather


def main() -> int:
    parser = argparse.ArgumentParser(description="Show recent weather for a place.")
    parser.add_argument("place", help='Place name, e.g. "Cairo" or "Springfield, Illinois".')
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    with make_client() as client:
        try:
            location = geocode(client, args.place)
            summary = fetch_recent_weather(client, location, args.days)
        except ServiceError as exc:
            print(f"Error: {exc}")
            return 1

    print(f"{location.label} ({location.latitude:.3f}, {location.longitude:.3f}) tz={location.timezone}")
    print(f"{'date':<12}{'Tmin':>6}{'Tmax':>6}{'RH%':>7}{'rain':>7}{'RH>=90h':>9}")
    for d in summary.days:
        print(
            f"{d.date:<12}{d.temp_min_c:>6.1f}{d.temp_max_c:>6.1f}"
            f"{d.humidity_mean_pct:>7.1f}{d.precipitation_mm:>7.1f}{d.high_humidity_hours:>9}"
        )
    print(
        f"total rain: {summary.total_precipitation_mm:.1f} mm | rainy days: {summary.rainy_days} "
        f"| high-humidity hours: {summary.total_high_humidity_hours}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())