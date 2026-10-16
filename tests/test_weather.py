import httpx
import pytest

from agri_advisor.services.errors import ServiceError
from agri_advisor.services.geocoding import Location
from agri_advisor.services.weather import fetch_recent_weather

BERLIN = Location(
    name="Berlin", latitude=52.52, longitude=13.41, country="Germany", timezone="Europe/Berlin"
)


def make_hourly(dates):
    """Per day: temp 20/21 alternating, RH 95 for 12h then 70, 2 mm rain at hour 0."""
    times, temps, hums, rains = [], [], [], []
    for date in dates:
        for hour in range(24):
            times.append(f"{date}T{hour:02d}:00")
            temps.append(20.0 + (hour % 2))
            hums.append(95.0 if hour < 12 else 70.0)
            rains.append(2.0 if hour == 0 else 0.0)
    return {
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "relative_humidity_2m": hums,
            "precipitation": rains,
        }
    }


def client_returning(payload, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_summarises_complete_past_days_and_drops_today():
    seen = []
    payload = make_hourly(["2026-10-13", "2026-10-14", "2026-10-15"])  # last = today
    summary = fetch_recent_weather(client_returning(payload, seen), BERLIN, days=2)

    assert [d.date for d in summary.days] == ["2026-10-13", "2026-10-14"]
    first = summary.days[0]
    assert (first.temp_min_c, first.temp_max_c, first.temp_mean_c) == (20.0, 21.0, 20.5)
    assert first.humidity_mean_pct == 82.5
    assert first.precipitation_mm == 2.0
    assert first.high_humidity_hours == 12
    assert summary.total_precipitation_mm == 4.0
    assert summary.rainy_days == 2
    assert summary.total_high_humidity_hours == 24

    params = seen[0].url.params
    assert params["past_days"] == "2"
    assert params["forecast_days"] == "1"
    assert params["timezone"] == "auto"


def test_days_without_usable_values_are_skipped():
    payload = make_hourly(["2026-10-13", "2026-10-14", "2026-10-15"])
    # Humidity missing for the whole second day.
    payload["hourly"]["relative_humidity_2m"][24:48] = [None] * 24
    summary = fetch_recent_weather(client_returning(payload), BERLIN, days=2)
    assert [d.date for d in summary.days] == ["2026-10-13"]


@pytest.mark.parametrize("days", [0, 15])
def test_invalid_days_make_no_request(days):
    def handler(request):
        raise AssertionError("no HTTP request expected")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceError, match="days must be"):
        fetch_recent_weather(client, BERLIN, days=days)


def test_unexpected_payload_raises_service_error():
    with pytest.raises(ServiceError, match="Unexpected response"):
        fetch_recent_weather(client_returning({}), BERLIN, days=2)


def test_mismatched_lengths_raise_service_error():
    payload = make_hourly(["2026-10-14", "2026-10-15"])
    payload["hourly"]["precipitation"].pop()
    with pytest.raises(ServiceError, match="Inconsistent"):
        fetch_recent_weather(client_returning(payload), BERLIN, days=1)