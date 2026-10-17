"""Shared test helpers."""

import httpx

BERLIN_GEOCODE = {
    "results": [
        {
            "name": "Berlin",
            "latitude": 52.52437,
            "longitude": 13.41053,
            "country": "Germany",
            "admin1": "Berlin",
            "timezone": "Europe/Berlin",
        }
    ]
}


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


def routing_client(weather_payload, geocode_payload=None):
    """HTTP client that answers geocoding and forecast requests from fixed payloads."""

    def handler(request):
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json=BERLIN_GEOCODE if geocode_payload is None else geocode_payload)
        return httpx.Response(200, json=weather_payload)

    return httpx.Client(transport=httpx.MockTransport(handler))