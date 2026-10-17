from helpers import make_hourly, routing_client

from agri_advisor.tools.weather_tool import build_weather_tool


def test_weather_tool_returns_compact_summary():
    payload = make_hourly(["2026-10-13", "2026-10-14", "2026-10-15"])
    tool = build_weather_tool(routing_client(payload))

    output = tool.invoke({"place": "Berlin", "days": 2})

    assert "Berlin, Germany" in output
    assert "2026-10-13" in output and "2026-10-14" in output
    assert "2026-10-15" not in output  # today is excluded
    assert "Totals:" in output
    assert "not on-farm sensors" in output


def test_weather_tool_returns_safe_message_on_service_error():
    tool = build_weather_tool(routing_client({}, geocode_payload={}))

    output = tool.invoke({"place": "Nowhereville"})

    assert output.startswith("Weather lookup failed:")
    assert "No place found" in output