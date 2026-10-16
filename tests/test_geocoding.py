import httpx
import pytest

from agri_advisor.services.errors import ServiceError
from agri_advisor.services.geocoding import geocode


def client_with(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_geocode_returns_best_match():
    def handler(request):
        assert request.url.params["name"] == "Berlin"
        return httpx.Response(
            200,
            json={
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
            },
        )

    location = geocode(client_with(handler), "Berlin")
    assert location.label == "Berlin, Germany"  # repeated admin1 is dropped
    assert location.latitude == pytest.approx(52.52437)
    assert location.timezone == "Europe/Berlin"


def test_no_results_key_raises_service_error():
    client = client_with(lambda request: httpx.Response(200, json={}))
    with pytest.raises(ServiceError, match="No place found"):
        geocode(client, "Nowhereville")


def test_too_short_query_makes_no_request():
    def handler(request):
        raise AssertionError("no HTTP request expected")

    with pytest.raises(ServiceError, match="place name"):
        geocode(client_with(handler), " a ")


def test_http_error_becomes_service_error():
    client = client_with(lambda request: httpx.Response(500, json={"reason": "boom"}))
    with pytest.raises(ServiceError, match="HTTP 500"):
        geocode(client, "Berlin")


def test_timeout_becomes_service_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ServiceError, match="timed out"):
        geocode(client_with(handler), "Berlin")