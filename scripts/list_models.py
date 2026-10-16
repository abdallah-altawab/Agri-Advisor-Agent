"""List model IDs visible to your API key. Does not call any model (no generation quota)."""

from google import genai

from agri_advisor.config import get_settings


def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.google_api_key.get_secret_value())
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None)
        if actions and "generateContent" not in actions:
            continue
        name = getattr(model, "name", "?")
        print(name.removeprefix("models/"))


if __name__ == "__main__":
    main()