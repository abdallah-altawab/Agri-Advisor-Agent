"""LLM factory: the single place where the chat model is constructed."""

from langchain_google_genai import ChatGoogleGenerativeAI

from agri_advisor.config import Settings, get_settings


def build_llm(settings: Settings | None = None) -> ChatGoogleGenerativeAI:
    """Build the chat model with an explicit timeout and bounded retries."""
    s = settings or get_settings()
    return ChatGoogleGenerativeAI(
        model=s.llm_model,
        api_key=s.google_api_key,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
    )