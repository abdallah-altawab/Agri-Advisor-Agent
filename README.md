# Agri Advisor Agent

Agentic tomato-disease advisor. Upload a leaf photo and get a diagnosis,
a confidence-aware explanation, and practical advice grounded in agronomy references.

## Planned features
- **Context-aware diagnosis:** when the image classifier cannot separate two diseases,
  the agent weighs weather and location evidence, or asks follow-up questions.
- **Grounded advice:** recommendations retrieved from curated references (RAG) with citations.
- **Human in the loop:** sensitive actions require explicit approval.

## Status
Work in progress. Currently: project skeleton.

## Setup
1. Install [uv](https://docs.astral.sh/uv/)
2. `uv python install 3.11`
3. `uv sync`
4. Copy `.env.example` to `.env` and fill in your values (never commit `.env`).
5. Run everything through uv, e.g. `uv run pytest`.