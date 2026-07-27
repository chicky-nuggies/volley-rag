# AGENTS.md

## Environment and commands

- Python `>=3.13`; dependencies and lockfile are managed with `uv`.
- Install/sync dependencies: `uv sync`
- Run the API from the repository root: `uv run uvicorn app.main:app --reload`
- Run the Gradio UI in a second terminal: `uv run python gradio_app.py`
- Run all tests: `uv run python -m unittest discover -s tests -v`
- Run one test:
  `uv run python -m unittest tests.test_langfuse_tracing.ChatTracingTests.test_chat_uses_sources_from_agent_tool_artifact`
- No lint, formatter, type-checker, or build command is configured in this repository.

The live API requires:

- Ollama with `qwen3-embedding:0.6b`
- `GENERATOR_BASE_URL`, `GENERATOR_API_KEY`, and `GENERATOR_MODEL`
- `QDRANT_ENDPOINT` and `QDRANT_API_KEY`
- An existing Qdrant collection named `volleyball_rules_hybrid`

Langfuse tracing uses `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
`LANGFUSE_BASE_URL`. It degrades to a no-op without valid credentials.

## Architecture

- `app/main.py` is the FastAPI entry point. Its lifespan initializes Langfuse,
  opens the SQLite LangGraph checkpointer at `checkpoints/langgraph.sqlite`, and
  stores one shared agent on `app.state.chat_agent`. Run from the repository
  root so the relative checkpoint path resolves correctly.
- `POST /chat` assigns a fresh UUID that serves as both the Langfuse trace seed
  and LangGraph `thread_id`. Requests are therefore single-turn even though
  checkpoints persist. It invokes the agent, extracts the final message, and
  obtains response sources from the latest `search_volleyball_rules`
  `ToolMessage` artifact rather than issuing another retrieval.
- `app/agent.py` builds the LangChain agent around an OpenAI-compatible
  `ChatOpenAI` generator. The model decides when to call
  `search_volleyball_rules`; retrieval is not forced for every message.
  Generator environment variables are read during application startup.
- `app/rag.py` performs hybrid retrieval: Ollama creates 1024-dimensional dense
  vectors, FastEmbed creates BM25 sparse vectors, and Qdrant fuses both searches
  with reciprocal-rank fusion. Qdrant and sparse-model instances are cached.
  `/retrieve` moves this synchronous work into Starlette's thread pool.
- Retrieval has two output shapes. Agent searches return `Source` objects with
  500-character previews for model context and chat citations. The direct
  `/retrieve` route returns `RetrievedChunk` objects containing full text.
  Pydantic request/response contracts and the retrieval limit of 1-20 live in
  `app/schemas.py`.
- `app/observability.py` owns Langfuse lifecycle and observation contexts.
  `/chat` is traced, including nested agent/model/tool activity and retrieval;
  the standalone `/retrieve` endpoint is intentionally untraced.
- `gradio_app.py` is a separate HTTP client for the FastAPI service, not an
  in-process UI wrapper. It uses `urllib` and supports
  `VOLLEY_RAG_API_BASE_URL`, `VOLLEY_RAG_CHAT_API_URL`, and
  `VOLLEY_RAG_RETRIEVE_API_URL`. Its chat callback currently ignores history.
- `vector-store-setup/` contains offline notebooks and generated JSONL used to
  chunk the FIVB rules and populate the Qdrant collection. Runtime startup
  assumes this indexing work has already been completed.
- Tests in `tests/test_langfuse_tracing.py` use `unittest` and mocks. They cover
  endpoint orchestration, tool artifacts, and tracing behavior without live
  Ollama, Qdrant, generator, FastAPI transport, or Gradio integration.
