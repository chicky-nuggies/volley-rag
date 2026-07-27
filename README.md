# Volley RAG

Minimal FastAPI base app for the volleyball rules RAG project.

## Run the API

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

Chat:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many contacts does a team have to return the ball?"}'
```

Retrieve chunks without calling the LLM:

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "team hits returning ball", "limit": 3}'
```

The chat endpoint expects:

- Ollama running locally with `qwen3-embedding:0.6b` for dense query embeddings.
- `GENERATOR_BASE_URL`, `GENERATOR_API_KEY`, and `GENERATOR_MODEL` configured for
  the generator LLM. For Vercel AI Gateway, use
  `https://ai-gateway.vercel.sh/v1` and a fully qualified model ID such as
  `anthropic/claude-sonnet-4.6`.
- `QDRANT_API_KEY` and `QDRANT_ENDPOINT` set in `.env`.
- The `volleyball_rules_hybrid` Qdrant collection already populated.

## Langfuse tracing

The `/chat` endpoint sends one trace per request to Langfuse. Each trace includes
the user question, answer, LangGraph and model calls, tool calls, and both rule
retrievals used by the endpoint. The standalone `/retrieve` endpoint is not
traced.

Add your Langfuse project credentials to `.env`:

```text
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Use the base URL for your Langfuse Cloud region or self-hosted deployment.
Without valid Langfuse credentials, tracing becomes a no-op and the chatbot
continues to work.

Trace payloads contain full user questions, model prompts and answers, and the
retrieved rule previews shown to the model. Do not send sensitive information
unless the configured Langfuse deployment is approved to store it.

## Run the Gradio chat UI

Start the API first:

```bash
uv run uvicorn app.main:app --reload
```

In another terminal, start Gradio:

```bash
uv run python gradio_app.py
```

The Gradio UI has two tabs:

- `Chat`: asks the LLM and shows sources.
- `Retrieve Chunks`: embeds the query and returns matching chunks directly.

By default, Gradio calls:

```text
http://127.0.0.1:8000
```

Override it with `VOLLEY_RAG_API_BASE_URL` if needed.
