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

The chat endpoint expects:

- Ollama running locally with `gemma4:e2b` and `qwen3-embedding:0.6b`.
- `QDRANT_API_KEY` and `QDRANT_ENDPOINT` set in `.env`.
- The `volleyball_rules_hybrid` Qdrant collection already populated.

## Run the Gradio chat UI

Start the API first:

```bash
uv run uvicorn app.main:app --reload
```

In another terminal, start Gradio:

```bash
uv run python gradio_app.py
```

By default, Gradio calls:

```text
http://127.0.0.1:8000/chat
```

Override it with `VOLLEY_RAG_CHAT_API_URL` if needed.
