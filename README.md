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
