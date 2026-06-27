from fastapi import FastAPI


app = FastAPI(title="Volley RAG API")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"name": "Volley RAG API", "status": "ok"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
