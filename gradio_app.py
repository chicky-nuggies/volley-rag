import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gradio as gr


API_BASE_URL = os.environ.get("VOLLEY_RAG_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
CHAT_API_URL = os.environ.get("VOLLEY_RAG_CHAT_API_URL", f"{API_BASE_URL}/chat")
RETRIEVE_API_URL = os.environ.get("VOLLEY_RAG_RETRIEVE_API_URL", f"{API_BASE_URL}/retrieve")


def format_source(source: dict, index: int) -> str:
    headings = " > ".join(source.get("headings") or [])
    title = headings or source.get("source") or "Retrieved source"
    chunk = source.get("chunk_id") or "unknown chunk"
    preview = str(source.get("preview") or "").strip().replace("\n", " ")
    return f"{index}. {title} · {chunk}\n   {preview}"


def format_response(answer: str, sources: list[dict]) -> str:
    if not sources:
        return answer

    source_text = "\n\n".join(format_source(source, index) for index, source in enumerate(sources[:3], start=1))
    return f"{answer}\n\nSources:\n{source_text}"


def post_chat(message: str) -> dict:
    return post_json(CHAT_API_URL, {"message": message})


def post_retrieve(query: str, limit: int) -> dict:
    return post_json(RETRIEVE_API_URL, {"query": query, "limit": limit})


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(message: str, history: list[dict]) -> str:
    try:
        response = post_chat(message)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        return f"Chat API returned HTTP {exc.code}.\n\n{details}"
    except URLError as exc:
        return (
            f"Could not reach the chat API at {CHAT_API_URL}.\n\n"
            "Start it with:\n"
            "uvicorn app.main:app --reload\n\n"
            f"Details: {exc.reason}"
        )

    return format_response(response.get("answer", ""), response.get("sources", []))


def format_chunk(chunk: dict, index: int) -> str:
    headings = " > ".join(chunk.get("headings") or [])
    title = headings or chunk.get("source") or "Retrieved chunk"
    chunk_id = chunk.get("chunk_id") or "unknown chunk"
    score = chunk.get("score")
    text = str(chunk.get("text") or "").strip()
    return f"## {index}. {title}\n\n**Chunk:** `{chunk_id}`  \n**Score:** `{score}`\n\n{text}"


def retrieve(query: str, limit: int) -> str:
    try:
        response = post_retrieve(query, int(limit))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        return f"Retrieve API returned HTTP {exc.code}.\n\n{details}"
    except URLError as exc:
        return (
            f"Could not reach the retrieve API at {RETRIEVE_API_URL}.\n\n"
            "Start it with:\n"
            "uvicorn app.main:app --reload\n\n"
            f"Details: {exc.reason}"
        )

    chunks = response.get("chunks", [])
    if not chunks:
        return "No chunks found."
    return "\n\n---\n\n".join(format_chunk(chunk, index) for index, chunk in enumerate(chunks, start=1))


with gr.Blocks(title="Volley RAG") as demo:
    gr.Markdown("# Volley RAG")
    gr.Markdown("Ask volleyball rules questions or inspect retrieved chunks directly.")

    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=chat,
            title="Volley RAG Chatbot",
            description="Ask volleyball rules questions using the indexed FIVB rules.",
            examples=[
                "How many contacts does a team have to return the ball?",
                "When does a team rotate?",
                "What is a block contact?",
            ],
        )

    with gr.Tab("Retrieve Chunks"):
        query = gr.Textbox(
            label="Query",
            placeholder="Example: team hits when returning the ball",
            lines=2,
        )
        limit = gr.Slider(
            label="Number of chunks",
            minimum=1,
            maximum=20,
            value=5,
            step=1,
        )
        retrieve_button = gr.Button("Retrieve")
        output = gr.Markdown()
        retrieve_button.click(fn=retrieve, inputs=[query, limit], outputs=output)


if __name__ == "__main__":
    demo.launch()
