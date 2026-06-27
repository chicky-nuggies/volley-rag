import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gradio as gr


CHAT_API_URL = os.environ.get("VOLLEY_RAG_CHAT_API_URL", "http://127.0.0.1:8000/chat")


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
    body = json.dumps({"message": message}).encode("utf-8")
    request = Request(
        CHAT_API_URL,
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


demo = gr.ChatInterface(
    fn=chat,
    title="Volley RAG Chatbot",
    description="Ask volleyball rules questions using the indexed FIVB rules.",
    examples=[
        "How many contacts does a team have to return the ball?",
        "When does a team rotate?",
        "What is a block contact?",
    ],
)


if __name__ == "__main__":
    demo.launch()
