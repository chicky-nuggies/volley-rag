from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import uuid

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.agent import build_agent, setup_checkpointer
from app.observability import (
    create_langfuse_handler,
    initialize_langfuse,
    shutdown_langfuse,
    trace_chat,
    trace_retrieval,
)
from app.rag import hybrid_search, retrieve_chunks
from app.schemas import ChatRequest, ChatResponse, RetrieveRequest, RetrieveResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


CHECKPOINT_DB_PATH = Path("checkpoints/langgraph.sqlite")
logger = logging.getLogger("uvicorn.error")


def extract_answer(agent_result: dict) -> str:
    messages = agent_result.get("messages", [])
    if not messages:
        return ""

    final_message = messages[-1]
    content = getattr(final_message, "content", final_message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def extract_tool_calls(agent_result: dict) -> list[dict]:
    tool_calls = []
    for message in agent_result.get("messages", []):
        for tool_call in getattr(message, "tool_calls", None) or []:
            tool_calls.append(tool_call)
    return tool_calls


def log_tool_calls(agent_result: dict) -> None:
    for tool_call in extract_tool_calls(agent_result):
        logger.info(
            "Chat agent tool call: tool=%s id=%s args=%s",
            tool_call.get("name"),
            tool_call.get("id"),
            json.dumps(tool_call.get("args", {}), ensure_ascii=False),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    initialize_langfuse()
    try:
        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
            await setup_checkpointer(checkpointer)
            app.state.chat_agent = build_agent(checkpointer)
            yield
    finally:
        shutdown_langfuse()


app = FastAPI(title="Volley RAG API", lifespan=lifespan)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"name": "Volley RAG API", "status": "ok"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    request_id = str(uuid.uuid4())
    with trace_chat(request.message, request_id) as (chat_observation, trace_id):
        with trace_retrieval(
            "response-source-retrieval", request.message
        ) as retrieval_observation:
            sources = await run_in_threadpool(hybrid_search, request.message)
            retrieval_observation.update(
                output=[source.model_dump() for source in sources],
                metadata={"resultCount": len(sources)},
            )

        agent_result = await app.state.chat_agent.ainvoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config={
                "callbacks": [create_langfuse_handler()],
                "configurable": {"thread_id": request_id},
            },
        )
        log_tool_calls(agent_result)
        answer = extract_answer(agent_result)
        chat_observation.update(output=answer)

    logger.info("Chat trace: request_id=%s trace_id=%s", request_id, trace_id)
    return ChatResponse(answer=answer, sources=sources)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    chunks = await run_in_threadpool(retrieve_chunks, request.query, request.limit)
    return RetrieveResponse(query=request.query, chunks=chunks)
