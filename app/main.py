from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.agent import CHECKPOINT_DB_PATH, build_agent, setup_checkpointer
from app.rag import hybrid_search
from app.schemas import ChatRequest, ChatResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        await setup_checkpointer(checkpointer)
        app.state.chat_agent = build_agent(checkpointer)
        yield


app = FastAPI(title="Volley RAG API", lifespan=lifespan)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"name": "Volley RAG API", "status": "ok"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    sources = await run_in_threadpool(hybrid_search, request.message)
    agent_result = await app.state.chat_agent.ainvoke(
        {"messages": [{"role": "user", "content": request.message}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    return ChatResponse(answer=extract_answer(agent_result), sources=sources)
