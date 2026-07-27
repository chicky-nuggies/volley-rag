from contextlib import contextmanager
from typing import Any, Iterator

from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler


@contextmanager
def trace_chat(message: str, request_id: str) -> Iterator[tuple[Any, str]]:
    langfuse = get_client()
    trace_id = Langfuse.create_trace_id(seed=request_id)
    with langfuse.start_as_current_observation(
        name="chat-response",
        as_type="agent",
        input=message,
        metadata={"requestId": request_id, "route": "/chat"},
        trace_context={"trace_id": trace_id},
    ) as observation:
        with propagate_attributes(
            trace_name="chat-response",
            tags=["volley-rag", "chat"],
        ):
            yield observation, trace_id


@contextmanager
def trace_retrieval(name: str, query: str) -> Iterator[Any]:
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        name=name,
        as_type="retriever",
        input=query,
    ) as observation:
        yield observation


def create_langfuse_handler() -> CallbackHandler:
    return CallbackHandler()


def initialize_langfuse() -> None:
    get_client()


def shutdown_langfuse() -> None:
    get_client().shutdown()
