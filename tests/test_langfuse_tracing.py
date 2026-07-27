from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app import agent, main, observability
from app.schemas import ChatRequest, RetrievedChunk, RetrieveRequest, Source


class RecordingObservation:
    def __init__(self) -> None:
        self.updates = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class ChatTracingTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_traces_request_retrieval_and_agent(self) -> None:
        source = Source(
            chunk_id="rule-1",
            source="rules.pdf",
            headings=["Playing the ball"],
            score=0.9,
            preview="A team has three hits.",
        )
        chat_observation = RecordingObservation()
        retrieval_observation = RecordingObservation()
        trace_calls = []

        @contextmanager
        def fake_trace_chat(message: str, request_id: str):
            trace_calls.append(("chat", message, request_id))
            yield chat_observation, "trace-123"

        @contextmanager
        def fake_trace_retrieval(name: str, query: str):
            trace_calls.append((name, query))
            yield retrieval_observation

        fake_agent = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value={"messages": [SimpleNamespace(content="Three hits.")]}
            )
        )
        main.app.state.chat_agent = fake_agent

        with (
            patch("app.main.uuid.uuid4", return_value="request-123"),
            patch("app.main.trace_chat", fake_trace_chat),
            patch("app.main.trace_retrieval", fake_trace_retrieval),
            patch("app.main.create_langfuse_handler", return_value="handler"),
            patch("app.main.run_in_threadpool", AsyncMock(return_value=[source])),
        ):
            response = await main.chat(ChatRequest(message="How many hits?"))

        self.assertEqual(response.answer, "Three hits.")
        self.assertEqual(response.sources, [source])
        self.assertEqual(
            trace_calls,
            [
                ("chat", "How many hits?", "request-123"),
                ("response-source-retrieval", "How many hits?"),
            ],
        )
        self.assertEqual(chat_observation.updates, [{"output": "Three hits."}])
        self.assertEqual(
            retrieval_observation.updates,
            [
                {
                    "output": [source.model_dump()],
                    "metadata": {"resultCount": 1},
                }
            ],
        )
        fake_agent.ainvoke.assert_awaited_once_with(
            {"messages": [{"role": "user", "content": "How many hits?"}]},
            config={
                "callbacks": ["handler"],
                "configurable": {"thread_id": "request-123"},
            },
        )

    async def test_retrieve_endpoint_does_not_start_tracing(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="rule-1",
            source="rules.pdf",
            headings=["Playing the ball"],
            score=0.9,
            text="A team has three hits.",
        )

        with (
            patch("app.main.trace_chat", side_effect=AssertionError("unexpected trace")),
            patch(
                "app.main.trace_retrieval",
                side_effect=AssertionError("unexpected trace"),
            ),
            patch("app.main.run_in_threadpool", AsyncMock(return_value=[chunk])) as run,
        ):
            response = await main.retrieve(RetrieveRequest(query="team hits", limit=3))

        self.assertEqual(response.query, "team hits")
        self.assertEqual(response.chunks, [chunk])
        run.assert_awaited_once_with(main.retrieve_chunks, "team hits", 3)


class ToolTracingTests(unittest.TestCase):
    def test_agent_rule_retrieval_records_sources(self) -> None:
        source = Source(
            chunk_id="rule-1",
            source="rules.pdf",
            headings=["Playing the ball"],
            score=0.9,
            preview="A team has three hits.",
        )
        observation = RecordingObservation()
        trace_calls = []

        @contextmanager
        def fake_trace_retrieval(name: str, query: str):
            trace_calls.append((name, query))
            yield observation

        with (
            patch("app.agent.trace_retrieval", fake_trace_retrieval),
            patch("app.agent.hybrid_search", return_value=[source]),
        ):
            result = agent.search_volleyball_rules.func("team hits")

        self.assertEqual(trace_calls, [("agent-rule-retrieval", "team hits")])
        self.assertIn("A team has three hits.", result)
        self.assertEqual(
            observation.updates,
            [
                {
                    "output": [source.model_dump()],
                    "metadata": {"resultCount": 1},
                }
            ],
        )


class ObservabilityConfigurationTests(unittest.TestCase):
    def test_chat_trace_uses_explicit_io_and_stable_attributes(self) -> None:
        observation = RecordingObservation()
        client = Mock()
        client.start_as_current_observation.return_value = _yield(observation)

        with (
            patch("app.observability.get_client", return_value=client),
            patch(
                "app.observability.Langfuse.create_trace_id",
                return_value="trace-123",
            ) as create_trace_id,
            patch("app.observability.propagate_attributes", return_value=_yield(None)) as propagate,
        ):
            with observability.trace_chat("How many hits?", "request-123") as value:
                self.assertEqual(value, (observation, "trace-123"))

        create_trace_id.assert_called_once_with(seed="request-123")
        client.start_as_current_observation.assert_called_once_with(
            name="chat-response",
            as_type="agent",
            input="How many hits?",
            metadata={"requestId": "request-123", "route": "/chat"},
            trace_context={"trace_id": "trace-123"},
        )
        propagate.assert_called_once_with(
            trace_name="chat-response",
            tags=["volley-rag", "chat"],
        )

    def test_langfuse_lifecycle_uses_singleton_client(self) -> None:
        client = Mock()

        with patch("app.observability.get_client", return_value=client) as get_client:
            observability.initialize_langfuse()
            observability.shutdown_langfuse()

        self.assertEqual(get_client.call_count, 2)
        client.shutdown.assert_called_once_with()


@contextmanager
def _yield(value):
    yield value


if __name__ == "__main__":
    unittest.main()
