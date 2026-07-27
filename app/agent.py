import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.observability import trace_retrieval
from app.rag import format_sources_for_tool, hybrid_search


SYSTEM_PROMPT = """You are a volleyball rules assistant.

Answer volleyball-related questions using the official FIVB rule context available
through the search_volleyball_rules tool.

Guidelines:
- Use the tool when the user asks about rules, gameplay, scoring, positions,
  faults, rotations, contacts, teams, refereeing, or official volleyball details.
- Prefer retrieved rule context over general knowledge.
- If the context is insufficient, say what is missing instead of inventing details.
- Keep answers concise and practical.
"""


@tool(response_format="content_and_artifact")
def search_volleyball_rules(query: str) -> tuple[str, list[dict]]:
    """Search official volleyball rules context for a user question."""
    with trace_retrieval("agent-rule-retrieval", query) as observation:
        sources = hybrid_search(query)
        source_data = [source.model_dump() for source in sources]
        observation.update(
            output=source_data,
            metadata={"resultCount": len(sources)},
        )
    return format_sources_for_tool(sources), source_data


async def setup_checkpointer(checkpointer: AsyncSqliteSaver) -> None:
    await checkpointer.setup()


def build_agent(checkpointer: AsyncSqliteSaver):
    generator = ChatOpenAI(
        base_url=os.environ["GENERATOR_BASE_URL"],
        api_key=os.environ["GENERATOR_API_KEY"],
        model=os.environ["GENERATOR_MODEL"],
    )
    return create_agent(
        model=generator,
        tools=[search_volleyball_rules],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
