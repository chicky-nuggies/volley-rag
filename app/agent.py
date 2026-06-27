from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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


@tool
def search_volleyball_rules(query: str) -> str:
    """Search official volleyball rules context for a user question."""
    return format_sources_for_tool(hybrid_search(query))


async def setup_checkpointer(checkpointer: AsyncSqliteSaver) -> None:
    await checkpointer.setup()


def build_agent(checkpointer: AsyncSqliteSaver):
    return create_agent(
        model="ollama:gemma4:e2b",
        tools=[search_volleyball_rules],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
