from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class Source(BaseModel):
    chunk_id: str | None = None
    source: str | None = None
    headings: list[str] | None = None
    score: float | None = None
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
