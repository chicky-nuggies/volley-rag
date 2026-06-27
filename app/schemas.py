from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    chunk_id: str | None = None
    source: str | None = None
    headings: list[str] | None = None
    score: float | None = None
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class RetrievedChunk(BaseModel):
    chunk_id: str | None = None
    source: str | None = None
    headings: list[str] | None = None
    score: float | None = None
    text: str


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
