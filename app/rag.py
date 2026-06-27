import os
from functools import lru_cache

import ollama
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from app.schemas import Source


load_dotenv()

COLLECTION_NAME = "volleyball_rules_hybrid"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_MODEL = "qwen3-embedding:0.6b"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_SIZE = 1024


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_ENDPOINT"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=60,
    )


@lru_cache
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(SPARSE_MODEL)


def to_sparse_vector(sparse_embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )


def embed_dense(text: str) -> list[float]:
    dense_vector = ollama.embed(
        model=DENSE_MODEL,
        input=text,
        truncate=False,
    )["embeddings"][0]
    if len(dense_vector) != DENSE_VECTOR_SIZE:
        raise ValueError(f"Expected dense vector size {DENSE_VECTOR_SIZE}, got {len(dense_vector)}")
    return dense_vector


def embed_sparse_query(text: str) -> models.SparseVector:
    sparse_embedding = next(get_sparse_model().query_embed(text))
    return to_sparse_vector(sparse_embedding)


def hybrid_search(query: str, limit: int = 5, prefetch_limit: int = 20) -> list[Source]:
    result = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=embed_dense(query),
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=embed_sparse_query(query),
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    sources = []
    for hit in result.points:
        payload = hit.payload or {}
        text = str(payload.get("text") or payload.get("raw_text") or "")
        sources.append(
            Source(
                chunk_id=payload.get("chunk_id"),
                source=payload.get("source"),
                headings=payload.get("headings"),
                score=hit.score,
                preview=text[:500],
            )
        )
    return sources


def format_sources_for_tool(sources: list[Source]) -> str:
    if not sources:
        return "No volleyball rule context was found."

    formatted = []
    for index, source in enumerate(sources, start=1):
        heading_text = " > ".join(source.headings or [])
        formatted.append(
            "\n".join(
                [
                    f"Result {index}",
                    f"Chunk ID: {source.chunk_id}",
                    f"Headings: {heading_text or 'None'}",
                    f"Score: {source.score}",
                    f"Text: {source.preview}",
                ]
            )
        )
    return "\n\n".join(formatted)
