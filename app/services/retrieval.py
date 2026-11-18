import re
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Chunk, Document
from app.services.embeddings import embed_query


@dataclass
class Hit:
    chunk_id: int
    document_id: uuid.UUID
    filename: str
    page: int | None
    content: str
    distance: float | None = None  # cosine distance, only set for vector hits


@dataclass
class RetrievalResult:
    hits: list[Hit]
    best_distance: float | None
    refuse: bool


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: merges several ranked id lists into one."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def _or_query(question: str) -> str | None:
    """Turn a question into 'word1 | word2 | ...' for to_tsquery (alphanumerics only = safe)."""
    words = list(dict.fromkeys(re.findall(r"[A-Za-z0-9]{3,}", question.lower())))
    return " | ".join(words) if words else None


def _scope(stmt, user_id: uuid.UUID, document_ids: list[uuid.UUID] | None):
    stmt = stmt.join(Document, Document.id == Chunk.document_id).where(
        Chunk.user_id == user_id, Document.status == "ready"
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    return stmt


_COLUMNS = (Chunk.id, Chunk.document_id, Document.filename, Chunk.page, Chunk.content)


def _vector_search(db: Session, user_id, question, document_ids) -> list[Hit]:
    distance = Chunk.embedding.cosine_distance(embed_query(question))
    stmt = _scope(select(*_COLUMNS, distance.label("distance")), user_id, document_ids)
    rows = db.execute(stmt.order_by(distance).limit(settings.retrieval_candidates)).all()
    return [Hit(*row[:5], distance=float(row[5])) for row in rows]


def _keyword_search(db: Session, user_id, question, document_ids) -> list[Hit]:
    query = _or_query(question)
    if not query:
        return []
    tsq = func.to_tsquery("english", query)
    rank = func.ts_rank_cd(Chunk.content_tsv, tsq)
    stmt = _scope(select(*_COLUMNS), user_id, document_ids).where(Chunk.content_tsv.op("@@")(tsq))
    rows = db.execute(stmt.order_by(rank.desc()).limit(settings.retrieval_candidates)).all()
    return [Hit(*row[:5]) for row in rows]


def retrieve(
    db: Session,
    user_id: uuid.UUID,
    question: str,
    *,
    top_k: int,
    mode: str = "hybrid",
    document_ids: list[uuid.UUID] | None = None,
) -> RetrievalResult:
    vector_hits = _vector_search(db, user_id, question, document_ids) if mode != "keyword" else []
    keyword_hits = _keyword_search(db, user_id, question, document_ids) if mode != "vector" else []

    by_id: dict[int, Hit] = {h.chunk_id: h for h in keyword_hits}
    by_id.update({h.chunk_id: h for h in vector_hits})  # vector hits carry the distance

    if mode == "vector":
        order = [h.chunk_id for h in vector_hits]
    elif mode == "keyword":
        order = [h.chunk_id for h in keyword_hits]
    else:
        order = rrf_fuse([[h.chunk_id for h in vector_hits], [h.chunk_id for h in keyword_hits]])

    hits = [by_id[i] for i in order[:top_k]]
    best = vector_hits[0].distance if vector_hits else None

    if not hits:
        refuse = True
    elif mode == "keyword":
        refuse = False
    else:
        # Relevance gate: if even the closest chunk is far away, don't ask the LLM at all.
        refuse = best is None or best > settings.max_vector_distance
    return RetrievalResult(hits=hits, best_distance=best, refuse=refuse)
