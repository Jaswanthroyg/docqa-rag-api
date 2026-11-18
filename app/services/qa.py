"""Orchestrates one question: cache -> retrieve -> (refuse | LLM) -> persist -> events."""
import re
import time
import uuid
from collections.abc import Iterator

from app.core.config import settings
from app.db import SessionLocal
from app.models import Conversation, Message, QueryLog
from app.schemas import AskRequest
from app.services import cache, retrieval
from app.services.llm import REFUSAL, SYSTEM_PROMPT, LLMError, Usage, build_user_message, get_llm

_CITE_RE = re.compile(r"\[(\d+)\]")


def _citations(answer: str, hits: list[retrieval.Hit]) -> list[dict]:
    referenced = sorted({int(n) for n in _CITE_RE.findall(answer) if 1 <= int(n) <= len(hits)})
    indices = referenced or ([] if answer.startswith(REFUSAL) else list(range(1, len(hits) + 1)))
    return [
        {
            "index": i,
            "document_id": str(hits[i - 1].document_id),
            "filename": hits[i - 1].filename,
            "page": hits[i - 1].page,
            "snippet": hits[i - 1].content[:240],
        }
        for i in indices
    ]


def _retrieved_public(hits: list[retrieval.Hit]) -> list[dict]:
    return [
        {
            "index": i,
            "chunk_id": h.chunk_id,
            "document_id": str(h.document_id),
            "filename": h.filename,
            "page": h.page,
            "distance": None if h.distance is None else round(h.distance, 4),
            "content": h.content,
        }
        for i, h in enumerate(hits, start=1)
    ]


def run(user_id: uuid.UUID, req: AskRequest) -> Iterator[dict]:
    """Yield {"type": "token"|"error"|"done", ...} events. Used by both JSON and SSE endpoints."""
    started = time.perf_counter()
    top_k = req.top_k or settings.retrieval_top_k
    doc_ids = [str(d) for d in (req.document_ids or [])]
    key = cache.make_key(str(user_id), req.question, req.mode, top_k, doc_ids, cache.corpus_version(str(user_id)))

    usage = Usage()
    retrieval_ms = 0
    chunk_ids: list[int] = []
    cached = None if req.no_cache else cache.get_json(key)

    if cached:
        answer, citations, retrieved = cached["answer"], cached["citations"], cached["retrieved"]
        yield {"type": "token", "text": answer}
    else:
        with SessionLocal() as db:
            t0 = time.perf_counter()
            result = retrieval.retrieve(
                db, user_id, req.question, top_k=top_k, mode=req.mode, document_ids=req.document_ids
            )
            retrieval_ms = int((time.perf_counter() - t0) * 1000)
        chunk_ids = [h.chunk_id for h in result.hits]
        retrieved = _retrieved_public(result.hits)

        if result.refuse:
            answer, citations = REFUSAL, []
            yield {"type": "token", "text": answer}
        else:
            parts: list[str] = []
            try:
                for token in get_llm().stream(SYSTEM_PROMPT, build_user_message(req.question, result.hits), usage):
                    parts.append(token)
                    yield {"type": "token", "text": token}
            except LLMError as exc:
                yield {"type": "error", "message": str(exc)}
                return
            answer = "".join(parts).strip()
            citations = _citations(answer, result.hits)
        cache.set_json(key, {"answer": answer, "citations": citations, "retrieved": retrieved})

    refused = answer.startswith(REFUSAL)
    latency_ms = int((time.perf_counter() - started) * 1000)
    cost = (usage.input_tokens * settings.price_in_per_mtok + usage.output_tokens * settings.price_out_per_mtok) / 1e6

    with SessionLocal() as db:
        conversation_id = req.conversation_id
        if conversation_id is None:
            conversation = Conversation(user_id=user_id, title=req.question[:120])
            db.add(conversation)
            db.flush()
            conversation_id = conversation.id
        db.add(Message(conversation_id=conversation_id, role="user", content=req.question))
        assistant = Message(conversation_id=conversation_id, role="assistant", content=answer, citations=citations)
        db.add(assistant)
        db.add(
            QueryLog(
                user_id=user_id,
                question=req.question,
                mode=req.mode,
                cached=bool(cached),
                refused=refused,
                latency_ms=latency_ms,
                retrieval_ms=retrieval_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                chunk_ids=chunk_ids,
            )
        )
        db.commit()
        message_id = assistant.id

    yield {
        "type": "done",
        "answer": answer,
        "citations": citations,
        "retrieved": retrieved,
        "refused": refused,
        "cached": bool(cached),
        "conversation_id": str(conversation_id),
        "message_id": str(message_id),
        "latency_ms": latency_ms,
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "cost_usd": round(cost, 6)},
    }
