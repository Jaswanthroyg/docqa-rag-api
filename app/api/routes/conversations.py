import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.db import get_db
from app.models import Conversation, Message, QueryLog, User
from app.schemas import ConversationDetail, ConversationOut, MessageOut

router = APIRouter(tags=["conversations"])


def _get_owned(db: Session, user: User, conversation_id: uuid.UUID) -> Conversation:
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conv


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return db.scalars(stmt).all()


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: uuid.UUID, user: User = Depends(current_user), db: Session = Depends(get_db)):
    conv = _get_owned(db, user, conversation_id)
    messages = db.scalars(
        select(Message).where(Message.conversation_id == conv.id).order_by(Message.created_at, Message.role.desc())
    ).all()
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        messages=[MessageOut.model_validate(m) for m in messages],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: uuid.UUID, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.delete(_get_owned(db, user, conversation_id))
    db.commit()


@router.get("/usage")
def usage(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Per-user query statistics from query_logs (latency, cache hit rate, tokens, cost)."""
    row = db.execute(
        select(
            func.count(QueryLog.id),
            func.count().filter(QueryLog.cached),
            func.count().filter(QueryLog.refused),
            func.coalesce(func.avg(QueryLog.latency_ms), 0),
            func.coalesce(func.avg(QueryLog.latency_ms).filter(QueryLog.cached), 0),
            func.coalesce(func.avg(QueryLog.latency_ms).filter(~QueryLog.cached), 0),
            func.coalesce(func.sum(QueryLog.input_tokens), 0),
            func.coalesce(func.sum(QueryLog.output_tokens), 0),
            func.coalesce(func.sum(QueryLog.cost_usd), 0),
        ).where(QueryLog.user_id == user.id)
    ).one()
    total, cached, refused = row[0], row[1], row[2]
    return {
        "queries": total,
        "cache_hit_rate": round(cached / total, 3) if total else 0.0,
        "refusal_rate": round(refused / total, 3) if total else 0.0,
        "avg_latency_ms": round(float(row[3]), 1),
        "avg_latency_ms_cached": round(float(row[4]), 1),
        "avg_latency_ms_uncached": round(float(row[5]), 1),
        "input_tokens": int(row[6]),
        "output_tokens": int(row[7]),
        "cost_usd": round(float(row[8]), 6),
    }
