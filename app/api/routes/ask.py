import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import ask_limited_user
from app.db import get_db
from app.models import Conversation, Document, User
from app.schemas import AskRequest
from app.services import qa

router = APIRouter(tags=["ask"])


def _validate_scope(db: Session, user: User, req: AskRequest) -> None:
    if req.document_ids:
        owned = set(
            db.scalars(
                select(Document.id).where(Document.user_id == user.id, Document.id.in_(req.document_ids))
            )
        )
        if owned != set(req.document_ids):
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                "One or more document_ids don't exist or aren't yours. Remove document_ids to search all "
                "your documents, or copy ids from GET /documents.",
            )
    if req.conversation_id:
        conv = db.get(Conversation, req.conversation_id)
        if conv is None or conv.user_id != user.id:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                "Conversation not found. Remove conversation_id to start a new conversation, or use the "
                "conversation_id returned by an earlier /ask response.",
            )


def _sse(events: Iterator[dict]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event)}\n\n"


@router.post("/ask")
def ask(req: AskRequest, user: User = Depends(ask_limited_user), db: Session = Depends(get_db)):
    """Answer a question from the user's documents. Set `stream: true` for Server-Sent Events."""
    _validate_scope(db, user, req)
    events = qa.run(user.id, req)

    if req.stream:
        return StreamingResponse(
            _sse(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    final = None
    for event in events:
        if event["type"] == "error":
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, event["message"])
        if event["type"] == "done":
            final = event
    return final
