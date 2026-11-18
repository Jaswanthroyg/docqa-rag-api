import hashlib
import logging
import tempfile
import uuid
from pathlib import PurePath

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, upload_limited_user
from app.core.config import settings
from app.db import get_db
from app.models import Document, User
from app.schemas import DocumentOut
from app.services import cache
from app.services.extract import SUPPORTED_EXTENSIONS
from app.services.storage import get_storage
from app.workers.tasks import ingest_document

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _get_owned(db: Session, user: User, doc_id: uuid.UUID) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None or doc.user_id != user.id:  # 404 (not 403) so IDs can't be probed
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    file: UploadFile,
    user: User = Depends(upload_limited_user),
    db: Session = Depends(get_db),
):
    filename = PurePath(file.filename or "").name  # strip any path the client sent
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    limit = settings.max_upload_mb * 1024 * 1024
    digest, size = hashlib.sha256(), 0
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as tmp:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
            digest.update(chunk)
            tmp.write(chunk)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")

        sha = digest.hexdigest()
        existing = db.scalar(select(Document).where(Document.user_id == user.id, Document.sha256 == sha))
        if existing:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"You already uploaded this file (document id {existing.id})"
            )

        doc_id = uuid.uuid4()
        key = f"{user.id}/{doc_id}{ext}"
        tmp.seek(0)
        get_storage().save(key, tmp)

    doc = Document(
        id=doc_id, user_id=user.id, filename=filename, storage_key=key, sha256=sha, size_bytes=size
    )
    db.add(doc)
    db.commit()

    try:
        ingest_document.delay(str(doc.id))
    except Exception:  # noqa: BLE001  (broker unreachable)
        log.exception("could not enqueue ingestion")
        doc.status, doc.error = "failed", "Processing queue unavailable; delete and re-upload."
        db.commit()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Processing queue unavailable") from None
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    stmt = (
        select(Document)
        .where(Document.user_id == user.id)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return db.scalars(stmt).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: uuid.UUID, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return _get_owned(db, user, doc_id)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: uuid.UUID, user: User = Depends(current_user), db: Session = Depends(get_db)):
    doc = _get_owned(db, user, doc_id)
    key = doc.storage_key
    db.delete(doc)  # chunks are removed by ON DELETE CASCADE
    db.commit()
    try:
        get_storage().delete(key)
    except Exception:  # noqa: BLE001  (orphaned file is harmless; DB is the source of truth)
        log.warning("could not delete stored file", exc_info=True)
    cache.bump_corpus_version(str(user.id))
