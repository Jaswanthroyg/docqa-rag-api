import logging
import uuid

from sqlalchemy import delete

from app.core.config import settings
from app.db import SessionLocal
from app.models import Chunk, Document
from app.services import cache
from app.services.chunking import split_text
from app.services.embeddings import embed_documents
from app.services.extract import ExtractionError, extract_pages
from app.services.storage import get_storage
from app.workers.celery_app import celery_app

log = logging.getLogger(__name__)


def _set_status(doc_id: uuid.UUID, status: str, error: str | None = None) -> None:
    with SessionLocal() as db:
        doc = db.get(Document, doc_id)
        if doc:
            doc.status, doc.error = status, error
            db.commit()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=10)
def ingest_document(self, document_id: str) -> None:
    """Extract text -> chunk -> embed -> store. Safe to re-run (it replaces existing chunks)."""
    doc_id = uuid.UUID(document_id)
    with SessionLocal() as db:
        doc = db.get(Document, doc_id)
        if doc is None:
            return
        user_id, filename, key = doc.user_id, doc.filename, doc.storage_key
        doc.status, doc.error = "processing", None
        db.commit()

    try:
        pages = extract_pages(get_storage().read(key), filename)
        pieces: list[tuple[int | None, str]] = [
            (page.number, text)
            for page in pages
            for text in split_text(page.text, settings.chunk_size, settings.chunk_overlap)
        ]
        if not pieces:
            raise ExtractionError("Document produced no text chunks.")
        vectors = embed_documents([text for _, text in pieces])

        with SessionLocal() as db:
            db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            db.add_all(
                Chunk(
                    document_id=doc_id,
                    user_id=user_id,
                    page=page,
                    chunk_index=i,
                    content=text,
                    embedding=vector,
                )
                for i, ((page, text), vector) in enumerate(zip(pieces, vectors, strict=True))
            )
            doc = db.get(Document, doc_id)
            doc.status, doc.chunk_count, doc.error = "ready", len(pieces), None
            db.commit()
        cache.bump_corpus_version(str(user_id))
        log.info("ingested document", extra={"document_id": document_id, "chunks": len(pieces)})
    except ExtractionError as exc:  # permanent: retrying will not help
        _set_status(doc_id, "failed", str(exc))
    except Exception as exc:  # noqa: BLE001  (transient: embedding model, DB, storage...)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc) from exc
        log.exception("ingestion failed", extra={"document_id": document_id})
        _set_status(doc_id, "failed", f"Processing failed: {exc.__class__.__name__}")
