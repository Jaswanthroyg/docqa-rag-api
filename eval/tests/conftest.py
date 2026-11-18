import os
import tempfile

# Must be set before the app is imported.
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5432/rag_test")
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="docqa-test-")
os.environ["SECRET_KEY"] = "test-secret-key-that-is-long-enough-for-hs256"

import fakeredis  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.core import redis_client  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.db import engine, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.workers.tasks import ingest_document  # noqa: E402

settings.max_vector_distance = 0.95  # hash embeddings: unrelated text has distance ~1.0


@pytest.fixture(scope="session", autouse=True)
def _schema():
    init_db()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE users, documents, chunks, conversations, messages, query_logs CASCADE"))
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_client.redis.Redis, "from_url", classmethod(lambda cls, *a, **k: fake))
    redis_client.get_redis.cache_clear()
    # Run ingestion inline instead of through a real Celery worker.
    monkeypatch.setattr(ingest_document, "delay", lambda doc_id: ingest_document.apply(args=(doc_id,)))
    monkeypatch.setattr(settings, "rate_limit_ask_per_min", 1000)
    monkeypatch.setattr(settings, "rate_limit_upload_per_min", 1000)
    yield
    redis_client.get_redis.cache_clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _make_user(client: TestClient, email: str) -> dict:
    password = "correct-horse-battery"
    assert client.post("/auth/register", json={"email": email, "password": password}).status_code == 201
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture
def alice(client):
    return _make_user(client, "alice@example.com")


@pytest.fixture
def bob(client):
    return _make_user(client, "bob@example.com")


def upload(client: TestClient, headers: dict, name: str, content: bytes, content_type="text/plain"):
    return client.post("/documents", headers=headers, files={"file": (name, content, content_type)})
