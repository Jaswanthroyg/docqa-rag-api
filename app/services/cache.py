import hashlib
import json
import logging

from app.core.config import settings
from app.core.redis_client import get_redis

log = logging.getLogger(__name__)


def corpus_version(user_id: str) -> int:
    """Bumped whenever a user's documents change, which invalidates their cached answers."""
    try:
        return int(get_redis().get(f"corpus_ver:{user_id}") or 0)
    except Exception:  # noqa: BLE001
        return 0


def bump_corpus_version(user_id: str) -> None:
    try:
        get_redis().incr(f"corpus_ver:{user_id}")
    except Exception:  # noqa: BLE001
        log.warning("could not bump corpus version", exc_info=True)


def make_key(user_id: str, question: str, mode: str, top_k: int, doc_ids: list[str], version: int) -> str:
    normalized = " ".join(question.lower().split())
    raw = json.dumps(
        [
            user_id,
            normalized,
            mode,
            top_k,
            sorted(doc_ids),
            version,
            settings.llm_provider,
            settings.effective_llm_model,
        ],
        separators=(",", ":"),
    )
    return "qa:" + hashlib.sha256(raw.encode()).hexdigest()


def get_json(key: str) -> dict | None:
    try:
        raw = get_redis().get(key)
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001
        log.warning("cache read failed", exc_info=True)
        return None


def set_json(key: str, value: dict) -> None:
    try:
        get_redis().set(key, json.dumps(value), ex=settings.cache_ttl_seconds)
    except Exception:  # noqa: BLE001
        log.warning("cache write failed", exc_info=True)
