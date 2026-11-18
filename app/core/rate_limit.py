import logging
import time

from fastapi import HTTPException

from app.core.redis_client import get_redis

log = logging.getLogger(__name__)


def check_rate_limit(user_id: str, bucket: str, limit: int, window_seconds: int = 60) -> None:
    """Fixed-window limiter backed by Redis. Fails open if Redis is unavailable."""
    window = int(time.time() // window_seconds)
    key = f"rl:{bucket}:{user_id}:{window}"
    try:
        r = get_redis()
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds + 1)
        count = pipe.execute()[0]
    except Exception:  # noqa: BLE001
        log.warning("rate limiter unavailable, allowing request", exc_info=True)
        return
    if count > limit:
        retry_after = window_seconds - int(time.time() % window_seconds)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({limit}/{window_seconds}s). Try again later.",
            headers={"Retry-After": str(retry_after)},
        )
