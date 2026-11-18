import hashlib
import math
import re
from functools import lru_cache

from app.core.config import settings


@lru_cache
def _fastembed_model():
    from fastembed import TextEmbedding

    kwargs = {"cache_dir": settings.embedding_cache_dir} if settings.embedding_cache_dir else {}
    return TextEmbedding(model_name=settings.embedding_model, **kwargs)


def _hash_embed(text: str) -> list[float]:
    """Deterministic bag-of-words hashing vector. Offline stand-in for tests/CI only."""
    dim = settings.embedding_dim
    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.md5(token.encode()).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    if settings.embedding_provider == "hash":
        return [_hash_embed(t) for t in texts]
    model = _fastembed_model()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(v.tolist() for v in model.embed(texts[i : i + batch_size]))
    _check_dim(out)
    return out


def embed_query(text: str) -> list[float]:
    if settings.embedding_provider == "hash":
        return _hash_embed(text)
    vec = next(iter(_fastembed_model().query_embed(text))).tolist()
    _check_dim([vec])
    return vec


def _check_dim(vectors: list[list[float]]) -> None:
    if vectors and len(vectors[0]) != settings.embedding_dim:
        raise RuntimeError(
            f"Embedding model returned {len(vectors[0])} dims but EMBEDDING_DIM={settings.embedding_dim}"
        )
