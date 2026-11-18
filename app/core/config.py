from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"  # "production" enforces a real SECRET_KEY

    # Infrastructure
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"

    # Auth
    secret_key: str = "change-me"
    access_token_minutes: int = 30
    refresh_token_days: int = 7

    # File storage
    storage_backend: str = "local"  # local | s3
    upload_dir: str = "./data/uploads"
    s3_bucket: str = ""
    max_upload_mb: int = 20

    # Embeddings: "fastembed" (local model, free) or "hash" (offline, for tests/CI)
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_cache_dir: str | None = None

    # LLM: "anthropic", "gemini" (has a free tier) or "mock" (no API key, extractive answers)
    llm_provider: str = "mock"
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    # Leave empty to use the provider default (see effective_llm_model), or set a model id you have access to.
    llm_model: str = ""
    llm_max_tokens: int = 800
    llm_timeout_seconds: float = 60.0
    # Optional pricing (USD per million tokens) so query_logs can record cost
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0

    # Chunking and retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    retrieval_candidates: int = 20
    # Cosine distance above which the best match is considered irrelevant.
    # Tune this with eval/run_eval.py for your embedding model.
    max_vector_distance: float = 0.45

    # Caching and rate limiting
    cache_ttl_seconds: int = 3600
    rate_limit_ask_per_min: int = 20
    rate_limit_upload_per_min: int = 10

    @property
    def effective_llm_model(self) -> str:
        """LLM_MODEL if set, otherwise a default that matches the chosen provider."""
        if self.llm_model:
            return self.llm_model
        defaults = {
            "anthropic": "claude-sonnet-5-5",
            "gemini": "gemini-flash-latest",  # alias for the current Flash model
        }
        return defaults.get(self.llm_provider, "mock")


settings = Settings()

if settings.environment == "production" and settings.secret_key == "change-me":
    raise RuntimeError("Set a strong SECRET_KEY before running in production.")
