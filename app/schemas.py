import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)

    @field_validator("password")
    @classmethod
    def _bcrypt_limit(cls, v: str) -> str:
        if len(v.encode()) > 72:
            raise ValueError("password must be at most 72 bytes")
        return v


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    email: EmailStr
    created_at: datetime


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    filename: str
    size_bytes: int
    status: str
    error: str | None
    chunk_count: int
    created_at: datetime


class AskRequest(BaseModel):
    # Without this, the /docs page pre-fills fake UUIDs (3fa85f64-...) into conversation_id and
    # document_ids, which then fail with 404. This makes "Try it out" start from a clean body.
    model_config = ConfigDict(
        json_schema_extra={"example": {"question": "What technical skills are listed in my resume?"}}
    )

    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[uuid.UUID] | None = None
    conversation_id: uuid.UUID | None = None
    stream: bool = False
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=10)
    no_cache: bool = False


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    role: str
    content: str
    citations: list | None
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    title: str
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]
