import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limit import check_rate_limit
from app.core.security import decode_token
from app.db import get_db
from app.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token",
    headers={"WWW-Authenticate": "Bearer"},
)


def current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        user_id = decode_token(token, "access")
    except (jwt.PyJWTError, ValueError):
        raise _UNAUTHORIZED from None
    user = db.get(User, user_id)
    if user is None:
        raise _UNAUTHORIZED
    return user


def ask_limited_user(user: User = Depends(current_user)) -> User:
    check_rate_limit(str(user.id), "ask", settings.rate_limit_ask_per_min)
    return user


def upload_limited_user(user: User = Depends(current_user)) -> User:
    check_rate_limit(str(user.id), "upload", settings.rate_limit_upload_per_min)
    return user
