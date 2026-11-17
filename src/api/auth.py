"""JWT authentication for the Text-to-SQL API.

Users live in a JSON file / env var as {"username": "<bcrypt hash>"} - never plaintext.
Generate hashes with scripts/hash_password.py.
"""
import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

JWT_SECRET = os.getenv("JWT_SECRET", "")
if len(JWT_SECRET) < 32:
    raise RuntimeError("JWT_SECRET must be set and at least 32 characters long.")
JWT_ALG = "HS256"
TOKEN_TTL_MINUTES = int(os.environ["TOKEN_TTL_MINUTES"])


def _load_users() -> dict[str, str]:
    raw = os.getenv("AUTH_USERS_JSON")  # e.g. a Cloud Run secret exposed as env var
    if raw:
        return json.loads(raw)
    with open(os.getenv("AUTH_USERS_FILE", "users.json"), "r", encoding="utf-8") as f:
        return json.load(f)


USERS: dict[str, str] = _load_users()

# Verified against when the username is unknown, so response time doesn't reveal valid usernames.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt()).decode()


def authenticate(username: str, password: str) -> bool:
    stored = USERS.get(username)
    ok = bcrypt.checkpw(password.encode()[:72], (stored or _DUMMY_HASH).encode())  # bcrypt ignores >72 bytes
    return ok and stored is not None


class LoginLimiter:
    """Blocks a (ip, username) pair after N failed logins within a window. Per-process, in-memory."""

    def __init__(self, max_failures: int = 5, window_s: int = 900):
        self.max_failures, self.window_s = max_failures, window_s
        self._fails: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str) -> deque:
        q, cutoff = self._fails[key], time.monotonic() - self.window_s
        while q and q[0] < cutoff:
            q.popleft()
        return q

    def check(self, key: str) -> None:
        with self._lock:
            if len(self._prune(key)) >= self.max_failures:
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed logins. Try again later.")

    def fail(self, key: str) -> None:
        with self._lock:
            self._prune(key).append(time.monotonic())

    def clear(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


limiter = LoginLimiter()


def create_access_token(username: str) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": username, "iat": now, "exp": now + timedelta(minutes=TOKEN_TTL_MINUTES)}
    return jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALG)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED, "Invalid or expired token.", headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], options={"require": ["exp", "sub"]})
    except jwt.PyJWTError:
        raise unauthorized
    username = payload["sub"]
    if username not in USERS:  # user removed after token was issued
        raise unauthorized
    return username