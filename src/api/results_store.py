"""Short-lived, per-user store of successful query results (so actions can run on them)."""
import secrets
import threading
import time
from dataclasses import dataclass, field


@dataclass
class StoredResult:
    id: str
    user: str
    question: str
    sql: str
    columns: list
    rows: list
    truncated: bool
    created: float = field(default_factory=time.monotonic)
    outputs: dict = field(default_factory=dict)  # action -> finished output, so repeat clicks don't re-call the LLM


class ResultStore:
    def __init__(self, ttl_s: int = 1800, max_per_user: int = 20):
        self.ttl_s, self.max_per_user = ttl_s, max_per_user
        self._items: dict[str, StoredResult] = {}
        self._lock = threading.Lock()

    def _evict(self, now: float, user: str) -> None:
        for rid in [k for k, v in self._items.items() if now - v.created > self.ttl_s]:
            del self._items[rid]
        mine = sorted((v for v in self._items.values() if v.user == user), key=lambda v: v.created)
        for v in mine[: max(0, len(mine) - self.max_per_user + 1)]:
            del self._items[v.id]

    def put(self, user, question, sql, columns, rows, truncated) -> str:
        now = time.monotonic()
        with self._lock:
            self._evict(now, user)
            rid = secrets.token_urlsafe(12)
            self._items[rid] = StoredResult(rid, user, question, sql or "", columns, rows, truncated)
            return rid

    def get(self, user: str, rid: str):
        """Only the owner can read a result; anything else looks like 'not found'."""
        with self._lock:
            v = self._items.get(rid)
            if v is None or v.user != user or time.monotonic() - v.created > self.ttl_s:
                return None
            return v