"""Per-user rate limiting for /chat (sliding window, in-memory, per process)."""
import math
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Response, status

from .auth import get_current_user

RATE_LIMIT_REQUESTS = int(os.environ["RATE_LIMIT_REQUESTS"])
RATE_LIMIT_WINDOW_SECONDS = int(os.environ["RATE_LIMIT_WINDOW_SECONDS"])


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_s: int):
        self.limit, self.window_s = limit, window_s
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> tuple[bool, int, int]:
        """Returns (allowed, remaining, retry_after_seconds). Rejected requests are NOT counted."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False, 0, max(1, math.ceil(q[0] + self.window_s - now))
            q.append(now)
            return True, self.limit - len(q), 0


chat_limiter = SlidingWindowLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def rate_limited_user(response: Response, user: str = Depends(get_current_user)) -> str:
    """Authenticates first (so anonymous traffic can't burn anyone's quota), then enforces the limit."""
    allowed, remaining, retry_after = chat_limiter.hit(user)
    headers = {
        "X-RateLimit-Limit": str(chat_limiter.limit),
        "X-RateLimit-Remaining": str(remaining),
    }
    if not allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit exceeded ({chat_limiter.limit} requests per {chat_limiter.window_s}s). "
            f"Retry in {retry_after}s.",
            headers={**headers, "Retry-After": str(retry_after)},
        )
    for k, v in headers.items():
        response.headers[k] = v
    return user