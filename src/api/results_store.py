"""Per-user store of successful query results, kept in Postgres (schema `app`).

Serverless instances don't share memory, so an in-process dict would make /results/{id}/{action} 404
whenever a request lands on a different instance. The login role (app_login) owns the `app` schema;
the 12 per-user roles have no privileges on it, so they can't read it through the text-to-SQL path.

One-time setup, as the database owner:   CREATE SCHEMA app AUTHORIZATION app_login;
"""
import json
import logging
import secrets
import threading
from dataclasses import dataclass, field

from sqlalchemy import text

from src.new_projects.db_access import engine

log = logging.getLogger(__name__)

_DDL = (
    """CREATE TABLE IF NOT EXISTS app.results (
        id         text PRIMARY KEY,
        username   text NOT NULL,
        question   text NOT NULL DEFAULT '',
        sql        text NOT NULL DEFAULT '',
        columns    jsonb NOT NULL,
        rows       jsonb NOT NULL,
        truncated  boolean NOT NULL DEFAULT false,
        outputs    jsonb NOT NULL DEFAULT CAST('{}' AS jsonb),
        created_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS results_user_created ON app.results (username, created_at DESC)",
)


@dataclass
class StoredResult:
    id: str
    user: str
    question: str
    sql: str
    columns: list
    rows: list
    truncated: bool
    outputs: dict = field(default_factory=dict)  # action -> finished output (repeat clicks skip the LLM)


class ResultStore:
    def __init__(self, ttl_s: int = 1800, max_per_user: int = 20):
        self.ttl_s, self.max_per_user = ttl_s, max_per_user
        self._ready = False
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            with engine.begin() as conn:
                for stmt in _DDL:
                    conn.execute(text(stmt))
            self._ready = True

    def put(self, user, question, sql, columns, rows, truncated) -> str:
        self._ensure()
        rid = secrets.token_urlsafe(12)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM app.results WHERE created_at < now() - make_interval(secs => :ttl)"),
                         {"ttl": self.ttl_s})
            conn.execute(text("""DELETE FROM app.results WHERE username = :u AND id IN (
                                   SELECT id FROM app.results WHERE username = :u
                                   ORDER BY created_at DESC OFFSET :keep)"""),
                         {"u": user, "keep": max(self.max_per_user - 1, 0)})
            conn.execute(text("""INSERT INTO app.results (id, username, question, sql, columns, rows, truncated)
                                 VALUES (:id, :u, :q, :s, CAST(:c AS jsonb), CAST(:r AS jsonb), :t)"""),
                         {"id": rid, "u": user, "q": question or "", "s": sql or "",
                          "c": json.dumps(columns), "r": json.dumps(rows, default=str), "t": bool(truncated)})
        return rid

    def get(self, user: str, rid: str):
        """Only the owner can read a result; anything else looks like 'not found'."""
        self._ensure()
        with engine.begin() as conn:
            r = conn.execute(text("""SELECT id, username, question, sql, columns, rows, truncated, outputs
                                     FROM app.results
                                     WHERE id = :id AND username = :u
                                       AND created_at > now() - make_interval(secs => :ttl)"""),
                             {"id": rid, "u": user, "ttl": self.ttl_s}).mappings().first()
        if r is None:
            return None
        return StoredResult(r["id"], r["username"], r["question"], r["sql"], r["columns"], r["rows"],
                            r["truncated"], r["outputs"] or {})

    def save_output(self, rid: str, action: str, output: dict) -> None:
        with engine.begin() as conn:
            conn.execute(text("""UPDATE app.results
                                 SET outputs = outputs || jsonb_build_object(CAST(:a AS text), CAST(:o AS jsonb))
                                 WHERE id = :id"""),
                         {"a": action, "o": json.dumps(output, default=str), "id": rid})