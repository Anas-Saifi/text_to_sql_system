"""Run every query as the authenticated user's own Postgres role.

The app connects with ONE low-privilege login role (DATABASE_URL) that is a member of the
per-user roles but inherits nothing. Each query runs in its own transaction as:

    SET TRANSACTION READ ONLY; SET LOCAL ROLE <user>; SET LOCAL statement_timeout; <query>

so Postgres GRANTs / row-level security for that role apply. SET LOCAL reverts at
commit/rollback, so a pooled connection never carries one user's role into the next request.

Successful queries return a JSON string: {"columns": [...], "rows": [[...]], "truncated": bool}
Failures return a string starting with "Error: ...".
"""
import datetime
import decimal
import json
import logging
import os
import re

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

log = logging.getLogger(__name__)

# Returned (instead of a query error) when we could not connect / switch to the user's role.
# The agent skips the LLM evaluator for these: rewriting the SQL cannot fix them.
SETUP_ERROR_PREFIX = "Error: database setup failed"
# SQLSTATE 42501 (insufficient_privilege) on the query itself: the user's role may not read that data.
PERMISSION_ERROR_PREFIX = "Error: access denied"


def is_access_error(result) -> bool:
    """Role/connection/permission failures: rewriting the SQL cannot fix these, so skip the evaluator."""
    r = str(result)
    return r.startswith(SETUP_ERROR_PREFIX) or r.startswith(PERMISSION_ERROR_PREFIX)


_ROLE_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
MAX_ROWS = int(os.getenv("DB_MAX_ROWS", 500))
STATEMENT_TIMEOUT = os.getenv("DB_STATEMENT_TIMEOUT", "15s")

# Cloud Run + Cloud SQL socket example:
# postgresql+psycopg2://app_login:PASS@/dbname?host=/cloudsql/PROJECT:REGION:INSTANCE
if os.getenv("VERCEL"):  # serverless: don't hold connections per instance; Neon's pooler endpoint pools for us
    engine = create_engine(os.environ["DATABASE_URL"], poolclass=NullPool)
else:
    engine = create_engine(os.environ["DATABASE_URL"], pool_size=5, max_overflow=5, pool_pre_ping=True)


def _jsonable(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "<binary>"
    return v  # anything else exotic (uuid, interval ...) is stringified by json.dumps(default=str)


def _short(exc: Exception) -> str:
    """First line of the Postgres error only (no echoed SQL, parameters or SQLAlchemy links)."""
    msg = str(getattr(exc, "orig", None) or exc).strip()
    return msg.splitlines()[0] if msg else type(exc).__name__


def run_query_as(role: str, query: str) -> str:
    if not isinstance(role, str) or not _ROLE_RE.match(role):
        return f"{SETUP_ERROR_PREFIX} (role {role!r}): not a valid role name."
    quoted = engine.dialect.identifier_preparer.quote_identifier(role)
    stage = "setup"
    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))  # must come before the first real query
                conn.execute(text(f"SET LOCAL ROLE {quoted}"))
                conn.execute(text(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'"))
                stage = "query"
                # escape ':' so SQLAlchemy's text() never mistakes ':foo' inside a literal for a bind param
                result = conn.execute(text(query.replace(":", r"\:")))
                columns = list(result.keys())
                rows = result.fetchmany(MAX_ROWS + 1)
        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        payload = {
            "columns": columns,
            "rows": [[_jsonable(v) for v in r] for r in rows],
            "truncated": truncated,
        }
        return json.dumps(payload, default=str)
    except Exception as exc:  # noqa: BLE001
        if stage == "setup":
            log.error("DB setup failed for role %r: %s", role, exc)
            return f"{SETUP_ERROR_PREFIX} (role {role!r}): {_short(exc)}"
        if getattr(getattr(exc, "orig", None), "pgcode", None) == "42501":
            log.warning("access denied | role=%s | %s | sql=%s", role, _short(exc), query[:500])
            return f"{PERMISSION_ERROR_PREFIX}: {_short(exc)}"
        return f"Error: {_short(exc)}"  # syntax error, missing column, timeout ...


def parse_result(result: str):
    """Parse a successful run_query_as() string -> dict, or None (error / unexpected format)."""
    try:
        d = json.loads(result)
    except (TypeError, ValueError):
        return None
    if isinstance(d, dict) and isinstance(d.get("columns"), list) and isinstance(d.get("rows"), list):
        return d
    return None


@tool
def sql_db_query(query: str, config: RunnableConfig) -> str:
    """Execute a read-only SQL query against the database. Returns JSON with the column names and
    rows. If the query is incorrect or not permitted, an error message is returned."""
    # `config` is injected by LangGraph and hidden from the LLM's tool schema.
    return run_query_as(config.get("configurable", {}).get("user_id", ""), query)