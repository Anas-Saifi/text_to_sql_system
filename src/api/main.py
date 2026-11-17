"""FastAPI backend for the Text-to-SQL LangGraph agent.

Run from the project root (the agent reads src/new_projects/database_schema.json by relative path):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
"""
import logging
import os
import uuid
from enum import Enum
from pathlib import Path as FsPath
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# >>> adjust this import to wherever your agent file lives <<<
from src.new_projects.agent import graph
from src.new_projects.db_access import PERMISSION_ERROR_PREFIX, SETUP_ERROR_PREFIX, parse_result

from . import actions as act
from .auth import (
    TOKEN_TTL_MINUTES,
    authenticate,
    create_access_token,
    get_current_user,
    limiter,
)
from .rate_limit import rate_limited_user
from .results_store import ResultStore

log = logging.getLogger(__name__)
DEBUG = os.environ["true"]  # include DB error + SQL in "denied"/setup-error responses

app = FastAPI(title="Text-to-SQL API", version="1.1.0")
store = ResultStore(ttl_s=int(os.getenv("RESULT_TTL_SECONDS", 1800)))

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


# ---------------------------------------------------------------- schemas
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    # Omit to start a new conversation; send the returned id back to continue it.
    conversation_id: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class ChatResponse(BaseModel):
    conversation_id: str
    status: Literal["success", "error", "denied", "incomplete", "rejected", "blocked"]
    message: Optional[str] = None
    sql: Optional[str] = None
    columns: Optional[list[str]] = None
    rows: Optional[list[list[Any]]] = None
    truncated: bool = False  # result was cut off at the row limit
    result_id: Optional[str] = None  # pass to POST /results/{result_id}/{action}
    actions: list[str] = []  # which of summarize / analyze / plot apply to this result
    cache: Optional[Literal["exact", "semantic"]] = None
    corrected: bool = False  # evaluator replaced the first SQL


class ActionName(str, Enum):
    summarize = "summarize"
    analyze = "analyze"
    plot = "plot"


class ActionResponse(BaseModel):
    result_id: str
    action: ActionName
    summary: Optional[str] = None
    analysis: Optional[str] = None  # markdown
    stats: Optional[dict[str, Any]] = None
    chart: Optional[dict[str, Any]] = None  # {type, title, x_label, y_label, data:[{x,y}], notes}


# ---------------------------------------------------------------- auth routes
@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    key = f"{request.client.host if request.client else '-'}:{form.username.lower()}"
    limiter.check(key)
    if not authenticate(form.username, form.password):
        limiter.fail(key)
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Incorrect username or password.", headers={"WWW-Authenticate": "Bearer"}
        )
    limiter.clear(key)
    return TokenResponse(access_token=create_access_token(form.username), expires_in=TOKEN_TTL_MINUTES * 60)


@app.get("/auth/me", tags=["auth"])
def me(user: str = Depends(get_current_user)):
    return {"username": user}


# ---------------------------------------------------------------- chat
def _build_response(state: dict, conversation_id: str, user: str) -> ChatResponse:
    cache = state.get("cache_hit") or None

    if not cache:
        if not state.get("valid_input"):
            if state.get("incomplete_request"):
                return ChatResponse(
                    conversation_id=conversation_id,
                    status="incomplete",
                    message="Input seems incomplete, care to elaborate?",
                )
            return ChatResponse(
                conversation_id=conversation_id,
                status="rejected",
                message=state.get("invalid_reason") or "Invalid request.",
            )
        if not state.get("sql_safe"):
            return ChatResponse(
                conversation_id=conversation_id,
                status="blocked",
                message=f"SQL execution blocked: {state.get('sql_safety_reason') or 'unsafe query'}",
            )

    corrected = (not cache) and state.get("result_correct") is False and bool(state.get("corrected_sql"))
    if cache:
        sql = state.get("final_sql")
    else:
        sql = state.get("corrected_sql") if corrected else state.get("generated_sql")

    result = str(state.get("query_result", ""))
    base = dict(conversation_id=conversation_id, cache=cache, corrected=corrected)

    if result.startswith(PERMISSION_ERROR_PREFIX):
        return ChatResponse(
            **base,
            status="denied",
            message=result if DEBUG else "You don't have permission to access the data this question needs.",
            sql=sql if DEBUG else None,
        )
    if result.startswith(SETUP_ERROR_PREFIX):  # details are logged server-side, not sent to the client
        return ChatResponse(
            **base,
            status="error",
            message=result if DEBUG else "Database access is not configured for your account. Contact an administrator.",
        )
    if result.lstrip().lower().startswith("error"):
        return ChatResponse(**base, status="error", message=result, sql=sql or None)

    data = parse_result(result)
    if data is None:  # unexpected / legacy format: pass it through, no actions
        return ChatResponse(**base, status="success", message=result, sql=sql or None)

    columns, rows, truncated = data["columns"], data["rows"], bool(data.get("truncated"))
    available = act.available_actions(columns, rows)
    result_id = store.put(user, state.get("user_input", ""), sql, columns, rows, truncated) if available else None
    return ChatResponse(
        **base,
        status="success",
        message=None if rows else "The query ran but returned no rows.",
        sql=sql or None,
        columns=columns,
        rows=rows,
        truncated=truncated,
        result_id=result_id,
        actions=available,
    )


# Plain `def` on purpose: graph.invoke is blocking, so FastAPI runs it in its threadpool.
@app.post("/chat", response_model=ChatResponse, tags=["agent"])
def chat(body: ChatRequest, user: str = Depends(rate_limited_user)):
    conversation_id = body.conversation_id or uuid.uuid4().hex
    config = {
        "configurable": {
            # thread_id is prefixed with the username so one user can never read another's conversation
            "thread_id": f"{user}:{conversation_id}",
            # user_id namespaces the result cache per user AND selects the Postgres role the query runs as
            "user_id": user,
        }
    }
    try:
        state = graph.invoke(
            {"messages": [HumanMessage(content=body.message)], "user_input": body.message},
            config=config,
        )
    except Exception:
        log.exception("agent failure for user=%s", user)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The agent is temporarily unavailable. Please retry.")
    return _build_response(state, conversation_id, user)


# ---------------------------------------------------------------- actions on a result
@app.post("/results/{result_id}/{action}", response_model=ActionResponse, tags=["agent"])
def run_action(
    action: ActionName,
    result_id: str = Path(pattern=r"^[A-Za-z0-9_-]{8,64}$"),
    user: str = Depends(rate_limited_user),
):
    entry = store.get(user, result_id)  # owner-only; someone else's id looks like "not found"
    if entry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Result not found or expired. Ask the question again.")
    if action.value not in act.available_actions(entry.columns, entry.rows):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"'{action.value}' does not apply to this result.")

    if action.value not in entry.outputs:  # repeat clicks reuse the first answer (no extra LLM call)
        try:
            if action is ActionName.summarize:
                entry.outputs["summarize"] = {"summary": act.summarize(entry)}
            elif action is ActionName.analyze:
                entry.outputs["analyze"] = act.analyze(entry)
            else:
                entry.outputs["plot"] = {"chart": act.plot(entry)}
        except act.ActionError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
        except Exception:
            log.exception("action %s failed for user=%s", action.value, user)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The model is temporarily unavailable. Please retry.")
    return ActionResponse(result_id=result_id, action=action, **entry.outputs[action.value])


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------- prototype UI
# Served from the same origin as the API at /ui/ (so no CORS setup is needed). Add ?demo=1 for mock data.
_FRONTEND = FsPath(__file__).resolve().parents[2] / "frontend"
if _FRONTEND.is_dir():
    app.mount("/ui", StaticFiles(directory=_FRONTEND, html=True), name="ui")