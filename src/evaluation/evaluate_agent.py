"""
Siemens Energy Text-to-SQL + PostgreSQL RBAC evaluation runner.

Default mode invokes `sql_executor_chain` directly to obtain a proposed SQL tool call
without executing it through the agent's current global SQL tool (which may use
neondb_owner). It then validates and executes the SQL using the test case's actual
PostgreSQL login role.

Required environment variables:
  DATABASE_URL          Owner/template PostgreSQL URL; credentials are replaced per role.
  ROLE_PASSWORDS_JSON   JSON mapping role names to their current passwords.
  AGENT_MODULE          Python import path containing sql_executor_chain and SCHEMA_TEXT.
                        Example: src.new_projects.agent

Optional:
  AGENT_CHAIN_ATTRIBUTE   default: sql_executor_chain
  DATASET_PATH            default: siemens_energy_eval_100.json
  OUTPUT_DIR              default: evaluation_output
  STATEMENT_TIMEOUT_MS    default: 10000
  MAX_ROWS                default: 20000

Install:
  pip install sqlalchemy psycopg2-binary python-dotenv sqlglot langchain-core

Run:
  python evaluate_agent.py

Important:
- This tests SQL generation + execution under each PostgreSQL role.
- It deliberately does NOT call graph.invoke(), because the uploaded graph's ToolNode
  executes with the globally configured sql_tool (currently neondb_owner).
- The benchmark does not test FastAPI/JWT authentication or the graph's input validator.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import traceback
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

ROOT = Path(__file__).resolve().parent
_dataset_env = os.getenv("DATASET_PATH")
DATASET_PATH = Path(_dataset_env) if _dataset_env else (ROOT / "siemens_energy_eval_100.json")
if not DATASET_PATH.is_absolute():
    # Prefer the path relative to the current working directory; otherwise resolve
    # it beside this script. This supports both `python -m ...` and direct execution.
    cwd_candidate = Path.cwd() / DATASET_PATH
    script_candidate = ROOT / DATASET_PATH
    DATASET_PATH = cwd_candidate if cwd_candidate.exists() else script_candidate
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "evaluation_output"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = Path.cwd() / OUTPUT_DIR
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "10000"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "20000"))

APP_MODULE = os.getenv("AGENT_MODULE", "src.new_projects.agent")
CHAIN_ATTRIBUTE = os.getenv("AGENT_CHAIN_ATTRIBUTE", "sql_executor_chain")

DATABASE_URL = os.getenv("DATABASE_URL")
ROLE_PASSWORDS_JSON = os.getenv("ROLE_PASSWORDS_JSON")


def load_agent():
    if not APP_MODULE:
        raise RuntimeError("Set AGENT_MODULE to the import path of your agent module.")
    module = importlib.import_module(APP_MODULE)
    chain = getattr(module, CHAIN_ATTRIBUTE, None)
    schema_text = getattr(module, "SCHEMA_TEXT", None)
    validator = getattr(module, "validate_read_only_sql", None)

    if chain is None:
        raise AttributeError(
            f"{APP_MODULE}.{CHAIN_ATTRIBUTE} was not found. "
            "Your uploaded graph exposes sql_executor_chain; check AGENT_MODULE."
        )
    if schema_text is None:
        raise AttributeError(f"{APP_MODULE}.SCHEMA_TEXT was not found.")
    if validator is None:
        raise AttributeError(
            f"{APP_MODULE}.validate_read_only_sql was not found. "
            "The evaluator uses the same SQLGlot read-only validator as your graph."
        )
    return chain, schema_text, validator


def load_role_passwords() -> dict[str, str]:
    if not ROLE_PASSWORDS_JSON:
        raise RuntimeError(
            "Set ROLE_PASSWORDS_JSON to a JSON object mapping each PostgreSQL role "
            "to its current password. Do not put passwords in the benchmark JSON."
        )
    try:
        values = json.loads(ROLE_PASSWORDS_JSON)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ROLE_PASSWORDS_JSON is not valid JSON: {exc}") from exc
    if not isinstance(values, dict):
        raise RuntimeError("ROLE_PASSWORDS_JSON must be a JSON object.")
    return {str(k): str(v) for k, v in values.items()}


def make_role_url(base_url: str, role: str, passwords: dict[str, str]) -> str:
    if role not in passwords:
        raise KeyError(
            f"No password configured for role '{role}'. Add it to ROLE_PASSWORDS_JSON."
        )
    url = make_url(base_url)
    # Keep host, database, SSL options, and driver; replace only login credentials.
    return url.set(username=role, password=passwords[role])


def extract_sql(message: Any) -> str:
    """Extract the sql_db_query tool argument from a LangChain AIMessage."""
    calls = getattr(message, "tool_calls", None) or []
    for call in calls:
        name = call.get("name") or call.get("tool_name")
        args = call.get("args") or {}
        if name in (None, "sql_db_query") and isinstance(args, dict):
            query = args.get("query") or args.get("sql")
            if query:
                return str(query).strip()

    # Compatibility fallback for providers that place tool calls in additional_kwargs.
    additional = getattr(message, "additional_kwargs", {}) or {}
    raw_calls = additional.get("tool_calls") or []
    for call in raw_calls:
        fn = call.get("function") or {}
        name = fn.get("name")
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}
        if name in (None, "sql_db_query") and isinstance(args, dict):
            query = args.get("query") or args.get("sql")
            if query:
                return str(query).strip()
    return ""


def generate_sql(chain, schema_text: str, question: str) -> tuple[str, str]:
    response = chain.invoke(
        {
            "schema": schema_text,
            "history": "(none; independent benchmark case)",
            "user_input": question,
        }
    )
    return extract_sql(response), str(getattr(response, "content", "") or "")


def normalize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


ENGINE_CACHE = {}

def execute_as_role(
    role_url: str,
    query: str,
) -> tuple[bool, list[str] | None, list[list[Any]] | None, str | None, str | None]:
    """Execute one query in a read-only transaction as the named login role."""
    engine = ENGINE_CACHE.get(role_url)
    if engine is None:
        engine = create_engine(role_url, pool_pre_ping=True, future=True)
        ENGINE_CACHE[role_url] = engine
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                connection.exec_driver_sql(
                    f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT_MS}'"
                )
                result = connection.execute(text(query))
                rows = result.fetchmany(MAX_ROWS + 1)
                if len(rows) > MAX_ROWS:
                    transaction.rollback()
                    return False, None, None, "MAX_ROWS_EXCEEDED", (
                        f"Query returned more than {MAX_ROWS} rows."
                    )
                columns = [str(column) for column in result.keys()]
                normalized = [
                    [normalize_value(value) for value in tuple(row)]
                    for row in rows
                ]
                transaction.rollback()
                return True, columns, normalized, None, None
            except Exception:
                transaction.rollback()
                raise
    except Exception as exc:
        original = getattr(exc, "orig", exc)
        sqlstate = (
            getattr(original, "sqlstate", None)
            or getattr(original, "pgcode", None)
        )
        message = str(original)
        return False, None, None, str(sqlstate) if sqlstate else None, message
    finally:
        pass


def is_permission_denied(sqlstate: str | None, message: str | None) -> bool:
    return sqlstate == "42501" or "permission denied" in (message or "").lower()


def _canonical_row(row: list[Any]) -> str:
    """Stable serialization used for multiset comparison; duplicate rows are preserved."""
    return json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)


def compare_results(
    expected_columns: list[str],
    expected_rows: list[list[Any]],
    actual_columns: list[str],
    actual_rows: list[list[Any]],
    order_sensitive: bool = False,
    sample_limit: int = 5,
) -> dict[str, Any]:
    """
    Compare results by output-column name, not tuple position.

    If projections differ, compare the shared columns and report the projection
    difference separately. Such a match is PARTIAL, never silently treated as a
    full projection match.
    """
    def normalized_names(columns: list[str]) -> list[str]:
        return [str(name).strip().casefold() for name in columns]

    expected_names = normalized_names(expected_columns)
    actual_names = normalized_names(actual_columns)

    # Keep occurrence order and avoid silently conflating duplicate output labels.
    common_names = [name for name in expected_names if name in actual_names]
    common_names = list(dict.fromkeys(common_names))
    missing_columns = [name for name in expected_names if name not in actual_names]
    extra_columns = [name for name in actual_names if name not in expected_names]

    positional_fallback = False
    if not common_names:
        if len(expected_columns) == len(actual_columns) and expected_columns:
            # Same-width projections with different aliases: compare by ordinal,
            # but mark as partial because the output labels do not match.
            positional_fallback = True
            common_names = [f"<position:{index + 1}>" for index in range(len(expected_columns))]
            expected_projected = [list(row) for row in expected_rows]
            actual_projected = [list(row) for row in actual_rows]
        else:
            return {
                "equivalent": False,
                "projection_match": False,
                "comparison_columns": [],
                "missing_columns": missing_columns,
                "extra_columns": extra_columns,
                "expected_only_sample": [],
                "actual_only_sample": [],
                "reason": "No shared output columns and result widths differ.",
            }
    else:
        # Build index maps. SQLAlchemy normally disambiguates duplicate result labels.
        expected_index = {name: expected_names.index(name) for name in common_names}
        actual_index = {name: actual_names.index(name) for name in common_names}
        expected_projected = [
            [row[expected_index[name]] for name in common_names]
            for row in expected_rows
        ]
        actual_projected = [
            [row[actual_index[name]] for name in common_names]
            for row in actual_rows
        ]

    if order_sensitive:
        equivalent = expected_projected == actual_projected
        expected_only = []
        actual_only = []
        if not equivalent:
            for index, (left, right) in enumerate(zip(expected_projected, actual_projected)):
                if left != right:
                    expected_only.append({"row_index": index, "values": left})
                    actual_only.append({"row_index": index, "values": right})
                    if len(expected_only) >= sample_limit:
                        break
            if len(expected_projected) != len(actual_projected):
                expected_only.append({"expected_row_count": len(expected_projected)})
                actual_only.append({"actual_row_count": len(actual_projected)})
    else:
        expected_counter = Counter(_canonical_row(row) for row in expected_projected)
        actual_counter = Counter(_canonical_row(row) for row in actual_projected)
        equivalent = expected_counter == actual_counter
        expected_only = [
            json.loads(value)
            for value, count in (expected_counter - actual_counter).most_common(sample_limit)
            for _ in range(min(count, sample_limit))
        ]
        actual_only = [
            json.loads(value)
            for value, count in (actual_counter - expected_counter).most_common(sample_limit)
            for _ in range(min(count, sample_limit))
        ]

    projection_match = not missing_columns and not extra_columns
    return {
        "equivalent": equivalent,
        "projection_match": projection_match,
        "comparison_columns": common_names,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "expected_only_sample": expected_only,
        "actual_only_sample": actual_only,
        "reason": (
            "Matching rows and projections."
            if equivalent and projection_match
            else "Matching values on shared columns; output projections differ."
            if equivalent
            else "Values differ on one or more shared columns."
        ),
    }


def evaluate_case(case, chain, schema_text, validator, passwords):
    role = case["role"]
    expected_access = case["expected_access"]
    question = case["question"]
    expected_sql = case["expected_sql"]

    result = {
        "id": case["id"],
        "role": role,
        "category": case["category"],
        "question": question,
        "expected_access": expected_access,
        "expected_sql": expected_sql,
        "generated_sql": "",
        "expected_result_columns": None,
        "generated_result_columns": None,
        "comparison_columns": [],
        "missing_output_columns": [],
        "extra_output_columns": [],
        "expected_only_sample": [],
        "generated_only_sample": [],
        "projection_match": None,
        "passed": False,
        "status": "",
        "detail": "",
        "sql_valid": False,
        "expected_result_rows": None,
        "generated_result_rows": None,
        "empty_reference_result": False,
    }

    role_url = make_role_url(DATABASE_URL, role, passwords)

    # The mutation case is a deterministic test of the same SQLGlot guard used by
    # the uploaded graph. It never sends the mutation to PostgreSQL.
    if expected_access == "blocked_by_sql_guard":
        safe, reason = validator(expected_sql)
        result["sql_valid"] = safe
        if not safe:
            result["passed"] = True
            result["status"] = "PASS_SQL_GUARD_BLOCKED_MUTATION"
            result["detail"] = reason
        else:
            result["status"] = "FAIL_SQL_GUARD_ACCEPTED_MUTATION"
            result["detail"] = reason
        return result

    try:
        generated_sql, model_text = generate_sql(chain, schema_text, question)
        result["generated_sql"] = generated_sql
        result["model_response"] = model_text[:1000]
    except Exception as exc:
        result["status"] = "FAIL_GENERATION_ERROR"
        result["detail"] = f"{type(exc).__name__}: {exc}"
        return result

    if not generated_sql:
        refusal_markers = ("not authorized", "no permission", "cannot access", "can't access", "restricted", "permission denied")
        if expected_access == "denied" and any(marker in model_text.lower() for marker in refusal_markers):
            result["passed"] = True
            result["status"] = "PASS_AGENT_REFUSAL"
            result["detail"] = "The model refused the unauthorized request without generating SQL."
        else:
            result["status"] = "FAIL_NO_SQL_TOOL_CALL"
            result["detail"] = "The SQL generation chain did not return a sql_db_query tool call."
        return result

    safe, reason = validator(generated_sql)
    result["sql_valid"] = safe
    if not safe:
        if expected_access == "denied":
            # Safe refusal/blocking is acceptable for an unauthorized data request.
            result["passed"] = True
            result["status"] = "PASS_AGENT_SQL_GUARD_BLOCKED"
            result["detail"] = reason
        else:
            result["status"] = "FAIL_GENERATED_SQL_BLOCKED"
            result["detail"] = reason
        return result

    if expected_access == "denied":
        # First ensure the benchmark's golden query is actually denied for this role.
        expected_ok, _, _, expected_state, expected_error = execute_as_role(
            role_url, expected_sql
        )
        if expected_ok:
            result["status"] = "FAIL_RBAC_GRANT_TOO_BROAD"
            result["detail"] = (
                "The reference query succeeded. The live role has more access than "
                "the benchmark expects; review its GRANTs."
            )
            return result
        if not is_permission_denied(expected_state, expected_error):
            result["status"] = "FAIL_REFERENCE_QUERY_ERROR_NOT_PERMISSION"
            result["detail"] = (
                f"Golden query failed for a reason other than permission denial: "
                f"{expected_state}: {expected_error}"
            )
            return result

        generated_ok, _, _, generated_state, generated_error = execute_as_role(
            role_url, generated_sql
        )
        if not generated_ok and is_permission_denied(generated_state, generated_error):
            result["passed"] = True
            result["status"] = "PASS_RBAC_PERMISSION_DENIED"
            result["detail"] = f"PostgreSQL rejected generated SQL with SQLSTATE {generated_state}."
        elif generated_ok:
            result["status"] = "FAIL_GENERATED_QUERY_BYPASSED_EXPECTED_DENIAL"
            result["detail"] = "Generated SQL executed successfully under a role expected to be denied."
        else:
            result["status"] = "FAIL_GENERATED_QUERY_OTHER_ERROR"
            result["detail"] = f"{generated_state}: {generated_error}"
        return result

    # Allowed query: execute golden and generated SQL under the same restricted role.
    expected_ok, expected_columns, expected_rows, expected_state, expected_error = execute_as_role(
        role_url, expected_sql
    )
    if not expected_ok:
        result["status"] = "FAIL_ROLE_PERMISSION_SETUP"
        result["detail"] = (
            f"Golden query could not execute for role '{role}': "
            f"{expected_state}: {expected_error}"
        )
        return result

    generated_ok, generated_columns, generated_rows, generated_state, generated_error = execute_as_role(
        role_url, generated_sql
    )
    if not generated_ok:
        result["status"] = "FAIL_GENERATED_QUERY_EXECUTION"
        result["detail"] = f"{generated_state}: {generated_error}"
        result["expected_result_rows"] = len(expected_rows or [])
        result["expected_result_columns"] = expected_columns
        return result

    result["expected_result_rows"] = len(expected_rows or [])
    result["generated_result_rows"] = len(generated_rows or [])
    result["expected_result_columns"] = expected_columns
    result["generated_result_columns"] = generated_columns
    result["empty_reference_result"] = len(expected_rows or []) == 0

    # Result ordering is enforced only when the case explicitly requests it.
    # ORDER BY in the reference SQL alone is not enough: two valid queries may
    # return the same records in different orders.
    order_sensitive = bool(case.get("order_sensitive", False))
    comparison = compare_results(
        expected_columns or [],
        expected_rows or [],
        generated_columns or [],
        generated_rows or [],
        order_sensitive=order_sensitive,
    )
    result["comparison_columns"] = comparison["comparison_columns"]
    result["missing_output_columns"] = comparison["missing_columns"]
    result["extra_output_columns"] = comparison["extra_columns"]
    result["expected_only_sample"] = comparison["expected_only_sample"]
    result["generated_only_sample"] = comparison["actual_only_sample"]
    result["projection_match"] = comparison["projection_match"]

    if comparison["equivalent"] and comparison["projection_match"]:
        result["passed"] = True
        result["status"] = "PASS_RESULT_EQUIVALENCE"
        result["detail"] = comparison["reason"]
    elif comparison["equivalent"]:
        # Do not hide projection differences: count separately from full passes.
        result["passed"] = True
        result["status"] = "PASS_PARTIAL_RESULT_EQUIVALENCE"
        result["detail"] = (
            comparison["reason"]
            + f" Missing from generated output: {comparison['missing_columns']};"
            + f" additional generated columns: {comparison['extra_columns']}."
        )
    else:
        result["status"] = "FAIL_RESULT_MISMATCH"
        result["detail"] = (
            comparison["reason"]
            + f" Reference rows: {len(expected_rows or [])};"
            + f" generated rows: {len(generated_rows or [])}."
            + f" Compared columns: {comparison['comparison_columns']}."
        )
    return result


def main():
    global DATABASE_URL, ROLE_PASSWORDS_JSON
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required.")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if len(cases) != 100:
        raise ValueError(f"Expected exactly 100 cases; found {len(cases)}.")

    passwords = load_role_passwords()
    chain, schema_text, validator = load_agent()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index:03d}/100] role={case['role']:<20} category={case['category']:<16}", end=" ")
        try:
            row = evaluate_case(case, chain, schema_text, validator, passwords)
        except Exception as exc:
            row = {
                "id": case["id"],
                "role": case["role"],
                "category": case["category"],
                "question": case["question"],
                "expected_access": case["expected_access"],
                "generated_sql": "",
                "passed": False,
                "status": "FAIL_HARNESS_ERROR",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        results.append(row)
        print(row["status"])

    total = len(results)
    passed = sum(bool(row.get("passed")) for row in results)
    failed = total - passed
    by_category = {}
    for category in sorted({r["category"] for r in results}):
        group = [r for r in results if r["category"] == category]
        by_category[category] = {
            "total": len(group),
            "passed_including_partial": sum(bool(r.get("passed")) for r in group),
            "full_result_equivalence": sum(r.get("status") == "PASS_RESULT_EQUIVALENCE" for r in group),
            "partial_result_equivalence": sum(r.get("status") == "PASS_PARTIAL_RESULT_EQUIVALENCE" for r in group),
            "failed": sum(not bool(r.get("passed")) for r in group),
        }

    status_counts = dict(Counter(r.get("status", "UNKNOWN") for r in results))
    full_passes = status_counts.get("PASS_RESULT_EQUIVALENCE", 0)
    partial_passes = status_counts.get("PASS_PARTIAL_RESULT_EQUIVALENCE", 0)
    summary = {
        "benchmark": dataset["benchmark"],
        "total": total,
        "passed_including_partial": passed,
        "failed": failed,
        "pass_rate_including_partial_percent": round(100 * passed / total, 2) if total else 0,
        "full_result_equivalence_passes": full_passes,
        "partial_result_equivalence_passes": partial_passes,
        "full_result_equivalence_rate_percent": round(100 * full_passes / total, 2) if total else 0,
        "by_category": by_category,
        "status_counts": status_counts,
        "warning": (
            "This runner calls sql_executor_chain directly and executes generated SQL "
            "under the case's PostgreSQL role. It does not run graph.invoke(), so it "
            "does not test the graph's deterministic/AI input validators, cache, or "
            "self-evaluation/correction nodes. It also avoids executing through the "
            "graph's global owner-connected sql_tool. Result comparisons are unordered "
            "unless a case explicitly sets order_sensitive=true. Projection differences "
            "are reported as partial equivalence, not hidden as full equivalence."
        ),
    }

    json_path = OUTPUT_DIR / "evaluation_results.json"
    csv_path = OUTPUT_DIR / "evaluation_results.csv"
    summary_path = OUTPUT_DIR / "evaluation_summary.json"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = sorted({key for row in results for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })

    print("\n" + "=" * 60)
    print(f"TOTAL: {total} | PASSED (including partial): {passed} | FAILED: {failed}")
    print(f"PASS RATE (including partial): {summary['pass_rate_including_partial_percent']}%")
    print(
        "FULL result equivalence: "
        f"{summary['full_result_equivalence_passes']} "
        f"({summary['full_result_equivalence_rate_percent']}%)"
    )
    print(f"PARTIAL result equivalence: {summary['partial_result_equivalence_passes']}")
    print("\nBY CATEGORY")
    for category, metrics in by_category.items():
        print(f"  {category:<18} {metrics['passed']:>2}/{metrics['total']:<2} passed")
    print("\nSTATUS COUNTS")
    for status, count in summary["status_counts"].items():
        print(f"  {status}: {count}")
    print(f"\nReports written to: {OUTPUT_DIR.resolve()}")
    print(f"  - {json_path.name}")
    print(f"  - {csv_path.name}")
    print(f"  - {summary_path.name}")
    for engine in ENGINE_CACHE.values():
        engine.dispose()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nEVALUATION SETUP ERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        for engine in ENGINE_CACHE.values():
            engine.dispose()
        sys.exit(1)
