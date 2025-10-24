#!/usr/bin/env python3
from __future__ import annotations

from openai import OpenAI
from mistralai import Mistral
import argparse
import sys
import os
import re
import duckdb
import difflib
from typing import Optional, Set, List


# =========================
# Prompt templates
# =========================

user_template = "Write an SQL query that returns - {}"
system_template = """
You are generating SQL for DuckDB. Use ONLY the columns in the provided DDL.
Return SQL only (no markdown, no fences, no explanation).

CREATE TABLE {tbl} ({schema});

Rules:
- The queue status lives in the column `status`. Do NOT use `state`, `rc_site_state`, `gstat`, or other lookalikes.
- When filtering for online queues, use LOWER(status) = 'online'.
- Prefer selecting the `name` column for queue identifiers if relevant.
- Be case-insensitive by wrapping the compared column with LOWER(...).
- Output a single valid SQL statement, and nothing else.

Example:
-- User: list all queues that are online
SELECT name FROM {tbl} WHERE LOWER(status) = 'online';
"""


# =========================
# Utilities
# =========================

def qident(name: str) -> str:
    """Quote an SQL identifier for DuckDB.

    Args:
        name: Unquoted identifier (schema, table, or column name).

    Returns:
        The identifier quoted with double quotes, with embedded quotes escaped.
    """
    return '"' + name.replace('"', '""') + '"'


def is_markdown_code_chunk(text: str) -> bool:
    """Check if text contains a Markdown fenced code block.

    Args:
        text: Text to inspect.

    Returns:
        True if a fenced block like ```...``` is present, otherwise False.
    """
    pattern = r"```[^`]*```"
    return bool(re.search(pattern, text, re.DOTALL))


def extract_code_from_markdown(markdown_text: str) -> Optional[str]:
    """Extract code content from the first Markdown fenced code block.

    Args:
        markdown_text: Text that may contain a fenced code block.

    Returns:
        The inner code content if found, else None.
    """
    pattern = r"```(.*?)\n(?P<code>.*?)\n```"
    match = re.search(pattern, markdown_text, re.DOTALL)
    return match.group("code") if match else None


# =========================
# Schema helpers
# =========================

def build_tbl_schema(db: duckdb.DuckDBPyConnection, tbl_name: str) -> str:
    """Build a compact CREATE TABLE column list using DESCRIBE.

    Args:
        db: Open DuckDB connection.
        tbl_name: Table name to describe.

    Returns:
        A string like: "col1 TYPE, col2 TYPE, ...", suitable for prompt DDL.
    """
    desc = db.sql("DESCRIBE SELECT * FROM " + qident(tbl_name) + ";")
    col_attr = desc.df()[["column_name", "column_type"]]
    col_attr["column_joint"] = col_attr["column_name"] + " " + col_attr["column_type"]
    return (
        str(list(col_attr["column_joint"].values))
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
    )


def list_columns(db: duckdb.DuckDBPyConnection, tbl_name: str) -> List[str]:
    """List column names for a table using DESCRIBE.

    Args:
        db: Open DuckDB connection.
        tbl_name: Table name to list columns for.

    Returns:
        List of column names in order.
    """
    rows = db.sql("DESCRIBE SELECT * FROM " + qident(tbl_name) + ";").fetchall()
    return [r[0] for r in rows]


# =========================
# SQL auto-repair
# =========================

def fix_common_mistakes(sql: str, actual_cols: Set[str]) -> str:
    """Repair common LLM SQL mistakes using the real schema.

    The function:
      1) Rewrites `state` -> `status` if `state` is not a column but `status` is.
      2) Normalizes `'ONLINE'` (any case) to `'online'`.
      3) Fuzzy-corrects unknown identifiers to the closest actual column.

    Args:
        sql: The SQL string produced by an LLM.
        actual_cols: The actual set of column names present in the table.

    Returns:
        A potentially corrected SQL string that better matches the real schema.
    """
    fixed = sql

    # 1) Use `status` instead of `state` when appropriate
    tokens = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", fixed)
    if "state" in tokens:
        if "status" in actual_cols and "state" not in actual_cols:
            fixed = re.sub(r"\bstate\b", "status", fixed)

    # 2) Normalize ONLINE literal to lowercase 'online'
    fixed = re.sub(r"=\s*'ONLINE'", "='online'", fixed, flags=re.IGNORECASE)

    # 3) Fuzzy fix other unknown identifiers (avoid SQL keywords)
    keywords = {
        "select","from","where","and","or","not","in","as","on","join","left","right",
        "inner","outer","group","by","order","limit","offset","having","distinct",
        "like","ilike","lower","upper","count","sum","avg","min","max"
    }
    token_set = set(tokens)
    unknowns = [t for t in token_set if t.lower() not in keywords and t not in actual_cols]

    for t in unknowns:
        candidates = difflib.get_close_matches(t, list(actual_cols), n=1, cutoff=0.86)
        if candidates:
            fixed = re.sub(rf"\b{re.escape(t)}\b", candidates[0], fixed)

    return fixed


# =========================
# LLM clients
# =========================

def ask_gemini(system: str, user: str, model: str, api_key: Optional[str] = None) -> str:
    """Call Gemini (OpenAI-compatible endpoint) and return raw text.

    Args:
        system: System prompt string.
        user: User prompt string.
        model: Model name (e.g., "gemini-2.5-flash").
        api_key: Optional API key; falls back to GEMINI_API_KEY env var.

    Returns:
        Raw text from `response.choices[0].message.content`.

    Raises:
        RuntimeError: If the API key is missing or the response is invalid.
    """
    gemini_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    client = OpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        max_completion_tokens=5000,
    )
    if not resp.choices or not getattr(resp.choices[0], "message", None):
        raise RuntimeError("Gemini returned no choices/message.")
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("Gemini returned empty content.")
    return content


def ask_mistral(system: str, user: str, model: str, api_key: Optional[str] = None) -> str:
    """Call Mistral and return raw text.

    Args:
        system: System prompt string.
        user: User prompt string.
        model: Model name (e.g., "mistral-small-latest").
        api_key: Optional API key; falls back to MISTRAL_API_KEY env var.

    Returns:
        Raw text from `response.choices[0].message.content`.

    Raises:
        RuntimeError: If the API key is missing or the response is invalid.
    """
    mistral_api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY not set.")
    client = Mistral(api_key=mistral_api_key)
    resp = client.chat.complete(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        max_tokens=5000,
    )
    if not resp.choices or not getattr(resp.choices[0], "message", None):
        raise RuntimeError("Mistral returned no choices/message.")
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("Mistral returned empty content.")
    return content


# =========================
# Execution
# =========================

def execute_and_display(db: duckdb.DuckDBPyConnection, sql: str, max_rows_print: int = 1000) -> None:
    """Execute SQL on DuckDB and print results.

    Args:
        db: Open DuckDB connection.
        sql: SQL string to execute.
        max_rows_print: Maximum number of rows to print before truncation.

    Raises:
        SystemExit: With code 4 if SQL execution fails.
    """
    try:
        df = db.sql(sql).df()
    except Exception as e:
        print(f"[SQL execution failed] {e}", file=sys.stderr)
        sys.exit(4)

    total_rows = len(df)
    print(f"\n=== Query result ({total_rows} rows) ===")
    if total_rows == 0:
        print("Empty DataFrame")
        return
    print(df.head(max_rows_print).to_string(index=False))
    if total_rows > max_rows_print:
        print(f"\n... truncated to first {max_rows_print} rows ...")


# =========================
# CLI
# =========================

def main() -> None:
    """CLI entry point for generating SQL via LLM and executing in DuckDB.

    Parses CLI args, opens the database, builds the schema-aware prompts,
    calls the chosen LLM, cleans and repairs the SQL, then executes it.
    """
    ap = argparse.ArgumentParser(
        description="Generate SQL from a question and run it on DuckDB with schema-aware auto-repair."
    )
    ap.add_argument("--db", default="queuedata.db", help="Path to DuckDB file (default: queuedata.db)")
    ap.add_argument("--question", required=True, help="Natural-language question to turn into SQL.")
    ap.add_argument("--llm", choices=["gemini", "mistral"], default="gemini", help="LLM provider.")
    ap.add_argument("--model", help="Model name (e.g., gemini-2.5-flash, mistral-small-latest).")
    args = ap.parse_args()

    defaults = {"gemini": "gemini-2.5-pro", "mistral": "mistral-large-latest"}
    model = args.model or defaults[args.llm]
    tbl_name = "queuedata"

    # Open DB
    try:
        db = duckdb.connect(args.db, read_only=True)
    except Exception as e:
        print(f"Error opening database '{args.db}': {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure table exists
    exists = db.execute(
        "SELECT 1 FROM information_schema.tables WHERE lower(table_name)=lower(?) LIMIT 1",
        [tbl_name]
    ).fetchone()
    if not exists:
        print(f"Table '{tbl_name}' not found in {args.db}.", file=sys.stderr)
        sys.exit(2)

    # Prompts
    schema_str = build_tbl_schema(db, tbl_name)
    system = system_template.format(tbl=tbl_name, schema=schema_str)
    user = user_template.format(args.question)

    # LLM call
    try:
        raw = ask_gemini(system, user, model) if args.llm == "gemini" else ask_mistral(system, user, model)
    except Exception as e:
        print(f"[{args.llm.capitalize()} call failed] {e}", file=sys.stderr)
        sys.exit(3)

    # Clean markdown -> SQL
    sql = extract_code_from_markdown(raw) if is_markdown_code_chunk(raw) else raw
    sql = (sql or raw or "").strip()

    # Auto-repair against actual schema
    cols = set(list_columns(db, tbl_name))
    repaired = fix_common_mistakes(sql, cols)

    print("=== Cleaned SQL ===")
    print(repaired)

    # Execute
    execute_and_display(db, repaired)


if __name__ == "__main__":
    main()
