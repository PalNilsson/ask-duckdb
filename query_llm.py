#!/usr/bin/env python3
from __future__ import annotations

from openai import OpenAI
from mistralai import Mistral
import argparse
import sys
import os
import re
import json
import duckdb
import difflib
from typing import Optional, Set, List, Dict, Any


# =========================
# Prompt templates (unchanged)
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
    """Quote an SQL identifier for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def is_markdown_code_chunk(text: str) -> bool:
    """Check if text contains a Markdown fenced code block."""
    return bool(re.search(r"```[^`]*```", text, re.DOTALL))


def extract_code_from_markdown(markdown_text: str) -> Optional[str]:
    """Extract code content from the first Markdown fenced code block."""
    m = re.search(r"```(.*?)\n(?P<code>.*?)\n```", markdown_text, re.DOTALL)
    return m.group("code") if m else None


# =========================
# Schema helpers
# =========================

def describe_columns(db: duckdb.DuckDBPyConnection, tbl_name: str) -> List[Dict[str, str]]:
    """Return column metadata (name, type) via DESCRIBE SELECT *."""
    rows = db.sql("DESCRIBE SELECT * FROM " + qident(tbl_name) + ";").fetchall()
    # DuckDB DESCRIBE rows: (column_name, column_type, null, key, default, extra)
    return [{"name": r[0], "type": r[1]} for r in rows]


def build_tbl_schema(db: duckdb.DuckDBPyConnection, tbl_name: str) -> str:
    """Build a compact CREATE TABLE column list for prompts."""
    cols = describe_columns(db, tbl_name)
    return ", ".join(f"{c['name']} {c['type']}" for c in cols)


def list_columns(db: duckdb.DuckDBPyConnection, tbl_name: str) -> List[str]:
    """List column names for a table using DESCRIBE."""
    return [c["name"] for c in describe_columns(db, tbl_name)]


# =========================
# NEW: Schema JSON skeleton generator
# =========================

def _default_canonicalization_for(col_type: str) -> Dict[str, Any]:
    """Suggest a canonicalization skeleton based on type."""
    # For text-like columns we often want case normalization. Leave 'none' by default.
    textlike = any(t in col_type.upper() for t in ("CHAR", "STRING", "TEXT", "VARCHAR"))
    return {
        "case": "none" if not textlike else "none",  # set to 'lower' manually later if desired
        "map_values": {}
    }

def _default_duckdb_access_for(col_name: str, col_type: str) -> Dict[str, Any]:
    """Suggest access tips for complex types (JSON/MAP/STRUCT)."""
    u = col_type.upper()
    access: Dict[str, Any] = {}
    if "JSON" in u:
        access = {
            "exists_key": f"json_extract({col_name}, '$.rucio') IS NOT NULL",
            "get_setup": f"json_extract({col_name}, '$.rucio.setup')"
        }
    elif "STRUCT" in u:
        access = {"example": f"{col_name}.field"}
    elif "MAP" in u:
        access = {"example": f"{col_name}['key']"}
    return access

def make_schema_skeleton(table: str, cols: List[Dict[str, str]]) -> Dict[str, Any]:
    """Create a skeleton metadata JSON structure for a table."""
    skeleton_cols: List[Dict[str, Any]] = []
    for c in cols:
        entry: Dict[str, Any] = {
            "name": c["name"],
            "type": c["type"],
            "description": "",
            "aliases": [],
            "importance": 5,
            "allowed_values": [],
            "canonicalization": _default_canonicalization_for(c["type"]),
        }
        access = _default_duckdb_access_for(c["name"], c["type"])
        if access:
            entry["duckdb_access"] = access
        skeleton_cols.append(entry)

    return {
        "version": "1.0",
        "table": table,
        "notes": "Auto-generated skeleton. Fill in descriptions, rules, aliases, and any canonicalization/allowed_values.",
        "columns": skeleton_cols,
        "rules": []
    }

def write_schema_skeleton(path: str, data: Dict[str, Any]) -> None:
    """Write the schema skeleton JSON to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote schema skeleton to: {path}")


# =========================
# SQL auto-repair (existing)
# =========================

def fix_common_mistakes(sql: str, actual_cols: Set[str]) -> str:
    """Repair common LLM SQL mistakes using the real schema."""
    fixed = sql

    tokens = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", fixed)
    if "state" in tokens:
        if "status" in actual_cols and "state" not in actual_cols:
            fixed = re.sub(r"\bstate\b", "status", fixed)

    fixed = re.sub(r"=\s*'ONLINE'", "='online'", fixed, flags=re.IGNORECASE)

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
# LLM clients (existing)
# =========================

def ask_gemini(system: str, user: str, model: str, api_key: Optional[str] = None) -> str:
    """Call Gemini (OpenAI-compatible endpoint) and return raw text."""
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
    """Call Mistral and return raw text."""
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
# Execution (existing)
# =========================

def execute_and_display(db: duckdb.DuckDBPyConnection, sql: str, max_rows_print: int = 1000) -> None:
    """Execute SQL on DuckDB and print results."""
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
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Generate SQL from a question and run it on DuckDB; also supports schema skeleton generation."
    )
    ap.add_argument("--db", default="queuedata.db", help="Path to DuckDB file (default: queuedata.db)")
    ap.add_argument("--table", default="queuedata", help="Target table name (default: queuedata)")
    ap.add_argument("--question", help="Natural-language question to turn into SQL.")
    ap.add_argument("--llm", choices=["gemini", "mistral"], default="gemini", help="LLM provider.")
    ap.add_argument("--model", help="Model name (e.g., gemini-2.5-flash, mistral-small-latest).")

    # NEW: schema skeleton options
    ap.add_argument("--generate-schema", action="store_true",
                    help="Generate a skeleton JSON data dictionary for the table and exit.")
    ap.add_argument("--schema-out", help="Output path for the JSON schema (default: <table>.schema.json)")

    args = ap.parse_args()

    tbl_name = args.table

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

    # === NEW: handle schema skeleton generation and exit ===
    if args.generate_schema:
        cols = describe_columns(db, tbl_name)
        skeleton = make_schema_skeleton(tbl_name, cols)
        out_path = args.schema_out or f"{tbl_name}.schema.json"
        write_schema_skeleton(out_path, skeleton)
        return  # do not proceed with LLM flow

    # --- normal LLM flow (unchanged) ---
    if not args.question:
        print("Error: --question is required unless --generate-schema is used.", file=sys.stderr)
        sys.exit(2)

    schema_str = build_tbl_schema(db, tbl_name)
    system = system_template.format(tbl=tbl_name, schema=schema_str)
    user = user_template.format(args.question)

    try:
        raw = ask_gemini(system, user, args.model or "gemini-2.5-pro") if args.llm == "gemini" \
            else ask_mistral(system, user, args.model or "mistral-large-latest")
    except Exception as e:
        print(f"[{args.llm.capitalize()} call failed] {e}", file=sys.stderr)
        sys.exit(3)

    sql = extract_code_from_markdown(raw) if is_markdown_code_chunk(raw) else raw
    sql = (sql or raw or "").strip()

    cols = set(list_columns(db, tbl_name))
    repaired = fix_common_mistakes(sql, cols)

    print("=== Cleaned SQL ===")
    print(repaired)

    execute_and_display(db, repaired)


if __name__ == "__main__":
    main()
