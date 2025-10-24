#!/usr/bin/env python3
import argparse
import sys
import duckdb

def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    # Case-insensitive check, ignores schema (defaults to 'main')
    sql = """
        SELECT 1
        FROM information_schema.tables
        WHERE lower(table_name) = lower(?)
        LIMIT 1
    """
    row = conn.execute(sql, [table]).fetchone()
    return row is not None

def print_result(conn: duckdb.DuckDBPyConnection, query: str, max_rows: int = 1000) -> None:
    try:
        rel = conn.sql(query)
        # Use pandas if available; otherwise rely on DuckDB's built-in pretty printing via fetchall
        try:
            df = rel.fetchdf()
            total = len(df)
            print(f"=== Query result ({total} rows) ===")
            if total > max_rows:
                print(df.head(max_rows).to_string(index=False))
                print(f"\n... truncated to first {max_rows} rows ...")
            else:
                print(df.to_string(index=False))
        except Exception:
            rows = rel.fetchall()
            cols = [d[0] for d in rel.description] if rel.description else []
            print(f"=== Query result ({len(rows)} rows) ===")
            if not rows:
                print("∅ (no rows)")
                return
            # Simple column header
            if cols:
                print(" | ".join(cols))
                print("-" * (sum(len(c) for c in cols) + 3 * (len(cols) - 1)))
            for i, r in enumerate(rows):
                if i >= max_rows:
                    print(f"... truncated to first {max_rows} rows ...")
                    break
                print(" | ".join(str(v) for v in r))
    except Exception as e:
        print(f"[SQL execution failed] {e}", file=sys.stderr)
        sys.exit(4)

def main():
    ap = argparse.ArgumentParser(
        description="Execute an arbitrary SQL query against a DuckDB file."
    )
    ap.add_argument("--dq", required=True, help="Path to DuckDB file (e.g., queuedata.db)")
    ap.add_argument("--table", required=True, help="Table name to validate (e.g., queuedata)")
    ap.add_argument("--query", required=True, help="SQL query to execute")
    ap.add_argument("--max-rows", type=int, default=1000, help="Max rows to print (default: 1000)")
    args = ap.parse_args()

    # Open DB read-only
    try:
        conn = duckdb.connect(args.dq, read_only=True)
    except Exception as e:
        print(f"Error opening database '{args.dq}': {e}", file=sys.stderr)
        sys.exit(1)

    # Verify table exists (early fail helps catch typos)
    if not table_exists(conn, args.table):
        print(f"Table '{args.table}' not found in {args.dq}.", file=sys.stderr)
        sys.exit(2)

    # Run the provided query
    print_result(conn, args.query, max_rows=args.max_rows)

if __name__ == "__main__":
    main()
