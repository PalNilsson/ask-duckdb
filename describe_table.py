#!/usr/bin/env python3

"""Describe a DuckDB table."""

import argparse
import sys
import duckdb

def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def main():
    ap = argparse.ArgumentParser(description="Describe the 'queuedata' table in a DuckDB file.")
    ap.add_argument("--db", default="queuedata.db", help="Path to DuckDB file (default: queuedata.db)")
    args = ap.parse_args()

    tbl_name = "queuedata"

    try:
        db = duckdb.connect(args.db, read_only=True)
    except Exception as e:
        print(f"Error opening database '{args.db}': {e}", file=sys.stderr)
        sys.exit(1)

    # Verify the table exists (optional but helpful)
    exists = db.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [tbl_name]
    ).fetchone()[0]
    if not exists:
        print(f"Table '{tbl_name}' not found in {args.db}.", file=sys.stderr)
        sys.exit(2)

    # --- Your logic, as requested ---
    # describe the table
    tbl_description = db.sql("DESCRIBE SELECT * FROM " + qident(tbl_name) + ";")

    # extract the column names (column_name) along with their attributes (column_type) and reformat it
    # (this is the format we need to feed in the template for the schema)
    col_attr = tbl_description.df()[["column_name", "column_type"]]
    col_attr["column_joint"] = col_attr["column_name"] + " " + col_attr["column_type"]
    tbl_schema = (
        str(list(col_attr["column_joint"].values))
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
    )
    # --- end of your logic ---

    print("\n=== tbl_schema (for template) ===")
    print(tbl_schema)

if __name__ == "__main__":
    main()
