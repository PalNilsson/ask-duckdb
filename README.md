# 🦆 Ask DuckDB

**Ask DuckDB** lets you query DuckDB databases using natural language questions powered by modern LLMs (Gemini 2.5 / Mistral Large / others).
It’s part of the *Ask PanDA* ecosystem, providing an intelligent interface between human queries and structured SQL data.

---

## 🚀 Overview

Ask DuckDB automatically:
1. Reads a DuckDB database (e.g. `queuedata.db`).
2. Builds the table schema using `DESCRIBE SELECT * FROM ...`.
3. Sends your question to an LLM (Gemini or Mistral) with schema context.
4. Cleans and validates the generated SQL (including column name fixes and case normalization).
5. Executes the SQL against your local database and prints the results.

You can think of it as a lightweight, offline-friendly “LLM-to-SQL” bridge for DuckDB.

---

## 🧩 Features

- 🔍 Works with **Gemini** and **Mistral** APIs (OpenAI-compatible).
- 🧠 Auto-corrects common LLM mistakes (e.g. `state` → `status`).
- 🧾 Cleans markdown/code fences from LLM outputs.
- 🪶 Supports any `.db` file readable by DuckDB.
- 🧰 Helpful companion script `query_db.py` for debugging direct SQL.

---


## 🧱 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/ask-duckdb.git
   cd ask-duckdb
   ```

2. **Install dependencies**
```bash
pip install duckdb openai mistralai pandas`
```

3. **Set your API keys**
```bash
export GEMINI_API_KEY="your_gemini_key"
export MISTRAL_API_KEY="your_mistral_key"
```

## Examples

Show the DB schema for the default database.db file
1. `python3 describe_table.py`

Query the database directly
1. `python3 query_db.py --dq queuedata.db --table queuedata --query "SELECT * FROM queuedata WHERE status = 'online'"`

2. Query an LLM (the SQL corresponding to the question will be generated and then executed against the database)
1. `python3 query_llm.py --question "list all queues that are online" --llm mistral --model mistral-small-latest`
2. `python3 query_llm.py --question "list all queues that are online that do not use the rucio copytool" --llm gemini --model gemini-2.5-flash`

