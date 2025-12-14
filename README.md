
<img src="./logo.png" alt="Tablesense-AI Logo" style="width:400px;height:auto;">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# TableSense-AI

TableSense-AI is a Streamlit app + agent toolkit for table-centric question answering.

It has two complementary parts:

1. **Runtime application (UI + agents)** in `tablesense_ai/`.
2. **Offline benchmarking harness** in `benchmark/` to evaluate agents on table-QA datasets.

---

## Project Structure

High-level layout (simplified):

```
.
├─ tablesense_ai/                      # Application code (agents + Streamlit UI)
│  ├─ app_streamlit.py                 # Main Streamlit entrypoint (used by Dockerfile)
│  ├─ agent/
│  │  ├─ base.py                       # Base agent interface used by benchmarks
│  │  ├─ code_agent/                   # Code-generation style table analysis agents
│  │  ├─ serialization/                # Agents that serialize tables (HTML/MD/...) before asking the LLM
│  │  └─ rag/                          # Retrieval-Augmented Generation pipeline (indexing + retrieval + SQL tool)
│  └─ utils/                           # Canonicalisation/performance helpers used across app + benchmark
│
├─ benchmark/                          # Benchmarking toolkit (datasets + evaluator + metrics)
│  ├─ evaluate_example.py              # Example runner
│  ├─ evaluator/                       # Core evaluator, caching, dataset definitions, metrics
│  └─ tab_llm_datasets/                # Local dataset definitions / helpers
│
├─ docker-compose.yml                  # Dev stack (Milvus + MinIO + Postgres + Docling + UI container)
├─ Dockerfile                          # Builds a Streamlit UI image for this repo
├─ requirements.txt / pyproject.toml   # Python dependencies (pip / Poetry)
└─ docs/, scripts/, tests/             # Notes, one-off scripts, and basic tests
```

---

## Runtime App (Streamlit + Agents)

The main UI is in `tablesense_ai/app_streamlit.py`.

It exposes two modes:

1. **Pure tabular analysis** (upload CSV → ask questions)
2. **Contextualized analysis (RAG)** (upload PDFs → index → ask questions grounded in retrieved context)

### Where the RAG architecture lives

RAG-specific code is in `tablesense_ai/agent/rag/`:

- `indexer.py`
	- Converts documents (PDF → markdown/text/tables via Docling API)
	- Extracts markdown tables
	- Stores **tables into Postgres** for exact computation with SQL
	- Stores **table chunks + free text into Milvus** for vector retrieval

- `retriever.py`
	- `RetrieverTool`: retrieves relevant chunks from Milvus; also appends Postgres schemas for referenced tables
	- `SQLQueryTool`: executes **read-only SELECT** queries against Postgres (guards against non-SELECT)
	- `PostgreSQLHelper`: handles SQLAlchemy engine + schema inspection

**Why two stores?**

- **Milvus (vector store)**: semantic retrieval of relevant text/table snippets.
- **Postgres (relational store)**: exact computations over full tables (AVG, SUM, filters, joins, etc.).

This is the key architectural separation in the RAG stack: retrieval for context, SQL for correctness.

---

## Benchmarking (Offline Evaluation)

Benchmarking is intentionally separate from the runtime RAG/UI stack.

It lives in `benchmark/` and provides:

- Dataset definitions (local + remote via HuggingFace)
- An `Evaluator` that iterates over dataset examples
- Metrics (exact match, ROUGE, BERTScore, etc.)
- A caching layer to avoid re-running completed evaluations

### How benchmarking differs from RAG

- **Goal**
	- RAG (`tablesense_ai/agent/rag/`): produce answers grounded in indexed documents and/or exact SQL.
	- Benchmarking (`benchmark/`): measure agent quality on known datasets with ground truth.

- **Execution mode**
	- RAG: long-running app, interactive, stateful services (Milvus/Postgres/Docling).
	- Benchmarking: batch evaluation scripts; does not require Milvus/Postgres unless your tested agent depends on it.

- **Core abstraction**
	- Benchmarking expects an agent compatible with `tablesense_ai/agent/base.py` (implements `eval(...)`).

### Example: run a benchmark script

The repo includes runnable examples:

```bash
python benchmark/evaluate_example.py
```

If you use hosted models (e.g., Mistral API), set required environment variables first (see “Environment Variables”).

---

## Environment Variables

The Streamlit app reads settings from a `.env` file (loaded via `python-dotenv`).

Typical variables used by `tablesense_ai/app_streamlit.py`:

```dotenv
# Postgres (used for SQLQueryTool + table storage)
DB_USER=tablesense_user
DB_PASS=tablesense_password
DB_NAME=tablesense_db
DB_HOST=postgres_db
DB_PORT=5432

# Milvus (used for retrieval)
MILVUS_HOST=standalone
MILVUS_PORT=19530

# Optional: local model gateway settings used by some agents
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_API_KEY=

# Optional: hosted LLM keys (used by some benchmark scripts)
OPENAI_API_KEY=YOUR_KEY_HERE
```

Notes:

- Inside Docker Compose, service DNS names (e.g., `postgres_db`, `standalone`) work as hosts.
- If you run the UI on your host machine (not in Docker), you likely want `DB_HOST=localhost`.

---

## Docker Deployment

There are two common workflows:

1. **Use Docker Compose to start the full stack** (datastores + Docling + UI).
2. **Build the UI image locally** from this repo (useful while developing).

### 1) Run the full stack via Docker Compose

From the repository root:

```bash
docker compose up -d
```

Useful endpoints:

- Streamlit UI: http://localhost:8501
- Milvus health/UI: http://localhost:9091 (UI at `/webui`)
- MinIO console: http://localhost:9001
- Docling API: http://localhost:5001

To follow logs:

```bash
docker compose logs -f
```

To stop:

```bash
docker compose down
```

To also remove persisted volumes (destructive):

```bash
docker compose down -v
```

### 2) Build the Streamlit UI image (this repo)

The provided `Dockerfile` builds an image that runs:

```bash
streamlit run ./tablesense_ai/app_streamlit.py --server.port=8501 --server.address=0.0.0.0
```

Build it:

```bash
docker build -t tablesense-ai:local .
```

Run it against the Compose network (recommended):

1. Start dependencies (Milvus + Postgres + Docling):

```bash
docker compose up -d etcd minio standalone postgres_db docling
```

2. Run your locally-built UI container on the same network:

```bash
docker run --rm -p 8501:8501 --env-file .env --network milvus tablesense-ai:local
```

### (Optional) Use your local image from docker-compose.yml

If you want `docker compose up` to run your locally-built UI image, update the `table_sense_ui` service in `docker-compose.yml` to use a local build instead of a remote image:

```yaml
	table_sense_ui:
		container_name: table_sense_ui
		build: .
		# image: tablesense-ai:local
		restart: always
		ports:
			- "8501:8501"
		env_file:
			- .env
```

Then run:

```bash
docker compose up -d --build
```

---

## Local (Non-Docker) Development

Either use Poetry:

```bash
poetry install
poetry run streamlit run tablesense_ai/app_streamlit.py
```

Or use pip:

```bash
pip install -r requirements.txt
streamlit run tablesense_ai/app_streamlit.py
```

---

## Common Issues

- **Milvus errors / corrupted state**: try `docker compose down -v` (destructive), or prune volumes if you know you don’t need them.
- **Nothing prints in Streamlit**: `print()` goes to server logs; prefer `st.write(...)` or use logging, and be aware cached functions may not re-run.