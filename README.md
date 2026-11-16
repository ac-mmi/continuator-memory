# Continuator

**Open-source, local-first, provenance-first memory and evidence infrastructure for AI applications.**

Continuator turns conversations into durable, versioned memories, indexes documents as evidence units, and returns a **Context Pack** — structured memories, source messages, document excerpts, and a ready-to-paste prompt block — without requiring Continuator to answer the question.

## Why not ordinary RAG?

| Ordinary RAG | Continuator |
| --- | --- |
| Retrieves relevant chunks for one question | Maintains **durable memories** derived from conversations over time |
| Chunks are usually ephemeral index rows | Memories are **versioned** (updates supersede; history is kept) |
| Provenance is often “which chunk?” | Provenance is **source messages** and **document sections** |
| Often couples retrieval to one model/vendor | **Context Pack** is model-agnostic — use any LLM (or none) |

Continuator is infrastructure for *context*. Optional local answering via Ollama exists; it is not required for the quickstart.

## How it fits together

```text
Conversation transcript          Documents (.md / .html / .docx / .pdf)
         │                                      │
         ▼                                      ▼
   import (memories)                    ingest (EvidenceUnits)
         │                                      │
         └──────────────┬───────────────────────┘
                        ▼
                 local SQLite store
                        ▼
                 POST /context
                        ▼
                   Context Pack
          memories · source evidence · document evidence
                 · formattedContext
```

## Quickstart (~5 minutes)

Requires **Python ≥ 3.10** and a **repository checkout** (sample files and scripts live under `examples/`; they are not shipped inside the installed wheel). No API keys, Docker, Postgres, or Redis. **Ollama is optional** (only for `POST /ask` or LLM-based extraction).

**Fastest path** (from the repo root):

```bash
chmod +x examples/quickstart/run.sh
./examples/quickstart/run.sh
```

**Manual path** (one database variable — or none; default is `data/continuator.db`):

```bash
# 1. Install (from a Continuator git checkout)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
mkdir -p data

# 2. Start the local HTTP API
# Auto: reopens CONTINUATOR_DB_PATH when it already has an imported corpus;
# otherwise starts the built-in *fixture* demo (not your data).
# Force demo:  continuator-memory serve --sample fixture
# Force DB:    continuator-memory serve --sample db
continuator-memory serve --host 127.0.0.1 --port 8765
```

In another terminal (same venv):

```bash
source .venv/bin/activate

# 3. Import a sample conversation (extract=messages → no LLM)
python examples/quickstart/import_payload.py | curl -sS -X POST http://127.0.0.1:8765/import \
  -H 'Content-Type: application/json' \
  -d @- | python -m json.tool | head

# 4. Confirm the corpus is imported (not fixture) before trusting /context
curl -sS http://127.0.0.1:8765/health | python -m json.tool
# Expect: "origin": "imported", "sampleId": "imported", "conversationId": "imported"

# 5. Ingest Markdown via HTTP (same Workspace lock as /context; prefer this while serve runs)
python -c '
import json, urllib.request
from pathlib import Path
body = json.dumps({
    "text": Path("examples/quickstart/policy.md").read_text(encoding="utf-8"),
    "filename": "policy.md",
    "sourceUri": "inline://quickstart-policy",
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8765/documents/ingest",
    data=body, method="POST",
    headers={"Content-Type": "application/json", "Accept": "application/json"},
)
print(json.load(urllib.request.urlopen(req)))
'

# 6. Retrieve a Context Pack (no answering LLM)
curl -sS -X POST http://127.0.0.1:8765/context \
  -H 'Content-Type: application/json' \
  -d '{"query":"usual ferry","mode":"baseline"}' | python -m json.tool | head -n 60
```

The first `/context` call loads MiniLM (~90 MB) into the Hugging Face cache. If that fails, the API returns **HTTP 503** with install/download guidance (not a stack trace).

**Full walkthrough:** [`docs/quickstart.md`](docs/quickstart.md).

### Cleanup

```bash
# Stop the server (Ctrl+C), then:
rm -f data/continuator.db          # or data/quickstart.db if run.sh was used
```

## Example Context Pack (shape)

Public contract: [`docs/context-pack-schema.md`](docs/context-pack-schema.md) (`schemaVersion: "1"`).

`POST /context` returns JSON. Important fields:

```json
{
  "schemaVersion": "1",
  "query": "usual ferry",
  "mode": "baseline",
  "conversationId": "imported",
  "memories": [
    {
      "id": "…",
      "text": "…",
      "sourceMessageIds": ["imported:m0000"],
      "sources": [{ "messageId": "imported:m0000", "role": "user", "text": "…" }]
    }
  ],
  "evidence": [{ "messageId": "imported:m0000", "role": "user", "text": "…" }],
  "citations": [{ "memoryId": "…", "messageIds": ["imported:m0000"] }],
  "documentEvidence": [
    {
      "evidenceId": "…",
      "documentId": "…",
      "headingPath": ["Authentication", "Token Refresh"],
      "text": "Access tokens expire after one hour.",
      "rank": 1,
      "score": 0.42
    }
  ],
  "documentCitations": [{ "evidenceId": "…", "documentId": "…", "headingPath": ["…"] }],
  "formattedContext": "…deterministic prompt text…",
  "diagnostics": { "answerGeneration": "not_run", "documentSearch": "bm25" }
}
```

Use `formattedContext` with your own model, or consume `memories` / `evidence` / `documentEvidence` programmatically.

## Core capabilities

- **Conversation → memories** — import a transcript; default `extract=messages` stores one memory per message with no LLM
- **Document → EvidenceUnits** — Markdown, HTML, DOCX, digital-text PDF via CLI
- **Context Pack** — memories + conversation source evidence + document BM25 hits + citations
- **Local SQLite** — single-file storage; no external database
- **Optional answering** — `POST /ask` can call a local Ollama model; off for the quickstart
- **Web UI** — optional React app under `web/` against the same HTTP API

## Supported inputs

| Input | How | Notes |
| --- | --- | --- |
| Labeled transcript | `POST /import` | `You:` / `ChatGPT:` (or `user:` / `assistant:`) headers |
| JSON conversation | `POST /import` | Object or message list; see transcript parser |
| Markdown (HTTP or CLI) | `POST /documents/ingest` or `ingest-markdown` | HTTP is Markdown text JSON; HTML/DOCX/PDF remain CLI |
| HTML / DOCX / PDF | `continuator-memory ingest-*` | Same `--db` / `CONTINUATOR_DB_PATH` as the live workspace |

## Python SDK

With the server running:

```python
from continuator_memory.sdk import ContinuatorClient, ContinuatorAPIError

client = ContinuatorClient("http://127.0.0.1:8765")

client.import_conversation(open("examples/quickstart/conversation.txt").read(), extract="messages")
client.ingest_markdown(
    open("examples/quickstart/policy.md").read(),
    filename="policy.md",
    source_uri="inline://quickstart-policy",
)
pack = client.context("access tokens expire", mode="baseline")
print(pack["schemaVersion"], pack["formattedContext"][:200])

try:
    client.documents.get("missing")
except ContinuatorAPIError as exc:
    print(exc.status_code, exc.message)
```

Returns raw API JSON (Context Pack schema v1). No extra dependencies. See [`docs/quickstart.md`](docs/quickstart.md).

**External LLM:** Continuator retrieves; your app generates. See [`examples/external_llm/`](examples/external_llm/) (`ask_with_context_pack.py` + OpenAI-compatible HTTP via stdlib).

## API overview

Default: `http://127.0.0.1:8765`

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Workspace status |
| `POST` | `/import` | Import conversation text → memories |
| `POST` | `/context` | Retrieve a Context Pack (no answer LLM) |
| `POST` | `/ask` | Context Pack + optional Ollama answer |
| `POST` | `/documents/ingest` | Ingest Markdown text into EvidenceUnits |
| `GET` | `/memories`, `/sources`, `/graph` | Inspect corpus state |
| `GET` | `/documents`, `/documents/{id}`, `/evidence/{id}` | Inspect document evidence |
| `DELETE` | `/documents/{id}` | Remove a document and its units |
| `DELETE` | `/corpus/conversations/{id}` | Delete corpus conversation (keeps documents) |

CLI entry point: `continuator-memory` (also `python -m continuator_memory`).

## Configuration

Defaults work for local tryout. **One database variable is enough.**

| Variable | Default | Role |
| --- | --- | --- |
| `CONTINUATOR_DB_PATH` | `data/continuator.db` | **Canonical** SQLite path for import, CLI ingest, and Settings |
| `CONTINUATOR_IMPORT_DB` | (unset) | Legacy alias — used only when `CONTINUATOR_DB_PATH` is unset |
| `CONTINUATOR_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Memory retrieval embeddings |
| `CONTINUATOR_OLLAMA_MODEL` | `qwen2.5:1.5b` | Optional generation / LLM extraction |
| `OLLAMA_HOST` | `http://localhost:11434` | Optional Ollama base URL |

Precedence: explicit `--db` → `CONTINUATOR_DB_PATH` → `CONTINUATOR_IMPORT_DB` → `data/continuator.db`.  
If both DB variables are set to different paths, **`CONTINUATOR_DB_PATH` wins** for product writes.

**`.env` loading:** `continuator-memory` and `serve` load a repo-root (or cwd) `.env` into the process environment. Existing shell variables are **not** overwritten. Missing `.env` is ignored.

**Restart / sample mode**

| Setting | Behavior |
| --- | --- |
| *(omit `--sample`, unset `CONTINUATOR_WEB_SAMPLE`)* | **Auto:** open product DB when it has a usable imported corpus; else fixture |
| `--sample fixture` or `CONTINUATOR_WEB_SAMPLE=fixture` | Always built-in demo corpus |
| `--sample db` or `CONTINUATOR_WEB_SAMPLE=db` | Always open `CONTINUATOR_DB_PATH` / `--db` (fails clearly if unusable) |

After import, a normal restart without `--sample fixture` reloads your SQLite corpus. Check `GET /health` → `origin` (`fixture` vs `imported`).

**HTTP body limits:** JSON routes **512 KB**; `POST /import` **8 MB**. Oversized or incomplete bodies → `400`. Unsupported methods on known paths → `405` `{"error":"method not allowed"}` with an `Allow` header.

## Requirements and limitations

**Required**

- Python ≥ 3.10
- Dependencies from `pyproject.toml` (includes `sentence-transformers`, NumPy, and document parsers)
- Disk (and one-time network) for the embedding model cache on first `/context`

**Not required for Context Pack retrieval**

- Ollama / API keys / Docker / PostgreSQL / Redis

**Current limits (honest)**

- Single-process local HTTP server (`ThreadingHTTPServer`); bind defaults to `127.0.0.1`
- No authentication or multi-tenancy
- Default import extraction (`messages`) is heuristic, not LLM understanding
- Document search in `/context` is BM25 over EvidenceUnits (not vector search)
- Research benchmarks and adaptive controllers under `benchmarks/` are not the product API

## Development and testing

```bash
pip install -e ".[dev]"
pytest tests/unit -q
```

Architecture and product boundaries: [`ARCHITECTURE.md`](ARCHITECTURE.md).  
Historical Phase 1–3 research README: [`docs/phase1-rag-baseline.md`](docs/phase1-rag-baseline.md).

## License

[MIT](LICENSE)
