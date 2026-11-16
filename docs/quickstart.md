# Continuator quickstart

Verified against the current package. Goal: install, import a conversation, ingest a document, retrieve a Context Pack — without Ollama.

Sample scripts and files live under `examples/` in a **repository checkout** (not inside the installed wheel alone).

## What you need

| Required | Optional |
| --- | --- |
| Python ≥ 3.10 | [Ollama](https://ollama.com) — only for `POST /ask` or LLM extraction |
| Network once — `pip install` and first embedding download | Node / Vite — only for the React UI under `web/` |

No API keys, Docker, PostgreSQL, or Redis.

## One-command demo

From the repository root (creates `.venv` if needed):

```bash
chmod +x examples/quickstart/run.sh
./examples/quickstart/run.sh
```

Uses `CONTINUATOR_DB_PATH=data/quickstart.db` by default. Cleanup: `rm -f data/quickstart.db`.

## Manual steps

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
continuator-memory --help
```

`continuator-memory --help` lists **product** commands first (`serve`, document ingest) and labels Phase 1 `ingest` / `retrieve` / `ask` as research/legacy (not the Context Pack API).

### 2. Start the server

One database path is enough (`CONTINUATOR_DB_PATH`, default `data/continuator.db`).

```bash
mkdir -p data
continuator-memory serve --host 127.0.0.1 --port 8765
```

**Startup / restart:** with no `--sample` flag, Continuator **auto-reopens** `CONTINUATOR_DB_PATH` when that file already has an imported corpus; otherwise it starts the built-in **fixture demo**. Force demo with `--sample fixture`. Force DB with `--sample db` (errors clearly if the DB cannot be opened — it does not silently fall back to fixture).

On a first run (no DB yet), expect `origin=fixture` until you import. After import + restart (auto or `--sample db`), expect `origin=imported`. Leave this terminal running.

### 3. Import the sample conversation

Second terminal (same venv):

```bash
source .venv/bin/activate

python examples/quickstart/import_payload.py | curl -sS -X POST http://127.0.0.1:8765/import \
  -H 'Content-Type: application/json' \
  -d @- \
  | python -m json.tool
```

Uses `extract: "messages"` → **one memory per message, no LLM**. Expect `"ok": true`.

### 4. Verify health (required)

Do **not** treat `/context` as application data until health shows an imported corpus:

```bash
curl -sS http://127.0.0.1:8765/health | python -m json.tool
```

| Field | Fixture (before import) | After successful import |
| --- | --- | --- |
| `origin` | `fixture` | `imported` |
| `sampleId` | `fixture` | `imported` |
| `conversationId` | `web-sample` | `imported` |
| `importedFilename` | `null` | e.g. `conversation.txt` |

`origin` and `sampleId` are the same value today. Prefer checking **`origin`**.

### 5. Ingest the sample document (HTTP / SDK)

While the server is running, ingest via the **same Workspace** (HTTP or SDK). Avoid CLI `ingest-markdown` against a live server (separate SQLite connection).

```bash
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
```

CLI alternative when the server is **stopped**: `continuator-memory ingest-markdown examples/quickstart/policy.md` (writes `CONTINUATOR_DB_PATH`).

### 6. Retrieve a Context Pack

```bash
curl -sS -X POST http://127.0.0.1:8765/context \
  -H 'Content-Type: application/json' \
  -d '{"query":"usual ferry","mode":"baseline"}' \
  | python -m json.tool
```

Document-oriented query:

```bash
curl -sS -X POST http://127.0.0.1:8765/context \
  -H 'Content-Type: application/json' \
  -d '{"query":"access tokens expire","mode":"baseline"}' \
  | python -m json.tool
```

### First `/context` note

Loads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) into the Hugging Face cache (network once, unless already cached). If the model cannot be downloaded or loaded, the API returns **HTTP 503** with an actionable message (not a stack trace). Fix network/cache, or set `CONTINUATOR_EMBEDDING_MODEL` to a local model path.

### How to read the response

| Field | Meaning |
| --- | --- |
| `memories` | Retrieved durable memories |
| `evidence` | Source conversation messages |
| `citations` | Memory → message id links |
| `documentEvidence` | BM25 hits over ingested EvidenceUnits |
| `formattedContext` | Deterministic text for any LLM prompt |
| `diagnostics.answerGeneration` | `"not_run"` for `/context` |

### 7. Same flow with the Python SDK

Keep the server running, then in another terminal (same venv):

```python
from continuator_memory.sdk import ContinuatorClient, ContinuatorAPIError

client = ContinuatorClient("http://127.0.0.1:8765", timeout=60.0)

client.import_conversation(
    open("examples/quickstart/conversation.txt", encoding="utf-8").read(),
    filename="conversation.txt",
    extract="messages",  # no Ollama
)

health = client.health()
assert health["origin"] == "imported"  # not "fixture"

ingested = client.ingest_markdown(
    open("examples/quickstart/policy.md", encoding="utf-8").read(),
    filename="policy.md",
    source_uri="inline://quickstart-policy",
)
print(ingested["document"]["documentId"], ingested["replaced"])

pack = client.context("access tokens expire", mode="baseline")
assert pack["schemaVersion"] == "1"
print(pack["diagnostics"]["documentSearch"], len(pack.get("documentEvidence") or []))

docs = client.documents.list()
detail = client.documents.get(ingested["document"]["documentId"])
evidence = client.evidence.get(detail["units"][0]["evidenceId"])

try:
    client.context("")
except ContinuatorAPIError as exc:
    print(exc.status_code, exc.message)  # 400 + server error string
```

The server must already be running. The SDK uses stdlib HTTP only (no extra packages).

### 8. Cleanup

Stop the server (`Ctrl+C`), then:

```bash
rm -f data/continuator.db          # or data/quickstart.db if you overrode the path
```

## Database configuration

| Variable | Role |
| --- | --- |
| `CONTINUATOR_DB_PATH` | **Canonical** product SQLite path (import + CLI ingest + Settings) |
| `CONTINUATOR_IMPORT_DB` | Legacy alias; used only when `CONTINUATOR_DB_PATH` is unset |

**Precedence:** explicit `--db` / `db_path=` → `CONTINUATOR_DB_PATH` → `CONTINUATOR_IMPORT_DB` → `data/continuator.db`.

When both variables are set to different paths, **`CONTINUATOR_DB_PATH` wins** for import and ingest.

CLI and `serve` also load a cwd/repo-root `.env` without overriding existing shell variables.

## Optional: answer with Ollama

```bash
ollama pull qwen2.5:1.5b
curl -sS -X POST http://127.0.0.1:8765/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is my usual ferry?","retrieval_mode":"baseline","generate":true}' \
  | python -m json.tool
```

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `/context` looks fine but data is unfamiliar | Still on **fixture** — check `GET /health` → `origin` must be `imported` |
| Empty `documentEvidence` after ingest | Used CLI ingest against a different DB, or ingested before import — use HTTP/SDK ingest after import |
| `product SQLite store is required` | Still on fixture — run `POST /import` (or SDK `import_conversation`) first |
| Import mentions Ollama | Use `"extract":"messages"` |
| `/context` → 503 about embedding model | Hub/network/cache failure — see error text (`hf download …`) |
| Port in use | Change `--port` |
| Restart shows fixture after import | Used `--sample fixture` or empty DB — omit `--sample` or use `--sample db`; confirm `origin=imported` |
| `405 method not allowed` | Wrong HTTP verb for that path — see response `Allow` header |
| `request body too large` | JSON routes max 512 KB; `/import` max 8 MB |

## Next reading

- [`../README.md`](../README.md)
- [`context-pack-schema.md`](context-pack-schema.md) — Context Pack schema v1
- [`../examples/external_llm/`](../examples/external_llm/) — Context Pack → external OpenAI-compatible LLM
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
