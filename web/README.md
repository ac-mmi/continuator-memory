# Continuator web

Vite + React UI for the Python API in `src/continuator_memory/webapi`. It isn't
part of the backend release.

With `VITE_CONTINUATOR_API_URL` set, the UI talks only to that API and shows an
error if it can't be reached. With it unset, the UI runs on built-in demo data,
and the sidebar says "Demo data".

## Run locally

**Terminal 1: backend** (from the repository root)

```sh
cp .env.example .env                  # first time; sets the Ollama model
.venv/bin/python -m continuator_memory serve --sample fixture --port 8765
```

**Terminal 2: frontend**

```sh
cd web
cp .env.example .env                  # first time; sets VITE_CONTINUATOR_API_URL
npm install                           # first time
npm run dev
```

Open <http://localhost:5173>. No proxy is needed: the API allows
`localhost:5173` and `127.0.0.1:5173` by default. Vite fails rather than moving
to another port, because other origins need `CONTINUATOR_WEB_CORS`. If the API
isn't on port 8765, change `VITE_CONTINUATOR_API_URL` and restart Vite.

| Variable | Where | Purpose |
| --- | --- | --- |
| `VITE_CONTINUATOR_API_URL` | `web/.env` | API URL. Required for real data. |
| `CONTINUATOR_OLLAMA_MODEL` | root `.env` | Model for answers. Answers need this or `CONTINUATOR_ANSWER_MODEL`. |
| `CONTINUATOR_DB_PATH` | root `.env` or shell | Canonical product SQLite path (import + document ingest). Default `data/continuator.db`. |
| `CONTINUATOR_IMPORT_DB` | root `.env` or shell | Legacy alias used only when `CONTINUATOR_DB_PATH` is unset. |
| `CONTINUATOR_WEB_CORS` | root `.env` or shell | Allowed origins, comma-separated. |
| `OLLAMA_HOST` | root `.env` or shell | Ollama URL. Default `http://localhost:11434`. |

Ollama is needed for answer generation, multi-query and iterative planning, and
the two model-based import modes. See the root `.env.example` for the rest.

## Check that it works

1. **Health:** `curl http://127.0.0.1:8765/health`. Settings shows the same values.
2. **Import:** on **Sources**, pick an extraction mode and import a `.txt` or
   `.json` conversation. Explicit speaker headers or JSON are recommended, since
   headerless parsing is best-effort and can misattribute speakers.
   - *One memory per message* is the default and needs no model.
   - *Memory extraction* is the opt-in V1 mode (`extract=memories`).
   - *V2 atomizer* records every proposition (`extract=v2`).
3. **Result:** Sources shows a **Memory ready** card with truthful counts from
   the API (messages, claims, persistent memories), persistence note, and next
   actions (Explore Memories / Ask & Answer / Retrieve Context). Failed imports
   show an error and keep the previous workspace — no success card.
4. **Memories:** open a memory to see its source message. V1 memories also show
   their exact type.
5. **Ask or Context:** on **Home**, choose **Ask & Answer** (`POST /ask`) or
   **Retrieve Context** (`POST /context`). Ask returns an answer plus evidence.
   Retrieve Context returns a Context Pack (memories, evidence, `formattedContext`)
   with **Copy context** and no answering LLM. If no model is available, Ask still
   retrieves; generation is marked not run.

To reopen an import after restarting the API:

```sh
.venv/bin/python -m continuator_memory serve --sample db --db data/continuator.db --port 8765
```

## Limitations

- The API holds one conversation at a time. Each import replaces the workspace
  and any existing import of the same database.
- Chats live in the server's memory and are lost on restart or import.
- The API starts on the Alex sample unless it's started with `--sample db`.
- `POST /import` stays open until extraction ends and can't be cancelled. If a
  browser gives up on a very long import, the server keeps working.
- B′ has no classified links for imports, so it matches baseline.

## Tests

```sh
npm run typecheck && npm test && npm run build
```

The contract test in `src/data/http/contract.test.ts` is skipped unless pointed
at a running API. It replaces that server's workspace, so use a throwaway one:

```sh
# from the repository root
CONTINUATOR_IMPORT_DB=/tmp/contract.db .venv/bin/python -m continuator_memory serve --port 8797
# from web/
VITE_CONTINUATOR_CONTRACT_API_URL=http://127.0.0.1:8797 npx vitest run contract
```
