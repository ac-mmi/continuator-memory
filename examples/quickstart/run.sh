#!/usr/bin/env bash
# End-to-end Continuator quickstart (no Ollama).
# Prerequisites: repo root, Python 3.10+, network once for pip + MiniLM cache.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -c "import continuator_memory" 2>/dev/null || pip install -e .

mkdir -p data
PORT="${CONTINUATOR_QUICKSTART_PORT:-8765}"
DB="${CONTINUATOR_DB_PATH:-data/quickstart.db}"
export CONTINUATOR_DB_PATH="$DB"
# Legacy alias kept in sync for older tooling; CONTINUATOR_DB_PATH is enough.
export CONTINUATOR_IMPORT_DB="$DB"

echo "Starting server on 127.0.0.1:${PORT} (db=${DB})…"
continuator-memory serve --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 0.1
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null

echo "Importing sample conversation…"
python examples/quickstart/import_payload.py | curl -sf -X POST "http://127.0.0.1:${PORT}/import" \
  -H 'Content-Type: application/json' \
  -d @- >/tmp/continuator-quickstart-import.json

echo "Verifying health (origin must be imported, not fixture)…"
python -c "
import json, urllib.request
h = json.load(urllib.request.urlopen('http://127.0.0.1:${PORT}/health'))
assert h.get('origin') == 'imported' and h.get('sampleId') == 'imported', h
print('health ok: origin=%s conversationId=%s memories=%s' % (
    h.get('origin'), h.get('conversationId'), h.get('nMemories')))
"

echo "Ingesting sample document via HTTP (same Workspace as /context)…"
python -c "
import json, urllib.request
from pathlib import Path
body = json.dumps({
    'text': Path('examples/quickstart/policy.md').read_text(encoding='utf-8'),
    'filename': 'policy.md',
    'sourceUri': 'inline://quickstart-policy',
}).encode()
req = urllib.request.Request(
    'http://127.0.0.1:${PORT}/documents/ingest',
    data=body,
    method='POST',
    headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
)
print(json.load(urllib.request.urlopen(req)))
"

echo "Retrieving Context Pack…"
curl -sf -X POST "http://127.0.0.1:${PORT}/context" \
  -H 'Content-Type: application/json' \
  -d '{"query":"usual ferry","mode":"baseline"}' \
  | python -m json.tool | head -n 40

echo
echo "Document-oriented pack…"
curl -sf -X POST "http://127.0.0.1:${PORT}/context" \
  -H 'Content-Type: application/json' \
  -d '{"query":"access tokens expire","mode":"baseline"}' \
  | python -c 'import sys,json; d=json.load(sys.stdin); print("documentEvidence:", len(d.get("documentEvidence") or []), "memories:", len(d.get("memories") or []))'

echo
echo "Quickstart OK. DB: ${DB}  (rm -f ${DB} to clean up)"
