# Context Pack schema v1

Public contract for `POST /context` (`ContextPack.to_dict()`).

This document freezes the **actual** serialized shape. It does not invent fields or
change retrieval behavior.

Authoritative implementation: `src/continuator_memory/webapi/context_pack.py`.

## 1. Schema version and compatibility

| Item | Value |
| --- | --- |
| Current version | `"1"` (`schemaVersion` on every pack) |
| Negotiation | None — one live version |
| Additive fields | Allowed within v1 if optional / default-safe |
| Breaking | Rename/remove fields; change types or nullability; change identifier semantics; merge conversation and document evidence channels |

**Compatibility policy**

- Existing field **names** and **types** in this document remain stable for v1.
- Additive optional fields may be introduced within v1 without a version bump.
- Removing or renaming fields requires a new schema version.
- Changing identifier semantics is breaking.
- Conversation-message `evidence` and document `documentEvidence` must remain separate.
- Retrieval implementation details (scores, mode-specific diagnostics, planner output) are not automatic public guarantees unless listed as **stable** below.

## 2. Naming quirks (intentional in v1)

- Memory objects use **`id`**, not `memoryId` (citations use `memoryId`).
- `selection` uses **snake_case** keys (`budget_chars`, `dropped_by_budget`, …).
- Citation `unresolved[]` entries use **snake_case** (`claim_id`, `message_id`, `reason`).
- Document hits never appear inside conversation `evidence[]`.

## 3. Top-level field reference

Keys always present on every successful pack:

| Field | Type | Null? | Stability |
| --- | --- | --- | --- |
| `schemaVersion` | string | no | **Stable** — always `"1"` |
| `query` | string | no | **Stable** |
| `mode` | string | no | **Stable** (mode vocabulary may grow) |
| `conversationId` | string \| null | yes | **Stable** |
| `memories` | array | no (may be `[]`) | **Stable** |
| `evidence` | array | no (may be `[]`) | **Stable** — conversation messages only |
| `citations` | array | no (may be `[]`) | **Stable** |
| `documentEvidence` | array | no (may be `[]`) | **Stable** |
| `documentCitations` | array | no (may be `[]`) | **Stable** |
| `formattedContext` | string | no | **Stable** |
| `selection` | object | no | **Stable shape**; snake_case preserved |
| `seedMemoryIds` | string[] | no | Advanced |
| `expandedMemoryIds` | string[] | no | Advanced |
| `hops` | object[] | no | Advanced / best-effort |
| `diagnostics` | object | no | Core keys stable; mode keys advanced |
| `queryPlan` | object \| null | yes | Advanced |
| `answerPrompt` | object | no | Answer-path scaffolding (optional for consumers) |

**Conditional advanced top-level fields:** non-null keys from internal query diagnostics (except `queryPlan`) may also appear at the **top level** (for example rankings / iterative metadata). Treat these as **best-effort / advanced** — they are preserved for compatibility but are not a guaranteed closed set.

## 4. Memory item (`memories[]`)

| Field | Type | Notes |
| --- | --- | --- |
| `id` | string | Memory identifier |
| `text` | string | Statement |
| `rank` | number | Rank among hits |
| `score` | number | Retrieval score |
| `similarity` | number | Similarity component |
| `status` | string | Memory status |
| `sourceMessageIds` | string[] | Resolved provenance message ids |
| `sources[]` | object | `{ messageId, claimId, role, text, timestamp }` — `timestamp` may be null |

## 5. Conversation evidence (`evidence[]`)

Message excerpts only (never document units):

| Field | Type | Notes |
| --- | --- | --- |
| `messageId` | string | |
| `role` | string | |
| `text` | string | |
| `memoryIds` | string[] | |
| `truncated` | boolean | |
| `timestamp` | string \| null | |
| `inAnswerPrompt` | boolean | Present only on some budget-dropped rows (`false`) |

## 6. Memory citations (`citations[]`)

| Field | Type | Notes |
| --- | --- | --- |
| `memoryId` | string | |
| `messageIds` | string[] | |
| `unresolved[]` | object | `{ claim_id, message_id, reason }` (snake_case) |

## 7. Document evidence (`documentEvidence[]`)

| Field | Type |
| --- | --- |
| `evidenceId` | string |
| `documentId` | string |
| `sourceUri` | string |
| `sourceType` | string |
| `title` | string \| null-ish as stored |
| `headingPath` | string[] |
| `sectionId` | string |
| `charStart` | number |
| `charEnd` | number |
| `text` | string |
| `rank` | number |
| `score` | number |

## 8. Document citations (`documentCitations[]`)

| Field | Type |
| --- | --- |
| `evidenceId` | string |
| `documentId` | string |
| `sourceUri` | string |
| `sectionId` | string |
| `headingPath` | string[] |

## 9. Selection (`selection`)

Serialized as snake_case:

| Field | Type |
| --- | --- |
| `budget_chars` | number |
| `unique_sources` | string[] |
| `selected` | string[] |
| `dropped_by_budget` | string[] |
| `used_chars` | number |
| `first_memory` | object (messageId → memoryId) |

## 10. Diagnostics (`diagnostics`)

### Core keys (always set by `Workspace.context` for successful packs)

| Field | Meaning |
| --- | --- |
| `mode` | Retrieval mode |
| `k` | Memory top-k |
| `seedCount` / `expandedCount` | Seed vs expanded counts |
| `sourceCount` / `selectedSourceCount` / `droppedSourceCount` | Source budget stats |
| `budgetChars` / `usedChars` | Char budget |
| `promptMemoryCount` / `retrievedMemoryCount` | Answer-context sizing |
| `answerGeneration` | `"not_run"` on `/context` |
| `documentSearch` | `skipped` \| `bm25` \| `error` |
| `documentEvidenceCount` | Length of document hits |
| `documentTopK` | Document BM25 cutoff |

### Mode-specific / advanced (examples; not a closed list)

`lexicalWeight`, `contextualEnrichment`, `bm25`, `rrfC`, `roundCount`, `stopReason`, `union`, `rerank`, …

### Answer-generation path

`answerPrompt` and ask-nested evidence are for Continuator’s optional answering layer. External apps consuming `/context` can ignore them and use `formattedContext` or structured memories/evidence.

## 11. Identifiers and provenance

| Id | Typical use |
| --- | --- |
| `conversationId` | Active corpus; `GET /sources/{id}` |
| memory `id` / `memoryId` | Pack memories / citations; `GET /memories/{id}` |
| `messageId` | Conversation evidence / sources |
| `claimId` | Opaque claim link inside memory sources (no public claim GET) |
| `documentId` | `GET /documents/{id}` |
| `evidenceId` | `GET /evidence/{id}` |
| `sourceUri` / `sectionId` / `headingPath` | Document location within a file |

**Traversal**

1. Memory → `sourceMessageIds` / `citations` → message text in `evidence` or source inspection APIs.
2. Document hit → `evidenceId` / `documentId` → document/evidence GET endpoints.

## 12. Error responses

Envelope: `{"error": "<string>"}`.

| Situation | HTTP |
| --- | --- |
| Validation / corpus deleted / bad input / oversized or incomplete body | 400 |
| Missing memory/document/evidence/source | 404 |
| Unsupported method on a known path | 405 `method not allowed` (+ `Allow` header) |
| Embedding model unavailable | 503 (actionable message) |
| Unexpected server failure | 500 `"internal server error"` (details logged server-side) |

Body size limits: **512 KB** for standard JSON routes; **8 MB** for `POST /import`.

Document search failures do **not** fail the pack: `documentSearch` becomes `"error"` and document arrays stay empty.

## 13. Stable versus advanced summary

| Stable core | Advanced / best-effort | Answer-path |
| --- | --- | --- |
| `schemaVersion`, `query`, `mode`, `conversationId` | `seedMemoryIds`, `expandedMemoryIds`, `hops` | `answerPrompt` |
| `memories`, `evidence`, `citations` | Mode-specific `diagnostics.*` | Ask nesting under `/ask` |
| `documentEvidence`, `documentCitations` | Top-level query-diagnostic extras | |
| `formattedContext`, `selection` | `queryPlan` | |
| Core `diagnostics` incl. `documentSearch` | | |

## 14. Representative JSON (schema v1)

Illustrative baseline shape after conversation import + markdown ingest. Values abbreviated; structure matches serialization:

```json
{
  "schemaVersion": "1",
  "query": "access tokens expire",
  "mode": "baseline",
  "conversationId": "imported",
  "memories": [
    {
      "id": "mem:imported:m0000:c0",
      "text": "Our API access tokens expire after one hour. …",
      "rank": 1,
      "score": 0.51,
      "similarity": 0.51,
      "status": "current",
      "sourceMessageIds": ["imported:m0000"],
      "sources": [
        {
          "messageId": "imported:m0000",
          "claimId": "…",
          "role": "user",
          "text": "Our API access tokens expire after one hour. …",
          "timestamp": null
        }
      ]
    }
  ],
  "evidence": [
    {
      "messageId": "imported:m0000",
      "role": "user",
      "text": "Our API access tokens expire after one hour. …",
      "memoryIds": ["mem:imported:m0000:c0"],
      "truncated": false,
      "timestamp": null
    }
  ],
  "citations": [
    {
      "memoryId": "mem:imported:m0000:c0",
      "messageIds": ["imported:m0000"],
      "unresolved": []
    }
  ],
  "documentEvidence": [
    {
      "evidenceId": "ev:doc:…:u0002",
      "documentId": "doc:…",
      "sourceUri": "file:///…/policy.md",
      "sourceType": "markdown",
      "title": "Project Overview",
      "headingPath": ["Project Overview", "Authentication", "Token Refresh"],
      "sectionId": "…",
      "charStart": 0,
      "charEnd": 36,
      "text": "Access tokens expire after one hour.",
      "rank": 1,
      "score": 0.42
    }
  ],
  "documentCitations": [
    {
      "evidenceId": "ev:doc:…:u0002",
      "documentId": "doc:…",
      "sourceUri": "file:///…/policy.md",
      "sectionId": "…",
      "headingPath": ["Project Overview", "Authentication", "Token Refresh"]
    }
  ],
  "formattedContext": "RELEVANT LONG-TERM MEMORY\n…\nRELEVANT DOCUMENT EVIDENCE\n…\n",
  "selection": {
    "budget_chars": 8000,
    "unique_sources": ["imported:m0000"],
    "selected": ["imported:m0000"],
    "dropped_by_budget": [],
    "used_chars": 120,
    "first_memory": { "imported:m0000": "mem:imported:m0000:c0" }
  },
  "seedMemoryIds": ["mem:imported:m0000:c0"],
  "expandedMemoryIds": [],
  "hops": [],
  "diagnostics": {
    "mode": "baseline",
    "k": 5,
    "seedCount": 1,
    "expandedCount": 0,
    "sourceCount": 1,
    "selectedSourceCount": 1,
    "droppedSourceCount": 0,
    "budgetChars": 8000,
    "usedChars": 120,
    "promptMemoryCount": 1,
    "retrievedMemoryCount": 1,
    "answerGeneration": "not_run",
    "documentSearch": "bm25",
    "documentEvidenceCount": 1,
    "documentTopK": 10
  },
  "queryPlan": null,
  "answerPrompt": {
    "system": "…",
    "prompt": "…",
    "promptMemoryIds": ["mem:imported:m0000:c0"],
    "droppedMemoryIds": [],
    "droppedMessageIds": []
  }
}
```

## Related APIs

See root `README.md` for route overview. Document ingest remains CLI-based in this milestone (not part of the Context Pack schema).
