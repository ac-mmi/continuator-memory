"""Continuator V2 semantic ingestion (phase 1): atomization, validation, storage.

Isolated from V1: V1 modules are imported, never modified. Raw messages and the chunk/vector
index are still written by the V1 ``Store``/``Pipeline``; V2 adds claims and explicit
relations on top, each traceable to its source message.

Phase 2A (``consolidation``, ``memory``, ``memory_store``) deterministically consolidates the
validated claim stream into versioned memories; phase 2B (``temporal``) separates observation
time from stated event/valid time and gates currentness on temporal meaning; phase 2C
(``identity``) resolves entities, slots and relation targets only from explicit claim data.
None of them adds an LLM call.

Phase 3A (``memory_retrieval``) ranks consolidated memories for a question by embedding
similarity with status-aware eligibility; it retrieves no raw chunks and generates no answer.
Phase 3B (``hybrid_retrieval``) fuses V1 raw chunk retrieval with phase 3A memories into one
message-level evidence list, expanding memories only through their own source messages.

Phase 5A (``context``, ``context_store``) contextualizes each validated claim against its source
window and the speaker names its conversation establishes, without changing claims, memories or
retrieval: a contextual memory is a consolidated memory plus references to the original source
windows, and ``global_memory_id`` gives it an identifier unique across conversations.

Phase 5B (``entities``, ``entity_store``) groups those memories by conversation-local entities:
speakers from phase 5A, other names from phase 2C identity keys and explicit aliases only (no
fuzzy, embedding or LLM merging). Each memory references its entities as speaker, subject,
possessor or mentioned; unresolvable references stay explicitly ambiguous. Retrieval and answer
generation do not consume entities yet.

Phase 5C (``relationships``, ``relationship_store``) adds a per-conversation relationship graph:
the phase 2A-2C links unchanged, plus additive EXTENDS (newer -> extended memory) and DERIVES
(derived -> each supporting memory) edges created only from explicit, validated evidence. They
never change memory status, are stored apart from the version links retrieval reads, and are not
consumed by retrieval or answer generation yet.

Phase 5D (``expansion``, ``expansion_store``) assembles query-time context from an existing
retriever's results: the seeds stay first and in order, followed by at most a few one-hop
neighbours over the phase 5C graph chosen by a fixed per-type, per-direction policy, then the
deduplicated phase 5A source windows with speaker labels and the phase 5B entity references.
It does not change retrieval and is not used by answer generation yet.
"""
