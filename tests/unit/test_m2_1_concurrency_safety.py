"""M2.1: shared SQLite lock, atomic corpus deletion, Workspace invalidation."""

from __future__ import annotations

import threading
from threading import Event, Thread

import pytest

from continuator_memory.evidence.ingest import ingest_markdown
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.store import Store
from continuator_memory.v2.memory_store import MemoryStore
from continuator_memory.v2.store import SemanticStore
from continuator_memory.webapi.server import make_server
from continuator_memory.webapi.transcript import IMPORTED_CONVERSATION_ID
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder
from tests.unit.test_context_pack import TOKEN_DOC
from tests.unit.test_lifecycle_m2 import _workspace_with_docs
from tests.unit.test_webapi import FERRY_TRANSCRIPT, _http


def _counts(store: Store, cid: str) -> dict[str, int]:
    conn = store.connection
    def n(sql: str) -> int:
        return conn.execute(sql, (cid,)).fetchone()[0]

    return {
        "conversations": n(
            "SELECT COUNT(*) FROM conversations WHERE conversation_id = ?"
        ),
        "messages": n("SELECT COUNT(*) FROM messages WHERE conversation_id = ?"),
        "claims": n("SELECT COUNT(*) FROM v2_claims WHERE conversation_id = ?"),
        "memories": n("SELECT COUNT(*) FROM v2_memories WHERE conversation_id = ?"),
    }


# --- Connection concurrency -------------------------------------------------


def test_concurrent_context_and_document_deletion(tmp_path):
    ws, ingested = _workspace_with_docs(tmp_path)
    doc_id = ingested.document.document_id
    errors: list[BaseException] = []
    results: dict[str, object] = {}
    ready = Event()
    release = Event()

    def do_context():
        try:
            ready.set()
            release.wait(timeout=5)
            pack = ws.context("access tokens expire", retrieval_mode="baseline")
            results["pack"] = pack
        except BaseException as exc:  # noqa: BLE001 — collect for assertion
            errors.append(exc)

    def do_delete():
        try:
            ready.wait(timeout=5)
            release.set()
            results["delete"] = ws.delete_document(doc_id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = Thread(target=do_context)
    t2 = Thread(target=do_delete)
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)
    assert not errors, errors
    assert results["delete"]["deleted"] is True
    pack = results["pack"]
    assert pack is not None
    # After both complete, document is gone from the store.
    assert ws.get_document(doc_id) is None
    getattr(ws, "_store").close()


def test_concurrent_memory_list_and_corpus_deletion(tmp_path):
    ws, _ingested = _workspace_with_docs(tmp_path)
    cid = ws.conversation.conversation_id
    errors: list[BaseException] = []
    outcomes: list[str] = []
    start = Event()

    def do_list():
        try:
            start.wait(timeout=5)
            try:
                rows = ws.list_memories()
                outcomes.append(f"list:{len(rows)}")
            except ValueError as exc:
                if "corpus conversation has been deleted" in str(exc):
                    outcomes.append("list:refused")
                else:
                    raise
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def do_delete():
        try:
            start.wait(timeout=5)
            result = ws.delete_corpus_conversation(cid)
            outcomes.append(f"delete:{result['deleted']}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = Thread(target=do_list)
    t2 = Thread(target=do_delete)
    t1.start()
    t2.start()
    start.set()
    t1.join(timeout=30)
    t2.join(timeout=30)
    assert not errors, errors
    assert any(o.startswith("delete:True") for o in outcomes)
    assert any(o.startswith("list:") for o in outcomes)
    assert getattr(ws, "_corpus_deleted") is True
    getattr(ws, "_store").close()


def test_concurrent_import_and_retrieval(tmp_path):
    db = tmp_path / "race-import.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    ws.import_text(FERRY_TRANSCRIPT, filename="seed.txt", extract="messages", db_path=db)
    errors: list[BaseException] = []
    outcomes: list[str] = []
    start = Event()

    def do_context():
        try:
            start.wait(timeout=5)
            for _ in range(5):
                try:
                    pack = ws.context("usual ferry", retrieval_mode="baseline")
                    outcomes.append(f"ctx:{len(pack.evidence.matches)}")
                except ValueError as exc:
                    if "already running" in str(exc) or "deleted" in str(exc):
                        outcomes.append("ctx:skipped")
                    else:
                        raise
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def do_import():
        try:
            start.wait(timeout=5)
            result = ws.import_text(
                FERRY_TRANSCRIPT, filename="again.txt", extract="messages", db_path=db,
            )
            outcomes.append(f"import:{result['ok']}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = Thread(target=do_context)
    t2 = Thread(target=do_import)
    t1.start()
    t2.start()
    start.set()
    t1.join(timeout=60)
    t2.join(timeout=60)
    assert not errors, errors
    assert any(o == "import:True" for o in outcomes)
    assert any(o.startswith("ctx:") for o in outcomes)
    pack = ws.context("usual ferry", retrieval_mode="baseline")
    assert pack.evidence.matches
    getattr(ws, "_store").close()


def test_workspace_and_store_share_lock_identity(tmp_path):
    ws, _ = _workspace_with_docs(tmp_path)
    store = getattr(ws, "_store")
    assert ws._db_lock is store.db_lock

    held = Event()
    release = Event()
    errors: list[BaseException] = []

    def hold_lock():
        try:
            with store.db_lock:
                held.set()
                release.wait(timeout=5)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = Thread(target=hold_lock)
    t.start()
    assert held.wait(timeout=5)
    assert store.db_lock.acquire(blocking=False) is False
    assert ws._db_lock.acquire(blocking=False) is False
    release.set()
    t.join(timeout=5)
    assert not errors, errors
    assert store.db_lock.acquire(blocking=False) is True
    store.db_lock.release()
    store.close()


# --- Transaction atomicity --------------------------------------------------


@pytest.mark.parametrize(
    "fail_after",
    ["relationships", "entities", "memories", "optional", "conversation"],
)
def test_corpus_deletion_rolls_back_on_injected_failure(tmp_path, fail_after):
    ws, ingested = _workspace_with_docs(tmp_path)
    cid = ws.conversation.conversation_id
    store = getattr(ws, "_store")
    before = _counts(store, cid)
    assert before["conversations"] == 1
    assert before["messages"] >= 1
    assert before["claims"] >= 1
    assert before["memories"] >= 1
    doc_id = ingested.document.document_id

    ws._deletion_fail_after = fail_after
    with pytest.raises(RuntimeError, match="injected failure"):
        ws.delete_corpus_conversation(cid)

    after = _counts(store, cid)
    assert after == before
    assert store.get_conversation(cid) is not None
    assert MemoryStore(store.connection).has_snapshot(cid)
    assert SemanticStore(store.connection).has_claims(cid)
    assert EvidenceStore(store.connection).get_document(doc_id) is not None
    assert getattr(ws, "_corpus_deleted") is False
    # Workspace remains usable.
    pack = ws.context("usual ferry", retrieval_mode="baseline")
    assert pack.evidence.matches
    assert ws.list_memories()
    store.close()


def test_successful_corpus_deletion_commits_all_cleanup(tmp_path):
    ws, ingested = _workspace_with_docs(tmp_path)
    cid = ws.conversation.conversation_id
    store = getattr(ws, "_store")
    assert _counts(store, cid)["conversations"] == 1

    result = ws.delete_corpus_conversation(cid)
    assert result["deleted"] is True
    assert _counts(store, cid) == {
        "conversations": 0,
        "messages": 0,
        "claims": 0,
        "memories": 0,
    }
    assert EvidenceStore(store.connection).get_document(ingested.document.document_id) is not None
    assert getattr(ws, "_corpus_deleted") is True
    store.close()


# --- Workspace invalidation -------------------------------------------------


def test_corpus_endpoints_refuse_after_deletion(tmp_path):
    ws, ingested = _workspace_with_docs(tmp_path)
    cid = ws.conversation.conversation_id
    memory_id = next(iter(ws.consolidator.memories))
    ws.delete_corpus_conversation(cid)

    with pytest.raises(ValueError, match="corpus conversation has been deleted"):
        ws.context("usual ferry", retrieval_mode="baseline")
    with pytest.raises(ValueError, match="corpus conversation has been deleted"):
        ws.ask("usual ferry", retrieval_mode="baseline", generate=False)
    with pytest.raises(ValueError, match="corpus conversation has been deleted"):
        ws.list_memories()
    with pytest.raises(ValueError, match="corpus conversation has been deleted"):
        ws.get_memory(memory_id)
    with pytest.raises(ValueError, match="corpus conversation has been deleted"):
        ws.list_sources()
    with pytest.raises(ValueError, match="corpus conversation has been deleted"):
        ws.get_source(cid)
    with pytest.raises(ValueError, match="corpus conversation has been deleted"):
        ws.get_graph()

    health = ws.health()
    assert health["corpusDeleted"] is True
    assert health["deletedConversationId"] == cid
    assert health["nMemories"] == 0
    assert health["nMessages"] == 0

    # Documents / evidence remain.
    assert ws.list_documents()
    assert ws.get_document(ingested.document.document_id) is not None
    unit = ingested.units[0]
    assert ws.get_evidence_unit(unit.evidence_id) is not None
    getattr(ws, "_store").close()


def test_corpus_invalidation_http_contract(tmp_path):
    ws, ingested = _workspace_with_docs(tmp_path)
    cid = ws.conversation.conversation_id
    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = _http(httpd, "DELETE", f"/corpus/conversations/{cid}")
        assert status == 200
        assert body["deleted"] is True

        for method, path, payload in [
            ("POST", "/context", {"query": "ferry"}),
            ("POST", "/ask", {"question": "ferry", "generate": False}),
            ("GET", "/memories", None),
            ("GET", "/sources", None),
            ("GET", "/graph", None),
        ]:
            status, _, err = _http(httpd, method, path, payload)
            assert status == 400, path
            assert "corpus conversation has been deleted" in err["error"]

        status, _, health = _http(httpd, "GET", "/health")
        assert status == 200
        assert health["corpusDeleted"] is True

        status, _, docs = _http(httpd, "GET", "/documents")
        assert status == 200
        assert docs
        status, _, one = _http(httpd, "GET", f"/documents/{ingested.document.document_id}")
        assert status == 200
        status, _, ev = _http(httpd, "GET", f"/evidence/{ingested.units[0].evidence_id}")
        assert status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
        getattr(ws, "_store").close()


def test_store_db_lock_usable_without_workspace(tmp_path):
    """Standalone Store still exposes a lock for callers that share its connection."""
    product = Store(tmp_path / "standalone.db")
    assert isinstance(product.db_lock, type(threading.RLock()))
    with product.db_lock:
        EvidenceStore(product.connection)
        path = tmp_path / "x.md"
        path.write_text(TOKEN_DOC, encoding="utf-8")
        ingest_markdown(path, EvidenceStore(product.connection))
    product.close()
