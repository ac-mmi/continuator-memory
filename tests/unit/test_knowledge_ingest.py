"""Knowledge-document ingestion with a scripted model. These cover what code guarantees: ownership, citation
structure, review marking, failure handling and storage. They cannot show that a real model writes good documents."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from continuator_memory.chunking import chunk_conversation
from continuator_memory.generation import GenerationError
from continuator_memory.knowledge import documents as documents_module, ingest as ingest_module, store as store_module
from continuator_memory.knowledge.documents import (
    CONTEXT_ONLY_CITATION,
    MISSING_CITATION,
    NOT_A_BULLET,
    OUTSIDE_SECTION,
    REVIEW_EVIDENCE_CITED,
    REVIEW_NEEDED,
    SEMANTIC_NOT_VERIFIED,
    SYSTEM_PROMPT,
    TRUNCATED_DOCUMENT,
    UNKNOWN_CITATION,
    UNKNOWN_SECTION,
    build_prompt,
    labels,
    parse_document,
)
from continuator_memory.knowledge.ingest import (
    CHUNK_FALLBACK,
    CITED,
    CONTEXT_MESSAGES,
    MAX_PASSAGE_TOKENS,
    NUM_PREDICT,
    build_passages,
    estimate_tokens,
    ingest_documents,
)
from continuator_memory.knowledge.store import KD_SCHEMA, KnowledgeStore
from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from tests.helpers import HashingEmbedder

CID = "c1"
TS = "2024-05-10"


def msg(i, text=None, role="user", ts=TS):
    return Message(f"{CID}:m{i:04d}", CID, i, role, text or f"Ana: Line number {i}.", ts)


TARGETS = (msg(2, "Ana: I bake rye on Friday."), msg(3, "Bram: Do you sell it?", "assistant"))
CONTEXT = (msg(0, "Bram: Hello."), msg(1, "Ana: Hi."))


def conversation(n=10):
    return Conversation(CID, tuple(msg(i) for i in range(n)))


class Scripted:
    model_name = "scripted"

    def __init__(self, replies=None, fail=(), usage=None, blow_up=None):
        self.replies, self.fail, self.usage, self.blow_up = replies or {}, set(fail), usage or {}, blow_up or {}
        self.prompts, self.systems, self.options = [], [], []
        self.last_usage = None

    def generate(self, system, prompt, *, temperature=None, num_predict=None):
        index = len(self.prompts)
        self.prompts.append(prompt)
        self.systems.append(system)
        self.options.append((temperature, num_predict))
        if index in self.blow_up:
            raise self.blow_up[index]
        if index in self.fail:
            raise GenerationError(f"scripted failure {index}")
        self.last_usage = self.usage.get(index, {"prompt_eval_count": 10, "eval_count": 5, "done_reason": "stop"})
        return self.replies.get(index, "## Facts and preferences\n- A statement. [M1]\n")


@pytest.fixture
def scratch(tmp_path):
    s = Store(str(tmp_path / "scratch.db"))
    yield s
    s.close()


def run(scratch, client, n=10, **kwargs):
    report = ingest_documents(conversation(n), scratch, client, HashingEmbedder(), clock=lambda: "2030-01-01T00:00:00+00:00", **kwargs)
    return report, KnowledgeStore(scratch.connection)


# ---- prompt and ownership -----------------------------------------------------

def test_targets_and_context_have_separate_local_labels():
    assert {k: m.message_id for k, m in labels(TARGETS, CONTEXT).items()} == {
        "M1": TARGETS[0].message_id, "M2": TARGETS[1].message_id, "C1": CONTEXT[0].message_id, "C2": CONTEXT[1].message_id}
    prompt = build_prompt(TARGETS, CONTEXT)
    assert prompt.index("[C1]") < prompt.index("[M1]")
    assert CID + ":m" not in prompt


def test_one_call_per_chunk_and_each_message_owned_once(scratch):
    client = Scripted()
    report, _ = run(scratch, client)
    chunks = chunk_conversation(conversation(), 4, 0)
    assert report.model_calls == report.documents == len(client.prompts) == len(chunks) == 3
    owned = [m for c in chunks for m in c.source_message_ids]
    assert sorted(owned) == sorted(m.message_id for m in conversation().messages) and len(owned) == len(set(owned))


def test_first_chunk_has_no_context_and_later_chunks_have_two_raw_messages(scratch):
    client = Scripted()
    _, knowledge = run(scratch, client)
    assert "[C1]" not in client.prompts[0]
    docs = knowledge.get_documents(CID)
    assert docs[0]["context_message_ids"] == []
    assert docs[1]["context_message_ids"] == [f"{CID}:m0002", f"{CID}:m0003"]
    assert "[C2]" in client.prompts[1] and "[C3]" not in client.prompts[1]
    assert "Line number 3." in client.prompts[1] and "Line number 1." not in client.prompts[1]
    assert CONTEXT_MESSAGES == 2


def test_documents_are_independent_of_each_other(scratch):
    client = Scripted(replies={0: "## Facts and preferences\n- UNIQUE-EARLIER-DOCUMENT-TEXT. [M1]\n"})
    run(scratch, client)
    assert all("UNIQUE-EARLIER-DOCUMENT-TEXT" not in prompt for prompt in client.prompts[1:])


def test_generation_settings(scratch):
    client = Scripted()
    run(scratch, client, n=4)
    assert client.options == [(0.0, NUM_PREDICT)]
    assert client.systems == [SYSTEM_PROMPT]


# ---- structural parsing and review ---------------------------------------------

def parsed(text):
    return parse_document(text, TARGETS, CONTEXT)


def test_cited_bullet_resolves_to_original_message_ids():
    claim = parsed("## Facts and preferences\n- Ana bakes rye. [M1]\n").claims[0]
    assert (claim.section, claim.text, claim.citations) == ("Facts and preferences", "Ana bakes rye.", ("M1",))
    assert claim.message_ids == (TARGETS[0].message_id,)
    assert claim.review_status == REVIEW_EVIDENCE_CITED


def test_multiple_citations_including_context():
    claim = parsed("## Facts and preferences\n- Ana bakes rye. [M1, C2] [M2]\n").claims[0]
    assert claim.citations == ("M1", "C2", "M2")
    assert claim.message_ids == (TARGETS[0].message_id, CONTEXT[1].message_id, TARGETS[1].message_id)
    assert claim.text == "Ana bakes rye."


@pytest.mark.parametrize(("line", "reason"), [
    ("- No citation at all.", MISSING_CITATION),
    ("- Wrong marker style. (M1)", MISSING_CITATION),
    ("- Cites a message it never saw. [M9]", UNKNOWN_CITATION),
    ("- Only from context. [C1]", CONTEXT_ONLY_CITATION),
])
def test_citation_problems_keep_the_claim_and_mark_it_for_review(line, reason):
    claim = parsed(f"## Facts and preferences\n{line}\n").claims[0]
    assert claim.text
    assert reason in claim.review_reasons
    assert claim.review_status == REVIEW_NEEDED


def test_unknown_citation_keeps_the_valid_ones():
    claim = parsed("## Facts and preferences\n- Partly valid. [M1, M9]\n").claims[0]
    assert claim.message_ids == (TARGETS[0].message_id,)
    assert claim.review_reasons == (UNKNOWN_CITATION,)


def test_structure_problems_keep_the_claim():
    result = parsed("- Before any heading. [M1]\n\n## Invented Heading\n- Under a new heading. [M1]\n\n## Relationships\nStray paragraph. [M2]\n")
    assert [c.text for c in result.claims] == ["Before any heading.", "Under a new heading.", "Stray paragraph."]
    assert result.claims[0].review_reasons == (OUTSIDE_SECTION,)
    assert result.claims[1].review_reasons == (UNKNOWN_SECTION,)
    assert result.claims[2].review_reasons == (NOT_A_BULLET,)


def test_continuation_lines_join_their_bullet():
    claim = parsed("## Facts and preferences\n- Ana bakes\n  rye on Friday. [M1]\n").claims[0]
    assert claim.text == "Ana bakes rye on Friday."
    assert claim.citations == ("M1",)


def test_code_fence_is_unwrapped_and_recorded():
    result = parsed("```markdown\n## Facts and preferences\n- Ana bakes rye. [M1]\n```")
    assert result.claims[0].text == "Ana bakes rye."
    assert result.issues == ["WRAPPED_IN_CODE_FENCE"]
    assert result.content.startswith("## Facts")


def test_targets_nobody_cited_are_reported():
    result = parsed("## Facts and preferences\n- Ana bakes rye. [M1]\n")
    assert result.uncited_target_ids == (TARGETS[1].message_id,)


def test_nothing_durable_is_explicit():
    result = parsed("NOTHING_DURABLE")
    assert result.nothing_durable and result.claims == []


@pytest.mark.parametrize("text", [
    "Ana never bakes rye.",            # flips the source
    "Ana bakes sourdough in Lisbon.",  # shares almost no words with the source
    "Ana enjoys making bread weekly.", # a fair paraphrase
])
def test_semantics_are_never_judged_by_code(text):
    """Both directions: no lexical rule rejects a paraphrase, and none can catch a wrong claim. It says so."""
    claim = parsed(f"## Facts and preferences\n- {text} [M1]\n").claims[0]
    assert claim.review_reasons == ()
    data = claim.to_dict()
    assert data["semantic_support"] == SEMANTIC_NOT_VERIFIED
    assert not {"confirmed", "verified", "true", "supported"} & set(data)


def test_no_uncertain_wording_is_rewritten():
    claim = parsed("## Plans and timelines\n- Possibly: Ana might bake rye on Friday. [M1]\n").claims[0]
    assert claim.text == "Possibly: Ana might bake rye on Friday."


# ---- document statuses ----------------------------------------------------------

def test_stop_is_ok_and_stored_with_metadata(scratch):
    _, knowledge = run(scratch, Scripted(), n=4)
    doc = knowledge.get_documents(CID)[0]
    assert doc["status"] == "ok"
    assert doc["document_id"] == f"{CID}:c0000:kd-prompt-v1"
    assert (doc["chunk_index"], doc["model_name"], doc["format_version"]) == (0, "scripted", "kd-markdown-v1")
    assert doc["created_at"] == "2030-01-01T00:00:00+00:00"
    assert doc["content_hash"] and doc["source_text_hash"]
    assert doc["usage"]["done_reason"] == "stop"


def test_truncation_uses_done_reason_keeps_text_and_makes_no_passages(scratch):
    text = "## Facts and preferences\n- Complete claim. [M1]\n- Cut off in the mid"
    client = Scripted(replies={0: text}, usage={0: {"prompt_eval_count": 1, "eval_count": 4096, "done_reason": "length"}})
    report, knowledge = run(scratch, client)
    doc = knowledge.get_documents(CID)[0]
    assert doc["status"] == "truncated" and "done_reason=length" in doc["error"]
    assert doc["content_markdown"] == text
    assert all(TRUNCATED_DOCUMENT in c["review_reasons"] for c in doc["claims"]) and len(doc["claims"]) == 2
    assert knowledge.get_passages(doc["document_id"]) == []
    assert (report.truncated, report.documents) == (1, 3)


def test_a_complete_looking_document_is_still_truncated_if_the_model_says_so(scratch):
    client = Scripted(usage={0: {"done_reason": "length"}})
    _, knowledge = run(scratch, client, n=4)
    assert knowledge.get_documents(CID)[0]["status"] == "truncated"


def test_missing_done_reason_is_recorded_not_assumed(scratch):
    _, knowledge = run(scratch, Scripted(usage={0: {"eval_count": 5}}), n=4)
    assert "DONE_REASON_MISSING" in knowledge.get_documents(CID)[0]["issues"]


@pytest.mark.parametrize("error", [GenerationError("Ollama down"), OSError("reset"), TimeoutError("slow")])
def test_failed_call_is_recorded_and_later_chunks_continue(scratch, error):
    client = Scripted(blow_up={0: error})
    report, knowledge = run(scratch, client)
    docs = knowledge.get_documents(CID)
    assert docs[0]["status"] == "failed" and str(error) in docs[0]["error"]
    assert docs[0]["content_markdown"] is None
    assert docs[1]["status"] == "ok"
    assert (report.failed, report.ok, report.model_calls) == (1, 2, 3)
    assert report.errors


def test_empty_reply_is_a_failure(scratch):
    report, knowledge = run(scratch, Scripted(replies={0: "   "}), n=4)
    assert knowledge.get_documents(CID)[0]["status"] == "failed"
    assert report.failed == 1


def test_nothing_durable_is_a_status_with_no_passages(scratch):
    report, knowledge = run(scratch, Scripted(replies={0: "NOTHING_DURABLE"}), n=4)
    doc = knowledge.get_documents(CID)[0]
    assert doc["status"] == "nothing_durable"
    assert knowledge.get_passages(doc["document_id"]) == []
    assert report.nothing_durable == 1


def test_an_unexpected_exception_is_not_swallowed(scratch):
    with pytest.raises(KeyError):
        run(scratch, Scripted(blow_up={0: KeyError("bug")}))


def test_review_claims_are_counted_and_preserved(scratch):
    reply = "## Facts and preferences\n- Fine. [M1]\n- No citation.\n- Bad citation. [M9]\n"
    report, knowledge = run(scratch, Scripted(replies={0: reply}), n=4)
    doc = knowledge.get_documents(CID)[0]
    assert len(doc["claims"]) == 3
    assert (report.claims, report.claims_needing_review) == (3, 2)


# ---- passages and embeddings -----------------------------------------------------

def passages_for(reply, n=4):
    chunk = chunk_conversation(conversation(n), 4, 0)[0]
    targets = conversation(n).messages[:4]
    return build_passages("doc", parse_document(reply, targets).claims, chunk, TS), chunk


def test_sections_become_separate_passages_with_a_generated_prefix():
    passages, _ = passages_for("## Facts and preferences\n- One. [M1]\n- Two. [M2]\n\n## Relationships\n- Three. [M3]\n")
    assert [p.heading_path for p in passages] == ["Facts and preferences", "Relationships"]
    assert passages[0].content == "- One.\n- Two."
    assert passages[0].embedded_text == f"Facts and preferences | {TS}\n- One.\n- Two."
    assert passages[0].message_ids == (f"{CID}:m0000", f"{CID}:m0001")
    assert passages[0].citation_source == CITED
    assert "[M1]" not in passages[0].embedded_text


def test_long_sections_split_between_claims_never_inside_one():
    bullets = "\n".join(f"- Statement {i} {'word ' * 30}. [M1]" for i in range(10))
    passages, _ = passages_for(f"## Facts and preferences\n{bullets}\n")
    assert len(passages) > 1
    assert all(p.token_estimate <= MAX_PASSAGE_TOKENS + estimate_tokens(f"Facts and preferences | {TS}\n") for p in passages)
    assert [i for p in passages for i in p.claim_indexes] == list(range(10))
    for p in passages:
        assert all(line.endswith(".") for line in p.content.splitlines())


def test_oversized_single_claim_stays_whole():
    long_claim = "word " * 400
    passages, _ = passages_for(f"## Facts and preferences\n- {long_claim}. [M1]\n")
    assert len(passages) == 1
    assert passages[0].token_estimate > MAX_PASSAGE_TOKENS


def test_uncited_claim_gets_its_own_passage_and_never_borrows_a_neighbours_messages():
    passages, chunk = passages_for("## Facts and preferences\n- Cited one. [M1]\n- Uncited.\n- Cited two. [M2]\n")
    assert [p.claim_indexes for p in passages] == [(0,), (1,), (2,)]
    assert passages[0].message_ids == (f"{CID}:m0000",)
    assert passages[1].message_ids == chunk.source_message_ids
    assert (passages[1].citation_source, passages[1].review_status) == (CHUNK_FALLBACK, REVIEW_NEEDED)
    assert passages[2].message_ids == (f"{CID}:m0001",)


def test_passages_and_embeddings_are_stored_with_the_embedder_name(scratch):
    embedder = HashingEmbedder()
    ingest_documents(conversation(4), scratch, Scripted(), embedder)
    knowledge = KnowledgeStore(scratch.connection)
    passage = knowledge.get_passages(knowledge.get_documents(CID)[0]["document_id"])[0]
    vectors = knowledge.load_embeddings([passage["passage_id"]], embedder.model_name)
    assert np.allclose(vectors[passage["passage_id"]], embedder.embed([passage["embedded_text"]])[0])
    assert knowledge.load_embeddings([passage["passage_id"]], "other-model") == {}


def test_passage_message_ids_resolve_to_original_messages(scratch):
    _, knowledge = run(scratch, Scripted(), n=4)
    passage = knowledge.get_passages(knowledge.get_documents(CID)[0]["document_id"])[0]
    rows = scratch.get_messages(CID, passage["message_ids"])
    assert [m.content for m in rows] == ["Ana: Line number 0."]


# ---- storage safety ------------------------------------------------------------

def test_exactly_three_additive_tables():
    assert KD_SCHEMA.count("CREATE TABLE") == 3
    for table in ("kd_documents", "kd_passages", "kd_passage_embeddings"):
        assert table in KD_SCHEMA


def test_save_is_atomic(scratch):
    class Bad(HashingEmbedder):
        calls = 0

        def embed(self, texts):
            self.calls += 1
            # The raw chunks embed normally. The passages come back short, failing mid-save.
            return super().embed(texts) if self.calls == 1 else np.zeros((0, self.dim), np.float32)

    with pytest.raises(IndexError):
        ingest_documents(conversation(4), scratch, Scripted(), Bad())
    assert scratch.connection.execute("SELECT COUNT(*) FROM kd_documents").fetchone()[0] == 0
    assert scratch.connection.execute("SELECT COUNT(*) FROM kd_passages").fetchone()[0] == 0


def test_rerun_needs_replace_and_replace_is_clean(scratch):
    run(scratch, Scripted(), n=4)
    with pytest.raises(ValueError, match="replace=True"):
        run(scratch, Scripted(), n=4)
    run(scratch, Scripted(replies={0: "## Relationships\n- New. [M1]\n"}), n=4, replace=True)
    knowledge = KnowledgeStore(scratch.connection)
    doc = knowledge.get_documents(CID)[0]
    assert doc["claims"][0]["text"] == "New."
    assert scratch.connection.execute("SELECT COUNT(*) FROM kd_passages").fetchone()[0] == 1


def test_existing_raw_data_is_never_rewritten(scratch):
    conv = conversation(4)
    chunks = chunk_conversation(conv, 4, 0)
    embedder = HashingEmbedder()
    scratch.save_conversation(conv, chunks, embedder.embed([c.text for c in chunks]), embedder.model_name)
    scratch.connection.execute("UPDATE conversations SET title = 'kept' WHERE conversation_id = ?", (CID,))
    scratch.connection.commit()
    before = scratch.connection.execute("SELECT vector FROM chunk_embeddings").fetchall()
    ingest_documents(conv, scratch, Scripted(), embedder)
    assert scratch.get_conversation(CID).title == "kept"
    assert scratch.connection.execute("SELECT vector FROM chunk_embeddings").fetchall() == before


def test_different_messages_or_chunking_are_refused(scratch):
    conv = conversation(8)
    embedder = HashingEmbedder()
    overlap = chunk_conversation(conv, 4, 2)
    scratch.save_conversation(conv, overlap, embedder.embed([c.text for c in overlap]), embedder.model_name)
    with pytest.raises(ValueError, match="different chunks"):
        ingest_documents(conv, scratch, Scripted(), embedder)
    changed = Conversation(CID, (msg(0, "Ana: Something else."),))
    with pytest.raises(ValueError, match="different messages"):
        ingest_documents(changed, scratch, Scripted(), embedder)


def test_raw_chunks_stay_in_their_own_retrieval_tables(scratch):
    run(scratch, Scripted(), n=4)
    assert scratch.connection.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] == 1
    ids, _ = scratch.load_embeddings(HashingEmbedder().model_name, CID)
    assert ids == [f"{CID}:c0000"]
    tables = {r[0] for r in scratch.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not {t for t in tables if t.startswith(("v2_", "sx_"))}


def test_foreign_keys_cascade_from_chunks(scratch):
    _, knowledge = run(scratch, Scripted(), n=4)
    with scratch.connection:
        scratch.connection.execute("DELETE FROM conversations WHERE conversation_id = ?", (CID,))
    for table in ("kd_documents", "kd_passages", "kd_passage_embeddings"):
        assert scratch.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_store_requires_the_raw_tables():
    connection = sqlite3.connect(":memory:")
    KnowledgeStore(connection)
    with pytest.raises(sqlite3.IntegrityError):
        with connection:
            connection.execute("INSERT INTO kd_passages (passage_id, document_id, conversation_id, passage_index, heading_path, content, "
                               "embedded_text, claim_indexes_json, message_ids_json, citation_source, review_status, token_estimate) "
                               "VALUES ('p', 'missing', 'c', 0, 'h', 'c', 'e', '[]', '[]', 'cited', 'x', 1)")


# ---- the design rules ----------------------------------------------------------

def imports_of(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
    return names


@pytest.mark.parametrize("module", [documents_module, ingest_module, store_module])
def test_no_lexical_support_logic_or_dependency_on_other_experiments(module):
    names = imports_of(module)
    for banned in ("tokens", "content_tokens", "lexical_overlap", "classify_support", "stem", "FRAME_TOKENS"):
        assert banned not in names
    for name in names:
        assert not name.startswith(("continuator_memory.v2", "continuator_memory.memory", "continuator_memory.webapi"))


def test_other_experiments_do_not_import_this_one():
    root = Path(documents_module.__file__).resolve().parents[1]
    for path in [*root.glob("v2/*.py"), *root.glob("webapi/*.py"), *root.glob("memory/*.py"), root / "cli.py", root / "retrieval.py", root / "store.py"]:
        assert "continuator_memory.knowledge" not in path.read_text(encoding="utf-8"), path


def test_regexes_only_read_structure():
    source = Path(documents_module.__file__).read_text(encoding="utf-8")
    patterns = [n for n in ("_HEADING", "_BULLET", "_CONTINUATION", "_CITATION", "_FENCE") if n in source]
    assert len(patterns) == 5
    assert source.count("re.compile") == 5


def test_prompt_has_no_evaluation_entities_and_its_example_parses_cleanly():
    for name in ("Maya", "Leo", "Trailmark", "FieldNotes", "Northstar", "GreenCurrent", "Hawthorne", "Pixel"):
        assert name not in SYSTEM_PROMPT
    example = SYSTEM_PROMPT.split("Document:\n", 1)[1]
    targets = (msg(1, "Rafi: x", "assistant"), msg(2, "Ines: y"))
    result = parse_document(example, targets, (msg(0, "Ines: z"),))
    assert len(result.claims) == 6
    assert [c.review_reasons for c in result.claims] == [()] * 6


# ---- the runner: nothing reaches a model without --live ---------------------------

def load_runner():
    path = Path(documents_module.__file__).resolve().parents[3] / "scripts/run_knowledge_ingest.py"
    spec = importlib.util.spec_from_file_location("run_knowledge_ingest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_input(tmp_path, n=8):
    source = tmp_path / "conv.json"
    source.write_text(json.dumps({"conversation_id": "x", "messages": [
        {"id": f"x:m{i:04d}", "role": "user", "content": f"Ana: Line {i}.", "timestamp": TS} for i in range(n)]}), encoding="utf-8")
    return source


def test_runner_refuses_without_an_explicit_mode(tmp_path):
    with pytest.raises(SystemExit) as exc:
        load_runner().main(["--input", str(write_input(tmp_path)), "--output-root", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


def test_runner_refuses_both_modes(tmp_path):
    with pytest.raises(SystemExit):
        load_runner().main(["--dry-run", "--live", "--input", str(write_input(tmp_path))])


def test_runner_dry_run_writes_artifacts_with_no_network(tmp_path, capsys):
    root = tmp_path / "out"
    assert load_runner().main(["--dry-run", "--smoke", "--input", str(write_input(tmp_path)), "--output-root", str(root)]) == 0
    run_dir = next(root.iterdir())
    for name in ("config.json", "run.json", "metrics.jsonl", "eval.db", "system_prompt.txt", "prompt_c01.txt", "document_c01.md",
                 "claims_c01.json", "code/documents.py", "code/ingest.py", "code/store.py", "code/run_knowledge_ingest.py"):
        assert (run_dir / name).exists(), name
    output = capsys.readouterr().out
    assert "STATUS: OK" in output and f"ARTIFACTS: {run_dir}" in output
    row = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[1])
    for key in ("done_reason", "input_tokens", "output_tokens", "prompt_eval_s", "generation_s", "total_ollama_s", "wall_s", "claims", "claims_needing_review", "passages", "uncited_targets", "status", "error"):
        assert key in row
    with Store(str(run_dir / "eval.db")) as store:
        docs = KnowledgeStore(store.connection).get_documents("x")
    assert [d["status"] for d in docs] == ["ok", "ok"]


def test_runner_live_path_is_stopped_by_the_guard(tmp_path, monkeypatch):
    """--live refuses unknown input before any request; with the right hash it would hit the guard, not Ollama."""
    runner = load_runner()
    source = write_input(tmp_path)
    assert runner.main(["--live", "--input", str(source), "--output-root", str(tmp_path / "o")]) == 1
    source = write_input(tmp_path, 62)
    monkeypatch.setattr(runner, "EXPECTED_INPUT_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(pytest.fail.Exception):
        runner.main(["--live", "--input", str(source), "--output-root", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()
