"""V1 release blockers for extract="memories": prompt content, failure-safe re-import, fail-safe validation."""

from __future__ import annotations

import json
import urllib.request

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from continuator_memory.v2 import memory_prompt
from continuator_memory.v2.atomizer import (
    MEMORY_PROTOCOL,
    AtomizationRequestError,
    OllamaChatClient,
    SemanticAtomizer,
)
from continuator_memory.v2.memory_records import memory_records
from continuator_memory.v2.models import SUPPORT_SOURCE
from continuator_memory.v2.store import SemanticStore
from continuator_memory.v2.validation import classify_support, validate_atomization
from continuator_memory.v2.windowing import ContextWindow
from continuator_memory.webapi.extract import ingest_imported
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder

CID = "c"
ROWS = [("m1", "user", "I live in Porto."), ("m2", "user", "I hate early shifts.")]
GOOD = {"m1": ["The user lives in Porto."], "m2": ["The user hates early shifts."]}


class Client:
    """Claims per message; ``raw`` overrides the output, ``fail_at`` raises from the Nth call (1-based)."""

    model_name = "scripted"

    def __init__(self, claims=None, raw=None, fail_at=None, error=AtomizationRequestError):
        self.claims, self.raw, self.fail_at, self.error, self.calls = claims or {}, raw, fail_at, error, 0

    def complete_json(self, system, prompt, schema, examples=()):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise self.error("scripted failure")
        if self.raw is not None:
            return self.raw
        out = []
        for mid in schema["properties"]["messages"]["items"]["properties"]["source_message_id"]["enum"]:
            contents = self.claims.get(mid, [])
            out.append({"source_message_id": mid, "status": "claims" if contents else "no_durable_assertion",
                        "explicit_relations": [],
                        "claims": [{"claim_id": f"{mid}:c{i}", "content": text, "memory_type": "personal_fact",
                                    "source_message_id": mid} for i, text in enumerate(contents, 1)]})
        return json.dumps({"messages": out})


def conv(rows=ROWS):
    return Conversation(CID, tuple(Message(mid, CID, i, role, text, None) for i, (mid, role, text) in enumerate(rows)))


def ingest(store, client, rows=ROWS):
    return ingest_imported(conv(rows), store, extract="memories", atomizer=SemanticAtomizer(client, MEMORY_PROTOCOL))


def state(store):
    """Everything an import may touch, read back from the database."""
    tables = ("conversations", "messages", "v2_semantic_messages", "v2_claims", "v2_relations", "v2_atomization_log")
    return {t: store.connection.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall() for t in tables}


def contents(store):
    return [c.content for c in SemanticStore(store.connection).get_claims(CID)]


@pytest.fixture
def seeded(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    assert ingest(store, Client(GOOD)).kept_existing is False
    yield store
    store.close()


# ---- Task 1: prompt content ---------------------------------------------------------------------

# Speakers of every LoCoMo conversation, plus the scenario details the removed example was built from.
BENCHMARK_TERMS = ("Caroline", "Melanie", "Jon", "Gina", "John", "Maria", "Joanna", "Nate", "Tim", "Audrey", "Andrew",
                   "James", "Deborah", "Jolene", "Evan", "Sam", "Calvin", "Dave", "Door Dash", "clothing store",
                   "dance studio", "LoCoMo", "conv-")


def test_production_prompt_contains_no_benchmark_content():
    text = " ".join([MEMORY_PROTOCOL.system, *(part for example in MEMORY_PROTOCOL.examples for part in example)])
    found = [t for t in BENCHMARK_TERMS if f" {t}" in text or f'"{t}' in text or f"{t}:" in text]
    assert found == []


def test_memory_examples_pass_the_validator_and_show_the_intended_behaviour():
    outputs = {}
    for window, output in ((memory_prompt._EX1, memory_prompt._EX1_OUT), (memory_prompt._EX2, memory_prompt._EX2_OUT)):
        result = validate_atomization(json.dumps(output), window)
        assert result.status == "ok" and not result.issues
        outputs |= {m.source_message_id: m for m in result.messages}
    assert len(outputs["y2"].claims) == 3  # several memories from one message
    assert outputs["y4"].status == "no_durable_assertion" and not outputs["y4"].claims  # a suggestion is no memory
    assert outputs["x2"].status == "no_durable_assertion"
    assert all("Odalys" in c.content for c in outputs["y2"].claims)  # attributed to the speaker
    assert all("Tavi" in c.content for c in outputs["y3"].claims)


# ---- Task 2: failure-safe re-import --------------------------------------------------------------


def test_successful_reimport_replaces_records_and_source_messages(seeded):
    rows = [("n1", "user", "I play chess.")]
    report = ingest(seeded, Client({"n1": ["The user plays chess."]}), rows)
    assert report.kept_existing is False
    assert contents(seeded) == ["The user plays chess."]
    assert [m.message_id for m in seeded.get_conversation(CID).messages] == ["n1"]
    assert [row[1] for row in state(seeded)["v2_semantic_messages"]] == ["n1"]


def test_repeated_successful_imports_do_not_duplicate(seeded):
    first = state(seeded)
    for _ in range(2):
        ingest(seeded, Client(GOOD))
    second = state(seeded)
    assert second["v2_claims"] == first["v2_claims"] and len(second["v2_claims"]) == 2
    assert second["messages"] == first["messages"]
    records = memory_records(conv(), SemanticStore(seeded.connection).get_claims(CID))
    assert [r.memory_id for r in records] == ["mem:m1:c1", "mem:m2:c1"]


@pytest.mark.parametrize("client", [
    Client(fail_at=1),  # model unavailable from the first call
    Client(raw="{not json"),  # malformed output
    Client(raw='{"wrong": []}'),  # valid JSON, wrong schema
], ids=["model-failure", "malformed-json", "wrong-schema"])
def test_failed_extraction_preserves_everything_stored(seeded, client):
    before = state(seeded)
    report = ingest(seeded, client, [("n1", "user", "A different conversation about chess.")])
    assert report.kept_existing is True and report.failed_windows == 1
    assert state(seeded) == before
    assert contents(seeded) == ["The user lives in Porto.", "The user hates early shifts."]


def test_failure_in_one_window_of_many_preserves_everything_stored(seeded):
    before = state(seeded)
    rows = [(f"n{i}", "user", f"I own item{i}.") for i in range(8)]  # two windows
    client = Client({f"n{i}": [f"The user owns item{i}."] for i in range(8)}, fail_at=2)
    report = ingest(seeded, client, rows)
    assert report.windows == 2 and report.failed_windows == 1 and report.kept_existing is True
    assert state(seeded) == before


def test_model_timeout_preserves_everything_stored(seeded, monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("timed out")
    monkeypatch.setattr(urllib.request, "urlopen", timeout)
    before = state(seeded)
    client = OllamaChatClient("m", "http://localhost:1")
    report = ingest(seeded, client)
    assert report.kept_existing is True and state(seeded) == before


def test_all_claims_rejected_by_validation_preserves_everything_stored(seeded):
    before = state(seeded)
    report = ingest(seeded, Client({"m1": ["The user lives in Paris."], "m2": ["The user adores trains."]}))
    assert report.issues["UNSUPPORTED_CLAIM:rejected"] == 2 and report.claims == 0 and report.kept_existing is True
    assert state(seeded) == before


def test_output_with_no_message_entries_preserves_everything_stored(seeded):
    before = state(seeded)
    report = ingest(seeded, Client(raw='{"messages": []}'))
    assert report.statuses["missing"] == 2 and report.failed_windows == 0 and report.kept_existing is True
    assert state(seeded) == before


def test_clean_run_with_nothing_durable_is_a_successful_replacement(seeded):
    report = ingest(seeded, Client({}))
    assert report.kept_existing is False and report.failed_windows == 0
    assert contents(seeded) == []


def test_partly_rejected_claims_are_normal_and_still_replace(seeded):
    report = ingest(seeded, Client({"m1": ["The user lives in Porto."], "m2": ["The user adores trains."]}))
    assert report.kept_existing is False and report.issues["UNSUPPORTED_CLAIM:rejected"] == 1
    assert contents(seeded) == ["The user lives in Porto."]


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_interrupted_extraction_preserves_everything_stored(seeded, error):
    before = state(seeded)
    rows = [(f"n{i}", "user", f"I own item{i}.") for i in range(8)]
    with pytest.raises(error):
        ingest(seeded, Client({f"n{i}": [f"The user owns item{i}."] for i in range(8)}, fail_at=2, error=error), rows)
    assert state(seeded) == before


def test_persistence_failure_midway_rolls_back_completely(seeded, monkeypatch):
    before = state(seeded)
    original, calls = SemanticStore.write_window, []

    def fail_on_second(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SemanticStore, "write_window", fail_on_second)
    rows = [(f"n{i}", "user", f"I own item{i}.") for i in range(8)]
    with pytest.raises(OSError):
        ingest(seeded, Client({f"n{i}": [f"The user owns item{i}."] for i in range(8)}), rows)
    assert len(calls) == 2  # the first window really was written before the failure
    assert state(seeded) == before


def test_failed_reimport_survives_reopening_the_database(tmp_path):
    path = str(tmp_path / "m.db")
    with Store(path) as store:
        ingest(store, Client(GOOD))
    with Store(path) as store:
        ingest(store, Client(fail_at=1))
    with Store(path) as store:
        assert contents(store) == ["The user lives in Porto.", "The user hates early shifts."]


def test_first_import_keeps_partial_results_when_nothing_is_stored(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    rows = [(f"n{i}", "user", f"I own item{i}.") for i in range(8)]
    report = ingest(store, Client({f"n{i}": [f"The user owns item{i}."] for i in range(8)}, fail_at=2), rows)
    assert report.kept_existing is False and report.failed_windows == 1
    assert len(contents(store)) == 4
    store.close()


def test_interrupted_first_import_keeps_completed_windows(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    rows = [(f"n{i}", "user", f"I own item{i}.") for i in range(8)]
    with pytest.raises(RuntimeError):
        ingest(store, Client({f"n{i}": [f"The user owns item{i}."] for i in range(8)}, fail_at=2, error=RuntimeError), rows)
    assert len(contents(store)) == 4
    store.close()


def test_workspace_import_reports_failure_and_keeps_memories(tmp_path):
    db = tmp_path / "web.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    text = "User: I live in Porto.\n\nAssistant: Noted."
    good = Client({"imported:m0000": ["The user lives in Porto."]})
    ws.import_text(text, extract="memories", atomizer=SemanticAtomizer(good, MEMORY_PROTOCOL), db_path=db)
    with pytest.raises(ValueError, match="were kept"):
        ws.import_text("User: Something else entirely.\n\nAssistant: Fine.", extract="memories",
                       atomizer=SemanticAtomizer(Client(fail_at=1), MEMORY_PROTOCOL), db_path=db)
    assert [c.content for c in ws.claims] == ["The user lives in Porto."]
    with Store(str(db)) as store:
        semantic = SemanticStore(store.connection)
        assert [c.content for c in semantic.get_claims("imported")] == ["The user lives in Porto."]
        assert store.get_conversation("imported").messages[0].content == "I live in Porto."


# ---- Task 3: fail-safe validation -----------------------------------------------------------------


def support(source, claim):
    return classify_support(claim, (), Message("m", CID, 0, "user", source, None), (), ())


def validate(source, item):
    window = ContextWindow(CID, 0, (), (Message("m", CID, 0, "user", source, None),))
    raw = {"messages": [{"source_message_id": "m", "status": "claims", "claims": [item], "explicit_relations": []}]}
    return validate_atomization(json.dumps(raw), window)


@pytest.mark.parametrize("source, claim", [
    ("", "The user likes tea."),
    ("   \n\t ", "The user likes tea."),
    ("我喜欢茶。", "用户住在巴黎。"),
    ("मुझे चाय पसंद है", "The user likes tea."),
    ("我喜欢茶。", "The user likes tea."),
    ("I like tea.", "用户住在巴黎。"),
    ("I like tea.", "The user likes 茶."),  # one word would escape the comparison
    ("I like tea.", "The user is not."),  # only frame words: nothing to compare
], ids=["empty-source", "whitespace-source", "non-latin-both", "non-latin-source", "non-latin-source-english-claim",
        "non-latin-claim", "mixed-claim", "frame-only-claim"])
def test_uncomparable_claims_are_rejected(source, claim):
    assert support(source, claim)[0] is None
    result = validate(source, {"content": claim})
    assert result.claims == []
    assert [i.code for i in result.issues] == ["UNVERIFIABLE_CLAIM"]
    assert result.messages[0].status == "rejected"


def test_english_source_supports_its_claim():
    assert support("I like tea.", "The user likes tea.")[0] == SUPPORT_SOURCE


def test_accented_latin_text_stays_comparable():
    assert support("I live in São Paulo.", "The user lives in São Paulo.")[0] == SUPPORT_SOURCE


def test_english_claim_unsupported_by_its_source_is_rejected():
    assert support("I like tea.", "The user lives in Paris.")[0] is None
    assert [i.code for i in validate("I like tea.", {"content": "The user lives in Paris."}).issues] == ["UNSUPPORTED_CLAIM"]


@pytest.mark.parametrize("source, claim", [("I bought a car.", "The user buys a car."),
                                           ("My kids are in Oslo.", "The user's children are in Oslo.")])
def test_known_limitation_faithful_paraphrase_is_rejected(source, claim):
    """Documents a known V1 limit, not desired behaviour: the check compares word stems, not meaning."""
    assert support(source, claim)[0] is None


def test_known_limitation_flipped_negation_is_not_detected():
    """Documents a known V1 limit: validation is lexical, so polarity errors rely on the model."""
    assert support("I like coffee.", "The user does not like coffee.")[0] == SUPPORT_SOURCE


def test_uncomparable_annotations_are_dropped_and_the_claim_kept():
    result = validate("I like tea.", {"content": "The user likes tea.", "entities": ["巴黎"],
                                      "qualifiers": ["很"], "temporal": {"expression": "明天"}})
    claim = result.claims[0]
    assert claim.entities == () and claim.qualifiers == () and claim.temporal is None
    assert {i.code for i in result.issues} == {"UNSUPPORTED_ENTITY", "UNSUPPORTED_QUALIFIER", "TEMPORAL_NOT_IN_SOURCE"}
