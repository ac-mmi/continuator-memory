"""LoCoMo conversation-only adapter. No benchmarks, no Ollama."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from continuator_memory.webapi.locomo import (
    conversation_from_locomo,
    ingest_locomo,
    load_locomo_conversation,
    parse_locomo_datetime,
)
from continuator_memory.webapi.transcript import IMPORTED_CONVERSATION_ID
from continuator_memory.webapi.workspace import Workspace, open_workspace
from tests.helpers import HashingEmbedder

ROOT = Path(__file__).resolve().parents[2]
LOCOMO = ROOT / "locomo_conv26.json"

POISON = {
    "sample_id": "conv-26",
    "qa": [
        {
            "question": "SECRET_QA_QUESTION",
            "answer": "SECRET_QA_ANSWER",
            "evidence": ["D1:1"],
            "category": 1,
        }
    ],
    "event_summary": {"note": "SECRET_EVENT_SUMMARY"},
    "observation": {"note": "SECRET_OBSERVATION"},
    "session_summary": {"session_1_summary": "SECRET_SESSION_SUMMARY"},
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel! How have you been?"},
            {
                "speaker": "Melanie",
                "dia_id": "D1:2",
                "text": "Hey Caroline! Busy with the kids.",
                "blip_caption": "a photo of a lake sunrise",
            },
        ],
        "session_2_date_time": "1:14 pm on 25 May, 2023",
        "session_2": [
            {"speaker": "Melanie", "dia_id": "D2:1", "text": "I ran a charity race."},
        ],
    },
}


def test_parse_locomo_datetime_is_iso():
    assert parse_locomo_datetime("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00"
    assert parse_locomo_datetime("1:14 pm on 25 May, 2023") == "2023-05-25T13:14:00"
    assert parse_locomo_datetime("12:09 am on 13 September, 2023") == "2023-09-13T00:09:00"
    assert parse_locomo_datetime("not a date") == "not a date"


def test_adapter_keeps_dialogue_and_drops_eval_only_fields():
    conv = conversation_from_locomo(POISON)
    assert conv.conversation_id == IMPORTED_CONVERSATION_ID
    blob = "\n".join(m.content for m in conv.messages)
    assert "SECRET_QA_QUESTION" not in blob
    assert "SECRET_QA_ANSWER" not in blob
    assert "SECRET_EVENT_SUMMARY" not in blob
    assert "SECRET_OBSERVATION" not in blob
    assert "SECRET_SESSION_SUMMARY" not in blob
    assert [m.message_id for m in conv.messages] == ["D1:1", "D1:2", "D2:1"]
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content.startswith("Caroline:")
    assert conv.messages[1].role == "assistant"
    assert conv.messages[1].content.startswith("Melanie:")
    assert "a photo of a lake sunrise" in conv.messages[1].content
    assert conv.messages[0].timestamp == "2023-05-08T13:56:00"
    assert conv.messages[2].timestamp == "2023-05-25T13:14:00"
    assert conv.messages[2].content.startswith("Melanie:")


@pytest.mark.skipif(not LOCOMO.is_file(), reason="locomo_conv26.json is not in the repo")
def test_real_conv26_preserves_dia_id_timestamp_and_speakers():
    conv = load_locomo_conversation(LOCOMO)
    by_id = {m.message_id: m for m in conv.messages}
    assert "D1:3" in by_id
    d13 = by_id["D1:3"]
    assert d13.timestamp == "2023-05-08T13:56:00"
    assert d13.content.startswith("Caroline:")
    assert "LGBTQ support group" in d13.content
    d21 = by_id["D2:1"]
    assert d21.timestamp == "2023-05-25T13:14:00"
    assert d21.content.startswith("Melanie:")
    assert all(m.message_id.startswith("D") for m in conv.messages)
    blob = "\n".join(m.content for m in conv.messages)
    assert "SECRET_QA_ANSWER" not in blob


def test_ingest_uses_dedicated_db_and_does_not_read_import_fallback(tmp_path, monkeypatch):
    locomo_db = tmp_path / "locomo-conv26.db"
    poison_import = tmp_path / "continuator-web-import.db"
    importer = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    importer.import_text(
        "You: I keep a sourdough starter in the fridge.\n\nChatGPT: Noted.\n",
        filename="poison.txt",
        extract="messages",
        db_path=poison_import,
    )
    getattr(importer, "_store").close()
    monkeypatch.setenv("CONTINUATOR_IMPORT_DB", str(poison_import))
    monkeypatch.delenv("CONTINUATOR_DB_PATH", raising=False)

    result = ingest_locomo(_write_poison(tmp_path), locomo_db)
    assert Path(result["db"]) == locomo_db
    assert locomo_db.is_file()
    assert not str(poison_import) == str(locomo_db)

    store = Store(str(locomo_db))
    stored = store.get_conversation(IMPORTED_CONVERSATION_ID)
    store.close()
    assert stored is not None
    assert [m.message_id for m in stored.messages] == ["D1:1", "D1:2", "D2:1"]
    assert stored.messages[0].timestamp == "2023-05-08T13:56:00"
    assert stored.messages[0].content.startswith("Caroline:")

    ws = open_workspace(sample="db", db_path=locomo_db, embedder=HashingEmbedder(), k=5)
    assert ws.conversation.conversation_id == IMPORTED_CONVERSATION_ID
    assert {m.message_id for m in ws.conversation.messages} == {"D1:1", "D1:2", "D2:1"}
    blob = "\n".join(m.content for m in ws.conversation.messages)
    assert "sourdough" not in blob.lower()
    assert "Caroline:" in blob and "Melanie:" in blob
    assert ws.claims
    health = ws.health()
    assert health["conversationId"] == IMPORTED_CONVERSATION_ID
    assert health["nMemories"] >= 3
    getattr(ws, "_store").close()


def test_explicit_dedicated_db_never_falls_back_to_import_db(tmp_path, monkeypatch):
    dedicated = tmp_path / "locomo-conv26.db"
    poison_import = tmp_path / "continuator-web-import.db"
    conv = Conversation(
        "trip-planning-001",
        (Message("m1", "trip-planning-001", 0, "user", "lisbon hotel dates"),),
        title="Planning a trip to Lisbon",
    )
    Store(dedicated).save_conversation(conv, [], np.zeros((0, 1), dtype=np.float32), "none")
    importer = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    importer.import_text(
        "You: I keep a sourdough starter in the fridge.\n\nChatGPT: Noted.\n",
        filename="poison.txt",
        extract="messages",
        db_path=poison_import,
    )
    getattr(importer, "_store").close()
    monkeypatch.setenv("CONTINUATOR_IMPORT_DB", str(poison_import))
    monkeypatch.setenv("CONTINUATOR_DB_PATH", str(tmp_path / "continuator.db"))

    ws = open_workspace(sample="db", db_path=dedicated, embedder=HashingEmbedder(), k=5)
    assert ws.conversation.conversation_id == "trip-planning-001"
    assert "sourdough" not in ws.conversation.messages[0].content.lower()
    getattr(ws, "_store").close()


def test_ingest_refuses_default_databases(tmp_path):
    locomo = _write_poison(tmp_path)
    with pytest.raises(ValueError, match="refusing"):
        ingest_locomo(locomo, Path("data/continuator.db"))
    with pytest.raises(ValueError, match="refusing"):
        ingest_locomo(locomo, Path("data/continuator-web-import.db"))


def _write_poison(tmp_path: Path) -> Path:
    import json

    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(POISON), encoding="utf-8")
    return path
