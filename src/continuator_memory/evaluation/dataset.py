"""Benchmark dataset loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from continuator_memory.models import Conversation, ParseError, parse_conversation

VALID_SCOPES = frozenset({"conversation", "global"})


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkQuestion:
    question_id: str
    category: str
    conversation_id: str
    question: str
    answerable: bool
    expected_answer: str | None
    expected_source_message_ids: tuple[str, ...]
    match_any: tuple[str, ...]
    match_all: tuple[str, ...]
    stale_answers: tuple[str, ...]
    scope: str
    notes: str | None = None

    @property
    def retrieval_conversation_id(self) -> str | None:
        """Conversation filter passed to the pipeline (None = search all conversations)."""
        return self.conversation_id if self.scope == "conversation" else None


@dataclass(frozen=True)
class BenchmarkDataset:
    dataset_id: str
    description: str | None
    conversations: tuple[Conversation, ...]
    questions: tuple[BenchmarkQuestion, ...]
    path: str | None = None
    sha256: str | None = None


def _str_list(raw: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise DatasetError(f"{where}: '{key}' must be a list of non-empty strings")
    return tuple(value)


def _parse_question(raw: Any, conversations: dict[str, Conversation], index: int):
    where = f"questions[{index}]"
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: must be a JSON object")
    for key in ("id", "category", "conversation_id", "question"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise DatasetError(f"{where}: '{key}' must be a non-empty string")
    where = f"question '{raw['id']}'"

    conversation = conversations.get(raw["conversation_id"])
    if conversation is None:
        raise DatasetError(f"{where}: unknown conversation_id '{raw['conversation_id']}'")

    answerable = raw.get("answerable", True)
    if not isinstance(answerable, bool):
        raise DatasetError(f"{where}: 'answerable' must be a boolean")
    scope = raw.get("scope", "conversation")
    if scope not in VALID_SCOPES:
        raise DatasetError(f"{where}: 'scope' must be one of {sorted(VALID_SCOPES)}")

    sources = _str_list(raw, "expected_source_message_ids", where)
    known_ids = {m.message_id for m in conversation.messages}
    unknown = [s for s in sources if s not in known_ids]
    if unknown:
        raise DatasetError(f"{where}: unknown source message ids {unknown}")
    match_all = _str_list(raw, "match_all", where)
    stale = _str_list(raw, "stale_answers", where)
    expected_answer = raw.get("expected_answer")

    if answerable:
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise DatasetError(f"{where}: answerable questions need 'expected_answer'")
        if not sources:
            raise DatasetError(f"{where}: answerable questions need expected source ids")
        if "match_any" in raw:
            match_any = _str_list(raw, "match_any", where)
        else:
            match_any = () if match_all else (expected_answer,)
    else:
        if expected_answer is not None or sources or match_all or raw.get("match_any"):
            raise DatasetError(
                f"{where}: unanswerable questions must not define an expected answer, "
                "sources, or match phrases"
            )
        match_any = ()

    notes = raw.get("notes")
    return BenchmarkQuestion(
        question_id=raw["id"],
        category=raw["category"],
        conversation_id=raw["conversation_id"],
        question=raw["question"],
        answerable=answerable,
        expected_answer=expected_answer,
        expected_source_message_ids=sources,
        match_any=match_any,
        match_all=match_all,
        stale_answers=stale,
        scope=scope,
        notes=notes if isinstance(notes, str) else None,
    )


def parse_dataset(data: Any) -> BenchmarkDataset:
    """Parse a benchmark dict.

    Shape::

        {"dataset_id": "...", "description": "...",
         "conversations": [<Phase 1 conversation objects>],
         "questions": [{"id", "category", "conversation_id", "question",
                        "answerable" (default true), "scope" ("conversation"|"global"),
                        "expected_answer", "expected_source_message_ids",
                        "match_any", "match_all", "stale_answers", "notes"}]}
    """
    if not isinstance(data, dict):
        raise DatasetError("dataset must be a JSON object")
    dataset_id = data.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise DatasetError("'dataset_id' must be a non-empty string")

    raw_conversations = data.get("conversations")
    if not isinstance(raw_conversations, list) or not raw_conversations:
        raise DatasetError("'conversations' must be a non-empty list")
    conversations: dict[str, Conversation] = {}
    for raw in raw_conversations:
        try:
            conv = parse_conversation(raw)
        except ParseError as exc:
            raise DatasetError(str(exc)) from exc
        if conv.conversation_id in conversations:
            raise DatasetError(f"duplicate conversation_id '{conv.conversation_id}'")
        conversations[conv.conversation_id] = conv

    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise DatasetError("'questions' must be a non-empty list")
    questions = [_parse_question(q, conversations, i) for i, q in enumerate(raw_questions)]
    seen: set[str] = set()
    for q in questions:
        if q.question_id in seen:
            raise DatasetError(f"duplicate question id '{q.question_id}'")
        seen.add(q.question_id)

    description = data.get("description")
    return BenchmarkDataset(
        dataset_id=dataset_id,
        description=description if isinstance(description, str) else None,
        conversations=tuple(conversations.values()),
        questions=tuple(questions),
    )


def load_dataset(path: str | Path) -> BenchmarkDataset:
    raw_bytes = Path(path).read_bytes()
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: invalid JSON: {exc}") from exc
    dataset = parse_dataset(data)
    return BenchmarkDataset(
        dataset_id=dataset.dataset_id,
        description=dataset.description,
        conversations=dataset.conversations,
        questions=dataset.questions,
        path=str(path),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
