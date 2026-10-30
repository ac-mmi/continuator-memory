"""B′ link expansion over frozen A-MEM classifications (query-time only).

Loads ``classifications.jsonl`` rows, filters to eligible LINK edges, and computes
neighbours of retrieved seed memory IDs (one hop) or, when ``max_hops=2``, also
neighbours of hop-1 expansions (fixed depth two). Does not invent links, reverse
directed edges, call models, or touch the extraction database.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RATIONALE_OK = "ok"
RATIONALE_EMPTY = "empty_rationale_diagnostic"
UNDIRECTED_RELATIONS = frozenset({"RELATED", "CONTRADICTS"})
DIRECTED_RELATIONS = frozenset({"ELABORATES", "UPDATES"})
DIRECTED_DIRECTIONS = frozenset({"source_to_candidate", "candidate_to_source"})
ALLOWED_MAX_HOPS = frozenset({1, 2})


@dataclass(frozen=True)
class EligibleLink:
    pair_id: str
    source_memory_id: str
    candidate_memory_id: str
    relation: str
    direction: str
    model: str | None = None
    linker_config_id: str | None = None
    rationale_status: str = RATIONALE_OK

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpansionHop:
    """One hop edge: origin memory → neighbour (to resolve and budget).

    ``seed_memory_id`` is the hop origin (a retrieved seed at depth 1, or a hop-1
    expanded memory at depth 2). ``hop_depth`` is 1 or 2.
    """

    pair_id: str
    seed_memory_id: str
    neighbour_memory_id: str
    relation: str
    direction: str
    seed_rank: int
    hop_depth: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BPrimeConfig:
    """Disabled by default. When disabled, the classifier file must not be read.

    ``max_hops`` is ignored when ``enabled`` is False. When enabled, only 1 (default
    one-hop) or 2 (opt-in two-hop) are valid.
    """

    enabled: bool = False
    classifications_path: Path | None = None
    max_hops: int = 1

    def resolved_path(self) -> Path | None:
        if not self.enabled:
            return None
        if not self.classifications_path:
            raise ValueError("B′ needs a classifications_path when enabled")
        return Path(self.classifications_path)

    def effective_max_hops(self) -> int:
        if not self.enabled:
            return 0
        return self.max_hops

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_classifications(path: Path) -> list[dict[str, Any]]:
    """Load JSONL classifications. Raises if the path is missing or a line is invalid JSON."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"classifications file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"classification line {line_no} must be a JSON object")
            rows.append(obj)
    return rows


def _rationale_status(row: Mapping[str, Any]) -> str:
    status = row.get("rationale_status")
    if status is None:
        return RATIONALE_OK
    if not isinstance(status, str):
        return RATIONALE_EMPTY
    return status


def _relation_direction_ok(relation: str, direction: str | None) -> bool:
    if relation in UNDIRECTED_RELATIONS:
        return direction == "undirected"
    if relation in DIRECTED_RELATIONS:
        return direction in DIRECTED_DIRECTIONS
    return False


def filter_eligible(
    rows: Sequence[Mapping[str, Any]],
    memory_ids: frozenset[str],
) -> tuple[list[EligibleLink], list[dict[str, Any]]]:
    """Return eligible LINK edges and skip diagnostics for every rejected row."""
    eligible: list[EligibleLink] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        pair_id = row.get("pair_id")
        reason: str | None = None
        if row.get("decision") != "LINK":
            reason = "decision_not_link"
        elif row.get("parse_ok") is not True or row.get("postcheck_ok") is not True:
            reason = "parse_or_postcheck_failed"
        elif _rationale_status(row) == RATIONALE_EMPTY:
            reason = "empty_rationale_diagnostic"
        elif _rationale_status(row) != RATIONALE_OK:
            reason = "rationale_status_not_ok"
        else:
            src = row.get("source_memory_id")
            cand = row.get("candidate_memory_id")
            relation = row.get("relation")
            direction = row.get("direction")
            if not isinstance(src, str) or not src or not isinstance(cand, str) or not cand:
                reason = "missing_endpoint_id"
            elif src not in memory_ids or cand not in memory_ids:
                reason = "endpoint_not_in_memory_set"
                skipped.append(
                    {
                        "pair_id": pair_id,
                        "reason": reason,
                        "source_memory_id": src,
                        "candidate_memory_id": cand,
                        "source_in_set": src in memory_ids,
                        "candidate_in_set": cand in memory_ids,
                    }
                )
                continue
            elif not isinstance(relation, str) or not _relation_direction_ok(relation, direction):
                reason = "invalid_relation_or_direction"
            else:
                eligible.append(
                    EligibleLink(
                        pair_id=str(pair_id),
                        source_memory_id=src,
                        candidate_memory_id=cand,
                        relation=relation,
                        direction=str(direction),
                        model=row.get("model") if isinstance(row.get("model"), str) else None,
                        linker_config_id=(
                            row.get("linker_config_id")
                            if isinstance(row.get("linker_config_id"), str)
                            else None
                        ),
                        rationale_status=_rationale_status(row),
                    )
                )
                continue
        skipped.append({"pair_id": pair_id, "reason": reason, "decision": row.get("decision")})
    return eligible, skipped


def traversal_pairs(link: EligibleLink) -> list[tuple[str, str]]:
    """(from_memory, to_memory) hops allowed by relation/direction. Never reverses directed edges."""
    src, cand = link.source_memory_id, link.candidate_memory_id
    if link.relation in UNDIRECTED_RELATIONS:
        if link.direction != "undirected":
            return []
        return [(src, cand), (cand, src)]
    if link.relation in DIRECTED_RELATIONS:
        if link.direction == "source_to_candidate":
            return [(src, cand)]
        if link.direction == "candidate_to_source":
            return [(cand, src)]
    return []


def neighbours_for_seeds(
    seed_ids: Sequence[str],
    links: Sequence[EligibleLink],
    *,
    seed_ranks: Mapping[str, int] | None = None,
    exclude_neighbour_ids: Iterable[str] | None = None,
    hop_depth: int = 1,
) -> list[ExpansionHop]:
    """Neighbours of ``seed_ids`` (hop origins). Skips excluded neighbours. Deterministic order.

    When ``exclude_neighbour_ids`` is None (one-hop default), neighbours that are themselves
    origins are skipped — matching the original seed-only behaviour. For hop 2, pass origins
    as ``seed_ids`` and pass seeds ∪ hop-1 expansions as ``exclude_neighbour_ids``.
    """
    origin_set = set(seed_ids)
    exclude = set(exclude_neighbour_ids) if exclude_neighbour_ids is not None else set(seed_ids)
    ranks = dict(seed_ranks) if seed_ranks is not None else {mid: i + 1 for i, mid in enumerate(seed_ids)}
    hops: list[ExpansionHop] = []
    seen: set[tuple[str, str]] = set()  # (pair_id, neighbour)
    for link in links:
        for origin, neighbour in traversal_pairs(link):
            if origin not in origin_set:
                continue
            if neighbour in exclude:
                continue
            key = (link.pair_id, neighbour)
            if key in seen:
                continue
            seen.add(key)
            hops.append(
                ExpansionHop(
                    pair_id=link.pair_id,
                    seed_memory_id=origin,
                    neighbour_memory_id=neighbour,
                    relation=link.relation,
                    direction=link.direction,
                    seed_rank=int(ranks.get(origin, 10**9)),
                    hop_depth=hop_depth,
                )
            )
    hops.sort(key=lambda h: (h.hop_depth, h.seed_rank, h.pair_id, h.neighbour_memory_id))
    return hops


def empty_trace(*, enabled: bool = False, max_hops: int | None = None) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "max_hops": max_hops if enabled else 0,
        "classifications_path": None,
        "classifications_sha256": None,
        "n_rows_loaded": 0,
        "eligible_links": [],
        "skipped": [],
        "applied_hops": [],
        "expanded_memory_ids": [],
        "expanded_memory_ids_by_hop": {"1": [], "2": []},
        "resolved_message_ids": [],
        "unresolved_neighbours": [],
        "seed_memory_ids": [],
        "seed_origins": [],
        "final_selected": [],
        "final_dropped": [],
        "budget_chars": None,
        "used_chars": None,
    }
