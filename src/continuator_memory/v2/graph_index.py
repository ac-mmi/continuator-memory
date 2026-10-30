"""Graph retrieval index over conversation turns (experimental; HippoRAG-style, deterministic).

Architecture follows HippoRAG / HippoRAG 2 (Gutiérrez et al.): a phrase graph plus passage nodes,
queried with Personalized PageRank (PPR) seeded on the question's phrases. Differences, chosen to
avoid an LLM extraction pass over every turn:

  HippoRAG                              here
  LLM OpenIE triples -> phrase nodes    stemmed content words of each turn (v2.lexical), no LLM
  phrase-phrase relation edges          phrases connect through the turns that mention them
  synonymy edges (embedding cos >= .8)  same: existing Embedder over phrase surface forms, >= 0.8
  passage nodes (HippoRAG 2)            one node per conversation turn (the answer evidence)
  LLM query NER + node linking          question content words; exact stem match, else nearest
                                        phrase by embedding when cosine >= the synonym threshold
  reset weight = node specificity       same: 1 / number of turns mentioning the phrase
  PPR damping 0.5                       same
  (none)                                same-session adjacent-turn edges (conversation structure)

Every mention edge keeps its source turn id, speaker and timestamp. The index never merges values or
decides which statement is current: a phrase's ``observations`` are every mention in time order, and
search returns turns with their own timestamps, so an older statement is never presented as current.
Indexing is incremental (LightRAG-style union): ``add_messages`` appends turns, phrases and edges;
existing nodes are never rewritten and PPR runs at query time, so nothing is rebuilt. Pure numpy.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from continuator_memory.embedding import Embedder
from continuator_memory.models import Message
from continuator_memory.v2.lexical import STOP_WORDS, stem, tokenize

DAMPING = 0.5  # HippoRAG default
SYNONYM_THRESHOLD = 0.8  # HippoRAG default synonymy-edge cosine threshold
PPR_MAX_ITERATIONS = 100
PPR_TOLERANCE = 1e-10
MAX_TRACE_HOPS = 5
KIND_MENTION, KIND_SYNONYM, KIND_ADJACENT = "mention", "synonym", "adjacent"
# A leading "Name:" speaker label (LoCoMo turns are stored as "Caroline: ...").
_SPEAKER_PREFIX = re.compile(r"^\s*([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,2}):\s+")


def phrases(text: str) -> dict[str, str]:
    """Phrase stem -> first surface form, from content words (stop words and <3 chars removed)."""
    out: dict[str, str] = {}
    for tok in tokenize(text):
        if tok in STOP_WORDS or len(tok) < 3:
            continue
        key = stem(tok)
        if key in STOP_WORDS or len(key) < 3:
            continue
        out.setdefault(key, tok)
    return out


def split_speaker(content: str, role: str) -> tuple[str, str]:
    """(speaker, body): the leading speaker label if present, otherwise the message role."""
    match = _SPEAKER_PREFIX.match(content)
    if match:
        return match.group(1), content[match.end():]
    return role, content


@dataclass(frozen=True)
class Mention:
    """A phrase occurring in a turn: the edge's source attribution."""
    phrase: str
    message_id: str
    conversation_id: str
    position: int
    speaker: str
    timestamp: str | None


@dataclass(frozen=True)
class TurnNode:
    message_id: str
    conversation_id: str
    position: int
    speaker: str
    timestamp: str | None
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class GraphHit:
    rank: int
    message_id: str
    conversation_id: str
    score: float
    speaker: str
    timestamp: str | None
    hops: int | None  # shortest path length from a seed phrase (1 = the turn mentions a seed)
    matched_seeds: tuple[str, ...]  # seed phrases the turn itself mentions


@dataclass
class GraphOutcome:
    question: str
    hits: tuple[GraphHit, ...]
    seeds: dict[str, float] = field(default_factory=dict)  # phrase -> reset weight (before normalising)
    links: dict[str, tuple[str, str, float]] = field(default_factory=dict)  # query phrase -> (node, how, cos)
    unmatched: tuple[str, ...] = ()


class GraphIndex:
    def __init__(self, embedder: Embedder, *, synonym_threshold: float = SYNONYM_THRESHOLD,
                 damping: float = DAMPING, adjacency: bool = True):
        if not 0 < damping < 1:
            raise ValueError("damping must be in (0, 1)")
        self.embedder = embedder
        self.synonym_threshold = synonym_threshold
        self.damping = damping
        self.adjacency = adjacency
        self.phrase_index: dict[str, int] = {}
        self.phrase_surface: list[str] = []
        self.phrase_vectors: np.ndarray | None = None
        self.turns: list[TurnNode] = []
        self.turn_index: dict[tuple[str, str], int] = {}
        self.mentions: dict[str, list[Mention]] = defaultdict(list)
        self.edges: list[tuple[str, int, str, int, float, str]] = []  # (kind_a, a, kind_b, b, weight, kind)
        self._last_turn: dict[str, int] = {}  # conversation -> last turn index (for adjacency)
        self.embedded_phrases = 0

    # ---- indexing ---------------------------------------------------------------------------

    def add_messages(self, messages: Iterable[Message]) -> dict[str, int]:
        """Append turns, their phrases and edges. Existing nodes and edges are never changed."""
        new_turns, new_phrase_keys = 0, []
        for message in sorted(messages, key=lambda m: (m.conversation_id, m.position)):
            key = (message.conversation_id, message.message_id)
            if key in self.turn_index:
                continue  # already indexed: an incremental re-add is a no-op
            speaker, body = split_speaker(message.content, message.role)
            found = phrases(body)
            for phrase_key, surface in found.items():
                if phrase_key not in self.phrase_index:
                    self.phrase_index[phrase_key] = len(self.phrase_surface)
                    self.phrase_surface.append(surface)
                    new_phrase_keys.append(phrase_key)
            t = len(self.turns)
            self.turns.append(TurnNode(message.message_id, message.conversation_id, message.position,
                                       speaker, message.timestamp, tuple(found)))
            self.turn_index[key] = t
            new_turns += 1
            for phrase_key in found:  # one edge per (turn, phrase): repeats inside a turn count once
                self.edges.append(("turn", t, "phrase", self.phrase_index[phrase_key], 1.0, KIND_MENTION))
                self.mentions[phrase_key].append(Mention(phrase_key, message.message_id, message.conversation_id,
                                                         message.position, speaker, message.timestamp))
            prev = self._last_turn.get(message.conversation_id)
            if (self.adjacency and prev is not None and message.timestamp
                    and self.turns[prev].timestamp == message.timestamp):
                self.edges.append(("turn", prev, "turn", t, 1.0, KIND_ADJACENT))
            self._last_turn[message.conversation_id] = t
        new_synonyms = self._add_synonyms(new_phrase_keys)
        return {"turns": new_turns, "phrases": len(new_phrase_keys), "synonymEdges": new_synonyms}

    def _add_synonyms(self, new_keys: list[str]) -> int:
        if not new_keys:
            return 0
        vectors = np.asarray(self.embedder.embed([self.phrase_surface[self.phrase_index[k]] for k in new_keys]),
                             dtype=np.float32)
        self.embedded_phrases += len(new_keys)
        old = self.phrase_vectors
        self.phrase_vectors = vectors if old is None else np.vstack([old, vectors])
        start = 0 if old is None else len(old)
        sims = vectors @ self.phrase_vectors.T  # new x all (vectors are L2-normalised)
        added = 0
        for row in range(len(new_keys)):
            i = start + row
            for j in np.nonzero(sims[row] >= self.synonym_threshold)[0]:
                j = int(j)
                if j < i:  # each pair once: new-vs-older, and new-vs-new below the diagonal
                    self.edges.append(("phrase", j, "phrase", i, float(sims[row, j]), KIND_SYNONYM))
                    added += 1
        return added

    # ---- inspection -------------------------------------------------------------------------

    def observations(self, phrase: str) -> list[Mention]:
        """Every mention of a phrase (stem or surface word), oldest first; nothing is merged or dropped."""
        key = stem(phrase.lower()) if phrase.lower() not in self.mentions else phrase.lower()
        return sorted(self.mentions.get(key, []), key=lambda m: (m.timestamp or "", m.position))

    def stats(self) -> dict[str, int]:
        kinds: dict[str, int] = defaultdict(int)
        for edge in self.edges:
            kinds[edge[5]] += 1
        return {"turnNodes": len(self.turns), "phraseNodes": len(self.phrase_surface),
                "edges": len(self.edges), **{f"{k}Edges": v for k, v in kinds.items()},
                "embeddedPhrases": self.embedded_phrases}

    # ---- query ------------------------------------------------------------------------------

    def _link_query(self, question: str) -> tuple[dict[str, float], dict[str, tuple[str, str, float]], list[str]]:
        query = phrases(question)
        seeds: dict[str, float] = {}
        links: dict[str, tuple[str, str, float]] = {}
        unmatched: list[str] = []
        pending = []
        for key, surface in query.items():
            if key in self.phrase_index and self.mentions.get(key):
                seeds[key] = seeds.get(key, 0.0) + 1.0 / len(self.mentions[key])  # node specificity
                links[key] = (key, "exact", 1.0)
            else:
                pending.append((key, surface))
        if pending and self.phrase_vectors is not None and len(self.phrase_vectors):
            vecs = np.asarray(self.embedder.embed([s for _, s in pending]), dtype=np.float32)
            sims = vecs @ self.phrase_vectors.T
            keys = list(self.phrase_index)
            for (key, _), row in zip(pending, sims):
                j = int(np.argmax(row))
                if row[j] >= self.synonym_threshold:
                    node = keys[j]
                    seeds[node] = seeds.get(node, 0.0) + float(row[j]) / len(self.mentions[node])
                    links[key] = (node, "synonym", float(row[j]))
                else:
                    unmatched.append(key)
        else:
            unmatched += [k for k, _ in pending]
        return seeds, links, unmatched

    def _matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        n_phrase = len(self.phrase_surface)
        offset = {"phrase": 0, "turn": n_phrase}
        a = np.array([offset[e[0]] + e[1] for e in self.edges], dtype=np.int64)
        b = np.array([offset[e[2]] + e[3] for e in self.edges], dtype=np.int64)
        w = np.array([e[4] for e in self.edges], dtype=np.float64)
        src = np.concatenate([a, b])  # undirected
        dst = np.concatenate([b, a])
        weight = np.concatenate([w, w])
        return src, dst, weight, n_phrase + len(self.turns)

    def search(self, question: str, k: int | None = None, conversation_id: str | None = None) -> GraphOutcome:
        seeds, links, unmatched = self._link_query(question)
        if not seeds or not self.edges:
            return GraphOutcome(question, (), seeds, links, tuple(unmatched))
        src, dst, weight, n = self._matrix()
        degree = np.bincount(src, weights=weight, minlength=n)
        norm = np.where(degree[src] > 0, weight / np.where(degree[src] > 0, degree[src], 1.0), 0.0)
        reset = np.zeros(n)
        for key, value in seeds.items():
            reset[self.phrase_index[key]] = value
        reset /= reset.sum()
        dangling = degree == 0
        p = reset.copy()
        for _ in range(PPR_MAX_ITERATIONS):
            spread = np.bincount(dst, weights=p[src] * norm, minlength=n)
            nxt = self.damping * (spread + p[dangling].sum() * reset) + (1 - self.damping) * reset
            done = np.abs(nxt - p).sum() < PPR_TOLERANCE
            p = nxt
            if done:
                break
        n_phrase = len(self.phrase_surface)
        hops = self._hops(set(seeds), src, dst, n)
        seed_set = set(seeds)
        scored = [(p[n_phrase + t], t) for t, turn in enumerate(self.turns)
                  if p[n_phrase + t] > 0 and (conversation_id is None or turn.conversation_id == conversation_id)]
        scored.sort(key=lambda x: (-x[0], self.turns[x[1]].conversation_id, self.turns[x[1]].position))
        hits = []
        for rank, (score, t) in enumerate(scored[:k] if k is not None else scored, 1):
            turn = self.turns[t]
            hits.append(GraphHit(rank, turn.message_id, turn.conversation_id, float(score), turn.speaker,
                                 turn.timestamp, hops.get(n_phrase + t),
                                 tuple(sorted(seed_set & set(turn.phrases)))))
        return GraphOutcome(question, tuple(hits), seeds, links, tuple(unmatched))

    def _hops(self, seed_keys: set[str], src: np.ndarray, dst: np.ndarray, n: int) -> dict[int, int]:
        adjacency: dict[int, list[int]] = defaultdict(list)
        for s, d in zip(src.tolist(), dst.tolist()):
            adjacency[s].append(d)
        dist = {self.phrase_index[k]: 0 for k in seed_keys}
        frontier = list(dist)
        for depth in range(1, MAX_TRACE_HOPS + 1):
            nxt = []
            for node in frontier:
                for other in adjacency[node]:
                    if other not in dist:
                        dist[other] = depth
                        nxt.append(other)
            frontier = nxt
        return dist
