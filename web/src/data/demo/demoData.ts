import type {
  Conversation,
  Memory,
  MemoryEvent,
  MemoryStatus,
  MemoryType,
  Relationship,
  Source,
  SourceMessage,
  SourceRef,
} from "../types";

export interface DemoData {
  memories: Memory[];
  relationships: Relationship[];
  sources: Source[];
  conversations: Conversation[];
}

/** Curated demo answers, matched by keyword before falling back to lexical search. */
export interface DemoAnswer {
  keywords: string[];
  answer: string;
  memoryIds: string[];
}

type SourceSpec = {
  id: string;
  title: string;
  kind: Source["kind"];
  daysAgo: number;
  messages: [role: SourceMessage["role"], content: string][];
};

const SOURCE_SPECS: SourceSpec[] = [
  {
    id: "src-preferences",
    title: "Working preferences",
    kind: "conversation",
    daysAgo: 20,
    messages: [
      ["user", "A few things about how I work: I prefer concise answers with code over long explanations."],
      ["user", "I use TypeScript for frontends and Python for anything ML-related."],
      ["user", "And please never suggest cloud services for FileMind. Privacy is the whole point."],
      ["assistant", "Noted: concise answers, TypeScript for frontends, Python for ML, and local-only for FileMind."],
    ],
  },
  {
    id: "src-continuator-design",
    title: "Continuator memory design",
    kind: "conversation",
    daysAgo: 14,
    messages: [
      ["user", "Continuator should extract structured memories from conversations, not just retrieve chunks."],
      ["assistant", "So each memory has a subject, a predicate, a value, and links to the messages it came from?"],
      ["user", "Exactly. And when a fact changes, supersede the old one instead of deleting it. History matters."],
      ["user", "Retrieval should combine memory lookup with the original source excerpts so answers stay grounded."],
      ["assistant", "That gives every answer a path from memory back to the original evidence."],
      ["user", "Rank memories lexically, but weight them by how well their source messages match semantically."],
    ],
  },
  {
    id: "src-filemind-architecture",
    title: "FileMind architecture review",
    kind: "conversation",
    daysAgo: 12,
    messages: [
      ["user", "I'm starting FileMind, a local-first file intelligence system. Everything runs on my laptop and nothing leaves the machine."],
      ["assistant", "For a local-first design you'll want an embedded index and a local model runtime. Any preferences for either?"],
      ["user", "SQLite for the metadata index. I want one file I can inspect and back up. For reasoning I'll use Ollama."],
      ["assistant", "SQLite with FTS5 for filenames and extracted text works well, and Ollama keeps inference local. Which model?"],
      ["user", "Start with llama3.1:8b. Priya is helping with the indexing pipeline."],
      ["assistant", "Noted: llama3.1:8b through Ollama, and Priya owns indexing."],
    ],
  },
  {
    id: "src-embeddings",
    title: "Embedding model choice",
    kind: "conversation",
    daysAgo: 9,
    messages: [
      ["user", "For FileMind's semantic search I'm going with all-MiniLM-L6-v2 embeddings. Small and fast on CPU."],
      ["assistant", "Good fit for CPU. 384 dimensions keeps the stored vectors small."],
      ["user", "Store vectors in SQLite as float32 blobs and do brute-force cosine for now. No external vector database."],
      ["assistant", "That matches the scale. Brute force over a few hundred thousand vectors is fine."],
    ],
  },
  {
    id: "src-priya-sync",
    title: "Weekly sync with Priya",
    kind: "conversation",
    daysAgo: 6,
    messages: [
      ["user", "Priya finished the file watcher. It re-indexes changed files incrementally."],
      ["user", "She thinks we should chunk documents by heading instead of fixed windows."],
      ["assistant", "Heading-aware chunking should preserve context better for Markdown and PDFs."],
      ["user", "Agreed. Also, Marcus from the design team will review the FileMind UI next month."],
      ["user", "Target for the FileMind private beta is October 30."],
    ],
  },
  {
    id: "src-model-switch",
    title: "Reasoning model switch",
    kind: "conversation",
    daysAgo: 3,
    messages: [
      ["user", "Switching FileMind from llama3.1:8b to qwen2.5:7b. It follows the JSON schema more reliably."],
      ["assistant", "Understood. qwen2.5:7b is now the current reasoning model; llama3.1:8b stays in history."],
      ["user", "Keep temperature at 0 for extraction so results are reproducible."],
    ],
  },
  {
    id: "src-launch",
    title: "Launch planning",
    kind: "conversation",
    daysAgo: 1,
    messages: [
      ["user", "The FileMind private beta is now planned for November 14. It was originally October 30."],
      ["assistant", "Updated: private beta on November 14, previously October 30."],
      ["user", "Priya and I will demo it at the local-first meetup on November 20."],
    ],
  },
];

type MemorySpec = {
  id: string;
  title: string;
  statement: string;
  type: MemoryType;
  subject: string;
  /** `[sourceId, messageIndex]`, 1-based to match the message ids `m1`, `m2`, ... */
  sources: [string, number][];
  status?: MemoryStatus;
  supersedes?: string;
};

const MEMORY_SPECS: MemorySpec[] = [
  // People
  { id: "mem-you", title: "You", type: "person", subject: "You",
    statement: "You are building FileMind and Continuator.",
    sources: [["src-filemind-architecture", 1], ["src-continuator-design", 1]] },
  { id: "mem-priya", title: "Priya", type: "person", subject: "Priya",
    statement: "Priya collaborates on FileMind and owns the indexing pipeline.",
    sources: [["src-filemind-architecture", 5], ["src-filemind-architecture", 6], ["src-priya-sync", 1]] },
  { id: "mem-marcus", title: "Marcus", type: "person", subject: "Marcus",
    statement: "Marcus is on the design team and will review the FileMind UI.",
    sources: [["src-priya-sync", 4]] },

  // Projects
  { id: "mem-filemind", title: "FileMind", type: "project", subject: "FileMind",
    statement: "FileMind is a local-first file intelligence system that runs entirely on your machine.",
    sources: [["src-filemind-architecture", 1], ["src-filemind-architecture", 2]] },
  { id: "mem-continuator", title: "Continuator", type: "project", subject: "Continuator",
    statement: "Continuator is a memory layer that extracts structured, source-linked memories from conversations.",
    sources: [["src-continuator-design", 1]] },

  // FileMind facts
  { id: "mem-sqlite", title: "SQLite index", type: "fact", subject: "FileMind",
    statement: "FileMind uses SQLite as its metadata index.",
    sources: [["src-filemind-architecture", 3], ["src-filemind-architecture", 4], ["src-embeddings", 3]] },
  { id: "mem-fts5", title: "FTS5 search", type: "fact", subject: "FileMind",
    statement: "FileMind uses SQLite FTS5 to search filenames and extracted text.",
    sources: [["src-filemind-architecture", 4]] },
  { id: "mem-ollama", title: "Ollama runtime", type: "fact", subject: "FileMind",
    statement: "FileMind runs local reasoning through Ollama.",
    sources: [["src-filemind-architecture", 3], ["src-filemind-architecture", 4]] },
  { id: "mem-llama", title: "llama3.1:8b", type: "fact", subject: "FileMind",
    statement: "FileMind uses llama3.1:8b as its reasoning model.",
    sources: [["src-filemind-architecture", 5]], status: "superseded" },
  { id: "mem-qwen", title: "qwen2.5:7b", type: "fact", subject: "FileMind",
    statement: "FileMind uses qwen2.5:7b as its reasoning model because it follows JSON schemas more reliably.",
    sources: [["src-model-switch", 1], ["src-model-switch", 2]], supersedes: "mem-llama" },
  { id: "mem-minilm", title: "MiniLM embeddings", type: "fact", subject: "FileMind",
    statement: "FileMind uses all-MiniLM-L6-v2 embeddings for semantic search.",
    sources: [["src-embeddings", 1], ["src-embeddings", 2]] },
  { id: "mem-vectors", title: "Vectors in SQLite", type: "fact", subject: "FileMind",
    statement: "FileMind stores embeddings in SQLite as float32 blobs and searches them with brute-force cosine similarity.",
    sources: [["src-embeddings", 3], ["src-embeddings", 4]] },
  { id: "mem-no-vectordb", title: "No vector database", type: "fact", subject: "FileMind",
    statement: "FileMind does not use an external vector database.",
    sources: [["src-embeddings", 3]] },
  { id: "mem-watcher", title: "Incremental watcher", type: "fact", subject: "FileMind",
    statement: "FileMind's file watcher re-indexes changed files incrementally.",
    sources: [["src-priya-sync", 1]] },
  { id: "mem-chunking", title: "Heading chunking", type: "fact", subject: "FileMind",
    statement: "FileMind chunks documents by heading instead of fixed-size windows.",
    sources: [["src-priya-sync", 2], ["src-priya-sync", 3]] },
  { id: "mem-temp0", title: "Temperature 0", type: "fact", subject: "FileMind",
    statement: "FileMind runs extraction at temperature 0 so results are reproducible.",
    sources: [["src-model-switch", 3]] },

  // Continuator facts
  { id: "mem-structured", title: "Structured memories", type: "fact", subject: "Continuator",
    statement: "Continuator stores memories as subject, predicate, and value, each linked to its source messages.",
    sources: [["src-continuator-design", 1], ["src-continuator-design", 2], ["src-continuator-design", 3]] },
  { id: "mem-supersede", title: "Supersede, not delete", type: "fact", subject: "Continuator",
    statement: "When a fact changes, Continuator supersedes the old memory and keeps it as history.",
    sources: [["src-continuator-design", 3]] },
  { id: "mem-grounded", title: "Grounded answers", type: "fact", subject: "Continuator",
    statement: "Continuator answers combine retrieved memories with the original source excerpts.",
    sources: [["src-continuator-design", 4], ["src-continuator-design", 5]] },
  { id: "mem-ranking", title: "Hybrid ranking", type: "fact", subject: "Continuator",
    statement: "Continuator ranks memories lexically and weights them by the semantic match of their source messages.",
    sources: [["src-continuator-design", 6]] },

  // Preferences
  { id: "mem-concise", title: "Concise answers", type: "preference", subject: "You",
    statement: "You prefer concise answers with code over long explanations.",
    sources: [["src-preferences", 1], ["src-preferences", 4]] },
  { id: "mem-typescript", title: "TypeScript frontends", type: "preference", subject: "You",
    statement: "You use TypeScript for frontend work.",
    sources: [["src-preferences", 2]] },
  { id: "mem-python", title: "Python for ML", type: "preference", subject: "You",
    statement: "You use Python for ML-related work.",
    sources: [["src-preferences", 2], ["src-preferences", 4]] },
  { id: "mem-local-only", title: "Local-only", type: "preference", subject: "FileMind",
    statement: "Never suggest cloud services for FileMind. Privacy is the priority.",
    sources: [["src-preferences", 3], ["src-preferences", 4], ["src-filemind-architecture", 1]] },

  // Events
  { id: "mem-beta-oct", title: "Beta · Oct 30", type: "event", subject: "FileMind",
    statement: "The FileMind private beta is planned for October 30.",
    sources: [["src-priya-sync", 5]], status: "superseded" },
  { id: "mem-beta-nov", title: "Beta · Nov 14", type: "event", subject: "FileMind",
    statement: "The FileMind private beta is planned for November 14.",
    sources: [["src-launch", 1], ["src-launch", 2]], supersedes: "mem-beta-oct" },
  { id: "mem-meetup", title: "Meetup demo", type: "event", subject: "FileMind",
    statement: "You and Priya will demo FileMind at the local-first meetup on November 20.",
    sources: [["src-launch", 3]] },
  { id: "mem-ui-review", title: "UI review", type: "event", subject: "FileMind",
    statement: "Marcus will review the FileMind UI next month.",
    sources: [["src-priya-sync", 4]] },
  { id: "mem-watcher-shipped", title: "Watcher shipped", type: "event", subject: "Priya",
    statement: "Priya finished FileMind's incremental file watcher.",
    sources: [["src-priya-sync", 1]] },
];

const RELATIONSHIP_SPECS: [from: string, label: string, to: string, status?: MemoryStatus][] = [
  ["mem-you", "builds", "mem-filemind"],
  ["mem-you", "builds", "mem-continuator"],
  ["mem-you", "works with", "mem-priya"],
  ["mem-you", "prefers", "mem-concise"],
  ["mem-you", "prefers", "mem-typescript"],
  ["mem-you", "prefers", "mem-python"],
  ["mem-local-only", "applies to", "mem-filemind"],
  ["mem-priya", "collaborates on", "mem-filemind"],
  ["mem-priya", "built", "mem-watcher"],
  ["mem-priya", "proposed", "mem-chunking"],
  ["mem-priya", "presents at", "mem-meetup"],
  ["mem-marcus", "leads", "mem-ui-review"],
  ["mem-ui-review", "reviews", "mem-filemind"],
  ["mem-filemind", "indexes with", "mem-sqlite"],
  ["mem-sqlite", "provides", "mem-fts5"],
  ["mem-sqlite", "stores", "mem-vectors"],
  ["mem-filemind", "reasons with", "mem-ollama"],
  ["mem-ollama", "runs", "mem-qwen"],
  ["mem-ollama", "ran", "mem-llama", "superseded"],
  ["mem-llama", "superseded by", "mem-qwen"],
  ["mem-filemind", "embeds with", "mem-minilm"],
  ["mem-minilm", "produces", "mem-vectors"],
  ["mem-vectors", "replaces", "mem-no-vectordb"],
  ["mem-filemind", "component", "mem-watcher"],
  ["mem-filemind", "chunks by", "mem-chunking"],
  ["mem-qwen", "configured", "mem-temp0"],
  ["mem-filemind", "milestone", "mem-beta-nov"],
  ["mem-beta-oct", "superseded by", "mem-beta-nov"],
  ["mem-meetup", "demos", "mem-filemind"],
  ["mem-watcher-shipped", "delivered", "mem-watcher"],
  ["mem-continuator", "stores", "mem-structured"],
  ["mem-continuator", "keeps history", "mem-supersede"],
  ["mem-continuator", "answers with", "mem-grounded"],
  ["mem-continuator", "ranks by", "mem-ranking"],
  ["mem-grounded", "uses", "mem-ranking"],
];

export const DEMO_ANSWERS: DemoAnswer[] = [
  {
    keywords: ["beta", "launch", "release"],
    answer:
      "The FileMind private beta is planned for November 14. It was originally set for October 30; you moved it in yesterday's launch planning conversation, so I've kept the October date as history rather than deleting it.",
    memoryIds: ["mem-beta-nov", "mem-beta-oct", "mem-meetup"],
  },
  {
    keywords: ["priya", "indexing"],
    answer:
      "Priya owns FileMind's indexing pipeline. She shipped the incremental file watcher last week and proposed chunking documents by heading instead of fixed windows. You're also demoing FileMind together at the local-first meetup on November 20.",
    memoryIds: ["mem-priya", "mem-watcher", "mem-watcher-shipped", "mem-chunking", "mem-meetup"],
  },
  {
    keywords: ["model", "llm", "qwen", "llama", "reasoning"],
    answer:
      "FileMind currently uses qwen2.5:7b through Ollama, running extraction at temperature 0. It replaced llama3.1:8b three days ago because qwen follows JSON schemas more reliably.",
    memoryIds: ["mem-qwen", "mem-llama", "mem-ollama", "mem-temp0"],
  },
  {
    keywords: ["prefer", "preference", "preferences"],
    answer:
      "You prefer concise answers with code over long explanations. You use TypeScript for frontends and Python for ML work, and you've asked me never to suggest cloud services for FileMind.",
    memoryIds: ["mem-concise", "mem-typescript", "mem-python", "mem-local-only"],
  },
  {
    keywords: ["continuator", "memory", "memories"],
    answer:
      "Continuator extracts structured memories (subject, predicate, value) and links each one to the messages it came from. When a fact changes, the old memory is superseded rather than deleted, and answers combine retrieved memories with the original source excerpts.",
    memoryIds: ["mem-continuator", "mem-structured", "mem-supersede", "mem-grounded", "mem-ranking"],
  },
  {
    keywords: ["filemind"],
    answer:
      "I remember that FileMind is a local-first file intelligence system using SQLite for indexing and Ollama for local reasoning. It currently runs qwen2.5:7b, embeds files with all-MiniLM-L6-v2, and re-indexes changed files incrementally. You've also been clear that it should never depend on cloud services.",
    memoryIds: ["mem-filemind", "mem-sqlite", "mem-ollama", "mem-qwen", "mem-minilm", "mem-watcher", "mem-local-only"],
  },
];

const CONVERSATION_SPECS: {
  id: string;
  title: string;
  hoursAgo: number;
  question: string;
  answerKeyword: string;
}[] = [
  { id: "conv-filemind", title: "FileMind overview", hoursAgo: 2,
    question: "What do you remember about my FileMind project?", answerKeyword: "filemind" },
  { id: "conv-beta", title: "Beta timeline", hoursAgo: 26,
    question: "When is the FileMind beta?", answerKeyword: "beta" },
  { id: "conv-indexing", title: "Who owns indexing", hoursAgo: 5 * 24 + 3,
    question: "Who is working on indexing?", answerKeyword: "priya" },
];

const HOUR = 60 * 60 * 1000;
const DAY = 24 * HOUR;

export function messageId(index: number): string {
  return `m${index}`;
}

/** Builds the demo dataset relative to `now` so timestamps read naturally. */
export function buildDemoData(now: Date): DemoData {
  const base = now.getTime();
  const iso = (ms: number) => new Date(ms).toISOString();

  const sources: Source[] = SOURCE_SPECS.map((spec) => {
    const start = base - spec.daysAgo * DAY;
    return {
      id: spec.id,
      title: spec.title,
      kind: spec.kind,
      createdAt: iso(start),
      messages: spec.messages.map(([role, content], i) => ({
        id: messageId(i + 1),
        role,
        content,
        timestamp: iso(start + i * 2 * 60 * 1000),
      })),
    };
  });
  const sourceById = new Map(sources.map((s) => [s.id, s]));

  const messageTime = (ref: SourceRef): string => {
    const source = sourceById.get(ref.sourceId);
    const message = source?.messages.find((m) => m.id === ref.messageId);
    if (!message) throw new Error(`Unknown demo source ${ref.sourceId}/${ref.messageId}`);
    return message.timestamp;
  };

  const supersededBy = new Map<string, string>();
  for (const spec of MEMORY_SPECS) {
    if (spec.supersedes) supersededBy.set(spec.supersedes, spec.id);
  }

  const memories: Memory[] = MEMORY_SPECS.map((spec) => {
    const refs: SourceRef[] = spec.sources.map(([sourceId, index]) => ({
      sourceId,
      messageId: messageId(index),
    }));
    const ordered = [...refs].sort((a, b) => messageTime(a).localeCompare(messageTime(b)));

    const history: MemoryEvent[] = ordered.map((ref, i) => ({
      at: messageTime(ref),
      action: i === 0 ? (spec.supersedes ? "updated" : "created") : "reinforced",
      description:
        i === 0
          ? spec.supersedes
            ? "Replaced an earlier memory with a newer statement."
            : "First observed in conversation."
          : "Confirmed by another message.",
      source: ref,
    }));

    const replacement = supersededBy.get(spec.id);
    let validUntil: string | undefined;
    if (replacement) {
      const next = MEMORY_SPECS.find((m) => m.id === replacement)!;
      const at = messageTime({ sourceId: next.sources[0][0], messageId: messageId(next.sources[0][1]) });
      validUntil = at;
      history.push({
        at,
        action: "superseded",
        description: "Superseded by a newer memory. Kept as history.",
        source: { sourceId: next.sources[0][0], messageId: messageId(next.sources[0][1]) },
      });
    }

    const createdAt = history[0].at;
    const updatedAt = history[history.length - 1].at;
    return {
      id: spec.id,
      title: spec.title,
      statement: spec.statement,
      type: spec.type,
      status: spec.status ?? "current",
      subject: spec.subject,
      createdAt,
      updatedAt,
      validFrom: createdAt,
      validUntil,
      supersedesId: spec.supersedes,
      supersededById: replacement,
      sources: refs,
      history,
    };
  });

  const relationships: Relationship[] = RELATIONSHIP_SPECS.map(([fromId, label, toId, status], i) => ({
    id: `rel-${String(i + 1).padStart(3, "0")}`,
    fromId,
    toId,
    label,
    status: status ?? "current",
  }));

  const conversations: Conversation[] = CONVERSATION_SPECS.map((spec) => {
    const at = base - spec.hoursAgo * HOUR;
    const answer = DEMO_ANSWERS.find((a) => a.keywords[0] === spec.answerKeyword)!;
    return {
      id: spec.id,
      title: spec.title,
      updatedAt: iso(at + 4000),
      messages: [
        { id: `${spec.id}-q1`, role: "user", content: spec.question, createdAt: iso(at) },
        {
          id: `${spec.id}-a1`,
          role: "assistant",
          content: answer.answer,
          createdAt: iso(at + 4000),
          evidence: {
            memoryIds: answer.memoryIds,
            sourceIds: sourcesForMemories(memories, answer.memoryIds),
          },
        },
      ],
    };
  });

  return { memories, relationships, sources, conversations };
}

export function sourcesForMemories(memories: Memory[], memoryIds: string[]): string[] {
  const wanted = new Set(memoryIds);
  const ids = new Set<string>();
  for (const memory of memories) {
    if (!wanted.has(memory.id)) continue;
    for (const ref of memory.sources) ids.add(ref.sourceId);
  }
  return [...ids].sort();
}
