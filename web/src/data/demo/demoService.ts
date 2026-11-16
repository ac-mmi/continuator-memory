import { matchesText, tokenize } from "../../lib/text";
import type { ContinuatorService } from "../service";
import type {
  AskOptions,
  AskResult,
  ChatMessage,
  ContextOptions,
  ContextPack,
  Conversation,
  ConversationSummary,
  GraphSnapshot,
  HealthStatus,
  ImportResult,
  Memory,
  MemoryDetail,
  MemoryQuery,
  ResolvedEvidence,
  RetrievalMode,
  Source,
  SourceDetail,
  SourceSummary,
} from "../types";
import { DEMO_ANSWERS, buildDemoData, sourcesForMemories, type DemoData } from "./demoData";

export interface DemoServiceOptions {
  now?: () => Date;
  /** Simulated answer latency in ms. */
  answerDelayMs?: number;
}

const NO_MEMORY_ANSWER =
  "I don't have a memory about that yet. Once it comes up in a conversation or source, I'll remember it and show you where it came from.";

/** In-memory, deterministic implementation backed by `demoData`. */
export class DemoService implements ContinuatorService {
  private readonly data: DemoData;
  private readonly now: () => Date;
  private readonly answerDelayMs: number;
  private sequence = 0;

  constructor(options: DemoServiceOptions = {}) {
    this.now = options.now ?? (() => new Date());
    this.answerDelayMs = options.answerDelayMs ?? 600;
    this.data = buildDemoData(this.now());
  }

  async listMemories(query: MemoryQuery = {}): Promise<Memory[]> {
    const { text = "", types, includeSuperseded = true } = query;
    return this.data.memories
      .filter((m) => includeSuperseded || m.status === "current")
      .filter((m) => !types || types.length === 0 || types.includes(m.type))
      .filter((m) => matchesText(text, m.title, m.statement, m.subject))
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt) || a.id.localeCompare(b.id));
  }

  async getMemory(id: string): Promise<MemoryDetail | null> {
    const memory = this.memory(id);
    if (!memory) return null;

    const related = this.data.relationships
      .filter((r) => r.fromId === id || r.toId === id)
      .map((relationship) => ({
        relationship,
        memory: this.memory(relationship.fromId === id ? relationship.toId : relationship.fromId)!,
      }));

    const evidence: ResolvedEvidence[] = memory.sources.flatMap((ref) => {
      const source = this.source(ref.sourceId);
      const message = source?.messages.find((m) => m.id === ref.messageId);
      return source && message ? [{ source: this.summarize(source), message }] : [];
    });

    return {
      memory,
      related,
      evidence,
      supersedes: memory.supersedesId ? this.memory(memory.supersedesId) : undefined,
      supersededBy: memory.supersededById ? this.memory(memory.supersededById) : undefined,
    };
  }

  async listSources(): Promise<SourceSummary[]> {
    return this.data.sources
      .map((s) => this.summarize(s))
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }

  async getSource(id: string): Promise<SourceDetail | null> {
    const source = this.source(id);
    if (!source) return null;
    const memories = this.data.memories
      .map((memory) => ({
        memory,
        messageIds: memory.sources.filter((r) => r.sourceId === id).map((r) => r.messageId),
      }))
      .filter((entry) => entry.messageIds.length > 0);
    return { source, memories };
  }

  async getGraph(): Promise<GraphSnapshot> {
    return { memories: this.data.memories, relationships: this.data.relationships };
  }

  async listConversations(): Promise<ConversationSummary[]> {
    return this.data.conversations
      .map((c) => ({ id: c.id, title: c.title, updatedAt: c.updatedAt, messageCount: c.messages.length }))
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }

  async getConversation(id: string): Promise<Conversation | null> {
    const conversation = this.data.conversations.find((c) => c.id === id);
    return conversation ? structuredClone(conversation) : null;
  }

  async health(): Promise<HealthStatus> {
    return {
      ok: true,
      sampleId: "demo",
      origin: "demo",
      nMemories: this.data.memories.length,
      answerGeneration: "demo",
    };
  }

  async importConversation(): Promise<ImportResult> {
    throw new Error("Import requires the Continuator API. Set VITE_CONTINUATOR_API_URL and restart the web app.");
  }

  async getContext(
    query: string,
    retrievalMode: RetrievalMode = "baseline",
    _options: ContextOptions = {},
  ): Promise<ContextPack> {
    const trimmed = query.trim();
    if (!trimmed) throw new Error("Question is empty.");
    const { memoryIds } = this.answerFor(trimmed);
    const memories = memoryIds
      .map((id, index) => {
        const memory = this.memory(id);
        if (!memory) return null;
        const sources = memory.sources.flatMap((ref) => {
          const source = this.source(ref.sourceId);
          const message = source?.messages.find((m) => m.id === ref.messageId);
          if (!message) return [];
          return [
            {
              messageId: message.id,
              claimId: memory.id,
              role: message.role,
              text: message.content,
              timestamp: message.timestamp,
            },
          ];
        });
        return {
          id: memory.id,
          text: memory.statement,
          rank: index + 1,
          score: 1 - index * 0.1,
          similarity: 1 - index * 0.1,
          status: memory.status === "current" ? "current" : "historical",
          sourceMessageIds: sources.map((s) => s.messageId),
          sources,
        };
      })
      .filter((m): m is NonNullable<typeof m> => m != null);

    const evidence = memories.flatMap((m) =>
      m.sources.map((s) => ({
        messageId: s.messageId,
        role: s.role,
        text: s.text,
        memoryIds: [m.id],
        truncated: false,
        timestamp: s.timestamp,
      })),
    );
    const citations = memories.map((m) => ({
      memoryId: m.id,
      messageIds: m.sourceMessageIds,
      unresolved: [] as Array<{ claim_id: string | null; message_id: string | null; reason: string }>,
    }));
    const lines = [
      "RELEVANT LONG-TERM MEMORY",
      `Query: ${trimmed}`,
      `Retrieval mode: ${retrievalMode}`,
      "",
    ];
    for (const m of memories) {
      lines.push(`[${m.rank}] memory ${m.id}`);
      lines.push(`status=${m.status} score=${m.score.toFixed(6)} similarity=${m.similarity.toFixed(6)}`);
      lines.push(`Statement: ${m.text}`);
      if (m.sources.length) {
        lines.push("Supporting evidence:");
        for (const s of m.sources) {
          const ts = s.timestamp ? ` @ ${s.timestamp}` : "";
          lines.push(`  [${s.messageId}] ${s.role}${ts}: ${s.text}`);
        }
      } else {
        lines.push("Supporting evidence: (none)");
      }
      lines.push("");
    }
    return {
      query: trimmed,
      mode: retrievalMode,
      conversationId: null,
      memories,
      evidence,
      citations,
      formattedContext: `${lines.join("\n").trimEnd()}\n`,
      seedMemoryIds: memories.map((m) => m.id),
      expandedMemoryIds: [],
      hops: [],
      diagnostics: {
        mode: retrievalMode,
        k: memories.length || 1,
        seedCount: memories.length,
        expandedCount: 0,
        sourceCount: evidence.length,
        selectedSourceCount: evidence.length,
        droppedSourceCount: 0,
        budgetChars: 50000,
        usedChars: evidence.reduce((n, e) => n + e.text.length, 0),
        answerGeneration: "not_run",
      },
    };
  }

  async ask(question: string, conversationId?: string, _mode?: string, _options?: AskOptions): Promise<AskResult> {
    const trimmed = question.trim();
    if (!trimmed) throw new Error("Question is empty.");

    let conversation = conversationId
      ? this.data.conversations.find((c) => c.id === conversationId)
      : undefined;
    if (!conversation) {
      conversation = {
        id: `conv-local-${++this.sequence}`,
        title: titleFrom(trimmed),
        updatedAt: this.now().toISOString(),
        messages: [],
      };
      this.data.conversations.push(conversation);
    }

    const asked: ChatMessage = {
      id: `${conversation.id}-q${conversation.messages.length + 1}`,
      role: "user",
      content: trimmed,
      createdAt: this.now().toISOString(),
    };
    conversation.messages.push(asked);

    if (this.answerDelayMs > 0) await new Promise((r) => setTimeout(r, this.answerDelayMs));

    const { content, memoryIds } = this.answerFor(trimmed);
    const answer: ChatMessage = {
      id: `${conversation.id}-a${conversation.messages.length + 1}`,
      role: "assistant",
      content,
      createdAt: this.now().toISOString(),
      evidence: memoryIds.length
        ? { memoryIds, sourceIds: sourcesForMemories(this.data.memories, memoryIds) }
        : undefined,
    };
    conversation.messages.push(answer);
    conversation.updatedAt = answer.createdAt;

    return { conversation: structuredClone(conversation), answer };
  }

  private answerFor(question: string): { content: string; memoryIds: string[] } {
    const tokens = new Set(tokenize(question));

    let best: (typeof DEMO_ANSWERS)[number] | undefined;
    let bestHits = 0;
    for (const candidate of DEMO_ANSWERS) {
      const hits = candidate.keywords.filter((k) => tokens.has(k)).length;
      if (hits > bestHits) {
        best = candidate;
        bestHits = hits;
      }
    }
    if (best) return { content: best.answer, memoryIds: best.memoryIds };

    const scored = this.data.memories
      .filter((m) => m.status === "current")
      .map((m) => {
        const words = new Set(tokenize(`${m.title} ${m.statement} ${m.subject}`));
        let score = 0;
        for (const t of tokens) if (words.has(t)) score += 1;
        return { m, score };
      })
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score || a.m.id.localeCompare(b.m.id))
      .slice(0, 4);

    if (scored.length === 0) return { content: NO_MEMORY_ANSWER, memoryIds: [] };
    const lines = scored.map((s) => s.m.statement);
    return {
      content: `Here's what I remember: ${lines.join(" ")}`,
      memoryIds: scored.map((s) => s.m.id),
    };
  }

  private memory(id: string): Memory | undefined {
    return this.data.memories.find((m) => m.id === id);
  }

  private source(id: string): Source | undefined {
    return this.data.sources.find((s) => s.id === id);
  }

  private summarize(source: Source): SourceSummary {
    const memoryCount = this.data.memories.filter((m) =>
      m.sources.some((r) => r.sourceId === source.id),
    ).length;
    return {
      id: source.id,
      title: source.title,
      kind: source.kind,
      createdAt: source.createdAt,
      preview: source.messages[0]?.content ?? "",
      messageCount: source.messages.length,
      memoryCount,
    };
  }
}

function titleFrom(question: string): string {
  const clean = question.replace(/[?.!]+$/, "");
  return clean.length > 42 ? `${clean.slice(0, 40).trimEnd()}…` : clean;
}
