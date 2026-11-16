import { describe, expect, it } from "vitest";
import { buildDemoData } from "./demoData";
import { DemoService } from "./demoService";

const NOW = new Date("2026-10-03T12:00:00Z");
const service = () => new DemoService({ now: () => NOW, answerDelayMs: 0 });

describe("demo data", () => {
  it("is deterministic for a fixed reference time", () => {
    expect(buildDemoData(NOW)).toEqual(buildDemoData(NOW));
  });

  it("links every memory source to a real message", () => {
    const data = buildDemoData(NOW);
    for (const memory of data.memories) {
      expect(memory.sources.length).toBeGreaterThan(0);
      for (const ref of memory.sources) {
        const source = data.sources.find((s) => s.id === ref.sourceId);
        expect(source?.messages.some((m) => m.id === ref.messageId), `${memory.id} -> ${ref.sourceId}/${ref.messageId}`).toBe(true);
      }
    }
  });

  it("only relates memories that exist", () => {
    const data = buildDemoData(NOW);
    const ids = new Set(data.memories.map((m) => m.id));
    for (const r of data.relationships) {
      expect(ids.has(r.fromId) && ids.has(r.toId), r.id).toBe(true);
    }
  });

  it("keeps superseded memories linked to their replacement", () => {
    const data = buildDemoData(NOW);
    const old = data.memories.find((m) => m.id === "mem-llama")!;
    expect(old.status).toBe("superseded");
    expect(old.supersededById).toBe("mem-qwen");
    expect(old.validUntil).toBeDefined();
    expect(old.history.at(-1)?.action).toBe("superseded");
  });

  it("stays within the curated graph size", () => {
    const n = buildDemoData(NOW).memories.length;
    expect(n).toBeGreaterThanOrEqual(20);
    expect(n).toBeLessThanOrEqual(40);
  });
});

describe("DemoService", () => {
  it("filters memories by type, text, and status", async () => {
    const s = service();
    const people = await s.listMemories({ types: ["person"] });
    expect(people.every((m) => m.type === "person")).toBe(true);

    const sqlite = await s.listMemories({ text: "sqlite" });
    expect(sqlite.map((m) => m.id)).toContain("mem-sqlite");

    const current = await s.listMemories({ includeSuperseded: false });
    expect(current.some((m) => m.status === "superseded")).toBe(false);
  });

  it("resolves memory provenance to original messages", async () => {
    const detail = await service().getMemory("mem-sqlite");
    expect(detail).not.toBeNull();
    expect(detail!.evidence).toHaveLength(3);
    expect(detail!.evidence[0].message.content).toMatch(/SQLite/);
    expect(detail!.related.length).toBeGreaterThan(0);
  });

  it("returns null for unknown ids", async () => {
    const s = service();
    expect(await s.getMemory("missing")).toBeNull();
    expect(await s.getSource("missing")).toBeNull();
    expect(await s.getConversation("missing")).toBeNull();
  });

  it("maps a source to the memories derived from each message", async () => {
    const detail = await service().getSource("src-filemind-architecture");
    const sqlite = detail!.memories.find((m) => m.memory.id === "mem-sqlite");
    expect(sqlite?.messageIds).toEqual(["m3", "m4"]);
  });

  it("answers with evidence and starts a conversation", async () => {
    const s = service();
    const { conversation, answer } = await s.ask("What do you remember about my FileMind project?");
    expect(answer.content).toMatch(/SQLite/);
    expect(answer.evidence?.memoryIds).toContain("mem-filemind");
    expect(answer.evidence?.sourceIds.length).toBeGreaterThan(0);
    expect(conversation.messages).toHaveLength(2);
    expect((await s.listConversations()).some((c) => c.id === conversation.id)).toBe(true);
  });

  it("getContext returns a Context Pack without creating a chat answer", async () => {
    const s = service();
    const before = (await s.listConversations()).length;
    const pack = await s.getContext("What do you remember about my FileMind project?", "baseline");
    expect(pack.query).toMatch(/FileMind/);
    expect(pack.formattedContext).toContain("RELEVANT LONG-TERM MEMORY");
    expect(pack.diagnostics?.answerGeneration).toBe("not_run");
    expect(pack.memories.length).toBeGreaterThan(0);
    expect(pack.evidence.length).toBeGreaterThan(0);
    expect((await s.listConversations()).length).toBe(before);
  });

  it("prefers the most specific curated answer", async () => {
    const { answer } = await service().ask("When is the FileMind beta?");
    expect(answer.evidence?.memoryIds).toContain("mem-beta-nov");
  });

  it("abstains without evidence when nothing matches", async () => {
    const { answer } = await service().ask("What is my favourite film?");
    expect(answer.evidence).toBeUndefined();
  });

  it("continues an existing conversation", async () => {
    const s = service();
    const first = await s.ask("Tell me about Marcus");
    const second = await s.ask("What are my preferences?", first.conversation.id);
    expect(second.conversation.id).toBe(first.conversation.id);
    expect(second.conversation.messages).toHaveLength(4);
  });
});
