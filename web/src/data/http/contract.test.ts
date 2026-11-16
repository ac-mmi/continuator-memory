import { describe, expect, it } from "vitest";
import { generationDefaults } from "../../lib/generation";
import { HttpError, HttpService } from "./httpService";

/**
 * Runs HttpService against a real API. It replaces the server's workspace, so
 * point it at a throwaway server and CONTINUATOR_IMPORT_DB (see web/README.md):
 *   VITE_CONTINUATOR_CONTRACT_API_URL=http://127.0.0.1:8765 npx vitest run contract
 */
const apiUrl = import.meta.env.VITE_CONTINUATOR_CONTRACT_API_URL as string | undefined;

describe.skipIf(!apiUrl)("frontend/backend contract", () => {
  const service = new HttpService(apiUrl ?? "");
  const transcript = "User: My sister Dana lives in Lisbon.\nAssistant: Noted.\nUser: I prefer tea over coffee.\n";

  it("imports with the default mode, then browses the result", async () => {
    const result = await service.importConversation("contract.txt", transcript);
    expect(result.ok).toBe(true);
    expect(result.report).toMatchObject({ filename: "contract.txt", nMessages: 3, extract: "message_claims", failedWindows: 0 });
    expect(result.health.sampleId).toBe("imported");

    const memories = await service.listMemories();
    expect(memories).toHaveLength(result.report.nMemories);
    for (const m of memories) {
      expect(typeof m.id).toBe("string");
      expect(typeof m.statement).toBe("string");
      expect(["person", "project", "fact", "preference", "event"]).toContain(m.type);
      expect(["current", "superseded"]).toContain(m.status);
      expect(m.sources.length).toBeGreaterThan(0);
    }

    const detail = await service.getMemory(memories[0]!.id);
    expect(detail?.memory.id).toBe(memories[0]!.id);
    expect(detail?.evidence[0]?.message.content).toBeTruthy();

    const [source] = await service.listSources();
    const sourceDetail = await service.getSource(source!.id);
    expect(sourceDetail?.source.messages).toHaveLength(3);
    expect(sourceDetail?.memories.length).toBe(memories.length);

    const graph = await service.getGraph();
    expect(graph.memories).toHaveLength(memories.length);
    expect(Array.isArray(graph.relationships)).toBe(true);
  });

  it("accepts what the UI sends from /health in retrieval modes that need no model", { timeout: 120_000 }, async () => {
    const health = await service.health();
    const generation = generationDefaults(health.generation ?? {});
    for (const mode of ["baseline", "bprime", "hybrid"] as const) {
      const { answer, conversation } = await service.ask("Where does Dana live?", undefined, mode, {
        generate: false,
        generation,
      });
      expect(answer.evidence?.diagnostics?.mode).toBe(mode);
      expect(answer.evidence?.generation?.status).toBe("not_run");
      expect(answer.evidence?.memoryIds.length).toBeGreaterThan(0);
      expect(answer.evidence?.selectedMessages?.length).toBeGreaterThan(0);
      const stored = await service.getConversation(conversation.id);
      expect(stored?.messages).toHaveLength(2);
    }
    expect((await service.listConversations()).length).toBeGreaterThan(0);
  });

  it("returns server errors in the shape HttpService reports", async () => {
    await expect(service.importConversation("empty.txt", "   ")).rejects.toBeInstanceOf(HttpError);
    await expect(service.getMemory("no-such-memory")).resolves.toBeNull();
  });
});
