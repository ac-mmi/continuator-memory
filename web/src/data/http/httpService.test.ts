import { afterEach, describe, expect, it, vi } from "vitest";
import { generationDefaults } from "../../lib/generation";
import { HttpError, HttpService } from "./httpService";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("HttpService getContext", () => {
  it("POSTs to /context and never calls /ask", async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      expect(String(url)).toBe("http://127.0.0.1:8765/context");
      expect(String(url)).not.toContain("/ask");
      expect(init?.method).toBe("POST");
      const body = JSON.parse(String(init?.body));
      expect(body).toEqual({
        query: "weekday bicycle",
        mode: "lexical",
        limit: 5,
        generation: { model: "qwen2.5:1.5b" },
      });
      return new Response(
        JSON.stringify({
          query: "weekday bicycle",
          mode: "baseline",
          memories: [],
          evidence: [],
          citations: [],
          formattedContext: "RELEVANT LONG-TERM MEMORY\n",
          diagnostics: { answerGeneration: "not_run", mode: "baseline", k: 1 },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    });
    vi.stubGlobal("fetch", fetchMock);
    const pack = await new HttpService("http://127.0.0.1:8765").getContext(
      "weekday bicycle",
      "lexical",
      { limit: 5, generation: { model: "qwen2.5:1.5b" } },
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(pack.formattedContext).toContain("RELEVANT LONG-TERM MEMORY");
    expect(pack.diagnostics?.answerGeneration).toBe("not_run");
  });
});

describe("HttpService ask", () => {
  it("sends generation options and never includes credentials", async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      void init;
      return new Response(
        JSON.stringify({
          conversation: { id: "c1", title: "q", updatedAt: "", messages: [] },
          answer: { id: "a1", role: "assistant", content: "ok", createdAt: "" },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    });
    vi.stubGlobal("fetch", fetchMock);
    const service = new HttpService("http://127.0.0.1:8765");
    await service.ask("weekday bicycle", undefined, "bprime", {
      generate: true,
      generation: { model: "qwen2.5:1.5b", temperature: 0, numPredict: 128 },
    });
    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(body.retrieval_mode).toBe("bprime");
    expect(body.generate).toBe(true);
    expect(body.generation).toEqual({ model: "qwen2.5:1.5b", temperature: 0, numPredict: 128 });
    expect(body.generation.apiKey).toBeUndefined();
    expect(body.generation.host).toBeUndefined();
  });
});

describe("HttpService import errors", () => {
  it("names POST /import when the API returns 404 not found", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ error: "not found" }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    const service = new HttpService("http://127.0.0.1:8765");
    await expect(service.importConversation("conv.txt", "You: hi\n")).rejects.toSatisfy((err: unknown) => {
      expect(err).toBeInstanceOf(HttpError);
      expect((err as HttpError).status).toBe(404);
      expect((err as HttpError).message).toBe("POST /import returned 404: not found");
      return true;
    });
  });
});

function stubJson(body: unknown, status = 200) {
  const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const BASE = "http://127.0.0.1:8765";

describe("HttpService routes", () => {
  it.each<[string, (s: HttpService) => Promise<unknown>, string]>([
    ["health", (s) => s.health(), "/health"],
    ["listMemories", (s) => s.listMemories(), "/memories"],
    ["getMemory", (s) => s.getMemory("mem:a b"), "/memories/mem%3Aa%20b"],
    ["listSources", (s) => s.listSources(), "/sources"],
    ["getSource", (s) => s.getSource("imported"), "/sources/imported"],
    ["getGraph", (s) => s.getGraph(), "/graph"],
    ["listConversations", (s) => s.listConversations(), "/conversations"],
    ["getConversation", (s) => s.getConversation("chat-1"), "/conversations/chat-1"],
  ])("%s uses GET on the backend route", async (_name, call, path) => {
    const fetchMock = stubJson({});
    await call(new HttpService(`${BASE}/`));
    expect(fetchMock.mock.calls[0]?.[0]).toBe(`${BASE}${path}`);
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBeUndefined();
  });

  it("encodes memory filters the way the server parses them", async () => {
    const fetchMock = stubJson([]);
    await new HttpService(BASE).listMemories({ text: "tea", types: ["fact", "event"], includeSuperseded: false });
    const url = new URL(String(fetchMock.mock.calls[0]?.[0]));
    expect(url.searchParams.get("q")).toBe("tea");
    expect(url.searchParams.get("types")).toBe("fact,event");
    expect(url.searchParams.get("include_superseded")).toBe("false");
  });

  it.each(["getMemory", "getSource", "getConversation"] as const)("%s returns null on 404", async (method) => {
    stubJson({ error: "not found" }, 404);
    await expect(new HttpService(BASE)[method]("x")).resolves.toBeNull();
  });

  it("parses a null-bearing memory detail as the backend sends it", async () => {
    stubJson({ memory: { id: "m", validFrom: null, memoryType: null }, related: [], evidence: [], supersedes: null, supersededBy: null });
    const detail = await new HttpService(BASE).getMemory("m");
    expect(detail?.supersededBy).toBeNull();
    expect(detail?.memory.memoryType).toBeNull();
  });
});

describe("HttpService import modes", () => {
  const okImport = { ok: true, report: { nMessages: 1 } };

  it("defaults to messages so Ollama extraction is never implicit", async () => {
    const fetchMock = stubJson(okImport);
    await new HttpService(BASE).importConversation("c.txt", "You: hi\n");
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toEqual({
      filename: "c.txt",
      text: "You: hi\n",
      extract: "messages",
    });
  });

  it.each(["messages", "v2", "memories"] as const)("sends extract=%s to POST /import", async (mode) => {
    const fetchMock = stubJson(okImport);
    await new HttpService(BASE).importConversation("c.txt", "You: hi\n", mode);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(`${BASE}/import`);
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBe("POST");
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body)).extract).toBe(mode);
  });

  it("rejects a body over the server limit without sending it", async () => {
    const fetchMock = stubJson(okImport);
    await expect(new HttpService(BASE).importConversation("c.txt", "é".repeat(4_500_000))).rejects.toThrow(/too large/);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("HttpService ask payload", () => {
  it("lets the server decide generation when the caller does not say", async () => {
    const fetchMock = stubJson({ conversation: {}, answer: {} });
    await new HttpService(BASE).ask("q");
    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(body).toEqual({ question: "q", retrieval_mode: "baseline" });
  });

  it("never sends the null maxContextChars that /health reports", () => {
    const body = JSON.parse(
      JSON.stringify(generationDefaults({ model: null, temperature: 0, numPredict: 256, maxContextChars: null })),
    );
    expect(body).toEqual({ temperature: 0, numPredict: 256 });
  });
});

describe("HttpService errors", () => {
  it("surfaces the backend's error message", async () => {
    stubJson({ error: "retrieval_mode must be 'baseline'" }, 400);
    await expect(new HttpService(BASE).ask("q")).rejects.toThrow(
      "POST /ask returned 400: retrieval_mode must be 'baseline'",
    );
  });

  it("explains an unreachable backend instead of a bare fetch error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    await expect(new HttpService(`${BASE}/`).health()).rejects.toSatisfy((err: unknown) => {
      expect(err).toBeInstanceOf(HttpError);
      expect((err as HttpError).status).toBe(0);
      expect((err as HttpError).message).toContain(`Cannot reach the Continuator API at ${BASE}`);
      return true;
    });
  });

  it("does not turn an outage into a null result", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    await expect(new HttpService(BASE).getMemory("m")).rejects.toBeInstanceOf(HttpError);
  });

  it("reports a non-JSON error body by status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("Bad Gateway", { status: 502 })));
    await expect(new HttpService(BASE).health()).rejects.toThrow("GET /health returned 502");
  });
});
