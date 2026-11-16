import type { ContinuatorService } from "../service";
import type {
  AskOptions,
  AskResult,
  ContextOptions,
  ContextPack,
  Conversation,
  ConversationSummary,
  ExtractMode,
  GraphSnapshot,
  HealthStatus,
  ImportResult,
  Memory,
  MemoryDetail,
  MemoryQuery,
  RetrievalMode,
  SourceDetail,
  SourceSummary,
} from "../types";
import { DEFAULT_EXTRACT_MODE } from "../types";

/** `IMPORT_BODY_LIMIT` in `webapi/server.py`. The server drops larger requests without a response. */
const IMPORT_BODY_LIMIT = 8_000_000;

/**
 * Client for the Python API (`continuator_memory.webapi.server`). Each endpoint
 * returns the JSON shape of the matching type in `../types`; errors are
 * `{ error: string }`.
 *
 *   GET  /memories?q=&types=&include_superseded=   -> Memory[]
 *   GET  /memories/:id                             -> MemoryDetail
 *   GET  /sources                                  -> SourceSummary[]
 *   GET  /sources/:id                              -> SourceDetail
 *   GET  /graph                                    -> GraphSnapshot
 *   GET  /conversations                            -> ConversationSummary[]
 *   GET  /conversations/:id                        -> Conversation
 *   GET  /health                                   -> HealthStatus
 *   POST /ask { question, conversation_id?, retrieval_mode?, generate?, generation? } -> AskResult
 *   POST /context { query|question, mode|retrieval_mode?, limit|k?, generation? } -> ContextPack
 *   POST /import { filename, text, extract? }                 -> ImportResult
 */
export class HttpService implements ContinuatorService {
  constructor(private readonly baseUrl: string) {}

  listMemories(query: MemoryQuery = {}): Promise<Memory[]> {
    const params = new URLSearchParams();
    if (query.text) params.set("q", query.text);
    if (query.types?.length) params.set("types", query.types.join(","));
    if (query.includeSuperseded === false) params.set("include_superseded", "false");
    const qs = params.toString();
    return this.get(`/memories${qs ? `?${qs}` : ""}`);
  }

  getMemory(id: string): Promise<MemoryDetail | null> {
    return this.getOrNull(`/memories/${encodeURIComponent(id)}`);
  }

  listSources(): Promise<SourceSummary[]> {
    return this.get("/sources");
  }

  getSource(id: string): Promise<SourceDetail | null> {
    return this.getOrNull(`/sources/${encodeURIComponent(id)}`);
  }

  getGraph(): Promise<GraphSnapshot> {
    return this.get("/graph");
  }

  listConversations(): Promise<ConversationSummary[]> {
    return this.get("/conversations");
  }

  getConversation(id: string): Promise<Conversation | null> {
    return this.getOrNull(`/conversations/${encodeURIComponent(id)}`);
  }

  ask(
    question: string,
    conversationId?: string,
    retrievalMode: RetrievalMode = "baseline",
    options: AskOptions = {},
  ): Promise<AskResult> {
    return this.request("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
        retrieval_mode: retrievalMode,
        generate: options.generate,
        generation: options.generation,
      }),
    });
  }

  getContext(
    query: string,
    retrievalMode: RetrievalMode = "baseline",
    options: ContextOptions = {},
  ): Promise<ContextPack> {
    return this.request("/context", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        mode: retrievalMode,
        limit: options.limit,
        generation: options.generation,
      }),
    });
  }

  health(): Promise<HealthStatus> {
    return this.get("/health");
  }

  importConversation(
    filename: string,
    text: string,
    extract: ExtractMode = DEFAULT_EXTRACT_MODE,
  ): Promise<ImportResult> {
    const body = JSON.stringify({ filename, text, extract });
    // Encoded size, not file size: JSON escaping can push a smaller file over the limit.
    if (new Blob([body]).size > IMPORT_BODY_LIMIT) {
      return Promise.reject(new Error("This conversation is too large to import (the API limit is 8 MB once encoded)."));
    }
    return this.request("/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
  }

  private get<T>(path: string): Promise<T> {
    return this.request<T>(path);
  }

  private async getOrNull<T>(path: string): Promise<T | null> {
    try {
      return await this.request<T>(path);
    } catch (error) {
      if (error instanceof HttpError && error.status === 404) return null;
      throw error;
    }
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const method = init?.method ?? "GET";
    const base = this.baseUrl.replace(/\/$/, "");
    let response: Response;
    try {
      response = await fetch(`${base}${path}`, init);
    } catch {
      throw new HttpError(
        0,
        `Cannot reach the Continuator API at ${base} (${method} ${path}). Check that the backend is running and allows this page's origin.`,
      );
    }
    if (!response.ok) {
      let reason = "";
      try {
        const body = (await response.json()) as { error?: unknown };
        if (typeof body.error === "string" && body.error) reason = body.error;
      } catch {
        /* ignore non-JSON error bodies */
      }
      const detail = reason
        ? `${method} ${path} returned ${response.status}: ${reason}`
        : `${method} ${path} returned ${response.status}`;
      throw new HttpError(response.status, detail);
    }
    return (await response.json()) as T;
  }
}

export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}
