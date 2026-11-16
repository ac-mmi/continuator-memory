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
} from "./types";

/**
 * The only surface UI components use to read or write Continuator data.
 * Implementations: `DemoService` (local deterministic data) and
 * `HttpService` (the Python API in `continuator_memory.webapi`).
 */
export interface ContinuatorService {
  listMemories(query?: MemoryQuery): Promise<Memory[]>;
  getMemory(id: string): Promise<MemoryDetail | null>;

  listSources(): Promise<SourceSummary[]>;
  getSource(id: string): Promise<SourceDetail | null>;

  getGraph(): Promise<GraphSnapshot>;

  listConversations(): Promise<ConversationSummary[]>;
  getConversation(id: string): Promise<Conversation | null>;
  /** Ask a question. Omitting `conversationId` starts a new conversation. */
  ask(
    question: string,
    conversationId?: string,
    retrievalMode?: RetrievalMode,
    options?: AskOptions,
  ): Promise<AskResult>;
  /**
   * Retrieve a Context Pack (memories + evidence + formattedContext).
   * Does not call Continuator's answering LLM.
   */
  getContext(
    query: string,
    retrievalMode?: RetrievalMode,
    options?: ContextOptions,
  ): Promise<ContextPack>;
  health(): Promise<HealthStatus>;
  importConversation(filename: string, text: string, extract?: ExtractMode): Promise<ImportResult>;
}
