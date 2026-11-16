/**
 * Product-facing domain types. They mirror the concepts in the Python memory
 * store (memories, memory_sources, relationships, consolidation_log,
 * conversations, messages) without depending on its table layout.
 */

export type MemoryType = "person" | "project" | "fact" | "preference" | "event";

export const MEMORY_TYPES: readonly MemoryType[] = [
  "person",
  "project",
  "fact",
  "preference",
  "event",
];

/** `superseded` memories are kept as history, like `is_current = 0` rows. */
export type MemoryStatus = "current" | "superseded";

export type ISODateString = string;

export interface SourceRef {
  sourceId: string;
  messageId: string;
}

export type MemoryEventAction = "created" | "reinforced" | "updated" | "superseded";

export interface MemoryEvent {
  at: ISODateString;
  action: MemoryEventAction;
  description: string;
  source?: SourceRef;
}

export interface Memory {
  id: string;
  /** Short label used in the graph and compact lists. */
  title: string;
  /** The full remembered statement. */
  statement: string;
  type: MemoryType;
  /** Exact V1 extraction type (e.g. "goal"); `type` is its display bucket. Null for other extraction modes. */
  memoryType?: string | null;
  status: MemoryStatus;
  /** Entity the memory is about, e.g. "FileMind" or "Priya". */
  subject: string;
  createdAt: ISODateString;
  updatedAt: ISODateString;
  validFrom?: ISODateString | null;
  validUntil?: ISODateString | null;
  supersedesId?: string | null;
  supersededById?: string | null;
  sources: SourceRef[];
  history: MemoryEvent[];
}

export interface Relationship {
  id: string;
  fromId: string;
  toId: string;
  label: string;
  status: MemoryStatus;
}

export type SourceKind = "conversation" | "document" | "note";

export interface SourceMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: ISODateString;
}

export interface Source {
  id: string;
  title: string;
  kind: SourceKind;
  createdAt: ISODateString;
  messages: SourceMessage[];
}

export interface SourceSummary {
  id: string;
  title: string;
  kind: SourceKind;
  createdAt: ISODateString;
  preview: string;
  messageCount: number;
  memoryCount: number;
  origin?: string;
}

/** Product retrieval modes. ``hybrid`` is a deprecated alias of ``lexical``. */
export type RetrievalMode =
  | "baseline"
  | "lexical"
  | "bprime"
  | "multiquery"
  | "iterative"
  | "hybrid";

export interface RetrievalHop {
  pairId: string;
  seedMemoryId: string;
  neighbourMemoryId: string;
  relation?: string;
  direction?: string;
  hopDepth?: number;
  rationale?: string;
  resolvedMessageIds: string[];
}

export interface RetrievalDiagnostics {
  mode: RetrievalMode;
  k: number;
  seedCount: number;
  expandedCount: number;
  sourceCount: number;
  selectedSourceCount: number;
  droppedSourceCount: number;
  budgetChars: number;
  usedChars: number;
  answerGeneration: string;
  promptMemoryCount?: number;
  retrievedMemoryCount?: number;
  queryCount?: number;
  planStatus?: string;
  roundCount?: number;
  stopReason?: string;
  latencyMs?: {
    plan?: number;
    retrieve?: number;
    total?: number;
  };
}

export interface GenerationPublic {
  enabled?: boolean;
  configured?: boolean;
  provider?: string | null;
  model?: string | null;
  temperature?: number;
  numPredict?: number;
  /** Null when the server has no character cap and budgets by `numCtx` instead. */
  maxContextChars?: number | null;
  numCtx?: number;
  promptChars?: number;
  truncated?: boolean;
  status?: string;
  error?: string;
  rawAnswer?: string;
  citations?: Array<{
    id: string;
    claim: string;
    kept: boolean;
    reason: string;
  }>;
  strippedCitations?: Array<{
    id: string;
    claim: string;
    kept: boolean;
    reason: string;
  }>;
}

export interface GenerationOverrides {
  model?: string;
  temperature?: number;
  numPredict?: number;
  maxContextChars?: number;
}

export interface AskOptions {
  generate?: boolean;
  generation?: GenerationOverrides;
}

export interface RetrievedMemoryEvidence {
  id: string;
  statement: string;
  rank?: number | null;
  score?: number | null;
  similarity?: number | null;
  origin?: string;
  sourceMessageIds: string[];
  inPrompt: boolean;
}

export interface PromptExcerpt {
  id: string;
  role: SourceMessage["role"];
  content: string;
  memoryIds: string[];
  truncated: boolean;
}

/** Evidence attached to an assistant answer. */
export interface AnswerEvidence {
  memoryIds: string[];
  sourceIds: string[];
  expandedMemoryIds?: string[];
  hops?: RetrievalHop[];
  diagnostics?: RetrievalDiagnostics;
  selectedMessageIds?: string[];
  selectedMessages?: SourceMessage[];
  seedMemoryIds?: string[];
  question?: string;
  promptMemoryIds?: string[];
  droppedMemoryIds?: string[];
  droppedMessageIds?: string[];
  memories?: RetrievedMemoryEvidence[];
  promptExcerpts?: PromptExcerpt[];
  generation?: GenerationPublic;
  queryPlan?: QueryPlanEvidence | null;
  queryRankings?: QueryRanking[] | null;
  fusedRanking?: FusedRankRow[] | null;
  retrievalComparison?: RetrievalComparison | null;
  iterativeRounds?: IterativeRound[] | null;
  stopReason?: string | null;
}

export interface QueryPlanEvidence {
  informationNeed: string;
  entities: string[];
  facets: string[];
  queries: string[];
  source: string;
  status: string;
  error?: string | null;
}

export interface QueryRankRow {
  rank: number;
  memoryId: string;
  sourceMessageIds: string[];
  similarity?: number;
  score?: number;
}

export interface QueryRanking {
  query: string;
  rankings: QueryRankRow[];
}

export interface FusedRankRow {
  rank: number;
  memoryId: string;
  sourceMessageIds: string[];
  fusion: number;
  perQueryRanks: Array<number | null>;
  similarity?: number;
}

export interface IterativeRound {
  round: number;
  queries: string[];
  newMemoryIds: string[];
  selectedMessageIds: string[];
  evaluation?: {
    status: string;
    supportedFacts: string[];
    missingInformation: string;
    nextQueries: string[];
    source: string;
    evalStatus: string;
    error?: string | null;
  } | null;
}

export interface RetrievalComparison {
  baseline?: { k: number; rankings: QueryRankRow[] };
  fused?: { k: number; rankings: FusedRankRow[] };
  latencyMs?: { plan?: number; retrieve?: number; total?: number };
  metrics?: Record<string, unknown>;
}

export interface HealthStatus {
  ok: boolean;
  sampleId?: string;
  conversationId?: string;
  nMemories?: number;
  nClaims?: number;
  nMessages?: number;
  k?: number;
  budgetChars?: number;
  bprimeClassifications?: string | null;
  answerGeneration?: string;
  generation?: GenerationPublic;
  origin?: string;
  extraction?: string;
  importedFilename?: string | null;
}

/**
 * `POST /import` extraction modes. `messages` needs no model and is the default;
 * `v2` and `memories` call Ollama and must be chosen explicitly.
 */
export type ExtractMode = "messages" | "v2" | "memories";

export const DEFAULT_EXTRACT_MODE: ExtractMode = "messages";

export interface ImportReport {
  filename: string;
  nMessages: number;
  nClaims: number;
  nMemories: number;
  nRelations: number;
  failedWindows: number;
  windows: number;
  extract: string;
  bprimeLinks: number;
}

export interface ImportResult {
  ok: boolean;
  source: SourceSummary;
  report: ImportReport;
  health: HealthStatus;
  note?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: ISODateString;
  evidence?: AnswerEvidence;
}

export interface Conversation {
  id: string;
  title: string;
  updatedAt: ISODateString;
  messages: ChatMessage[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: ISODateString;
  messageCount: number;
}

/** A source message that a memory was derived from, resolved for display. */
export interface ResolvedEvidence {
  source: SourceSummary;
  message: SourceMessage;
}

export interface MemoryDetail {
  memory: Memory;
  related: { memory: Memory; relationship: Relationship }[];
  evidence: ResolvedEvidence[];
  supersedes?: Memory | null;
  supersededBy?: Memory | null;
}

export interface SourceDetail {
  source: Source;
  /** Memories derived from this source, keyed by the message they came from. */
  memories: { memory: Memory; messageIds: string[] }[];
}

export interface MemoryQuery {
  text?: string;
  types?: MemoryType[];
  includeSuperseded?: boolean;
}

export interface GraphSnapshot {
  memories: Memory[];
  relationships: Relationship[];
}

export interface AskResult {
  conversation: Conversation;
  answer: ChatMessage;
}

/** Home/chat query intent: answer via `/ask`, or Context Pack via `/context`. */
export type QueryIntent = "ask" | "context";

/** One retrieved memory row from `POST /context` (`ContextPack.to_dict()`). */
export interface ContextPackMemory {
  id: string;
  text: string;
  rank: number;
  score: number;
  similarity: number;
  status: string;
  sourceMessageIds: string[];
  sources: Array<{
    messageId: string;
    claimId: string;
    role: string;
    text: string;
    timestamp: string | null;
  }>;
}

/** Source evidence row from `POST /context`. */
export interface ContextPackEvidence {
  messageId: string;
  role: string;
  text: string;
  memoryIds: string[];
  truncated: boolean;
  timestamp?: string | null;
  inAnswerPrompt?: boolean;
}

export interface ContextPackCitation {
  memoryId: string;
  messageIds: string[];
  unresolved: Array<{ claim_id: string | null; message_id: string | null; reason: string }>;
}

/**
 * Public JSON from `POST /context` / `ContextPack.to_dict()`.
 * Diagnostic extras (queryPlan, answerPrompt, …) may be present but are optional for the UI.
 */
export interface ContextPack {
  query: string;
  mode: RetrievalMode | string;
  conversationId?: string | null;
  memories: ContextPackMemory[];
  evidence: ContextPackEvidence[];
  citations: ContextPackCitation[];
  formattedContext: string;
  selection?: {
    budget_chars: number;
    unique_sources: string[];
    selected: string[];
    dropped_by_budget: string[];
    used_chars: number;
    first_memory: Record<string, string>;
  };
  seedMemoryIds?: string[];
  expandedMemoryIds?: string[];
  hops?: RetrievalHop[];
  diagnostics?: RetrievalDiagnostics & Record<string, unknown>;
  queryPlan?: QueryPlanEvidence | null;
  answerPrompt?: {
    system: string;
    prompt: string;
    promptMemoryIds: string[];
    droppedMemoryIds: string[];
    droppedMessageIds: string[];
  };
}

export interface ContextOptions {
  generation?: GenerationOverrides;
  limit?: number;
}
