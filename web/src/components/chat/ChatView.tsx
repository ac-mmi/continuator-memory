import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { isDemoMode } from "../../data";
import type {
  ChatMessage,
  ContextPack,
  Conversation,
  GenerationOverrides,
  Memory,
  QueryIntent,
  RetrievalMode,
} from "../../data/types";
import { useService } from "../../data/ServiceProvider";
import { relativeTime } from "../../lib/format";
import { generationDefaults } from "../../lib/generation";
import { endpointForQueryIntent } from "../../lib/queryIntent";
import { useAsync } from "../../lib/useAsync";
import { useShell } from "../shell/ShellContext";
import { TopBar } from "../shell/TopBar";
import { TypeGlyph } from "../ui/Badges";
import { EmptyState, ErrorNotice } from "../ui/EmptyState";
import { Icon } from "../ui/Icon";
import type { ReaskOptions } from "./AnswerInspector";
import { ChatComposer } from "./ChatComposer";
import { ChatMessageView, type EvidenceLookup } from "./ChatMessageView";
import { ContextPackView } from "./ContextPackView";
import styles from "./chat.module.css";

const DEMO_SUGGESTIONS = [
  "What do you remember about my FileMind project?",
  "When is the FileMind beta?",
  "Which model does FileMind use now?",
  "What are my preferences?",
];

const SAMPLE_SUGGESTIONS = [
  "What does Alex commute on?",
  "Does Alex own a Fender guitar?",
];

export function ChatView({ conversationId }: { conversationId?: string }) {
  const service = useService();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { refreshConversations } = useShell();

  const seeded = (location.state as { conversation?: Conversation } | null)?.conversation;
  const initial = seeded && seeded.id === conversationId ? seeded : undefined;

  const [conversation, setConversation] = useState<Conversation | undefined>(initial);
  const [pending, setPending] = useState<string | null>(null);
  const [pendingIntent, setPendingIntent] = useState<QueryIntent | null>(null);
  const [contextPack, setContextPack] = useState<ContextPack | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [missing, setMissing] = useState(false);
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("baseline");
  const [queryIntent, setQueryIntent] = useState<QueryIntent>(
    searchParams.get("intent") === "context" ? "context" : "ask",
  );
  const [generation, setGeneration] = useState<GenerationOverrides>({});
  const scrollRef = useRef<HTMLDivElement>(null);
  const live = !isDemoMode();
  const health = useAsync(() => service.health(), [service]);
  const suggestions =
    !live ? DEMO_SUGGESTIONS : health.data?.sampleId === "fixture" ? SAMPLE_SUGGESTIONS : [];

  const lookupState = useAsync(async (): Promise<EvidenceLookup> => {
    const [memories, sources] = await Promise.all([service.listMemories(), service.listSources()]);
    return {
      memories: new Map(memories.map((m) => [m.id, m])),
      sources: new Map(sources.map((s) => [s.id, s])),
    };
  }, [service]);
  const lookup: EvidenceLookup = lookupState.data ?? { memories: new Map(), sources: new Map() };

  useEffect(() => {
    if (searchParams.get("intent") === "context") setQueryIntent("context");
  }, [searchParams]);

  useEffect(() => {
    const g = health.data?.generation;
    if (!g) return;
    setGeneration(generationDefaults(g));
  }, [health.data?.generation]);

  useEffect(() => {
    setError(null);
    setPending(null);
    setPendingIntent(null);
    setMissing(false);
    if (!conversationId) {
      setConversation(undefined);
      return;
    }
    if (initial) {
      setConversation(initial);
      return;
    }
    let active = true;
    service.getConversation(conversationId).then(
      (c) => {
        if (!active) return;
        setConversation(c ?? undefined);
        setMissing(!c);
      },
      (e: Error) => active && setError(e),
    );
    return () => {
      active = false;
    };
  }, [service, conversationId]);

  const messages: ChatMessage[] = conversation?.messages ?? [];

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight });
  }, [messages.length, pending, contextPack]);

  const ask = async (question: string, overrides?: ReaskOptions) => {
    setPending(question);
    setPendingIntent("ask");
    setError(null);
    setContextPack(null);
    try {
      const mode = overrides?.retrievalMode ?? retrievalMode;
      const gen = overrides?.generation ?? generation;
      const result = await service.ask(
        question,
        conversation?.id,
        live ? mode : undefined,
        live ? { generate: health.data?.generation?.enabled !== false, generation: gen } : undefined,
      );
      setConversation(result.conversation);
      refreshConversations();
      if (result.conversation.id !== conversationId) {
        navigate(`/c/${result.conversation.id}`, { state: { conversation: result.conversation } });
      }
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setPending(null);
      setPendingIntent(null);
    }
  };

  const retrieveContext = async (question: string) => {
    setPending(question);
    setPendingIntent("context");
    setError(null);
    setContextPack(null);
    try {
      const pack = await service.getContext(question, retrievalMode, {
        generation: live ? generation : undefined,
      });
      setContextPack(pack);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setPending(null);
      setPendingIntent(null);
    }
  };

  const submit = async (question: string, overrides?: ReaskOptions) => {
    if (endpointForQueryIntent(queryIntent) === "/context") {
      await retrieveContext(question);
      return;
    }
    await ask(question, overrides);
  };

  const applyDefaults = (options: ReaskOptions) => {
    setRetrievalMode(options.retrievalMode);
    setGeneration(options.generation);
  };

  const contextMemories = useMemo(() => {
    const seen = new Set<string>();
    const result: Memory[] = [];
    const expanded = new Set<string>();
    for (const m of [...messages].reverse()) {
      for (const id of m.evidence?.expandedMemoryIds ?? []) expanded.add(id);
      for (const id of m.evidence?.memoryIds ?? []) {
        const memory = lookup.memories.get(id);
        if (memory && !seen.has(id)) {
          seen.add(id);
          result.push(memory);
        }
      }
    }
    if (contextPack) {
      for (const id of contextPack.expandedMemoryIds ?? []) expanded.add(id);
      for (const row of contextPack.memories) {
        const memory = lookup.memories.get(row.id);
        if (memory && !seen.has(row.id)) {
          seen.add(row.id);
          result.push(memory);
        }
      }
    }
    return { memories: result, expanded };
  }, [messages, lookup, contextPack]);

  if (missing) {
    return (
      <>
        <TopBar title="Conversation" />
        <EmptyState
          title="Conversation not found"
          description="It may have been removed, or the link is incorrect."
          action={<Link to="/" className={styles.inlineLink}>Start a new conversation</Link>}
        />
      </>
    );
  }

  const isEmpty =
    !conversationId && messages.length === 0 && !pending && !contextPack;

  const composerProps = {
    retrievalMode: live ? retrievalMode : undefined,
    onRetrievalModeChange: live ? setRetrievalMode : undefined,
    queryIntent,
    onQueryIntentChange: setQueryIntent,
  };

  if (isEmpty) {
    return (
      <>
        <TopBar title="Home" />
        <div className={styles.homeScroll}>
          <section className={styles.hero}>
            <h2 className={styles.heroTitle}>Your AI should remember.</h2>
            <p className={styles.heroText}>
              Ask for an answer, or retrieve a reusable Context Pack for any LLM.
            </p>
            <div className={styles.heroComposer}>
              <ChatComposer
                onSubmit={submit}
                autoFocus
                {...composerProps}
              />
            </div>
            {live && health.status === "ready" && (
              <p className={styles.heroNote}>
                {healthNote(health.data)}
              </p>
            )}
            {live && health.status === "error" && <ErrorNotice error={health.error} onRetry={health.reload} />}
            <div className={styles.suggestions}>
              {suggestions.map((s) => (
                <button key={s} type="button" className={styles.suggestion} onClick={() => void submit(s)}>
                  {s}
                </button>
              ))}
            </div>
            {error && <ErrorNotice error={error} />}
          </section>
          <RecentMemories memories={[...lookup.memories.values()]} />
        </div>
      </>
    );
  }

  const showThread = Boolean(conversationId || messages.length > 0 || pending || contextPack);

  return (
    <>
      <TopBar
        title={
          contextPack && !conversation
            ? "Context Pack"
            : conversation?.title ?? (pending ? "New conversation" : "Conversation")
        }
        meta={conversation ? `Updated ${relativeTime(conversation.updatedAt)}` : undefined}
      />
      <div className={styles.chatLayout}>
        <div className={styles.chatColumn}>
          <div className={styles.thread} ref={scrollRef}>
            <div className={styles.threadInner}>
              {messages.map((m) => (
                <ChatMessageView
                  key={m.id}
                  message={m}
                  lookup={lookup}
                  generationDefaults={health.data?.generation}
                  onReask={live ? ask : undefined}
                  onApplyDefaults={live ? applyDefaults : undefined}
                />
              ))}
              {pending && (
                <>
                  <ChatMessageView
                    message={{ id: "pending", role: "user", content: pending, createdAt: "" }}
                    lookup={lookup}
                  />
                  <div className={styles.recalling} role="status">
                    <Icon name="memory" size={14} />
                    {pendingIntent === "context"
                      ? "Retrieving relevant memories and source evidence…"
                      : "Retrieving memories and writing an answer…"}
                  </div>
                </>
              )}
              {!pending && contextPack && <ContextPackView pack={contextPack} />}
              {error && <ErrorNotice error={error} />}
            </div>
          </div>
          {showThread && (
            <div className={styles.composerDock}>
              <ChatComposer
                onSubmit={submit}
                disabled={pending !== null}
                placeholder={queryIntent === "context" ? "Retrieve more context…" : "Ask a follow-up..."}
                {...composerProps}
              />
            </div>
          )}
        </div>
        <ContextPanel memories={contextMemories.memories} expandedIds={contextMemories.expanded} />
      </div>
    </>
  );
}

function ContextPanel({ memories, expandedIds }: { memories: Memory[]; expandedIds: Set<string> }) {
  return (
    <aside className={styles.contextPanel} aria-label="Memories in context">
      <p className={styles.panelLabel}>In context</p>
      {memories.length === 0 ? (
        <p className={styles.panelEmpty}>Memories used in answers will appear here.</p>
      ) : (
        <ul className={styles.contextList}>
          {memories.map((m) => (
            <li key={m.id}>
              <Link to={`/memories/${m.id}`} className={styles.contextItem}>
                <TypeGlyph type={m.type} />
                <span className={styles.contextText}>
                  <span className={styles.contextTitle}>{m.title}</span>
                  <span className={styles.contextStatement}>{m.statement}</span>
                </span>
                {expandedIds.has(m.id) && <span className={styles.evidenceTag}>B′</span>}
                {m.status !== "current" && <span className={styles.evidenceTag}>Historical</span>}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}

function healthNote(health: { sampleId?: string; importedFilename?: string | null; answerGeneration?: string; generation?: { configured?: boolean; model?: string | null } } | undefined): string {
  const origin =
    health?.sampleId === "imported"
      ? `Using imported conversation${health.importedFilename ? ` (${health.importedFilename})` : ""}.`
      : health?.sampleId === "fixture"
        ? "Connected to the Continuator API (Alex sample). Import your own conversation from Sources."
        : "Connected to the Continuator API (existing database).";
  if (health?.answerGeneration === "not_configured" || health?.generation?.configured === false) {
    return `${origin} Ask generates grounded answers when a model is configured; Retrieve Context never needs one.`;
  }
  if (health?.answerGeneration === "disabled") {
    return `${origin} Answer generation is disabled; Retrieve Context still works.`;
  }
  if (health?.generation?.model) {
    return `${origin} Ask uses ${health.generation.model}; Retrieve Context returns a pack with no answering LLM.`;
  }
  return `${origin} Use Ask & Answer for a Continuator reply, or Retrieve Context for any external LLM.`;
}

function RecentMemories({ memories }: { memories: Memory[] }) {
  const recent = memories
    .filter((m) => m.status === "current")
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .slice(0, 4);
  if (recent.length === 0) return null;
  return (
    <section className={styles.recentMemories}>
      <div className={styles.recentHeader}>
        <p className={styles.panelLabel}>Recently remembered</p>
        <Link to="/memories" className={styles.inlineLink}>
          All memories
          <Icon name="arrowRight" size={13} />
        </Link>
      </div>
      <div className={styles.recentGrid}>
        {recent.map((m) => (
          <Link key={m.id} to={`/memories/${m.id}`} className={styles.recentCard}>
            <span className={styles.recentStatement}>{m.statement}</span>
            <span className={styles.recentMeta}>
              <TypeGlyph type={m.type} />
              {relativeTime(m.updatedAt)}
            </span>
          </Link>
        ))}
      </div>
    </section>
  );
}
