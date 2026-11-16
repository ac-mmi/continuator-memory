import { useState } from "react";
import { Link } from "react-router-dom";
import type { ChatMessage, GenerationPublic, Memory, SourceSummary } from "../../data/types";
import { memoriesLabel, plural } from "../../lib/format";
import { TypeGlyph } from "../ui/Badges";
import { Icon } from "../ui/Icon";
import { LogoMark } from "../ui/Logo";
import { AnswerInspector, type ReaskOptions } from "./AnswerInspector";
import styles from "./chat.module.css";

export interface EvidenceLookup {
  memories: Map<string, Memory>;
  sources: Map<string, SourceSummary>;
}

export function ChatMessageView({
  message,
  lookup,
  generationDefaults,
  onReask,
  onApplyDefaults,
}: {
  message: ChatMessage;
  lookup: EvidenceLookup;
  generationDefaults?: GenerationPublic;
  onReask?: (question: string, options: ReaskOptions) => void;
  onApplyDefaults?: (options: ReaskOptions) => void;
}) {
  if (message.role === "user") {
    return (
      <div className={styles.userMessage}>
        <p>{message.content}</p>
      </div>
    );
  }
  const [inspect, setInspect] = useState(false);
  const status = message.evidence?.generation?.status;
  const notice =
    status === "not_configured" ||
    status === "error" ||
    status === "disabled" ||
    status === "not_run" ||
    status === "context_overflow";
  return (
    <div className={styles.assistantMessage}>
      <div className={styles.assistantHeader}>
        <div className={styles.assistantLabel}>
          <LogoMark size={14} />
          Continuator
        </div>
        {message.evidence && (
          <button
            type="button"
            className={inspect ? styles.inspectorToggleOpen : styles.inspectorToggle}
            onClick={() => setInspect((o) => !o)}
            aria-expanded={inspect}
            aria-label="Answer configuration and memory provenance"
            title="Answer settings and provenance"
          >
            <Icon name="sliders" size={14} />
          </button>
        )}
      </div>
      <p className={notice ? styles.assistantNotice : styles.assistantText}>{message.content}</p>
      {inspect && message.evidence && (
        <AnswerInspector
          evidence={message.evidence}
          lookup={lookup}
          defaults={generationDefaults}
          onReask={onReask}
          onApplyDefaults={onApplyDefaults}
        />
      )}
      {message.evidence && <EvidenceIndicator evidence={message.evidence} lookup={lookup} />}
    </div>
  );
}

function EvidenceIndicator({
  evidence,
  lookup,
}: {
  evidence: NonNullable<ChatMessage["evidence"]>;
  lookup: EvidenceLookup;
}) {
  const [open, setOpen] = useState(false);
  const memories = evidence.memoryIds.flatMap((id) => lookup.memories.get(id) ?? []);
  const sources = evidence.sourceIds.flatMap((id) => lookup.sources.get(id) ?? []);
  const expanded = new Set(evidence.expandedMemoryIds ?? []);
  const hops = evidence.hops ?? [];
  const diagnostics = evidence.diagnostics;
  const selected = evidence.selectedMessages ?? [];
  const sourceHref = evidence.sourceIds[0];
  const gen = diagnostics?.answerGeneration;

  return (
    <div className={styles.evidence}>
      <button
        type="button"
        className={open ? styles.evidenceToggleOpen : styles.evidenceToggle}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className={styles.evidenceCount}>
          <Icon name="memory" size={13} />
          {memoriesLabel(evidence.memoryIds.length)}
        </span>
        <span className={styles.evidenceSep} />
        <span className={styles.evidenceCount}>
          <Icon name="sources" size={13} />
          {plural(evidence.sourceIds.length, "source")}
        </span>
        {diagnostics && (
          <>
            <span className={styles.evidenceSep} />
            <span className={styles.evidenceCount}>
              k={diagnostics.k}
              {diagnostics.expandedCount > 0 ? ` · +${diagnostics.expandedCount} B′` : ""}
            </span>
          </>
        )}
        <Icon name="chevronDown" size={13} className={styles.evidenceChevron} />
      </button>

      {open && (
        <div className={styles.evidencePanel}>
          {diagnostics && (
            <div>
              <p className={styles.evidenceHeading}>Retrieval</p>
              <p className={styles.diagnosticsLine}>
                {modeLabel(diagnostics.mode)} · k={diagnostics.k} · seeds{" "}
                {diagnostics.seedCount} · expanded {diagnostics.expandedCount} · sources{" "}
                {diagnostics.selectedSourceCount}/{diagnostics.sourceCount} · budget {diagnostics.usedChars}/
                {diagnostics.budgetChars} chars
                {gen && gen !== "ok" ? ` · answer ${gen.replace("_", " ")}` : ""}
              </p>
            </div>
          )}
          <div>
            <p className={styles.evidenceHeading}>Memories retrieved</p>
            <ul className={styles.evidenceList}>
              {memories.map((m) => (
                <li key={m.id}>
                  <Link to={`/memories/${m.id}`} className={styles.evidenceItem}>
                    <TypeGlyph type={m.type} />
                    <span className={m.status === "current" ? undefined : styles.evidenceHistorical}>
                      {m.statement}
                    </span>
                    {expanded.has(m.id) && <span className={styles.evidenceTag}>B′</span>}
                    {m.status !== "current" && <span className={styles.evidenceTag}>Historical</span>}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          {hops.length > 0 && (
            <div>
              <p className={styles.evidenceHeading}>B′ hops</p>
              <ul className={styles.evidenceList}>
                {hops.map((hop) => (
                  <li key={`${hop.pairId}:${hop.neighbourMemoryId}`} className={styles.hopItem}>
                    <span className={styles.evidenceTag}>{hop.pairId}</span>
                    <span>
                      {hop.relation ?? "LINK"} · {hop.direction ?? "undirected"}
                      {hop.rationale ? ` — ${hop.rationale}` : ""}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {selected.length > 0 && (
            <div>
              <p className={styles.evidenceHeading}>Source messages</p>
              <ul className={styles.evidenceList}>
                {selected.map((msg) => (
                  <li key={msg.id}>
                    <Link
                      to={sourceHref ? `/sources/${sourceHref}#${msg.id}` : "/sources"}
                      className={styles.evidenceItem}
                    >
                      <Icon name="sources" size={13} />
                      <span>
                        <code className={styles.messageId}>{msg.id}</code> {msg.content}
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div>
            <p className={styles.evidenceHeading}>From sources</p>
            <ul className={styles.evidenceList}>
              {sources.map((s) => (
                <li key={s.id}>
                  <Link to={`/sources/${s.id}`} className={styles.evidenceItem}>
                    <Icon name="sources" size={13} />
                    <span>{s.title}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function modeLabel(mode: string): string {
  if (mode === "bprime") return "B′ linked";
  if (mode === "multiquery") return "Multi-query";
  if (mode === "iterative") return "Iterative";
  if (mode === "lexical" || mode === "hybrid") return "Lexical";
  return "Baseline";
}
