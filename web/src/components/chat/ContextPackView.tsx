import { useState } from "react";
import { Link } from "react-router-dom";
import type { ContextPack } from "../../data/types";
import { Icon } from "../ui/Icon";
import styles from "./chat.module.css";

/**
 * Minimal product rendering of `POST /context` / ContextPack.to_dict().
 * Not a chat answer — reusable memory context for any LLM or application.
 */
export function ContextPackView({ pack }: { pack: ContextPack }) {
  const [formattedOpen, setFormattedOpen] = useState(true);
  const [copied, setCopied] = useState(false);
  const expanded = new Set(pack.expandedMemoryIds ?? []);

  const copyContext = async () => {
    await navigator.clipboard.writeText(pack.formattedContext);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <article className={styles.contextPack} aria-label="Context pack">
      <header className={styles.contextPackHeader}>
        <div>
          <p className={styles.contextPackEyebrow}>Context Pack</p>
          <h3 className={styles.contextPackTitle}>Reusable memory context</h3>
        </div>
        <p className={styles.contextPackBadge} role="status">
          No answer was generated
        </p>
      </header>

      <p className={styles.contextPackLead}>
        Continuator retrieved grounded memories and source evidence for your own LLM or application.
        Nothing was sent to Continuator&apos;s answering model.
      </p>

      <section>
        <p className={styles.evidenceHeading}>Query</p>
        <p className={styles.inspectorQuestion}>{pack.query}</p>
        <p className={styles.diagnosticsLine}>
          {modeLabel(String(pack.mode))}
          {pack.diagnostics
            ? ` · k=${pack.diagnostics.k} · memories ${pack.memories.length} · sources ${pack.evidence.length}`
            : ` · memories ${pack.memories.length} · sources ${pack.evidence.length}`}
          {expanded.size > 0 ? ` · +${expanded.size} B′` : ""}
        </p>
      </section>

      <section>
        <p className={styles.evidenceHeading}>Relevant memories</p>
        {pack.memories.length === 0 ? (
          <p className={styles.panelEmpty}>No memories matched this query.</p>
        ) : (
          <ul className={styles.evidenceList}>
            {pack.memories.map((m) => (
              <li key={m.id}>
                <Link to={`/memories/${encodeURIComponent(m.id)}`} className={styles.evidenceItem}>
                  <Icon name="memory" size={13} />
                  <span>
                    <span className={styles.contextPackRank}>#{m.rank}</span> {m.text}
                  </span>
                  {expanded.has(m.id) && <span className={styles.evidenceTag}>B′</span>}
                  {m.status !== "current" && <span className={styles.evidenceTag}>{m.status}</span>}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <p className={styles.evidenceHeading}>Supporting evidence</p>
        {pack.evidence.length === 0 ? (
          <p className={styles.panelEmpty}>No source messages were selected.</p>
        ) : (
          <ul className={styles.evidenceList}>
            {pack.evidence.map((row) => (
              <li key={`${row.messageId}:${row.memoryIds.join(",")}`}>
                <div className={styles.contextPackEvidence}>
                  <code className={styles.messageId}>{row.messageId}</code>
                  <span className={styles.contextPackEvidenceMeta}>
                    {row.role}
                    {row.memoryIds.length > 0 ? ` · ${row.memoryIds.map((id) => id).join(", ")}` : ""}
                  </span>
                  <p className={styles.contextPackEvidenceText}>{row.text}</p>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      {pack.citations.length > 0 && (
        <section>
          <p className={styles.evidenceHeading}>Citations</p>
          <ul className={styles.evidenceList}>
            {pack.citations.map((c) => (
              <li key={c.memoryId} className={styles.diagnosticsLine}>
                <Link to={`/memories/${encodeURIComponent(c.memoryId)}`} className={styles.inlineLink}>
                  {c.memoryId}
                </Link>
                {" → "}
                {c.messageIds.length ? c.messageIds.join(", ") : "(unresolved)"}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <div className={styles.contextPackFormattedHeader}>
          <button
            type="button"
            className={formattedOpen ? styles.evidenceToggleOpen : styles.evidenceToggle}
            onClick={() => setFormattedOpen((o) => !o)}
            aria-expanded={formattedOpen}
          >
            <span className={styles.evidenceHeading} style={{ margin: 0 }}>
              Formatted context
            </span>
            <Icon name="chevronDown" size={13} className={styles.evidenceChevron} />
          </button>
          <button type="button" className={styles.contextPackCopy} onClick={() => void copyContext()}>
            <Icon name="link" size={13} />
            {copied ? "Copied" : "Copy context"}
          </button>
        </div>
        {formattedOpen && (
          <pre className={styles.contextPackFormatted} data-testid="formatted-context">
            {pack.formattedContext}
          </pre>
        )}
      </section>
    </article>
  );
}

function modeLabel(mode: string): string {
  if (mode === "bprime") return "B′ linked";
  if (mode === "multiquery") return "Multi-query";
  if (mode === "iterative") return "Iterative";
  if (mode === "lexical" || mode === "hybrid") return "Lexical";
  return "Baseline";
}
