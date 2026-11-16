import { useEffect, useMemo, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import type { Memory, SourceDetail as SourceDetailData } from "../../data/types";
import { formatDateTime, memoriesLabel, plural, sourceKindLabel } from "../../lib/format";
import { TypeGlyph } from "../ui/Badges";
import styles from "./source.module.css";

export function SourceDetail({ detail }: { detail: SourceDetailData }) {
  const { source, memories } = detail;
  const { hash } = useLocation();
  const anchored = hash.replace(/^#/, "") || null;
  const [focusMemory, setFocusMemory] = useState<string | null>(null);

  const memoriesByMessage = useMemo(() => {
    const map = new Map<string, Memory[]>();
    for (const { memory, messageIds } of memories) {
      for (const id of messageIds) map.set(id, [...(map.get(id) ?? []), memory]);
    }
    return map;
  }, [memories]);

  const focusedMessages = useMemo(() => {
    if (!focusMemory) return null;
    return new Set(memories.find((m) => m.memory.id === focusMemory)?.messageIds ?? []);
  }, [focusMemory, memories]);

  useEffect(() => {
    if (!anchored) return;
    document.getElementById(`msg-${anchored}`)?.scrollIntoView({ block: "center" });
  }, [anchored]);

  return (
    <div className={styles.detail}>
      <div className={styles.transcriptColumn}>
        <header className={styles.detailHeader}>
          <p className={styles.detailKind}>{sourceKindLabel(source.kind)}</p>
          <h2 className={styles.detailTitle}>{source.title}</h2>
          <p className={styles.detailMeta}>
            {formatDateTime(source.createdAt)} · {plural(source.messages.length, "message")} ·{" "}
            {memoriesLabel(memories.length)} derived
          </p>
        </header>

        <ol className={styles.transcript}>
          {source.messages.map((message) => {
            const derived = memoriesByMessage.get(message.id) ?? [];
            const highlighted = focusedMessages
              ? focusedMessages.has(message.id)
              : anchored === message.id;
            const dimmed = focusedMessages !== null && !highlighted;
            return (
              <li
                key={message.id}
                id={`msg-${message.id}`}
                className={[
                  styles.message,
                  derived.length ? styles.messageDerived : "",
                  highlighted ? styles.messageHighlighted : "",
                  dimmed ? styles.messageDimmed : "",
                ].join(" ")}
              >
                <div className={styles.messageMeta}>
                  <span className={styles.messageRole}>{message.role === "user" ? "You" : "Assistant"}</span>
                  <span>{formatDateTime(message.timestamp)}</span>
                  <code className={styles.messageId}>{message.id}</code>
                </div>
                <p className={styles.messageText}>{message.content}</p>
                {derived.length > 0 && (
                  <div className={styles.derived}>
                    {derived.map((m) => (
                      <Link key={m.id} to={`/memories/${m.id}`} className={styles.derivedChip}>
                        <TypeGlyph type={m.type} size={11} />
                        {m.title}
                      </Link>
                    ))}
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      </div>

      <aside className={styles.memoriesColumn}>
        <h3 className={styles.sideTitle}>Memories from this source</h3>
        <p className={styles.sideHint}>Hover a memory to see the messages it came from.</p>
        <ul className={styles.memoryList}>
          {memories.map(({ memory, messageIds }) => (
            <li key={memory.id}>
              <Link
                to={`/memories/${memory.id}`}
                className={focusMemory === memory.id ? styles.memoryItemActive : styles.memoryItem}
                onMouseEnter={() => setFocusMemory(memory.id)}
                onMouseLeave={() => setFocusMemory(null)}
                onFocus={() => setFocusMemory(memory.id)}
                onBlur={() => setFocusMemory(null)}
              >
                <TypeGlyph type={memory.type} />
                <span className={styles.memoryText}>
                  <span className={memory.status === "current" ? undefined : styles.historical}>
                    {memory.statement}
                  </span>
                  <span className={styles.memoryRefs}>
                    {messageIds.map((id) => (
                      <code key={id}>{id}</code>
                    ))}
                    {memory.status !== "current" && <span>Historical</span>}
                  </span>
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}
