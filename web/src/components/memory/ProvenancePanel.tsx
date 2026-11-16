import { Link } from "react-router-dom";
import type { Memory, ResolvedEvidence } from "../../data/types";
import { formatDateTime, plural } from "../../lib/format";
import { TypeGlyph } from "../ui/Badges";
import { Icon } from "../ui/Icon";
import styles from "./memory.module.css";

/**
 * Shows the chain Memory -> Source -> Original evidence, so a memory is never
 * presented without the messages it was derived from.
 */
export function ProvenancePanel({ memory, evidence }: { memory: Memory; evidence: ResolvedEvidence[] }) {
  const groups = new Map<string, ResolvedEvidence[]>();
  for (const item of evidence) {
    const list = groups.get(item.source.id) ?? [];
    list.push(item);
    groups.set(item.source.id, list);
  }

  return (
    <div className={styles.provenance}>
      <ol className={styles.provenanceSteps} aria-label="Provenance chain">
        <li>Memory</li>
        <li aria-hidden="true"><Icon name="arrowRight" size={12} /></li>
        <li>{plural(groups.size, "source")}</li>
        <li aria-hidden="true"><Icon name="arrowRight" size={12} /></li>
        <li>{plural(evidence.length, "original message")}</li>
      </ol>

      <div className={styles.chain}>
        <div className={styles.chainMemory}>
          <TypeGlyph type={memory.type} />
          <span>{memory.statement}</span>
        </div>

        {groups.size === 0 && (
          <p className={styles.chainEmpty}>No source evidence is linked to this memory.</p>
        )}

        {[...groups.values()].map((items) => {
          const source = items[0].source;
          return (
            <div key={source.id} className={styles.chainSource}>
              <Link to={`/sources/${source.id}`} className={styles.chainSourceHeader}>
                <Icon name="sources" size={14} />
                <span className={styles.chainSourceTitle}>{source.title}</span>
                <span className={styles.chainSourceDate}>{formatDateTime(source.createdAt)}</span>
              </Link>
              {items.map(({ message }) => (
                <Link key={message.id} to={`/sources/${source.id}#${message.id}`} className={styles.excerptLink}>
                <figure className={styles.excerpt}>
                  <figcaption className={styles.excerptMeta}>
                    <span className={styles.excerptRole}>{message.role === "user" ? "You" : "Assistant"}</span>
                    <span>{formatDateTime(message.timestamp)}</span>
                    <code className={styles.excerptId}>{message.id}</code>
                  </figcaption>
                  <blockquote className={styles.excerptText}>{message.content}</blockquote>
                </figure>
                </Link>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
