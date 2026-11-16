import { Link } from "react-router-dom";
import type { MemoryDetail as MemoryDetailData } from "../../data/types";
import { formatDate, relativeTime } from "../../lib/format";
import { StatusBadge, TypeGlyph, TypeTag } from "../ui/Badges";
import { Icon } from "../ui/Icon";
import { ProvenancePanel } from "./ProvenancePanel";
import { Timeline } from "./Timeline";
import styles from "./memory.module.css";

export function MemoryDetail({ detail }: { detail: MemoryDetailData }) {
  const { memory, related, evidence, supersedes, supersededBy } = detail;

  return (
    <div className={styles.detail}>
      <div className={styles.detailMain}>
        <header className={styles.detailHeader}>
          <div className={styles.detailTags}>
            <TypeTag type={memory.type} />
            <StatusBadge status={memory.status} />
            {memory.memoryType && <span className={styles.memoryType}>{memory.memoryType.replace("_", " ")}</span>}
          </div>
          <h2 className={styles.detailStatement}>{memory.statement}</h2>
          <dl className={styles.detailFacts}>
            {memory.subject && (
              <div>
                <dt>About</dt>
                <dd>{memory.subject}</dd>
              </div>
            )}
            <div>
              <dt>Created</dt>
              <dd>{formatDate(memory.createdAt)}</dd>
            </div>
            <div>
              <dt>Updated</dt>
              <dd>{relativeTime(memory.updatedAt)}</dd>
            </div>
            {memory.validUntil && (
              <div>
                <dt>Valid until</dt>
                <dd>{formatDate(memory.validUntil)}</dd>
              </div>
            )}
          </dl>
        </header>

        {supersededBy && (
          <Link to={`/memories/${supersededBy.id}`} className={styles.notice}>
            <Icon name="history" size={15} />
            <span>
              <strong>Superseded.</strong> The current memory is: {supersededBy.statement}
            </span>
            <Icon name="arrowRight" size={14} />
          </Link>
        )}
        {supersedes && (
          <Link to={`/memories/${supersedes.id}`} className={styles.noticeQuiet}>
            <Icon name="history" size={15} />
            <span>Replaces an earlier memory: {supersedes.statement}</span>
            <Icon name="arrowRight" size={14} />
          </Link>
        )}

        <section className={styles.section}>
          <h3 className={styles.sectionTitle}>Provenance</h3>
          <p className={styles.sectionHint}>Where this memory came from, down to the original messages.</p>
          <ProvenancePanel memory={memory} evidence={evidence} />
        </section>

        <section className={styles.section}>
          <h3 className={styles.sectionTitle}>History</h3>
          <Timeline events={memory.history} />
        </section>
      </div>

      <aside className={styles.detailSide}>
        <h3 className={styles.sectionTitle}>Related memories</h3>
        {related.length === 0 ? (
          <p className={styles.sectionHint}>No related memories yet.</p>
        ) : (
          <ul className={styles.relatedList}>
            {related.map(({ memory: other, relationship }) => {
              const outgoing = relationship.fromId === memory.id;
              return (
                <li key={relationship.id}>
                  <Link to={`/memories/${other.id}`} className={styles.relatedItem}>
                    <span className={styles.relatedLabel}>
                      {outgoing ? `${relationship.label} →` : `${relationship.label} this`}
                    </span>
                    <span className={styles.relatedTarget}>
                      <TypeGlyph type={other.type} />
                      <span className={other.status === "current" ? undefined : styles.muted}>{other.title}</span>
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
        )}
        <Link to={`/graph?focus=${memory.id}`} className={styles.graphLink}>
          <Icon name="graph" size={14} />
          View in graph
        </Link>
      </aside>
    </div>
  );
}
