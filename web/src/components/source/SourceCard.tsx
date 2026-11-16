import { Link } from "react-router-dom";
import type { SourceSummary } from "../../data/types";
import { formatDate, memoriesLabel, plural, sourceKindLabel } from "../../lib/format";
import { Icon } from "../ui/Icon";
import styles from "./source.module.css";

export function SourceCard({ source }: { source: SourceSummary }) {
  return (
    <Link to={`/sources/${source.id}`} className={styles.card}>
      <div className={styles.cardIcon}>
        <Icon name="sources" size={16} />
      </div>
      <div className={styles.cardBody}>
        <div className={styles.cardHead}>
          <p className={styles.cardTitle}>{source.title}</p>
          <span className={styles.cardDate}>{formatDate(source.createdAt)}</span>
        </div>
        {source.origin === "imported" && <p className={styles.cardOrigin}>Imported conversation</p>}
        {source.origin === "fixture" && <p className={styles.cardOrigin}>Built-in sample (Alex)</p>}
        <p className={styles.cardPreview}>{source.preview}</p>
        <div className={styles.cardMeta}>
          <span>{sourceKindLabel(source.kind)}</span>
          <span className={styles.metaDot} />
          <span>{plural(source.messageCount, "message")}</span>
          <span className={styles.metaDot} />
          <span className={styles.cardMemories}>{memoriesLabel(source.memoryCount)} derived</span>
        </div>
      </div>
    </Link>
  );
}
