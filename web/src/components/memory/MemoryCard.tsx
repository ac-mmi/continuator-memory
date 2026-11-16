import { Link } from "react-router-dom";
import type { Memory } from "../../data/types";
import { plural, relativeTime } from "../../lib/format";
import { StatusBadge, TypeTag } from "../ui/Badges";
import styles from "./memory.module.css";

export function MemoryCard({ memory }: { memory: Memory }) {
  const historical = memory.status !== "current";
  return (
    <Link to={`/memories/${memory.id}`} className={historical ? styles.cardHistorical : styles.card}>
      <div className={styles.cardTop}>
        <TypeTag type={memory.type} />
        <StatusBadge status={memory.status} />
      </div>
      <p className={styles.cardStatement}>{memory.statement}</p>
      <div className={styles.cardMeta}>
        <span>Updated {relativeTime(memory.updatedAt)}</span>
        <span className={styles.metaDot} />
        <span>{plural(new Set(memory.sources.map((s) => s.sourceId)).size, "source")}</span>
      </div>
    </Link>
  );
}
