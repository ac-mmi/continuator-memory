import type { MemoryStatus, MemoryType } from "../../data/types";
import { typeLabel } from "../../lib/format";
import styles from "./ui.module.css";

/** Small glyph matching the node geometry used in the graph. */
export function TypeGlyph({ type, size = 12 }: { type: MemoryType; size?: number }) {
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 1.4 };
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true" className={styles.glyph}>
      {type === "person" && <circle cx="6" cy="6" r="4.5" {...common} />}
      {type === "project" && <rect x="1.2" y="2.5" width="9.6" height="7" rx="1.8" {...common} />}
      {type === "fact" && <rect x="1.5" y="3.5" width="9" height="5" rx="0.6" {...common} />}
      {type === "preference" && <rect x="0.8" y="3.5" width="10.4" height="5" rx="2.5" {...common} />}
      {type === "event" && <rect x="3" y="3" width="6" height="6" transform="rotate(45 6 6)" {...common} />}
    </svg>
  );
}

export function TypeTag({ type }: { type: MemoryType }) {
  return (
    <span className={styles.typeTag}>
      <TypeGlyph type={type} />
      {typeLabel(type)}
    </span>
  );
}

export function StatusBadge({ status }: { status: MemoryStatus }) {
  return (
    <span className={status === "current" ? styles.statusCurrent : styles.statusHistorical}>
      <span className={styles.statusDot} />
      {status === "current" ? "Current" : "Historical"}
    </span>
  );
}
