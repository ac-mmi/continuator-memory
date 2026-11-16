import { Link } from "react-router-dom";
import type { MemoryEvent, MemoryEventAction } from "../../data/types";
import { formatDateTime } from "../../lib/format";
import styles from "./memory.module.css";

const ACTION_LABELS: Record<MemoryEventAction, string> = {
  created: "Created",
  reinforced: "Reinforced",
  updated: "Updated",
  superseded: "Superseded",
};

export function Timeline({ events }: { events: MemoryEvent[] }) {
  return (
    <ol className={styles.timeline}>
      {[...events].reverse().map((event, i) => (
        <li key={`${event.at}-${i}`} className={styles.timelineItem} data-action={event.action}>
          <span className={styles.timelineMarker} aria-hidden="true" />
          <div className={styles.timelineBody}>
            <div className={styles.timelineHead}>
              <span className={styles.timelineAction}>{ACTION_LABELS[event.action]}</span>
              <time className={styles.timelineTime} dateTime={event.at}>
                {formatDateTime(event.at)}
              </time>
            </div>
            <p className={styles.timelineText}>{event.description}</p>
            {event.source && (
              <Link to={`/sources/${event.source.sourceId}#${event.source.messageId}`} className={styles.timelineSource}>
                View source <code>{event.source.messageId}</code>
              </Link>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}
