import type { ReactNode } from "react";
import styles from "./ui.module.css";

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className={styles.empty}>
      <div className={styles.emptyMark} aria-hidden="true" />
      <p className={styles.emptyTitle}>{title}</p>
      {description && <p className={styles.emptyText}>{description}</p>}
      {action}
    </div>
  );
}

export function LoadingRows({ rows = 4 }: { rows?: number }) {
  return (
    <div className={styles.loading} aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className={styles.loadingRow} />
      ))}
    </div>
  );
}

export function ErrorNotice({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  return (
    <div className={styles.error} role="alert">
      <span>{error.message}</span>
      {onRetry && (
        <button type="button" className={styles.linkButton} onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}
