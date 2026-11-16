import styles from "./ui.module.css";

export function LogoMark({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <rect x="5" y="5" width="22" height="22" rx="4" stroke="currentColor" strokeWidth="2.5" />
      <rect x="12" y="12" width="8" height="8" rx="1.5" fill="var(--blue)" />
    </svg>
  );
}

export function Wordmark() {
  return (
    <span className={styles.wordmark}>
      <LogoMark />
      <span>Continuator</span>
    </span>
  );
}
