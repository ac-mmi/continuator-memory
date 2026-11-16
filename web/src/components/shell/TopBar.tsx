import type { ReactNode } from "react";
import { Icon } from "../ui/Icon";
import { useShell } from "./ShellContext";
import styles from "./shell.module.css";

export function TopBar({
  title,
  meta,
  actions,
  leading,
}: {
  title: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  leading?: ReactNode;
}) {
  const { openNav } = useShell();
  return (
    <header className={styles.topbar}>
      <button type="button" className={styles.menuButton} onClick={openNav} aria-label="Open navigation">
        <Icon name="menu" />
      </button>
      {leading}
      <div className={styles.topbarTitle}>
        <h1>{title}</h1>
        {meta && <span className={styles.topbarMeta}>{meta}</span>}
      </div>
      {actions && <div className={styles.topbarActions}>{actions}</div>}
    </header>
  );
}
