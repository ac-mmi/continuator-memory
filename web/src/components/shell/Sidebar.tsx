import { Link, NavLink, useLocation } from "react-router-dom";
import { isDemoMode } from "../../data";
import { useService } from "../../data/ServiceProvider";
import { relativeTime } from "../../lib/format";
import { useAsync } from "../../lib/useAsync";
import { ImportConversation } from "../source/ImportConversation";
import { Icon, type IconName } from "../ui/Icon";
import { Wordmark } from "../ui/Logo";
import { useShell } from "./ShellContext";
import styles from "./shell.module.css";

const NAV: { to: string; label: string; icon: IconName; end?: boolean }[] = [
  { to: "/", label: "Home", icon: "chat", end: true },
  { to: "/memories", label: "Memories", icon: "memory" },
  { to: "/graph", label: "Graph", icon: "graph" },
  { to: "/sources", label: "Sources", icon: "sources" },
];

export function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const service = useService();
  const { conversationsVersion } = useShell();
  const inConversation = useLocation().pathname.startsWith("/c/");
  const conversations = useAsync(() => service.listConversations(), [service, conversationsVersion]);

  return (
    <aside className={open ? styles.sidebarOpen : styles.sidebar} aria-label="Primary">
      <div className={styles.sidebarHeader}>
        <Link to="/" className={styles.brand} aria-label="Continuator home">
          <Wordmark />
        </Link>
        <button type="button" className={styles.sidebarClose} onClick={onClose} aria-label="Close navigation">
          <Icon name="close" />
        </button>
      </div>

      <Link to="/" className={styles.newChat}>
        <Icon name="plus" size={15} />
        New conversation
      </Link>
      <ImportConversation compact />

      <nav className={styles.nav}>
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              isActive || (item.to === "/" && inConversation) ? styles.navItemActive : styles.navItem
            }
          >
            <Icon name={item.icon} />
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className={styles.recents}>
        <p className={styles.sectionLabel}>Recent</p>
        {conversations.status === "error" && (
          <p className={styles.recentsEmpty} role="alert">Could not load conversations.</p>
        )}
        {conversations.status === "ready" && conversations.data.length === 0 && (
          <p className={styles.recentsEmpty}>No conversations yet.</p>
        )}
        {conversations.status === "ready" &&
          conversations.data.map((c) => (
            <NavLink
              key={c.id}
              to={`/c/${c.id}`}
              className={({ isActive }) => (isActive ? styles.recentActive : styles.recent)}
            >
              <span className={styles.recentTitle}>{c.title}</span>
              <span className={styles.recentMeta}>{relativeTime(c.updatedAt)}</span>
            </NavLink>
          ))}
      </div>

      <div className={styles.sidebarFooter}>
        <NavLink
          to="/settings"
          className={({ isActive }) => (isActive ? styles.navItemActive : styles.navItem)}
        >
          <Icon name="settings" />
          Settings
        </NavLink>
        {isDemoMode() && (
          <p className={styles.modeNote}>
            <span className={styles.modeDot} />
            Demo data
          </p>
        )}
      </div>
    </aside>
  );
}
