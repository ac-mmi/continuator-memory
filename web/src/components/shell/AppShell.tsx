import { useCallback, useEffect, useMemo, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import type { ImportResult } from "../../data/types";
import { ShellContext } from "./ShellContext";
import { Sidebar } from "./Sidebar";
import styles from "./shell.module.css";

export function AppShell() {
  const [navOpen, setNavOpen] = useState(false);
  const [conversationsVersion, setConversationsVersion] = useState(0);
  const [lastImport, setLastImport] = useState<ImportResult | null>(null);
  const [workspaceVersion, setWorkspaceVersion] = useState(0);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => setNavOpen(false), [location.pathname]);

  useEffect(() => {
    if (!navOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setNavOpen(false);
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [navOpen]);

  const refreshConversations = useCallback(() => setConversationsVersion((v) => v + 1), []);
  const onImported = useCallback(
    (result: ImportResult) => {
      setLastImport(result);
      setWorkspaceVersion((v) => v + 1);
      setConversationsVersion((v) => v + 1);
      navigate("/sources");
    },
    [navigate],
  );
  const context = useMemo(
    () => ({
      openNav: () => setNavOpen(true),
      conversationsVersion,
      refreshConversations,
      lastImport,
      onImported,
    }),
    [conversationsVersion, refreshConversations, lastImport, onImported],
  );

  return (
    <ShellContext.Provider value={context}>
      <div className={styles.shell}>
        <Sidebar open={navOpen} onClose={() => setNavOpen(false)} />
        <div
          className={navOpen ? styles.scrimVisible : styles.scrim}
          onClick={() => setNavOpen(false)}
          aria-hidden="true"
        />
        <main className={styles.main}>
          <Outlet key={workspaceVersion} />
        </main>
      </div>
    </ShellContext.Provider>
  );
}
