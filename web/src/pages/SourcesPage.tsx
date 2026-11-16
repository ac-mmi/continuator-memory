import { useState } from "react";
import { TopBar } from "../components/shell/TopBar";
import { useShell } from "../components/shell/ShellContext";
import { ImportConversation, ImportResultNotice } from "../components/source/ImportConversation";
import { SourceCard } from "../components/source/SourceCard";
import { EmptyState, ErrorNotice, LoadingRows } from "../components/ui/EmptyState";
import { SearchInput } from "../components/ui/SearchInput";
import { useService } from "../data/ServiceProvider";
import { plural } from "../lib/format";
import { matchesText } from "../lib/text";
import { useAsync } from "../lib/useAsync";
import styles from "./pages.module.css";

export function SourcesPage() {
  const service = useService();
  const { lastImport } = useShell();
  const [query, setQuery] = useState("");
  const sources = useAsync(() => service.listSources(), [service]);
  const visible =
    sources.status === "ready" ? sources.data.filter((s) => matchesText(query, s.title, s.preview)) : [];

  return (
    <>
      <TopBar
        title="Sources"
        meta={sources.status === "ready" ? plural(sources.data.length, "source") : undefined}
      />
      <div className={styles.scroll}>
        <div className={styles.pageNarrow}>
          <div className={styles.pageIntro}>
            <h2 className={styles.pageTitle}>Where memories come from</h2>
            <p className={styles.pageText}>
              Import a conversation to extract claims and consolidate them into persistent memory with
              source provenance. Open a source to see exactly which messages produced which memories.
            </p>
          </div>
          <ImportConversation />
          {lastImport?.ok && <ImportResultNotice result={lastImport} />}
          <div className={styles.filters}>
            <SearchInput value={query} onChange={setQuery} placeholder="Search sources" label="Search sources" />
          </div>

          {sources.status === "loading" && <LoadingRows rows={5} />}
          {sources.status === "error" && <ErrorNotice error={sources.error} onRetry={sources.reload} />}
          {sources.status === "ready" && visible.length === 0 && (
            <EmptyState title="No sources match" description="Try a different search." />
          )}
          {visible.length > 0 && (
            <div className={styles.list}>
              {visible.map((s) => (
                <SourceCard key={s.id} source={s} />
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
