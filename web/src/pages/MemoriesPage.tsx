import { useState } from "react";
import { MemoryCard } from "../components/memory/MemoryCard";
import { TopBar } from "../components/shell/TopBar";
import { TypeGlyph } from "../components/ui/Badges";
import { EmptyState, ErrorNotice, LoadingRows } from "../components/ui/EmptyState";
import { FilterMenu } from "../components/ui/FilterMenu";
import { SearchInput } from "../components/ui/SearchInput";
import { useService } from "../data/ServiceProvider";
import { MEMORY_TYPES, type MemoryType } from "../data/types";
import { memoriesLabel, typeLabel } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import styles from "./pages.module.css";

type StatusFilter = "current" | "all";

export function MemoriesPage() {
  const service = useService();
  const [query, setQuery] = useState("");
  const [types, setTypes] = useState<MemoryType[]>([]);
  const [status, setStatus] = useState<StatusFilter>("current");

  const memories = useAsync(
    () => service.listMemories({ text: query, types, includeSuperseded: status === "all" }),
    [service, query, types.join(","), status],
  );

  return (
    <>
      <TopBar
        title="Memories"
        meta={memories.status === "ready" ? memoriesLabel(memories.data.length) : undefined}
      />
      <div className={styles.scroll}>
        <div className={styles.page}>
          <div className={styles.pageIntro}>
            <h2 className={styles.pageTitle}>What Continuator remembers</h2>
            <p className={styles.pageText}>
              Every memory links back to the conversations it came from. Outdated memories are kept as history.
            </p>
          </div>

          <div className={styles.filters}>
            <SearchInput value={query} onChange={setQuery} placeholder="Search memories" label="Search memories" />
            <FilterMenu
              label="Type"
              options={MEMORY_TYPES.map((t) => ({ value: t, label: typeLabel(t), icon: <TypeGlyph type={t} /> }))}
              selected={types}
              onChange={setTypes}
            />
            <div className={styles.segmented} role="radiogroup" aria-label="Status">
              {(["current", "all"] as const).map((value) => (
                <button
                  key={value}
                  type="button"
                  role="radio"
                  aria-checked={status === value}
                  className={status === value ? styles.segmentActive : styles.segment}
                  onClick={() => setStatus(value)}
                >
                  {value === "current" ? "Current" : "Include history"}
                </button>
              ))}
            </div>
          </div>

          {memories.status === "loading" && <LoadingRows rows={6} />}
          {memories.status === "error" && <ErrorNotice error={memories.error} onRetry={memories.reload} />}
          {memories.status === "ready" && memories.data.length === 0 && (
            <EmptyState
              title="No memories match"
              description="Try a different search, or include historical memories."
            />
          )}
          {memories.status === "ready" && memories.data.length > 0 && (
            <div className={styles.cardGrid}>
              {memories.data.map((m) => (
                <MemoryCard key={m.id} memory={m} />
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
