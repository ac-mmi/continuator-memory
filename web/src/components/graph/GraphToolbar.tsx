import { MEMORY_TYPES, type MemoryType } from "../../data/types";
import { typeLabel } from "../../lib/format";
import { TypeGlyph } from "../ui/Badges";
import { FilterMenu } from "../ui/FilterMenu";
import { SearchInput } from "../ui/SearchInput";
import styles from "./graph.module.css";

export function GraphToolbar({
  query,
  onQueryChange,
  onSubmitQuery,
  types,
  onTypesChange,
  showHistorical,
  onShowHistoricalChange,
  matchCount,
  nodeCount,
}: {
  query: string;
  onQueryChange: (q: string) => void;
  onSubmitQuery: () => void;
  types: MemoryType[];
  onTypesChange: (types: MemoryType[]) => void;
  showHistorical: boolean;
  onShowHistoricalChange: (value: boolean) => void;
  matchCount: number | null;
  nodeCount: number;
}) {
  return (
    <div className={styles.toolbar}>
      <form
        className={styles.toolbarSearch}
        onSubmit={(e) => {
          e.preventDefault();
          onSubmitQuery();
        }}
      >
        <SearchInput value={query} onChange={onQueryChange} placeholder="Find in graph" label="Find in graph" />
      </form>
      <FilterMenu
        label="Type"
        options={MEMORY_TYPES.map((t) => ({ value: t, label: typeLabel(t), icon: <TypeGlyph type={t} /> }))}
        selected={types}
        onChange={onTypesChange}
      />
      <label className={styles.toggle}>
        <input
          type="checkbox"
          checked={showHistorical}
          onChange={(e) => onShowHistoricalChange(e.target.checked)}
        />
        <span className={styles.toggleTrack} aria-hidden="true" />
        History
      </label>
      <span className={styles.toolbarCount}>
        {matchCount !== null ? `${matchCount} of ${nodeCount} match` : `${nodeCount} memories`}
      </span>
    </div>
  );
}
