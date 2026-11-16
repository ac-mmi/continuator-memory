import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { GraphToolbar } from "../components/graph/GraphToolbar";
import { MemoryGraph } from "../components/graph/MemoryGraph";
import { TopBar } from "../components/shell/TopBar";
import { StatusBadge, TypeGlyph, TypeTag } from "../components/ui/Badges";
import { ErrorNotice } from "../components/ui/EmptyState";
import { Icon } from "../components/ui/Icon";
import { useService } from "../data/ServiceProvider";
import type { MemoryType } from "../data/types";
import { layoutGraph } from "../graph/layout";
import { toGraphModel, type GraphModel } from "../graph/model";
import { plural, relativeTime } from "../lib/format";
import { matchesText } from "../lib/text";
import { useAsync } from "../lib/useAsync";
import styles from "./pages.module.css";

export function GraphPage() {
  const service = useService();
  const graph = useAsync(async () => layoutGraph(toGraphModel(await service.getGraph())), [service]);

  return (
    <>
      <TopBar
        title="Graph"
        meta={
          graph.status === "ready"
            ? `${plural(graph.data.nodes.length, "memory", "memories")} · ${plural(graph.data.edges.length, "relationship")}`
            : undefined
        }
      />
      {graph.status === "error" && (
        <div className={styles.page}>
          <ErrorNotice error={graph.error} onRetry={graph.reload} />
        </div>
      )}
      {graph.status === "loading" && <div className={styles.graphLoading} />}
      {graph.status === "ready" && <GraphWorkspace model={graph.data} />}
    </>
  );
}

function GraphWorkspace({ model }: { model: GraphModel }) {
  const [params, setParams] = useSearchParams();
  const initialFocus = params.get("focus");
  const [selectedId, setSelectedId] = useState<string | null>(
    initialFocus && model.nodes.some((n) => n.id === initialFocus) ? initialFocus : null,
  );
  const [focusRequest, setFocusRequest] = useState<{ id: string; nonce: number } | null>(null);
  const [query, setQuery] = useState("");
  const [types, setTypes] = useState<MemoryType[]>([]);
  const [showHistorical, setShowHistorical] = useState(true);

  const visibleIds = useMemo(
    () =>
      new Set(
        model.nodes
          .filter((n) => types.length === 0 || types.includes(n.type))
          .filter((n) => showHistorical || n.status === "current")
          .map((n) => n.id),
      ),
    [model, types, showHistorical],
  );

  const matchIds = useMemo(() => {
    if (!query.trim()) return null;
    return new Set(
      model.nodes
        .filter((n) => visibleIds.has(n.id))
        .filter((n) => matchesText(query, n.label, n.memory.statement, n.memory.subject))
        .map((n) => n.id),
    );
  }, [model, query, visibleIds]);

  const select = (id: string | null, center = false) => {
    setSelectedId(id);
    if (id && center) setFocusRequest((prev) => ({ id, nonce: (prev?.nonce ?? 0) + 1 }));
    setParams(id ? { focus: id } : {}, { replace: true });
  };

  const selected = selectedId && visibleIds.has(selectedId) ? model.nodes.find((n) => n.id === selectedId) : undefined;
  const connections = useMemo(() => {
    if (!selected) return [];
    return model.edges
      .filter((e) => e.source === selected.id || e.target === selected.id)
      .map((e) => {
        const otherId = e.source === selected.id ? e.target : e.source;
        return { edge: e, node: model.nodes.find((n) => n.id === otherId)!, outgoing: e.source === selected.id };
      })
      .filter((c) => visibleIds.has(c.node.id));
  }, [model, selected, visibleIds]);

  return (
    <div className={styles.graphPage}>
      <GraphToolbar
        query={query}
        onQueryChange={setQuery}
        onSubmitQuery={() => {
          const first = matchIds && model.nodes.find((n) => matchIds.has(n.id));
          if (first) select(first.id, true);
        }}
        types={types}
        onTypesChange={setTypes}
        showHistorical={showHistorical}
        onShowHistoricalChange={setShowHistorical}
        matchCount={matchIds ? matchIds.size : null}
        nodeCount={visibleIds.size}
      />
      <div className={styles.graphBody}>
        <MemoryGraph
          model={model}
          visibleIds={visibleIds}
          matchIds={matchIds}
          selectedId={selected?.id ?? null}
          focusRequest={focusRequest}
          onSelect={(id) => select(id)}
        />

        <aside className={selected ? styles.graphPanelOpen : styles.graphPanel} aria-label="Selected memory">
          {!selected ? (
            <div className={styles.graphPanelEmpty}>
              <p className={styles.graphPanelTitle}>Select a memory</p>
              <p className={styles.muted}>
                Click a node to see what it says, how it connects, and where it came from.
              </p>
            </div>
          ) : (
            <>
              <div className={styles.graphPanelHead}>
                <TypeTag type={selected.type} />
                <button
                  type="button"
                  className={styles.iconButton}
                  onClick={() => select(null)}
                  aria-label="Clear selection"
                >
                  <Icon name="close" size={14} />
                </button>
              </div>
              <p className={styles.graphPanelTitle}>{selected.label}</p>
              <p className={styles.graphPanelStatement}>{selected.memory.statement}</p>
              <div className={styles.graphPanelMeta}>
                <StatusBadge status={selected.status} />
                <span>Updated {relativeTime(selected.memory.updatedAt)}</span>
                <span>{plural(new Set(selected.memory.sources.map((s) => s.sourceId)).size, "source")}</span>
              </div>
              <Link to={`/memories/${selected.id}`} className={styles.primaryButton}>
                Open memory and provenance
                <Icon name="arrowRight" size={14} />
              </Link>

              {connections.length > 0 && (
                <>
                  <p className={styles.panelSection}>Connections</p>
                  <ul className={styles.connectionList}>
                    {connections.map(({ edge, node, outgoing }) => (
                      <li key={edge.id}>
                        <button type="button" className={styles.connection} onClick={() => select(node.id, true)}>
                          <span className={styles.connectionLabel}>
                            {outgoing ? `${edge.label} →` : `${edge.label} this`}
                          </span>
                          <span className={styles.connectionTarget}>
                            <TypeGlyph type={node.type} />
                            <span className={node.status === "current" ? undefined : styles.muted}>{node.label}</span>
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
