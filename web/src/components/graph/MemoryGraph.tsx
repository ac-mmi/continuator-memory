import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { graphBounds } from "../../graph/layout";
import { estimateTextWidth, neighborsOf, visibleSegment, type GraphModel } from "../../graph/model";
import { centerOn, fitTransform, zoomAt, type Transform } from "../../graph/viewport";
import { Icon } from "../ui/Icon";
import { GraphNode, type NodeState } from "./GraphNode";
import styles from "./graph.module.css";

/** Edge labels are shown for all edges of a focused node up to this degree. */
const MAX_LABELLED_DEGREE = 6;

type Pointer = { x: number; y: number };

export function MemoryGraph({
  model,
  visibleIds,
  matchIds,
  selectedId,
  focusRequest,
  onSelect,
}: {
  /** Laid-out graph. */
  model: GraphModel;
  /** Nodes to render; others are hidden along with their edges. */
  visibleIds: Set<string>;
  /** Search matches; non-matching nodes fade. `null` means no active search. */
  matchIds: Set<string> | null;
  selectedId: string | null;
  /** Changing this value centres the view on the given node. */
  focusRequest: { id: string; nonce: number } | null;
  onSelect: (id: string | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const [transform, setTransform] = useState<Transform | null>(null);
  const [animate, setAnimate] = useState(false);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const pointers = useRef(new Map<number, Pointer>());
  const gesture = useRef<{ moved: boolean; start: Pointer; startT: Transform; pinchDist?: number } | null>(null);

  const nodes = useMemo(() => model.nodes.filter((n) => visibleIds.has(n.id)), [model, visibleIds]);
  const nodeById = useMemo(() => new Map(model.nodes.map((n) => [n.id, n])), [model]);
  const edges = useMemo(
    () => model.edges.filter((e) => visibleIds.has(e.source) && visibleIds.has(e.target)),
    [model, visibleIds],
  );

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) setSize({ width: rect.width, height: rect.height });
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setSize({ width, height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const fit = useCallback(
    (animated: boolean) => {
      if (!size) return;
      const padding = size.width < 640 ? 24 : 56;
      setAnimate(animated);
      setTransform(fitTransform(graphBounds(nodes), size.width, size.height, padding));
    },
    [size, nodes],
  );

  // Fit once when the canvas first gets a size.
  const fitted = useRef(false);
  useEffect(() => {
    if (size && !fitted.current) {
      fitted.current = true;
      fit(false);
    }
  }, [size, fit]);

  useEffect(() => {
    if (!focusRequest || !size || !transform) return;
    const node = nodeById.get(focusRequest.id);
    if (!node) return;
    setAnimate(true);
    setTransform((t) => (t ? centerOn(t, node.x, node.y, size.width, size.height) : t));
    // Only react to new requests, not to every transform change.
  }, [focusRequest, size !== null, transform !== null]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = svg.getBoundingClientRect();
      const scale = e.ctrlKey ? 0.01 : e.deltaMode === 1 ? 0.05 : 0.0015;
      const factor = Math.exp(-e.deltaY * scale);
      setAnimate(false);
      setTransform((t) => (t ? zoomAt(t, factor, e.clientX - rect.left, e.clientY - rect.top) : t));
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, []);

  const zoomBy = (factor: number) => {
    if (!size) return;
    setAnimate(true);
    setTransform((t) => (t ? zoomAt(t, factor, size.width / 2, size.height / 2) : t));
  };

  const localPoint = (e: PointerEvent): Pointer => {
    const rect = svgRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onPointerDown = (e: PointerEvent<SVGSVGElement>) => {
    if (!transform) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, localPoint(e));
    setAnimate(false);
    const pts = [...pointers.current.values()];
    if (pts.length === 1) {
      gesture.current = { moved: false, start: pts[0], startT: transform };
    } else if (pts.length === 2) {
      gesture.current = {
        moved: true,
        start: midpoint(pts[0], pts[1]),
        startT: transform,
        pinchDist: distance(pts[0], pts[1]),
      };
    }
  };

  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    if (!pointers.current.has(e.pointerId) || !gesture.current) return;
    pointers.current.set(e.pointerId, localPoint(e));
    const g = gesture.current;
    const pts = [...pointers.current.values()];

    if (pts.length >= 2 && g.pinchDist) {
      const mid = midpoint(pts[0], pts[1]);
      const factor = distance(pts[0], pts[1]) / g.pinchDist;
      const panned = { ...g.startT, x: g.startT.x + mid.x - g.start.x, y: g.startT.y + mid.y - g.start.y };
      setTransform(zoomAt(panned, factor, mid.x, mid.y));
      return;
    }

    const dx = pts[0].x - g.start.x;
    const dy = pts[0].y - g.start.y;
    if (!g.moved && Math.hypot(dx, dy) < 4) return;
    g.moved = true;
    setTransform({ ...g.startT, x: g.startT.x + dx, y: g.startT.y + dy });
  };

  const onPointerUp = (e: PointerEvent<SVGSVGElement>) => {
    pointers.current.delete(e.pointerId);
    const g = gesture.current;
    if (pointers.current.size === 0) {
      if (g && !g.moved) onSelect(null);
      gesture.current = null;
    } else if (transform) {
      const [remaining] = pointers.current.values();
      gesture.current = { moved: true, start: remaining, startT: transform };
    }
  };

  const focusId = hoveredId ?? selectedId;
  const active = useMemo(() => (focusId ? neighborsOf(edges, focusId) : null), [edges, focusId]);
  const focusDegree = active ? active.size - 1 : 0;

  const nodeState = (id: string): NodeState => {
    if (id === selectedId) return "selected";
    if (id === hoveredId) return "hovered";
    if (active) return active.has(id) ? "default" : "dimmed";
    if (matchIds) return matchIds.has(id) ? "default" : "dimmed";
    return "default";
  };

  const onHover = useCallback((id: string | null) => setHoveredId(id), []);

  return (
    <div className={styles.canvas} ref={containerRef}>
      <svg
        ref={svgRef}
        className={styles.svg}
        width="100%"
        height="100%"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        role="application"
        aria-label="Memory graph. Drag to pan, scroll to zoom, select a node to inspect it."
      >
        {transform && (
          <g
            style={{
              transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.k})`,
              transformOrigin: "0 0",
              transition: animate ? "transform 260ms var(--ease)" : "none",
            }}
          >
            <g>
              {edges.map((edge) => {
                const a = nodeById.get(edge.source)!;
                const b = nodeById.get(edge.target)!;
                const { from: pa, to: pb, mid, visibleLength } = visibleSegment(a, b);
                const incident = focusId !== null && (edge.source === focusId || edge.target === focusId);
                const selectedEdge =
                  selectedId !== null && (edge.source === selectedId || edge.target === selectedId) &&
                  (hoveredId === null || incident);
                const searchHit = !matchIds || (matchIds.has(edge.source) && matchIds.has(edge.target));
                const state = selectedEdge
                  ? "selected"
                  : incident
                    ? "hovered"
                    : active || !searchHit
                      ? "dimmed"
                      : "default";
                const showLabel =
                  incident &&
                  visibleLength >= estimateTextWidth(edge.label, 10.5) + 32 &&
                  (focusDegree <= MAX_LABELLED_DEGREE ||
                    (selectedId !== null && hoveredId !== null &&
                      [edge.source, edge.target].includes(selectedId) &&
                      [edge.source, edge.target].includes(hoveredId)));
                return (
                  <g key={edge.id} className={styles[`edge_${state}`]}>
                    <line
                      className={edge.status === "current" ? styles.edge : styles.edgeHistorical}
                      x1={pa.x}
                      y1={pa.y}
                      x2={pb.x}
                      y2={pb.y}
                    />
                    {showLabel && (
                      <text className={styles.edgeLabel} x={mid.x} y={mid.y} dy="0.35em">
                        {edge.label}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
            <g>
              {nodes.map((node) => (
                <GraphNode key={node.id} node={node} state={nodeState(node.id)} onSelect={onSelect} onHover={onHover} />
              ))}
            </g>
          </g>
        )}
      </svg>

      <div className={styles.controls} role="group" aria-label="Zoom controls">
        <button type="button" onClick={() => zoomBy(1.25)} aria-label="Zoom in">
          <Icon name="zoomIn" />
        </button>
        <button type="button" onClick={() => zoomBy(0.8)} aria-label="Zoom out">
          <Icon name="zoomOut" />
        </button>
        <button type="button" onClick={() => fit(true)} aria-label="Fit graph to view">
          <Icon name="fit" />
        </button>
      </div>

      <Legend />
    </div>
  );
}

function Legend() {
  return (
    <div className={styles.legend} aria-hidden="true">
      <span>
        <svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="4.5" /></svg>Person
      </span>
      <span>
        <svg width="14" height="12" viewBox="0 0 14 12"><rect x="1" y="2.5" width="12" height="7" rx="2" /></svg>Project
      </span>
      <span>
        <svg width="14" height="12" viewBox="0 0 14 12"><rect x="1" y="3.5" width="12" height="5" rx="0.8" /></svg>Fact
      </span>
      <span>
        <svg width="14" height="12" viewBox="0 0 14 12"><rect x="1" y="3.5" width="12" height="5" rx="2.5" /></svg>Preference
      </span>
      <span>
        <svg width="12" height="12" viewBox="0 0 12 12"><rect x="3" y="3" width="6" height="6" transform="rotate(45 6 6)" /></svg>Event
      </span>
      <span>
        <svg width="16" height="12" viewBox="0 0 16 12"><line x1="1" y1="6" x2="15" y2="6" strokeDasharray="3 2" /></svg>Historical
      </span>
    </div>
  );
}

function midpoint(a: Pointer, b: Pointer): Pointer {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function distance(a: Pointer, b: Pointer): number {
  return Math.hypot(a.x - b.x, a.y - b.y) || 1;
}
