import { memo, type KeyboardEvent, type PointerEvent } from "react";
import { NODE_GEOMETRY, type GraphNode as GraphNodeModel } from "../../graph/model";
import { typeLabel } from "../../lib/format";
import styles from "./graph.module.css";

export type NodeState = "default" | "selected" | "hovered" | "dimmed";

function initials(label: string): string {
  return label
    .split(/\s+/)
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

export const GraphNode = memo(function GraphNode({
  node,
  state,
  onSelect,
  onHover,
}: {
  node: GraphNodeModel;
  state: NodeState;
  onSelect: (id: string) => void;
  onHover: (id: string | null) => void;
}) {
  const historical = node.status !== "current";
  const className = [
    styles.node,
    styles[`node_${state}`],
    historical ? styles.nodeHistorical : "",
  ].join(" ");

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(node.id);
    }
  };
  const stop = (e: PointerEvent) => e.stopPropagation();

  const top = -node.height / 2;

  return (
    <g
      className={className}
      transform={`translate(${node.x} ${node.y})`}
      role="button"
      tabIndex={0}
      aria-label={`${typeLabel(node.type)}: ${node.label}${historical ? " (historical)" : ""}`}
      aria-pressed={state === "selected"}
      onPointerDown={stop}
      onClick={() => onSelect(node.id)}
      onPointerEnter={() => onHover(node.id)}
      onPointerLeave={() => onHover(null)}
      onFocus={() => onHover(node.id)}
      onBlur={() => onHover(null)}
      onKeyDown={onKeyDown}
    >
      {node.type === "person" && (
        <>
          <circle className={styles.shape} cx={0} cy={top + NODE_GEOMETRY.person.radius} r={NODE_GEOMETRY.person.radius} />
          <text className={styles.initials} x={0} y={top + NODE_GEOMETRY.person.radius} dy="0.35em">
            {initials(node.label)}
          </text>
          <text className={styles.label} x={0} y={node.height / 2 - 4}>
            {node.label}
          </text>
        </>
      )}

      {node.type === "event" && (
        <>
          <rect
            className={styles.shape}
            x={-NODE_GEOMETRY.event.half * 0.71}
            y={top + NODE_GEOMETRY.event.half * 0.29}
            width={NODE_GEOMETRY.event.half * 1.42}
            height={NODE_GEOMETRY.event.half * 1.42}
            rx={1.5}
            transform={`rotate(45 0 ${top + NODE_GEOMETRY.event.half})`}
          />
          <text className={styles.label} x={0} y={node.height / 2 - 4}>
            {node.label}
          </text>
        </>
      )}

      {(node.type === "project" || node.type === "fact" || node.type === "preference") && (
        <>
          <rect
            className={styles.shape}
            x={-node.width / 2}
            y={top}
            width={node.width}
            height={node.height}
            rx={node.type === "project" ? 6 : node.type === "preference" ? node.height / 2 : 3}
          />
          <text
            className={node.type === "project" ? styles.labelProject : styles.labelInside}
            x={0}
            y={0}
            dy="0.35em"
          >
            {node.label}
          </text>
        </>
      )}
    </g>
  );
});
