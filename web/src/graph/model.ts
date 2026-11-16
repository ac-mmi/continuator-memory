import type { GraphSnapshot, Memory, MemoryStatus, MemoryType } from "../data/types";

export interface GraphNode {
  id: string;
  label: string;
  type: MemoryType;
  status: MemoryStatus;
  /** Bounding box of the shape plus its label, centred on (x, y). */
  width: number;
  height: number;
  x: number;
  y: number;
  memory: Memory;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  status: MemoryStatus;
}

export interface GraphModel {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export const DEFAULT_MAX_NODES = 40;

/** Geometry per node type: shape semantics instead of colour. */
export const NODE_GEOMETRY = {
  person: { radius: 20, labelGap: 8 },
  event: { half: 12, labelGap: 8 },
  project: { height: 34, padX: 14, fontSize: 13 },
  fact: { height: 26, padX: 10, fontSize: 12 },
  preference: { height: 26, padX: 14, fontSize: 12 },
} as const;

const LABEL_FONT = 12;
const LABEL_LINE = 16;

/** Rough text width for the system sans stack; good enough for layout. */
export function estimateTextWidth(text: string, fontSize = LABEL_FONT): number {
  let units = 0;
  for (const ch of text) {
    if (/[ilj.,:;'|!·]/.test(ch)) units += 0.3;
    else if (/[mwMW]/.test(ch)) units += 0.85;
    else if (/[A-Z0-9]/.test(ch)) units += 0.65;
    else if (ch === " ") units += 0.3;
    else units += 0.55;
  }
  return Math.ceil(units * fontSize);
}

export function nodeSize(type: MemoryType, label: string): { width: number; height: number } {
  switch (type) {
    case "person": {
      const g = NODE_GEOMETRY.person;
      return {
        width: Math.max(g.radius * 2, estimateTextWidth(label)),
        height: g.radius * 2 + g.labelGap + LABEL_LINE,
      };
    }
    case "event": {
      const g = NODE_GEOMETRY.event;
      return {
        width: Math.max(g.half * 2, estimateTextWidth(label)),
        height: g.half * 2 + g.labelGap + LABEL_LINE,
      };
    }
    case "project":
    case "fact":
    case "preference": {
      const g = NODE_GEOMETRY[type];
      return { width: estimateTextWidth(label, g.fontSize) + g.padX * 2, height: g.height };
    }
  }
}

/**
 * Converts a memory snapshot into a curated graph model. When there are more
 * memories than `maxNodes`, the best-connected current memories are kept.
 */
export function toGraphModel(snapshot: GraphSnapshot, maxNodes = DEFAULT_MAX_NODES): GraphModel {
  const degree = new Map<string, number>();
  for (const r of snapshot.relationships) {
    degree.set(r.fromId, (degree.get(r.fromId) ?? 0) + 1);
    degree.set(r.toId, (degree.get(r.toId) ?? 0) + 1);
  }

  const ranked = [...snapshot.memories].sort(
    (a, b) =>
      Number(b.status === "current") - Number(a.status === "current") ||
      (degree.get(b.id) ?? 0) - (degree.get(a.id) ?? 0) ||
      a.id.localeCompare(b.id),
  );
  const kept = ranked.slice(0, maxNodes);
  const keptIds = new Set(kept.map((m) => m.id));

  const nodes: GraphNode[] = kept
    .sort((a, b) => a.id.localeCompare(b.id))
    .map((memory) => ({
      id: memory.id,
      label: memory.title,
      type: memory.type,
      status: memory.status,
      ...nodeSize(memory.type, memory.title),
      x: 0,
      y: 0,
      memory,
    }));

  const edges: GraphEdge[] = snapshot.relationships
    .filter((r) => keptIds.has(r.fromId) && keptIds.has(r.toId))
    .map((r) => ({ id: r.id, source: r.fromId, target: r.toId, label: r.label, status: r.status }));

  return { nodes, edges };
}

/** Centre of the node's shape (labelled circles/diamonds sit above their label). */
export function nodeAnchor(node: GraphNode): { x: number; y: number } {
  const top = node.y - node.height / 2;
  if (node.type === "person") return { x: node.x, y: top + NODE_GEOMETRY.person.radius };
  if (node.type === "event") return { x: node.x, y: top + NODE_GEOMETRY.event.half };
  return { x: node.x, y: node.y };
}

/** Distance from the shape centre to its outline along unit direction (ux, uy). */
function shapeExtent(node: GraphNode, ux: number, uy: number): number {
  const ax = Math.abs(ux);
  const ay = Math.abs(uy);
  if (node.type === "person") return NODE_GEOMETRY.person.radius;
  if (node.type === "event") return NODE_GEOMETRY.event.half / Math.max(ax + ay, 1e-6);
  const hw = node.width / 2;
  const hh = node.height / 2;
  return Math.min(ax > 1e-6 ? hw / ax : Infinity, ay > 1e-6 ? hh / ay : Infinity);
}

/** The part of an edge between the two shape outlines. */
export function visibleSegment(a: GraphNode, b: GraphNode) {
  const pa = nodeAnchor(a);
  const pb = nodeAnchor(b);
  const dx = pb.x - pa.x;
  const dy = pb.y - pa.y;
  const length = Math.hypot(dx, dy) || 1;
  const ux = dx / length;
  const uy = dy / length;
  const ea = shapeExtent(a, ux, uy);
  const eb = shapeExtent(b, ux, uy);
  return {
    from: pa,
    to: pb,
    mid: {
      x: pa.x + ux * (ea + (length - ea - eb) / 2),
      y: pa.y + uy * (ea + (length - ea - eb) / 2),
    },
    visibleLength: Math.max(0, length - ea - eb),
  };
}

export function neighborsOf(edges: GraphEdge[], id: string): Set<string> {
  const result = new Set<string>([id]);
  for (const e of edges) {
    if (e.source === id) result.add(e.target);
    if (e.target === id) result.add(e.source);
  }
  return result;
}
