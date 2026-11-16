import type { GraphModel, GraphNode } from "./model";

export interface LayoutOptions {
  iterations?: number;
  linkDistance?: number;
  repulsion?: number;
  gravity?: number;
  /** Minimum clear space between node bounding boxes. */
  gap?: number;
}

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * Deterministic force-directed layout computed once, synchronously. There is
 * no random seeding and no running simulation, so the same model always
 * yields the same positions and nodes never move on their own.
 */
export function layoutGraph(model: GraphModel, options: LayoutOptions = {}): GraphModel {
  const {
    iterations = 400,
    linkDistance = 160,
    repulsion = 22000,
    gravity = 0.01,
    gap = 32,
  } = options;

  const index = new Map(model.nodes.map((n, i) => [n.id, i]));
  const degree = new Array(model.nodes.length).fill(0);
  const links: [number, number][] = [];
  for (const e of model.edges) {
    const a = index.get(e.source);
    const b = index.get(e.target);
    if (a === undefined || b === undefined || a === b) continue;
    links.push([a, b]);
    degree[a]++;
    degree[b]++;
  }

  // Best-connected nodes start nearest the centre on a golden-angle spiral.
  const order = model.nodes
    .map((_node, i) => i)
    .sort((a, b) => degree[b] - degree[a] || model.nodes[a].id.localeCompare(model.nodes[b].id));
  const xs = new Float64Array(model.nodes.length);
  const ys = new Float64Array(model.nodes.length);
  order.forEach((nodeIndex, rank) => {
    const r = 40 * Math.sqrt(rank + 0.5);
    xs[nodeIndex] = r * Math.cos(rank * GOLDEN_ANGLE);
    ys[nodeIndex] = r * Math.sin(rank * GOLDEN_ANGLE);
  });

  const n = model.nodes.length;
  const dx = new Float64Array(n);
  const dy = new Float64Array(n);

  for (let iter = 0; iter < iterations; iter++) {
    const cooling = 1 - iter / iterations;
    const maxStep = 4 + 36 * cooling;
    dx.fill(0);
    dy.fill(0);

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let vx = xs[i] - xs[j];
        let vy = ys[i] - ys[j];
        let d2 = vx * vx + vy * vy;
        if (d2 < 0.01) {
          vx = 0.1 * (i - j);
          vy = 0.1;
          d2 = vx * vx + vy * vy;
        }
        const f = repulsion / d2;
        const d = Math.sqrt(d2);
        dx[i] += (vx / d) * f;
        dy[i] += (vy / d) * f;
        dx[j] -= (vx / d) * f;
        dy[j] -= (vy / d) * f;
      }
    }

    for (const [a, b] of links) {
      const vx = xs[b] - xs[a];
      const vy = ys[b] - ys[a];
      const d = Math.max(Math.sqrt(vx * vx + vy * vy), 0.01);
      const f = (d - linkDistance) * 0.06;
      dx[a] += (vx / d) * f;
      dy[a] += (vy / d) * f;
      dx[b] -= (vx / d) * f;
      dy[b] -= (vy / d) * f;
    }

    for (let i = 0; i < n; i++) {
      dx[i] -= xs[i] * gravity;
      dy[i] -= ys[i] * gravity;
      const len = Math.sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
      const scale = len > maxStep ? maxStep / len : 1;
      xs[i] += dx[i] * scale;
      ys[i] += dy[i] * scale;
    }
  }

  resolveOverlaps(model.nodes, xs, ys, gap);

  const nodes: GraphNode[] = model.nodes.map((node, i) => ({
    ...node,
    x: Math.round(xs[i]),
    y: Math.round(ys[i]),
  }));
  return { nodes, edges: model.edges };
}

/** Pushes apart overlapping bounding boxes along the axis of least overlap. */
function resolveOverlaps(nodes: GraphNode[], xs: Float64Array, ys: Float64Array, gap: number) {
  for (let pass = 0; pass < 80; pass++) {
    let moved = false;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const ox = (nodes[i].width + nodes[j].width) / 2 + gap - Math.abs(xs[i] - xs[j]);
        const oy = (nodes[i].height + nodes[j].height) / 2 + gap - Math.abs(ys[i] - ys[j]);
        if (ox <= 0 || oy <= 0) continue;
        moved = true;
        if (ox < oy) {
          const s = (xs[i] < xs[j] || (xs[i] === xs[j] && i < j) ? -1 : 1) * (ox / 2);
          xs[i] += s;
          xs[j] -= s;
        } else {
          const s = (ys[i] < ys[j] || (ys[i] === ys[j] && i < j) ? -1 : 1) * (oy / 2);
          ys[i] += s;
          ys[j] -= s;
        }
      }
    }
    if (!moved) return;
  }
}

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export function graphBounds(nodes: GraphNode[]): Bounds {
  if (nodes.length === 0) return { minX: -100, minY: -100, maxX: 100, maxY: 100 };
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const n of nodes) {
    minX = Math.min(minX, n.x - n.width / 2);
    maxX = Math.max(maxX, n.x + n.width / 2);
    minY = Math.min(minY, n.y - n.height / 2);
    maxY = Math.max(maxY, n.y + n.height / 2);
  }
  return { minX, minY, maxX, maxY };
}
