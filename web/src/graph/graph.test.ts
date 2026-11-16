import { describe, expect, it } from "vitest";
import { buildDemoData } from "../data/demo/demoData";
import type { GraphSnapshot, Memory } from "../data/types";
import { graphBounds, layoutGraph } from "./layout";
import { neighborsOf, toGraphModel } from "./model";
import { fitTransform, zoomAt } from "./viewport";

const NOW = new Date("2026-10-03T12:00:00Z");
const snapshot = (): GraphSnapshot => {
  const data = buildDemoData(NOW);
  return { memories: data.memories, relationships: data.relationships };
};

describe("toGraphModel", () => {
  it("creates one node per memory and keeps only valid edges", () => {
    const model = toGraphModel(snapshot());
    expect(model.nodes).toHaveLength(snapshot().memories.length);
    const ids = new Set(model.nodes.map((n) => n.id));
    expect(model.edges.every((e) => ids.has(e.source) && ids.has(e.target))).toBe(true);
  });

  it("caps the node count, preferring well-connected current memories", () => {
    const model = toGraphModel(snapshot(), 10);
    expect(model.nodes).toHaveLength(10);
    expect(model.nodes.map((n) => n.id)).toContain("mem-filemind");
    expect(model.nodes.every((n) => n.status === "current")).toBe(true);
  });

  it("finds neighbours in both directions", () => {
    const model = toGraphModel(snapshot());
    const n = neighborsOf(model.edges, "mem-sqlite");
    expect(n.has("mem-filemind")).toBe(true);
    expect(n.has("mem-fts5")).toBe(true);
  });
});

describe("layoutGraph", () => {
  it("is deterministic", () => {
    const a = layoutGraph(toGraphModel(snapshot()));
    const b = layoutGraph(toGraphModel(snapshot()));
    expect(a.nodes.map((n) => [n.x, n.y])).toEqual(b.nodes.map((n) => [n.x, n.y]));
  });

  it("produces non-overlapping node boxes", () => {
    const { nodes } = layoutGraph(toGraphModel(snapshot()));
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        const overlapX = (a.width + b.width) / 2 - Math.abs(a.x - b.x);
        const overlapY = (a.height + b.height) / 2 - Math.abs(a.y - b.y);
        expect(overlapX > 1 && overlapY > 1, `${a.id} overlaps ${b.id}`).toBe(false);
      }
    }
  });

  it("handles an empty graph", () => {
    const { nodes } = layoutGraph({ nodes: [], edges: [] });
    expect(nodes).toEqual([]);
    expect(graphBounds(nodes).maxX).toBeGreaterThan(graphBounds(nodes).minX);
  });

  it("handles a single node", () => {
    const memory = snapshot().memories[0] as Memory;
    const { nodes } = layoutGraph(toGraphModel({ memories: [memory], relationships: [] }));
    expect(Number.isFinite(nodes[0].x) && Number.isFinite(nodes[0].y)).toBe(true);
  });
});

describe("viewport", () => {
  it("fits bounds inside the viewport", () => {
    const t = fitTransform({ minX: -200, minY: -100, maxX: 200, maxY: 100 }, 800, 600, 40);
    const left = -200 * t.k + t.x;
    const right = 200 * t.k + t.x;
    expect(left).toBeGreaterThanOrEqual(39.9);
    expect(right).toBeLessThanOrEqual(760.1);
    expect((left + right) / 2).toBeCloseTo(400);
  });

  it("keeps the zoom anchor fixed", () => {
    const t = { x: 10, y: 20, k: 1 };
    const z = zoomAt(t, 2, 100, 100);
    const gx = (100 - t.x) / t.k;
    expect(gx * z.k + z.x).toBeCloseTo(100);
  });
});
