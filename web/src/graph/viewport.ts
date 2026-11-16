import type { Bounds } from "./layout";

export interface Transform {
  x: number;
  y: number;
  k: number;
}

export const MIN_ZOOM = 0.3;
export const MAX_ZOOM = 2.5;

export function clampZoom(k: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, k));
}

/** Transform that centres `bounds` inside a `width` x `height` viewport. */
export function fitTransform(
  bounds: Bounds,
  width: number,
  height: number,
  padding = 48,
  maxZoom = 1.25,
): Transform {
  const bw = Math.max(bounds.maxX - bounds.minX, 1);
  const bh = Math.max(bounds.maxY - bounds.minY, 1);
  const k = clampZoom(
    Math.min((width - padding * 2) / bw, (height - padding * 2) / bh, maxZoom),
  );
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;
  return { x: width / 2 - cx * k, y: height / 2 - cy * k, k };
}

/** Zooms by `factor` keeping the screen point (px, py) fixed. */
export function zoomAt(t: Transform, factor: number, px: number, py: number): Transform {
  const k = clampZoom(t.k * factor);
  const ratio = k / t.k;
  return { k, x: px - (px - t.x) * ratio, y: py - (py - t.y) * ratio };
}

/** Pans so that the graph point (gx, gy) sits at the viewport centre. */
export function centerOn(t: Transform, gx: number, gy: number, width: number, height: number): Transform {
  return { k: t.k, x: width / 2 - gx * t.k, y: height / 2 - gy * t.k };
}
