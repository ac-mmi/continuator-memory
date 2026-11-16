import { describe, expect, it } from "vitest";
import type { ImportResult } from "../data/types";
import {
  extractModeLabel,
  importPersistenceNote,
  importStatistics,
  importSucceeded,
} from "./importSummary";

function baseResult(overrides: Partial<ImportResult> = {}): ImportResult {
  return {
    ok: true,
    source: {
      id: "imported",
      title: "Imported · conv.txt",
      kind: "conversation",
      createdAt: "2024-01-01T00:00:00Z",
      preview: "hi",
      messageCount: 2,
      memoryCount: 2,
      origin: "imported",
    },
    report: {
      filename: "conv.txt",
      nMessages: 2,
      nClaims: 2,
      nMemories: 2,
      nRelations: 0,
      failedWindows: 0,
      windows: 1,
      extract: "message_claims",
      bprimeLinks: 0,
    },
    health: {
      ok: true,
      sampleId: "imported",
      conversationId: "imported",
      nMemories: 2,
      nClaims: 2,
      nMessages: 2,
    },
    ...overrides,
  };
}

describe("importStatistics", () => {
  it("surfaces only truthful backend report fields", () => {
    const stats = importStatistics(baseResult());
    expect(stats).toEqual(
      expect.arrayContaining([
        { label: "Messages processed", value: 2 },
        { label: "Claims extracted", value: 2 },
        { label: "Persistent memories", value: 2 },
        { label: "Extraction windows", value: 1 },
        { label: "Extraction mode", value: "One memory per message" },
        { label: "Conversation", value: "imported" },
        { label: "Source file", value: "conv.txt" },
      ]),
    );
    expect(stats.some((s) => s.label === "Relations")).toBe(false);
  });

  it("does not invent statistics when fields are missing", () => {
    const result = baseResult({
      report: {
        filename: "x.txt",
        nMessages: undefined as unknown as number,
        nClaims: undefined as unknown as number,
        nMemories: 3,
        nRelations: 0,
        failedWindows: 0,
        windows: 0,
        extract: "v2_atomizer",
        bprimeLinks: 0,
      },
    });
    const stats = importStatistics(result);
    expect(stats.find((s) => s.label === "Messages processed")).toBeUndefined();
    expect(stats.find((s) => s.label === "Claims extracted")).toBeUndefined();
    expect(stats.find((s) => s.label === "Persistent memories")).toEqual({
      label: "Persistent memories",
      value: 3,
    });
    expect(stats.find((s) => s.label === "Extraction mode")?.value).toBe("V2 atomizer");
  });

  it("includes relations only when the backend reports a positive count", () => {
    const stats = importStatistics(
      baseResult({
        report: { ...baseResult().report, nRelations: 4 },
      }),
    );
    expect(stats).toContainEqual({ label: "Relations", value: 4 });
  });
});

describe("importSucceeded / persistence", () => {
  it("rejects failed or incomplete results", () => {
    expect(importSucceeded(null)).toBe(false);
    expect(importSucceeded(baseResult({ ok: false }))).toBe(false);
  });

  it("describes imported workspaces as saved", () => {
    expect(importPersistenceNote(baseResult())).toMatch(/import database/i);
  });

  it("does not claim persistence for demo mode", () => {
    expect(
      importPersistenceNote(
        baseResult({
          health: { ok: true, sampleId: "demo" },
        }),
      ),
    ).toMatch(/Demo data/i);
  });
});

describe("extractModeLabel", () => {
  it("maps API extract codes", () => {
    expect(extractModeLabel("message_claims")).toBe("One memory per message");
    expect(extractModeLabel("memory_extraction")).toBe("Memory extraction");
    expect(extractModeLabel("v2_atomizer")).toBe("V2 atomizer");
    expect(extractModeLabel(undefined)).toBeNull();
  });
});
