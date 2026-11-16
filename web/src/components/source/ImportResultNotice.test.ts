import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import type { ImportResult } from "../../data/types";
import { ImportResultNotice } from "./ImportConversation";

function sample(ok = true): ImportResult {
  return {
    ok,
    source: {
      id: "imported",
      title: "Imported · ferry.txt",
      kind: "conversation",
      createdAt: "2024-01-01T00:00:00Z",
      preview: "ferry",
      messageCount: 2,
      memoryCount: 1,
      origin: "imported",
    },
    report: {
      filename: "ferry.txt",
      nMessages: 2,
      nClaims: 1,
      nMemories: 1,
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
      nMemories: 1,
      nClaims: 1,
      nMessages: 2,
    },
  };
}

function render(result: ImportResult): string {
  return renderToStaticMarkup(
    createElement(MemoryRouter, null, createElement(ImportResultNotice, { result })),
  );
}

describe("ImportResultNotice", () => {
  it("shows a truthful Memory ready summary with next actions", () => {
    const html = render(sample(true));
    expect(html).toContain("data-testid=\"import-ready\"");
    expect(html).toContain("Memory ready");
    expect(html).toContain("Messages processed");
    expect(html).toContain(">2<");
    expect(html).toContain("Claims extracted");
    expect(html).toContain("Persistent memories");
    expect(html).toContain("Conversation");
    expect(html).toContain("Persistent memory");
    expect(html).toContain("Explore Memories");
    expect(html).toContain('href="/memories"');
    expect(html).toContain('href="/"');
    expect(html).toContain('href="/?intent=context"');
    expect(html).toContain("import database");
  });

  it("renders nothing for failed imports", () => {
    expect(render(sample(false))).toBe("");
  });

  it("surfaces failed-window warnings from the backend report", () => {
    const result = sample(true);
    result.report.failedWindows = 1;
    result.report.windows = 4;
    const html = render(result);
    expect(html).toContain("1 of 4 extraction windows failed");
  });
});
