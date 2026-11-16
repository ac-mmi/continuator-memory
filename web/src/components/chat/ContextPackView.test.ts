import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ContextPack } from "../../data/types";
import { ContextPackView } from "./ContextPackView";

const SAMPLE: ContextPack = {
  query: "weekday bicycle",
  mode: "baseline",
  memories: [
    {
      id: "mem:seed",
      text: "Alex commutes on a weekday bicycle.",
      rank: 1,
      score: 0.9,
      similarity: 0.9,
      status: "current",
      sourceMessageIds: ["m-commute"],
      sources: [
        {
          messageId: "m-commute",
          claimId: "seed",
          role: "user",
          text: "Alex commutes on a weekday bicycle.",
          timestamp: "2024-03-01",
        },
      ],
    },
  ],
  evidence: [
    {
      messageId: "m-commute",
      role: "user",
      text: "Alex commutes on a weekday bicycle.",
      memoryIds: ["mem:seed"],
      truncated: false,
      timestamp: "2024-03-01",
    },
  ],
  citations: [{ memoryId: "mem:seed", messageIds: ["m-commute"], unresolved: [] }],
  formattedContext:
    "RELEVANT LONG-TERM MEMORY\nQuery: weekday bicycle\n\n[1] memory mem:seed\nStatement: Alex commutes on a weekday bicycle.\n",
  diagnostics: {
    mode: "baseline",
    k: 1,
    seedCount: 1,
    expandedCount: 0,
    sourceCount: 1,
    selectedSourceCount: 1,
    droppedSourceCount: 0,
    budgetChars: 50000,
    usedChars: 35,
    answerGeneration: "not_run",
  },
};

function render(pack: ContextPack = SAMPLE): string {
  return renderToStaticMarkup(
    createElement(MemoryRouter, null, createElement(ContextPackView, { pack })),
  );
}

describe("ContextPackView", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders query, memories, evidence, and formattedContext", () => {
    const html = render();
    expect(html).toContain("weekday bicycle");
    expect(html).toContain("Alex commutes on a weekday bicycle.");
    expect(html).toContain("m-commute");
    expect(html).toContain("No answer was generated");
    expect(html).toContain("data-testid=\"formatted-context\"");
    expect(html).toContain("RELEVANT LONG-TERM MEMORY");
  });

  it("Copy context writes formattedContext to the clipboard", async () => {
    const writeText = vi.fn(async () => undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    // Exercise the same string the button copies.
    expect(SAMPLE.formattedContext).toContain("RELEVANT LONG-TERM MEMORY");
    await navigator.clipboard.writeText(SAMPLE.formattedContext);
    expect(writeText).toHaveBeenCalledWith(SAMPLE.formattedContext);
  });
});
