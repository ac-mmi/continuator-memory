import { describe, expect, it } from "vitest";
import { partitionPromptMemories } from "./answerEvidence";
import type { AnswerEvidence } from "../data/types";

describe("partitionPromptMemories", () => {
  it("distinguishes retrieved memories from those passed to the model", () => {
    const evidence: AnswerEvidence = {
      memoryIds: ["seed", "gold"],
      sourceIds: ["src"],
      promptMemoryIds: ["seed"],
      droppedMemoryIds: ["gold"],
      memories: [
        {
          id: "seed",
          statement: "Alex commutes on a weekday bicycle.",
          sourceMessageIds: ["m-commute"],
          inPrompt: true,
        },
        {
          id: "gold",
          statement: "Alex owns a Fender Stratocaster.",
          sourceMessageIds: ["m-guitar"],
          inPrompt: false,
        },
      ],
    };
    const parts = partitionPromptMemories(evidence);
    expect(parts.tracked).toBe(true);
    expect(parts.inPrompt.map((m) => m.id)).toEqual(["seed"]);
    expect(parts.retrievedOnly.map((m) => m.id)).toEqual(["gold"]);
    expect(parts.retrieved.map((m) => m.id)).toEqual(["seed", "gold"]);
  });

  it("does not imply demo memories were supplied to a model", () => {
    const evidence: AnswerEvidence = {
      memoryIds: ["mem-filemind"],
      sourceIds: ["src-filemind"],
    };
    const parts = partitionPromptMemories(evidence);
    expect(parts.tracked).toBe(false);
    expect(parts.inPrompt).toEqual([]);
    expect(parts.retrievedOnly.map((m) => m.id)).toEqual(["mem-filemind"]);
  });
});
