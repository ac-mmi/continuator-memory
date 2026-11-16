import type { AnswerEvidence, RetrievedMemoryEvidence } from "../data/types";

/** Split retrieved memories from those actually supplied in the generation prompt. */
export function partitionPromptMemories(evidence: AnswerEvidence): {
  retrieved: RetrievedMemoryEvidence[];
  inPrompt: RetrievedMemoryEvidence[];
  retrievedOnly: RetrievedMemoryEvidence[];
  tracked: boolean;
} {
  const rows = evidence.memories ?? [];
  if (rows.length > 0) {
    return {
      retrieved: rows,
      inPrompt: rows.filter((row) => row.inPrompt),
      retrievedOnly: rows.filter((row) => !row.inPrompt),
      tracked: true,
    };
  }
  const prompt = new Set(evidence.promptMemoryIds ?? []);
  const retrieved = evidence.memoryIds.map((id) => ({
    id,
    statement: id,
    sourceMessageIds: [],
    inPrompt: prompt.size > 0 ? prompt.has(id) : false,
  }));
  return {
    retrieved,
    inPrompt: retrieved.filter((row) => row.inPrompt),
    retrievedOnly: retrieved.filter((row) => !row.inPrompt),
    tracked: prompt.size > 0 || (evidence.promptMemoryIds?.length === 0 && evidence.memoryIds.length > 0),
  };
}
