import type { ImportResult } from "../data/types";

export interface ImportStat {
  label: string;
  value: string | number;
}

/** Human labels for `report.extract` values returned by the API. */
export function extractModeLabel(extract: string | undefined): string | null {
  if (!extract) return null;
  if (extract === "message_claims" || extract === "messages") return "One memory per message";
  if (extract === "memory_extraction" || extract === "memories") return "Memory extraction";
  if (extract === "v2_atomizer" || extract === "v2") return "V2 atomizer";
  return extract;
}

/**
 * Truthful post-import statistics from `POST /import`.
 * Only includes fields the backend actually returned as numbers.
 */
export function importStatistics(result: ImportResult): ImportStat[] {
  const report = result.report;
  if (!report) return [];
  const stats: ImportStat[] = [];
  if (typeof report.nMessages === "number") {
    stats.push({ label: "Messages processed", value: report.nMessages });
  }
  if (typeof report.nClaims === "number") {
    stats.push({ label: "Claims extracted", value: report.nClaims });
  }
  if (typeof report.nMemories === "number") {
    stats.push({ label: "Persistent memories", value: report.nMemories });
  }
  if (typeof report.nRelations === "number" && report.nRelations > 0) {
    stats.push({ label: "Relations", value: report.nRelations });
  }
  if (typeof report.windows === "number" && report.windows > 0) {
    stats.push({ label: "Extraction windows", value: report.windows });
  }
  const mode = extractModeLabel(report.extract);
  if (mode) stats.push({ label: "Extraction mode", value: mode });
  const conversationId = result.health?.conversationId ?? result.source?.id;
  if (conversationId) {
    stats.push({ label: "Conversation", value: conversationId });
  }
  if (report.filename) {
    stats.push({ label: "Source file", value: report.filename });
  }
  return stats;
}

/** Persistence wording that matches the active workspace mode. */
export function importPersistenceNote(result: ImportResult): string {
  const sampleId = result.health?.sampleId ?? result.health?.origin;
  if (sampleId === "imported") {
    return "Memory saved to this Continuator workspace (server import database).";
  }
  if (sampleId === "fixture") {
    return "Workspace still on the sample fixture; import did not replace it.";
  }
  if (sampleId === "demo") {
    return "Demo data only — connect the Continuator API to persist imports.";
  }
  return "Memory is available in the current Continuator workspace.";
}

export function importSucceeded(result: ImportResult | null | undefined): boolean {
  return Boolean(result && result.ok === true && result.report);
}
