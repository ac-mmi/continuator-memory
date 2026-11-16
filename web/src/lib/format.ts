import type { MemoryType, SourceKind } from "../data/types";

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

export function relativeTime(iso: string, now: Date = new Date()): string {
  const diff = now.getTime() - new Date(iso).getTime();
  if (diff < MINUTE) return "just now";
  if (diff < HOUR) return plural(Math.floor(diff / MINUTE), "minute") + " ago";
  if (diff < DAY) return plural(Math.floor(diff / HOUR), "hour") + " ago";
  const days = Math.floor(diff / DAY);
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  return formatDate(iso);
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function plural(n: number, word: string, pluralWord = `${word}s`): string {
  return `${n} ${n === 1 ? word : pluralWord}`;
}

export function memoriesLabel(n: number): string {
  return plural(n, "memory", "memories");
}

const TYPE_LABELS: Record<MemoryType, string> = {
  person: "Person",
  project: "Project",
  fact: "Fact",
  preference: "Preference",
  event: "Event",
};

export function typeLabel(type: MemoryType): string {
  return TYPE_LABELS[type];
}

const KIND_LABELS: Record<SourceKind, string> = {
  conversation: "Conversation",
  document: "Document",
  note: "Note",
};

export function sourceKindLabel(kind: SourceKind): string {
  return KIND_LABELS[kind];
}
