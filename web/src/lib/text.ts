const STOPWORDS = new Set([
  "a", "an", "and", "are", "about", "do", "does", "for", "from", "how", "i", "in", "is", "it",
  "me", "my", "of", "on", "or", "the", "to", "was", "what", "when", "where", "which", "who",
  "with", "you", "your", "remember", "know", "tell",
]);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .split(/[^a-z0-9.:]+/)
    .map((t) => t.replace(/^[.:]+|[.:]+$/g, ""))
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

/** Case-insensitive substring match across several fields. */
export function matchesText(query: string, ...fields: string[]): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return fields.some((f) => f.toLowerCase().includes(q));
}
