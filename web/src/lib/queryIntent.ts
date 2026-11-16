import type { QueryIntent } from "../data/types";

/** Which product endpoint the Home composer should call for a given intent. */
export function endpointForQueryIntent(intent: QueryIntent): "/ask" | "/context" {
  return intent === "context" ? "/context" : "/ask";
}
