import type { GenerationOverrides, GenerationPublic } from "../data/types";

/**
 * Request overrides from the settings `/health` reports. The API reports unset
 * values as null but rejects null in a request, so they are left out.
 */
export function generationDefaults(g: GenerationPublic): GenerationOverrides {
  return {
    model: g.model ?? undefined,
    temperature: g.temperature,
    numPredict: g.numPredict,
    maxContextChars: g.maxContextChars ?? undefined,
  };
}
