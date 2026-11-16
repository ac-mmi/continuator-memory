import { DemoService } from "./demo/demoService";
import { HttpService } from "./http/httpService";
import type { ContinuatorService } from "./service";

export type { ContinuatorService } from "./service";
export * from "./types";

/** Set `VITE_CONTINUATOR_API_URL` to use a real backend instead of demo data. */
export function createService(): ContinuatorService {
  const apiUrl = import.meta.env.VITE_CONTINUATOR_API_URL as string | undefined;
  return apiUrl ? new HttpService(apiUrl) : new DemoService();
}

export function isDemoMode(): boolean {
  return !import.meta.env.VITE_CONTINUATOR_API_URL;
}
