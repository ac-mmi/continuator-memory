import { createContext, useContext, useState, type ReactNode } from "react";
import { createService, type ContinuatorService } from "./index";

const ServiceContext = createContext<ContinuatorService | null>(null);

export function ServiceProvider({
  service,
  children,
}: {
  service?: ContinuatorService;
  children: ReactNode;
}) {
  const [instance] = useState(() => service ?? createService());
  return <ServiceContext.Provider value={instance}>{children}</ServiceContext.Provider>;
}

export function useService(): ContinuatorService {
  const service = useContext(ServiceContext);
  if (!service) throw new Error("useService must be used inside <ServiceProvider>");
  return service;
}
