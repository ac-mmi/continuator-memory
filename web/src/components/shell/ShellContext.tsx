import { createContext, useContext } from "react";
import type { ImportResult } from "../../data/types";

export interface ShellContextValue {
  openNav: () => void;
  /** Bumped whenever the conversation list should be reloaded. */
  conversationsVersion: number;
  refreshConversations: () => void;
  /** Result of the latest import in this session, shown on the Sources page. */
  lastImport: ImportResult | null;
  /** Records the result and reloads every page, since an import replaces the workspace. */
  onImported: (result: ImportResult) => void;
}

export const ShellContext = createContext<ShellContextValue>({
  openNav: () => {},
  conversationsVersion: 0,
  refreshConversations: () => {},
  lastImport: null,
  onImported: () => {},
});

export function useShell(): ShellContextValue {
  return useContext(ShellContext);
}
