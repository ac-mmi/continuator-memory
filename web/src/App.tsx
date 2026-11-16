import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { AppShell } from "./components/shell/AppShell";
import { ServiceProvider } from "./data/ServiceProvider";
import { GraphPage } from "./pages/GraphPage";
import { HomePage, NotFoundPage } from "./pages/HomePage";
import { MemoriesPage } from "./pages/MemoriesPage";
import { MemoryDetailPage } from "./pages/MemoryDetailPage";
import { SettingsPage } from "./pages/SettingsPage";
import { SourceDetailPage } from "./pages/SourceDetailPage";
import { SourcesPage } from "./pages/SourcesPage";

const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: "/", element: <HomePage /> },
      { path: "/c/:conversationId", element: <HomePage /> },
      { path: "/memories", element: <MemoriesPage /> },
      { path: "/memories/:memoryId", element: <MemoryDetailPage /> },
      { path: "/graph", element: <GraphPage /> },
      { path: "/sources", element: <SourcesPage /> },
      { path: "/sources/:sourceId", element: <SourceDetailPage /> },
      { path: "/settings", element: <SettingsPage /> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);

export function App() {
  return (
    <ServiceProvider>
      <RouterProvider router={router} />
    </ServiceProvider>
  );
}
