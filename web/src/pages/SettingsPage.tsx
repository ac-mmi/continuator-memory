import { TopBar } from "../components/shell/TopBar";
import { ErrorNotice } from "../components/ui/EmptyState";
import { isDemoMode } from "../data";
import { useService } from "../data/ServiceProvider";
import { useAsync } from "../lib/useAsync";
import styles from "./pages.module.css";

export function SettingsPage() {
  const service = useService();
  const live = !isDemoMode();
  const health = useAsync(() => service.health(), [service]);

  return (
    <>
      <TopBar title="Settings" />
      <div className={styles.scroll}>
        <div className={styles.pageNarrow}>
          <div className={styles.pageIntro}>
            <h2 className={styles.pageTitle}>Workspace settings</h2>
            <p className={styles.pageText}>
              {live
                ? "This interface is reading the Continuator HTTP API (V2 product path). Retrieval modes are baseline, lexical, B′ linked, multi-query and iterative. Answers are generated from retrieved source evidence when a local Ollama model is configured."
                : "Configuration will be available once the backend is connected."}
            </p>
          </div>
          {health.status === "error" && <ErrorNotice error={health.error} onRetry={health.reload} />}

          <section className={styles.settingsSection}>
            <div>
              <h3 className={styles.settingsTitle}>Connection</h3>
              <p className={styles.muted}>The data source this interface reads from.</p>
            </div>
            <dl className={styles.settingsRows}>
              <div>
                <dt>Mode</dt>
                <dd>{live ? "Continuator API" : "Demo data (local, read-only)"}</dd>
              </div>
              {live && (
                <div>
                  <dt>API URL</dt>
                  <dd>{String(import.meta.env.VITE_CONTINUATOR_API_URL)}</dd>
                </div>
              )}
            </dl>
          </section>

          <section className={styles.settingsSection}>
            <div>
              <h3 className={styles.settingsTitle}>Memory</h3>
              <p className={styles.muted}>How memories are formed, consolidated, and kept as history.</p>
            </div>
            <dl className={styles.settingsRows}>
              <div>
                <dt>Extraction</dt>
                <dd>
                  {health.data?.extraction ?? (live ? "—" : "Automatic, per message")}
                </dd>
              </div>
              <div>
                <dt>Changed facts</dt>
                <dd>Superseded and kept as history</dd>
              </div>
            </dl>
          </section>

          <section className={styles.settingsSection}>
            <div>
              <h3 className={styles.settingsTitle}>Retrieval</h3>
              <p className={styles.muted}>Baseline top-k and optional B′ linked expansion.</p>
            </div>
            <dl className={styles.settingsRows}>
              <div>
                <dt>Workspace</dt>
                <dd>{health.data?.sampleId ?? (live ? "…" : "demo")}</dd>
              </div>
              {health.data?.importedFilename && (
                <div>
                  <dt>Imported file</dt>
                  <dd>{health.data.importedFilename}</dd>
                </div>
              )}
              <div>
                <dt>Memories</dt>
                <dd>{health.data?.nMemories ?? "—"}</dd>
              </div>
              <div>
                <dt>Cutoff k</dt>
                <dd>{health.data?.k ?? "—"}</dd>
              </div>
              <div>
                <dt>Answer generation</dt>
                <dd>{health.data?.answerGeneration ?? (live ? "not_configured" : "demo")}</dd>
              </div>
              <div>
                <dt>Answer model</dt>
                <dd>
                  {health.data?.generation?.model
                    ? `${health.data.generation.provider ?? "ollama"} / ${health.data.generation.model}`
                    : live
                      ? "Not configured. Set CONTINUATOR_OLLAMA_MODEL or CONTINUATOR_ANSWER_MODEL on the server."
                      : "demo"}
                </dd>
              </div>
            </dl>
          </section>

          <section className={styles.settingsSection}>
            <div>
              <h3 className={styles.settingsTitle}>Data</h3>
              <p className={styles.muted}>Where Continuator stores memories and sources.</p>
            </div>
            <dl className={styles.settingsRows}>
              <div>
                <dt>Storage</dt>
                <dd>{live ? "Sample or SQLite on the server; imports go to CONTINUATOR_IMPORT_DB" : "Not configured"}</dd>
              </div>
            </dl>
          </section>
        </div>
      </div>
    </>
  );
}
