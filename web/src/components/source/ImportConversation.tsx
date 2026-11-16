import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { isDemoMode } from "../../data";
import { useService } from "../../data/ServiceProvider";
import { DEFAULT_EXTRACT_MODE, type ExtractMode, type ImportResult } from "../../data/types";
import {
  importPersistenceNote,
  importStatistics,
  importSucceeded,
} from "../../lib/importSummary";
import { useShell } from "../shell/ShellContext";
import { Icon } from "../ui/Icon";
import styles from "./source.module.css";

const MAX_BYTES = 8_000_000;

const EXTRACT_OPTIONS: { value: ExtractMode; label: string; status: string }[] = [
  {
    value: "messages",
    label: "One memory per message (no model)",
    status: "Importing messages and creating one memory per message…",
  },
  {
    value: "memories",
    label: "Memory extraction: durable memories (Ollama, slow)",
    status: "Importing and running Ollama memory extraction. This can take several minutes…",
  },
  {
    value: "v2",
    label: "V2 atomizer: every proposition (Ollama, slow)",
    status: "Importing and running the V2 Ollama atomizer. This can take several minutes…",
  },
];

export function ImportConversation({ compact = false }: { compact?: boolean }) {
  const service = useService();
  const { onImported } = useShell();
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [extract, setExtract] = useState<ExtractMode>(DEFAULT_EXTRACT_MODE);

  const pick = () => {
    setError(null);
    inputRef.current?.click();
  };

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    if (inputRef.current) inputRef.current.value = "";
    if (isDemoMode()) {
      setError("Connect the Continuator API (VITE_CONTINUATOR_API_URL) to import a conversation.");
      return;
    }
    if (file.size > MAX_BYTES) {
      setError("File is larger than 8 MB.");
      return;
    }
    setBusy(true);
    setError(null);
    setStatus(`Reading ${file.name}…`);
    try {
      const text = await file.text();
      setStatus(EXTRACT_OPTIONS.find((o) => o.value === extract)?.status ?? null);
      const result = await service.importConversation(file.name, text, extract);
      setStatus(null);
      onImported(result);
    } catch (err) {
      setStatus(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={compact ? styles.importCompact : styles.importPanel}>
      <input
        ref={inputRef}
        type="file"
        accept=".txt,.json,text/plain,application/json"
        className="visually-hidden"
        onChange={(e) => void onFile(e.target.files?.[0])}
      />
      <button type="button" className={styles.importButton} onClick={pick} disabled={busy}>
        <Icon name="plus" size={15} />
        {busy ? "Importing…" : "Import conversation"}
      </button>
      {!compact && (
        <label className={styles.importOption}>
          Extraction
          <select
            className={styles.importSelect}
            value={extract}
            disabled={busy}
            onChange={(e) => setExtract(e.target.value as ExtractMode)}
          >
            {EXTRACT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
      )}
      {status && <p className={styles.importStatus}>{status}</p>}
      {error && (
        <p className={styles.importError} role="alert">
          Import failed. The previous workspace was not replaced. {error}
        </p>
      )}
    </div>
  );
}

/**
 * Post-import success card. Shows only statistics returned by `POST /import`.
 * Does not invent pipeline stage progress.
 */
export function ImportResultNotice({ result }: { result: ImportResult }) {
  if (!importSucceeded(result)) return null;

  const { report } = result;
  const stats = importStatistics(result);
  const persistence = importPersistenceNote(result);

  return (
    <div className={styles.importReady} role="status" data-testid="import-ready">
      <header className={styles.importReadyHeader}>
        <p className={styles.importReadyEyebrow}>Import complete</p>
        <h3 className={styles.importReadyTitle}>Memory ready</h3>
      </header>

      <p className={styles.importReadyLead}>
        Continuator extracted and consolidated information from your conversation into persistent
        memory with source provenance.
      </p>

      <p className={styles.importReadyFlow} aria-label="Product transformation">
        <span>Conversation</span>
        <Icon name="arrowRight" size={12} />
        <span>Persistent memory</span>
      </p>

      {stats.length > 0 && (
        <dl className={styles.importReadyStats} data-testid="import-stats">
          {stats.map((stat) => (
            <div key={stat.label}>
              <dt>{stat.label}</dt>
              <dd>{stat.value}</dd>
            </div>
          ))}
        </dl>
      )}

      <p className={styles.importReadyPersist}>{persistence}</p>

      {report.failedWindows > 0 && (
        <p className={styles.importWarning} role="status">
          {report.failedWindows} of {report.windows} extraction windows failed, so some memories may
          be missing.
        </p>
      )}
      {result.note && <p className={styles.importReadyNote}>{result.note}</p>}

      <div className={styles.importReadyActions}>
        <Link to="/memories" className={styles.importReadyPrimary}>
          Explore Memories
        </Link>
        <Link to="/" className={styles.importReadySecondary}>
          Ask &amp; Answer
        </Link>
        <Link to="/?intent=context" className={styles.importReadySecondary}>
          Retrieve Context
        </Link>
      </div>
    </div>
  );
}
