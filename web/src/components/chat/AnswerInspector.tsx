import { useState } from "react";
import { Link } from "react-router-dom";
import type {
  AnswerEvidence,
  GenerationOverrides,
  GenerationPublic,
  Memory,
  RetrievalMode,
} from "../../data/types";
import { partitionPromptMemories } from "../../lib/answerEvidence";
import styles from "./chat.module.css";

export interface ReaskOptions {
  retrievalMode: RetrievalMode;
  generation: GenerationOverrides;
}

export function AnswerInspector({
  evidence,
  lookup,
  defaults,
  onReask,
  onApplyDefaults,
}: {
  evidence: AnswerEvidence;
  lookup: { memories: Map<string, Memory> };
  defaults?: GenerationPublic;
  onReask?: (question: string, options: ReaskOptions) => void;
  onApplyDefaults?: (options: ReaskOptions) => void;
}) {
  const diagnostics = evidence.diagnostics;
  const generation = evidence.generation;
  const hops = evidence.hops ?? [];
  const parts = partitionPromptMemories(evidence);
  const question = evidence.question ?? "";
  const mode = diagnostics?.mode ?? "baseline";
  const [openId, setOpenId] = useState<string | null>(null);
  const [formMode, setFormMode] = useState<RetrievalMode>(mode);
  const [model, setModel] = useState(generation?.model ?? defaults?.model ?? "");
  const [temperature, setTemperature] = useState(String(generation?.temperature ?? defaults?.temperature ?? 0));
  const [numPredict, setNumPredict] = useState(String(generation?.numPredict ?? defaults?.numPredict ?? 256));
  const [maxChars, setMaxChars] = useState(
    String(generation?.maxContextChars ?? defaults?.maxContextChars ?? ""),
  );

  const currentOptions = (): ReaskOptions => ({
    retrievalMode: formMode,
    generation: {
      model: model.trim() || undefined,
      temperature: Number(temperature),
      numPredict: Number(numPredict),
      maxContextChars: maxChars.trim() ? Number(maxChars) : undefined,
    },
  });

  return (
    <div className={styles.inspector}>
      <section>
        <p className={styles.evidenceHeading}>Question</p>
        <p className={styles.inspectorQuestion}>{question || "(not recorded)"}</p>
      </section>

      <section>
        <p className={styles.evidenceHeading}>Retrieval</p>
        <p className={styles.diagnosticsLine}>
          {modeLabel(mode)}
          {diagnostics
            ? ` · k=${diagnostics.k} · seeds ${diagnostics.seedCount} · expanded ${diagnostics.expandedCount} · sources ${diagnostics.selectedSourceCount}/${diagnostics.sourceCount} · budget ${diagnostics.usedChars}/${diagnostics.budgetChars} chars`
            : ""}
          {diagnostics?.queryCount ? ` · ${diagnostics.queryCount} queries` : ""}
          {diagnostics?.roundCount ? ` · ${diagnostics.roundCount} rounds` : ""}
          {diagnostics?.stopReason ? ` · stop ${diagnostics.stopReason}` : ""}
          {diagnostics?.latencyMs?.total != null ? ` · ${Math.round(diagnostics.latencyMs.total)} ms` : ""}
        </p>
      </section>

      {generation?.rawAnswer && generation.rawAnswer !== undefined && (
        <section>
          <p className={styles.evidenceHeading}>Citation check</p>
          {(generation.strippedCitations?.length ?? 0) > 0 && (
            <p className={styles.inspectorNote}>
              Removed unsupported cites:{" "}
              {generation.strippedCitations?.map((row) => `[${row.id}] ${row.reason}`).join(", ")}
            </p>
          )}
          {generation.rawAnswer !== undefined && (
            <p className={styles.inspectorNote}>Raw model text: {generation.rawAnswer}</p>
          )}
        </section>
      )}

      {evidence.queryPlan && (
        <section>
          <p className={styles.evidenceHeading}>Query plan</p>
          <p className={styles.inspectorNote}>
            {evidence.queryPlan.informationNeed || "(no information need)"} · {evidence.queryPlan.source} ·{" "}
            {evidence.queryPlan.status}
          </p>
          {evidence.queryPlan.entities.length > 0 && (
            <p className={styles.diagnosticsLine}>Entities: {evidence.queryPlan.entities.join(", ")}</p>
          )}
          {evidence.queryPlan.facets.length > 0 && (
            <p className={styles.diagnosticsLine}>Facets: {evidence.queryPlan.facets.join(", ")}</p>
          )}
          <ul className={styles.evidenceList}>
            {evidence.queryPlan.queries.map((query, index) => (
              <li key={`${index}:${query}`}>
                <code className={styles.messageId}>q{index}</code> {query}
              </li>
            ))}
          </ul>
        </section>
      )}

      {evidence.iterativeRounds && evidence.iterativeRounds.length > 0 && (
        <section>
          <p className={styles.evidenceHeading}>Iterative rounds</p>
          <ul className={styles.evidenceList}>
            {evidence.iterativeRounds.map((round) => (
              <li key={round.round}>
                <code className={styles.messageId}>r{round.round}</code>{" "}
                {round.evaluation?.status ?? "no evaluation"} · new {round.newMemoryIds.length} ·{" "}
                {round.queries.join(" | ")}
              </li>
            ))}
          </ul>
        </section>
      )}

      {evidence.retrievalComparison && (
        <section>
          <p className={styles.evidenceHeading}>Baseline vs fused</p>
          <p className={styles.inspectorNote}>
            Same question under single-query baseline and multi-query RRF. Rankings below are the selected top-k.
          </p>
          <p className={styles.diagnosticsLine}>
            Baseline: {(evidence.retrievalComparison.baseline?.rankings ?? [])
              .slice(0, 8)
              .map((row) => row.sourceMessageIds[0] ?? row.memoryId)
              .join(", ") || "(none)"}
          </p>
          <p className={styles.diagnosticsLine}>
            Fused: {(evidence.retrievalComparison.fused?.rankings ?? [])
              .slice(0, 8)
              .map((row) => row.sourceMessageIds[0] ?? row.memoryId)
              .join(", ") || "(none)"}
          </p>
        </section>
      )}

      <section>
        <p className={styles.evidenceHeading}>Retrieved memories</p>
        {!parts.tracked && (
          <p className={styles.inspectorNote}>
            This answer did not record which memories were passed to a model. Retrieved IDs are listed
            below without implying they influenced generation.
          </p>
        )}
        <MemoryList
          rows={parts.retrieved}
          lookup={lookup}
          expanded={new Set(evidence.expandedMemoryIds ?? [])}
          promptIds={new Set(parts.inPrompt.map((m) => m.id))}
        />
      </section>

      <section>
        <p className={styles.evidenceHeading}>
          {generation?.status === "ok" ? "Passed to the answer model" : "Prepared prompt memories"}
        </p>
        {generation?.status && generation.status !== "ok" && (
          <p className={styles.inspectorNote}>
            These IDs were placed in the prompt{generation.status === "error" ? " for the failed model call" : `, but generation ${generation.status.replace("_", " ")}`}.
            They did not influence a completed answer unless generation succeeded.
          </p>
        )}
        {parts.tracked ? (
          parts.inPrompt.length > 0 ? (
            <MemoryList
              rows={parts.inPrompt}
              lookup={lookup}
              expanded={new Set(evidence.expandedMemoryIds ?? [])}
              promptIds={new Set(parts.inPrompt.map((m) => m.id))}
            />
          ) : (
            <p className={styles.inspectorNote}>No memories were included in the generation prompt.</p>
          )
        ) : (
          <p className={styles.inspectorNote}>Not available for this answer.</p>
        )}
        {parts.retrievedOnly.length > 0 && parts.tracked && (
          <>
            <p className={styles.evidenceHeading}>Retrieved but not passed to the model</p>
            <MemoryList
              rows={parts.retrievedOnly}
              lookup={lookup}
              expanded={new Set(evidence.expandedMemoryIds ?? [])}
              promptIds={new Set()}
            />
          </>
        )}
      </section>

      <section>
        <p className={styles.evidenceHeading}>Source messages</p>
        <ul className={styles.evidenceList}>
          {(evidence.selectedMessages ?? []).map((msg) => {
            const excerpt = evidence.promptExcerpts?.find((item) => item.id === msg.id);
            const passed = Boolean(excerpt);
            return (
              <li key={msg.id}>
                <button
                  type="button"
                  className={styles.inspectorSource}
                  onClick={() => setOpenId((id) => (id === msg.id ? null : msg.id))}
                  aria-expanded={openId === msg.id}
                >
                  <code className={styles.messageId}>{msg.id}</code>
                  <span>
                    {openId === msg.id ? msg.content : preview(msg.content)}
                    {passed ? " · in prompt" : " · retrieved, not in prompt"}
                    {excerpt?.truncated ? " · truncated" : ""}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
        {(evidence.droppedMessageIds?.length ?? 0) > 0 && (
          <p className={styles.inspectorNote}>
            Dropped by budget or context limit: {evidence.droppedMessageIds!.join(", ")}
          </p>
        )}
      </section>

      {hops.length > 0 && (
        <section>
          <p className={styles.evidenceHeading}>B′ expansions</p>
          <ul className={styles.evidenceList}>
            {hops.map((hop) => (
              <li key={`${hop.pairId}:${hop.neighbourMemoryId}`} className={styles.hopItem}>
                <span className={styles.evidenceTag}>{hop.pairId}</span>
                <span>
                  seed {hop.seedMemoryId} → {hop.neighbourMemoryId} · {hop.relation ?? "LINK"} · hop{" "}
                  {hop.hopDepth ?? 1}
                  {hop.rationale ? ` — ${hop.rationale}` : ""}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <p className={styles.evidenceHeading}>Answer model</p>
        <p className={styles.diagnosticsLine}>
          {generationStatus(generation)}
          {generation?.truncated ? " · prompt truncated to the context budget" : ""}
          {generation?.promptChars != null ? ` · prompt ${generation.promptChars} chars` : ""}
        </p>
        {generation?.error && <p className={styles.inspectorError}>{generation.error}</p>}
        {generation?.status === "not_configured" && (
          <p className={styles.inspectorNote}>
            Set <code>CONTINUATOR_OLLAMA_MODEL</code> or <code>CONTINUATOR_ANSWER_MODEL</code> on the
            server (Ollama, no API key in the browser), or enter a local model below and ask again.
          </p>
        )}
      </section>

      {onReask && (
        <form
          className={styles.inspectorForm}
          onSubmit={(e) => {
            e.preventDefault();
            if (!question) return;
            onReask(question, currentOptions());
          }}
        >
          <p className={styles.evidenceHeading}>Ask this question again</p>
          <label className={styles.inspectorField}>
            Retrieval
            <select value={formMode} onChange={(e) => setFormMode(e.target.value as RetrievalMode)}>
              <option value="baseline">Baseline</option>
              <option value="bprime">B′ linked</option>
              <option value="multiquery">Multi-query</option>
              <option value="iterative">Iterative</option>
              <option value="lexical">Lexical</option>
            </select>
          </label>
          <label className={styles.inspectorField}>
            Model
            <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="qwen2.5:1.5b" />
          </label>
          <div className={styles.inspectorRow}>
            <label className={styles.inspectorField}>
              Temperature
              <input value={temperature} onChange={(e) => setTemperature(e.target.value)} />
            </label>
            <label className={styles.inspectorField}>
              Max tokens
              <input value={numPredict} onChange={(e) => setNumPredict(e.target.value)} />
            </label>
            <label className={styles.inspectorField}>
              Memory context chars
              <input value={maxChars} onChange={(e) => setMaxChars(e.target.value)} />
            </label>
          </div>
          <div className={styles.inspectorActions}>
            <button type="submit" className={styles.inspectorPrimary} disabled={!question}>
              Ask again
            </button>
            {onApplyDefaults && (
              <button type="button" className={styles.inspectorSecondary} onClick={() => onApplyDefaults(currentOptions())}>
                Use for next questions
              </button>
            )}
          </div>
        </form>
      )}
    </div>
  );
}

function MemoryList({
  rows,
  lookup,
  expanded,
  promptIds,
}: {
  rows: { id: string; statement: string; rank?: number | null; score?: number | null; origin?: string; sourceMessageIds: string[] }[];
  lookup: { memories: Map<string, Memory> };
  expanded: Set<string>;
  promptIds: Set<string>;
}) {
  return (
    <ul className={styles.evidenceList}>
      {rows.map((row) => {
        const memory = lookup.memories.get(row.id);
        const statement = memory?.statement ?? row.statement;
        const score =
          row.score != null
            ? `score ${row.score.toFixed(3)}`
            : row.rank != null
              ? `rank ${row.rank}`
              : null;
        return (
          <li key={row.id}>
            <Link to={`/memories/${row.id}`} className={styles.evidenceItem}>
              <span>
                <code className={styles.messageId}>{row.id}</code> {preview(statement)}
                {score ? ` · ${score}` : ""}
                {row.origin === "bprime" || expanded.has(row.id) ? " · B′" : ""}
                {promptIds.has(row.id) ? " · in prompt" : ""}
                {row.sourceMessageIds.length > 0 ? ` · ${row.sourceMessageIds.join(", ")}` : ""}
              </span>
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

function preview(text: string, limit = 140): string {
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length <= limit ? compact : `${compact.slice(0, limit - 1)}…`;
}

function modeLabel(mode: string): string {
  if (mode === "bprime") return "B′ linked";
  if (mode === "multiquery") return "Multi-query";
  if (mode === "iterative") return "Iterative";
  if (mode === "lexical" || mode === "hybrid") return "Lexical";
  return "Baseline";
}

function generationStatus(generation?: GenerationPublic): string {
  if (!generation) return "Not recorded for this answer.";
  const provider = generation.provider ?? "unspecified provider";
  const model = generation.model ?? "no model";
  const status = generation.status ?? "unknown";
  const bits = [`${provider} / ${model}`, status];
  if (generation.temperature != null) bits.push(`temperature ${generation.temperature}`);
  if (generation.numPredict != null) bits.push(`max tokens ${generation.numPredict}`);
  if (generation.maxContextChars != null) bits.push(`context ${generation.maxContextChars} chars`);
  return bits.join(" · ");
} 