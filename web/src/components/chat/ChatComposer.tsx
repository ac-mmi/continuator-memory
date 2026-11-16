import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import type { QueryIntent, RetrievalMode } from "../../data/types";
import { Icon } from "../ui/Icon";
import styles from "./chat.module.css";

export function ChatComposer({
  onSubmit,
  disabled = false,
  autoFocus = false,
  placeholder = "Ask anything about your memory...",
  retrievalMode,
  onRetrievalModeChange,
  queryIntent,
  onQueryIntentChange,
}: {
  onSubmit: (text: string) => void;
  disabled?: boolean;
  autoFocus?: boolean;
  placeholder?: string;
  retrievalMode?: RetrievalMode;
  onRetrievalModeChange?: (mode: RetrievalMode) => void;
  queryIntent?: QueryIntent;
  onQueryIntentChange?: (intent: QueryIntent) => void;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const text = value.trim();
    if (!text || disabled) return;
    onSubmit(text);
    setValue("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const canSend = value.trim().length > 0 && !disabled;
  const showMode = retrievalMode !== undefined && onRetrievalModeChange !== undefined;
  const showIntent = queryIntent !== undefined && onQueryIntentChange !== undefined;
  const contextMode = queryIntent === "context";

  return (
    <div className={styles.composerStack}>
      {showIntent && (
        <>
          <div className={styles.intentToggle} role="group" aria-label="Query intent">
            <button
              type="button"
              className={queryIntent === "ask" ? styles.modeActive : styles.modeButton}
              onClick={() => onQueryIntentChange?.("ask")}
              disabled={disabled}
            >
              Ask &amp; Answer
            </button>
            <button
              type="button"
              className={contextMode ? styles.modeActive : styles.modeButton}
              onClick={() => onQueryIntentChange?.("context")}
              disabled={disabled}
            >
              Retrieve Context
            </button>
          </div>
          <p className={styles.intentHint}>
            {contextMode
              ? "Context → Continuator returns grounded memory and evidence for any LLM."
              : "Ask → Continuator retrieves memory and can generate an answer."}
          </p>
        </>
      )}
      <form className={styles.composer} onSubmit={submit}>
        <label className="visually-hidden" htmlFor="composer-input">
          {contextMode ? "Retrieve context from Continuator" : "Ask Continuator"}
        </label>
        <textarea
          id="composer-input"
          ref={ref}
          rows={1}
          value={value}
          placeholder={
            contextMode ? "What context should Continuator retrieve?" : placeholder
          }
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          className={styles.composerInput}
        />
        <div className={styles.composerFooter}>
          {showMode ? (
            <div className={styles.modeToggle} role="group" aria-label="Retrieval mode">
              <button
                type="button"
                className={retrievalMode === "baseline" ? styles.modeActive : styles.modeButton}
                onClick={() => onRetrievalModeChange?.("baseline")}
                disabled={disabled}
              >
                Baseline
              </button>
              <button
                type="button"
                className={retrievalMode === "bprime" ? styles.modeActive : styles.modeButton}
                onClick={() => onRetrievalModeChange?.("bprime")}
                disabled={disabled}
              >
                B′ linked
              </button>
              <button
                type="button"
                className={retrievalMode === "multiquery" ? styles.modeActive : styles.modeButton}
                onClick={() => onRetrievalModeChange?.("multiquery")}
                disabled={disabled}
              >
                Multi-query
              </button>
              <button
                type="button"
                className={retrievalMode === "iterative" ? styles.modeActive : styles.modeButton}
                onClick={() => onRetrievalModeChange?.("iterative")}
                disabled={disabled}
              >
                Iterative
              </button>
              <button
                type="button"
                className={
                  retrievalMode === "lexical" || retrievalMode === "hybrid"
                    ? styles.modeActive
                    : styles.modeButton
                }
                onClick={() => onRetrievalModeChange?.("lexical")}
                disabled={disabled}
              >
                Lexical
              </button>
            </div>
          ) : (
            <span className={styles.composerHint}>
              <Icon name="link" size={13} />
              {contextMode
                ? "Returns a Context Pack you can paste into any LLM"
                : "Answers cite the memories and sources they use"}
            </span>
          )}
          <button type="submit" className={styles.sendButton} disabled={!canSend} aria-label="Send">
            <Icon name="arrowUp" size={16} strokeWidth={1.8} />
          </button>
        </div>
      </form>
    </div>
  );
}
