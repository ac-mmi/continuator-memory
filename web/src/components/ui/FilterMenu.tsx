import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { Icon } from "./Icon";
import styles from "./ui.module.css";

export interface FilterOption<T extends string> {
  value: T;
  label: string;
  icon?: ReactNode;
}

/** Multi-select dropdown. An empty selection means "all". */
export function FilterMenu<T extends string>({
  label,
  options,
  selected,
  onChange,
}: {
  label: string;
  options: FilterOption<T>[];
  selected: T[];
  onChange: (next: T[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const menuId = useId();

  useEffect(() => {
    if (!open) return;
    const onPointer = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const toggle = (value: T) =>
    onChange(selected.includes(value) ? selected.filter((v) => v !== value) : [...selected, value]);

  const summary =
    selected.length === 0 || selected.length === options.length
      ? "All"
      : selected.length === 1
        ? options.find((o) => o.value === selected[0])?.label
        : `${selected.length} selected`;

  return (
    <div className={styles.filter} ref={ref}>
      <button
        type="button"
        className={styles.filterButton}
        aria-expanded={open}
        aria-controls={menuId}
        onClick={() => setOpen((o) => !o)}
      >
        <Icon name="filter" size={14} />
        <span className={styles.filterLabel}>{label}</span>
        <span className={styles.filterSummary}>{summary}</span>
        <Icon name="chevronDown" size={14} />
      </button>
      {open && (
        <div className={styles.filterMenu} id={menuId} role="menu">
          {options.map((o) => {
            const checked = selected.includes(o.value);
            return (
              <button
                key={o.value}
                type="button"
                role="menuitemcheckbox"
                aria-checked={checked}
                className={styles.filterItem}
                onClick={() => toggle(o.value)}
              >
                <span className={checked ? styles.checkboxOn : styles.checkbox}>
                  {checked && <Icon name="check" size={11} strokeWidth={2} />}
                </span>
                {o.icon}
                {o.label}
              </button>
            );
          })}
          {selected.length > 0 && (
            <button type="button" className={styles.filterReset} onClick={() => onChange([])}>
              Clear filter
            </button>
          )}
        </div>
      )}
    </div>
  );
}
