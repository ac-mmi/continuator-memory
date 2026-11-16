import { Icon } from "./Icon";
import styles from "./ui.module.css";

export function SearchInput({
  value,
  onChange,
  placeholder = "Search",
  label = "Search",
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label?: string;
}) {
  return (
    <label className={styles.search}>
      <span className="visually-hidden">{label}</span>
      <Icon name="search" size={15} className={styles.searchIcon} />
      <input
        type="search"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className={styles.searchInput}
      />
      {value && (
        <button type="button" className={styles.searchClear} onClick={() => onChange("")} aria-label="Clear search">
          <Icon name="close" size={13} />
        </button>
      )}
    </label>
  );
}
