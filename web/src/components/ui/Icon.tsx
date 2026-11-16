const PATHS = {
  chat: "M4 5.5A1.5 1.5 0 0 1 5.5 4h9A1.5 1.5 0 0 1 16 5.5v6a1.5 1.5 0 0 1-1.5 1.5H9l-3.5 3v-3h0A1.5 1.5 0 0 1 4 11.5z",
  memory: "M4 4.5h12v3H4zM4 9.5h12v3H4zM4 14.5h7v1.5H4z",
  graph: "M6 6m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0M14 5h3v3h-3zM12.5 14.5m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0M7.6 7.3l3.6 5.6M8 6h6M15.5 8l-2.2 4.6",
  sources: "M5 3.5h7l3 3v10H5zM12 3.5v3h3M7.5 10h5M7.5 13h5",
  settings: "M10 7.25a2.75 2.75 0 1 0 0 5.5a2.75 2.75 0 0 0 0-5.5zM10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2M4.7 4.7l1.4 1.4M13.9 13.9l1.4 1.4M4.7 15.3l1.4-1.4M13.9 6.1l1.4-1.4",
  sliders: "M4 6h12M7 4v4M4 14h12M13 12v4",
  search: "M9 4a5 5 0 1 0 0 10a5 5 0 0 0 0-10zM12.6 12.6L16 16",
  plus: "M10 4.5v11M4.5 10h11",
  arrowUp: "M10 15.5v-11M5.5 9L10 4.5L14.5 9",
  arrowRight: "M4.5 10h11M11 5.5l4.5 4.5l-4.5 4.5",
  arrowLeft: "M15.5 10h-11M9 5.5L4.5 10L9 14.5",
  menu: "M3.5 6h13M3.5 10h13M3.5 14h13",
  close: "M5.5 5.5l9 9M14.5 5.5l-9 9",
  chevronDown: "M6 8l4 4l4-4",
  chevronRight: "M8 6l4 4l-4 4",
  fit: "M4 8V4h4M12 4h4v4M16 12v4h-4M8 16H4v-4",
  zoomIn: "M9 4a5 5 0 1 0 0 10a5 5 0 0 0 0-10zM12.6 12.6L16 16M9 6.8v4.4M6.8 9h4.4",
  zoomOut: "M9 4a5 5 0 1 0 0 10a5 5 0 0 0 0-10zM12.6 12.6L16 16M6.8 9h4.4",
  filter: "M3.5 5h13M6 10h8M8.5 15h3",
  clock: "M10 3.5a6.5 6.5 0 1 0 0 13a6.5 6.5 0 0 0 0-13zM10 6.5V10l2.5 1.5",
  link: "M8.5 11.5l3-3M7 9.5l-1.5 1.5a2.5 2.5 0 0 0 3.5 3.5L10.5 13M13 10.5l1.5-1.5A2.5 2.5 0 0 0 11 5.5L9.5 7",
  check: "M5 10.5l3 3l7-7",
  history: "M4 10a6 6 0 1 0 1.8-4.3M4 4v3.5h3.5M10 7v3.2l2.2 1.3",
  quote: "M5 8.5h3v3.5H5zM5 8.5c0-2 1-3 2.5-3.5M11.5 8.5h3v3.5h-3zM11.5 8.5c0-2 1-3 2.5-3.5",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({
  name,
  size = 16,
  className,
  strokeWidth = 1.5,
}: {
  name: IconName;
  size?: number;
  className?: string;
  strokeWidth?: number;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
