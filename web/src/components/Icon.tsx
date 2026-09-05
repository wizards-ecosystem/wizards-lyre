import { IconName } from "../types";

export function Icon({ name }: { name: IconName }) {
  const paths: Record<IconName, JSX.Element> = {
    add: <path d="M12 5v14M5 12h14" />,
    back: <path d="m15 18-6-6 6-6" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    delete: <path d="M5 7h14M9 7V4h6v3m2 0-1 13H8L7 7m3 4v5m4-5v5" />,
    library: <path d="M5 4h5v16H5zM14 4h5v16h-5z" />,
    pause: <path d="M8 6h3v12H8zM14 6h3v12h-3z" />,
    play: <path d="m8 5 11 7-11 7z" />,
    search: <path d="m20 20-4.5-4.5m2.5-5A7.5 7.5 0 1 1 3 10.5a7.5 7.5 0 0 1 15 0Z" />,
    settings: <path d="M4 7h10m4 0h2M4 17h2m4 0h10M14 4v6M6 14v6" />,
    spark: <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />,
    star: <path d="m12 3 2.7 5.5 6.1.9-4.4 4.3 1 6.1-5.4-2.9-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z" />,
    wave: <path d="M3 12h3l2-6 3 12 3-9 2 6h5" />,
  };
  return (
    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      {paths[name]}
    </svg>
  );
}
