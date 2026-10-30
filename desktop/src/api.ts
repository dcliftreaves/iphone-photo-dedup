export type Session = {
  id: string;
  name: string;
  created_at: string;
  roots: string[];
  state: string;
  job_id: string;
  groups: number;
  reviewed: number;
  counts: Record<string, number>;
  warnings?: number;
  last_export?: string;
  last_export_mode?: string;
  last_export_template?: string;
  error?: string;
};
export type FileRecord = {
  id: string;
  path: string;
  status: string;
  confidence: string;
  reason: string;
  size: number;
  kind: string;
  extension: string;
  width?: number;
  height?: number;
  captured_at?: string;
  model?: string;
  sha256?: string;
  warnings?: string[];
  related_files?: { path: string; relationship: string }[];
  metadata?: Record<string, unknown>;
};
export type GroupSummary = {
  id: string;
  kind: string;
  count: number;
  title: string;
  preview_id: string;
  size: number;
  decision: unknown;
  has_image: boolean;
  media_kind: string;
};
export type Group = {
  id: string;
  records: FileRecord[];
  selected: string[];
  recommendation: string;
  decision: { action: string; selected: string[] } | null;
};
export type Job = {
  id: string;
  session_id: string;
  kind: string;
  state: string;
  done: number;
  total: number;
  phase: string;
  cached: number;
  error?: string;
  result?: {
    destination: string;
    files: number;
    mode: string;
    omitted: number;
    ignored: number;
    errors: number;
    warnings: number;
  };
};
export type Plan = {
  files: number;
  bytes: number;
  omitted: number;
  unresolved: number;
  ignored: number;
  errors: number;
  warnings: number;
  free_bytes?: number;
  records: { source: string; destination: string }[];
};
export const native = "__TAURI_INTERNALS__" in window;
let token =
  window.location.hash.slice(1) || sessionStorage.getItem("dedup-token") || "";
if (token) {
  sessionStorage.setItem("dedup-token", token);
  history.replaceState(null, "", location.pathname);
}
export async function rpc<T>(
  method: string,
  params: Record<string, unknown> = {},
): Promise<T> {
  if (native) {
    const { invoke } = await import("@tauri-apps/api/core");
    return invoke<T>("rpc", { method, params });
  }
  const response = await fetch("/api/rpc", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ version: 1, method, params }),
  });
  const body = await response.json();
  if (body.error) throw new Error(body.error);
  return body.result;
}
export async function pickDirectory(): Promise<string | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  return (await open({ directory: true, multiple: false })) as string | null;
}
export async function reveal(path: string) {
  if (native) {
    const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
    await revealItemInDir(path);
  } else await navigator.clipboard.writeText(path);
}
export function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  return `${(bytes / 1024 ** i).toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
}
export const basename = (path: string) => path.split(/[\\/]/).pop() || path;
export const kindLabel: Record<string, string> = {
  exact: "Identical files",
  visual: "Similar photos",
  metadata: "Possible matches",
  unique: "Unique files",
};
