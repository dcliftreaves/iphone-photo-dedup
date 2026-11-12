import { useState } from "react";
import { CheckCircle2, CircleHelp, XCircle } from "lucide-react";
import { native, pickDirectory, rpc } from "./api";

type Volume = {
  path: string;
  name: string;
  filesystem: string;
  supported: boolean | null;
  read_only: boolean | null;
  evidence: string;
};
export function DiskCheck() {
  const [volumes, setVolumes] = useState<Volume[] | null>(null);
  const [busy, setBusy] = useState(false),
    [message, setMessage] = useState("");
  const [folder, setFolder] = useState("");
  async function list() {
    setBusy(true);
    setMessage("");
    try {
      const result = await rpc<{ volumes: Volume[]; message: string }>(
        "local_volumes",
      );
      setVolumes(result.volumes);
      setMessage(result.message);
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function probe() {
    setBusy(true);
    setMessage("");
    try {
      const path = native ? await pickDirectory() : folder.trim();
      if (path) {
        const result = await rpc<{ message: string }>("test_folder", { path });
        setMessage(result.message);
      }
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="disk-check">
      <summary>Which disks support hard links?</summary>
      <p>
        APFS and Mac OS Extended (HFS+) support file hard links, as does NTFS on
        Windows. exFAT and FAT32 do not. Imports and output must share one
        volume. Network drives and third-party drivers need a folder test.
      </p>
      <button
        className="button secondary"
        disabled={busy}
        onClick={() => void list()}
      >
        {busy ? "Checking…" : "Check local disks"}
      </button>
      {volumes && (
        <ul className="volume-list">
          {volumes.map((v) => (
            <li key={v.path}>
              {v.supported === true && !v.read_only ? (
                <CheckCircle2 size={18} aria-hidden="true" />
              ) : v.supported === false || v.read_only ? (
                <XCircle size={18} aria-hidden="true" />
              ) : (
                <CircleHelp size={18} aria-hidden="true" />
              )}
              <div>
                <strong>
                  {v.name} · {v.filesystem || "Unknown filesystem"}
                </strong>
                <small>{v.path}</small>
                <span>
                  {v.read_only
                    ? "Read only — choose a writable volume"
                    : v.supported === true
                      ? "Filesystem supports hard links"
                      : v.supported === false
                        ? "Filesystem does not support hard links"
                        : "Support unknown — test a folder"}
                </span>
                <small>{v.evidence}</small>
              </div>
            </li>
          ))}
        </ul>
      )}
      {volumes?.length === 0 && (
        <p>
          No local mounted disks were reported. You can test a folder directly.
        </p>
      )}
      <p>
        Optional folder test: creates two names for one tiny temporary file,
        then removes both. macOS or Windows may ask for folder access. Access
        errors mean “unverified,” not “unsupported.”
      </p>
      {!native && (
        <label>
          Folder to test
          <input
            value={folder}
            onChange={(e) => setFolder(e.target.value)}
            placeholder="Absolute path to an existing folder"
          />
        </label>
      )}
      <button
        className="button secondary"
        disabled={busy || (!native && !folder.trim())}
        onClick={() => void probe()}
      >
        {native
          ? "Choose a folder and test hard links"
          : "Test hard links in this folder"}
      </button>
      {message && <p role="status">{message}</p>}
    </details>
  );
}
