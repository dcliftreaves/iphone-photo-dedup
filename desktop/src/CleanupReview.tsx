import { useEffect, useState } from "react";
import { FolderOpen, ArrowLeft, ArrowRight } from "lucide-react";
import { basename, formatBytes, native, reveal, rpc } from "./api";
type Cleanup = {
  roots: { path: string; files: number; at_risk: number }[];
  items: { path: string; size: number; reason: string; status: string }[];
  total: number;
  retained: number;
  identical_retained: number;
  message: string;
};
export function CleanupReview({ sessionId }: { sessionId: string }) {
  const [data, setData] = useState<Cleanup | null>(null),
    [error, setError] = useState(""),
    [offset, setOffset] = useState(0),
    [notice, setNotice] = useState("");
  useEffect(() => {
    let live = true;
    setData(null);
    rpc<Cleanup>("cleanup_review", { session_id: sessionId, offset, limit: 20 })
      .then((d) => {
        if (live) setData(d);
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, [sessionId, offset]);
  async function show(path: string) {
    try {
      await reveal(path);
      if (!native) setNotice("Path copied to clipboard.");
    } catch (e) {
      setNotice((e as Error).message);
    }
  }
  return (
    <section className="cleanup-review" aria-label="Source cleanup review">
      <h3>Before you delete the old folders</h3>
      <p>
        We don’t delete anything. Use this review to decide what to do with your
        imports yourself.
      </p>
      {error && <p role="alert">{error}</p>}
      {!data && !error && <p role="status">Loading saved export evidence…</p>}
      {data && (
        <>
          <p>
            <strong>{data.retained} files included</strong> ·{" "}
            {data.identical_retained}{" "}
            {data.identical_retained === 1
              ? "omitted copy has"
              : "omitted copies have"}{" "}
            identical bytes in the library.
          </p>
          <h4>Your source folders</h4>
          <ul className="cleanup-roots">
            {data.roots.map((root) => (
              <li key={root.path}>
                <span>
                  <strong>{basename(root.path)}</strong>
                  <small>{root.path}</small>
                  <small>
                    {root.files} inventoried entries · {root.at_risk} without a
                    retained byte-identical counterpart
                  </small>
                </span>
                <button
                  className="button secondary"
                  onClick={() => void show(root.path)}
                >
                  <FolderOpen size={16} />
                  {native ? "Show folder" : "Copy path"}
                </button>
              </li>
            ))}
          </ul>
          <h4>
            {data.total === 1
              ? "1 file or entry needs"
              : `${data.total} files or entries need`}{" "}
            your attention
          </h4>
          <p>
            These were omitted, unsupported, or unreadable. Similar appearance
            alone does not establish that their distinct contents are preserved.
            Deleting the source folders may lose them.
          </p>
          {data.total === 0 && (
            <p>
              Every inventoried file was included or has an identical copy in
              this export. Check the limits below before removing sources.
            </p>
          )}
          <ul className="cleanup-files">
            {data.items.map((item) => (
              <li key={item.path}>
                <div>
                  <strong>{basename(item.path)}</strong>
                  <small>{item.path}</small>
                  <small>
                    {formatBytes(item.size)} · {item.status}
                  </small>
                  <p>{item.reason}</p>
                </div>
                <button
                  className="button secondary"
                  aria-label={`${native ? "Show" : "Copy path for"} ${item.path}`}
                  onClick={() => void show(item.path)}
                >
                  <FolderOpen size={16} />
                  {native ? "Show file" : "Copy path"}
                </button>
              </li>
            ))}
          </ul>
          {data.total > 20 && (
            <div className="cleanup-pages">
              <button
                className="button secondary"
                disabled={!offset}
                onClick={() => setOffset(offset - 20)}
              >
                <ArrowLeft size={16} />
                Previous
              </button>
              <span>
                {offset + 1}–{Math.min(offset + 20, data.total)} of {data.total}
              </span>
              <button
                className="button secondary"
                disabled={offset + 20 >= data.total}
                onClick={() => setOffset(offset + 20)}
              >
                Next
                <ArrowRight size={16} />
              </button>
            </div>
          )}
          <p className="warning-note">{data.message}</p>
        </>
      )}
      {notice && <p role="status">{notice}</p>}
    </section>
  );
}
