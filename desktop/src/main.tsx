import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCheck,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Copy,
  Folder,
  FolderOpen,
  Image as ImageIcon,
  Layers2,
  Leaf,
  LoaderCircle,
  LockKeyhole,
  MoreHorizontal,
  Plus,
  RotateCcw,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  X,
  ZoomIn,
  Film,
  CheckCircle2,
  Menu,
} from "lucide-react";
import {
  basename,
  formatBytes,
  Group,
  GroupSummary,
  Job,
  kindLabel,
  native,
  pickDirectory,
  Plan,
  reveal,
  rpc,
  Session,
  FileRecord,
} from "./api";
import "./style.css";

function Landscape({ variant = 0 }: { variant?: number }) {
  const colors = [
    ["#dce4df", "#708b7c", "#3d6359"],
    ["#ecddc6", "#ac927a", "#736b50"],
    ["#d9e3e6", "#8aa1a4", "#4b7476"],
  ][variant % 3];
  return (
    <svg
      viewBox="0 0 360 260"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      <rect width="360" height="260" fill={colors[0]} />
      <circle cx="265" cy="65" r="25" fill="#faf3db" />
      <path d="M0 175 115 65 215 175 285 110 360 180V260H0Z" fill={colors[1]} />
      <path d="m0 204 84-74 128 84 90-37 58 25v58H0Z" fill={colors[2]} />
      <path d="M0 235q120-30 360-5v30H0Z" fill="#ffffff" opacity=".13" />
    </svg>
  );
}
function Photo({
  sessionId,
  fileId,
  size = 640,
  alt = "",
}: {
  sessionId: string;
  fileId: string;
  size?: number;
  alt?: string;
}) {
  const [state, setState] = useState<{
    image: string | null;
    message?: string;
  }>({ image: null });
  useEffect(() => {
    let live = true;
    setState({ image: null });
    rpc<{ image: string | null; message: string }>("preview", {
      session_id: sessionId,
      file_id: fileId,
      size,
    })
      .then((s) => {
        if (live) setState(s);
      })
      .catch((e) => {
        if (live) setState({ image: null, message: e.message });
      });
    return () => {
      live = false;
    };
  }, [sessionId, fileId, size]);
  return state.image ? (
    <img src={state.image} alt={alt} draggable={false} />
  ) : (
    <div className="photo-placeholder">
      {state.message ? (
        <>
          <ImageIcon size={30} />
          <span>{state.message}</span>
        </>
      ) : (
        <LoaderCircle className="spin" size={22} />
      )}
    </div>
  );
}
const initialSettings = { workers: 4, threshold: 0.02, full_images: false };

function App() {
  const [sessions, setSessions] = useState<Session[]>([]),
    [session, setSession] = useState<Session | null>(null);
  const [view, setView] = useState("setup"),
    [filter, setFilter] = useState("all"),
    [query, setQuery] = useState("");
  const [groups, setGroups] = useState<GroupSummary[]>([]),
    [total, setTotal] = useState(0),
    [page, setPage] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [roots, setRoots] = useState<string[]>([]),
    [path, setPath] = useState(""),
    [settings, setSettings] = useState(initialSettings);
  const [group, setGroup] = useState<Group | null>(null),
    [selected, setSelected] = useState<string[]>([]),
    [zoom, setZoom] = useState(1);
  const [job, setJob] = useState<Job | null>(null),
    [busy, setBusy] = useState(false),
    [loadingGroups, setLoadingGroups] = useState(false),
    [error, setError] = useState(""),
    [notice, setNotice] = useState("");
  const helpTrigger = useRef<HTMLButtonElement>(null),
    groupTrigger = useRef<HTMLButtonElement | null>(null);
  const [mobileNav, setMobileNav] = useState(false);
  useEffect(() => setMobileNav(false), [view]);
  const [info, setInfo] = useState(false),
    [advanced, setAdvanced] = useState(false),
    [issues, setIssues] = useState<FileRecord[]>([]),
    [issueTotal, setIssueTotal] = useState(0),
    [issuePage, setIssuePage] = useState(0);
  const [destination, setDestination] = useState(""),
    [mode, setMode] = useState("copy"),
    [plan, setPlan] = useState<Plan | null>(null),
    [exported, setExported] = useState("");
  const handledJobs = useRef(new Set<string>());
  const scrolls = useRef<(HTMLDivElement | null)[]>([]),
    scrollLock = useRef(false);
  const refreshSessions = async () => {
    const s = await rpc<Session[]>("sessions");
    setSessions(s);
    return s;
  };
  const reportError = (e: unknown) =>
    setError(e instanceof Error ? e.message : String(e));
  const perform = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      reportError(e);
    } finally {
      setBusy(false);
    }
  };
  const openSession = async (s: Session) => {
    setSession(s);
    setPage(0);
    setFilter("all");
    setQuery("");
    setView("review");
    setPlan(null);
    setExported("");
    localStorage.setItem("dedup-session", s.id);
    if (["scanning", "exporting"].includes(s.state)) {
      try {
        setJob(await rpc<Job>("job", { job_id: s.job_id }));
      } catch {}
    }
  };
  useEffect(() => {
    if (!native) return;
    let stop: (() => void) | undefined;
    let alive = true;
    void import("@tauri-apps/api/webview")
      .then(async ({ getCurrentWebview }) => {
        const unlisten = await getCurrentWebview().onDragDropEvent((event) => {
          if (event.payload.type === "enter") setDragging(true);
          if (event.payload.type === "leave") setDragging(false);
          if (event.payload.type === "drop") {
            const paths = event.payload.paths;
            setDragging(false);
            setRoots((roots) => [...new Set([...roots, ...paths])]);
            setView("setup");
          }
        });
        if (alive) stop = unlisten;
        else unlisten();
      })
      .catch(reportError);
    return () => {
      alive = false;
      stop?.();
    };
  }, []);
  useEffect(() => {
    refreshSessions()
      .then((s) => {
        const previous = s.find(
          (v) => v.id === localStorage.getItem("dedup-session"),
        );
        if (previous) void openSession(previous);
      })
      .catch(reportError);
  }, []);
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(""), 3500);
    return () => clearTimeout(t);
  }, [notice]);
  useEffect(() => {
    if (!session || view !== "review" || session.state !== "ready") return;
    let live = true;
    setLoadingGroups(true);
    rpc<{ items: GroupSummary[]; total: number }>("groups", {
      session_id: session.id,
      filter,
      query,
      offset: page * 30,
      limit: 30,
    })
      .then((r) => {
        if (live) {
          setGroups(r.items);
          setTotal(r.total);
        }
      })
      .catch(reportError)
      .finally(() => {
        if (live) setLoadingGroups(false);
      });
    return () => {
      live = false;
    };
  }, [session, view, filter, query, page]);
  useEffect(() => {
    if (!session || view !== "issues") return;
    rpc<{ items: FileRecord[]; total: number }>("issues", {
      session_id: session.id,
      offset: issuePage * 50,
      limit: 50,
    })
      .then((r) => {
        setIssues(r.items);
        setIssueTotal(r.total);
      })
      .catch(reportError);
  }, [session, view, issuePage]);
  useEffect(() => {
    if (!job || job.state !== "running") return;
    let live = true;
    const timer = setInterval(async () => {
      try {
        const updated = await rpc<Job>("job", { job_id: job.id });
        if (live) setJob(updated);
      } catch (e) {
        if (live) reportError(e);
      }
    }, 700);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [job?.id, job?.state]);
  useEffect(() => {
    if (!job || job.state === "running" || handledJobs.current.has(job.id))
      return;
    handledJobs.current.add(job.id);
    void (async () => {
      const saved = await refreshSessions();
      const current = saved.find((s) => s.id === job.session_id);
      if (current) setSession(current);
      if (job.state === "error") setError(job.error || "The operation failed");
      if (job.state === "cancelled")
        setNotice(job.error || "Stopped safely. Your originals are unchanged.");
      if (job.state === "complete" && job.kind === "export") {
        setExported(job.result?.destination || "");
        setView("export");
      } else if (job.kind === "scan") setView("review");
    })().catch(reportError);
  }, [job?.id, job?.state]);
  useEffect(() => {
    setPlan(null);
  }, [destination, mode, session?.reviewed]);
  const start = async (scanRoots = roots, name?: string) => {
    const result = await rpc<{ session_id: string; job_id: string }>(
      "start_scan",
      { roots: scanRoots, name, ...settings },
    );
    const s = await rpc<Session>("session", { session_id: result.session_id });
    await openSession(s);
    setJob(await rpc<Job>("job", { job_id: result.job_id }));
    await refreshSessions();
  };
  const demo = () =>
    perform(async () => {
      const demo = await rpc<{ roots: string[]; name: string }>("demo");
      await start(demo.roots, demo.name);
    });
  const addFolder = () =>
    perform(async () => {
      let folder = path.trim();
      if (native) folder = (await pickDirectory()) || "";
      if (folder) {
        setRoots((r) => [...new Set([...r, folder])]);
        setPath("");
      }
    });
  const openGroup = async (id: string) => {
    if (!session) return;
    const g = await rpc<Group>("group", {
      session_id: session.id,
      group_id: id,
    });
    setGroup(g);
    setSelected(g.selected);
    setZoom(1);
  };
  const save = async (action: string, choice = selected) => {
    if (!session || !group) return;
    await rpc("decide", {
      session_id: session.id,
      group_id: group.id,
      action,
      selected: choice,
    });
    setGroup(null);
    setSession(await rpc<Session>("session", { session_id: session.id }));
    await refreshSessions();
    setNotice("Choice saved. Your originals are unchanged.");
  };
  const undo = () =>
    perform(async () => {
      if (!session) return;
      const result = await rpc<{ undone: boolean }>("undo", {
        session_id: session.id,
      });
      setSession(await rpc<Session>("session", { session_id: session.id }));
      await refreshSessions();
      setNotice(
        result.undone
          ? "Last review choice undone."
          : "No review choices to undo.",
      );
    });
  const navigateGroup = (delta: number) => {
    const index = groups.findIndex((g) => g.id === group?.id);
    const next = groups[index + delta];
    if (next) void perform(() => openGroup(next.id));
  };
  const syncScroll = (index: number, element: HTMLDivElement) => {
    if (scrollLock.current) return;
    scrollLock.current = true;
    scrolls.current.forEach((other, i) => {
      if (other && i !== index) {
        other.scrollLeft = element.scrollLeft;
        other.scrollTop = element.scrollTop;
      }
    });
    requestAnimationFrame(() => {
      scrollLock.current = false;
    });
  };
  const inventory = Object.values(session?.counts || {}).reduce(
    (a, b) => a + b,
    0,
  );
  const isRunning = job?.state === "running";
  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "mobile-expanded" : ""}`}>
        <button
          className="brand"
          onClick={() => {
            setView("setup");
            setGroup(null);
          }}
          aria-label="Photo Dedup home"
        >
          <span className="brand-icon">
            <Layers2 size={22} />
          </span>
          <span>
            Photo Dedup
            <span className="brand-caption">A little room to breathe</span>
          </span>
        </button>
        <button
          className="new-scan"
          onClick={() => {
            setView("setup");
            setRoots([]);
          }}
        >
          <Plus size={17} /> New scan <kbd>＋</kbd>
        </button>
        <button
          className="icon-button mobile-menu-button"
          aria-label="Toggle navigation"
          aria-expanded={mobileNav}
          onClick={() => setMobileNav(!mobileNav)}
        >
          <Menu size={21} />
        </button>
        <div className="nav-label">WORKSPACE</div>
        <button
          className={`nav-item ${view === "review" ? "active" : ""}`}
          disabled={!session}
          onClick={() => setView("review")}
        >
          <Layers2 size={18} /> Review library{" "}
          {session && <span className="nav-count">{session.groups}</span>}
        </button>
        <button
          className={`nav-item ${view === "export" ? "active" : ""}`}
          disabled={!session || session.state !== "ready"}
          onClick={() => setView("export")}
        >
          <ArrowDownToLine size={18} /> Create library
        </button>
        <button
          className={`nav-item ${view === "issues" ? "active" : ""}`}
          disabled={!session || session.state !== "ready"}
          onClick={() => {
            setIssuePage(0);
            setView("issues");
          }}
        >
          <CircleHelp size={18} /> Skipped & warnings{" "}
          {session?.warnings ? <span className="nav-dot" /> : null}
        </button>
        <div className="sessions-heading">
          <span className="nav-label">RECENT SCANS</span>
          <span>{sessions.length}</span>
        </div>
        <div className="session-list">
          {sessions.map((s) => (
            <button
              key={s.id}
              className={`session-item ${s.id === session?.id ? "chosen" : ""}`}
              onClick={() => void perform(() => openSession(s))}
            >
              <span className="session-dot" />
              <span>
                <strong>{s.name}</strong>
                <small>
                  {s.state === "ready" ? `${s.groups} match groups` : s.state} ·{" "}
                  {new Date(s.created_at).toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                  })}
                </small>
              </span>
            </button>
          ))}
          {!sessions.length && (
            <p className="empty-sessions">
              Your scans will be saved here.
              <br />
              Pick up whenever you like.
            </p>
          )}
        </div>
        <div className="sidebar-bottom">
          <div className="private-note">
            <ShieldCheck size={21} />
            <span>
              <strong>Just on your computer</strong>
              <small>No uploads. No account needed.</small>
            </span>
          </div>
          <button
            ref={helpTrigger}
            className="nav-item"
            onClick={() => setInfo(true)}
          >
            <CircleHelp size={17} /> About & help{" "}
            <span className="version">v0.2</span>
          </button>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Your workspace</span>
            <ChevronRight size={14} />
            <span>
              {view === "setup"
                ? "New scan"
                : view === "export"
                  ? "Create library"
                  : view === "issues"
                    ? "Skipped & warnings"
                    : "Review library"}
            </span>
          </div>
          <span className="safe-label">
            <LockKeyhole size={13} />
            <span>Originals stay untouched</span>
          </span>
        </header>
        <main>
          {error && (
            <div className="alert error" role="alert">
              <TriangleAlert size={18} />
              <span>{error}</span>
              <button onClick={() => setError("")} aria-label="Dismiss error">
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div className="toast" role="status">
              <CheckCircle2 size={17} />
              {notice}
            </div>
          )}
          {view === "setup" && (
            <>
              <section className="welcome">
                <div className="eyebrow">
                  <span /> LESS CLUTTER. MORE CLARITY.
                </div>
                <h1>
                  A little room
                  <br />
                  to breathe<span>.</span>
                </h1>
                <p>
                  Find the photos worth keeping.
                  <br />
                  Bring a little order to years of memories.
                </p>
                <div className="welcome-assurance">
                  <ShieldCheck size={16} /> Review first. Your originals always
                  stay safe.
                </div>
              </section>
              <div className="welcome-art" aria-hidden="true">
                <div className="art-photo art-back">
                  <Landscape variant={1} />
                </div>
                <div className="art-photo art-front">
                  <Landscape />
                  <div className="art-caption">
                    <span>One memory. One favorite.</span>
                    <CheckCircle2 size={16} />
                  </div>
                </div>
                <div className="art-tag">
                  <Sparkles size={14} /> A clearer collection
                </div>
              </div>
              <section className="setup-grid">
                <div className="start-card">
                  <div className="section-heading">
                    <span className="step-number">1</span>
                    <div>
                      <h2>Where are your photos?</h2>
                      <p>Choose one or more folders to compare.</p>
                    </div>
                  </div>
                  <div className={`folder-drop ${dragging ? "dragging" : ""}`}>
                    <div className="folder-illustration">
                      <FolderOpen size={33} strokeWidth={1.3} />
                    </div>
                    <h3>
                      {dragging
                        ? "Drop your folders here"
                        : native
                          ? "Drop your photo folders here"
                          : "Add your photo folders"}
                    </h3>
                    <p>Camera imports, old downloads, or a backup folder.</p>
                    {!native && (
                      <div className="path-input">
                        <label className="sr-only" htmlFor="source-path">
                          Folder path
                        </label>
                        <input
                          id="source-path"
                          value={path}
                          onChange={(e) => setPath(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") void addFolder();
                          }}
                          placeholder="Paste a local folder path…"
                        />
                      </div>
                    )}
                    <button
                      className="button secondary"
                      onClick={() => void addFolder()}
                      disabled={busy || (!native && !path.trim())}
                    >
                      <Plus size={16} /> Choose a folder
                    </button>
                  </div>
                  {roots.length > 0 && (
                    <ul className="root-list">
                      {roots.map((root) => (
                        <li key={root}>
                          <Folder size={17} />
                          <span title={root}>{root}</span>
                          <button
                            onClick={() =>
                              setRoots((r) => r.filter((p) => p !== root))
                            }
                            aria-label={`Remove ${root}`}
                          >
                            <X size={15} />
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                  <button
                    className="advanced-toggle"
                    onClick={() => setAdvanced(!advanced)}
                    aria-expanded={advanced}
                  >
                    <Settings2 size={15} /> Scan preferences{" "}
                    <ChevronRight
                      size={14}
                      className={advanced ? "rotate" : ""}
                    />
                  </button>
                  {advanced && (
                    <div className="settings">
                      <label>
                        Disk access
                        <select
                          value={settings.workers}
                          onChange={(e) =>
                            setSettings({
                              ...settings,
                              workers: Number(e.target.value),
                            })
                          }
                        >
                          <option value={4}>Balanced · 4 workers</option>
                          <option value={1}>Gentle · 1 worker</option>
                          <option value={8}>Fast SSD · 8 workers</option>
                        </select>
                      </label>
                      <label>
                        <span>
                          Visual threshold{" "}
                          <small>Similar matches always need review</small>
                        </span>
                        <input
                          type="number"
                          min="0"
                          max="1"
                          step="0.005"
                          value={settings.threshold}
                          onChange={(e) =>
                            setSettings({
                              ...settings,
                              threshold: Number(e.target.value),
                            })
                          }
                        />
                      </label>
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={settings.full_images}
                          onChange={(e) =>
                            setSettings({
                              ...settings,
                              full_images: e.target.checked,
                            })
                          }
                        />{" "}
                        Compare full images{" "}
                        <small>Slower; uses more memory</small>
                      </label>
                    </div>
                  )}
                  <div className="start-actions">
                    <span>
                      {roots.length
                        ? `${roots.length} folder${roots.length === 1 ? "" : "s"} selected`
                        : "You choose what to keep"}
                    </span>
                    <button
                      className="button primary"
                      disabled={!roots.length || busy || isRunning}
                      onClick={() => void perform(() => start())}
                    >
                      Find duplicates <ArrowRight size={16} />
                    </button>
                  </div>
                </div>
                <div className="how-card">
                  <span className="eyebrow">A THOUGHTFUL LITTLE CLEANUP</span>
                  <h2>
                    Make space for
                    <br />
                    what matters.
                  </h2>
                  <ol>
                    {[
                      [
                        "Find the overlaps",
                        "We look for identical files and photos that might be similar.",
                      ],
                      [
                        "You make the call",
                        "Compare side by side. Keep your favorite, or keep both.",
                      ],
                      [
                        "Create a fresh library",
                        "Export an organized collection. Your source files stay right where they are.",
                      ],
                    ].map(([title, body], i) => (
                      <li key={title}>
                        <span>{i + 1}</span>
                        <div>
                          <h3>{title}</h3>
                          <p>{body}</p>
                        </div>
                      </li>
                    ))}
                  </ol>
                  <button
                    className="demo-link"
                    disabled={busy || isRunning}
                    onClick={() => void demo()}
                  >
                    <Sparkles size={16} /> Take a look with demo photos{" "}
                    <ArrowRight size={15} />
                  </button>
                  <p className="demo-note">
                    A tiny sample library. Nothing personal.
                  </p>
                </div>
              </section>
              <div className="bottom-note">
                <Leaf size={15} /> Made for your photos. Designed for your peace
                of mind.
              </div>
            </>
          )}
          {view !== "setup" && session && (
            <>
              <div className="page-title">
                <div>
                  <div className="eyebrow">
                    {view === "review"
                      ? "YOUR LIBRARY, A LITTLE LIGHTER"
                      : view === "export"
                        ? "THE NEXT CHAPTER"
                        : "EVERY FILE ACCOUNTED FOR"}
                  </div>
                  <h1>
                    {view === "review"
                      ? "A clearer picture."
                      : view === "export"
                        ? "Make it your library."
                        : "Nothing left unexplained."}
                  </h1>
                  <p>
                    {view === "review"
                      ? session.name
                      : view === "export"
                        ? "Your choices, organized into a fresh folder. Originals stay put."
                        : "Unsupported files and metadata warnings are listed here for review."}
                  </p>
                </div>
                {view === "review" && session.state === "ready" && (
                  <button
                    className="button primary"
                    onClick={() => setView("export")}
                  >
                    Create library <ArrowRight size={16} />
                  </button>
                )}
              </div>
              {isRunning && job && (
                <section
                  className="progress-card"
                  role="status"
                  aria-live="polite"
                >
                  <LoaderCircle className="spin" size={24} />
                  <div>
                    <h3>{job.phase}</h3>
                    <p>
                      {job.done.toLocaleString()} of{" "}
                      {job.total ? job.total.toLocaleString() : "…"}{" "}
                      {job.kind === "scan"
                        ? "entries inspected"
                        : "files exported"}{" "}
                      {job.cached > 0 && `· ${job.cached} cached`}
                    </p>
                    <progress value={job.done} max={Math.max(job.total, 1)} />
                  </div>
                  <button
                    className="button secondary"
                    onClick={() =>
                      void perform(async () => {
                        setJob(await rpc<Job>("cancel", { job_id: job.id }));
                      })
                    }
                  >
                    Cancel safely
                  </button>
                </section>
              )}
              {view === "review" && (
                <>
                  {!["ready", "scanning", "exporting"].includes(
                    session.state,
                  ) && (
                    <div className="recovery-card">
                      <TriangleAlert size={24} />
                      <h2>
                        {session.state === "cancelled"
                          ? "Paused here. Ready when you are."
                          : "This scan needs a fresh start."}
                      </h2>
                      <p>
                        {session.error ||
                          "Cached inspections will be reused when you restart."}
                      </p>
                      <button
                        className="button primary"
                        disabled={busy || isRunning}
                        onClick={() =>
                          void perform(async () => {
                            const r = await rpc<{
                              session_id: string;
                              job_id: string;
                            }>("restart_scan", { session_id: session.id });
                            await openSession(
                              await rpc<Session>("session", {
                                session_id: r.session_id,
                              }),
                            );
                            setJob(await rpc<Job>("job", { job_id: r.job_id }));
                            await refreshSessions();
                          })
                        }
                      >
                        Restart scan <RotateCcw size={16} />
                      </button>
                    </div>
                  )}
                  {session.state === "ready" && (
                    <>
                      <section className="stats">
                        <div>
                          <span className="stat-icon">
                            <ImageIcon size={20} />
                          </span>
                          <div>
                            <strong>{inventory.toLocaleString()}</strong>
                            <span>entries scanned</span>
                          </div>
                        </div>
                        <div>
                          <span className="stat-icon">
                            <Layers2 size={20} />
                          </span>
                          <div>
                            <strong>{session.groups.toLocaleString()}</strong>
                            <span>match groups to explore</span>
                          </div>
                        </div>
                        <div>
                          <span className="stat-icon">
                            <CheckCheck size={20} />
                          </span>
                          <div>
                            <strong>
                              {session.reviewed}
                              <small> / {session.groups}</small>
                            </strong>
                            <span>groups reviewed</span>
                          </div>
                        </div>
                      </section>
                      <div className="results-toolbar">
                        <div className="tabs" aria-label="Filter match groups">
                          {[
                            ["all", "All matches"],
                            ["exact", "Identical"],
                            ["visual", "Similar"],
                            ["metadata", "Possible"],
                            ["reviewed", "Reviewed"],
                            ["unique", "Unique"],
                          ].map(([key, label]) => (
                            <button
                              key={key}
                              className={filter === key ? "selected" : ""}
                              aria-pressed={filter === key}
                              onClick={() => {
                                setFilter(key);
                                setPage(0);
                              }}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                        <button
                          className="icon-button"
                          onClick={() => void undo()}
                          disabled={busy}
                          title="Undo last review choice"
                          aria-label="Undo last review choice"
                        >
                          <RotateCcw size={17} />
                        </button>
                      </div>
                      <div className="results-subbar">
                        <span>
                          {loadingGroups
                            ? "Loading matches…"
                            : `${total} ${filter === "unique" ? "unique files" : "match groups"}`}
                          <span className="muted"> · largest groups first</span>
                        </span>
                        <label className="search">
                          <Search size={15} />
                          <input
                            aria-label="Search filenames and paths"
                            placeholder="Find a filename…"
                            value={query}
                            onChange={(e) => {
                              setQuery(e.target.value);
                              setPage(0);
                            }}
                          />
                        </label>
                      </div>
                      {!loadingGroups && !groups.length && (
                        <div className="empty-state">
                          <CheckCircle2 size={36} />
                          <h2>
                            {query
                              ? "No matches for that search."
                              : filter === "all"
                                ? "Everything has its own place."
                                : "Nothing in this view."}
                          </h2>
                          <p>
                            {filter === "all"
                              ? "No duplicate groups were found. You can still create an organized copy of your library."
                              : "Try another filter or search."}
                          </p>
                        </div>
                      )}
                      <h2 className="sr-only">Photo match groups</h2>
                      <div className="group-grid">
                        {groups.map((g) => (
                          <button
                            className="group-card"
                            key={g.id}
                            onClick={(e) => {
                              groupTrigger.current = e.currentTarget;
                              void perform(() => openGroup(g.id));
                            }}
                          >
                            <div className="group-image">
                              {g.has_image ? (
                                <Photo
                                  sessionId={session.id}
                                  fileId={g.preview_id}
                                  size={640}
                                  alt={g.title}
                                />
                              ) : (
                                <div className="video-placeholder">
                                  {g.media_kind === "video" ? (
                                    <Film size={35} />
                                  ) : (
                                    <ImageIcon size={35} />
                                  )}
                                  <span>
                                    {g.media_kind === "video"
                                      ? "Video · metadata comparison"
                                      : "RAW file · original preserved"}
                                  </span>
                                </div>
                              )}
                              <span className="image-count">
                                <Layers2 size={13} />
                                {g.count} files
                              </span>
                              {!!g.decision && (
                                <span className="reviewed-badge">
                                  <Check size={13} /> Reviewed
                                </span>
                              )}
                            </div>
                            <div className="group-info">
                              <span className={`match-tag ${g.kind}`}>
                                <span />
                                {kindLabel[g.kind]}
                              </span>
                              <h3>{g.title}</h3>
                              <div>
                                <span>{formatBytes(g.size)} combined</span>
                                <ArrowRight size={16} />
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                      {total > 30 && (
                        <div className="pagination">
                          <button
                            className="button secondary"
                            disabled={page === 0}
                            onClick={() => setPage((p) => p - 1)}
                          >
                            <ChevronLeft size={16} /> Previous
                          </button>
                          <span>
                            {page + 1} / {Math.ceil(total / 30)}
                          </span>
                          <button
                            className="button secondary"
                            disabled={(page + 1) * 30 >= total}
                            onClick={() => setPage((p) => p + 1)}
                          >
                            Next <ChevronRight size={16} />
                          </button>
                        </div>
                      )}
                      <div className="results-footer">
                        <ShieldCheck size={15} />
                        <span>
                          Suggestions are a starting point. Similar photos
                          always deserve a closer look.
                        </span>
                      </div>
                    </>
                  )}
                </>
              )}
              {view === "issues" && (
                <section className="issues-card">
                  <div className="section-heading">
                    <CircleHelp size={21} />
                    <div>
                      <h2>{issueTotal} entries to know about</h2>
                      <p>
                        These entries remain untouched in their original
                        folders.
                      </p>
                    </div>
                  </div>
                  {issues.map((r) => (
                    <div className="issue-row" key={r.id}>
                      <TriangleAlert size={17} />
                      <div>
                        <strong>{basename(r.path)}</strong>
                        <p>{r.warnings?.join("; ") || r.reason}</p>
                        <small>{r.path}</small>
                      </div>
                      <span className="pill">{r.status}</span>
                    </div>
                  ))}
                  {!issues.length && (
                    <div className="empty-state">
                      <CheckCircle2 size={30} />
                      <p>No skipped files or warnings.</p>
                    </div>
                  )}
                  {issueTotal > 50 && (
                    <div className="pagination">
                      <button
                        className="button secondary"
                        disabled={issuePage === 0}
                        onClick={() => setIssuePage((p) => p - 1)}
                      >
                        Previous
                      </button>
                      <span>
                        {issuePage + 1} / {Math.ceil(issueTotal / 50)}
                      </span>
                      <button
                        className="button secondary"
                        disabled={(issuePage + 1) * 50 >= issueTotal}
                        onClick={() => setIssuePage((p) => p + 1)}
                      >
                        Next
                      </button>
                    </div>
                  )}
                </section>
              )}
              {view === "export" && (
                <section className="export-grid">
                  <div className="export-card">
                    {exported ? (
                      <div className="export-success">
                        <span>
                          <CheckCheck size={35} />
                        </span>
                        <h2>A fresh start for your photos.</h2>
                        <p>
                          Your organized library is ready. Original files were
                          left in place.
                        </p>
                        <code>{exported}</code>
                        <button
                          className="button primary"
                          onClick={() =>
                            void perform(async () => {
                              await reveal(exported);
                              if (!native)
                                setNotice("Export path copied to clipboard.");
                            })
                          }
                        >
                          <FolderOpen size={17} />
                          {native
                            ? "Show in Finder / Explorer"
                            : "Copy folder path"}
                        </button>
                        <button
                          className="text-button"
                          onClick={() => {
                            setExported("");
                            setDestination("");
                            setPlan(null);
                          }}
                        >
                          Create another export
                        </button>
                      </div>
                    ) : (
                      <>
                        <div className="section-heading">
                          <span className="step-number">3</span>
                          <div>
                            <h2>Choose a new home</h2>
                            <p>
                              We create a new folder, never overwrite an
                              existing one.
                            </p>
                          </div>
                        </div>
                        <label className="field-label" htmlFor="export-path">
                          New library folder
                        </label>
                        <div className="export-input">
                          <input
                            id="export-path"
                            placeholder="Path to a new, empty destination…"
                            value={destination}
                            onChange={(e) => setDestination(e.target.value)}
                          />
                          {native && (
                            <button
                              aria-label="Choose parent folder"
                              className="icon-button"
                              onClick={() =>
                                void perform(async () => {
                                  const p = await pickDirectory();
                                  if (p)
                                    setDestination(
                                      p +
                                        (p.includes("\\") ? "\\" : "/") +
                                        "Photo Dedup Library",
                                    );
                                })
                              }
                            >
                              <FolderOpen size={20} />
                            </button>
                          )}
                        </div>
                        <div className="mode-options">
                          <label className={mode === "copy" ? "checked" : ""}>
                            <input
                              type="radio"
                              name="mode"
                              value="copy"
                              checked={mode === "copy"}
                              onChange={() => setMode("copy")}
                            />
                            <span>
                              <strong>
                                Independent copies{" "}
                                <span className="recommended">Recommended</span>
                              </strong>
                              <small>
                                Edit your new library without affecting your
                                originals.
                              </small>
                            </span>
                            <Copy size={19} />
                          </label>
                          <label
                            className={mode === "hardlink" ? "checked" : ""}
                          >
                            <input
                              type="radio"
                              name="mode"
                              value="hardlink"
                              checked={mode === "hardlink"}
                              onChange={() => setMode("hardlink")}
                            />
                            <span>
                              <strong>
                                Hard links{" "}
                                <span className="advanced-badge">Advanced</span>
                              </strong>
                              <small>
                                Same filesystem only. Editing either file
                                changes both.
                              </small>
                            </span>
                            <Layers2 size={19} />
                          </label>
                        </div>
                        <button
                          className="button secondary full"
                          disabled={!destination.trim() || busy || isRunning}
                          onClick={() =>
                            void perform(async () =>
                              setPlan(
                                await rpc<Plan>("export_plan", {
                                  session_id: session.id,
                                  destination,
                                  mode,
                                }),
                              ),
                            )
                          }
                        >
                          Preview my library <ArrowRight size={16} />
                        </button>
                        {plan && (
                          <div className="export-plan">
                            <h3>Here’s what will happen</h3>
                            <div>
                              <span>Files in the new library</span>
                              <strong>{plan.files}</strong>
                            </div>
                            <div>
                              <span>
                                {mode === "copy"
                                  ? "Copy size"
                                  : "Source data represented"}
                              </span>
                              <strong>{formatBytes(plan.bytes)}</strong>
                            </div>
                            <div>
                              <span>
                                Alternatives left only in their source folders
                              </span>
                              <strong>{plan.omitted}</strong>
                            </div>
                            <div>
                              <span>Groups still awaiting a choice</span>
                              <strong>{plan.unresolved}</strong>
                            </div>
                            {plan.unresolved > 0 && (
                              <p className="plan-note">
                                Unreviewed similar and possible matches are kept
                                together for later review. Exact byte duplicates
                                use the suggested representative.
                              </p>
                            )}
                            <button
                              className="button primary full"
                              disabled={busy || isRunning}
                              onClick={() =>
                                void perform(async () => {
                                  const r = await rpc<{ job_id: string }>(
                                    "start_export",
                                    {
                                      session_id: session.id,
                                      destination,
                                      mode,
                                    },
                                  );
                                  setJob(
                                    await rpc<Job>("job", { job_id: r.job_id }),
                                  );
                                  setSession(
                                    await rpc<Session>("session", {
                                      session_id: session.id,
                                    }),
                                  );
                                })
                              }
                            >
                              Create my library <ArrowDownToLine size={16} />
                            </button>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                  <div className="export-explainer">
                    <ShieldCheck size={29} />
                    <h2>
                      A new collection.
                      <br />
                      The same peace of mind.
                    </h2>
                    <p>
                      Every exported file is checked against its scan
                      fingerprint. If a source changed, we stop and explain.
                    </p>
                    <ul>
                      <li>
                        <Check size={15} /> Your originals stay in place
                      </li>
                      <li>
                        <Check size={15} /> Sidecars and companions are
                        preserved
                      </li>
                      <li>
                        <Check size={15} /> A complete report comes along
                      </li>
                      <li>
                        <Check size={15} /> Failed exports clean up their
                        partial output
                      </li>
                    </ul>
                    <div className="small-note">
                      Creating a library does not free space in your source
                      folders. This app does not delete source files.
                    </div>
                  </div>
                </section>
              )}
            </>
          )}
        </main>
      </div>
      <Dialog.Root
        open={!!group}
        onOpenChange={(open) => {
          if (!open) setGroup(null);
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content
            className="compare-dialog"
            aria-describedby="compare-description"
            onCloseAutoFocus={(e) => {
              e.preventDefault();
              if (groupTrigger.current?.isConnected)
                groupTrigger.current.focus();
            }}
          >
            <div className="compare-header">
              <div>
                <span className="eyebrow">A CLOSER LOOK</span>
                <Dialog.Title>
                  {group ? basename(group.records[0].path) : "Compare files"}
                </Dialog.Title>
              </div>
              <div className="compare-header-actions">
                <button
                  className="icon-button"
                  aria-label="Previous group"
                  disabled={groups.findIndex((g) => g.id === group?.id) <= 0}
                  onClick={() => navigateGroup(-1)}
                >
                  <ChevronLeft size={19} />
                </button>
                <button
                  className="icon-button"
                  aria-label="Next group"
                  disabled={
                    groups.findIndex((g) => g.id === group?.id) >=
                    groups.length - 1
                  }
                  onClick={() => navigateGroup(1)}
                >
                  <ChevronRight size={19} />
                </button>
                <Dialog.Close
                  className="icon-button"
                  aria-label="Close comparison"
                >
                  <X size={21} />
                </Dialog.Close>
              </div>
            </div>
            <Dialog.Description
              id="compare-description"
              className="compare-description"
            >
              Choose the files for your new library. Your originals stay
              untouched. At least one file stays in every group.
            </Dialog.Description>
            {group && session && (
              <>
                <div className="compare-tools">
                  <span>
                    <ShieldCheck size={15} />{" "}
                    {group.records.some((r) => r.confidence === "metadata_only")
                      ? "Metadata agrees. Media content is not verified."
                      : group.records.some(
                            (r) => r.confidence === "visual_similarity",
                          )
                        ? "Similar appearance. Check detail and edits before choosing."
                        : "Identical SHA-256 fingerprints. The file bytes match."}
                  </span>
                  <label>
                    <ZoomIn size={16} />
                    <input
                      aria-label="Comparison zoom"
                      type="range"
                      min="1"
                      max="3"
                      step=".25"
                      value={zoom}
                      onChange={(e) => setZoom(Number(e.target.value))}
                    />
                    {Math.round(zoom * 100)}%
                  </label>
                </div>
                <div className="compare-files">
                  {group.records.map((r, i) => (
                    <article
                      key={r.id}
                      className={`compare-file ${selected.includes(r.id) ? "is-kept" : ""}`}
                    >
                      <div className="compare-file-heading">
                        <label>
                          <input
                            type="checkbox"
                            checked={selected.includes(r.id)}
                            onChange={(e) =>
                              setSelected((s) =>
                                e.target.checked
                                  ? [...s, r.id]
                                  : s.filter((id) => id !== r.id),
                              )
                            }
                          />
                          <strong>{basename(r.path)}</strong>
                        </label>
                        {r.id === group.recommendation && (
                          <span className="recommended">
                            <Sparkles size={11} /> Suggested
                          </span>
                        )}
                      </div>
                      <div
                        className="compare-photo"
                        tabIndex={0}
                        role="region"
                        aria-label={`Image preview for ${basename(r.path)}`}
                        ref={(el) => {
                          scrolls.current[i] = el;
                        }}
                        onScroll={(e) => syncScroll(i, e.currentTarget)}
                      >
                        <div
                          className="zoom-content"
                          style={{
                            width: `${zoom * 100}%`,
                            height: `${zoom * 100}%`,
                          }}
                        >
                          <Photo
                            sessionId={session.id}
                            fileId={r.id}
                            size={zoom > 1 ? 1600 : 640}
                            alt={basename(r.path)}
                          />
                        </div>
                      </div>
                      <div className="file-details">
                        <div>
                          <strong>{r.extension.toUpperCase().slice(1)}</strong>
                          <span>
                            {r.width && r.height
                              ? `${r.width} × ${r.height}`
                              : "Dimensions unavailable"}
                          </span>
                          <span>{formatBytes(r.size)}</span>
                        </div>
                        <p>
                          {r.id === group.recommendation
                            ? "Suggested by format, filename, and file date. Original status is unverified."
                            : r.confidence === "visual_similarity"
                              ? "These photos look alike. Edits, fine detail, or metadata may differ."
                              : r.confidence === "exact_bytes"
                                ? "This file has exactly the same bytes as its matching representative."
                                : "Capture metadata agrees. The media content has not been verified."}
                        </p>
                        <small className="capture-details">
                          {r.model && r.model !== "Unknown"
                            ? `${r.model} · `
                            : ""}
                          {r.captured_at || "Capture date unavailable"}
                        </small>
                        <details className="evidence-details">
                          <summary>Matching evidence</summary>
                          <p>{r.reason}</p>
                          {r.sha256 && <code>SHA-256: {r.sha256}</code>}
                        </details>
                        <small className="file-path" title={r.path}>
                          {r.path}
                        </small>
                        {!!r.related_files?.length && (
                          <small className="companion-note">
                            <Layers2 size={12} />
                            {r.related_files.length} related companion file(s)
                            recorded
                          </small>
                        )}
                        {!!r.warnings?.length && (
                          <small className="warning-note">
                            <TriangleAlert size={12} />
                            {r.warnings.join("; ")}
                          </small>
                        )}
                        <button
                          className="text-button"
                          onClick={() =>
                            void perform(async () => {
                              await reveal(r.path);
                              if (!native)
                                setNotice("File path copied to clipboard.");
                            })
                          }
                        >
                          {native
                            ? "Show original in Finder / Explorer"
                            : "Copy original path"}{" "}
                          <FolderOpen size={13} />
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
                <div className="compare-footer">
                  <div>
                    <button
                      className="button secondary"
                      disabled={busy}
                      onClick={() => void perform(() => save("keep_all"))}
                    >
                      Keep all
                    </button>
                    <button
                      className="text-button"
                      disabled={busy}
                      onClick={() => void perform(() => save("not_duplicates"))}
                    >
                      These aren’t duplicates
                    </button>
                  </div>
                  <div>
                    <span className="muted">
                      {selected.length} of {group.records.length} selected
                    </span>
                    <button
                      className="button primary"
                      disabled={busy || !selected.length}
                      onClick={() => void perform(() => save("keep_selected"))}
                    >
                      Save choice <Check size={16} />
                    </button>
                  </div>
                </div>
              </>
            )}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root open={info} onOpenChange={setInfo}>
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content
            className="help-dialog"
            onCloseAutoFocus={(e) => {
              e.preventDefault();
              helpTrigger.current?.focus();
            }}
          >
            <Dialog.Close
              className="icon-button help-close"
              aria-label="Close help"
            >
              <X size={20} />
            </Dialog.Close>
            <span className="brand-icon">
              <Layers2 size={25} />
            </span>
            <Dialog.Title>A little room to breathe.</Dialog.Title>
            <Dialog.Description>
              Photo Dedup helps you review repeated photo imports and create an
              organized library.
            </Dialog.Description>
            <h3>Your files stay yours</h3>
            <p>
              Scans, previews, and choices stay on this computer. There is no
              account, telemetry, or photo upload. Source files are never
              deleted or edited by the app.
            </p>
            <h3>Know your matches</h3>
            <p>
              <strong>Identical:</strong> the file bytes match.{" "}
              <strong>Similar:</strong> image previews look alike, but edits may
              differ. <strong>Possible:</strong> metadata agrees; media content
              is not verified.
            </p>
            <h3>A few useful details</h3>
            <p>
              HEIC and common images have previews. Videos and RAW files
              currently use byte and metadata matching. Export originals from
              Apple Photos before scanning; direct Photos library editing is not
              supported. Review choices save automatically and can be undone.
            </p>
            <small>Photo Dedup 0.2.0 · Apache-2.0 · Works offline</small>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      {busy && !group && (
        <span className="busy-indicator" role="status">
          <LoaderCircle size={15} className="spin" /> Working locally…
        </span>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
