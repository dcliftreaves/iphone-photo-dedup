"""Inventory every input entry, inspect with bounded workers, then classify."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from concurrent.futures import CancelledError
from pathlib import Path

from .cache import Cache
from .matching import classify
from .media import SUPPORTED, engine_signature, exiftool_path, inspect


def inventory(roots, excludes):
    seen = set()
    for root in roots:
        def walk_error(error):
            raise OSError(f"Cannot enumerate {error.filename}: {error.strerror}")

        for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
            current = Path(directory)
            dirs.sort()
            for name in list(dirs):
                path = current / name
                if any(path == e or e in path.parents for e in excludes):
                    dirs.remove(name)
                elif path.suffix.lower() == '.photoslibrary':
                    dirs.remove(name)
                    if path not in seen:
                        seen.add(path)
                        yield path, "Apple Photos library was not entered; export originals from Photos first."
                elif path.is_symlink():
                    dirs.remove(name)
                    if path not in seen:
                        seen.add(path)
                        yield path, "Symbolic-link directory was not followed."
            for name in sorted(files):
                path = current / name
                if path in seen or any(path == e or e in path.parents for e in excludes):
                    continue
                seen.add(path)
                if path.is_symlink():
                    yield path, "Symbolic link was not followed."
                elif not path.is_file():
                    yield path, "Not a regular file."
                elif path.suffix.lower() not in SUPPORTED:
                    yield path, "Unsupported file extension."
                else:
                    yield path, None


def scan(roots, *, cache_path=None, excludes=(), workers=4, threshold=0.02,
         full_images=False, progress=None, cancel_event=None):
    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Scan cancelled; cached inspections are available for a restart")
    executable = exiftool_path()
    cache = Cache(cache_path, engine_signature(executable))
    try:
        entries = []
        for entry in inventory(roots, excludes):
            check_cancel()
            entries.append(entry)
        records = []
        jobs = []
        for path, reason in entries:
            check_cancel()
            if reason:
                try:
                    size = path.lstat().st_size
                except OSError:
                    size = 0
                records.append(dict(path=str(path), status="ignored", reason=reason,
                                    warnings=[], confidence="unassessed", size=size))
                continue
            try:
                cached = cache.get(path, path.stat())
            except OSError as error:
                records.append(dict(path=str(path), status="error", reason=str(error),
                                    warnings=[], confidence="unassessed", size=0))
                continue
            if cached:
                records.append(dict(cached, status="pending"))
            else:
                jobs.append(path)
        if progress:
            progress(len(records), len(entries), cache.hits)

        def work(path):
            check_cancel()
            try:
                result = inspect(path, executable)
                return dict(result, status="pending")
            except Exception as error:
                # An unreadable/corrupt file must appear in the report, not abort the scan.
                return dict(path=str(path), status="error", reason=str(error),
                            confidence="unassessed", size=0, warnings=[])

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Batches bound pending work for large libraries while preserving input order.
            for offset in range(0, len(jobs), workers * 4):
                for record in executor.map(work, jobs[offset:offset + workers * 4]):
                    check_cancel()
                    if record["status"] != "error" and record.get("_cacheable", False):
                        path = Path(record["path"])
                        try:
                            current = path.stat()
                            if (current.st_ino, current.st_size, current.st_mtime_ns,
                                current.st_ctime_ns) == (record["inode"], record["size"],
                                                        record["mtime_ns"], record["ctime_ns"]):
                                cache.put(path, current, {k: v for k, v in record.items() if k != "status"})
                        except OSError:
                            pass
                    records.append(record)
                    if progress:
                        progress(len(records), len(entries), cache.hits)
        check_cancel()
        classify(records, threshold, full_images, check_cancel=check_cancel)
        records.sort(key=lambda r: r["path"])
        for record in records:
            record.pop("thumbnail", None)
            record.pop("_cacheable", None)
        return dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                    roots=[str(r) for r in roots],
                    options=dict(threshold=threshold, comparison="full" if full_images else "thumbnail",
                                 workers=workers, exiftool=bool(executable)),
                    cache_hits=cache.hits, records=records)
    finally:
        cache.close()
