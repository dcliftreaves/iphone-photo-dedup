"""CLI is a thin client of the scan / manifest / export workflow."""

import argparse
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import shutil

from . import __version__
from .export import export_manifest, load_manifest
from .media import SUPPORTED, exiftool_path
from .reports import summary, write_reports
from .scanner import scan


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def threshold_value(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def parser():
    result = argparse.ArgumentParser(
        prog="photo-dedup", description="Review photo and video duplicates locally. Sources are never deleted.")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    scan_cmd = commands.add_parser("scan", help="Scan inputs and write an explainable review report")
    scan_cmd.add_argument("inputs", nargs="+", type=Path, help="One or more input directories")
    scan_cmd.add_argument("-o", "--output", type=Path, default=Path("results"), help="New report directory (default: results)")
    scan_cmd.add_argument("--cache", type=Path, default=Path(".dedup-cache.sqlite3"), help="SQLite inspection cache")
    scan_cmd.add_argument("--no-cache", action="store_true", help="Do not read or write the cache")
    scan_cmd.add_argument("--dry-run", action="store_true", help="Compute and print summary without writing any files")
    scan_cmd.add_argument("--workers", type=positive_int, default=min(4, os.cpu_count() or 1), help="Concurrent inspection workers (default: up to 4; use 1 for HDDs)")
    scan_cmd.add_argument("--threshold", type=threshold_value, default=0.02, help="Visual RMSE limit, 0–1 (default: 0.02); all visual matches need review")
    scan_cmd.add_argument("--full-images", action="store_true", help="Compare all normalized pixels instead of 128px thumbnails; uses more memory")
    scan_cmd.add_argument("--exclude", action="append", type=Path, default=[], help="Exclude a file or directory (repeatable)")
    scan_cmd.add_argument("--quiet", action="store_true", help="Hide progress")
    export_cmd = commands.add_parser("export", help="Export a saved scan into a new curated/review directory")
    export_cmd.add_argument("manifest", type=Path)
    export_cmd.add_argument("-o", "--output", type=Path, required=True, help="New destination outside input directories")
    export_cmd.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    export_cmd.add_argument("--dry-run", action="store_true", help="Verify and preview export without writing")
    commands.add_parser("doctor", help="Check Python, image decoders, and optional ExifTool")
    ui = commands.add_parser("ui", help="Open the local graphical review application")
    ui.add_argument("--data-dir", type=Path, default=Path(".photo-dedup"))
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-open", action="store_true")
    return result


def run_scan(args):
    roots = sorted({p.expanduser().resolve() for p in args.inputs})
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Not a valid input directory: {root}")
    output = args.output.expanduser().resolve()
    if not args.dry_run and output.exists():
        raise ValueError(f"Report directory already exists: {output}. Choose a new --output.")
    if any(output == root or output in root.parents for root in roots):
        raise ValueError("Report directory cannot contain an input directory")
    excludes = [p.expanduser().resolve() for p in args.exclude] + [output]
    requested_cache = args.cache.expanduser().absolute()
    if requested_cache.is_symlink():
        raise ValueError("Cache path must not be a symbolic link")
    cache_path = None if args.no_cache or args.dry_run else requested_cache.resolve()
    if cache_path:
        if cache_path.exists() and not cache_path.is_file():
            raise ValueError("Cache path is not a regular file")
        if cache_path.suffix.lower() in SUPPORTED:
            raise ValueError("Cache path must not have a media-file extension")
        if cache_path == output or output in cache_path.parents:
            raise ValueError("Cache must be outside the report directory")
        excludes += [cache_path, Path(str(cache_path) + "-journal"),
                     Path(str(cache_path) + "-wal"), Path(str(cache_path) + "-shm")]
    last_percent = -1

    def progress(done, total, hits):
        nonlocal last_percent
        percent = done * 100 // max(total, 1)
        if not args.quiet and percent != last_percent:
            print(f"\rInspected {done:,}/{total:,} entries · {hits:,} cached", end="", file=sys.stderr, flush=True)
            last_percent = percent

    manifest = scan(roots, cache_path=cache_path, excludes=excludes, workers=args.workers,
                    threshold=args.threshold, full_images=args.full_images, progress=progress)
    if not args.quiet:
        print("\nMatching complete.", file=sys.stderr)
    print(summary(manifest), end="")
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            write_reports(manifest, staging)
            if output.exists():
                raise ValueError("Report directory appeared during scan; refusing to overwrite")
            staging.rename(output)
        except BaseException:
            shutil.rmtree(staging)
            raise
        print(f"\nReports: {output}\nRead summary.txt, filter files.csv, or use manifest.json.")
    return 1 if any(r["status"] == "error" or r.get("warnings") for r in manifest["records"]) else 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "ui":
            from .web import main as web_main
            return web_main(["--data-dir", str(args.data_dir), "--port", str(args.port)] +
                            (["--no-open"] if args.no_open else []))
        if args.command == "scan":
            return run_scan(args)
        if args.command == "export":
            manifest = load_manifest(args.manifest)
            plan = export_manifest(manifest, args.output.expanduser(), mode=args.mode, dry_run=args.dry_run)
            if args.dry_run:
                for source, destination in plan.items():
                    print(f"{source} -> {destination}")
                print(f"Preview: {len(plan):,} files; no files written.")
            else:
                print(f"Exported {len(plan):,} files to {args.output.absolute()}")
                if args.mode == "hardlink":
                    print("Hard links share contents with sources. Editing either changes both.")
            return 0
        from PIL import __version__ as pillow_version
        import pillow_heif
        print(f"photo-dedup {__version__}\nPython: {sys.version.split()[0]}\n"
              f"Pillow: {pillow_version}\nHEIF/HEIC: pillow-heif {pillow_heif.__version__}\n"
              f"ExifTool: {exiftool_path() or 'not installed (optional; recommended)'}")
        if not exiftool_path():
            print("Install ExifTool on macOS: brew install exiftool\n"
                  "Without it, images use Pillow metadata; video/RAW matching and Live Photo IDs are limited.")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled. Source files were not modified.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
