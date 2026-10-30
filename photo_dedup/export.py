"""Explicit, verified export; no deletion or in-place source modification."""

import json
import os
import errno
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

from .media import SUPPORTED, sha256
from .reports import write_reports
from .layout import DEFAULT_TEMPLATE, render_template, validate_template


def safe_component(text):
    value = re.sub(r"[^\w. -]", "_", str(text)).strip(". ")
    return value.encode("utf-8")[:180].decode("utf-8", "ignore") or "Unknown"


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or not isinstance(manifest.get("records"), list):
        raise ValueError("Unsupported or invalid manifest schema")
    if (not isinstance(manifest.get("roots"), list) or not manifest["roots"] or
            not all(isinstance(root, str) and Path(root).is_absolute() for root in manifest["roots"]) or
            not isinstance(manifest.get("options"), dict) or
            not isinstance(manifest["options"].get("exiftool"), bool) or
            not isinstance(manifest.get("cache_hits"), int)):
        raise ValueError("Invalid manifest scan settings")
    seen = set()
    for record in manifest["records"]:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Invalid manifest file record")
        source = Path(record["path"])
        if not source.is_absolute() or str(source) in seen:
            raise ValueError("Manifest paths must be absolute and unique")
        seen.add(str(source))
        if record["status"] not in {"keep", "exact_duplicate", "review", "sidecar", "ignored", "error"}:
            raise ValueError("Unknown file status in manifest")
        if record["status"] not in {"ignored", "error"} and not re.fullmatch(
                r"[0-9a-f]{64}", record.get("sha256", "")):
            raise ValueError("Missing or invalid SHA-256 in manifest; scan again")
        if record["status"] not in {"ignored", "error"} and record.get("extension") not in SUPPORTED:
            raise ValueError("Invalid media extension in manifest")
    return manifest


def plan_export(manifest):
    template = validate_template(manifest.get("layout", {}).get("template", DEFAULT_TEMPLATE))
    plan = {}
    used = set()
    by_path = {r["path"]: r for r in manifest["records"]}
    records = sorted(manifest["records"], key=lambda r: (r["status"] == "sidecar", r["path"]))
    for record in records:
        if record["status"] in {"ignored", "error"} or record.get("export_selected") is False:
            continue
        source = Path(record["path"])
        bucket = {"exact_duplicate": "duplicates/exact", "review": "review"}.get(
            record["status"], "curated")
        if record.get("review_decision") and record.get("export_selected"):
            bucket = "curated"
        relative = render_template(record, template)
        parent = Path(bucket) / relative.parent
        filename = relative.name
        if record["status"] == "sidecar":
            # Keep sidecars beside their local media, including collision-renamed media.
            local = [r for r in record.get("related_files", [])
                     if Path(r["path"]).parent == source.parent and r["path"] in plan]
            if not local:
                for relation in record.get("related_files", []):
                    original = by_path.get(relation["path"], {})
                    representative = original.get("matched_to")
                    if original.get("confidence") == "exact_bytes" and representative in plan:
                        local.append(dict(path=representative))
            if local:
                primary = min(local, key=lambda r: r["path"])
                companion = plan[primary["path"]]
                parent = companion.parent
                filename = companion.stem + record["extension"]
        target = parent / filename
        counter = 1
        while unicodedata.normalize("NFC", str(target)).casefold() in used:
            target = parent / f"{Path(filename).stem}__{counter}{Path(filename).suffix}"
            counter += 1
        used.add(unicodedata.normalize("NFC", str(target)).casefold())
        plan[record["path"]] = target
    return plan


def cleanup_records(manifest, plan):
    """Snapshot of what the successful export preserved, with byte-level evidence."""
    retained_hashes = {r.get('sha256') for r in manifest['records'] if r['path'] in plan}
    result = []
    for record in manifest['records']:
        if record['path'] in plan:
            disposition, reason = 'retained', 'Included in the new library.'
        elif record.get('sha256') and record['sha256'] in retained_hashes:
            disposition, reason = 'identical_retained', 'An identical byte-for-byte copy is included in the new library.'
        else:
            disposition = 'not_retained'
            reason = ('Not included. Deleting the source folder may lose this file or its distinct contents. '
                      + record.get('reason', ''))
        result.append(dict(path=record['path'], size=record.get('size', 0),
                           disposition=disposition, reason=reason, status=record['status']))
    return result


def verified_source(record):
    source = Path(record["path"])
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Source is missing or is now a symbolic link: {source}")
    stat = source.stat()
    if stat.st_size != record["size"] or sha256(source) != record["sha256"]:
        raise ValueError(f"Source changed since scan; scan again: {source}")
    return source


def export_manifest(manifest, destination, *, mode="hardlink", dry_run=False, progress=None,
                    check_cancel=lambda: None, staging_ready=lambda path: None):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Export destination already exists: {destination}")
    resolved = destination.resolve()
    for root in manifest["roots"]:
        root = Path(root).resolve()
        if root == resolved or root in resolved.parents or resolved in root.parents:
            raise ValueError("Export destination and input directories must not contain one another")
    plan = plan_export(manifest)
    if dry_run:
        # Verify sources and hard-link feasibility even for a preview.
        parent = destination.parent
        while not parent.exists():
            parent = parent.parent
        for record in manifest["records"]:
            check_cancel()
            if record["path"] in plan:
                source = verified_source(record)
                if mode == "hardlink" and source.stat().st_dev != parent.stat().st_dev:
                    raise ValueError("Hard links require the same filesystem; use --mode copy")
        return plan
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        marker = staging / '.photo-dedup-staging.json'
        marker.write_text(json.dumps(dict(schema_version=1, destination=str(destination.resolve()))))
        staging_ready(staging)
        for record in manifest["records"]:
            check_cancel()
            relative = plan.get(record["path"])
            if relative is None:
                continue
            source = verified_source(record)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "hardlink":
                if source.stat().st_dev != staging.stat().st_dev:
                    raise ValueError("Hard links require the same filesystem; use --mode copy")
                try:
                    os.link(source, target, follow_symlinks=False)
                except OSError as error:
                    if error.errno in {errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV}:
                        raise ValueError('The output filesystem could not create this hard link. '
                                         'Put imports and output on one hard-link-capable volume, '
                                         'or explicitly choose independent copies. Source contents were not changed.') from error
                    raise
            else:
                shutil.copy2(source, target, follow_symlinks=False)
            if target.is_symlink() or sha256(target) != record["sha256"]:
                raise ValueError(f"Source changed during export: {source}")
            record["export_path"] = str(relative)
            if progress:
                progress(record["path"])
        manifest["cleanup_review"] = dict(roots=manifest["roots"], records=cleanup_records(manifest, plan),
                                        scope="Saved scan evidence; files added or changed later are not covered. Ignored directories may contain uncounted files.")
        manifest["export"] = dict(mode=mode, destination=str(destination))
        write_reports(manifest, staging)
        if destination.exists() or destination.is_symlink():
            raise ValueError("Export destination appeared during export; refusing to overwrite")
        check_cancel()
        marker.unlink()
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return plan
