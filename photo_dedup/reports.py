"""Portable scan results for people, scripts, and a future local review UI."""

from collections import Counter
import csv
import json
from pathlib import Path

LABELS = {"keep": "Keep", "exact_duplicate": "Exact duplicate", "review": "Needs review",
          "sidecar": "Preserved sidecar", "ignored": "Ignored", "error": "Error"}


def summary(manifest):
    records = manifest["records"]
    counts = Counter(r["status"] for r in records)
    lines = ["Photo library scan", "==================", f"Input entries: {len(records)}"]
    lines.extend(f"{label}: {counts[status]}" for status, label in LABELS.items())
    if "review" in manifest:
        selected = sum(r.get("export_selected", False) for r in records)
        omitted = sum(r.get("export_selected") is False and r["status"] not in {"ignored", "error"}
                      for r in records)
        lines += [f"Selected for this export: {selected}",
                  f"Unselected alternatives (may include identical retained copies): {omitted}",
                  f"Review groups with saved choices: {len(manifest['review']['decisions'])}"]
    # Hard-linked aliases do not represent additional occupied file data.
    kept_inodes = {(r.get("device"), r.get("inode")) for r in records
                   if r["status"] != "exact_duplicate"}
    reclaimable = {}
    for record in records:
        inode = (record.get("device"), record.get("inode"))
        if record["status"] == "exact_duplicate" and inode not in kept_inodes:
            reclaimable[inode] = record["size"]
    lines += [f"Potential redundant bytes (exact matches only): {sum(reclaimable.values()):,}",
              "This estimate excludes other hard links and filesystem compression or clones.",
              f"Cached inspections reused: {manifest['cache_hits']}",
              "", "Exact duplicate = identical file bytes (SHA-256).",
              "Needs review = visual similarity or metadata agreement; keep both until reviewed.",
              "Sidecars and Live Photo / RAW companions are preserved unless a reviewed RAW/image alternative was explicitly omitted.",
              "No source files were modified or deleted."]
    if not manifest["options"]["exiftool"]:
        lines += ["", "ExifTool was not found. Video/RAW metadata and Live Photo identification",
                  "are limited. Run photo-dedup doctor for installation guidance."]
    warnings = sum(bool(r.get("warnings")) for r in records)
    lines += [f"", f"Files with warnings: {warnings}"]
    return "\n".join(lines) + "\n"


def csv_cell(value):
    text = str(value) if value is not None else ""
    return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else text


def write_reports(manifest, directory):
    directory = Path(directory)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    (directory / "summary.txt").write_text(summary(manifest))
    cleanup = manifest.get("cleanup_review")
    if cleanup:
        with (directory / "source-cleanup.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "disposition", "size", "status", "reason"])
            writer.writeheader()
            writer.writerows({key: csv_cell(value) for key, value in row.items()} for row in cleanup["records"])
    columns = ["path", "status", "confidence", "reason", "matched_to", "group_id", "size",
               "captured_at", "kind", "score", "related_files", "warnings", "export_selected", "export_path"]
    with (directory / "files.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in manifest["records"]:
            row = {key: record.get(key, "") for key in columns}
            row["related_files"] = "; ".join(
                f"{r['relationship']}: {r['path']}" for r in record.get("related_files", []))
            row["warnings"] = "; ".join(record.get("warnings", []))
            writer.writerow({key: csv_cell(value) for key, value in row.items()})
