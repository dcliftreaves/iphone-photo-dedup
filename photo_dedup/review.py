"""Review choices are an overlay; the original evidence is immutable."""

from copy import deepcopy
import hashlib
from pathlib import Path


def record_id(record):
    return hashlib.sha256(record["path"].encode()).hexdigest()[:24]


def apply_decisions(manifest, decisions):
    result = deepcopy(manifest)
    groups = {}
    by_path = {r["path"]: r for r in result["records"]}
    for record in result["records"]:
        group = record.get("group_id")
        if group:
            groups.setdefault(group, []).append(record)
        record["export_selected"] = record["status"] not in {"exact_duplicate", "ignored", "error"}
    for group, decision in decisions.items():
        members = groups.get(group)
        if not members:
            raise ValueError("A review group no longer exists; review the latest scan")
        allowed = {record_id(r) for r in members}
        selected = set(decision["selected"])
        if not selected or not selected <= allowed:
            raise ValueError("Choose at least one existing file in each reviewed group")
        for record in members:
            record["export_selected"] = record_id(record) in selected
            record["review_decision"] = decision["action"]
    # Selecting a media file also preserves its sidecars and local Live Photo / RAW
    # companions. An unselected alternate cannot pull a whole group back in.
    queue = [r for r in result["records"] if r["export_selected"] and r.get("kind") != "sidecar"]
    visited = set()
    while queue:
        record = queue.pop()
        if record["path"] in visited:
            continue
        visited.add(record["path"])
        for related in record.get("related_files", []):
            other = by_path.get(related["path"])
            if other is None or other["status"] in {"ignored", "error"}:
                continue
            # Cross-directory repeated Live Photo IDs express a relationship, not
            # a requirement to export every duplicate component in every import.
            if Path(other["path"]).parent != Path(record["path"]).parent:
                continue
            if not other["export_selected"]:
                other["export_selected"] = True
                other["preserved_as_companion"] = True
                queue.append(other)
    # Editing metadata is preserved even when its local image is an omitted exact
    # copy. The export planner can place it beside a selected byte-identical image.
    result["review"] = dict(decisions=deepcopy(decisions), policy="selected-and-unresolved",
                            sources_unchanged=True)
    return result
