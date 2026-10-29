"""Deterministic decisions with a direct evidence link for every match."""

import base64
from collections import defaultdict
import hashlib
from pathlib import Path
import re

import numpy as np

from .media import THUMB_SIZE, full_image_rmse, rmse

PREFERENCE = {ext: index for index, ext in enumerate(
    [".dng", ".heif", ".heic", ".tiff", ".tif", ".png", ".jpg", ".jpeg"])}


def preference(record):
    path = Path(record["path"])
    return (PREFERENCE.get(record.get("extension"), 99), len(path.stem),
            record.get("mtime_ns", 0), str(path).casefold(), str(path))


def group_id(path):
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def name_key(path):
    stem = Path(path).stem.casefold()
    camera = re.fullmatch(r"(img_e?\d{4}|dsc_?\d{4}|[a-z]{4}\d{4})"
                          r"(?:[ _-]\d+|\s*\(\d+\))?", stem)
    if camera:
        return camera.group(1)
    return re.sub(r"(?:[ -]\d+|_\d{1,3}|\s*\(\d+\))$", "", stem).strip()


def candidate_key(record):
    if record.get("capture_key"):
        identity = ("capture", record["capture_key"], record.get("model"))
    elif record.get("thumbnail"):
        identity = ("name", name_key(record["path"]))
    else:
        return None
    return (*identity, record.get("kind"), record.get("width"), record.get("height"))


def compare(left, right, full_images):
    if full_images:
        return full_image_rmse(left["path"], right["path"])
    shape = (*THUMB_SIZE, 4)
    a = np.frombuffer(base64.b64decode(left["thumbnail"]), dtype=np.uint8).reshape(shape)
    b = np.frombuffer(base64.b64decode(right["thumbnail"]), dtype=np.uint8).reshape(shape)
    return rmse(a, b)


def classify(records, threshold=0.02, full_images=False, check_cancel=lambda: None):
    exact = {}
    aliases = defaultdict(set)
    candidates = defaultdict(list)
    ordered = sorted(records, key=preference)
    # Complete byte grouping first, so renamed aliases work regardless of traversal order.
    for record in ordered:
        check_cancel()
        if record["status"] in ("error", "ignored"):
            continue
        record.update(status="keep", confidence="unique", matched_to=None,
                      reason="No duplicate was confirmed.", group_id=group_id(record["path"]))
        if record["kind"] == "sidecar":
            record.update(status="sidecar", confidence="preserved",
                          reason="Preserved editing metadata; never suppressed independently.")
            continue
        keeper = exact.get(record["sha256"])
        alias = candidate_key(record)
        if alias is not None:
            aliases[record["sha256"]].add(alias)
        if keeper:
            record.update(status="exact_duplicate", confidence="exact_bytes",
                          matched_to=keeper["path"], group_id=keeper["group_id"],
                          reason="SHA-256 matches: file contents are byte-for-byte identical.")
            continue
        exact[record["sha256"]] = record
    for record in ordered:
        check_cancel()
        if record["status"] != "keep":
            continue
        keys = aliases[record["sha256"]]
        options = {r["path"]: r for key in keys for r in candidates[key]}
        for keeper in sorted(options.values(), key=preference):
            if record.get("thumbnail") and keeper.get("thumbnail"):
                if record.get("multi_frame") or keeper.get("multi_frame"):
                    continue
                try:
                    score = compare(record, keeper, full_images)
                except (OSError, ValueError) as error:
                    record["warnings"].append(f"Image comparison unavailable: {error}")
                    continue
                if score > threshold:
                    continue
                record.update(status="review", confidence="visual_similarity", score=score,
                              matched_to=keeper["path"], group_id=keeper["group_id"],
                              reason=f"Similar normalized images (RMSE {score:.5f}, limit {threshold:.5f}); "
                                     "review edits, detail, and metadata before choosing.")
                break
            elif record.get("capture_key"):
                # Video duration disagreement rules out a candidate. Codec preference is
                # deliberately not evidence of equal quality or identical content.
                if record["kind"] == "video":
                    a = record["metadata"].get("Duration")
                    b = keeper["metadata"].get("Duration")
                    if a is None or b is None or a != b:
                        continue
                record.update(status="review", confidence="metadata_only",
                              matched_to=keeper["path"], group_id=keeper["group_id"],
                              reason="Capture metadata agrees; media content has not been verified.")
                break
        # Compare against representatives directly, never through a similarity chain.
        if record["status"] == "keep":
            for key in keys:
                candidates[key].append(record)
    for record in ordered:
        if record["status"] == "exact_duplicate":
            record["group_id"] = exact[record["sha256"]]["group_id"]
    associate(records)
    return sorted(records, key=lambda r: r["path"])


def associate(records):
    """Keep Live Photo pairs and sidecars as relationships, never duplicates."""
    by_stem = defaultdict(list)
    by_content = defaultdict(list)
    for record in records:
        if record["status"] in ("error", "ignored"):
            continue
        path = Path(record["path"])
        by_stem[(str(path.parent), path.stem.casefold())].append(record)
        if record.get("content_id"):
            by_content[str(record["content_id"])].append(record)
    for record in records:
        path = Path(record["path"])
        companions = by_stem[(str(path.parent), path.stem.casefold())]
        content = by_content.get(str(record.get("content_id")), [])
        relations = {}
        for other in companions + content:
            if other is record:
                continue
            kinds = {record.get("kind"), other.get("kind")}
            if "sidecar" in kinds:
                relation = "sidecar"
            elif "raw" in kinds and "image" in kinds:
                relation = "raw_jpeg_pair"
            elif "video" in kinds and "image" in kinds:
                relation = "live_photo" if content and other in content else "possible_live_photo"
            else:
                continue
            relations[other["path"]] = dict(path=other["path"], relationship=relation)
        record["related_files"] = sorted(relations.values(), key=lambda r: r["path"])
