"""Deterministic decisions with a direct evidence link for every match."""

import base64
from collections import defaultdict
import hashlib
from pathlib import Path
import re

import numpy as np

from .media import THUMB_SIZE, full_image_rmse, rmse
from .raw import rawpy

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
        return full_image_rmse(left["path"], right["path"],
                               left_raw=left.get("kind") == "raw", right_raw=right.get("kind") == "raw")
    shape = (*THUMB_SIZE, 4)
    a = np.frombuffer(base64.b64decode(left["thumbnail"]), dtype=np.uint8).reshape(shape)
    b = np.frombuffer(base64.b64decode(right["thumbnail"]), dtype=np.uint8).reshape(shape)
    return rmse(a, b)


def classify(records, threshold=0.02, full_images=False, check_cancel=lambda: None, deep_scan=False):
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
    # RAW samples can survive metadata rewrites with a different file hash.
    raw_samples = {}
    for record in ordered:
        check_cancel()
        if record["status"] != "keep" or not record.get("raw_samples_sha256"):
            continue
        keeper = raw_samples.get(record["raw_samples_sha256"])
        if keeper:
            record.update(status="review", confidence="raw_samples", matched_to=keeper["path"],
                          group_id=keeper["group_id"], reason="Decoded active RAW sample arrays and CFA layout match. "
                          "File bytes, metadata, white balance and rendering instructions may differ. Review before choosing.")
        else:
            raw_samples[record["raw_samples_sha256"]] = record
    for record in ordered:
        check_cancel()
        if record["status"] != "keep":
            continue
        keys = aliases[record["sha256"]]
        options = {r["path"]: r for key in keys for r in candidates[key]}
        for keeper in sorted(options.values(), key=preference):
            if record.get("raw_samples_sha256") and record.get("raw_samples_sha256") == keeper.get("raw_samples_sha256"):
                record.update(status="review", confidence="raw_samples", matched_to=keeper["path"],
                              group_id=keeper["group_id"], reason="Decoded active RAW samples and CFA layout match; file bytes, metadata and rendering instructions may differ.")
                break
            if record.get("thumbnail") and keeper.get("thumbnail"):
                if record.get("multi_frame") or keeper.get("multi_frame"):
                    continue
                try:
                    score = compare(record, keeper, full_images)
                except (OSError, ValueError, rawpy.LibRawError) as error:
                    record["warnings"].append(f"Image comparison unavailable: {error}")
                    continue
                if score > threshold:
                    continue
                record.update(status="review", confidence="raw_render_similarity" if record["kind"] == "raw" else "visual_similarity", score=score,
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
    if deep_scan:
        deep_compare(ordered, threshold, full_images, check_cancel)
    # Earlier matches can point to a representative later grouped by RAW/deep evidence.
    by_path = {r["path"]: r for r in ordered}
    for record in ordered:
        current, visited = record, set()
        while current.get("matched_to") and current["path"] not in visited:
            visited.add(current["path"])
            current = by_path[current["matched_to"]]
        record["group_id"] = current.get("group_id")
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


def deep_compare(ordered, threshold, full_images, check_cancel):
    """Bounded perceptual index, followed by pixel comparison; photos only."""
    index = defaultdict(list)
    protected = {r.get("matched_to") for r in ordered if r.get("matched_to")}
    for record in ordered:
        check_cancel()
        if record["status"] != "keep" or not record.get("thumbnail") or record.get("multi_frame"):
            continue
        fingerprint = record.get("visual_hash")
        if fingerprint is None:
            continue
        keys = [(part, (fingerprint >> (part * 16)) & 65535) for part in range(4)]
        options = {r["path"]: r for key in keys for r in index[key]}
        # Bound work for flat/low-detail images and enormous repeated import sets.
        if len(options) > 200:
            record["warnings"].append("Deep visual candidate limit reached; this pass may miss matches. Original retained.")
        for keeper in ([] if record["path"] in protected else sorted(options.values(), key=preference)[:200]):
            check_cancel()
            if (fingerprint ^ keeper["visual_hash"]).bit_count() > 3:
                continue
            # Do not compare incompatible aspect ratios or known burst subseconds.
            width = record.get("render_width") or record.get("width", 0)
            height = record.get("render_height") or record.get("height", 0)
            kw = keeper.get("render_width") or keeper.get("width", 0)
            kh = keeper.get("render_height") or keeper.get("height", 0)
            if not height or not kh or abs(width / height - kw / kh) > .01:
                continue
            a, b = record.get("metadata", {}), keeper.get("metadata", {})
            if a.get("SubSecTimeOriginal") and b.get("SubSecTimeOriginal") and a["SubSecTimeOriginal"] != b["SubSecTimeOriginal"]:
                continue
            try:
                score = compare(record, keeper, full_images)
            except (OSError, ValueError, rawpy.LibRawError) as error:
                record["warnings"].append(f"Deep comparison unavailable: {error}")
                continue
            if score <= threshold:
                record.update(status="review", confidence="deep_visual_similarity", score=score,
                              matched_to=keeper["path"], group_id=keeper["group_id"],
                              reason=f"Content-indexed image similarity (RMSE {score:.5f}); names and capture dates were not required. "
                                     "Rendered appearance is not proof of identical originals. Review detail, edits and metadata.")
                break
        if record["status"] == "keep":
            for key in keys:
                # Only retain a bounded set per bucket; already matched images never become representatives.
                if len(index[key]) < 201:
                    index[key].append(record)
