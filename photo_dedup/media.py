"""Read-only file inspection and normalized image comparison."""

import base64
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps
from pillow_heif import register_heif_opener

from .raw import SIGNATURE as RAW_SIGNATURE, inspect_raw, render_raw

register_heif_opener(thumbnails=False, decode_threads=1)

IMAGES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"}
RAW = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2"}
VIDEOS = {".mov", ".mp4", ".m4v", ".avi", ".mts"}
SIDECARS = {".aae", ".xmp"}
SUPPORTED = IMAGES | RAW | VIDEOS | SIDECARS
THUMB_SIZE = (128, 128)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_image(image):
    """Orient, convert embedded color to sRGB, and retain alpha for comparison."""
    image = ImageOps.exif_transpose(image)
    alpha = image.convert("RGBA").getchannel("A")
    profile = image.info.get("icc_profile")
    if profile:
        source = image if image.mode in ("RGB", "CMYK", "LAB") else image.convert("RGB")
        image = ImageCms.profileToProfile(source, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                         ImageCms.createProfile("sRGB"), outputMode="RGB")
    else:
        image = image.convert("RGB")
    result = image.convert("RGBA")
    result.putalpha(alpha)
    return result


def rmse(left, right):
    """Normalized [0,1] error; floating point prevents unsigned overflow."""
    a = np.asarray(left, dtype=np.float32) / 255.0
    b = np.asarray(right, dtype=np.float32) / 255.0
    if a.shape != b.shape:
        return 1.0
    return float(np.sqrt(np.mean(np.square(a - b))))


def open_normalized(path, *, half_size=False, raw=None):
    if raw if raw is not None else Path(path).suffix.lower() in RAW:
        return render_raw(path, half_size=half_size)
    with Image.open(path) as opened:
        return normalized_image(opened)


def full_image_rmse(left, right, *, left_raw=None, right_raw=None):
    return rmse(open_normalized(left, raw=left_raw), open_normalized(right, raw=right_raw))


def visual_hash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes(), "big")


def metadata(path, executable):
    if not executable:
        return {}
    result = subprocess.run(
        [executable, "-j", "-n", "-FileTypeExtension", "-Model", "-DateTimeOriginal",
         "-SubSecTimeOriginal", "-OffsetTimeOriginal", "-CreateDate", "-CreationDate",
         "-ImageWidth", "-ImageHeight", "-Duration", "-CompressorName",
         "-ContentIdentifier", "-MediaGroupUUID", "-Warning", "-Error", str(path)],
        capture_output=True, text=True, timeout=60, check=False)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "ExifTool could not read this file")
    rows = json.loads(result.stdout)
    return rows[0] if rows else {}


def inspect(path: Path, executable=None):
    stat = path.stat()
    ext = path.suffix.lower()
    record = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                  ctime_ns=stat.st_ctime_ns, device=stat.st_dev, inode=stat.st_ino,
                  sha256=sha256(path), extension=ext, warnings=[], _cacheable=True,
                  kind="sidecar" if ext in SIDECARS else "image", metadata={})
    try:
        tags = metadata(path, executable)
        record["metadata"] = tags
        if tags.get("Error"):
            record["warnings"].append(str(tags["Error"]))
        if tags.get("Warning"):
            record["warnings"].append(str(tags["Warning"]))
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        record["_cacheable"] = False
        record["warnings"].append(f"Metadata unavailable: {error}")
        tags = {}
    detected = "." + str(tags.get("FileTypeExtension", "")).lower()
    if detected in SUPPORTED:
        record["extension"] = detected
        ext = detected
    if ext in VIDEOS:
        record["kind"] = "video"
    elif ext in RAW:
        record["kind"] = "raw"
    elif ext in SIDECARS:
        record["kind"] = "sidecar"
    if ext in IMAGES:
        try:
            with Image.open(path) as opened:
                # Pillow also detects misnamed supported image files without ExifTool.
                actual = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tiff",
                          "HEIF": ".heic", "WEBP": ".webp"}.get(opened.format)
                if actual:
                    record["extension"] = actual
                exif = opened.getexif()
                try:
                    exif_tags = exif.get_ifd(0x8769)
                except (KeyError, TypeError):
                    exif_tags = {}
                for name, key in [("DateTimeOriginal", 36867), ("SubSecTimeOriginal", 37521),
                                  ("OffsetTimeOriginal", 36881), ("Model", 272)]:
                    value = exif_tags.get(key, exif.get(key))
                    if value is not None:
                        tags.setdefault(name, str(value))
                image = normalized_image(opened)
                record["width"], record["height"] = image.size
                record["multi_frame"] = getattr(opened, "n_frames", 1) > 1
                thumb = image.resize(THUMB_SIZE, Image.Resampling.LANCZOS)
                record["thumbnail"] = base64.b64encode(thumb.tobytes()).decode("ascii")
                record["visual_hash"] = visual_hash(image)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            record["_cacheable"] = False
            record["warnings"].append(f"Image comparison unavailable: {error}")
    if record["kind"] == "raw":
        try:
            sample_hash, image = inspect_raw(path)
            record["raw_samples_sha256"] = sample_hash
            record["raw_decoder"] = RAW_SIGNATURE
            record["render_width"], record["render_height"] = image.size
            record["thumbnail"] = base64.b64encode(image.resize(THUMB_SIZE, Image.Resampling.LANCZOS).tobytes()).decode("ascii")
            record["visual_hash"] = visual_hash(image)
        except Exception as error:
            record["_cacheable"] = False
            record["warnings"].append(f"RAW decoding unavailable; original retained: {error}")
    record["metadata"] = tags
    record.setdefault("width", tags.get("ImageWidth", 0))
    record.setdefault("height", tags.get("ImageHeight", 0))
    record["model"] = str(tags.get("Model", "Unknown"))
    date = tags.get("DateTimeOriginal") or tags.get("CreationDate") or tags.get("CreateDate")
    record["captured_at"] = str(date) if date else None
    record["capture_key"] = (f"{date}|{tags.get('SubSecTimeOriginal', '')}|"
                             f"{tags.get('OffsetTimeOriginal', '')}") if date else None
    record["content_id"] = tags.get("ContentIdentifier") or tags.get("MediaGroupUUID")
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
        raise ValueError("File changed during inspection; scan again")
    return record


def engine_signature(executable):
    from PIL import __version__ as pillow_version
    import pillow_heif
    version = "none"
    if executable:
        version = subprocess.check_output([executable, "-ver"], text=True, timeout=10).strip()
    return f"inspection-v3|{RAW_SIGNATURE}|Pillow:{pillow_version}|heif:{pillow_heif.__version__}|exiftool:{version}"


def exiftool_path():
    bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "tools" / "exiftool"
    executable = bundled / ("exiftool.exe" if sys.platform == "win32" else "exiftool")
    if executable.is_file():
        return str(executable)
    return (shutil.which("exiftool") or
            shutil.which("exiftool", path=str(Path(sys.executable).parent)))
