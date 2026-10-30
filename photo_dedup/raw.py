"""LibRaw sample evidence and deterministic renders; never use embedded previews."""
import hashlib
import json
import threading

import numpy as np
from PIL import Image
import rawpy

# RAW unpacking can use substantial memory. Share this bound with review previews.
_DECODE = threading.Semaphore(1)
SIGNATURE = f'rawpy:{rawpy.__version__}|LibRaw:{rawpy.libraw_version}|camera-wb-srgb-no-auto-bright-v1'


def _render(raw, half_size):
    pixels = raw.postprocess(half_size=half_size, use_camera_wb=True, use_auto_wb=False,
                             no_auto_bright=True, output_color=rawpy.ColorSpace.sRGB, output_bps=8)
    return Image.fromarray(pixels).convert('RGBA')


def inspect_raw(path):
    with _DECODE, rawpy.imread(str(path)) as raw:
        samples = raw.raw_image_visible
        digest = hashlib.sha256()
        digest.update(json.dumps(dict(shape=samples.shape, dtype=str(samples.dtype),
                                      pattern=raw.raw_pattern.tolist() if raw.raw_pattern is not None else None,
                                      colors=raw.color_desc.decode('ascii', 'replace')),
                                 sort_keys=True).encode())
        digest.update(np.ascontiguousarray(samples).tobytes())
        return digest.hexdigest(), _render(raw, True)


def render_raw(path, half_size=True):
    with _DECODE, rawpy.imread(str(path)) as raw:
        return _render(raw, half_size)
