"""Small synthetic RAW fixture for decoder and packaged-runtime verification."""
import struct
import numpy as np


def make_dng(path, description='original', change=False):
    """Tiny uncompressed Bayer DNG; synthetic sensor samples, no external fixture."""
    width = height = 64
    samples = np.arange(width * height, dtype=np.uint16).reshape(height, width) % 3000 + 128
    if change:
        samples[20:40, 20:40] += 400
    tags = []
    def tag(number, kind, count, data):
        tags.append((number, kind, count, data))
    short = lambda *v: struct.pack('<' + 'H' * len(v), *v)
    long = lambda *v: struct.pack('<' + 'I' * len(v), *v)
    text = lambda v: v.encode() + b'\0'
    for number, value in [(256,width),(257,height),(278,height),(279,samples.nbytes)]:
        tag(number,4,1,long(value))
    for number,value in [(258,16),(259,1),(262,32803),(277,1),(284,1),(50717,4095),(50778,21)]:
        tag(number,3,1,short(value))
    tag(270,2,len(text(description)),text(description))
    tag(271,2,5,b'Test\0')
    tag(272,2,7,b'Camera\0')
    tag(33421,3,2,short(2,2))
    tag(33422,1,4,bytes([0,1,1,2]))
    tag(50706,1,4,bytes([1,4,0,0]))
    tag(50707,1,4,bytes([1,3,0,0]))
    tag(50708,2,17,b'Synthetic Camera\0')
    tag(50710,1,3,bytes([0,1,2]))
    tag(50711,3,1,short(1))
    tag(50714,3,1,short(64))
    matrix = [1,0,0,0,1,0,0,0,1]
    tag(50721,10,9,b''.join(struct.pack('<ii',v,1) for v in matrix))
    tag(50728,5,3,b''.join(long(1,1) for _ in range(3)))
    tag(273,4,1,long(0))
    tags.sort()
    offset = 8 + 2 + len(tags)*12 + 4
    extra = bytearray()
    entries = []
    for number,kind,count,data in tags:
        if len(data) <= 4:
            entries.append([number,kind,count,data.ljust(4,b'\0')])
        else:
            entries.append([number,kind,count,long(offset+len(extra))])
            extra.extend(data)
            if len(extra)%2: extra.append(0)
    pixel_offset = offset+len(extra)
    for entry in entries:
        if entry[0] == 273: entry[3] = long(pixel_offset)
    header = b'II*\0'+long(8)+short(len(tags))
    path.write_bytes(header+b''.join(struct.pack('<HHI',*e[:3])+e[3] for e in entries)+long(0)+extra+samples.astype('<u2').tobytes())

