# parse_vendor_binary.py
import numpy as np
import struct
import os

def parse_vendor_binary(path):
    """
    Vendor binary parsing is vendor-specific. This is a safe fallback that tries common float patterns.
    Prefer implementing vendor SDK parser when available.
    """
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        raw = f.read()

    # try interpreting as float32 sequence: freq, mag, freq, mag...
    try:
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.size % 2 == 0 and arr.size >= 4:
            freq = arr[::2]
            mag = arr[1::2]
            return {'metadata': {}, 'frequency': freq, 'magnitude_db': mag, 'phase_deg': None}
    except Exception:
        pass

    raise NotImplementedError("Vendor binary parser: schema unknown. Use vendor SDK or reverse-engineer format.")
