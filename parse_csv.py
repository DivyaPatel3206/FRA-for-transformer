# parse_csv.py
import pandas as pd
import numpy as np

def parse_csv(path):
    """
    Generic CSV FRA parser.
    Tries to detect columns: frequency, magnitude (dB), phase (deg), real, imag.
    Returns canonical dict.
    """
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]

    # metadata hint extraction (if present)
    meta = {}
    for k in ['transformer','id','tap','operator','date']:
        for c in df.columns:
            if k in c.lower():
                meta[k] = str(df[c].iloc[0])
                break

    if 'frequency' not in cols:
        # assume first column is frequency
        freq = df.iloc[:, 0].to_numpy(dtype=float)
    else:
        freq = df.iloc[:, cols.index('frequency')].to_numpy(dtype=float)

    mag = None; phase = None
    if 'magnitude' in cols or 'magnitude_db' in cols or 'mag_db' in cols or 'mag' in cols:
        for name in ['magnitude','magnitude_db','mag_db','mag']:
            if name in cols:
                mag = df.iloc[:, cols.index(name)].to_numpy(dtype=float); break

    if mag is None and ('real' in cols and 'imag' in cols):
        real = df.iloc[:, cols.index('real')].to_numpy(dtype=float)
        imag = df.iloc[:, cols.index('imag')].to_numpy(dtype=float)
        complex_ = real + 1j*imag
        mag = 20 * np.log10(np.abs(complex_) + 1e-12)
        phase = np.angle(complex_, deg=True)
    else:
        if 'phase' in cols:
            phase = df.iloc[:, cols.index('phase')].to_numpy(dtype=float)

    if mag is None:
        raise ValueError("CSV: could not find magnitude or real/imag columns. Inspect file.")

    return {'metadata': meta, 'frequency': freq, 'magnitude_db': mag, 'phase_deg': phase}
