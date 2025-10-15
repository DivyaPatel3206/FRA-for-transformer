# preprocessing.py
import numpy as np
from scipy import interpolate, signal

def resample_to_log_grid(freq, mag_db, n_points=1024, fmin=None, fmax=None):
    freq = np.asarray(freq)
    mag_db = np.asarray(mag_db)
    if fmin is None: fmin = max(freq.min(), 1e-3)
    if fmax is None: fmax = freq.max()
    grid = np.logspace(np.log10(fmin), np.log10(fmax), num=n_points)
    interp = interpolate.interp1d(freq, mag_db, kind='linear', bounds_error=False, fill_value='extrapolate')
    mag = interp(grid)
    return grid, mag

def normalize(z):
    z = np.asarray(z, dtype=float)
    mean = np.mean(z); std = np.std(z) + 1e-12
    return (z - mean) / std

def denoise(mag_db, method='median'):
    if method == 'median':
        return signal.medfilt(mag_db, kernel_size=5)
    return mag_db
