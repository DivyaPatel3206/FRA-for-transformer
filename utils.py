# utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_signal(freq, mag_db, title="FRA"):
    plt.figure(figsize=(9,4))
    plt.semilogx(freq, mag_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True, which='both', ls='--')
    plt.show()

def save_hdf5(path, parsed):
    import h5py
    with h5py.File(path,'w') as f:
        md = f.create_group('metadata')
        for k,v in (parsed.get('metadata') or {}).items():
            md.attrs[k] = str(v)
        f.create_dataset('frequency', data=parsed['frequency'])
        f.create_dataset('magnitude_db', data=parsed['magnitude_db'])
        if parsed.get('phase_deg') is not None:
            f.create_dataset('phase_deg', data=parsed['phase_deg'])
