# synthetic_data.py
import numpy as np
import json
from math import sin
import random

def generate_signature(n_points=1024, fault=None):
    # log freq
    freq = np.logspace(0, 4, n_points)
    base = np.sin(np.log(freq))*5  # synthetic baseline wiggle
    # add resonant peaks
    for p in [50, 300, 1200]:
        base += 6 * np.exp(-0.5*((np.log(freq)-np.log(p))/0.6)**2)
    if fault == 'axial':
        base += 10 * np.exp(-0.5*((np.log(freq)-np.log(80))/0.4)**2)
    if fault == 'radial':
        base += 8 * np.exp(-0.5*((np.log(freq)-np.log(1200))/0.3)**2)
    if fault == 'core_ground':
        base += np.linspace(0, -12, n_points)
    # noise
    base += np.random.normal(0, 0.5, size=n_points)
    mag_db = base
    return {'frequency': freq, 'mag': mag_db}

def create_dataset(n_per_class=50, out='synthetic.npy'):
    labels = ['no_fault','axial','radial','core_ground']
    data = []
    for i,label in enumerate(labels):
        for _ in range(n_per_class):
            f = generate_signature(1024, fault=(label if label!='no_fault' else None))
            data.append({'frequency': f['frequency'], 'mag': f['mag'], 'label': i})
    np.save(out, data)
    print("Saved", out)

if __name__ == "__main__":
    create_dataset(20, out='synthetic.npy')
