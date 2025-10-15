# parse_xml.py
import xmltodict
import numpy as np

def _to_array(x):
    if x is None:
        return None
    if isinstance(x, list):
        return np.array([float(i) for i in x], dtype=float)
    if isinstance(x, str):
        return np.array([float(i) for i in x.strip().split()], dtype=float)
    if isinstance(x, dict) and '#text' in x:
        return np.array([float(i) for i in x['#text'].split()], dtype=float)
    # try numeric children
    try:
        return np.array([float(v) for v in x.values()], dtype=float)
    except Exception:
        raise ValueError("Unsupported XML numeric format")

def parse_xml(path):
    """
    Generic XML FRA parser. Vendor schemas vary — adapt as needed.
    """
    with open(path, 'r') as f:
        doc = xmltodict.parse(f.read())

    # search heuristics
    def find_key(d, candidates):
        if not isinstance(d, dict):
            return None
        for k in d.keys():
            if any(c.lower() in k.lower() for c in candidates):
                return d[k]
        for v in d.values():
            if isinstance(v, dict):
                res = find_key(v, candidates)
                if res is not None:
                    return res
        return None

    freq_node = find_key(doc, ['frequency','frequencies'])
    mag_node = find_key(doc, ['magnitude','magnitudelist','mag'])
    phase_node = find_key(doc, ['phase','phaselist'])

    if freq_node is None or mag_node is None:
        raise ValueError("XML: required nodes not found. Inspect schema.")

    frequency = _to_array(freq_node)
    magnitude = _to_array(mag_node)
    phase = _to_array(phase_node) if phase_node is not None else None

    # try metadata extraction (simple)
    metadata = {}
    top = doc
    if isinstance(top, dict):
        for key in ['instrument','operator','date','tap','transformer']:
            node = find_key(top, [key])
            if isinstance(node, str):
                metadata[key] = node

    return {'metadata': metadata, 'frequency': frequency, 'magnitude_db': magnitude, 'phase_deg': phase}
