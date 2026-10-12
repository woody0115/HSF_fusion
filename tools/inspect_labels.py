"""Step 2a: inspect DSEC-Det labels (object_detections/left/tracks.npy) and calibration.

Run:  .venv\\Scripts\\python.exe tools\\inspect_labels.py
"""
import glob
import os.path as osp
from collections import Counter

import numpy as np
import yaml

ROOT = yaml.safe_load(open(osp.join(osp.dirname(__file__), '..', 'configs', 'paths.yaml')))['dsec_root']

files = sorted(glob.glob(f'{ROOT}/*/*/object_detections/left/tracks.npy'))
print(f'label files: {len(files)}')
t = np.load(files[0])
print(f'\nexample: {files[0]}')
print('dtype :', t.dtype)
print('first rows:\n', t[:3])

cls_count = Counter()
stats = []
for f in files:
    split, seq = f.replace('\\', '/').split('/')[-5:-3]
    t = np.load(f)
    ts = np.unique(t['t'])
    dt = np.diff(ts) / 1000.0 if len(ts) > 1 else np.array([0])
    cls_count.update({(split, int(c)): n for c, n in zip(*np.unique(t['class_id'], return_counts=True))})
    stats.append((split, seq, len(t), len(ts), np.median(dt), t['x'].min(), t['y'].min(),
                  (t['x'] + t['w']).max(), (t['y'] + t['h']).max(), t['class_confidence'].min()))

print(f'\n{"split":5} {"sequence":20} {"boxes":>7} {"frames":>6} {"dt_ms":>6}  box extent (x0,y0)-(x1,y1)   min_conf')
for s in stats:
    print(f'{s[0]:5} {s[1]:20} {s[2]:7d} {s[3]:6d} {s[4]:6.1f}  ({s[5]:6.1f},{s[6]:6.1f})-({s[7]:6.1f},{s[8]:6.1f})  {s[9]:.2f}')

for split in ('train', 'test'):
    rows = [s for s in stats if s[0] == split]
    print(f'\n{split}: {len(rows)} sequences, {sum(r[3] for r in rows)} labelled frames, {sum(r[2] for r in rows)} boxes')
    print('  boxes per class_id:', {c: n for (sp, c), n in sorted(cls_count.items()) if sp == split})

calib = glob.glob(f'{ROOT}/test/*/calibration/cam_to_cam.yaml')[:1]
if calib:
    c = yaml.safe_load(open(calib[0]))
    print(f'\ncalibration keys ({calib[0]}):', list(c.keys()))
    for k in ('intrinsics',):
        if k in c:
            for cam, v in c[k].items():
                if isinstance(v, dict) and 'resolution' in v:
                    print(f'  {cam}: resolution {v["resolution"]}')
