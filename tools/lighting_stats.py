"""Per-frame lighting statistics of the RGB frames (used to split the evaluation by condition).

For every labelled test frame: mean luminance (0-255) and the fraction of under-exposed (< 20) and
over-exposed (> 250) pixels. Writes <processed_root>/test_lighting.json and prints per-sequence
summaries so thresholds can be chosen.

Run:  .venv\\Scripts\\python.exe tools\\lighting_stats.py
"""
import glob
import json
import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
from hsf.data.dataset import CFG  # noqa: E402


def stats(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return float(g.mean()), float((g < 20).mean()), float((g > 250).mean())


def main():
    out = {}
    for mpath in sorted(glob.glob(osp.join(CFG['processed_root'], 'test', '*', 'meta.json'))):
        m = json.load(open(mpath))
        paths = [osp.join(m['rgb_dir'], f'{i:06d}.png') for i in m['labelled_frames']]
        with ThreadPoolExecutor(8) as ex:
            s = list(ex.map(stats, paths))
        out[m['sequence']] = {str(i): v for i, v in zip(m['labelled_frames'], s)}
        a = np.array(s)
        print(f'{m["sequence"]:18s} frames {len(a):5d}  luminance median {np.median(a[:, 0]):6.1f} '
              f'(p10 {np.percentile(a[:, 0], 10):5.1f}, p90 {np.percentile(a[:, 0], 90):5.1f})  '
              f'dark px {a[:, 1].mean():.2f}  saturated px {a[:, 2].mean():.3f} (max {a[:, 2].max():.2f})', flush=True)
    with open(osp.join(CFG['processed_root'], 'test_lighting.json'), 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()
