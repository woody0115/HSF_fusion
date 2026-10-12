"""B5 / B6 of the planning agent's questions: label statistics per split and class, box-area bins for
pedestrian / rider, AP50 per class of the OWOD runs per Task, and the best epoch of the T1 base training.

Areas are measured on the original 640x480 frames (COCO bins: small < 32^2, medium 32^2-96^2, large > 96^2).
Writes runs/diag/b5_b6_data.json.
Run:  .venv\\Scripts\\python.exe tools\\diag_data_stats.py
"""
import csv
import json
import os
import os.path as osp
import sys

import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from hsf.data.dataset import list_frames, read_labels  # noqa: E402

NAMES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']


def split_stats(split):
    items = list_frames(split, require_gear=False)
    boxes, frames, areas = np.zeros(8, int), np.zeros(8, int), {0: [], 1: []}
    for seq_dir, _, idx in items:
        lab = read_labels(osp.join(seq_dir, 'labels', f'{idx:06d}.txt'))
        cls = lab[:, 0].astype(int)
        for c in range(8):
            n = int((cls == c).sum())
            boxes[c] += n
            frames[c] += n > 0
        for c in (0, 1):
            m = cls == c
            areas[c] += list(lab[m, 3] * 640 * lab[m, 4] * 480)
    out = {'labelled_frames': len(items), 'sequences': len({i[0] for i in items}),
           'per_class': {NAMES[c]: {'boxes': int(boxes[c]), 'frames_with_class': int(frames[c])} for c in range(8)}}
    for c in (0, 1):
        a = np.array(areas[c])
        out['per_class'][NAMES[c]]['area_bins_%'] = {
            'small(<32^2)': round(100 * float((a < 32 ** 2).mean()), 1) if len(a) else None,
            'medium(32^2-96^2)': round(100 * float(((a >= 32 ** 2) & (a <= 96 ** 2)).mean()), 1) if len(a) else None,
            'large(>96^2)': round(100 * float((a > 96 ** 2).mean()), 1) if len(a) else None,
            'median_area_px': round(float(np.median(a)), 1) if len(a) else None}
    return out


def main():
    out = {'train': split_stats('train'), 'test': split_stats('test')}
    ap = {}
    for tag, run in (('#11', 'owod_add'), ('#12', 'owod_add_evtign')):
        for t in ('T1', 'T2', 'T3', 'T4'):
            m = json.load(open(osp.join(ROOT, 'runs', 'step7', run, f'{t}_metrics.json')))
            ap[f'{tag} {t}'] = {NAMES[int(c)]: v for c, v in m['AP50_per_class'].items()}
    out['owod_AP50_per_class'] = ap
    rows = list(csv.DictReader(open(osp.join(ROOT, 'runs', 'step7', 'owod_add', 'T1_base', 'results.csv'))))
    out['T1_base'] = [{k: r[k] for k in ('epoch', 'mAP50', 'mAP50-95', 'lr')} for r in rows]
    os.makedirs(osp.join(ROOT, 'runs', 'diag'), exist_ok=True)
    json.dump(out, open(osp.join(ROOT, 'runs', 'diag', 'b5_b6_data.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
