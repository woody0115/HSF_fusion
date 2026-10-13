"""Choose the OVD fine-tuning sequences (meeting 2026-10-01: cut the dataset by video, e.g. 10-20 of ~50 videos for
the OVD / text-encoder LoRA stage, the rest for open-world wildcard learning).

Candidates are the 42 training sequences that are not validation sequences (configs/splits.yaml). A seeded random
search picks N sequences for OVD so that
  - the OVD share of frames is close to the target (default 25 %),
  - every class keeps a similar share of boxes on both sides (OVD share per class close to the frame share),
  - the night share (median luminance < 35) of OVD frames is close to that of all candidates,
  - and at least one night sequence is in each side.
Prints the chosen split with per-class statistics; --write adds `ovd_sequences` to configs/splits.yaml.

Run:  .venv\\Scripts\\python.exe tools\\select_ovd_split.py [--n 12] [--write]
"""
import argparse
import glob
import json
import os.path as osp
import sys

import cv2
import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from hsf.data.dataset import CFG, VAL_SEQUENCES, read_labels  # noqa: E402

NAMES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle']


def sequence_stats():
    rows = []
    for mp in sorted(glob.glob(osp.join(CFG['processed_root'], 'train', '*', 'meta.json'))):
        m = json.load(open(mp))
        if m['sequence'] in VAL_SEQUENCES:
            continue
        sd, fr = osp.dirname(mp), m['labelled_frames']
        cnt = np.zeros(7)
        for i in fr:
            c = read_labels(osp.join(sd, 'labels', f'{i:06d}.txt'))[:, 0].astype(int)
            cnt += np.bincount(c[c < 7], minlength=7)
        lum = float(np.median([cv2.imread(osp.join(m['rgb_dir'], f'{i:06d}.png'), 0).mean()
                               for i in fr[::max(1, len(fr) // 15)]]))
        rows.append({'seq': m['sequence'], 'frames': len(fr), 'boxes': cnt, 'night': lum < 35, 'lum': round(lum)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--target', type=float, default=0.25)
    ap.add_argument('--trials', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    rows = sequence_stats()
    F = np.array([r['frames'] for r in rows], float)
    B = np.stack([r['boxes'] for r in rows])
    N = np.array([r['night'] for r in rows])
    night_share = (F * N).sum() / F.sum()
    rng = np.random.default_rng(a.seed)
    best, best_cost = None, 1e9
    for _ in range(a.trials):
        idx = rng.choice(len(rows), a.n, replace=False)
        m = np.zeros(len(rows), bool)
        m[idx] = True
        if not N[m].any() or not N[~m].any():
            continue
        fs = F[m].sum() / F.sum()
        cs = B[m].sum(0) / B.sum(0)
        ns = (F[m] * N[m]).sum() / F[m].sum()
        cost = 4 * abs(fs - a.target) + np.abs(cs - fs).mean() + 2 * np.abs(cs - fs).max() + abs(ns - night_share)
        if cost < best_cost:
            best, best_cost = m.copy(), cost
    ovd = [r['seq'] for r, k in zip(rows, best) if k]
    fs = F[best].sum() / F.sum()
    print(f'OVD: {len(ovd)} sequences, {int(F[best].sum())} frames ({100 * fs:.1f} % of {int(F.sum())}); '
          f'OWOD: {int((~best).sum())} sequences, {int(F[~best].sum())} frames; cost {best_cost:.3f}')
    print(f'night share: OVD {100 * (F[best] * N[best]).sum() / F[best].sum():.1f} %, '
          f'OWOD {100 * (F[~best] * N[~best]).sum() / F[~best].sum():.1f} %')
    print('class     OVD boxes  OWOD boxes  OVD share')
    for c, n in enumerate(NAMES):
        print(f'{n:11s} {int(B[best, c].sum()):9d} {int(B[~best, c].sum()):11d}   {100 * B[best, c].sum() / B[:, c].sum():5.1f} %')
    print('OVD sequences:', ovd)
    if a.write:
        p = osp.join(ROOT, 'configs', 'splits.yaml')
        txt = open(p, encoding='utf-8').read()
        if 'ovd_sequences' in txt:
            raise SystemExit('configs/splits.yaml already has ovd_sequences; edit it by hand to change them')
        txt += ('\n# OVD fine-tuning sequences (text-encoder LoRA + detector), chosen by tools/select_ovd_split.py\n'
                f'# (seed {a.seed}, n {a.n}, target {a.target}). "owod" = train_core minus these.\n'
                'ovd_sequences:\n' + ''.join(f'  - {s}\n' for s in ovd))
        open(p, 'w', encoding='utf-8').write(txt)
        print('written to configs/splits.yaml')


if __name__ == '__main__':
    main()
