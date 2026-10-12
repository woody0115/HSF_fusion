"""Render GEAR previews next to the RGB frame, with ground-truth boxes.

For each requested frame, GEAR is warmed up over the preceding --warmup frames (it is recurrent),
then a 2x3 panel is saved:  RGB+boxes | GEAR (3 ch as color)+boxes | events of this window
                            h (long-term) | feat (short-term) | num (activity)

Run:  .venv\\Scripts\\python.exe tools\\preview_gear.py test thun_02_a --frames 300 1500 3000
"""
import argparse
import os
import os.path as osp
import sys

import cv2
import h5py
import hdf5plugin  # noqa: F401  (registers the blosc filter)
import numpy as np
import yaml

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
from hsf.data.gear import GEAR, EventReader, first_window_start  # noqa: E402

CFG = yaml.safe_load(open(osp.join(osp.dirname(__file__), '..', 'configs', 'paths.yaml')))
NAMES = CFG['names']


def draw_boxes(img, boxes):
    for b in boxes:
        x0, y0 = int(b['x']), int(b['y'])
        x1, y1 = int(b['x'] + b['w']), int(b['y'] + b['h'])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.putText(img, NAMES[b['class_id']], (x0, max(y0 - 3, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return img


def label(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('split')
    ap.add_argument('seq')
    ap.add_argument('--frames', type=int, nargs='+', default=[300])
    ap.add_argument('--warmup', type=int, default=20)
    a = ap.parse_args()

    seq_dir = osp.join(CFG['dsec_root'], a.split, a.seq)
    out_dir = osp.join(CFG['processed_root'], 'preview')
    os.makedirs(out_dir, exist_ok=True)
    ts = np.loadtxt(osp.join(seq_dir, 'images', 'timestamps.txt'), dtype=np.int64)
    tracks = np.load(osp.join(seq_dir, 'object_detections', 'left', 'tracks.npy'))
    img_dir = osp.join(seq_dir, 'images', 'left', 'distorted')
    gear = GEAR()

    with h5py.File(osp.join(seq_dir, 'events', 'left', 'events.h5'), 'r') as f:
        reader = EventReader(f)
        for idx in a.frames:
            gear.reset()
            for i in range(max(0, idx - a.warmup), idx + 1):
                x, y, p = reader.window(first_window_start(ts, i), ts[i])
                g = gear(x, y, p)
            n_ev = len(x)
            boxes = tracks[tracks['t'] == ts[idx]]
            rgb = cv2.imread(osp.join(img_dir, f'{idx:06d}.png'))
            g8 = GEAR.to_uint8(g)
            ev = np.full((480, 640, 3), 255, np.uint8)       # raw events of this window: red +, blue -
            ev[y[p == 1], x[p == 1]] = (0, 0, 255)
            ev[y[p == 0], x[p == 0]] = (255, 0, 0)
            chans = [cv2.cvtColor(g8[..., k], cv2.COLOR_GRAY2BGR) for k in range(3)]
            top = np.hstack([label(draw_boxes(rgb.copy(), boxes), f'RGB  frame {idx}  ({len(boxes)} boxes)'),
                             label(draw_boxes(g8.copy(), boxes), 'GEAR (h, feat, num as B,G,R)'),
                             label(ev, f'events in window: {n_ev:,}')])
            bot = np.hstack([label(chans[0], 'h: long-term memory'),
                             label(chans[1], 'feat: polarity sum (this window)'),
                             label(chans[2], 'num: event count (activity)')])
            out = osp.join(out_dir, f'{a.split}_{a.seq}_{idx:06d}.jpg')
            cv2.imwrite(out, np.vstack([top, bot]), [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f'frame {idx}: {n_ev:,} events, {len(boxes)} boxes -> {out}')


if __name__ == '__main__':
    main()
