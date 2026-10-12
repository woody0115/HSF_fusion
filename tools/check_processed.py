"""Draw processed outputs (GEAR png + YOLO txt labels) back onto RGB to verify step 2.

Run:  .venv\\Scripts\\python.exe tools\\check_processed.py test thun_02_a 1500
"""
import json
import os.path as osp
import sys

import cv2
import numpy as np
import yaml

CFG = yaml.safe_load(open(osp.join(osp.dirname(__file__), '..', 'configs', 'paths.yaml')))
split, seq, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = osp.join(CFG['processed_root'], split, seq)
meta = json.load(open(osp.join(out, 'meta.json')))
rgb = cv2.imread(osp.join(meta['rgb_dir'], f'{idx:06d}.png'))
gear = cv2.imread(osp.join(out, 'gear', f'{idx:06d}.png'))
lines = open(osp.join(out, 'labels', f'{idx:06d}.txt')).read().split('\n')
H, W = rgb.shape[:2]
for img in (rgb, gear):
    for ln in filter(None, lines):
        c, xc, yc, w, h = ln.split()
        xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
        p0, p1 = (int(xc - w / 2), int(yc - h / 2)), (int(xc + w / 2), int(yc + h / 2))
        cv2.rectangle(img, p0, p1, (0, 255, 0), 1)
        cv2.putText(img, CFG['names'][int(c)], (p0[0], max(p0[1] - 3, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
chans = [cv2.cvtColor(gear[..., k], cv2.COLOR_GRAY2BGR) for k in range(3)]
panel = np.vstack([np.hstack([rgb, gear, chans[0]]), np.hstack([chans[1], chans[2], np.zeros_like(rgb)])])
for (x, y), t in zip([(6, 18), (646, 18), (1286, 18), (6, 498), (646, 498)],
                     ['RGB + YOLO labels', 'GEAR png + YOLO labels', 'h', 'feat', 'num (black = no events)']):
    cv2.putText(panel, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
dst = osp.join(CFG['processed_root'], 'preview', f'processed_{split}_{seq}_{idx:06d}.jpg')
cv2.imwrite(dst, panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(dst, '|', len([l for l in lines if l]), 'boxes |', meta['n_images'], 'frames in sequence')
