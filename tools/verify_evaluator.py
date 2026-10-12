"""Check that hsf.engine.evaluate reproduces Ultralytics' validator.

Evaluates the step-3 RGB model on one test sequence twice: with Ultralytics' YOLO.val() and with
our evaluator (wrapping the single-stream model in the dual-stream interface). The numbers should
agree closely (small differences come from letterbox padding: Ultralytics validates with
rectangular 640x480 batches, we pad to 640x640).

Run:  .venv\\Scripts\\python.exe tools\\verify_evaluator.py [--seq thun_02_a]
"""
import argparse
import glob
import os.path as osp
import sys

import torch
import torch.nn as nn
import yaml

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from ultralytics import YOLO  # noqa: E402

from hsf.data.dataset import CFG, DualDSEC  # noqa: E402
from hsf.engine import evaluate  # noqa: E402
from hsf.models.dual_yolov8 import load_ultralytics  # noqa: E402


class SingleAsDual(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, rgb, evt):
        return self.m(rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='thun_02_a')
    ap.add_argument('--weights', default=osp.join(ROOT, 'runs', 'step3', 'rgb_yolov8s', 'weights', 'best.pt'))
    a = ap.parse_args()
    names = {i: n for i, n in enumerate(CFG['names'])}

    # 1) Ultralytics validator on this sequence
    imgs = sorted(glob.glob(osp.join(CFG['processed_root'], 'yolo', 'rgb', 'test', 'images', f'{a.seq}_*.png')))
    lst = osp.join(ROOT, 'runs', f'verify_{a.seq}.txt')
    open(lst, 'w').write('\n'.join(imgs))
    data = osp.join(ROOT, 'runs', f'verify_{a.seq}.yaml')
    yaml.safe_dump({'path': osp.dirname(lst), 'train': lst, 'val': lst, 'names': names}, open(data, 'w'))
    r = YOLO(a.weights).val(data=data, imgsz=640, batch=16, conf=0.001, iou=0.7, plots=False, verbose=False,
                            project=osp.join(ROOT, 'runs'), name='verify', exist_ok=True)
    print(f'Ultralytics val : mAP50 {r.box.map50:.4f}  mAP50-95 {r.box.map:.4f}  ({len(imgs)} images)')

    # 2) our evaluator
    ds = DualDSEC('test', augment=False, sequences=[a.seq])
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4, collate_fn=DualDSEC.collate)
    model = SingleAsDual(load_ultralytics(a.weights)).cuda().eval()
    res = evaluate(model, dl, torch.device('cuda'), names, desc='ours')
    print(f'our evaluator   : mAP50 {res["mAP50"]:.4f}  mAP50-95 {res["mAP50-95"]:.4f}  ({len(ds)} images)')


if __name__ == '__main__':
    main()
