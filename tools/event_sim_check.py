"""How usable are simulated events (hsf/data/event_sim.py) for the fusion models? Two steps on DSEC test:

  1. calibration: for several contrast thresholds C, simulate events from consecutive RGB frames and
     compare the mean of the GEAR `num` channel with the real GEAR images; pick the closest C.
  2. evaluation: run one fusion model over every test sequence in order with three event inputs
     (real GEAR, simulated GEAR with the chosen C, no events at all) and report mAP50 / mAP50-95.

Writes runs/diag/event_sim_check.json. The screen demo (tools/demo_screen.py) uses the chosen C.
Run:  .venv\\Scripts\\python.exe tools\\event_sim_check.py [--model af_lr1e3] [--contrasts 0.05 0.1 0.15 0.2 0.3]
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import DetMetrics  # noqa: E402

from hsf.data.dataset import CFG, DualDSEC, SequenceDSEC, read_labels  # noqa: E402
from hsf.data.event_sim import EventSimulator  # noqa: E402
from hsf.data.gear import GEAR  # noqa: E402
from hsf.engine import IOUV, _frame_stats  # noqa: E402
from tools.eval_conditions import build  # noqa: E402

NAMES = {i: n for i, n in enumerate(CFG['names'])}
ZERO_GEAR = GEAR.to_uint8(np.zeros((3, 480, 640), np.float32))       # no events: h = feat = 128, num = 0


class RawFrames(Dataset):
    """Every frame of the test sequences in temporal order: raw BGR image, raw GEAR png, labels."""

    def __init__(self, max_per_seq=None):
        self.frames = SequenceDSEC('test').frames
        if max_per_seq:
            seen = {}
            keep = []
            for f in self.frames:
                seen[f[0]] = seen.get(f[0], 0) + 1
                if seen[f[0]] <= max_per_seq:
                    keep.append(f)
            self.frames = keep

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, k):
        si, seq_dir, rgb_dir, idx, has, first = self.frames[k]
        rgb = cv2.imread(osp.join(rgb_dir, f'{idx:06d}.png'))
        evt = cv2.imread(osp.join(seq_dir, 'gear', f'{idx:06d}.png'))
        lab = read_labels(osp.join(seq_dir, 'labels', f'{idx:06d}.txt')) if has else np.zeros((0, 5), np.float32)
        return rgb, evt, lab, has, first, osp.basename(seq_dir)


def loader(ds, workers):
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers, pin_memory=False)


def calibrate(contrasts, frames_per_seq, workers):
    sims = {c: EventSimulator(c) for c in contrasts}
    real, sim = [], {c: [] for c in contrasts}
    for rgb, evt, lab, has, first, seq in loader(RawFrames(frames_per_seq), workers):
        rgb, evt = rgb.numpy(), evt.numpy()
        if first:
            for s in sims.values():
                s.reset()
        for c, s in sims.items():
            g = s(rgb)
            if not first:
                sim[c].append(float(g[..., 2].mean()))
        if not first:
            real.append(float(evt[..., 2].mean()))
    r = float(np.mean(real))
    out = {'real_num_mean': round(r, 2),
           'sim_num_mean': {str(c): round(float(np.mean(v)), 2) for c, v in sim.items()}}
    best = min(contrasts, key=lambda c: abs(np.log(max(np.mean(sim[c]), 1e-6) / r)))
    out['chosen_contrast'] = best
    print(json.dumps(out), flush=True)
    return best, out


@torch.no_grad()
def evaluate(model_key, contrasts, workers, device):
    """contrasts: one or more thresholds; each gets its own simulator and its own 'simulated C=..' result."""
    model = build(model_key, device)
    ds_t = DualDSEC('test')
    aug = ds_t.sample_aug()
    sims = {f'simulated C={c}': EventSimulator(c) for c in contrasts}
    variants = ('real', *sims, 'none')
    metrics = {v: DetMetrics(names=NAMES) for v in variants}
    iouv = IOUV.to(device)
    t0, n = time.time(), 0
    for rgb, evt, lab, has, first, seq in loader(RawFrames(), workers):
        rgb, evt, lab = rgb.numpy(), evt.numpy(), lab.numpy()
        if first:
            for s in sims.values():
                s.reset()
        g_sims = [s(rgb) for s in sims.values()]          # the simulators must see every frame
        if not has:
            continue
        batch_r, batch_e, labs = [], [], None
        for e in (evt, *g_sims, ZERO_GEAR):
            r_t, e_t, l_t = ds_t.transform(rgb, e, lab, aug)
            batch_r.append(r_t)
            batch_e.append(e_t)
            labs = l_t
        with torch.autocast('cuda'):
            out = model(torch.stack(batch_r).to(device), torch.stack(batch_e).to(device))
        dets = ops.non_max_suppression(out[0].float(), 0.001, 0.7, multi_label=True, max_det=300)
        labs = labs.to(device)
        tbox = ops.xywh2xyxy(labs[:, 1:] * 640)
        for v, d in zip(variants, dets):
            metrics[v].update_stats(_frame_stats(d, labs[:, 0], tbox, iouv))
        n += 1
        if n % 2000 == 0:
            print(f'  {n} labelled frames ({time.time() - t0:.0f} s)', flush=True)
    res = {}
    for v, dm in metrics.items():
        dm.process()
        p, r, m50, m5095 = dm.box.mean_results()
        res[v] = {'mAP50': round(100 * m50, 2), 'mAP50-95': round(100 * m5095, 2)}
    print(model_key, json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='af_lr1e3', help='key of tools/eval_conditions.py MODELS (a dual-stream model)')
    ap.add_argument('--contrasts', type=float, nargs='*', default=[0.05, 0.1, 0.15, 0.2, 0.3])
    ap.add_argument('--calib-frames', type=int, default=300, help='frames per test sequence for the calibration')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--eval-contrasts', type=float, nargs='*', default=None,
                    help='evaluate these thresholds (default: only the calibrated one)')
    ap.add_argument('--out', default='event_sim_check.json')
    a = ap.parse_args()
    device = torch.device('cuda')
    best, calib = calibrate(a.contrasts, a.calib_frames, a.workers)
    res = evaluate(a.model, a.eval_contrasts or [best], a.workers, device)
    os.makedirs(osp.join(ROOT, 'runs', 'diag'), exist_ok=True)
    out = {'model': a.model, 'calibration': calib, 'evaluation': res}
    json.dump(out, open(osp.join(ROOT, 'runs', 'diag', a.out), 'w'), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
