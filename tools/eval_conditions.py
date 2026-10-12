"""Evaluate trained models separately per lighting condition of the test frames.

Buckets (per frame, from tools/lighting_stats.py): night = mean luminance < 35, dim = 35-55,
day >= 55, plus "all". Single-stream Ultralytics models (RGB / event) and dual-stream checkpoints
(add / af / hsf; hsf is evaluated by streaming each sequence in order) use the same evaluator
(conf 0.001, IoU 0.7, multi-label, max 300; verified against Ultralytics).

Run:  .venv\\Scripts\\python.exe tools\\eval_conditions.py [--models rgb event add af hsf ...]
Writes runs/eval_conditions.md / .json
"""
import argparse
import json
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import DetMetrics  # noqa: E402

from hsf.data.dataset import CFG, DualDSEC, SequenceDSEC  # noqa: E402
from hsf.engine import IOUV, _frame_stats  # noqa: E402
from hsf.models.dual_yolov8 import DualYOLOv8, load_ultralytics  # noqa: E402

NAMES = {i: n for i, n in enumerate(CFG['names'])}
RUNS = osp.join(ROOT, 'runs')
MODELS = {   # key: (kind, path, fusion)
    'rgb': ('single_rgb', 'step3/rgb_yolov8s/weights/best.pt', None),
    'event': ('single_evt', 'step3/event_yolov8s/weights/best.pt', None),
    'add': ('dual', 'step4/dual_add_e20/best.pt', 'add'),
    'af': ('dual', 'step5_6/dual_af_e20/best.pt', 'af'),
    'hsf': ('dual', 'step5_6/dual_hsf_e20/best.pt', 'hsf'),
    'add_lr1e3': ('dual', 'step4/dual_add_lr1e3_e5/best.pt', 'add'),
    'af_lr1e3': ('dual', 'step5_6/dual_af_lr1e3_e5/best.pt', 'af'),
    'hsf_lr1e3': ('dual', 'step5_6/dual_hsf_lr1e3_e5/best.pt', 'hsf'),
    # step 8: illumination-aware fusion
    'route_add': ('route', ('rgb', 'add_lr1e3'), None),      # hard switch: day -> RGB, dim/night -> fusion
    'route_af': ('route', ('rgb', 'af_lr1e3'), None),
    'iaf_add': ('dual', 'step8_iaf/dual_iaf_add_lr1e3_e5/best.pt', 'iaf_add'),
    'iaf_af': ('dual', 'step8_iaf/dual_iaf_af_lr1e3_e5/best.pt', 'iaf_af'),
}
BUCKETS = ('all', 'day', 'dim', 'night')


def bucket_of(lum):
    return 'night' if lum < 35 else ('dim' if lum < 55 else 'day')


class Single(nn.Module):
    def __init__(self, m, use_evt):
        super().__init__()
        self.m, self.use_evt = m, use_evt

    def forward(self, rgb, evt):
        return self.m(evt if self.use_evt else rgb)


class Route(nn.Module):
    """Hard switch by lighting (reference for the learned gate): frames whose 640x480 image has mean
    luminance >= 55 (the 'day' bucket threshold) use the day model, the others the dark model."""

    def __init__(self, day, dark):
        super().__init__()
        self.day, self.dark = day, dark

    def forward(self, rgb, evt):
        img = rgb[:, :, 80:560]                       # letterboxed 640x480 -> rows 80..559 (no augmentation)
        lum = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).mean((1, 2)) * 255
        a, b = self.day(rgb, evt)[0], self.dark(rgb, evt)[0]
        return (torch.where((lum >= 55)[:, None, None], a, b),)


def gate_means(model):
    """Per-image mean gate of an illumination-gated model (average over the three scales), or None."""
    g = [f.last_gate for f in getattr(model, 'fusion', []) if getattr(f, 'last_gate', None) is not None]
    return torch.stack(g).mean(0).float().cpu() if g else None


def build(key, device):
    kind, rel, fusion = MODELS[key]
    if kind == 'route':
        day, dark = build(rel[0], device), build(rel[1], device)
        return None if day is None or dark is None else Route(day, dark).eval()
    path = osp.join(RUNS, rel)
    if not osp.isfile(path):
        return None
    if kind.startswith('single'):
        return Single(load_ultralytics(path), kind == 'single_evt').to(device).eval()
    m = DualYOLOv8(nc=8, fusion=fusion)
    m.load_state_dict(torch.load(path, map_location='cpu')['model'])
    return m.to(device).eval()


@torch.no_grad()
def run(model, lighting, device, workers):
    metrics = {b: DetMetrics(names=NAMES) for b in BUCKETS}
    seq_metrics = {}
    gates = {b: [] for b in BUCKETS}
    iouv = IOUV.to(device)

    def add(stat, seq, idx):
        b = bucket_of(lighting[seq][str(idx)][0])
        for k in ('all', b):
            metrics[k].update_stats(stat)
        seq_metrics.setdefault(seq, DetMetrics(names=NAMES)).update_stats(stat)

    if getattr(model, 'stateful', False):
        ds = SequenceDSEC('test')
        dl = DataLoader(ds, batch_size=1, num_workers=workers, pin_memory=True)
        state = None
        for k, (rgb, evt, lab, has, si, first) in enumerate(dl):
            if bool(first[0]):
                state = None
            with torch.autocast('cuda'):
                out, state = model(rgb.to(device), evt.to(device), state)
            if bool(has[0]):
                _, seq_dir, _, idx, _, _ = ds.frames[k]
                d = ops.non_max_suppression(out[0].float(), 0.001, 0.7, multi_label=True, max_det=300)[0]
                lab = lab[0].to(device)
                add(_frame_stats(d, lab[:, 0], ops.xywh2xyxy(lab[:, 1:] * 640), iouv), osp.basename(seq_dir), idx)
    else:
        ds = DualDSEC('test')
        dl = DataLoader(ds, batch_size=16, num_workers=workers, pin_memory=True, collate_fn=DualDSEC.collate)
        k = 0
        for b in dl:
            with torch.autocast('cuda'):
                out = model(b['rgb'].to(device), b['evt'].to(device))
            dets = ops.non_max_suppression(out[0].float(), 0.001, 0.7, multi_label=True, max_det=300)
            g = gate_means(model)
            for i, d in enumerate(dets):
                seq_dir, _, idx = ds.items[k]
                k += 1
                if g is not None:
                    bk = bucket_of(lighting[osp.basename(seq_dir)][str(idx)][0])
                    gates['all'].append(float(g[i]))
                    gates[bk].append(float(g[i]))
                m = b['batch_idx'] == i
                tcls = b['cls'][m].squeeze(-1).to(device)
                add(_frame_stats(d, tcls, ops.xywh2xyxy(b['bboxes'][m].to(device) * 640), iouv),
                    osp.basename(seq_dir), idx)

    def summ(dm):
        if not dm.stats['tp']:
            return None
        dm.process()
        p, r, m50, m5095 = dm.box.mean_results()
        return {'mAP50': round(100 * m50, 2), 'mAP50-95': round(100 * m5095, 2),
                'frames': int(sum(len(np.atleast_1d(t)) > -1 for t in dm.stats['tp']))}
    gate = {b: round(float(np.mean(v)), 4) for b, v in gates.items() if v} or None
    return {b: summ(metrics[b]) for b in BUCKETS}, {s: summ(dm) for s, dm in seq_metrics.items()}, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=list(MODELS))
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    lighting = json.load(open(osp.join(CFG['processed_root'], 'test_lighting.json')))
    counts = {b: 0 for b in BUCKETS}
    for seq in lighting.values():
        for v in seq.values():
            counts['all'] += 1
            counts[bucket_of(v[0])] += 1
    print('frames per bucket:', counts, flush=True)
    device = torch.device('cuda')
    out_json = osp.join(RUNS, 'eval_conditions.json')
    results = json.load(open(out_json)) if osp.isfile(out_json) else {}
    for key in a.models:
        model = build(key, device)
        if model is None:
            print(f'{key}: checkpoint not found, skipped', flush=True)
            continue
        t0 = time.time()
        res, per_seq, gate = run(model, lighting, device, a.workers)
        results[key] = {'buckets': res, 'sequences': per_seq}
        if gate:
            results[key]['gate'] = gate
            print(f'{key:10s} mean gate g: ' + '  '.join(f'{b} {v:.3f}' for b, v in gate.items()), flush=True)
        print(f'{key:10s} ' + '  '.join(f'{b}: {r["mAP50"]:.1f}/{r["mAP50-95"]:.1f}' for b, r in res.items() if r) +
              f'  ({time.time() - t0:.0f} s)', flush=True)
        json.dump(results, open(out_json, 'w'), indent=1)
        del model
        torch.cuda.empty_cache()

    lines = [f'frames: all {counts["all"]}, day {counts["day"]}, dim {counts["dim"]}, night {counts["night"]}', '',
             '| model | all | day | dim | night | zurich_city_12_a (night seq) | interlaken_01_a (dark) |',
             '|---|---|---|---|---|---|---|']
    for key, r in results.items():
        cell = lambda x: f'{x["mAP50"]:.1f} / {x["mAP50-95"]:.1f}' if x else '-'   # noqa: E731
        lines.append(f'| {key} | ' + ' | '.join(cell(r['buckets'].get(b)) for b in BUCKETS) + ' | ' +
                     cell(r['sequences'].get('zurich_city_12_a')) + ' | ' + cell(r['sequences'].get('interlaken_01_a')) + ' |')
    md = '# mAP50 / mAP50-95 by lighting condition (DSEC-Det test)\n\n' + '\n'.join(lines) + '\n'
    open(osp.join(RUNS, 'eval_conditions.md'), 'w', encoding='utf-8').write(md)
    print('\n' + md, flush=True)


if __name__ == '__main__':
    main()
