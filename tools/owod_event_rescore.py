"""Step 9, option 1: let the event camera take part in the unknown decision at inference time.

The unknown score of every candidate box is multiplied by clamp(relative event density, 4)^alpha
before NMS (hsf/owod/event.py). alpha is chosen on TRAINING frames only (2000 random frames): it
maximises the T1 model's unknown recall (top-10 + top-30 + all unknown predictions per image) there.
Using unknown annotations of held-out non-test data for one scalar follows ORE, which fits its unknown
identifier the same way; a label-free proxy (car recall of the object wildcard) saturates at 96-98 %
and cannot rank alphas. All alphas are then evaluated on the test split for T1-T3 (T4 has no unknowns).

Decision ("effect"), with the train-chosen alpha vs alpha = 0, averaged over T1-T3:
    U-Recall +0.5 points or more, or U-Recall@30 +1.0 points or more.
Writes runs/step9_evt_unknown/<run name>_rescore.json / .md (train_owod.py --only-if reads "effect").

Run:  .venv\\Scripts\\python.exe tools\\owod_event_rescore.py --run runs\\step7\\owod_add
"""
import argparse
import json
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

from hsf.data.dataset import DualDSEC  # noqa: E402
from hsf.models.dual_yolov8 import DualYOLOv8  # noqa: E402
from hsf.owod.event import rescore_unknown  # noqa: E402
from hsf.owod.model import UNK, set_active, to_open_world  # noqa: E402
from hsf.owod.train import evaluate_owod  # noqa: E402

NAMES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']
PRESENT = [0, 1, 2, 3, 4, 5, 6]


def load_task(run, task, fusion, device):
    model = to_open_world(DualYOLOv8(nc=80, fusion=fusion))
    model.load_state_dict(torch.load(osp.join(run, f'{task}.pt'), map_location='cpu')['model'])
    return model.to(device).eval()


@torch.no_grad()
def tune_alpha(model, device, alphas, frames, workers, known=(2,)):
    """Unknown recall of the T1 model on training frames (top-10 / top-30 / all unknown predictions per
    image), per alpha. Training split only, never the test split; the unknown labels are used to choose
    this one scalar, like ORE fits its unknown identifier on held-out data with unknown annotations."""
    known = list(known)
    set_active(model, known + [UNK])
    ds = DualDSEC('train', augment=False)
    random.seed(0)
    ds = Subset(ds, random.sample(range(len(ds)), frames))
    dl = DataLoader(ds, batch_size=16, num_workers=workers, pin_memory=True, collate_fn=DualDSEC.collate)
    hit = {a: {10: 0, 30: 0, 'all': 0} for a in alphas}
    n_gt = 0
    for b in dl:
        with torch.autocast('cuda'):
            evt = b['evt'].to(device)
            y, _ = model(b['rgb'].to(device), evt)
        gts = []
        for i in range(len(evt)):
            m = b['batch_idx'] == i
            unk = ~torch.isin(b['cls'][m].squeeze(-1).long(), torch.tensor(known))
            gts.append(ops.xywh2xyxy(b['bboxes'][m][unk] * 640).to(device))
        n_gt += sum(len(g) for g in gts)
        for a in alphas:
            dets = ops.non_max_suppression(rescore_unknown(y.float().clone(), evt, len(known), a), 0.001, 0.7,
                                           multi_label=True, max_det=100)
            for d, g in zip(dets, gts):
                d = d[d[:, 5] == len(known)]                          # unknown predictions, sorted by score
                if len(g) and len(d):
                    iou = box_iou(g, d[:, :4])
                    for k in (10, 30, 'all'):
                        hit[a][k] += int((iou[:, :(None if k == 'all' else k)].max(1).values >= 0.5).sum())
    rec = {a: {k: 100.0 * v / max(n_gt, 1) for k, v in h.items()} for a, h in hit.items()}
    best = max(alphas, key=lambda a: (round(rec[a][10] + rec[a][30] + rec[a]['all'], 2), -a))
    return best, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=osp.join(ROOT, 'runs', 'step7', 'owod_add'))
    ap.add_argument('--fusion', default='add')
    ap.add_argument('--alphas', type=float, nargs='*', default=[0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument('--tune-frames', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    a.alphas = [float(x) for x in a.alphas]          # '0' and '0.0' must be the same key
    if not osp.isfile(osp.join(a.run, 'T1.pt')):
        print(f'{a.run}: no T1.pt (run skipped or not finished) -> nothing to evaluate', flush=True)
        return
    device = torch.device('cuda')
    out_dir = osp.join(ROOT, 'runs', 'step9_evt_unknown')
    os.makedirs(out_dir, exist_ok=True)
    name = osp.basename(osp.normpath(a.run))
    tasks = yaml.safe_load(open(osp.join(ROOT, 'configs', 'owod_tasks.yaml')))['tasks']

    model = load_task(a.run, 'T1', a.fusion, device)
    best, proxy = tune_alpha(model, device, a.alphas, a.tune_frames, a.workers)
    print('train-split unknown recall (T1): ' +
          '  '.join(f'a={al}: @10 {r[10]:.1f} @30 {r[30]:.1f} all {r["all"]:.1f}' for al, r in proxy.items()) + f'  -> alpha {best}',
          flush=True)

    results, prev = {}, []
    for tname, tcfg in tasks.items():
        cur = list(tcfg['classes'])
        unknown = [c for c in PRESENT if c not in prev + cur]
        if unknown:
            model = load_task(a.run, tname, a.fusion, device)
            res = evaluate_owod(model, device, prev, cur, unknown, workers=a.workers, alphas=tuple(a.alphas))
            results[tname] = {str(al): r for al, r in res.items()}
            for al, r in res.items():
                print(f'{tname} alpha {al}: U-Recall {r["U-Recall"]:.2f}  @10 {r["U-Recall@10"]:.2f}  '
                      f'@30 {r["U-Recall@30"]:.2f}  WI {r["WI"]:.4f}  A-OSE {r["A-OSE"]}', flush=True)
        prev += cur

    mean = lambda al, key: float(np.mean([results[t][str(al)][key] for t in results]))   # noqa: E731
    gain = {k: mean(best, k) - mean(0.0 if 0.0 in a.alphas else a.alphas[0], k) for k in ('U-Recall', 'U-Recall@30')}
    effect = best != 0 and (gain['U-Recall'] >= 0.5 or gain['U-Recall@30'] >= 1.0)
    summary = {'run': a.run, 'alpha': best, 'proxy': {str(k): v for k, v in proxy.items()}, 'gain': gain,
               'effect': bool(effect), 'results': results}
    json.dump(summary, open(osp.join(out_dir, f'{name}_rescore.json'), 'w'), indent=1)

    lines = [f'# Option 1: event rescoring of the unknown score ({name})', '',
             f'alpha chosen on {a.tune_frames} training frames (T1 unknown recall, never the test split): **{best}**', '',
             '| Task | alpha | U-Recall | U-Recall@10 | U-Recall@30 | WI | A-OSE |', '|---|---|---|---|---|---|---|']
    for t, rs in results.items():
        for al, r in rs.items():
            mark = ' (chosen)' if float(al) == best else ''
            lines.append(f'| {t} | {al}{mark} | {r["U-Recall"]:.2f} | {r["U-Recall@10"]:.2f} | {r["U-Recall@30"]:.2f} | '
                         f'{r["WI"]:.4f} | {r["A-OSE"]} |')
    lines += ['', f'mean gain over T1-T3 at the chosen alpha: U-Recall {gain["U-Recall"]:+.2f}, '
                  f'U-Recall@30 {gain["U-Recall@30"]:+.2f} -> effect: **{effect}**']
    open(osp.join(out_dir, f'{name}_rescore.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
