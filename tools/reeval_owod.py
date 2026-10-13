"""Re-evaluate finished DSEC-OWOD runs with the current metrics (adds U-AP and U-Recall@k to old runs).

Loads <run>/T1.pt ... T4.pt and evaluates each Task on the test split exactly like train_owod.py does
(same evaluator, max 100 detections per image). Writes runs/diag/owod_reeval.json / .md.
Run:  .venv\\Scripts\\python.exe tools\\reeval_owod.py [--runs owod_add owod_add_evtign]
"""
import argparse
import json
import os
import os.path as osp
import sys

import torch
import yaml

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from hsf.owod.train import evaluate_owod  # noqa: E402
from tools.owod_event_rescore import PRESENT, load_task  # noqa: E402

NAMES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']
KEYS = ('U-AP', 'U-Recall', 'U-Recall@30', 'WI', 'A-OSE', 'mAP_PK', 'mAP_CK', 'mAP_Both')


def fmt(v, nd=1):
    return '-' if v is None or v != v else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=['owod_add', 'owod_add_evtign'])
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    device = torch.device('cuda')
    tasks = yaml.safe_load(open(osp.join(ROOT, 'configs', 'owod_tasks.yaml')))['tasks']
    out = {}
    for run in a.runs:
        prev, out[run] = [], {}
        for tname, tcfg in tasks.items():
            cur = list(tcfg['classes'])
            unknown = [c for c in PRESENT if c not in prev + cur]
            model = load_task(osp.join(ROOT, 'runs', 'step7', run), tname, 'add', device)
            r = evaluate_owod(model, device, prev, cur, unknown, workers=a.workers)
            out[run][tname] = {k: r[k] for k in KEYS}
            out[run][tname]['AP50_per_class'] = {NAMES[c]: v for c, v in r['AP50_per_class'].items()}
            print(run, tname, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[run][tname].items()
                               if k != 'AP50_per_class'}, flush=True)
            prev += cur
    os.makedirs(osp.join(ROOT, 'runs', 'diag'), exist_ok=True)
    json.dump(out, open(osp.join(ROOT, 'runs', 'diag', 'owod_reeval.json'), 'w'), indent=1)
    lines = ['| run | Task | U-AP ↑ | U-Recall ↑ | WI ↓ | A-OSE ↓ | PK | CK | Both |', '|---|---|---|---|---|---|---|---|---|']
    for run, res in out.items():
        for t, r in res.items():
            lines.append(f'| {run} | {t} | {fmt(r["U-AP"], 2)} | {fmt(r["U-Recall"])} | {fmt(r["WI"], 4)} | {r["A-OSE"]} | '
                         f'{fmt(r["mAP_PK"])} | {fmt(r["mAP_CK"])} | {fmt(r["mAP_Both"])} |')
    md = '# DSEC-OWOD re-evaluation with U-AP (test split, AP50, max 100 detections per image)\n\n' + '\n'.join(lines) + '\n'
    open(osp.join(ROOT, 'runs', 'diag', 'owod_reeval.md'), 'w', encoding='utf-8').write(md)
    print(md, flush=True)


if __name__ == '__main__':
    main()
