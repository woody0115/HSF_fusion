"""Step 7: DSEC-OWOD, open-world incremental detection over Task 1-4 (configs/owod_tasks.yaml).

Efficient YOLO-UniOW-style protocol (details in hsf/owod/train.py):
  T1 : full training of the dual-stream detector with the contrastive open-world head on car only
       (both backbones from COCO-pretrained YOLOv8s), then tune the "object" wildcard
  Tt : model frozen; learn the new classes' text embeddings + the "unknown" embedding; evaluate
Writes runs/step7/<name>/results.md with the YOLO-UniOW Tab. 2/3 columns.

Run:  .venv\\Scripts\\python.exe tools\\train_owod.py --fusion add            (full run)
      .venv\\Scripts\\python.exe tools\\train_owod.py --fusion add --smoke    (pipeline check)
      step 9 option 3 (event-active background is not taught as background in T1):
      train_owod.py --fusion add --event-ignore 0.1 --name owod_add_evtign
            [--only-if runs/step9_evt_unknown/owod_add_rescore.json]   (skip unless option 1 had an effect)
"""
import argparse
import json
import os.path as osp
import sys

import torch
import yaml

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from hsf.engine import start_logging, train  # noqa: E402
from hsf.models.dual_yolov8 import DualYOLOv8  # noqa: E402
from hsf.owod.model import set_active, to_open_world  # noqa: E402
from hsf.owod.train import evaluate_owod, save_task, tune_embeddings, tune_object  # noqa: E402

NAMES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']
PRESENT = [0, 1, 2, 3, 4, 5, 6]                    # class 7 (train) has no boxes in DSEC-Det


def fmt(v, nd=1):
    return '-' if v is None or (isinstance(v, float) and v != v) else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fusion', default='add', choices=['add', 'af'],
                    help='single-frame fusions only (the frozen-model tuning stages assume no temporal state)')
    ap.add_argument('--base-epochs', type=int, default=10)
    ap.add_argument('--tune-epochs', type=int, default=1)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--name', default=None)
    ap.add_argument('--event-ignore', type=float, default=0.0,
                    help='T1 base + object wildcard: ignore background anchors in this top share of event density')
    ap.add_argument('--only-if', default=None, help='json with "effect": run only if it is true')
    ap.add_argument('--split', default='val', choices=['val', 'legacy'],
                    help='val: train/tune on train_core and select the T1 base checkpoint on the validation '
                         'sequences; legacy: all 47 sequences, T1 base selected on test (#11, #12)')
    ap.add_argument('--resume', action='store_true',
                    help='continue an interrupted run: T1 base from its resume.pt, finished Tasks (T<k>.pt) are reused')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.only_if:
        eff = json.load(open(a.only_if)).get('effect', False) if osp.isfile(a.only_if) else False
        print(f'{a.only_if}: effect = {eff}', flush=True)
        if not eff:
            print('option 1 showed no effect -> option 3 skipped', flush=True)
            return

    name = a.name or f'owod_{a.fusion}' + ('_smoke' if a.smoke else '')
    run_dir = osp.join(ROOT, 'runs', 'step7', name)
    start_logging(run_dir, ROOT)
    tasks = yaml.safe_load(open(osp.join(ROOT, 'configs', 'owod_tasks.yaml')))['tasks']
    smoke = dict(sequences=['zurich_city_16_a', 'zurich_city_17_a'], max_batches=40) if a.smoke else {}
    sm_val = dict(sequences=['thun_02_a'], max_batches=40) if a.smoke else {}
    print(f'config: {vars(a)}\ntasks: {tasks}', flush=True)

    coco = osp.join(ROOT, 'weights', 'yolov8s.pt')
    model = to_open_world(DualYOLOv8(nc=80, fusion=a.fusion, rgb_init=coco, evt_init=coco))
    device = torch.device('cuda')
    model.to(device)

    tr_split = 'train' if (a.split == 'legacy' or a.smoke) else 'train_core'
    val_split = 'test' if (a.split == 'legacy' or a.smoke) else 'val'
    prev, rows = [], []
    for tname, tcfg in tasks.items():
        cur = list(tcfg['classes'])
        unknown = [c for c in PRESENT if c not in prev + cur]
        print(f'\n===== {tname}: previously known {[NAMES[c] for c in prev]}, current {[NAMES[c] for c in cur]}, '
              f'unknown {[NAMES[c] for c in unknown]} =====', flush=True)
        done = osp.join(run_dir, f'{tname}.pt')
        if a.resume and osp.isfile(done) and osp.isfile(osp.join(run_dir, f'{tname}_metrics.json')):
            model.load_state_dict(torch.load(done, map_location=device)['model'])
            res = json.load(open(osp.join(run_dir, f'{tname}_metrics.json')))
            res = {k: (float('nan') if v is None else v) for k, v in res.items()}
            print(f'{tname}: finished earlier, reusing {done}', flush=True)
            rows.append((tname, res))
            prev = prev + cur
            continue
        if tname == 'T1':
            # full training on the first task's classes only
            set_active(model, cur)
            model.class_table.set_trainable(cur)
            all_names = model.names
            model.names = {i: NAMES[c] for i, c in enumerate(cur)}
            base_dir = osp.join(run_dir, 'T1_base')
            import os
            os.makedirs(base_dir, exist_ok=True)
            train(model, base_dir, epochs=1 if a.smoke else a.base_epochs, batch=8, lr=0.01, workers=a.workers,
                  classes=cur, label_map={c: i for i, c in enumerate(cur)},
                  sequences_train=smoke.get('sequences'), sequences_val=sm_val.get('sequences'),
                  max_train_batches=smoke.get('max_batches'), event_ignore=a.event_ignore,
                  train_split=tr_split, val_split=val_split, resume=a.resume)
            model.load_state_dict(torch.load(osp.join(base_dir, 'best.pt'), map_location=device)['model'])
            model.names = all_names
            tune_object(model, device, known=cur, epochs=a.tune_epochs, workers=a.workers,
                        event_ignore=a.event_ignore, train_split=tr_split, **smoke)
        tune_embeddings(model, device, prev, cur, epochs=a.tune_epochs, workers=a.workers, train_split=tr_split,
                        **smoke)
        res = evaluate_owod(model, device, prev, cur, unknown, workers=a.workers, **sm_val)
        save_task(run_dir, tname, res, model)
        print(f'{tname}: U-Recall {fmt(res["U-Recall"])}  WI {fmt(res["WI"], 4)}  A-OSE {res["A-OSE"]}  '
              f'mAP PK {fmt(res["mAP_PK"])} CK {fmt(res["mAP_CK"])} Both {fmt(res["mAP_Both"])}  '
              f'per-class AP50 { {NAMES[c]: v for c, v in res["AP50_per_class"].items()} }', flush=True)
        rows.append((tname, res))
        prev = prev + cur

    lines = ['| Task | WI ↓ | A-OSE ↓ | U-Recall ↑ | U-Recall@30 | mAP PK | mAP CK | mAP Both |',
             '|---|---|---|---|---|---|---|---|']
    for tname, r in rows:
        lines.append(f'| {tname} | {fmt(r["WI"], 4)} | {r["A-OSE"]} | {fmt(r["U-Recall"])} | {fmt(r.get("U-Recall@30"))} | '
                     f'{fmt(r["mAP_PK"])} | {fmt(r["mAP_CK"])} | {fmt(r["mAP_Both"])} |')
    table = '\n'.join(lines)
    with open(osp.join(run_dir, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(f'# DSEC-OWOD results ({a.fusion})\n\nmAP = AP50 (VOC-style, OWOD convention).\n\n{table}\n')
    print('\n' + table, flush=True)


if __name__ == '__main__':
    main()
