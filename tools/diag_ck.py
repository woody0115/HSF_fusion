"""Diagnostics for the DSEC-OWOD new-class (CK) collapse, asked by the planning agent
(questions_for_training_agent.md, section B) plus the step-8 gate distribution (D3).

  b3    class-agnostic box recall of the T1 model per class (object score / max over channels),
        IoU 0.5 and 0.7, top 100 per image; oracle best IoU over all 8,400 decoded boxes      (H3)
  b4    embedding space of the T1 model: class-mean cosine of e(x) = BN(Conv1x1(x)), PCA of all
        samples, text-embedding cosine, object vs unknown rows per Task
  b1    linear probe on the frozen T1 features: train ContrastCls (proj, BN, scale, bias) + the 7
        class embeddings with all 7 classes labelled; control: neck + whole cls branch too   (H1 vs H2)
  b2    T3 embedding tuning from the #11 T2 state for 1/5/10 epochs at lr 1e-3 and 1e-2       (H2)
  gate  per-image illumination gate of #9 / #10 per lighting bucket (mean, std, percentiles)

Every command writes runs/diag/<command>.json (and prints the key numbers).
Run:  .venv\\Scripts\\python.exe tools\\diag_ck.py b3   (b4, b1, b2, gate)
"""
import argparse
import json
import os
import os.path as osp
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

from hsf.data.dataset import CFG, DualDSEC  # noqa: E402
from hsf.owod.model import OBJ, UNK, ContrastCls, set_active  # noqa: E402
from hsf.owod.train import evaluate_owod, loader, tune_embeddings  # noqa: E402
from tools.owod_event_rescore import load_task  # noqa: E402

NAMES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle']
OUT = osp.join(ROOT, 'runs', 'diag')
RUN11 = osp.join(ROOT, 'runs', 'step7', 'owod_add')
RUN12 = osp.join(ROOT, 'runs', 'step7', 'owod_add_evtign')


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(osp.join(OUT, f'{name}.json'), 'w'), indent=1)
    print(f'-> runs/diag/{name}.json', flush=True)


def test_loader(workers, batch=16):
    return DataLoader(DualDSEC('test'), batch_size=batch, num_workers=workers, pin_memory=True,
                      collate_fn=DualDSEC.collate)


def gts_of(b, i, S=640):
    m = b['batch_idx'] == i
    return b['cls'][m].squeeze(-1).long(), ops.xywh2xyxy(b['bboxes'][m] * S)


# ----------------------------------------------------------------------------------------------- b3
@torch.no_grad()
def b3(a, dev):
    out = {}
    for tag, run in (('#11', RUN11), ('#12', RUN12)):
        model = load_task(run, 'T1', 'add', dev)
        set_active(model, [2, OBJ, UNK])                       # channels: car, object, unknown
        st = {c: {'n': 0, 'obj05': 0, 'obj07': 0, 'max05': 0, 'max07': 0, 'oracle_iou': 0.0,
                  'oracle05': 0, 'oracle07': 0} for c in range(7)}
        t0 = time.time()
        for bi, b in enumerate(test_loader(a.workers)):
            with torch.autocast('cuda'):
                y, _ = model(b['rgb'].to(dev), b['evt'].to(dev))
            y = y.float()
            boxes = ops.xywh2xyxy(y[:, :4].transpose(1, 2))    # (B, A, 4)
            dets = {}
            for v, s in (('obj', y[:, 4 + 1]), ('max', y[:, 4:7].max(1).values)):
                dets[v] = ops.non_max_suppression(torch.cat([y[:, :4], s[:, None]], 1), 0.001, 0.7, max_det=100)
            for i in range(len(y)):
                cls, g = gts_of(b, i)
                if not len(g):
                    continue
                g = g.to(dev)
                best = box_iou(g, boxes[i]).max(1).values                       # oracle over all anchors
                hit = {}
                for v in ('obj', 'max'):
                    d = dets[v][i]
                    m = box_iou(g, d[:, :4]).max(1).values if len(d) else torch.zeros(len(g), device=dev)
                    hit[v] = m
                for k, c in enumerate(cls.tolist()):
                    if c > 6:
                        continue
                    s_ = st[c]
                    s_['n'] += 1
                    for v in ('obj', 'max'):
                        s_[f'{v}05'] += int(hit[v][k] >= 0.5)
                        s_[f'{v}07'] += int(hit[v][k] >= 0.7)
                    s_['oracle_iou'] += float(best[k])
                    s_['oracle05'] += int(best[k] >= 0.5)
                    s_['oracle07'] += int(best[k] >= 0.7)
            if bi % 200 == 0:
                print(f'  b3 {tag} {bi}/{869} ({time.time() - t0:.0f} s)', flush=True)
        res = {}
        for c, s_ in st.items():
            n = max(s_['n'], 1)
            res[NAMES[c]] = {'n_gt': s_['n'],
                             **{k: round(100 * s_[k] / n, 1) for k in ('obj05', 'obj07', 'max05', 'max07',
                                                                      'oracle05', 'oracle07')},
                             'oracle_mean_iou': round(s_['oracle_iou'] / n, 3)}
        out[tag] = res
        print(tag, json.dumps(res, ensure_ascii=False), flush=True)
    save('b3_box_recall', out)


# ----------------------------------------------------------------------------------------------- b4
@torch.no_grad()
def b4(a, dev):
    model = load_task(RUN11, 'T1', 'add', dev)
    set_active(model, [2, OBJ, UNK])
    det = model.model[-1]
    feats = [None] * 3
    hooks = [seq[-1].bn.register_forward_hook(lambda m, i, o, k=k: feats.__setitem__(k, o.float()))
             for k, seq in enumerate(det.cv3)]
    cap, per = 3000, {c: [] for c in range(8)}                 # 7 classes + background (index 7)
    strides = [8, 16, 32]
    random.seed(0)
    for bi, b in enumerate(test_loader(a.workers)):
        with torch.autocast('cuda'):
            model(b['rgb'].to(dev), b['evt'].to(dev))
        for i in range(b['rgb'].shape[0]):
            cls, g = gts_of(b, i)
            for c, box in zip(cls.tolist(), g.tolist()):
                if c > 6 or len(per[c]) >= cap:
                    continue
                w, h = box[2] - box[0], box[3] - box[1]
                k = 0 if (w * h) ** 0.5 < 64 else (1 if (w * h) ** 0.5 < 128 else 2)
                cx, cy = int((box[0] + box[2]) / 2 // strides[k]), int((box[1] + box[3]) / 2 // strides[k])
                f = feats[k]
                cx, cy = min(cx, f.shape[3] - 1), min(cy, f.shape[2] - 1)
                per[c].append(f[i, :, cy, cx].cpu())
            if len(per[7]) < cap:                              # background: random P4 cells outside every box
                for _ in range(3):
                    cx, cy = random.randrange(40), random.randrange(10, 30)
                    px, py = (cx + 0.5) * 16, (cy + 0.5) * 16
                    if all(not (b0 <= px <= b2 and b1 <= py <= b3) for b0, b1, b2, b3 in g.tolist()):
                        per[7].append(feats[1][i, :, cy, cx].cpu())
        if all(len(v) >= cap for v in per.values()) or bi >= 868:
            break
    for h in hooks:
        h.remove()
    names = NAMES + ['background']
    X = {names[c]: torch.stack(v) for c, v in per.items() if v}
    means = torch.stack([F_norm(x.mean(0)) for x in X.values()])
    cos = (means @ means.T).tolist()
    allx = torch.cat([X[n] for n in NAMES if n in X])
    allx = allx - allx.mean(0)
    ev = torch.linalg.svdvals(allx) ** 2
    ev = (ev / ev.sum()).cumsum(0)
    allb = torch.cat(list(X.values()))
    allb = allb - allb.mean(0)
    evb = torch.linalg.svdvals(allb) ** 2
    evb = (evb / evb.sum()).cumsum(0)
    # how well each class-mean e(x) points at the class text embeddings of the T1 table (CLIP init)
    tab = F_norm(model.class_table.emb.detach().float().cpu())
    cos_text = (means @ tab[list(range(7)) + [OBJ, UNK]].T).tolist()
    d = np.load(osp.join(ROOT, 'configs', 'owod_text_embeds.npz'))
    te = torch.from_numpy(np.concatenate([d['embeds'], d['object_tuned_uniow']])).float()
    te = F_norm(te)
    tnames = list(d['names']) + ['object_tuned_uniow']
    rows = {}
    for tag, run in (('#11', RUN11), ('#12', RUN12)):
        for t in ('T1', 'T2', 'T3', 'T4'):
            e = torch.load(osp.join(run, f'{t}.pt'), map_location='cpu')['model']['class_table.emb'].float()
            o, u = e[OBJ], e[UNK]
            rows[f'{tag} {t}'] = {'cos': round(float(F_norm(o) @ F_norm(u)), 4), 'l2_raw': round(float((o - u).norm()), 4),
                                  'l2_normalized': round(float((F_norm(o) - F_norm(u)).norm()), 4),
                                  'norm_object': round(float(o.norm()), 3), 'norm_unknown': round(float(u.norm()), 3)}
    out = {'samples': {n: len(x) for n, x in X.items()},
           'class_mean_cosine': {'names': list(X), 'matrix': [[round(v, 3) for v in r] for r in cos]},
           'pca_cumulative_7classes': {k: round(float(ev[k - 1]), 4) for k in (1, 5, 20)},
           'pca_cumulative_with_background': {k: round(float(evb[k - 1]), 4) for k in (1, 5, 20)},
           'mean_vs_T1_table_cosine': {'rows': list(X), 'cols': NAMES + ['object', 'unknown'],
                                       'matrix': [[round(v, 3) for v in r] for r in cos_text]},
           'text_cosine': {'names': tnames, 'matrix': [[round(float(v), 3) for v in r] for r in (te @ te.T)]},
           'object_vs_unknown': rows}
    print(json.dumps({k: out[k] for k in ('samples', 'pca_cumulative_7classes', 'pca_cumulative_with_background')}),
          flush=True)
    save('b4_embedding', out)


def F_norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ----------------------------------------------------------------------------------------------- b1
class _Head(nn.Module):
    """Stand-in so v8DetectionLoss can score the 7 output channels with chosen loss gains."""

    def __init__(self, det, nc, device, box, cls, dfl):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
        self.model = [SimpleNamespace(stride=det.stride, nc=nc, reg_max=det.reg_max)]
        self.args = SimpleNamespace(box=box, cls=cls, dfl=dfl)


def b1(a, dev):
    rows = list(range(7))
    out = {}
    for variant in a.variants:
        model = load_task(osp.join(ROOT, 'runs', 'step7', a.run), 'T1', 'add', dev)
        set_active(model, rows)
        det = model.model[-1]
        for p in model.parameters():
            p.requires_grad = False
        heads = [seq[-1] for seq in det.cv3]
        train_mods = list(heads)
        groups = [{'params': [p for h in heads for p in h.parameters()], 'lr': 1e-3, 'weight_decay': 5e-4},
                  {'params': [model.class_table.emb], 'lr': 1e-3, 'weight_decay': 0.0}]
        if variant == 'neck':                                   # control: neck + whole cls branch
            neck = list(model.model[:-1]) + [m for seq in det.cv3 for m in seq[:-1]]
            train_mods += neck
            groups.append({'params': [p for m in neck for p in m.parameters()], 'lr': 1e-4, 'weight_decay': 5e-4})
        for g in groups:
            for p in g['params']:
                p.requires_grad = True
        model.class_table.set_trainable(rows)
        gains = dict(box=0.0, cls=1.0, dfl=0.0) if variant == 'head' else dict(box=7.5, cls=0.5, dfl=1.5)
        crit = v8DetectionLoss(_Head(det, 7, dev, **gains))
        opt = torch.optim.AdamW(groups)
        scaler = torch.amp.GradScaler('cuda')
        dl = loader('train', 16, a.workers, classes=rows, label_map={c: c for c in rows}, augment=True)
        hist, evals = [], {}

        def set_modes():
            model.eval()
            for m in train_mods:
                m.train()

        for ep in range(1, a.epochs + 1):
            set_modes()
            t0, run = time.time(), 0.0
            for bi, b in enumerate(dl):
                with torch.autocast('cuda'):
                    _, maps = model(b['rgb'].to(dev, non_blocking=True), b['evt'].to(dev, non_blocking=True))
                    loss = crit(maps, b)[0].sum()
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 10.0)
                scaler.step(opt)
                scaler.update()
                run = 0.9 * run + 0.1 * float(loss) if bi else float(loss)
                if bi % 400 == 0:
                    print(f'  b1 {variant} epoch {ep} {bi}/{len(dl)} loss {run:.3f} ({(time.time() - t0) / 60:.1f} min)',
                          flush=True)
                    hist.append((ep, bi, round(run, 4)))
            r = evaluate_owod(model, dev, [], rows, [], workers=a.workers)
            evals[ep] = {'run': a.run, 'mAP_all7': round(r['mAP_CK'], 2),
                         'AP50_per_class': {NAMES[c]: v for c, v in r['AP50_per_class'].items()}}
            print(f'b1 {variant} epoch {ep}: {evals[ep]}', flush=True)
            set_active(model, rows)
        out[variant] = {'evals': evals, 'loss_history': hist}
        save('b1_linear_probe' if a.run == 'owod_add' else f'b1_linear_probe_{a.run}', out)


# ----------------------------------------------------------------------------------------------- b2
def b2(a, dev):
    prev, cur, unknown = [2, 3, 4], [0, 1], [5, 6]
    rows = prev + cur + [OBJ, UNK]
    out = {}
    for lr in a.lrs:
        model = load_task(RUN11, 'T2', 'add', dev)
        evals, hist = {}, []

        def on_epoch(ep):
            if ep in a.checkpoints:
                r = evaluate_owod(model, dev, prev, cur, unknown, workers=a.workers)
                evals[ep] = {k: (round(r[k], 2) if isinstance(r[k], float) else r[k])
                             for k in ('mAP_CK', 'mAP_PK', 'mAP_Both', 'U-Recall', 'U-Recall@30', 'WI', 'A-OSE')}
                evals[ep]['AP50_per_class'] = {NAMES[c]: v for c, v in r['AP50_per_class'].items()}
                print(f'b2 lr {lr} epoch {ep}: {evals[ep]}', flush=True)
                set_active(model, rows)                       # evaluation changed the active rows
                model.eval()
                save('b2_tune_more', {**out, str(lr): {'evals': evals, 'loss_history': hist}})

        tune_embeddings(model, dev, prev, cur, epochs=max(a.checkpoints), lr=lr, workers=a.workers,
                        on_epoch=on_epoch, history=hist)
        out[str(lr)] = {'evals': evals, 'loss_history': hist}
        save('b2_tune_more', out)


# --------------------------------------------------------------------------------------------- gate
@torch.no_grad()
def gate(a, dev):
    from tools.eval_conditions import bucket_of, build
    lighting = json.load(open(osp.join(CFG['processed_root'], 'test_lighting.json')))
    out = {}
    for key in ('iaf_add', 'iaf_af'):
        model = build(key, dev)
        ds = DualDSEC('test')
        per = {b: [] for b in ('day', 'dim', 'night')}
        chan_std = {b: [] for b in ('day', 'dim', 'night')}
        k = 0
        for b in DataLoader(ds, batch_size=16, num_workers=a.workers, collate_fn=DualDSEC.collate):
            with torch.autocast('cuda'):
                model(b['rgb'].to(dev), b['evt'].to(dev))
            g = torch.stack([f.last_gate for f in model.fusion]).float().cpu()     # (3 scales, B) image means
            for i in range(g.shape[1]):
                seq_dir, _, idx = ds.items[k]
                k += 1
                bk = bucket_of(lighting[osp.basename(seq_dir)][str(idx)][0])
                per[bk].append(g[:, i].tolist())
        res = {}
        for bk, v in per.items():
            v = np.array(v)                                     # (N, 3 scales)
            m = v.mean(1)
            res[bk] = {'n': len(v), 'mean': round(float(m.mean()), 4), 'std_over_images': round(float(m.std()), 4),
                       'p5': round(float(np.percentile(m, 5)), 4), 'p95': round(float(np.percentile(m, 95)), 4),
                       'per_scale_mean': [round(float(x), 4) for x in v.mean(0)],
                       'per_scale_std': [round(float(x), 4) for x in v.std(0)]}
        out[key] = res
        print(key, json.dumps(res), flush=True)
    save('gate_distribution', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['b1', 'b2', 'b3', 'b4', 'gate'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=3, help='b1 epochs')
    ap.add_argument('--variants', nargs='*', default=['head', 'neck'], help='b1 variants')
    ap.add_argument('--run', default='owod_add', help='b1: OWOD run under runs/step7 whose T1.pt is probed '
                    '(owod_add = #11, trained with the old event input format; owod_add_evtign = #12)')
    ap.add_argument('--lrs', type=float, nargs='*', default=[1e-3, 1e-2], help='b2 learning rates')
    ap.add_argument('--checkpoints', type=int, nargs='*', default=[1, 5, 10], help='b2 epochs to evaluate')
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device('cuda')
    {'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'gate': gate}[a.cmd](a, dev)


if __name__ == '__main__':
    main()
