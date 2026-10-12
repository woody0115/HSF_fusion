"""DSEC-OWOD training stages (efficient YOLO-UniOW-style protocol).

Task 1 (known: car)
  base      : full training of the dual-stream detector with the contrastive head on T1 labels only
              (backbones start from COCO-pretrained YOLOv8s, never from our all-class models)
  wildcard  : model frozen, tune the "object" embedding with every known box as one class
Every task t (T1..T4)
  embeddings: model frozen; learn the embeddings of the current classes (classification loss on the
              current task's labels) and the "unknown" embedding, whose soft targets are the object
              wildcard's scores on anchors that are neither current-class ground truth nor confident
              previously-known predictions (YOLO-UniOW Eq. 4-5, plus the previously-known exclusion
              because earlier classes are unlabelled in later tasks). Earlier class embeddings stay frozen,
              so nothing is forgotten and no exemplar replay is needed.
  evaluate  : U-Recall, WI, A-OSE, PK / CK / Both mAP (hsf/owod/metrics.py)
"""
import json
import os.path as osp
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ultralytics.utils import ops
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import box_iou

from hsf.data.dataset import DualDSEC
from hsf.owod.event import EventIgnoreLoss, rescore_unknown
from hsf.owod.metrics import OWODEvaluator
from hsf.owod.model import OBJ, UNK, set_active


class _HeadView(nn.Module):
    """Minimal stand-in so v8DetectionLoss can score a subset of the output channels."""

    def __init__(self, det, nc, device):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
        self.model = [SimpleNamespace(stride=det.stride, nc=nc, reg_max=det.reg_max)]
        self.args = SimpleNamespace(box=0.0, cls=1.0, dfl=0.0)       # frozen boxes: classification only


def _slice_channels(maps, reg_max, ch):
    """Keep box channels + the classification channels in `ch` (list of output channel indices)."""
    box = 4 * reg_max
    idx = list(range(box)) + [box + c for c in ch]
    return [m[:, idx] for m in maps]


def freeze_all_but_table(model):
    for p in model.parameters():
        p.requires_grad = False
    model.class_table.emb.requires_grad = True
    model.eval()                     # frozen BN statistics


def loader(split, batch, workers, classes=None, label_map=None, augment=False, sequences=None):
    ds = DualDSEC(split, augment=augment, classes=classes, label_map=label_map, sequences=sequences)
    return DataLoader(ds, batch_size=batch, shuffle=split == 'train', drop_last=split == 'train',
                      num_workers=workers, pin_memory=True, collate_fn=DualDSEC.collate,
                      persistent_workers=workers > 0)


def tune_object(model, device, known, epochs=1, batch=16, workers=8, lr=1e-3, sequences=None, max_batches=None,
                event_ignore=0.0):
    """Object wildcard: every box of a known class is the single class 'object'.
    event_ignore > 0: event-active background anchors are not negatives (EventIgnoreLoss)."""
    freeze_all_but_table(model)
    set_active(model, [OBJ])
    model.class_table.set_trainable([OBJ])
    head = _HeadView(model.model[-1], 1, device)
    crit = EventIgnoreLoss(head, q=event_ignore) if event_ignore > 0 else v8DetectionLoss(head)
    opt = torch.optim.AdamW([model.class_table.emb], lr=lr, weight_decay=0.0)
    dl = loader('train', batch, workers, classes=known, label_map={c: 0 for c in known}, augment=True,
                sequences=sequences)
    _loop(model, dl, device, opt, epochs, max_batches, 'object wildcard',
          lambda maps, b, y: crit(maps, b)[0].sum())


def tune_embeddings(model, device, prev, cur, epochs=1, batch=16, workers=8, lr=1e-3, sigma_iou=0.5,
                    known_conf=0.2, sequences=None, max_batches=None):
    """Learn current-class embeddings (cls loss) + 'unknown' (soft targets from the object wildcard)."""
    freeze_all_but_table(model)
    rows = prev + cur + [OBJ, UNK]                    # output channels during tuning
    set_active(model, rows)
    model.class_table.set_trainable(cur + [UNK])
    det = model.model[-1]
    ch_cur = [rows.index(c) for c in cur]
    ch_prev = [rows.index(c) for c in prev]
    ch_obj, ch_unk = rows.index(OBJ), rows.index(UNK)
    crit = v8DetectionLoss(_HeadView(det, len(cur), device))
    opt = torch.optim.AdamW([model.class_table.emb], lr=lr, weight_decay=0.0)
    dl = loader('train', batch, workers, classes=cur, label_map={c: i for i, c in enumerate(cur)}, augment=True,
                sequences=sequences)
    box = 4 * det.reg_max

    def loss_fn(maps, b, y):
        l_k = crit(_slice_channels(maps, det.reg_max, ch_cur), b)[0].sum()
        # unknown: soft targets = object score, removed where the anchor box overlaps current GT or a
        # previously-known class is confidently predicted (earlier classes are unlabelled here)
        B = y.shape[0]
        scores = y[:, 4:]                                             # (B, nc, A) sigmoid scores
        boxes = ops.xywh2xyxy(y[:, :4].transpose(1, 2))               # (B, A, 4) pixels
        target = scores[:, ch_obj].detach().clone()                   # (B, A)
        S = maps[0].shape[-1] * det.stride[0]
        for i in range(B):
            g = b['bboxes'][b['batch_idx'] == i]
            if len(g):
                iou = box_iou(boxes[i], ops.xywh2xyxy(g.to(boxes.device) * S))
                target[i][iou.max(1).values >= sigma_iou] = 0
        if ch_prev:
            target[scores[:, ch_prev].detach().max(1).values >= known_conf] = 0
        logit_unk = torch.cat([m.flatten(2) for m in maps], 2)[:, box + ch_unk]     # (B, A)
        l_u = F.binary_cross_entropy_with_logits(logit_unk.float(), target.float(), reduction='sum')
        return l_k + l_u / max(float(target.sum()), 1.0) * B

    _loop(model, dl, device, opt, epochs, max_batches, f'embeddings cur={cur}', loss_fn)


def _loop(model, dl, device, opt, epochs, max_batches, desc, loss_fn):
    nb = len(dl) if max_batches is None else min(len(dl), max_batches)
    for ep in range(epochs):
        t0, run = time.time(), 0.0
        for bi, b in enumerate(dl):
            if bi >= nb:
                break
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                y, maps = model(b['rgb'].to(device, non_blocking=True), b['evt'].to(device, non_blocking=True))
                loss = loss_fn(maps, b, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run = 0.9 * run + 0.1 * float(loss) if bi else float(loss)
            if bi % 200 == 0:
                print(f'  {desc}: epoch {ep + 1}/{epochs} {bi}/{nb} loss {run:.4f} ({(time.time() - t0) / 60:.1f} min)',
                      flush=True)


@torch.no_grad()
def evaluate_owod(model, device, prev, cur, unknown, batch=16, workers=8, max_det=100, known_conf=0.2,
                  tau=0.99, sequences=None, max_batches=None, alphas=(0,)):
    """Open-world evaluation on the test split with all ground truth (later classes = unknown).
    alphas: event rescoring strengths for the unknown channel (hsf/owod/event.py, option 1); the model
    runs once per batch and every alpha gets its own evaluator. Returns the summary of alphas[0], or
    {alpha: summary} when several alphas are given."""
    model.eval()
    known = prev + cur
    rows = known + [UNK]
    set_active(model, rows)
    evs = {al: OWODEvaluator(prev, cur, unknown) for al in alphas}
    dl = loader('test', batch, workers, sequences=sequences)
    t0 = time.time()
    for bi, b in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            evt = b['evt'].to(device, non_blocking=True)
            y, _ = model(b['rgb'].to(device, non_blocking=True), evt)
        S = b['rgb'].shape[-1]
        for al, ev in evs.items():
            # clone: Ultralytics NMS converts the boxes of its input to xyxy in place
            dets = ops.non_max_suppression(rescore_unknown(y.float().clone(), evt, len(known), al), 0.001, 0.7,
                                           multi_label=True, max_det=max_det)
            _owod_update(ev, dets, b, known, known_conf, tau, S)
        if bi % 200 == 0:
            print(f'  owod eval {bi + 1}/{len(dl)} ({time.time() - t0:.0f} s)', flush=True)
    out = {al: ev.summarize() for al, ev in evs.items()}
    return out[alphas[0]] if len(alphas) == 1 else out


def _owod_update(ev, dets, b, known, known_conf, tau, S):
    """Relabel NMS output (known class ids / unknown = -1), drop duplicate unknowns, update ev."""
    for i, d in enumerate(dets):
        lab = torch.full((len(d),), -1.0, device=d.device)
        k = d[:, 5] < len(known)
        lab[k] = torch.tensor(known, device=d.device, dtype=torch.float)[d[k, 5].long()]
        d = torch.cat([d[:, :5], lab[:, None]], 1)
        # YOLO-UniOW unknown filtering (Eq. 6): drop unknown boxes duplicating a confident known box
        unk, conf_known = d[:, 5] == -1, (d[:, 5] >= 0) & (d[:, 4] > known_conf)
        if unk.any() and conf_known.any():
            dup = (box_iou(d[unk, :4], d[conf_known, :4]) >= tau).any(1)
            keep = torch.ones(len(d), dtype=torch.bool, device=d.device)
            keep[unk.nonzero().flatten()[dup]] = False
            d = d[keep]
        m = b['batch_idx'] == i
        gt = torch.cat([b['cls'][m], ops.xywh2xyxy(b['bboxes'][m] * S)], 1)
        ev.update(d, gt)


def save_task(run_dir, task, res, model):
    torch.save({'model': model.state_dict(), 'active': model.class_table.active, 'task': task,
                'fusion': model.fusion_type}, osp.join(run_dir, f'{task}.pt'))
    with open(osp.join(run_dir, f'{task}_metrics.json'), 'w') as f:
        json.dump({k: (v if not isinstance(v, float) or np.isfinite(v) else None) for k, v in res.items()}, f, indent=1)
