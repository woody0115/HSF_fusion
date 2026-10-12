"""Open-world head for the dual-stream detector (YOLO-UniOW style, adapted to HSF).

The last 1x1 classification conv of every Detect scale is replaced by a contrastive classifier:
    e(x)    = BN(Conv1x1(x))                         per-anchor region embedding (D = 512)
    logit_k = exp(s) * <e(x), T_k / |T_k|> + b        one logit per active text embedding T_k
(BN-contrastive head of YOLO-World / YOLO-UniOW). The text embeddings live in one shared table:
    rows 0..7  DSEC classes (pedestrian, rider, car, bus, truck, bicycle, motorcycle, train),
               initialised from the CLIP text encoder of YOLO-UniOW (configs/owod_text_embeds.npz)
    row  8     "object"  wildcard  (class-agnostic objectness, tuned on all known boxes)
    row  9     "unknown" wildcard  (trained from the object wildcard's pseudo labels)
`set_active(model, ids)` chooses which rows form the output channels (and their order); rows that
are not in `trainable` receive no gradient, so previously known classes stay frozen.
"""
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBJ, UNK = 8, 9
ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))


class ClassTable(nn.Module):
    def __init__(self, embeds):
        super().__init__()
        self.emb = nn.Parameter(torch.as_tensor(embeds, dtype=torch.float32).clone())
        self.active = [2]
        self.register_buffer('train_mask', torch.zeros(len(embeds), 1))
        self.emb.register_hook(lambda g: g * self.train_mask)

    def set_trainable(self, rows):
        self.train_mask.zero_()
        self.train_mask[list(rows)] = 1.0

    def weights(self):
        return F.normalize(self.emb[self.active], dim=-1)


class ContrastCls(nn.Module):
    def __init__(self, c_in, table, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(c_in, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.table = [table]                          # list: do not register the shared table three times
        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.full((1,), -4.6))   # prior probability ~0.01

    def forward(self, x):
        e = self.bn(self.proj(x))
        w = self.table[0].weights().to(e.dtype)
        return torch.einsum('bchw,kc->bkhw', e, w) * self.logit_scale.exp() + self.bias


def load_text_table(path=None):
    d = np.load(path or osp.join(ROOT, 'configs', 'owod_text_embeds.npz'))
    names = list(d['names'])
    e = d['embeds']                                   # rows: 8 classes + "object"
    assert names[:8] == ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']
    return np.concatenate([e[:8], e[8:9], e[8:9]], 0)     # "unknown" starts as a copy of "object"


def to_open_world(model, table_init=None):
    """Replace the Detect classifier of a DualYOLOv8 with the contrastive open-world head (in place)."""
    det = model.model[-1]
    table = ClassTable(load_text_table() if table_init is None else table_init)
    model.class_table = table
    for seq in det.cv3:
        c_in = seq[-1].in_channels
        seq[-1] = ContrastCls(c_in, table)
    set_active(model, [2])
    model.names = {i: n for i, n in enumerate(['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle',
                                               'motorcycle', 'train', 'object', 'unknown'])}
    return model


def set_active(model, rows):
    """Output channel j of the detector = table row rows[j]."""
    det = model.model[-1]
    model.class_table.active = list(rows)
    det.nc = len(rows)
    det.no = det.nc + det.reg_max * 4
    det.shape = None                                  # force anchor rebuild in Detect._inference
    model.nc = det.nc
