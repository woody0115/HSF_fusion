"""Paired RGB + GEAR dataset for DSEC-Det (dual-stream training).

Each sample is one labelled frame with both modalities available:
    rgb   float32 (3, S, S) in [0, 1], RGB order
    evt   float32 (3, S, S) in [0, 1]: the GEAR png prepared exactly like Ultralytics prepares any image
          (BGR->RGB flip, /255, pad 114), i.e. the input format the step-3 event model was trained on;
          the event backbone is initialised from that model, so the formats must match
    labels float32 (n, 5): cls, xc, yc, w, h normalized to the S x S network input
The 640x480 frames are letterboxed to S x S (default 640). Training augmentation applies the
*same* geometric transform to both modalities (random scale / translate / horizontal flip, as
Ultralytics' defaults) and HSV jitter to RGB only (GEAR channels are not colours).
"""
import glob
import json
import os.path as osp
import random

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from hsf.data.gear import GEAR

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
CFG = yaml.safe_load(open(osp.join(ROOT, 'configs', 'paths.yaml')))
SPLITS = yaml.safe_load(open(osp.join(ROOT, 'configs', 'splits.yaml')))
VAL_SEQUENCES = set(SPLITS['val_sequences'])
RGB_FILL = (114, 114, 114)
EVT_FILL = (114, 114, 114)        # same letterbox fill Ultralytics used when training the event model


def split_filter(split):
    """split name -> (folder under processed_root, keep set or None, drop set or None).
    'train' = all 47 training sequences (versions #1-#12), 'train_core' = train minus the validation
    sequences of configs/splits.yaml, 'val' = those sequences, 'test' = the 13 test sequences."""
    if split == 'val':
        return 'train', VAL_SEQUENCES, None
    if split == 'train_core':
        return 'train', None, VAL_SEQUENCES
    return split, None, None


def list_frames(split, sequences=None, require_gear=True):
    """[(seq_dir, rgb_dir, frame_idx), ...] of labelled frames in a split (see split_filter)."""
    folder, keep, drop = split_filter(split)
    items = []
    for mpath in sorted(glob.glob(osp.join(CFG['processed_root'], folder, '*', 'meta.json'))):
        seq_dir = osp.dirname(mpath)
        name = osp.basename(seq_dir)
        if (sequences and name not in sequences) or (keep is not None and name not in keep) or                 (drop is not None and name in drop):
            continue
        if require_gear and not osp.isfile(osp.join(seq_dir, 'gear_done.json')):
            continue
        m = json.load(open(mpath))
        items += [(seq_dir, m['rgb_dir'], i) for i in m['labelled_frames']]
    return items


def read_labels(path):
    a = np.loadtxt(path, ndmin=2, dtype=np.float32) if osp.getsize(path) else np.zeros((0, 5), np.float32)
    return a.reshape(-1, 5)


def hsv_gains(hgain=0.015, sgain=0.7, vgain=0.4):
    return np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1


def augment_hsv(img, r):
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class DualDSEC(Dataset):
    def __init__(self, split='train', size=640, augment=False, sequences=None, classes=None,
                 scale=0.5, translate=0.1, fliplr=0.5, hsv=True, label_map=None):
        """classes: keep only these class ids (open-world task splits).
        label_map: {original class id: training class index} applied after filtering."""
        self.items = list_frames(split, sequences)
        self.size, self.augment = size, augment
        self.scale, self.translate, self.fliplr, self.hsv = scale, translate, fliplr, hsv
        self.classes = None if classes is None else np.array(sorted(classes))
        self.label_map = label_map

    def __len__(self):
        return len(self.items)

    def _matrix(self, w, h):
        """Affine matrix mapping the original frame into the S x S network input."""
        S = self.size
        s = min(S / w, S / h)
        if self.augment:
            s *= random.uniform(1 - self.scale, 1 + self.scale)
        tx, ty = (S - w * s) / 2, (S - h * s) / 2      # centred letterbox
        if self.augment:
            tx += random.uniform(-self.translate, self.translate) * S
            ty += random.uniform(-self.translate, self.translate) * S
        return np.array([[s, 0, tx], [0, s, ty]], np.float32)

    def load(self, i):
        seq_dir, rgb_dir, idx = self.items[i]
        rgb = cv2.imread(osp.join(rgb_dir, f'{idx:06d}.png'))                       # BGR uint8
        evt = cv2.imread(osp.join(seq_dir, 'gear', f'{idx:06d}.png'))               # [h, feat, num]
        lab = read_labels(osp.join(seq_dir, 'labels', f'{idx:06d}.txt'))
        return rgb, evt, lab

    def sample_aug(self, w=640, h=480):
        """Random augmentation parameters (drawn once per sample, or once per clip)."""
        return {'M': self._matrix(w, h),
                'flip': self.augment and random.random() < self.fliplr,
                'hsv': hsv_gains() if (self.augment and self.hsv) else None}

    def __getitem__(self, i):
        rgb, evt, lab = self.load(i)
        return self.transform(rgb, evt, lab, self.sample_aug(rgb.shape[1], rgb.shape[0]))

    def transform(self, rgb, evt, lab, aug):
        h, w = rgb.shape[:2]
        if self.classes is not None:
            lab = lab[np.isin(lab[:, 0].astype(int), self.classes)]
        if self.label_map is not None and len(lab):
            lab = lab.copy()
            lab[:, 0] = [self.label_map[int(c)] for c in lab[:, 0]]

        # boxes to pixel xyxy in the original frame
        boxes = np.stack([(lab[:, 1] - lab[:, 3] / 2) * w, (lab[:, 2] - lab[:, 4] / 2) * h,
                          (lab[:, 1] + lab[:, 3] / 2) * w, (lab[:, 2] + lab[:, 4] / 2) * h], 1)
        M = aug['M']
        S = self.size
        rgb = cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=RGB_FILL)
        evt = cv2.warpAffine(evt, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=EVT_FILL)
        s = M[0, 0]
        nb = boxes * s + np.array([M[0, 2], M[1, 2], M[0, 2], M[1, 2]], np.float32)
        nb = nb.clip(0, S)
        # drop boxes that became tiny or were mostly cut off (Ultralytics' box_candidates rule)
        bw, bh = nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]
        area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * s * s
        keep = (bw > 2) & (bh > 2) & (bw * bh / (area0 + 1e-9) > 0.1)
        nb, cls = nb[keep], lab[keep, 0]

        if aug['flip']:
            rgb, evt = rgb[:, ::-1], evt[:, ::-1]
            nb = np.stack([S - nb[:, 2], nb[:, 1], S - nb[:, 0], nb[:, 3]], 1)
        if aug['hsv'] is not None:
            rgb = augment_hsv(np.ascontiguousarray(rgb), aug['hsv'])

        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb[:, :, ::-1].transpose(2, 0, 1))).float() / 255.0
        evt_t = torch.from_numpy(np.ascontiguousarray(evt[:, :, ::-1].transpose(2, 0, 1))).float() / 255.0
        xywh = np.stack([(nb[:, 0] + nb[:, 2]) / 2, (nb[:, 1] + nb[:, 3]) / 2,
                         nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]], 1) / S
        labels = torch.from_numpy(np.concatenate([cls[:, None], xywh], 1).astype(np.float32))
        return rgb_t, evt_t, labels

    @staticmethod
    def collate(batch):
        rgb, evt, labels = zip(*batch)
        idx = torch.cat([torch.full((len(l),), i, dtype=torch.float32) for i, l in enumerate(labels)])
        lab = torch.cat(labels, 0)
        return {
            'rgb': torch.stack(rgb), 'evt': torch.stack(evt),
            'batch_idx': idx, 'cls': lab[:, :1], 'bboxes': lab[:, 1:],     # Ultralytics loss format
        }


class ClipDSEC(DualDSEC):
    """Clips of T consecutive frames for temporal fusion (TFS).

    Each epoch, every sequence is cut into non-overlapping windows of T consecutive images starting
    at a random offset; windows containing no labelled frame are skipped. All frames of a clip share
    one augmentation. Unlabelled frames (no label file) still advance the hidden state but carry no
    loss: has_label[t] = False.
    """

    def __init__(self, split='train', clip_len=4, **kw):
        super().__init__(split, **kw)
        self.T = clip_len
        self.seqs = []                       # (seq_dir, rgb_dir, n_images, labelled set)
        for seq_dir in sorted({it[0] for it in self.items}):
            m = json.load(open(osp.join(seq_dir, 'meta.json')))
            self.seqs.append((seq_dir, m['rgb_dir'], m['n_images'], set(m['labelled_frames'])))
        self.resample()

    def resample(self):
        """New random clip boundaries (call once per epoch)."""
        self.clips = []
        for si, (_, _, n, lab) in enumerate(self.seqs):
            for start in range(random.randrange(self.T), n - self.T + 1, self.T):
                if any(start + t in lab for t in range(self.T)):
                    self.clips.append((si, start))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        si, start = self.clips[i]
        seq_dir, rgb_dir, _, lab_set = self.seqs[si]
        aug = self.sample_aug()
        frames = []
        for idx in range(start, start + self.T):
            rgb = cv2.imread(osp.join(rgb_dir, f'{idx:06d}.png'))
            evt = cv2.imread(osp.join(seq_dir, 'gear', f'{idx:06d}.png'))
            has = idx in lab_set
            lab = read_labels(osp.join(seq_dir, 'labels', f'{idx:06d}.txt')) if has else np.zeros((0, 5), np.float32)
            frames.append((*self.transform(rgb, evt, lab, aug), has))
        return frames

    @staticmethod
    def collate(batch):
        """-> list over time of single-frame batches (DualDSEC.collate format) + 'has_label' (B,)."""
        out = []
        for t in range(len(batch[0])):
            step = DualDSEC.collate([(r, e, l) for r, e, l, _ in (clip[t] for clip in batch)])
            step['has_label'] = torch.tensor([clip[t][3] for clip in batch])
            out.append(step)
        return out


class SequenceDSEC(DualDSEC):
    """Every frame of each sequence in temporal order (for streaming evaluation of TFS models).
    Items: (rgb, evt, labels, has_label, seq_index, is_first_frame)."""

    def __init__(self, split='test', sequences=None, **kw):
        super().__init__(split, augment=False, sequences=sequences, **kw)
        self.frames = []
        for si, seq_dir in enumerate(sorted({it[0] for it in self.items})):
            m = json.load(open(osp.join(seq_dir, 'meta.json')))
            lab = set(m['labelled_frames'])
            self.frames += [(si, seq_dir, m['rgb_dir'], i, i in lab, i == 0) for i in range(m['n_images'])]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, k):
        si, seq_dir, rgb_dir, idx, has, first = self.frames[k]
        rgb = cv2.imread(osp.join(rgb_dir, f'{idx:06d}.png'))
        evt = cv2.imread(osp.join(seq_dir, 'gear', f'{idx:06d}.png'))
        lab = read_labels(osp.join(seq_dir, 'labels', f'{idx:06d}.txt')) if has else np.zeros((0, 5), np.float32)
        r, e, l = self.transform(rgb, evt, lab, self.sample_aug())
        return r, e, l, has, si, first
