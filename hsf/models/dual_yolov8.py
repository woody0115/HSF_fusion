"""Dual-stream YOLOv8 for event + RGB detection (HSF architecture, Fig. 2 of the paper).

Two YOLOv8 backbones (layers 0-9 of the Ultralytics graph: C1..C5 + SPPF) run in parallel on the
event frame and the RGB frame. At the end of stages C3 (layer 4, P3/8), C4 (layer 6, P4/16) and
C5 (layer 9, P5/32) a fusion module sees both feature maps and returns
    evt_out, rgb_out   features that continue in each backbone stream (the "⊕" in Fig. 2)
    neck_in            the fused feature fed to the shared YOLOv8 neck + Detect head
Fusion variants live in hsf/models/fusion.py: 'add' is the paper's ablation baseline; AF and TFS
(steps 5-6) plug into the same interface.

The neck/head are the unmodified Ultralytics modules, so v8DetectionLoss, NMS and the metrics work
as-is; `self.model[-1]` is the Detect layer and `self.args` holds the loss gains like an Ultralytics
DetectionModel.
"""
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

from hsf.models.fusion import build_fusion, illum_descriptor

FUSE_AT = (4, 6, 9)            # backbone layer indices of C3 / C4 / C5 outputs (P3, P4, P5)


def load_ultralytics(path):
    """Ultralytics checkpoint (.pt from model.train) -> trainable DetectionModel in float32.

    Ultralytics saves checkpoints with requires_grad=False on every parameter; re-enable training for
    all of them except the fixed DFL projection (frozen by design in Ultralytics)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float()
    for name, p in model.named_parameters():
        p.requires_grad = '.dfl' not in name
    return model


class DualYOLOv8(nn.Module):
    def __init__(self, nc=8, cfg='yolov8s.yaml', fusion='add', rgb_init=None, evt_init=None, modality_drop=0.15):
        """rgb_init / evt_init: single-stream checkpoints (paper: backbones are first trained
        separately on each modality); the neck + head start from the RGB model.
        modality_drop: illumination-gated fusions (iaf_*) only - share of training images whose gate
        is forced to 0 (pure RGB path)."""
        super().__init__()
        rgb_model = load_ultralytics(rgb_init) if rgb_init else DetectionModel(cfg, nc=nc, verbose=False)
        evt_model = load_ultralytics(evt_init) if evt_init else DetectionModel(cfg, nc=nc, verbose=False)
        assert rgb_model.model[-1].nc == nc, 'class count of the RGB init checkpoint differs'
        self.rgb = nn.ModuleList(copy.deepcopy(list(rgb_model.model[:10])))
        self.evt = nn.ModuleList(copy.deepcopy(list(evt_model.model[:10])))
        self.model = nn.ModuleList(copy.deepcopy(list(rgb_model.model[10:])))   # neck + Detect
        chans = [self._out_channels(self.rgb, i) for i in FUSE_AT]
        self.fusion = nn.ModuleList(build_fusion(fusion, c, scale_idx=k) for k, c in enumerate(chans))
        self.fusion_type = fusion
        self.stateful = bool(getattr(self.fusion[0], 'stateful', False))   # needs consecutive frames
        self.illum = fusion.startswith('iaf_')
        self.modality_drop = modality_drop
        self.nc = nc
        self.stride = self.model[-1].stride
        self.names = rgb_model.names
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)                 # Ultralytics default gains

    @staticmethod
    def _out_channels(stream, idx):
        m = stream[idx]
        for attr in ('cv2', 'conv'):                 # C2f / SPPF expose their output conv as cv2
            sub = getattr(m, attr, None)
            if sub is not None and hasattr(sub, 'conv'):
                return sub.conv.out_channels
        raise ValueError(f'cannot infer channels of layer {idx}')

    def forward(self, rgb, evt, state=None):
        """rgb, evt: (B, 3, H, W). state: per-scale hidden states for temporal fusion (TFS), or None
        (= start of a sequence). Returns the Detect output (train: list of 3 maps; eval: (decoded
        preds, maps)); stateful models (self.stateful) return (detect_output, new_state)."""
        x_r, x_e = rgb, evt
        neck_inputs, new_state = {}, []
        extra = {}
        if self.illum:          # illumination-gated fusion: lighting descriptor + modality dropout mask
            extra['illum'] = illum_descriptor(rgb)
            if self.training and self.modality_drop > 0:
                extra['keep'] = (torch.rand(rgb.shape[0], device=rgb.device) >= self.modality_drop).float()
        k = 0
        for i in range(10):
            x_r, x_e = self.rgb[i](x_r), self.evt[i](x_e)
            if i in FUSE_AT:
                h = None if state is None else state[k]
                x_e, x_r, neck_inputs[i], h_new = self.fusion[k](x_e, x_r, h, **extra)
                new_state.append(h_new)
                k += 1
        # replay the Ultralytics neck graph; it only reads layers 4, 6, 9 from the backbone
        y = [None] * 10
        for i in FUSE_AT:
            y[i] = neck_inputs[i]
        x = y[9]
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return (x, new_state) if self.stateful else x
