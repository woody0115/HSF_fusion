"""Step 1 sanity check: environment + the key assumption of the dual-stream design.

Checks:
  1. PyTorch sees the GPU.
  2. h5py + hdf5plugin can write/read a blosc-compressed dataset (DSEC events use blosc).
  3. YOLOv8s can be split into backbone (layers 0-9) and neck+head (layers 10-22), two
     backbones can be fused, and Ultralytics' v8DetectionLoss works on the result,
     including a backward pass on the GPU.

Run:  .venv\\Scripts\\python.exe tools\\check_env.py
"""
import os
import tempfile
from types import SimpleNamespace

import h5py
import hdf5plugin
import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

NC = 8  # DSEC-Det classes


def check_gpu():
    ok = torch.cuda.is_available()
    print(f'[1] torch {torch.__version__} | CUDA {ok} | {torch.cuda.get_device_name(0) if ok else "-"}')
    return 'cuda' if ok else 'cpu'


def check_blosc():
    path = os.path.join(tempfile.gettempdir(), 'blosc_test.h5')
    data = np.random.randint(0, 640, 100_000).astype(np.uint16)
    with h5py.File(path, 'w') as f:
        f.create_dataset('x', data=data, **hdf5plugin.Blosc(cname='zstd'))
    with h5py.File(path, 'r') as f:
        same = np.array_equal(f['x'][:], data)
    os.remove(path)
    print(f'[2] h5py {h5py.__version__} + blosc round-trip OK: {same}')


class TwoStreamProbe(nn.Module):
    """Minimal dual-stream YOLOv8s: two backbones, features added at P3/P4/P5, shared neck+head.

    Only a feasibility probe for the loss/stride plumbing; the real HSF model (GEAR/TFS/AF)
    is built in later steps.
    """

    def __init__(self, base: DetectionModel):
        super().__init__()
        import copy
        layers = base.model
        self.rgb = nn.ModuleList(copy.deepcopy(layers[:10]))
        self.evt = nn.ModuleList(copy.deepcopy(layers[:10]))
        self.model = nn.ModuleList(layers[10:])   # neck + Detect; v8DetectionLoss reads model[-1]
        self.save = base.save
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)   # Ultralytics default loss gains

    @staticmethod
    def _run(stream, x):
        outs = []
        for m in stream:
            x = m(x)
            outs.append(x)
        return outs          # outs[4]=P3, outs[6]=P4, outs[9]=P5

    def forward(self, rgb, evt):
        a, b = self._run(self.rgb, rgb), self._run(self.evt, evt)
        y = [ai + bi for ai, bi in zip(a, b)]    # element-wise add (paper's ablation baseline)
        x = y[-1]
        for m in self.model:                     # replay Ultralytics' graph for layers 10..22
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return x


def check_dual_stream(device):
    base = DetectionModel('yolov8s.yaml', nc=NC, verbose=False)
    model = TwoStreamProbe(base).to(device).train()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    rgb = torch.rand(2, 3, 640, 640, device=device)
    evt = torch.rand(2, 3, 640, 640, device=device)
    preds = model(rgb, evt)
    print(f'[3] dual-stream YOLOv8s: {n_params:.1f} M params | strides {model.model[-1].stride.tolist()} | '
          f'outputs {[tuple(p.shape) for p in preds]}')
    batch = {  # two dummy boxes (normalized xywh) in image 0 and 1
        'batch_idx': torch.tensor([0., 1.]),
        'cls': torch.tensor([[2.], [0.]]),
        'bboxes': torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.6, 0.1, 0.2]]),
    }
    crit = v8DetectionLoss(model)
    loss, items = crit(preds, batch)
    loss.sum().backward()
    grads = sum(p.grad is not None for p in model.parameters())
    print(f'    v8DetectionLoss box/cls/dfl = {[round(v, 3) for v in items.tolist()]} | '
          f'params with grad: {grads}/{len(list(model.parameters()))}')
    if device == 'cuda':
        print(f'    peak GPU memory (batch 2, 640x640, train): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')


if __name__ == '__main__':
    dev = check_gpu()
    check_blosc()
    check_dual_stream(dev)
    print('ALL CHECKS PASSED')
