"""Find the dual-stream training bottleneck: data loading vs GPU compute.

Run:  .venv\\Scripts\\python.exe tools\\profile_dual.py
"""
import os.path as osp
import sys
import time

import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402

from hsf.data.dataset import DualDSEC  # noqa: E402
from hsf.models.dual_yolov8 import DualYOLOv8  # noqa: E402

BATCH, N = 8, 60


def loader_speed(workers):
    ds = DualDSEC('train', augment=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=workers,
                                     collate_fn=DualDSEC.collate, persistent_workers=True, pin_memory=True)
    it = iter(dl)
    for _ in range(5):          # warm up the workers
        next(it)
    t0 = time.time()
    for _ in range(N):
        next(it)
    return (time.time() - t0) / N


def gpu_speed(batch=BATCH, benchmark=False, channels_last=False):
    torch.backends.cudnn.benchmark = benchmark
    model = DualYOLOv8(nc=8).cuda().train()
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    model = model.to(memory_format=fmt)
    crit = v8DetectionLoss(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')
    rgb = torch.rand(batch, 3, 640, 640, device='cuda').to(memory_format=fmt)
    evt = torch.rand(batch, 3, 640, 640, device='cuda').to(memory_format=fmt)
    b = {'batch_idx': torch.arange(batch).float().repeat_interleave(8), 'cls': torch.randint(0, 7, (batch * 8, 1)).float(),
         'bboxes': torch.rand(batch * 8, 4) * 0.3 + 0.2}
    for i in range(N + 5):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast('cuda'):
            loss, _ = crit(model(rgb, evt), b)
        scaler.scale(loss.sum()).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
    torch.cuda.synchronize()
    return (time.time() - t0) / N


def real_loop(workers=8, iters=150):
    """The actual training step on real data, timed per phase."""
    from ultralytics.utils.torch_utils import ModelEMA
    model = DualYOLOv8(nc=8).cuda().train()
    crit = v8DetectionLoss(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    scaler = torch.amp.GradScaler('cuda')
    ema = ModelEMA(model)
    ds = DualDSEC('train', augment=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=workers,
                                     collate_fn=DualDSEC.collate, persistent_workers=True, pin_memory=True)
    t = {'data': 0.0, 'fwd+loss': 0.0, 'bwd': 0.0, 'step+ema': 0.0}
    it = iter(dl)
    for i in range(iters + 10):
        if i == 10:
            t = {k: 0.0 for k in t}
            t_start = time.time()
        a = time.time()
        b = next(it)
        rgb, evt = b['rgb'].cuda(non_blocking=True), b['evt'].cuda(non_blocking=True)
        torch.cuda.synchronize(); t1 = time.time(); t['data'] += t1 - a
        with torch.autocast('cuda'):
            loss, items = crit(model(rgb, evt), b)
        torch.cuda.synchronize(); t2 = time.time(); t['fwd+loss'] += t2 - t1
        scaler.scale(loss.sum()).backward()
        torch.cuda.synchronize(); t3 = time.time(); t['bwd'] += t3 - t2
        if (i + 1) % 8 == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); ema.update(model)
        torch.cuda.synchronize(); t['step+ema'] += time.time() - t3
    total = time.time() - t_start
    print(f'real loop ({workers} workers): {total / iters * 1000:.0f} ms/iter | ' +
          ' | '.join(f'{k} {v / iters * 1000:.0f} ms' for k, v in t.items()), flush=True)


if __name__ == '__main__':
    if '--real' in sys.argv:
        real_loop()
        sys.exit()
    if '--gpu-variants' in sys.argv:
        for bs, bm, cl in [(8, False, False), (8, True, False), (8, True, True), (16, True, False), (16, True, True)]:
            g = gpu_speed(bs, bm, cl)
            print(f'GPU batch {bs:2d} benchmark={bm!s:5} channels_last={cl!s:5}: {g * 1000:4.0f} ms/batch '
                  f'= {g * 1000 / bs:5.1f} ms/image', flush=True)
        sys.exit()
    g = gpu_speed()
    print(f'GPU forward+backward (batch {BATCH}): {g * 1000:.0f} ms/batch', flush=True)
    for w in (8, 12, 16):
        d = loader_speed(w)
        print(f'data loader, {w:2d} workers: {d * 1000:.0f} ms/batch', flush=True)
