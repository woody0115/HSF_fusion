"""Steps 4-6: train the dual-stream (event + RGB) detector.

Defaults follow the HSF paper's full-model stage (Sec. IV-B): backbones initialised from the
single-stream models of step 3, SGD lr 0.005, momentum 0.937, batch 8, 40 epochs, 640x640.

Run:  .venv\\Scripts\\python.exe tools\\train_dual.py --fusion add --epochs 20   (step 4 baseline)
      .venv\\Scripts\\python.exe tools\\train_dual.py --fusion af  --epochs 20   (step 5)
      .venv\\Scripts\\python.exe tools\\train_dual.py --fusion hsf --epochs 20   (step 6, full HSF:
            TFS + AF, trained on clips of --clip-len consecutive frames, evaluated by streaming)
      .venv\Scripts\python.exe tools\train_dual.py --fusion iaf_add --epochs 5 --lr 1e-3 --warmup 1
            (step 8: illumination-aware gate on add / af, see IllumGate in hsf/models/fusion.py)
      add --smoke for a quick pipeline check
"""
import argparse
import os.path as osp
import sys

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from hsf.engine import start_logging, train, train_temporal  # noqa: E402
from hsf.models.dual_yolov8 import DualYOLOv8  # noqa: E402

STEP3 = osp.join(ROOT, 'runs', 'step3')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fusion', default='add')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--rgb-init', default=osp.join(STEP3, 'rgb_yolov8s', 'weights', 'best.pt'))
    ap.add_argument('--evt-init', default=osp.join(STEP3, 'event_yolov8s', 'weights', 'best.pt'))
    ap.add_argument('--name', default=None)
    ap.add_argument('--warmup', type=float, default=3, help='warmup epochs')
    ap.add_argument('--clips', type=int, default=4, help='TFS/HSF: clips per batch')
    ap.add_argument('--clip-len', type=int, default=4, help='TFS/HSF: consecutive frames per clip')
    ap.add_argument('--modality-drop', type=float, default=0.15,
                    help='iaf_* fusions: share of training images with the event contribution switched off')
    ap.add_argument('--split', default='val', choices=['val', 'legacy'],
                    help='val: train on train_core, select checkpoints on the validation sequences '
                         '(configs/splits.yaml), report test once; legacy: train on all 47, select on test (#1-#12)')
    ap.add_argument('--resume', action='store_true', help='continue from <run>/resume.pt')
    ap.add_argument('--smoke', action='store_true', help='2 short epochs on a few sequences')
    a = ap.parse_args()

    name = a.name or f'dual_{a.fusion}' + ('_smoke' if a.smoke else '')
    step = 'step4' if a.fusion == 'add' else ('step8_iaf' if a.fusion.startswith('iaf_') else 'step5_6')
    run_dir = osp.join(ROOT, 'runs', step, name)
    start_logging(run_dir, ROOT)
    for k, path in (('rgb', a.rgb_init), ('event', a.evt_init)):
        if not osp.isfile(path):
            print(f'NOTE: {k} init checkpoint not found ({path}); that stream starts from scratch')
    model = DualYOLOv8(nc=8, fusion=a.fusion,
                       rgb_init=a.rgb_init if osp.isfile(a.rgb_init) else None,
                       evt_init=a.evt_init if osp.isfile(a.evt_init) else None,
                       modality_drop=a.modality_drop)
    print(f'config: {vars(a)}', flush=True)
    smoke = dict(epochs=2, sequences_train=['zurich_city_16_a', 'zurich_city_17_a'], sequences_val=['thun_02_a'],
                 max_train_batches=150) if a.smoke else dict(epochs=a.epochs)
    splits = dict(train_split='train', val_split='test') if (a.split == 'legacy' or a.smoke) else \
        dict(train_split='train_core', val_split='val')
    smoke.update(splits, resume=a.resume)
    if model.stateful:      # TFS / full HSF: clips of consecutive frames, streaming evaluation
        train_temporal(model, run_dir, clips=a.clips, clip_len=a.clip_len, lr=a.lr, workers=a.workers,
                       eval_every=1 if (a.smoke or a.epochs <= 6) else 2, warmup_epochs=a.warmup, **smoke)
    else:
        train(model, run_dir, batch=a.batch, lr=a.lr, workers=a.workers, warmup_epochs=a.warmup, **smoke)


if __name__ == '__main__':
    main()
