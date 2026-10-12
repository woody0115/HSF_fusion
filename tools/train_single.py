"""Step 3: single-stream YOLOv8s baselines (RGB only / event-GEAR only) with the Ultralytics trainer.

Settings follow the HSF paper's single-stream stage (Sec. IV-B): YOLOv8s, 640x640, SGD,
lr 0.01, momentum 0.937, batch 16, 10 epochs, starting from COCO-pretrained yolov8s.pt.
Not specified in the paper (our choice): close_mosaic=2 (Ultralytics' default of 10 would
disable mosaic for the whole 10-epoch run); for GEAR frames the HSV colour augmentation is
turned off because their channels are not colours.

Run:  .venv\\Scripts\\python.exe tools\\train_single.py --modality rgb   [--epochs 10] [--tag _smoke]
      .venv\\Scripts\\python.exe tools\\train_single.py --modality event
Results: runs/step3/<modality>_yolov8s<tag>/ (weights/best.pt, results.csv, plots)
"""
import argparse
import os
import os.path as osp
import sys

from ultralytics import YOLO

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))


class Tee:
    """Mirror console output into a log file so training can be watched live (VS Code task)."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, 'a', encoding='utf-8', buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.file.write(s.replace('\r', '\n') if '\r' in s else s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modality', choices=['rgb', 'event'], required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tag', default='', help='dataset/run suffix, e.g. _smoke')
    ap.add_argument('--fraction', type=float, default=1.0, help='use a fraction of the train set')
    a = ap.parse_args()

    run_dir = osp.join(ROOT, 'runs', 'step3', f'{a.modality}_yolov8s{a.tag}')
    os.makedirs(run_dir, exist_ok=True)
    log = osp.join(run_dir, 'console.log')
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    with open(osp.join(ROOT, 'runs', 'latest_log.txt'), 'w', encoding='utf-8') as f:
        f.write(log)                                            # read by tools/watch_training.ps1

    data = osp.join(ROOT, 'configs', 'generated', f'dsec_{a.modality}{a.tag}.yaml')
    model = YOLO(osp.join(ROOT, 'weights', 'yolov8s.pt'))     # downloaded on first use
    extra = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0) if a.modality == 'event' else {}
    model.train(
        data=data, imgsz=640, epochs=a.epochs, batch=a.batch, workers=a.workers,
        optimizer='SGD', lr0=0.01, momentum=0.937, close_mosaic=2,
        fraction=a.fraction, seed=0, deterministic=False,
        project=osp.join(ROOT, 'runs', 'step3'), name=f'{a.modality}_yolov8s{a.tag}', exist_ok=True,
        plots=True, **extra,
    )


if __name__ == '__main__':
    main()
