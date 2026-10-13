"""Real-time demo of the event + RGB fusion detectors on a video played on screen (or a video file).

There is no event camera here: events are SIMULATED from consecutive frames (hsf/data/event_sim.py) and
encoded with the same GEAR code as the DSEC training data. The contrast threshold is the one with the best
mAP in tools/event_sim_check.py (runs/diag/event_sim_check*.json; currently C = 0.1). On DSEC test the AF
model gets mAP50-95 53.5 with real events, 52.9 with simulated events (C = 0.1), 52.5 with no events and
the RGB-only model 53.9, so this demo shows how the fusion model runs, not a benefit from events. Frames are taken at 20 Hz by
default, like DSEC's 50 ms event windows, center-cropped to 4:3 and resized to 640x480, then prepared
exactly like the training data (letterbox to 640x640 with gray 114, RGB order, /255).

Sources (--source):
    screen            capture a screen region (chosen with the mouse at start; Esc = whole monitor)
    <video file>      .mp4 / .avi ... (loops)
    0, 1, ...         USB camera
Models (--model):
    af       #7  AF fusion, lr 1e-3 (best fusion model overall)        [default]
    add      #6  add fusion, lr 1e-3 (best at night on DSEC)
    iaf_add  #9  add + illumination gate
    rgb      #1  RGB-only YOLOv8s (reference: same frames, no events)
    owod     #12 open world, Task 4 (7 known classes + "unknown")
Window keys: q quit, s save the current view to runs/demo/, e show/hide the simulated events, r reset the
event memory. Sliders: conf %, IoU % (NMS), unknown % (owod only).

Run:  .venv\\Scripts\\python.exe tools\\demo_screen.py [--source screen] [--model af]
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

from hsf.data.dataset import CFG  # noqa: E402
from hsf.data.event_sim import EventSimulator  # noqa: E402
from hsf.models.dual_yolov8 import DualYOLOv8, load_ultralytics  # noqa: E402
from hsf.owod.model import UNK, set_active  # noqa: E402

RUNS = osp.join(ROOT, 'runs')
NAMES = list(CFG['names'])
OWOD_KNOWN = [2, 3, 4, 0, 1, 5, 6]          # T4 known classes in task order (configs/owod_tasks.yaml)
W, H = 640, 480                              # DSEC frame size the models were trained on


def parse_args():
    p = argparse.ArgumentParser(description='HSF fusion demo on a screen region / video')
    p.add_argument('--source', default='screen')
    p.add_argument('--region', default=None, help='x,y,w,h of the screen region (skips the mouse selection)')
    p.add_argument('--model', default='af', choices=['af', 'add', 'iaf_add', 'rgb', 'owod'])
    p.add_argument('--contrast', type=float, default=None, help='event threshold C (default: calibrated value)')
    p.add_argument('--hz', type=float, default=20.0, help='frames per second fed to the model (DSEC: 20)')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.7)
    p.add_argument('--unk', type=float, default=0.1,
                   help='owod: unknown score threshold (unknown scores are low: U-AP is 0.2-0.4 on DSEC)')
    p.add_argument('--scale', type=float, default=1.5, help='display size = 640x480 times this')
    p.add_argument('--record', default=None, help='also write the shown view to this .mp4')
    p.add_argument('--max-frames', type=int, default=0, help='stop after this many frames (0 = run until q)')
    p.add_argument('--headless', action='store_true', help='no window (with --record / --max-frames, for tests)')
    return p.parse_args()


# ----------------------------------------------------------------------------------------- sources
class ScreenSource:
    def __init__(self, region):
        import mss
        self.sct = mss.mss()
        mon = self.sct.monitors[1]
        if region is None:
            region = self.select(mon)
        self.box = region

    def select(self, mon):
        full = cv2.cvtColor(np.asarray(self.sct.grab(mon)), cv2.COLOR_BGRA2BGR)
        s = min(1.0, 1280 / full.shape[1])
        small = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        print('用滑鼠框出影片播放的區域，按 Enter 確認；按 Esc 或 c 使用整個螢幕。', flush=True)
        x, y, w, h = cv2.selectROI('select the video area (Enter = OK, Esc = whole screen)', small, False, False)
        cv2.destroyAllWindows()
        if w < 16 or h < 16:
            return {'left': mon['left'], 'top': mon['top'], 'width': mon['width'], 'height': mon['height']}
        return {'left': mon['left'] + int(x / s), 'top': mon['top'] + int(y / s), 'width': int(w / s), 'height': int(h / s)}

    def read(self):
        return True, cv2.cvtColor(np.asarray(self.sct.grab(self.box)), cv2.COLOR_BGRA2BGR)


class VideoSource:
    def __init__(self, src, hz):
        self.cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not self.cap.isOpened():
            raise SystemExit(f'無法開啟畫面來源：{src}')
        self.is_file = not src.isdigit()
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.step = max(1, int(round(fps / hz))) if self.is_file else 1   # files: keep about `hz` frames per second

    def read(self):
        frame = None
        for _ in range(self.step):
            ok, frame = self.cap.read()
            if not ok and self.is_file:                                   # loop the video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return False, None
        return True, frame


def to_dsec_frame(frame):
    """Center-crop to 4:3 and resize to 640x480."""
    h, w = frame.shape[:2]
    if w * 3 > h * 4:
        cw = h * 4 // 3
        frame = frame[:, (w - cw) // 2:(w - cw) // 2 + cw]
    else:
        ch = w * 3 // 4
        frame = frame[(h - ch) // 2:(h - ch) // 2 + ch]
    return cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)


def to_tensor(img):
    """Same as the dataset without augmentation: letterbox 640x480 -> 640x640 (gray 114 rows on top and
    bottom), BGR -> RGB channel order, /255."""
    lb = cv2.copyMakeBorder(img, 80, 80, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))).float() / 255.0


# ------------------------------------------------------------------------------------------ models
def load_model(name, device):
    if name == 'rgb':
        m = load_ultralytics(osp.join(RUNS, 'step3', 'rgb_yolov8s', 'weights', 'best.pt'))
        return m.to(device).eval(), NAMES, None
    if name == 'owod':
        from hsf.owod.model import to_open_world
        m = to_open_world(DualYOLOv8(nc=80, fusion='add'))
        m.load_state_dict(torch.load(osp.join(RUNS, 'step7', 'owod_add_evtign', 'T4.pt'), map_location='cpu')['model'])
        set_active(m, OWOD_KNOWN + [UNK])
        return m.to(device).eval(), [NAMES[c] for c in OWOD_KNOWN] + ['unknown'], len(OWOD_KNOWN)
    path, fusion = {'af': ('step5_6/dual_af_lr1e3_e5/best.pt', 'af'), 'add': ('step4/dual_add_lr1e3_e5/best.pt', 'add'),
                    'iaf_add': ('step8_iaf/dual_iaf_add_lr1e3_e5/best.pt', 'iaf_add')}[name]
    m = DualYOLOv8(nc=8, fusion=fusion)
    m.load_state_dict(torch.load(osp.join(RUNS, path), map_location='cpu')['model'])
    return m.to(device).eval(), NAMES, None


def calibrated_contrast():
    """Threshold with the best mAP50-95 among the simulated-event evaluations of tools/event_sim_check.py
    (runs/diag/event_sim_check*.json); 0.1 if none has been run."""
    best, best_map = 0.1, -1.0
    for f in ('event_sim_check.json', 'event_sim_check_fine.json'):
        p = osp.join(RUNS, 'diag', f)
        if not osp.isfile(p):
            continue
        d = json.load(open(p))
        for k, v in d['evaluation'].items():
            if k.startswith('simulated'):
                c = float(k.split('=')[1]) if '=' in k else float(d['calibration']['chosen_contrast'])
                if v['mAP50-95'] > best_map:
                    best, best_map = c, v['mAP50-95']
    return best


@torch.no_grad()
def detect(model, name, rgb, evt, conf, iou, unk_conf, n_known, device):
    r = to_tensor(rgb)[None].to(device)
    with torch.autocast('cuda', enabled=device.type == 'cuda'):
        out = model(r) if name == 'rgb' else model(r, to_tensor(evt)[None].to(device))
    y = out[0].float()
    lo = conf if n_known is None else min(conf, unk_conf)
    d = ops.non_max_suppression(y, lo, iou, max_det=100)[0]
    if n_known is not None and len(d):                     # owod: own threshold for unknown, drop duplicates
        is_unk = d[:, 5] == n_known
        d = d[(is_unk & (d[:, 4] >= unk_conf)) | (~is_unk & (d[:, 4] >= conf))]
        is_unk = d[:, 5] == n_known
        if is_unk.any() and (~is_unk).any():
            dup = (box_iou(d[is_unk, :4], d[~is_unk, :4]) > 0.5).any(1)
            keep = torch.ones(len(d), dtype=torch.bool, device=d.device)
            keep[is_unk.nonzero().flatten()[dup]] = False
            d = d[keep]
    d = d.cpu().numpy()
    d[:, [1, 3]] -= 80                                     # letterbox -> 640x480 frame
    return d


# ------------------------------------------------------------------------------------------ drawing
def color_for(c):
    rng = np.random.default_rng(int(c) * 7919 + 17)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def draw(img, dets, names, n_known, scale):
    view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    for x1, y1, x2, y2, s, c in dets:
        unk = n_known is not None and int(c) == n_known
        col = (255, 255, 255) if unk else color_for(c)
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(view, p1, p2, col, 2)
        label = f'{names[int(c)]} {s:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(view, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), col, -1)
        cv2.putText(view, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return view


def event_view(gear_img, scale):
    """Polarity of the current window: red = brighter, blue = darker, on dark gray."""
    feat = gear_img[..., 1].astype(np.int16) - 128
    vis = np.full((H, W, 3), 30, np.uint8)
    pos, neg = np.clip(feat * 2, 0, 255).astype(np.uint8), np.clip(-feat * 2, 0, 255).astype(np.uint8)
    vis[..., 2] = np.maximum(vis[..., 2], pos)
    vis[..., 0] = np.maximum(vis[..., 0], neg)
    return cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def main():
    a = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    contrast = a.contrast if a.contrast is not None else calibrated_contrast()
    model, names, n_known = load_model(a.model, device)
    if a.source == 'screen':
        region = None
        if a.region:
            x, y, w, h = (int(v) for v in a.region.split(','))
            region = {'left': x, 'top': y, 'width': w, 'height': h}
        src = ScreenSource(region)
    else:
        src = VideoSource(a.source, a.hz)
    sim = EventSimulator(contrast)
    print(f'模型 {a.model}；event 為模擬值（門檻 C = {contrast}）；每秒 {a.hz:g} 張。按 q 結束。', flush=True)

    win = 'HSF fusion demo'
    if not a.headless:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('conf %', win, int(a.conf * 100), 100, lambda v: None)
        cv2.createTrackbar('IoU %', win, int(a.iou * 100), 100, lambda v: None)
        if n_known is not None:
            cv2.createTrackbar('unknown %', win, int(a.unk * 100), 100, lambda v: None)
    writer, n_frames = None, 0
    show_evt, period, t_last, fps = True, 1.0 / a.hz, 0.0, 0.0
    os.makedirs(osp.join(RUNS, 'demo'), exist_ok=True)
    while True:
        wait = period - (time.time() - t_last)             # pace the capture at `hz` (DSEC: 50 ms windows)
        if a.source == 'screen' and wait > 0:
            time.sleep(wait)
        t0 = time.time()
        ok, frame = src.read()
        if not ok:
            break
        rgb = to_dsec_frame(frame)
        evt = sim(rgb)
        if a.headless:
            conf, iou, unk = a.conf, a.iou, a.unk
        else:
            conf = max(cv2.getTrackbarPos('conf %', win), 1) / 100
            iou = max(cv2.getTrackbarPos('IoU %', win), 1) / 100
            unk = max(cv2.getTrackbarPos('unknown %', win), 1) / 100 if n_known is not None else 1.0
        dets = detect(model, a.model, rgb, evt, conf, iou, unk, n_known, device)
        view = draw(rgb, dets, names, n_known, a.scale)
        if show_evt and a.model != 'rgb':
            view = np.concatenate([view, event_view(evt, a.scale)], 1)
        fps = 0.9 * fps + 0.1 / max(time.time() - t_last, 1e-3) if t_last else 0.0
        t_last = t0
        info = f'{a.model} | simulated events C={contrast} | {len(dets)} boxes | {fps:.1f} FPS'
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(view, (4, 4), (16 + tw, 14 + th), (20, 20, 20), -1)
        cv2.putText(view, info, (10, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if a.record:
            if writer is None:
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*'mp4v'), a.hz, (view.shape[1], view.shape[0]))
            writer.write(view)
        n_frames += 1
        if a.max_frames and n_frames >= a.max_frames:
            break
        if a.headless:
            continue
        cv2.imshow(win, view)
        key = cv2.waitKey(1 if a.source == 'screen' else max(1, int(period * 1000 - (time.time() - t0) * 1000))) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord('e'):
            show_evt = not show_evt
        if key == ord('r'):
            sim.reset()
        if key == ord('s'):
            p = osp.join(RUNS, 'demo', time.strftime('demo_%Y%m%d_%H%M%S.png'))
            cv2.imwrite(p, view)
            print(f'已存：{p}', flush=True)
    if writer is not None:
        writer.release()
        print(f'已錄影：{a.record}', flush=True)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
