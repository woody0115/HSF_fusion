"""Step 2: preprocess DSEC-Det into YOLO labels + GEAR event frames.

Two independent stages per sequence (the RGB baseline only needs the first):
  labels  (needs timestamps, tracks, all remapped RGB frames)
      <processed_root>/<split>/<seq>/rgb/000123.png      copy of every remapped RGB frame (training reads
                                                         from the SSD; the raw download stays on the HDD)
      <processed_root>/<split>/<seq>/labels/000123.txt   "cls xc yc w h" normalized to 640x480
      <processed_root>/<split>/<seq>/boxes.npy           img_idx, cls, x0, y0, x1, y1 (px), track_id
      <processed_root>/<split>/<seq>/meta.json           image count, labelled frames, paths, stats
  gear    (needs events/left/events.h5)
      <processed_root>/<split>/<seq>/gear/000123.png     GEAR frame for every image (see hsf/data/gear.py)
      <processed_root>/<split>/<seq>/gear_done.json      marker + event statistics
Safe to re-run at any time: finished stages are skipped, sequences still downloading are reported.
Run:  .venv\\Scripts\\python.exe tools\\preprocess_dsec.py [--workers 6] [--only thun_02_a] [--labels-only]
"""
import argparse
import json
import os
import os.path as osp
import shutil
import sys
import time
from multiprocessing import Pool

import cv2
import h5py
import hdf5plugin  # noqa: F401  (blosc filter for DSEC events)
import numpy as np
import yaml

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
from hsf.data.gear import GEAR, EventReader, first_window_start  # noqa: E402

CFG = yaml.safe_load(open(osp.join(osp.dirname(__file__), '..', 'configs', 'paths.yaml')))
W, H = 640, 480
MIN_SIDE = 2.0          # boxes thinner than this after clipping to the image are dropped
BOX_DTYPE = np.dtype([('img_idx', 'i4'), ('cls', 'u1'), ('x0', 'f4'), ('y0', 'f4'),
                      ('x1', 'f4'), ('y1', 'f4'), ('track_id', 'u4')])


def seq_paths(split, seq):
    src = osp.join(CFG['dsec_root'], split, seq)
    out = osp.join(CFG['processed_root'], split, seq)
    return {
        'src': src,
        'h5': osp.join(src, 'events', 'left', 'events.h5'),
        'ts': osp.join(src, 'images', 'timestamps.txt'),
        'rgb': osp.join(src, 'images', 'left', 'distorted'),
        'tracks': osp.join(src, 'object_detections', 'left', 'tracks.npy'),
        'out': out,
        'meta': osp.join(out, 'meta.json'),
        'gear_done': osp.join(out, 'gear_done.json'),
    }


def labels_ready(p):
    """None if the label stage can run, else the reason it cannot (yet)."""
    for k in ('ts', 'tracks'):
        if not osp.isfile(p[k]):
            return f'missing {osp.relpath(p[k], p["src"])}'
    n_ts = len(np.loadtxt(p['ts'], dtype=np.int64))
    n_rgb = len([f for f in os.listdir(p['rgb']) if f.endswith('.png')]) if osp.isdir(p['rgb']) else 0
    if n_rgb != n_ts:
        return f'RGB frames {n_rgb}/{n_ts}'
    return None


def gear_ready(p):
    """None if the GEAR stage can run, else the reason it cannot (yet)."""
    if not osp.isfile(p['h5']):
        return 'missing events/left/events.h5'
    if time.time() - osp.getmtime(p['h5']) < 120:
        return 'events.h5 still being written'
    try:
        with h5py.File(p['h5'], 'r') as f:
            f['ms_to_idx'][-1]
    except Exception as e:  # truncated file during extraction
        return f'events.h5 not readable yet ({type(e).__name__})'
    return None


def convert_labels(tracks, ts):
    """DSEC tracks -> clipped boxes per image index."""
    idx = np.searchsorted(ts, tracks['t'].astype(np.int64))
    assert np.all(ts[np.clip(idx, 0, len(ts) - 1)] == tracks['t'].astype(np.int64)), 'label time not on an image'
    x0 = np.clip(tracks['x'], 0, W)
    y0 = np.clip(tracks['y'], 0, H)
    x1 = np.clip(tracks['x'] + tracks['w'], 0, W)
    y1 = np.clip(tracks['y'] + tracks['h'], 0, H)
    keep = ((x1 - x0) >= MIN_SIDE) & ((y1 - y0) >= MIN_SIDE)
    boxes = np.zeros(int(keep.sum()), BOX_DTYPE)
    boxes['img_idx'], boxes['cls'], boxes['track_id'] = idx[keep], tracks['class_id'][keep], tracks['track_id'][keep]
    boxes['x0'], boxes['y0'], boxes['x1'], boxes['y1'] = x0[keep], y0[keep], x1[keep], y1[keep]
    labelled = np.unique(idx)                # frames that were annotated (may end up with 0 boxes)
    return boxes, labelled, int((~keep).sum())


def do_labels(split, seq):
    p = seq_paths(split, seq)
    t0 = time.time()
    os.makedirs(osp.join(p['out'], 'labels'), exist_ok=True)
    rgb_out = osp.join(p['out'], 'rgb')
    os.makedirs(rgb_out, exist_ok=True)
    ts = np.loadtxt(p['ts'], dtype=np.int64)
    for i in range(len(ts)):                 # all frames: temporal modules need unlabelled ones too
        dst = osp.join(rgb_out, f'{i:06d}.png')
        if not osp.exists(dst):
            shutil.copyfile(osp.join(p['rgb'], f'{i:06d}.png'), dst)
    boxes, labelled, dropped = convert_labels(np.load(p['tracks']), ts)
    np.save(osp.join(p['out'], 'boxes.npy'), boxes)
    for i in labelled:
        b = boxes[boxes['img_idx'] == i]
        lines = [f'{c} {(a + c2) / 2 / W:.6f} {(bb + d) / 2 / H:.6f} {(c2 - a) / W:.6f} {(d - bb) / H:.6f}'
                 for c, a, bb, c2, d in zip(b['cls'], b['x0'], b['y0'], b['x1'], b['y1'])]
        with open(osp.join(p['out'], 'labels', f'{i:06d}.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
    meta = {
        'split': split, 'sequence': seq, 'n_images': int(len(ts)),
        'labelled_frames': [int(i) for i in labelled],
        'n_boxes': int(len(boxes)), 'n_boxes_dropped_tiny': dropped,
        'boxes_per_class': {int(c): int(n) for c, n in zip(*np.unique(boxes['cls'], return_counts=True))},
        'rgb_dir': rgb_out, 'rgb_source': p['rgb'], 'timestamps': p['ts'], 'events_h5': p['h5'],
    }
    with open(p['meta'], 'w') as f:
        json.dump(meta, f, indent=1)
    return f'labels {split}/{seq}: {len(labelled)} labelled frames, {len(boxes)} boxes ' \
           f'({dropped} tiny dropped), {time.time() - t0:.0f} s'


def do_gear(split, seq):
    p = seq_paths(split, seq)
    t0 = time.time()
    os.makedirs(osp.join(p['out'], 'gear'), exist_ok=True)
    ts = np.loadtxt(p['ts'], dtype=np.int64)
    gear = GEAR(H, W)                    # recurrent along the sequence -> every image is encoded
    n_events = np.zeros(len(ts), np.int64)
    with h5py.File(p['h5'], 'r') as f:
        reader = EventReader(f)
        for i in range(len(ts)):
            x, y, pol = reader.window(first_window_start(ts, i), ts[i])
            n_events[i] = len(x)
            img = GEAR.to_uint8(gear(x, y, pol))
            cv2.imwrite(osp.join(p['out'], 'gear', f'{i:06d}.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    info = {
        'n_frames': int(len(ts)), 'events_per_frame_median': int(np.median(n_events)),
        'gear': {'c': 8, 'tau': 10, 'window': 'previous image -> current image (sync=front)',
                 'channels': '[h, feat, num]; h/feat mid-gray=0, num black=0'},
        'seconds': round(time.time() - t0, 1),
    }
    with open(p['gear_done'], 'w') as f:
        json.dump(info, f, indent=1)
    return f'gear   {split}/{seq}: {len(ts)} frames, {info["events_per_frame_median"]:,} ev/frame, {info["seconds"]:.0f} s'


def run(job):
    stage, split, seq = job
    return (do_labels if stage == 'labels' else do_gear)(split, seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--only', nargs='*', help='sequence names to process')
    ap.add_argument('--labels-only', action='store_true', help='skip the GEAR stage')
    a = ap.parse_args()

    label_jobs, gear_jobs, waiting = [], [], []
    n_lab_done = n_gear_done = 0
    for split in ('train', 'test'):
        root = osp.join(CFG['dsec_root'], split)
        for seq in sorted(os.listdir(root)):
            if a.only and seq not in a.only:
                continue
            p = seq_paths(split, seq)
            if osp.isfile(p['meta']):
                n_lab_done += 1
            else:
                why = labels_ready(p)
                label_jobs.append(('labels', split, seq)) if not why else waiting.append(f'labels {split}/{seq}: {why}')
            if a.labels_only:
                continue
            if osp.isfile(p['gear_done']):
                n_gear_done += 1
            else:
                why = gear_ready(p)
                gear_jobs.append(('gear', split, seq)) if not why else waiting.append(f'gear   {split}/{seq}: {why}')

    print(f'labels: {n_lab_done} done, {len(label_jobs)} to do | '
          f'gear: {n_gear_done} done, {len(gear_jobs)} to do | waiting for download: {len(waiting)}', flush=True)
    t0 = time.time()
    jobs = label_jobs + gear_jobs           # labels are fast; do them first
    if jobs:
        with Pool(min(a.workers, len(jobs))) as pool:
            for msg in pool.imap_unordered(run, jobs):
                print('  ' + msg, flush=True)
        print(f'finished {len(jobs)} jobs in {(time.time() - t0) / 60:.1f} min', flush=True)
    if waiting:
        print('still waiting (re-run this script after more data is downloaded):')
        for w in waiting[:8]:
            print('  ' + w)
        if len(waiting) > 8:
            print(f'  ... and {len(waiting) - 8} more')


if __name__ == '__main__':
    main()
