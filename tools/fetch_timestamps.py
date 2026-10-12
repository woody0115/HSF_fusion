"""Fetch only images/timestamps.txt for every DSEC-Det sequence.

The timestamps live inside the (huge) rectified-image archives, which we otherwise don't need.
Range requests read just those small members, so nothing large is downloaded.
Output: <dsec_root>/<split>/<sequence>/images/timestamps.txt

Run:  .venv\\Scripts\\python.exe tools\\fetch_timestamps.py
"""
import io
import os
import os.path as osp
import zipfile

import yaml

from list_remote_zip import HttpRangeFile

ROOT = yaml.safe_load(open(osp.join(osp.dirname(__file__), '..', 'configs', 'paths.yaml')))['dsec_root']
BASE = 'https://download.ifi.uzh.ch/rpg/DSEC'
ZIPS = {  # archive -> split (coarse archives have no split prefix in their paths)
    'train_coarse/train_images.zip': 'train',
    'test_coarse/test_images.zip': 'test',
    'train_object_detection_coarse/train_images.zip': 'train',
    'test_object_detection_coarse/test_images.zip': 'test',
}

done = 0
for rel, split in ZIPS.items():
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(f'{BASE}/{rel}'), buffer_size=1 << 16))
    for name in z.namelist():
        if not name.endswith('images/timestamps.txt'):
            continue
        parts = name.split('/')
        seq = parts[1] if parts[0] in ('train', 'test') else parts[0]
        out = osp.join(ROOT, split, seq, 'images', 'timestamps.txt')
        os.makedirs(osp.dirname(out), exist_ok=True)
        with z.open(name) as src, open(out, 'wb') as dst:
            dst.write(src.read())
        done += 1
    print(f'{rel}: ok')

seqs = [(s, q) for s in ('train', 'test') for q in os.listdir(osp.join(ROOT, s))
        if osp.isdir(osp.join(ROOT, s, q))]
missing = [f'{s}/{q}' for s, q in seqs if not osp.isfile(osp.join(ROOT, s, q, 'images', 'timestamps.txt'))]
print(f'wrote {done} timestamp files; sequences without timestamps: {missing or "none"}')
