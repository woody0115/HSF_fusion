"""Resumable multi-connection HTTP downloader.

The DSEC server limits each connection to ~1 MB/s, but 4 parallel range requests reach ~15 MB/s.
The file is split into 64 MB chunks fetched by worker threads and written in place; finished chunk
indices are saved to <out>.progress.json so an interrupted download resumes where it stopped.
A pre-existing partial file without a progress file (e.g. left by curl, which writes sequentially)
is reused: chunks lying entirely inside its current size are treated as done.

Usage:  python pdownload.py <url> <out_file> [--workers 4]
Exit code 0 on success.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.request

CHUNK = 64 << 20


def head_size(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])


def fetch_chunk(url, f, lock, start, end):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end - 1}'})
    pos = start
    with urllib.request.urlopen(req, timeout=60) as r:
        while pos < end:
            b = r.read(min(1 << 20, end - pos))
            if not b:
                raise IOError(f'connection closed at {pos}')
            with lock:
                f.seek(pos)
                f.write(b)
            pos += len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    size = head_size(a.url)
    n_chunks = (size + CHUNK - 1) // CHUNK
    prog_file = a.out + '.progress.json'
    done = set()
    if os.path.exists(prog_file):
        done = set(json.load(open(prog_file))['done'])
    elif os.path.exists(a.out):
        prefix = os.path.getsize(a.out)
        done = {i for i in range(n_chunks) if min((i + 1) * CHUNK, size) <= prefix}
    mode = 'r+b' if os.path.exists(a.out) else 'w+b'
    f = open(a.out, mode)
    f.truncate(size)

    todo = queue.Queue()
    for i in range(n_chunks):
        if i not in done:
            todo.put(i)
    lock, state_lock = threading.Lock(), threading.Lock()
    errors = []
    t0, done0 = time.time(), len(done)

    def save():
        with open(prog_file, 'w') as pf:
            json.dump({'size': size, 'done': sorted(done)}, pf)

    def worker():
        while True:
            try:
                i = todo.get_nowait()
            except queue.Empty:
                return
            start, end = i * CHUNK, min((i + 1) * CHUNK, size)
            for attempt in range(10):
                try:
                    fetch_chunk(a.url, f, lock, start, end)
                    break
                except Exception as e:  # network hiccup: retry this chunk
                    time.sleep(min(60, 5 * (attempt + 1)))
            else:
                errors.append(i)
                continue
            with state_lock:
                done.add(i)
                save()
                el = time.time() - t0
                rate = (len(done) - done0) * CHUNK / max(el, 1e-6) / 1e6
                print(f'  {len(done)}/{n_chunks} chunks  {len(done) * CHUNK / 1e9:.1f}/{size / 1e9:.1f} GB  '
                      f'{rate:.1f} MB/s', flush=True)

    print(f'{a.url}\n  size {size / 1e9:.2f} GB, {n_chunks} chunks, {len(done)} already done, {a.workers} workers',
          flush=True)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    f.close()
    if errors:
        print(f'FAILED chunks: {errors}', flush=True)
        sys.exit(1)
    if os.path.exists(prog_file):   # absent when every chunk was already present on entry
        os.remove(prog_file)
    print('  complete', flush=True)


if __name__ == '__main__':
    main()
