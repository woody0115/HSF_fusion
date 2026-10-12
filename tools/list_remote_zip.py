"""List files inside a remote zip without downloading it (HTTP range requests on the central directory).

Run:  .venv\\Scripts\\python.exe tools\\list_remote_zip.py <url> [filter-substring]
"""
import io
import sys
import urllib.request
import zipfile


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method='HEAD')
        self.size = int(urllib.request.urlopen(req).headers['Content-Length'])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size - 1 if n < 0 else min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b''
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        data = urllib.request.urlopen(req).read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


if __name__ == '__main__':
    url = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 else ''
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))
    names = z.namelist()
    print(f'{len(names)} entries')
    shown = [n for n in names if flt in n] if flt else names[:40]
    for n in shown[:200]:
        print(' ', n)
