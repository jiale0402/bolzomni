"""Read selected members of a pinned public ZIP without downloading 75 GB."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

REVISION = 'ff7d3bf150e4284bf8f05cf44db7f011e41cba37'
SIZE = 75009352113
URL = f'https://huggingface.co/datasets/boltzgen/boltzgen1_train/resolve/{REVISION}/targets.zip'


class RangeFile(io.RawIOBase):
    def __init__(self, cache):
        self.pos = 0
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.downloaded = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else SIZE + offset
        return self.pos

    def read(self, n=-1):
        n = min(SIZE-self.pos, n if n >= 0 else SIZE-self.pos)
        if n <= 0: return b''
        if n > 80_000_000: raise ValueError('Refusing unexpectedly large range')
        first, last = self.pos, self.pos+n-1
        path = self.cache/f'{first}-{last}.bin'
        if path.exists():
            value = path.read_bytes()
        else:
            req = urllib.request.Request(URL+f'?range={first}-{last}', headers={'Range': f'bytes={first}-{last}'})
            with urllib.request.urlopen(req, timeout=90) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f'bytes {first}-{last}/{SIZE}'
                value = response.read(n+1)
            assert len(value) == n
            path.write_bytes(value)
            self.downloaded += len(value)
        assert len(value) == n
        self.pos += n
        return value


def open_archive(cache):
    source = RangeFile(cache)
    return source, zipfile.ZipFile(source)


if __name__ == '__main__':
    source, archive = open_archive(Path(__file__).parent/'data/ranges')
    names = archive.namelist()
    print(json.dumps({'entries': len(names), 'first': names[:12],
                      'manifests': [n for n in names if 'manifest' in n],
                      'downloaded_bytes': source.downloaded}, indent=2))
