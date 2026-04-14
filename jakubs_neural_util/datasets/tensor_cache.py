from __future__ import annotations

from typing import TYPE_CHECKING

import diskcache as dc
import torch
import zstandard as zstd
from typing import Generic, Optional, TypeVar

import io

T = TypeVar("T")

class TensorCache(Generic[T]):
    """
    A safe, multiprocess-friendly, compressed tensor cache.
    - Values are serialized with torch.save
    - Compressed with zstd
    - Stored as raw bytes in diskcache
    - Exposes [] operator
    """

    def __init__(
        self,
        path: str,
        *,
        size_limit: Optional[int] = None,
        use_compression: bool = False
    ):
        self.cache = dc.Cache(path, size_limit=size_limit)
        if use_compression:
            self.cctx = zstd.ZstdCompressor(level=1, threads=4)
            self.dctx = zstd.ZstdDecompressor()
        else:
            self.cctx = None
            self.dctx = None
        self.use_compression = use_compression

    # --------------------------
    # Public API
    # --------------------------

    def __setitem__(self, key: str, value: T) -> None:
        buf = io.BytesIO()
        if self.use_compression:
            with self.cctx.stream_writer(buf, closefd=False) as compressor:
                torch.save(value, compressor, _use_new_zipfile_serialization=False)
        else:
            torch.save(value, buf, _use_new_zipfile_serialization=False)
        buf.seek(0)
        # diskcache reads the BytesIO and stores bytes
        self.cache.set(key, buf, read=True)

    def __getitem__(self, key: str):
        stream = self.cache.get(key, read=True)

        if self.use_compression:
            with self.dctx.stream_reader(stream, closefd=True) as reader:
                bytes_out = io.BytesIO()
                while True:
                    chunk = reader.read(1024*1024*4)
                    if not chunk:
                        break
                    bytes_out.write(chunk)

            bytes_out.seek(0)
        else:
            bytes_out = stream
        return torch.load(bytes_out, weights_only=True)

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        if key not in self.cache:
            return default
        return self[key]

    def close(self):
        self.cache.close()

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)
