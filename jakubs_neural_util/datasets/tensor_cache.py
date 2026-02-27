from __future__ import annotations
from typing import Generic, TypeVar, Optional
import io
import torch
import lz4.frame

import diskcache as dc

T = TypeVar("T") 

# import zstandard as zstd

# cctx = zstd.ZstdCompressor(level=1, write_content_size=False)
# dctx = zstd.ZstdDecompressor()

# def compress(raw: bytes) -> bytes:
#     return cctx.compress(raw)

# def decompress(blob: bytes) -> bytes:
#     return dctx.decompress(blob)


class TensorCache(Generic[T]):
    """
    A safe, multiprocess-friendly, compressed tensor cache.
    - Values are serialized with torch.save
    - Compressed with lz4
    - Stored as raw bytes in diskcache
    - Exposes [] operator
    """

    def __init__(
        self,
        path: str,
        size_limit: Optional[int] = None,
    ):
        self.cache = dc.Cache(path, size_limit=size_limit)

    # --------------------------
    # Public API
    # --------------------------

    def __setitem__(self, key: str, value: T) -> None:
        buf = io.BytesIO()
        lzfile = lz4.frame.LZ4FrameFile(buf, mode="w")
        torch.save(value, lzfile, _use_new_zipfile_serialization=False)
        self.cache.set(key, buf, read=True)

    def __getitem__(self, key: str) -> T:
        stream = self.cache.get(key, read=True)
        lzfile = lz4.frame.LZ4FrameFile(stream, mode="r")
        return torch.load(lzfile, weights_only=True)

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
