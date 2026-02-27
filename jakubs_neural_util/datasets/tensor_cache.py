from __future__ import annotations

from typing import TYPE_CHECKING

import diskcache as dc
import torch
import zstandard as zstd
from typing import Generic, Optional, TypeVar

if TYPE_CHECKING:
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
        size_limit: Optional[int] = None,
    ):
        self.cache = dc.Cache(path, size_limit=size_limit)
        self.cctx = zstd.ZstdCompressor(level=1, threads=4)
        self.dctx = zstd.ZstdDecompressor()

    # --------------------------
    # Public API
    # --------------------------

    def __setitem__(self, key: str, value: T) -> None:
        # torch.save → zstd stream writer → BytesIO
        buf = io.BytesIO()
        with self.cctx.stream_writer(buf) as compressor:
            torch.save(value, compressor, _use_new_zipfile_serialization=False)

        # diskcache reads the BytesIO and stores bytes
        self.cache.set(key, buf, read=True)

    def __getitem__(self, key: str) -> T:
        # diskcache returns readable stream containing compressed blob
        stream: 'io.BufferedReader' = self.cache.get(key, read=True) # type: ignore

        # zstd stream reader → torch.load
        with self.dctx.stream_reader(stream, closefd=True) as reader:
            return torch.load(reader, weights_only=True)

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
