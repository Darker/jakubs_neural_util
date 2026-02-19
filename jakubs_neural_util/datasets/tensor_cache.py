from __future__ import annotations
from typing import Generic, TypeVar, Optional
import io
import torch
import lz4.frame
import diskcache as dc

T = TypeVar("T") 

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
    # Serialization / compression
    # --------------------------

    @staticmethod
    def _serialize(value: T) -> bytes:
        buf = io.BytesIO()
        torch.save(value, buf)
        raw = buf.getvalue()
        return lz4.frame.compress(raw)

    @staticmethod
    def _deserialize(blob: bytes) -> T:
        raw = lz4.frame.decompress(blob)
        buf = io.BytesIO(raw)
        return torch.load(buf, weights_only=True)

    # --------------------------
    # Public API
    # --------------------------

    def __setitem__(self, key: str, value: T) -> None:
        blob = self._serialize(value)
        self.cache[key] = blob

    def __getitem__(self, key: str) -> T:
        blob = self.cache[key]  # raises KeyError if missing
        return self._deserialize(blob)

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
