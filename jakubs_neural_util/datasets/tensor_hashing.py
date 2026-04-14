import hashlib
from os import PathLike
import pickle
from typing import Any, Iterable, Optional
from pathlib import Path
from dataclasses import asdict, is_dataclass

def freeze(obj: Any) -> Any:
    if is_dataclass(obj):
        return freeze(asdict(obj))
    if isinstance(obj, dict):
        # sort keys for deterministic ordering
        return tuple((k, freeze(obj[k])) for k in sorted(obj))
    if isinstance(obj, tuple):
        return tuple(freeze(v) for v in obj)
    if isinstance(obj, list):
        return tuple(freeze(v) for v in obj)
    if isinstance(obj, PathLike) or isinstance(obj, Path):
        return str(obj)
    return obj  # numbers, strings, bools, None

def hash_dataset_entry(
    data: object,
    dependent_paths: Optional[Iterable[Path]] = None,
) -> str:
    
    frozen_data = freeze(data)

    h = hashlib.sha256(usedforsecurity=False)
    
    # Deterministic serialization
    h.update(pickle.dumps(frozen_data, protocol=5))

    if dependent_paths is not None:
        for filepath in sorted(dependent_paths, key=str):
            mtime = filepath.stat().st_mtime
            h.update(str(filepath).encode("utf-8"))
            h.update(str(int(mtime)).encode("utf-8"))

    return h.hexdigest()
