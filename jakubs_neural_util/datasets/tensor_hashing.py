import hashlib
import pickle
from typing import Any, Iterable, Optional
from pathlib import Path

def freeze(obj: Any) -> Any:
    if isinstance(obj, dict):
        # sort keys for deterministic ordering
        return tuple((k, freeze(obj[k])) for k in sorted(obj))
    if isinstance(obj, list):
        return tuple(freeze(v) for v in obj)
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
