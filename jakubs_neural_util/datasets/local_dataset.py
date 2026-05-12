from abc import ABC, abstractmethod

from pathlib import Path
from typing import Generic, Optional, Tuple, List, Dict, Any, TypeVar, cast, Union, TypedDict

import torch.nn.functional as F
from torch.utils.data import Dataset

import random

from jakubs_neural_util.datasets.cached_dataset import CachedDataset
from jakubs_neural_util.datasets.tensor_hashing import hash_dataset_entry
from jakubs_neural_util.datasets.tensor_cache import TensorCache

# Type that typically has one value per item
SourceType = TypeVar("SourceType")
TensorType = TypeVar("TensorType")
ParamsType = TypeVar("ParamsType")

class LocalDataset(CachedDataset[SourceType, ParamsType, TensorType], Generic[SourceType, ParamsType, TensorType]):
    def __init__(self, 
                 folder: str,
                 *,
                 glob_pattern: str = "*.json",
                 recursive_glob: bool = False,
                 cache_dir: str = "",
                 cache_max_size: int = 500*(1024**3),
                 is_validation: bool = False, 
                 subrange: Optional[Tuple[float, float]] = None,
                 subrange_is_percent: bool = False,
                 shuffle_seed: int = -1):
        """
        Args:
            folder (str): Path to the folder containing *_image_meta.json files.
            is_validation (bool): Flag for validation split.
            subrange (Optional[Tuple[int,int]]): Optional (start, end) indices to restrict dataset.
        """
        super().__init__(
            cache_dir=cache_dir,
            cache_max_size=cache_max_size,
            subrange=subrange,
            subrange_is_percent=subrange_is_percent,
            shuffle_seed=shuffle_seed)
        
        self.folder = Path(folder)
        self.is_validation = is_validation
        self.shuffle_seed = shuffle_seed
        self.subrange = subrange
        self.subrange_is_percent = subrange_is_percent

        # Collect all JSON files ending with _image_meta.json

        self.glob_pattern = glob_pattern
        self.recursive_glob = recursive_glob
        self.files: List[Path] = []

    def count_items(self) -> Optional[List[SourceType]]:
        globres = self.folder.glob(self.glob_pattern) if not self.recursive_glob else self.folder.rglob(self.glob_pattern)
        self.files = sorted(globres)

    @abstractmethod
    def get_real_len(self) -> int:
        pass

    @abstractmethod
    def load_item(self, item: SourceType) -> TensorType:
        pass

    @abstractmethod
    def init_items(self):
        pass
