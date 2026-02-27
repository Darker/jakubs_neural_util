from abc import ABC, abstractmethod

from pathlib import Path
from typing import Generic, Optional, Tuple, List, Dict, Any, TypeVar, cast, Union, TypedDict

import torch.nn.functional as F
from torch.utils.data import Dataset

import random

from jakubs_neural_util.datasets.cached_dataset import CachedDataset
from jakubs_neural_util.datasets.tensor_hashing import hash_dataset_entry
from jakubs_neural_util.datasets.tensor_cache import TensorCache

SourceType = TypeVar("SourceType")
TensorType = TypeVar("TensorType")
ParamsType = TypeVar("ParamsType")

class LocalDataset(CachedDataset[SourceType, ParamsType, TensorType], Generic[SourceType, ParamsType, TensorType]):
    def __init__(self, 
                 folder: str,
                 *,
                 glob_pattern: str = "*.json",
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
        super().__init__(cache_dir=cache_dir)
        self.folder = Path(folder)
        self.is_validation = is_validation
        self.shuffle_seed = shuffle_seed
        self.subrange = subrange
        self.subrange_is_percent = subrange_is_percent

        # Collect all JSON files ending with _image_meta.json
        all_files = sorted(self.folder.glob(glob_pattern))

        self.files: List[Path] = all_files
        self.items: List[SourceType] = []

    @abstractmethod
    def create_items(self) -> List[SourceType]:
        '''
        Must populate self.items
        '''
        pass
    
    @abstractmethod
    def get_item_info(self, item: SourceType) -> tuple[ParamsType, Optional[List[Path]]]:
        '''
        Returns parsed item params and optionally list of paths it depends on
        The paths are only used for hashing for cache
        '''
        pass

    @abstractmethod
    def load_item(self, item: SourceType) -> TensorType:
        pass

    def apply_range_shuffle(self):
        if self.shuffle_seed > 0:
            random.Random(self.shuffle_seed).shuffle(self.items)
        total_files = len(self.items)

        if self.subrange is not None:
            if self.subrange_is_percent:
                start = int(total_files * self.subrange[0])
                end   = int(total_files * self.subrange[1])
            else:
                start, end = self.subrange

            self.items = self.items[start:end]

    def init_items(self):
        self.items = self.create_items()
        self.apply_range_shuffle()
