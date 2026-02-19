from abc import ABC, abstractmethod

from pathlib import Path
from typing import Generic, Optional, Tuple, List, Dict, Any, TypeVar, cast, Union, TypedDict

import torch.nn.functional as F
from torch.utils.data import Dataset

import random

from jakubs_neural_util.datasets.tensor_hashing import hash_dataset_entry
from jakubs_neural_util.datasets.tensor_cache import TensorCache

SourceType = TypeVar("SourceType")
ItemType = TypeVar("ItemType")
ParamsType = TypeVar("ParamsType")

class LocalDataset(Generic[SourceType, ParamsType, ItemType], Dataset[ItemType], ABC):
    def __init__(self, 
                 folder: str,
                 *,
                 glob_pattern: str = "*.json",
                 cache_dir: str = "",
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
        self.folder = Path(folder)
        self.is_validation = is_validation
        self.shuffle_seed = shuffle_seed
        self.subrange = subrange
        self.subrange_is_percent = subrange_is_percent

        # Collect all JSON files ending with _image_meta.json
        all_files = sorted(self.folder.glob(glob_pattern))

        self.files: List[Path] = all_files
        self.items: List[SourceType] = []

        self.did_init = False

        if len(cache_dir) > 0:
            self.cache_system: Optional[TensorCache[ItemType]] = TensorCache(cache_dir, 500*(1024**3))
        else:
            self.cache_system = None

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
    def load_item(self, item: SourceType) -> ItemType:
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
        self.did_init = True

    def __len__(self) -> int:
        if not self.did_init:
            self.init_items()
        return len(self.items)

    def __getitem__(self, idx: int):
        if not self.did_init:
            self.init_items()

        must_save_cache = False
        item_input = self.items[idx]
        item_hash = ""
        # hashing
        if self.cache_system is not None:
            param_dict, dependent_paths = self.get_item_info(item_input)
            item_hash = hash_dataset_entry((param_dict, item_input), dependent_paths)

            if item_hash in self.cache_system:
                #print(f"Cache hit, hash {item_hash}")
                return self.cache_system[item_hash]
            else:
                #print(f"Cache miss, hash {item_hash}")
                must_save_cache = True
                # print("Cache miss: "+str(item_input))

        items_tensors = self.load_item(item_input)

        if must_save_cache and self.cache_system is not None:
            self.cache_system[item_hash] = items_tensors
        return items_tensors
