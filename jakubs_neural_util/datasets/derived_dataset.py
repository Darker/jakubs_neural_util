from abc import ABC, abstractmethod

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Optional, Tuple, List, Dict, Any, TypeVar, cast, Union, TypedDict

import torch.nn.functional as F
from torch.utils.data import Dataset

import random

from jakubs_neural_util.datasets.cached_dataset import CachedDataset

InputType = TypeVar("InputType")
TensorType = TypeVar("TensorType")
ParamsType = TypeVar("ParamsType")

SourceTensorType = TypeVar("SourceTensorType")
SourceParamsType = TypeVar("SourceParamsType")
SourceItemType = TypeVar("SourceItemType")


@dataclass
class DerivedItemMapping():
    source_indices: list[int]

class DerivedDataset(CachedDataset[DerivedItemMapping, list[SourceParamsType], TensorType], Generic[SourceParamsType, TensorType, SourceItemType]):
    """
        This dataset creates its items from another dataset. Typical use would be image pairing,
        creating more image variants etc. Requires CachedDataset, and this means both variants
        and their sources will be cached.
    """
    def __init__(self, 
                 source: CachedDataset[SourceItemType, SourceParamsType, SourceTensorType],
                 *,
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
        super().__init__(cache_dir=cache_dir)

        self.is_validation = is_validation
        self.shuffle_seed = shuffle_seed
        self.subrange = subrange
        self.subrange_is_percent = subrange_is_percent
        self.source = source

        self.items: List[DerivedItemMapping] = []

    @abstractmethod
    def create_items(self) -> List[DerivedItemMapping]:
        pass
    
    def get_item_info(self, item: DerivedItemMapping) -> tuple[list[SourceParamsType], Optional[List[Path]]]:
        items: list[SourceItemType] = []
        for index in item.source_indices:
            items.append(self.source.items[index])

        # Now load all the items metadata
        source_infos: list[SourceParamsType] = []
        source_paths: set[Path] = set()
        for info in items:
            info, dependencies = self.source.get_item_info(info)
            source_infos.append(info)
            if dependencies:
                for dep in dependencies:
                    source_paths.add(dep)

        return source_infos, [x for x in source_paths]

    @abstractmethod
    def load_item(self, item: DerivedItemMapping) -> TensorType:
        pass