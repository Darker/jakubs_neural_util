from abc import ABC, abstractmethod

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, Optional, Tuple, List, Dict, Any, Type, TypeVar, cast, Union, TypedDict, get_args

import torch.nn.functional as F
from torch.utils.data import Dataset

import random

from jakubs_neural_util.datasets.cached_dataset import CachedDataset, SourceType as CSourceType


TensorType = TypeVar("TensorType")

SourceTensorType = TypeVar("SourceTensorType")
SourceParamsType = TypeVar("SourceParamsType")
SourceItemType = TypeVar("SourceItemType")


TTypeVarStaticTensor = TypeVar("TTypeVarStaticTensor")
@dataclass
class DerivedItemMapping():
    source_indices: list[int]


class DerivedDataset(
    CachedDataset[DerivedItemMapping, list[SourceParamsType], TensorType],
    Generic[TensorType, SourceItemType, SourceParamsType, SourceTensorType]):
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

    @staticmethod
    def get_type(base: Type[CachedDataset[SourceParamsType, SourceItemType, SourceTensorType]], tensorType: TTypeVarStaticTensor):
        return DerivedDataset[TTypeVarStaticTensor, SourceParamsType, SourceItemType, SourceTensorType]

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

INHSourceType = TypeVar("INHSourceType", covariant=True)
INHItemType = TypeVar("INHItemType")
INHTensorType = TypeVar("INHTensorType")
INHFinalTensorType = TypeVar("INHFinalTensorType")
def DerivedDataset_inherit(
        sourceType: Type[CachedDataset[INHSourceType, INHItemType, INHTensorType]],
        tensorType: Type[INHFinalTensorType])->Type[DerivedDataset[INHFinalTensorType, INHSourceType, INHItemType, INHTensorType]]:
    return DerivedDataset # type: ignore

# class TestDerived(CachedDataset[Literal["source_type"], Literal["params_type"], Literal["tensor_type"]]):
#     def create_items(self) -> List[Literal["source_type"]]:
#         raise NotImplementedError

#     def get_item_info(self, item: Literal["source_type"]) -> Tuple[Literal["params_type"], List[Path] | None]:
#         raise NotImplementedError

#     def load_item(self, item: Literal["source_type"]) ->  Literal["tensor_type"]:
#         raise NotImplementedError

#     def init_items(self):
#         raise NotImplementedError


# class TestDerivedImpl(DerivedDataset_inherit(TestDerived, list[float])):
#     def __init__(self, source: TestDerived):
#         super(TestDerivedImpl, self).__init__(source=source)

#     def create_items(self) -> List[DerivedItemMapping]:
#         raise NotImplementedError

#     def load_item(self, item: DerivedItemMapping):
#         raise NotImplementedError

#     def init_items(self):
#         raise NotImplementedError

# # testDerInst = TestDerived()
# # derivedInst = DerivedDataset(testDerInst, Type[list[float]])

# implTest = TestDerivedImpl(TestDerived())

# implTest.source.get_item_info("source_type")