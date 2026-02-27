from abc import ABC, abstractmethod

from pathlib import Path
from typing import Generic, Optional, Tuple, List, Dict, Any, TypeVar, cast, Union, TypedDict, TYPE_CHECKING, final

from torch.utils.data import Dataset

from jakubs_neural_util.datasets.tensor_hashing import hash_dataset_entry
from jakubs_neural_util.datasets.tensor_cache import TensorCache

# In memory info, such as filepath
SourceType = TypeVar("SourceType")
TensorType = TypeVar("TensorType")
# Info that is stored on the disk or otherwise generated from SourceType lazily
ParamsType = TypeVar("ParamsType")

class CachedDataset(Generic[SourceType, ParamsType, TensorType], Dataset[TensorType], ABC):
    def __init__(self, 
                 *,
                 cache_dir: str = "",
                 cache_max_size: int = 500*(1024**3)
        ):
        """
        Args:
            folder (str): Path to the folder containing *_image_meta.json files.
            is_validation (bool): Flag for validation split.
            subrange (Optional[Tuple[int,int]]): Optional (start, end) indices to restrict dataset.
        """

        self.sourceTypeHelper: SourceType = None # type: ignore

        self.items: List[SourceType] = []

        self.did_init = False

        self.cache_system: Optional[TensorCache[TensorType]] = None
        self.cache_dir = cache_dir
        self.cache_max_size = cache_max_size

    def _typing_source_type(self) -> SourceType:
        if TYPE_CHECKING:
            return self.items[0]
        else:
            raise SyntaxError("Cannot call type helper in runtime!")

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

    @abstractmethod
    def init_items(self):
        '''
        Initialized list of items to cache
        '''

    @final
    def base_init_items(self):
        if not self.did_init:
            if len(self.cache_dir) > 0:
                self.cache_system = TensorCache(self.cache_dir, self.cache_max_size)
            self.init_items()
        self.did_init = True

    def get_item_hash(self, idx):
        if not self.did_init:
            self.base_init_items()
        item_input = self.items[idx]
        param_dict, dependent_paths = self.get_item_info(item_input)
        item_hash = hash_dataset_entry((param_dict, item_input), dependent_paths)
        return item_hash


    def __len__(self) -> int:
        if not self.did_init:
            self.base_init_items()
        return len(self.items)

    def __getitem__(self, idx: int):
        if not self.did_init:
            self.base_init_items()

        must_save_cache = False
        
        item_hash = ""
        # hashing
        if self.cache_system is not None:
            item_hash = self.get_item_hash(idx)

            if item_hash in self.cache_system:
                #print(f"Cache hit, hash {item_hash}")
                return self.cache_system[item_hash]
            else:
                #print(f"Cache miss, hash {item_hash}")
                must_save_cache = True
                # print("Cache miss: "+str(item_input))

        item_input = self.items[idx]

        items_tensors = self.load_item(item_input)

        if must_save_cache and self.cache_system is not None:
            self.cache_system[item_hash] = items_tensors
        return items_tensors
