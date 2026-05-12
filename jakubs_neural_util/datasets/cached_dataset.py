from abc import ABC, abstractmethod

from pathlib import Path
import random
from typing import Generic, NewType, Optional, Tuple, List, Dict, Any, Type, TypeVar, cast, Union, TypedDict, TYPE_CHECKING, final

from torch.utils.data import Dataset

from jakubs_neural_util.datasets.tensor_hashing import hash_dataset_entry
from jakubs_neural_util.datasets.tensor_cache import TensorCache

# In memory info, such as filepath
SourceType = TypeVar("SourceType")
TensorType = TypeVar("TensorType")
# Info that is stored on the disk or otherwise generated from SourceType lazily
ParamsType = TypeVar("ParamsType")

IndiceInner = NewType("IndiceInner", int)

class CachedDataset(Generic[SourceType, ParamsType, TensorType], Dataset[TensorType], ABC):
    def __init__(self, 
                 *,
                 cache_dir: str = "",
                 cache_max_size: int = 500*(1024**3),
                 compress: bool = True,
                 subrange: Optional[Tuple[float, float]] = None,
                 subrange_is_percent: bool = False,
                 shuffle_seed: int = -1
        ):
        """
        Args:
            folder (str): Path to the folder containing *_image_meta.json files.
            is_validation (bool): Flag for validation split.
            subrange (Optional[Tuple[int,int]]): Optional (start, end) indices to restrict dataset.
        """

        self.sourceTypeHelper: SourceType = None # type: ignore

        self.called_base_init_items = False
        self.called_count_items = False
        self.called_init_items = False

        self.cache_system: Optional[TensorCache[TensorType]] = None
        self.cache_dir = cache_dir
        self.cache_max_size = cache_max_size
        self.cache_compress = compress

        self.subrange = subrange
        self.subrange_is_percent = subrange_is_percent
        self.shuffle_seed = shuffle_seed
        self.needs_indice_map = shuffle_seed != -1 or subrange is not None
        self.indice_map: Optional[list[IndiceInner]] = None

    def _typing_source_type(self) -> SourceType:
        if TYPE_CHECKING:
            return self.items[0]
        else:
            raise SyntaxError("Cannot call type helper in runtime!")

    @abstractmethod
    def count_items(self) -> Optional[List[SourceType]]:
        '''
        Must be idempotent and ensure get_len_impl returns a constant value from now on

        May return the full item list if available, but this is not required
        '''
        pass

    @abstractmethod
    def get_item_source_impl(self, idx: int) -> SourceType:
        '''
        Must return source info (ie file path, db id etc) for given indice in the dataset
        '''
        pass

    @abstractmethod
    def get_real_len(self) -> int:
        '''
        Must return dataset len. Should not take subrange into account,
        that is handled by this class. The actual get_len either calls this,
        or returns the len of the cropped indices
        '''
        pass

    @final
    def get_item_source(self, idx: int) -> SourceType:
        if not self.called_count_items:
            self.count_items()
            self.called_count_items = True
        if not self.called_base_init_items:
            self.base_init_items()
        if self.needs_indice_map and self.indice_map is None:
            self.__generate_shuffle()
        if self.indice_map is not None:
            idx = self.indice_map[idx]
        return self.get_item_source_impl(idx)
    
    def iter_sources(self):
        if self.indice_map is not None:
            for idx in self.indice_map:
                yield self.get_item_source_impl(idx)
        else:
            data_len = self.get_real_len()
            for idx in range(0, data_len):
                yield self.get_item_source_impl(idx)
    
    @final
    def get_internal_idx(self, idx: 'IndiceInner') -> int:
        '''
        Use this if you know what unshuffled and uncropped idx
        maps to for get_item_source and get_item_hash

        This maps your inner idx to the idx used by other function

        This is O(n_items)!

        Returns -1 if this indice is unused (because of the subrange param)
        '''
        if self.needs_indice_map and self.indice_map is None:
            self.__generate_shuffle()
        if self.indice_map is not None:
            return self.reverse_indice_map[idx]
        return idx

    @final
    def get_len(self) -> int:
        if self.needs_indice_map and self.indice_map is None:
            self.__generate_shuffle()
        # Elif because the above already calls count_items()
        elif not self.called_count_items:
            self.count_items()
            self.called_count_items = True
        
        return self.get_real_len() if self.indice_map is None else len(self.indice_map)

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
        If any processing pass is needed, this is where it is implemented
        After this is called, get_item_source(x) shold always return the same thing for same x

        This function must not change length if items
        '''

    @final
    def base_init_items(self):
        if not self.called_base_init_items:
            if not self.called_count_items:
                self.count_items()
                self.called_count_items = True
            if len(self.cache_dir) > 0:
                self.cache_system = TensorCache(self.cache_dir, size_limit=self.cache_max_size, use_compression=self.cache_compress)
            self.init_items()
        self.called_base_init_items = True

    def __generate_shuffle(self):
        if not self.needs_indice_map or self.indice_map is not None:
            return
        if not self.called_count_items:
            self.count_items()
            self.called_count_items = True
        
        total_count = self.get_real_len()

        indices = [x for x in range(0, total_count)]
        if self.shuffle_seed > 0:
            random.Random(self.shuffle_seed).shuffle(indices)

        if self.subrange is not None:
            if self.subrange_is_percent:
                start = int(total_count * self.subrange[0])
                end   = int(total_count * self.subrange[1])
            else:
                start, end = self.subrange

            self.indice_map = indices[start:end]
        else:
            self.indice_map = indices

        # Build reverse map: inner_index -> outer_index
        reverse_map = [-1] * total_count
        for outer_idx, inner_idx in enumerate(self.indice_map):
            reverse_map[inner_idx] = outer_idx

        self.reverse_indice_map = reverse_map

        pass

    def get_item_hash(self, idx):
        if not self.called_base_init_items:
            self.base_init_items()

        hash_tuple, dependent_paths = self.get_item_hash_tuple(idx)

        if hasattr(self, "FORMAT_VERSION") and isinstance(self.FORMAT_VERSION, int):
            hash_tuple = (self.FORMAT_VERSION, hash_tuple)

        item_hash = hash_dataset_entry(hash_tuple, dependent_paths)
        return item_hash
    
    def get_item_hash_tuple(self, index: int) -> 'tuple[tuple, list[Path]]':
        '''
        Override this to modify source items for hash purposes (ie. remove things that do not affect load_item)
        '''
        item_input = self.get_item_source(index)
        param_dict, dependent_paths = self.get_item_info(item_input)
        return  (param_dict, item_input), dependent_paths

    def __len__(self) -> int:
        return self.get_len()

    def __getitem__(self, idx: int):
        if not self.called_base_init_items:
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

        item_input = self.get_item_source(idx)

        items_tensors = self.load_item(item_input)

        if must_save_cache and self.cache_system is not None:
            self.cache_system[item_hash] = items_tensors
        return items_tensors
