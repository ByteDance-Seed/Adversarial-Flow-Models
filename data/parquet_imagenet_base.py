import random
from functools import partial
from itertools import chain
from typing import List, Literal, Optional, Union
from pyarrow.parquet import ParquetFile
from torch.utils.data import IterableDataset

from common.fs import listdir_with_metafile
from data.utils import (
    get_filesystem,
    get_portion_for_rank_and_worker,
    get_random_for_rank_and_worker,
)


class ParquetDataset(IterableDataset):
    def __init__(
        self,
        path: Union[str, List[str]],
        seed: Optional[int],
        num_row_groups_per_file: int,
        infinite: bool = True,
        path_mode: Literal["dir", "file"] = "dir",
        shuffle: bool = True,
        overwrite_metafile: bool = True,
        disable_metafile: bool = False,
        separate_meta_storage: bool = False,
    ):
        assert path_mode in ["dir", "file"]

        # Save settings.
        self.seed = seed
        self.infinite = infinite
        self.shuffle = shuffle

        list_dir_func = partial(
            listdir_with_metafile,
            overwrite=overwrite_metafile,
            disable=disable_metafile,
            separate_meta_storage=separate_meta_storage,
        )

        # List file paths.
        file_paths = path if isinstance(path, list) else [path]
        if path_mode == "dir":
            file_paths = map(list_dir_func, file_paths)
            file_paths = chain(*file_paths)
            file_paths = list(file_paths)
            file_paths = sorted(file_paths)

        assert len(file_paths) > 0
        self.file_paths = file_paths

        # Create file readers.
        self.filereaders = []

        for path in file_paths:
            for group in range(num_row_groups_per_file):
                self.filereaders.append(
                    ParquetGroupReader(
                        path=path,
                        group=group,
                        shuffle=shuffle,
                    )
                )

    def __iter__(self):
        filereaders = self.filereaders
        filereaders = get_portion_for_rank_and_worker(filereaders, force=True)
        random = get_random_for_rank_and_worker(self.seed)

        while True:
            # Initialize filereaders iterators.
            iterators = [iter(reader) for reader in filereaders]
            if self.shuffle:
                random.shuffle(iterators)

            # Yield samples.
            while any(iterators):
                iterator = iterators[0]
                try:
                    yield next(iterator)
                except:
                    iterators.remove(iterator)

            # Break after the first epoch if not infinite.
            if not self.infinite:
                break


class ParquetGroupReader:
    def __init__(
        self,
        path: str,
        group: int,
        shuffle: bool,
    ):
        self.path = path
        self.group = group
        self.shuffle = shuffle

    def __iter__(self):
        fs = get_filesystem(self.path)

        with ParquetFile(self.path, filesystem=fs) as file:
            row_group = file.read_row_group(self.group).to_pandas().to_dict(orient='records')

            if self.shuffle:
                random.shuffle(row_group)

            for sample in row_group:
                yield sample
