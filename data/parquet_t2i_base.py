from functools import partial
from itertools import chain
from multiprocessing import Pool
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
    """
    Parquet dataset.

    Arguments:
        path: a directory path that contains *.parquet files.
        seed: seed for deterministic sampling. If None, just random.
        partition: partition strategy. Split by *.parquet file or by row groups in each file.
        force_partition: if True, raise error if partition is indivisible.
        num_parallel_files: number of parallel files to read.
        infinite: If True, data will be returned infinitely.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        seed: Optional[int],
        partition: Literal["file", "group", "dump"] = "file",
        force_partition: bool = False,
        num_parallel_files: int = 2,
        infinite: bool = True,
        path_mode: Literal["dir", "file"] = "dir",
        shuffle: bool = True,
        overwrite_metafile: bool = True,
        disable_metafile: bool = False,
    ):
        assert partition in ["file", "group", "dump"]
        assert path_mode in ["dir", "file"]

        # Save settings.
        self.seed = seed
        self.infinite = infinite
        self.partition = partition
        self.force_partition = force_partition
        self.num_parallel_files = num_parallel_files
        self.shuffle = shuffle

        list_dir_func = partial(
            listdir_with_metafile,
            overwrite=overwrite_metafile,
            disable=disable_metafile,
        )

        # List file paths.
        file_paths = path if isinstance(path, list) else [path]
        if path_mode == "dir":
            file_paths = map(list_dir_func, file_paths)
            file_paths = chain(*file_paths)
            file_paths = filter(lambda path: path.endswith(".parquet"), file_paths)
            file_paths = sorted(file_paths)

        assert len(file_paths) > 0
        self.file_paths = file_paths

        # Create file readers.
        self.filereaders = [
            ParquetFileReader(
                path=path,
                seed=seed,
                partition=partition,
                force_partition=force_partition,
                shuffle=shuffle,
            )
            for path in file_paths
        ]

    def __iter__(self):
        epoch = 0
        filereaders = self.filereaders
        random = get_random_for_rank_and_worker(self.seed)

        # Partition by files if needed.
        if self.partition == "file":
            filereaders = get_portion_for_rank_and_worker(filereaders, self.force_partition)

        while True:
            # Initialize filereaders iterators.
            iterators = [reader.__iter__(epoch=epoch) for reader in filereaders]
            if self.shuffle:
                random.shuffle(iterators)

            # Yield samples.
            while any(iterators):
                if self.shuffle:
                    iterator = random.choice(iterators[: self.num_parallel_files])
                else:
                    iterator = iterators[0]
                try:
                    yield next(iterator)
                except:
                    iterators.remove(iterator)

            # Break after the first epoch if not infinite.
            if not self.infinite:
                break

            # Increment epoch.
            epoch += 1


class ParquetFileReader:
    """
    Read a single *.parquet file.

    Arguments:
        path: a *.parquet file path.
        seed: seed for deterministic sampling. If None, just random.
        partition: partition strategy.
        force_partition: if True, raise error if partition is indivisible.
    """

    def __init__(
        self,
        path: str,
        seed: Optional[int],
        partition: bool,
        force_partition: bool,
        shuffle: bool,
    ):
        self.path = path
        self.seed = seed
        self.partition = partition
        self.force_partition = force_partition
        self.shuffle = shuffle

    def __len__(self):
        fs = get_filesystem(self.path)
        with ParquetFile(self.path, filesystem=fs) as file:
            return file.metadata.num_rows

    def __iter__(self, epoch=0):

        fs = get_filesystem(self.path)
        with ParquetFile(self.path, filesystem=fs) as file:

            # List all groups.
            groups = list(range(file.num_row_groups))

            # Partition groups if needed.
            if self.partition == "group":
                groups = get_portion_for_rank_and_worker(groups, self.force_partition)

            if self.shuffle:
                # Shuffle groups
                seed = (self.seed + epoch) if self.seed is not None else None
                get_random_for_rank_and_worker(seed).shuffle(groups)

            # Iteration over all samples from all row groups.
            for group in groups:

                file_batches = file.iter_batches(
                    batch_size=1, row_groups=[group]
                )

                for sample in file_batches:
                    data = sample.to_pandas().iloc[0].to_dict()
                    yield data
