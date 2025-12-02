import random
from typing import Any, List, Optional, Union
from pyarrow.fs import HadoopFileSystem, LocalFileSystem
from torch.utils.data import get_worker_info

from common.distributed import get_global_rank, get_world_size
from common.partition import partition_by_groups


def get_worker_id() -> int:
    """
    Get the current dataloader worker id.
    """
    return get_worker_info().id if get_worker_info() is not None else 0


def get_worker_count() -> int:
    """
    Get the total dataloader worker count.
    """
    return get_worker_info().num_workers if get_worker_info() is not None else 1


def get_seed_for_rank_and_worker(seed: Optional[int]) -> Optional[int]:
    """
    Get seed for current rank and worker.
    """
    if seed is None:
        return None
    return seed + get_global_rank() * get_worker_count() + get_worker_id()


def get_random_for_rank_and_worker(seed: Optional[int]) -> random.Random:
    """
    Get random.Random for the current rank and worker.
    """
    return random.Random(get_seed_for_rank_and_worker(seed))



def get_portion_for_rank_and_worker(items: List[Any], force: bool = False) -> List[Any]:
    """
    Get the portion of items for current rank and worker.
    """
    rank = get_global_rank()
    world_size = get_world_size()
    worker_id = get_worker_id()
    worker_count = get_worker_count()

    if world_size * worker_count <= len(items):
        # If there are enough items to be divided, we divide the items
        items = partition_by_groups(items, world_size)[rank]
        items = partition_by_groups(items, worker_count)[worker_id]
    elif not force:
        # If not enough items to be divided, all ranks and workers shuffle it
        # with different seed.
        items = list(items)
        get_random_for_rank_and_worker(0).shuffle(items)
    else:
        raise ValueError(f"Items {len(items)} not divisible by world_size * worker_count")
    return items


def get_filesystem(path: str) -> Union[LocalFileSystem, HadoopFileSystem]:
    """
    Get filesystem based on the path.
    """
    if path.startswith("hdfs://"):
        return HadoopFileSystem.from_uri(path)
    else:
        return LocalFileSystem()
