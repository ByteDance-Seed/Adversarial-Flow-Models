from typing import Any, Callable, Literal, Optional
import random
import numpy as np
import torch
from torch import nn

from common.logger import get_logger
from data.parquet_imagenet_base import ParquetDataset


class ImagenetParquetDataset(ParquetDataset):
    def __init__(
        self,
        path: str,
        *,
        infinite: bool = True,
        path_mode: Literal["dir", "file"] = "dir",
        seed: Optional[int] = None,
        shuffle: bool = True,
        image_transform: Callable[[torch.Tensor], Any] = nn.Identity(),
        overwrite_metafile: bool = True,
    ):
        super().__init__(
            path=path,
            seed=seed,
            infinite=infinite,
            path_mode=path_mode,
            shuffle=shuffle,
            num_row_groups_per_file=69, # Our parquet has 69 row groups per file.
            overwrite_metafile=overwrite_metafile,
            disable_metafile=False,
            separate_meta_storage=False,
        )
        self.image_transform = image_transform
        self.logger = get_logger(self.__class__.__name__)

    def __iter__(self):
        for sample in super().__iter__():
            try:
                # Image decode.
                key = random.choice(["latent", "latent_xflip"])
                latent = sample[key]
                latent = np.frombuffer(latent, dtype=np.float32).reshape(4, 32, 32)
                latent = torch.from_numpy(latent)

                # Class condition.
                label = sample["label"]
                label = torch.tensor(label)

                yield {
                    "latent": latent,
                    "label": label,
                }

            except Exception as ex:
                self.logger.warn(f"ImageParquetDataset got unexpected expcetion: {ex}")
                continue
