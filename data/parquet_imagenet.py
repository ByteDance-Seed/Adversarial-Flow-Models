from typing import Any, Callable, Literal, Optional
import io
import numpy as np
import torch
from PIL import Image
from torch import nn

from common.logger import get_logger
from data.parquet_base import ParquetDataset


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
            separate_meta_storage=True,
        )
        self.image_transform = image_transform
        self.logger = get_logger(self.__class__.__name__)

    def __iter__(self):
        for sample in super().__iter__():
            try:
                # Image decode.
                image_bytes = sample["data"]
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image = image.convert("RGB")

                # Image transform.
                if self.image_transform is not None:
                    image = self.image_transform(image)

                # Class condition.
                label = sample["label"]
                if isinstance(label, np.ndarray):
                    label = label[0]
                label = torch.tensor(int(label))

                yield {
                    "image": image,
                    "label": label,
                }

            except Exception as ex:
                self.logger.warn(f"ImageParquetDataset got unexpected expcetion: {ex}")
                continue
