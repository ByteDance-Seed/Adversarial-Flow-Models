from typing import Callable, Optional

from data.parquet_imagenet import ImagenetParquetDataset


def create_dataset(
    image_transform: Optional[Callable] = None,
    **kwargs,
):
    return ImagenetParquetDataset(
        path="imagenet/parquet/train", # Replace to your path.
        image_transform=image_transform,
        **kwargs,
    )
