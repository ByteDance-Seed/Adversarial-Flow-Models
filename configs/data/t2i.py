from data.parquet_t2i import ImageParquetDataset


def create_dataset(**kwargs):
    # Feel free to replace the following to your dataloading class.

    return ImageParquetDataset(
        path="./t2i",
        **kwargs,
    )
