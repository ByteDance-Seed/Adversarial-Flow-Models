import io
import json
import random
import os
import numpy as np
from typing import Any, Callable, Literal, Optional
import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import center_crop, to_tensor

from common.logger import get_logger
from .parquet_t2i_base import ParquetDataset


class ImageParquetDataset(ParquetDataset):
    """
    Image parquet dataset.

    Arguments:
        path: a directory path that contains *.parquet files.
        seed: seed for deterministic sampling. If None, just random.
        image_transform: a callable function to perform transformation on image.
        text_transform: a callable function to perform transformation on text.
        partition: partition strategy. Split by *.parquet file or by row groups in each file.
        force_partition: if True, raise error if partition is indivisible.
        num_parallel_files: number of parallel files to read.
        infinite: If True, data will be returned infinitely.
    """

    def __init__(
        self,
        path: str,
        *,
        infinite: bool = True,
        partition: Optional[Literal["file", "group"]] = "file",
        force_partition: bool = False,
        path_mode: Literal["dir", "file"] = "dir",
        seed: Optional[int] = None,
        num_parallel_files: int = 2,
        shuffle: bool = True,
        image_transform: Callable[[torch.Tensor], Any] = nn.Identity(),
        text_transform: Callable[[str], Any] = nn.Identity(),
        overwrite_metafile: bool = False,
        filter_color: bool = True,
    ):
        super().__init__(
            path=path,
            seed=seed,
            infinite=infinite,
            num_parallel_files=num_parallel_files,
            partition=partition,
            force_partition=force_partition,
            path_mode=path_mode,
            shuffle=shuffle,
            overwrite_metafile=overwrite_metafile,
            disable_metafile=False,
        )
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.filter_color = filter_color
        self.logger = get_logger(self.__class__.__name__)

    def __iter__(self):
        for sample in super().__iter__():
            try:
                source = sample["data_source"]

                # Caption.
                caption = None

                has_recaption = sample.get("caption_dict") is not None
                has_original = sample.get("origin_caption_dict") is not None

                # Disable original caption
                if has_original and (random.random() < 0.0 or not has_recaption):
                    # Load original caption
                    captions = json.loads(sample["origin_caption_dict"])
                    caption = captions.get("en_text", None)

                if caption is None and has_recaption:
                    captions = json.loads(sample["caption_dict"])
                    # example: no_title_qwen_caption_en_v2_text
                    keys = [k for k in captions.keys() if k.endswith("_text") and "_en_" in k]
                    caption = captions[random.choice(keys)]
                    # filter prefix
                    prefies = ["The image shows ", "The image depicts ", "The image features ", "This image captures ", "The image showcases "]
                    for prefix in prefies:
                        if caption.startswith(prefix):
                            caption = caption[len(prefix) :].capitalize()
                            break
                    
                    # Make caption shorter.
                    # if random.random() < 0.95:
                    if random.random() < 0.5:
                        try:
                            if caption.index(".") > 15:
                                caption = caption.split(".")[0]
                        except:
                            pass

                        try:
                            if caption.index(",") > 15:
                                caption = caption.split(",")[0]
                        except:
                            pass


                if caption is None:
                    # Unconditional case.
                    self.logger.warn(f"Found image with no caption.")
                    caption = ""

                # Image decode.
                image_bytes = sample["image"]
                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
                        continue
                    else:
                        image = image.convert("RGB")
                
                w, h = image.size
                if w / h > 2 or h / w > 2:
                    continue

                if self.filter_color and has_large_uniform_color_region(image, 0.3):
                    continue

                image = to_tensor(image)

                # Image transform.
                image = self.image_transform(image)

                yield {
                    "image": image,
                    "text": caption,
                    "source": source,
                }

            except Exception as ex:
                self.logger.warn(f"ImageParquetDataset got unexpected expcetion: {ex}")
                continue



def has_large_uniform_color_region(
    pil_img: Image.Image,
    threshold: float = 0.35,
    resize_to: int = 256,
    n_colors: int = 32,
):
    """
    Returns True if the image has a dominant color occupying > threshold of pixels.

    Parameters
    ----------
    pil_img : PIL.Image
    threshold : float
        Percentage of pixels that must share the same color (0–1).
        Typical values:
            0.30–0.40 -> cartoons / flat graphics
            0.50+     -> very flat images / icons
    resize_to : int
        Downsample longest side to this size for speed.
    n_colors : int
        Number of colors for quantization.

    """
    # Convert to RGB
    img = pil_img.convert("RGB")

    # Downsample (keeps aspect ratio)
    img.thumbnail((resize_to, resize_to), Image.Resampling.BILINEAR)

    # Quantize to reduce tiny color variations
    img_q = img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)

    # Convert to numpy
    arr = np.array(img_q)

    # Count pixels per color index
    counts = np.bincount(arr.flatten())
    dominant_ratio = counts.max() / counts.sum()

    return dominant_ratio >= threshold