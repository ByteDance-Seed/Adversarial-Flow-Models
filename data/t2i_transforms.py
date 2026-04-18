import math
from typing import Union
import torch
from PIL import Image
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode


class AreaResize:
    def __init__(
        self,
        max_area: float,
        downsample_only: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.max_area = max_area
        self.downsample_only = downsample_only
        self.interpolation = interpolation

    def __call__(self, image: Union[torch.Tensor, Image.Image]):

        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        scale = math.sqrt(self.max_area / (height * width))

        # keep original height and width for small pictures.
        scale = 1 if scale >= 1 and self.downsample_only else scale

        resized_height, resized_width = round(height * scale), round(width * scale)

        return TVF.resize(
            image,
            size=(resized_height, resized_width),
            interpolation=self.interpolation,
        )


class DivisibleCrop:
    def __init__(self, factor):
        if not isinstance(factor, tuple):
            factor = (factor, factor)

        self.height_factor, self.width_factor = factor[0], factor[1]

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        cropped_height = height - (height % self.height_factor)
        cropped_width = width - (width % self.width_factor)

        image = TVF.center_crop(img=image, output_size=(cropped_height, cropped_width))
        return image
