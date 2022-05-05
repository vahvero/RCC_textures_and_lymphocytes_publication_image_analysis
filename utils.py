__author__ = "MSc. Otso Brummer, <https://github.com/vahvero>"
__date__ = "2022-05-5"


import os
import pickle
import random
from math import ceil
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import PIL
import torch
import torchvision as tv
from matplotlib.colors import ListedColormap
from openslide import OpenSlide
from scipy import ndimage
from torch import BoolTensor, Tensor
from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import rgb_to_grayscale, to_tensor

def load_image_to_tensor(path:str) -> Tensor:
    """Loads path image as tensor

    Args:
        path (str)): Path to the image.

    Returns:
        Tensor: RGB converted TorchVision tensor
    """
    with PIL.Image.open(path) as img:
        img = img.convert("RGB")
        return to_tensor(img)

def no_texture(
    tensor: Tensor, min_limit: float = 0.9, max_proportion_empty=0.1
) -> BoolTensor:
    """Check if the total color values seem to suggest that
    the tensor image does not contain any tissue.

    Args:
        tensor (Tensor): Image tensor batch [N, 3, H, W].

    Returns:
        BoolTensor: Tensor of the image in batch containing texture.
    """
    # Single image
    if len(tensor.shape) == 3 and tensor.shape[0] == 3:
        # transform to batch
        tensor = tensor.unsqueeze(0)

    assert len(tensor.shape) == 4 and tensor.shape[1] == 3, "Invalid dimensions"

    sum_limit = int(tensor.shape[2] * tensor.shape[3] * max_proportion_empty)
    ret = (
        torch.sum(rgb_to_grayscale(tensor).flatten(start_dim=1) < min_limit, dim=1)
        < sum_limit
    )

    return ret

def read_wsi_region(
    filename: str,
    coordinates: Tuple[int, int],
    size: Tuple[int, int],
) -> PIL.Image.Image:
    """Reads a part WSI and converts it to RGB.

    Args:
        filename (str): WSI filepath
        coordinates (Tuple[int, int]): Upper left corner of the region.
        size (Tuple[int, int]): Size of the region. (X, Y).

    Returns:
        PIL.Image of the area
    """
    with OpenSlide(filename) as fobj:
        img = fobj.read_region(coordinates, 0, size).convert("RGB")

    return img


class SlideDataset(Dataset):
    """Create torch dataset from open OpenSlide
    file connection where subsample size is `size`

    Subclassing image generation gives access
    to torch multiprocessing capabilities which
    offers signigicant speed increase compared to
    naive approach

    Args:
        image_connection: Path to slide a image.
        size (int, int): Size of the evaluated crop

    """

    cache_folder = "dataset_cache"

    def __init__(
        self,
        slide_filename: str,
        size: Tuple[int, int],
    ):

        self.slide_filename = slide_filename
        self.size = size
        self.os_conn = None
        self.return_type: Literal["img", "tensor"] = "tensor"
        self.transform = tv.transforms.Compose([tv.transforms.ToTensor()])

        with OpenSlide(self.slide_filename) as fobj:
            width, height = fobj.dimensions

        self.dimensions = (
            ceil(width / self.size[0]) - 1,
            ceil(height / self.size[1]) - 1,
        )

        self.image_positions = []
        for xidx in range(0, self.dimensions[0]):
            for yidx in range(0, self.dimensions[1]):
                xvalue = self.size[0] * xidx
                yvalue = self.size[1] * yidx
                assert width > xvalue + self.size[0]
                assert height > yvalue + self.size[1]
                self.image_positions.append((xvalue, yvalue))

        if len(self) != self.dimensions[0] * self.dimensions[1]:
            raise ValueError(("Image dimensions do not respond to dataset length"))

    @property
    def tcga_identifier(self) -> str:
        return extract_tcga_identifier(self.slide_filename)

    @property
    def cache_filename(self) -> str:
        return (
            self.cache_folder
            + f"/{self.tcga_identifier}"
            + f"_{self.size[0]}x{self.size[1]}.pickle"
        )

    def load_cache(self) -> bool:
        try:
            with open(self.cache_filename, "rb") as fobj:
                self.image_positions = pickle.load(fobj)
                return True
        except OSError:
            return False

    def save_cache(self) -> bool:
        os.makedirs(self.cache_folder, exist_ok=True)

        with open(self.cache_filename, "wb") as fobj:
            pickle.dump(self.image_positions, fobj)

    @torch.no_grad()
    def filter_empty(self, use_cache: bool) -> None:
        """Removes all images from the dataset
        that seem contain only background
        """
        if use_cache:
            success = self.load_cache()
            if success:
                return None

        new_positions = []
        batch_size = 512

        self.open_connection()

        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            num_workers=cpu_count(),
        )
        for dl_idx, batch in enumerate(dataloader):
            bools = no_texture(batch)
            for batch_idx, value in enumerate(bools):
                if not value:
                    idx = dl_idx * batch_size + batch_idx
                    new_positions.append(self.image_positions[idx])
        self.image_positions = new_positions

        if use_cache:
            self.save_cache()

        self.close_connection()

    def open_connection(self) -> None:

        self.os_conn = OpenSlide(self.slide_filename)

    def close_connection(self) -> None:

        self.os_conn.close()
        self.os_conn = None

    def __getitem__(self, index: int):
        """
        Reads region with defined index

        The index responds to flattened index
        of image crop. Ie. `reshape` can be called
        with `dimensions` attribute to find the
        actual (x,y) subsampled image.
        """
        if self.os_conn:
            img = self.os_conn.read_region(
                self.image_positions[index], 0, self.size
            ).convert("RGB")
        else:
            img = read_wsi_region(self.slide_filename, self.image_positions[index], self.size)

        if self.return_type == "tensor":
            img = self.transform(img)

        return img

    def export(self, folderpath: str, max_size: int):
        """Exports all data points in the dataset

        Args:
            folderpath (str): Target folder.
            max_size (int): Amount of images exported.
        The files will be in form `img<index>_<location in the WSI>
        """
        os.makedirs(folderpath, exist_ok=True)

        self.open_connection()
        self.return_type = "img"
        positions = list(self.image_positions)
        random.shuffle(positions)
        for idx, img in enumerate(self):
            if idx < max_size:
                location = positions[idx]
                filename = f"{folderpath}/img{idx}_{location[0]}x{location[1]}.png"
                img.save(filename, "PNG")

        self.return_type = "tensor"
        self.close_connection()

    def __len__(self) -> int:
        """
        Returns the amount of crops
        """
        return len(self.image_positions)


def extract_tcga_identifier(filename: str) -> str:
    """Extract TCGA identifier from TCGA filename.

    Args:
        filename (str): Filename of the WSI.

    Returns:
        str: Extracted identifier
    """
    return os.path.split(filename)[-1][:12]


def extract_margin(tissue_classification, tissue_mapping, window=(5, 5)):
    margin_index = (tissue_classification == tissue_mapping["cancer"]).numpy()
    # Take the maximum filter and bitwise XOR to get margin
    margin_index = ndimage.maximum_filter(margin_index, window) ^ margin_index
    margin_index = torch.from_numpy(margin_index)
    return margin_index
