"""
Custom Two-Class Point Cloud Dataset (Alien/Target) for Pointcept

- Compatible with ScanNet-processed format loaders in Pointcept.
- Expects each scene folder (e.g., scene0000_00) to contain:
    - coord.npy       (float32, [N,3])
    - color.npy       (uint8, [N,3])
    - normal.npy      (float32, [N,3])  (can be dummy zeros)
    - instance.npy    (int32, [N,])     (use zeros if not available)
    - segment20.npy   (int16, [N,])     (0 = alien, 1 = target)
- For test split, only coord/color/normal is required.

Author: [Your Name]
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset

@DATASETS.register_module()
class CustomTwoClassDataset(DefaultDataset):
    """
    Custom dataset for 3D binary-class segmentation (alien/target).
    Labels:
        segment20: 0 = alien, 1 = target
        instance:  always 0 (if not available)
    """

    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
    ]
    class2id = np.array([0, 1])  # 0 = alien, 1 = target

    def __init__(self, lr_file=None, la_file=None, **kwargs):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"custom-two-class-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))

        data_dict["name"] = name
        data_dict["split"] = split

        # Enforce correct dtype and shape
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)

        # Handle segmentation labels
        if "segment20" in data_dict.keys():
            # 0 = alien, 1 = target
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        else:
            # For test set (no segment20): fill with -1
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        # Handle instance labels
        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                np.zeros(data_dict["coord"].shape[0], dtype=np.int32)
            )

        # If using la file (active learning), mask out unselected
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index

        return data_dict

# If you want a "test" dataset that loads from the test folder (with only coord/color/normal):
@DATASETS.register_module()
class CustomTwoClassTestDataset(CustomTwoClassDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal"
    ]
    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"custom-two-class-test-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))

        data_dict["name"] = name
        data_dict["split"] = split
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)
        data_dict["segment"] = np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
        data_dict["instance"] = np.zeros(data_dict["coord"].shape[0], dtype=np.int32)
        return data_dict
