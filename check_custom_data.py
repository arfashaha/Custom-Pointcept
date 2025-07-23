import os
import numpy as np
from glob import glob

def check_colors(dataset_root, max_scenes=10):
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            continue
        print(f"\nChecking split: {split}")
        scene_dirs = sorted(glob(os.path.join(split_dir, "scene*")))[:max_scenes]
        for scene_dir in scene_dirs:
            color_path = os.path.join(scene_dir, "color.npy")
            if os.path.exists(color_path):
                color = np.load(color_path)
                print(
                    f"{os.path.basename(scene_dir)}: min={color.min()}, max={color.max()}, "
                    f"unique={np.unique(color).size}, shape={color.shape}, "
                    f"all_zero={np.all(color == 0)}"
                )
            else:
                print(f"{os.path.basename(scene_dir)}: color.npy not found")

if __name__ == "__main__":
    dataset_root = "/home/s2737104/Pointcept-main/data/scannet_custom"
    check_colors(dataset_root, max_scenes=10)
