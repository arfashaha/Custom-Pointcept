# import torch
# import torch_geometric
# import torch_points3d
# import open3d
# import wandb
# import torchvision
# import torchaudio
# print(torch.__version__)
# print(torch_geometric.__version__)
# print(torch_points3d.__version__)
# print(open3d.__version__)
# print(wandb.__version__)
# print(torchvision.__version__)
# print(torchaudio.__version__)

# import resource
# soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# print(f"Open file limits: soft={soft}, hard={hard}")

# import os
# import sys
# import glob

# def summarize_dir(root_dir, max_examples=5):
#     print(f"Inspecting: {root_dir}")
#     all_files = []
#     ext_count = {}
#     sizes = []
#     for dirpath, _, filenames in os.walk(root_dir):
#         for filename in filenames:
#             fullpath = os.path.join(dirpath, filename)
#             all_files.append(fullpath)
#             ext = os.path.splitext(filename)[-1].lower()
#             ext_count[ext] = ext_count.get(ext, 0) + 1
#             try:
#                 sizes.append(os.path.getsize(fullpath))
#             except Exception as e:
#                 print(f"Could not get size for {fullpath}: {e}")
    
#     print(f"\nTotal files: {len(all_files)}")
#     print("\nFile extension breakdown:")
#     for ext, cnt in ext_count.items():
#         print(f"  {ext or '[no ext]'}: {cnt}")
#     print("\nFirst {} files:".format(max_examples))
#     for f in all_files[:max_examples]:
#         print("  ", f)
#     print("\nSample file sizes (bytes):")
#     for sz in sizes[:max_examples]:
#         print("  ", sz)
#     if sizes:
#         print(f"\nMin file size: {min(sizes)}")
#         print(f"Max file size: {max(sizes)}")
#         print(f"Mean file size: {sum(sizes)//len(sizes)}")
#     print("\nTop-level dirs:")
#     for item in os.listdir(root_dir):
#         if os.path.isdir(os.path.join(root_dir, item)):
#             print("  DIR:", item)
#         else:
#             print("  FILE:", item)

#     # Optional: Try to open a sample of files (to see if they can be opened/closed)
#     print("\nSample open/close test for first few files:")
#     for f in all_files[:min(3, len(all_files))]:
#         try:
#             with open(f, 'rb') as fin:
#                 fin.read(10)  # Try to read a few bytes
#             print(f"  [OK] {f}")
#         except Exception as e:
#             print(f"  [ERR] {f}: {e}")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python inspect_dataset.py <test_set_directory>")
#         sys.exit(1)
#     summarize_dir(sys.argv[1])


import spconv.pytorch as spconv
import torch

# print("spconv version:", spconv.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0))

# features = torch.rand(10, 4).cuda()
# indices = torch.randint(0, 20, (10, 4)).int().cuda()
# x = spconv.SparseConvTensor(
#     features=features,
#     indices=indices,
#     spatial_shape=[20, 20, 20],
#     batch_size=1
# )
# conv = spconv.SubMConv3d(4, 8, 3, padding=1, indice_key='subm').cuda()
# out = conv(x)
# print("spconv basic SubMConv3d ran successfully!")

import torch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import scatter

device = torch.device('cuda')
pos = torch.rand((100, 3), device=device)
batch = torch.zeros(100, dtype=torch.long, device=device)
voxel = voxel_grid(pos, size=0.1, batch=batch)
_, inverse = torch.unique(voxel, return_inverse=True)
val = torch.rand(100, device=device)
reduced = scatter(val, inverse, reduce="mean")
print("torch_geometric voxel_grid and scatter ran successfully!")
