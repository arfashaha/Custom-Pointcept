import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# === CONFIGURATION ===
sample_number = "0001"  # Accepts leading zeros, e.g., "0459", "19200"
split = "val"           # "train", "val", or "test"
experiment_name = "semseg-spunet-v1m1-0-base"
# experiment_name = "semseg-octformer-v1m1-0-base"

# === PATH SETUP ===
scene_name = f"scene{sample_number}_00"
base_dir = "/home/s2737104/Pointcept-main"
data_dir = f"{base_dir}/data/scannet_custom/{split}/{scene_name}"
result_dir = f"{base_dir}/exp/scannet_custom/{experiment_name}/result"
output_dir = f"{base_dir}/viz_results/{split}"
os.makedirs(output_dir, exist_ok=True)

coord_path = os.path.join(data_dir, "coord.npy")
color_path = os.path.join(data_dir, "color.npy")
pred_path = os.path.join(result_dir, f"{scene_name}_pred.npy")
ply_orig_path = os.path.join(output_dir, f"original_{scene_name}.ply")
ply_pred_path = os.path.join(output_dir, f"predicted_{scene_name}.ply")
viz_path = os.path.join(output_dir, f"{scene_name}_clean_comparison.png")

# === LOAD DATA ===
coord = np.load(coord_path)
color = np.load(color_path)
pred = np.load(pred_path)

# Normalize color if necessary
if color.max() > 1.0:
    color = color / 255.0

# === MAP PREDICTION TO COLORS ===
color_map = np.array([[0, 0, 1], [1, 0, 0]])  # 0 → blue, 1 → red
pred_colors = color_map[pred]

# === SAVE ORIGINAL PLY ===
with open(ply_orig_path, "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(coord)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for i in range(len(coord)):
        r, g, b = (color[i] * 255).astype(int)
        f.write(f"{coord[i,0]} {coord[i,1]} {coord[i,2]} {r} {g} {b}\n")

# === SAVE PREDICTED PLY ===
with open(ply_pred_path, "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(coord)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for i in range(len(coord)):
        r, g, b = (pred_colors[i] * 255).astype(int)
        f.write(f"{coord[i,0]} {coord[i,1]} {coord[i,2]} {r} {g} {b}\n")

# === VISUALIZATION ===
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=color, s=2)
ax1.set_title("Original Point Cloud")
ax1.axis("off")
ax1.set_box_aspect([1, 1, 1])

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=pred_colors, s=2)
ax2.set_title("Model Prediction")
ax2.axis("off")
ax2.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
plt.show()
