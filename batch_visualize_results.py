import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# === CONFIGURATION ===
split = "val"  # "train", "val", or "test"
experiment_name = "semseg-spunet-v1m1-0-combine"
# experiment_name = "semseg-cac-v1m1-0-spunet-base"
# experiment_name = "semseg-octformer-v1m1-0-base"

# === PATH SETUP ===
base_dir = "/home/s2737104/Pointcept-main"
data_dir = f"{base_dir}/data/scannet_custom_102400_300/{split}"
result_dir = f"{base_dir}/exp_102400_efficient/scannet_custom/{experiment_name}/result"
output_dir = f"{base_dir}/viz_results/exp_102400_efficient/{experiment_name}"
os.makedirs(output_dir, exist_ok=True)

# === MAP PREDICTION TO COLORS ===
color_map = np.array([[0, 0, 1], [1, 0, 0]])  # 0 → blue, 1 → red

# === PROCESS EACH SCENE ===
scene_names = [d for d in os.listdir(data_dir) if d.startswith("scene") and d.endswith("_00")]

for scene_name in scene_names:
    coord_path = os.path.join(data_dir, scene_name, "coord.npy")
    color_path = os.path.join(data_dir, scene_name, "color.npy")
    pred_path = os.path.join(result_dir, f"{scene_name}_pred.npy")

    # Skip if any required file is missing
    if not (os.path.exists(coord_path) and os.path.exists(color_path) and os.path.exists(pred_path)):
        print(f"[SKIP] Missing data for {scene_name}")
        continue

    print(f"[INFO] Processing {scene_name}")
    coord = np.load(coord_path)
    color = np.load(color_path)
    pred = np.load(pred_path)

    # Normalize color if necessary
    if color.max() > 1.0:
        color = color / 255.0

    # Map prediction to RGB colors
    pred_colors = color_map[pred]

    # === Save original PLY ===
    ply_orig_path = os.path.join(output_dir, f"original_{scene_name}.ply")
    with open(ply_orig_path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(coord)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(len(coord)):
            r, g, b = (color[i] * 255).astype(int)
            f.write(f"{coord[i,0]} {coord[i,1]} {coord[i,2]} {r} {g} {b}\n")

    # === Save predicted PLY ===
    ply_pred_path = os.path.join(output_dir, f"predicted_{scene_name}.ply")
    with open(ply_pred_path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(coord)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(len(coord)):
            r, g, b = (pred_colors[i] * 255).astype(int)
            f.write(f"{coord[i,0]} {coord[i,1]} {coord[i,2]} {r} {g} {b}\n")

    # === Save side-by-side visualization ===
    viz_path = os.path.join(output_dir, f"{scene_name}_comparison.png")
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
    plt.close()

print(f"[DONE] Visualizations saved to {output_dir}")
