import os
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
output_root = "/home/s2737104/Pointcept-main/data/scannet_custom"
splits = ["train", "val", "test"]
bin_size = 10240
save_path = os.path.join(output_root, "point_count_distribution.png")

# === COLLECT POINT COUNTS ===
point_counts = []
for split in splits:
    split_dir = os.path.join(output_root, split)
    if not os.path.exists(split_dir):
        continue
    for scene in os.listdir(split_dir):
        coord_file = os.path.join(split_dir, scene, "coord.npy")
        if os.path.exists(coord_file):
            coords = np.load(coord_file)
            point_counts.append(coords.shape[0])

# === DEFINE BINS ===
if point_counts:
    max_points = max(point_counts)
    bins = list(range(0, ((max_points // bin_size) + 2) * bin_size, bin_size))
else:
    bins = [0, bin_size]

# === GENERATE HISTOGRAM ===
plt.figure(figsize=(10, 6))
plt.hist(point_counts, bins=bins, edgecolor='black')
plt.title("Distribution of Number of Points per Scene")
plt.xlabel("Number of Points")
plt.ylabel("Frequency")
plt.xticks(bins, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path)
plt.close()

# === PRINT BIN COUNTS ===
hist_counts, bin_edges = np.histogram(point_counts, bins=bins)
hist_result = {
    f"{int(bin_edges[i])}-{int(bin_edges[i+1])}": int(hist_counts[i])
    for i in range(len(hist_counts))
}
print("Bin Distribution:")
for k, v in hist_result.items():
    print(f"{k}: {v}")

print(f"\nHistogram saved to: {save_path}")
