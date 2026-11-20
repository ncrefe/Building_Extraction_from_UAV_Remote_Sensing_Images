import numpy as np
import cv2
import matplotlib.pyplot as plt

# File paths
GT_PATH = "data/img_mosaic_label.tif"  # blue ground-truth areas
VEC_PATH = "vectorized_map.png"  # boundary-only vector map

# Load ground-truth and extract blue mask
gt_color = cv2.imread(GT_PATH, cv2.IMREAD_COLOR)
gt_color = cv2.cvtColor(gt_color, cv2.COLOR_BGR2RGB)

blue_mask = (gt_color[:, :, 0] == 0) & \
            (gt_color[:, :, 1] == 0) & \
            (gt_color[:, :, 2] == 255)

gt_bin = blue_mask.astype(np.uint8)

# Load the vectorized boundary map
vec = cv2.imread(VEC_PATH, cv2.IMREAD_GRAYSCALE)

# Convert vector map to a binary image
_, vec_bw = cv2.threshold(vec, 127, 255, cv2.THRESH_BINARY)

# Find polygon contours
contours, _ = cv2.findContours(vec_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fill polygons to obtain the predicted mask
pred_mask = np.zeros_like(vec_bw)

for cnt in contours:
    cv2.fillPoly(pred_mask, [cnt], 255)

pred_bin = (pred_mask > 127).astype(np.uint8)

# Compute IoU
intersection = np.logical_and(gt_bin == 1, pred_bin == 1).sum()
union = np.logical_or(gt_bin == 1, pred_bin == 1).sum()

iou_score = intersection / union if union > 0 else 0.0

print(" IoU Score (Vectorized Filled) =", round(iou_score, 4))

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gt_bin, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pred_bin, cmap="gray")
plt.title("Prediction (Filled Polygons)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gt_bin - pred_bin, cmap="bwr")
plt.title("Difference (GT - Prediction)")
plt.axis("off")

plt.tight_layout()
plt.show()
