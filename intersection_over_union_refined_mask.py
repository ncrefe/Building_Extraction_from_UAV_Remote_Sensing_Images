import numpy as np
import cv2
import matplotlib.pyplot as plt

# File paths
GT_PATH = "data/img_mosaic_label.tif"
PRED_PATH = "02_refinement_results/Refined_Segmentation_OriginalSize.png"

# Read ground-truth label (RGB)
gt_color = cv2.imread(GT_PATH, cv2.IMREAD_COLOR)
gt_color = cv2.cvtColor(gt_color, cv2.COLOR_BGR2RGB)
print("GT color shape:", gt_color.shape)

# Extract blue mask as ground-truth (pure blue: R=0, G=0, B=255)
blue_mask = (gt_color[:, :, 0] == 0) & \
            (gt_color[:, :, 1] == 0) & \
            (gt_color[:, :, 2] == 255)

gt_bin = blue_mask.astype(np.uint8)
print("GT mask unique values:", np.unique(gt_bin))

# Read predicted mask (grayscale → binary)
pred = cv2.imread(PRED_PATH, cv2.IMREAD_GRAYSCALE)
_, pred_bin = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)
print("Prediction shape:", pred_bin.shape)

# Resize prediction if dimensions do not match
if pred_bin.shape != gt_bin.shape:
    print("Prediction resized to match GT dimensions")
    pred_bin = cv2.resize(
        pred_bin,
        (gt_bin.shape[1], gt_bin.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

# Compute Intersection over Union (IoU)
intersection = np.logical_and(gt_bin == 1, pred_bin == 1).sum()
union = np.logical_or(gt_bin == 1, pred_bin == 1).sum()
iou_score = intersection / union if union > 0 else 0.0

print(" IoU Score =", round(iou_score, 4))

# Visualization of GT, prediction, and difference map
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gt_bin, cmap="gray")
plt.title("Ground Truth (Blue Mask)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pred_bin, cmap="gray")
plt.title("Prediction")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gt_bin - pred_bin, cmap="bwr")
plt.title("Difference (GT - Prediction)")
plt.axis("off")

plt.tight_layout()
plt.show()
