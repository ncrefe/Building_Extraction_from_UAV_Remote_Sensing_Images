import numpy as np
import cv2
import matplotlib
import skimage
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)

# Step 0: Load Image
# Load the input image and convert from BGR to RGB. The channels correspond to NIR, RED, GREEN.
img = cv2.imread("data/img_mosaic.tif", cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

nir = img[:, :, 0].astype(np.float32)
red = img[:, :, 1].astype(np.float32)
green = img[:, :, 2].astype(np.float32)

# Step 1: Discrete Histograms for Each Channel
# Visualize pixel intensity distributions using 16 equal bins.
plt.figure(figsize=(15, 4))
bins = np.linspace(0, 255, 17)  # 16 equal bins

plt.subplot(1, 3, 1)
nir_hist, _ = np.histogram(nir, bins=bins)
plt.bar(bins[:-1], nir_hist, width=15)
plt.title("NIR Histogram (Discrete Bins)")
plt.xlabel("Value")
plt.ylabel("Count")

plt.subplot(1, 3, 2)
red_hist, _ = np.histogram(red, bins=bins)
plt.bar(bins[:-1], red_hist, width=15, color='r')
plt.title("RED Histogram (Discrete Bins)")
plt.xlabel("Value")

plt.subplot(1, 3, 3)
green_hist, _ = np.histogram(green, bins=bins)
plt.bar(bins[:-1], green_hist, width=15, color='g')
plt.title("GREEN Histogram (Discrete Bins)")
plt.xlabel("Value")

plt.tight_layout()
plt.show()

# NIR channel highlights vegetation and open areas (high values) and buildings (low values).
# Red channel can discriminate between buildings and open areas depending on intensity.
# These channels provide different cues for segmentation.

# Step 2: Initial Segmentation
# Normalize channels to [0,1] and apply simple thresholding based on red intensity.
red_n = red / 255.0
nir_n = nir / 255.0

# Select pixels with sufficiently strong red intensity (threshold 0.15)
mask_red_strong = red_n > 0.15

# Select pixels where red dominates over green by at least 15 units
mask_red_dominant = red > green + 15

# Combine both conditions to obtain initial binary mask
# Pixels satisfying both criteria are set to 1, others to 0
initial_mask = (mask_red_strong & mask_red_dominant).astype(np.uint8)

plt.figure(figsize=(8, 6))
plt.imshow(initial_mask, cmap='gray')
plt.title("Initial Mask BEFORE Morphology")
plt.axis("off")
plt.show()

# Apply morphological operations to remove small noise and close gaps.
kernel = np.ones((7, 7), np.uint8)
initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel)
initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel)

# Step 3: Display Original Image and Mask Side by Side
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image (RGB)")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Initial Segmentation Mask")
plt.imshow(initial_mask, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

print("Done.")


# Function to save images in PNG format
def save_image(img_array, filename):
    """Save a grayscale or RGB image array as a PNG file."""
    if len(img_array.shape) == 2:  # Grayscale
        cv2.imwrite(filename, img_array)
    else:  # RGB
        cv2.imwrite(filename, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))


save_image(initial_mask * 255, "binary_mask.png")


# Step 4: Function to Experiment with Different Thresholds
def explore_red_thresholds(red, green, thresholds_strong=[0.2, 0.3], thresholds_dominant=[10, 20]):
    """
    Explore different threshold combinations for red channel to refine segmentation.

    red, green: np.float32 arrays (0-255)
    thresholds_strong: list of float values (normalized 0-1)
    thresholds_dominant: list of int values (red - green difference)

    Displays a grid of masks for all threshold combinations.
    """
    red_n = red / 255.0
    n_rows = len(thresholds_strong)
    n_cols = len(thresholds_dominant)

    plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    for i, t_strong in enumerate(thresholds_strong):
        for j, t_dom in enumerate(thresholds_dominant):
            mask_strong = red_n > t_strong
            mask_dominant = red > green + t_dom
            mask = (mask_strong & mask_dominant).astype(np.uint8)

            # Apply morphology
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Red>{t_strong:.2f}, Red-Green>{t_dom}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

# Step 5: Example Call to Explore Thresholds
# explore_red_thresholds(red, green, thresholds_strong=[0.1, 0.15, 0.2, 0.25], thresholds_dominant=[10, 15, 20, 30])
