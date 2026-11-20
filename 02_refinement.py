import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import gaussian, sobel
import os

np.set_printoptions(suppress=True)

# Create output directories to save results and individual snake visualizations
output_dir = "02_refinement_results"
os.makedirs(output_dir, exist_ok=True)
individual_dir = os.path.join(output_dir, "individual_snakes")
os.makedirs(individual_dir, exist_ok=True)

def manual_label(binary_img, connectivity=8):
    # Connected component labeling
    # - binary_img: 0/1 mask
    # - connectivity: 4 or 8
    # Returns:
    #     labels: same shape as binary_img, each component has a unique integer label (0=background)

    H, W = binary_img.shape
    labels = np.zeros_like(binary_img, dtype=int)
    current_label = 0

    # Define neighbor offsets based on connectivity
    if connectivity == 4:
        neighbors = [(-1, 0), (0, -1)]
    elif connectivity == 8:
        neighbors = [(-1, 0), (0, -1), (-1, -1), (-1, 1)]
    else:
        raise ValueError("Connectivity must be 4 or 8")

    # Union-Find data structure for equivalences
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        x_root = find(x)
        y_root = find(y)
        if x_root != y_root:
            parent[y_root] = x_root

    for r in range(H):
        for c in range(W):
            if binary_img[r, c] == 0:
                continue
            neighbor_labels = []
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and labels[nr, nc] > 0:
                    neighbor_labels.append(labels[nr, nc])
            if not neighbor_labels:
                current_label += 1
                labels[r, c] = current_label
                parent[current_label] = current_label
            else:
                min_label = min(neighbor_labels)
                labels[r, c] = min_label
                for lbl in neighbor_labels:
                    union(lbl, min_label)

    # Flatten equivalences
    for r in range(H):
        for c in range(W):
            if labels[r, c] > 0:
                labels[r, c] = find(labels[r, c])

    return labels

def manual_regionprops(labels):
    # Compute basic properties for each labeled component
    # Returns a list of dicts with keys: label, area, bbox (y0,x0,y1,x1), centroid (y,x)
    props_list = []
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]  # exclude background

    for lab in unique_labels:
        mask = (labels == lab)
        coords = np.argwhere(mask)  # list of (y,x) pixels
        area = coords.shape[0]
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        centroid = coords.mean(axis=0)  # (y, x)
        props_list.append({
            "label": lab,
            "area": area,
            "bbox": (y0, x0, y1, x1),
            "centroid": (centroid[0], centroid[1])
        })
    return props_list


# Function to detect building regions in a binary mask using connected components
def find_building_regions(binary_mask, min_area=100):
    # Convert input mask to binary format (0 or 1)
    # Any non-zero pixel is considered foreground
    mask = (binary_mask > 0).astype(np.uint8)

    # Perform connected component analysis to label each contiguous region
    # connectivity=8 means pixels connected diagonally are also considered connected
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Initialize list to store information about detected regions
    regions = []

    # Iterate through each connected component (skip label 0, which is background)
    for lab in range(1, num_labels):
        # Retrieve area of the current component
        area = int(stats[lab, cv2.CC_STAT_AREA])

        # Ignore small regions below the minimum area threshold to remove noise
        if area < min_area:
            continue

        # Extract bounding box information of the region
        x = int(stats[lab, cv2.CC_STAT_LEFT])  # left coordinate
        y = int(stats[lab, cv2.CC_STAT_TOP])  # top coordinate
        w = int(stats[lab, cv2.CC_STAT_WIDTH])  # width of bounding box
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])  # height of bounding box

        # Convert to standard bounding box format (y_min, x_min, y_max, x_max)
        bbox = (y, x, y + h, x + w)

        # Retrieve centroid coordinates (center of the region)
        centroid = (float(centroids[lab, 1]), float(centroids[lab, 0]))

        # Extract the binary mask corresponding only to this region
        crop = (labels[y:y + h, x:x + w] == lab).astype(np.uint8)

        # Store all relevant information for this region in a dictionary
        regions.append({
            "label": lab,  # Component label number
            "area": area,  # Pixel area of the component
            "bbox": bbox,  # Bounding box coordinates
            "centroid": centroid,  # Centroid coordinates
            "mask_crop": crop  # Binary mask for the cropped region
        })

    # Return a list of dictionaries containing all detected building regions
    return regions


# Active Contour (Snake) implementation
def simple_active_contour(image, snake, alpha=0.01, beta=0.1, gamma=0.05,
                          max_iterations=1500, convergence=0.1):
    # Compute edge map using the Sobel operator.
    # This gives an approximation of the gradient magnitude of the image.
    # Strong gradients correspond to object boundaries, and these will pull the snake points.
    edge = sobel(image)

    # Convert snake coordinates to float for continuous updates.
    snake = snake.astype(np.float64)

    # Total number of points forming the snake contour.
    n_points = len(snake)

    # Main optimization loop.
    for iteration in range(max_iterations):

        # Store previous snake to check convergence later.
        prev = snake.copy()

        # Update each point of the snake contour.
        for i in range(n_points):
            # Neighboring points (circular indexing to keep snake closed).
            prev_p = snake[(i - 1) % n_points]
            curr_p = snake[i]
            next_p = snake[(i + 1) % n_points]

            # INTERNAL ENERGY:
            # The internal energy encourages smooth and continuous contours.
            #
            # elasticity term (first derivative):
            # Keeps distances between points roughly uniform.
            # elasticity = (prev_p + next_p)/2 - curr_p
            #
            # curvature term (second derivative):
            # Penalizes sharp changes in direction (encourages smooth curves).
            # curvature = prev_p - 2*curr_p + next_p
            #
            # total_internal = alpha * elasticity + beta * curvature
            elasticity = (prev_p + next_p) / 2 - curr_p
            curvature = prev_p - 2 * curr_p + next_p
            internal = alpha * elasticity + beta * curvature

            # Keep point inside image boundaries.
            r, c = curr_p
            r = np.clip(r, 0, image.shape[0] - 1.001)
            c = np.clip(c, 0, image.shape[1] - 1.001)

            # EXTERNAL ENERGY:
            # External energy attracts the contour toward edges in the image.
            # It is derived from the gradient of the edge map.
            # Because snake points are continuous (float coordinates),
            # we interpolate the gradient using bilinear interpolation.
            r0, c0 = int(r), int(c)
            r1 = min(r0 + 1, image.shape[0] - 1)
            c1 = min(c0 + 1, image.shape[1] - 1)
            fr, fc = r - r0, c - c0

            # Approximate gradient in the row direction.
            grad_r = ((edge[r1, c0] - edge[r0, c0]) * (1 - fc)
                        + (edge[r1, c1] - edge[r0, c1]) * fc)

            # Approximate gradient in the column direction.
            grad_c = ((edge[r0, c1] - edge[r0, c0]) * (1 - fr)
                        + (edge[r1, c1] - edge[r1, c0]) * fr)

            # External force vector.
            external = np.array([grad_r, grad_c])

            # TOTAL FORCE:
            # total_force = internal + gamma * external
            total = internal + gamma * external

            # Update point position.
            snake[i] = curr_p + total

            # Ensure updated position stays inside the image.
            snake[i, 0] = np.clip(snake[i, 0], 0, image.shape[0] - 1)
            snake[i, 1] = np.clip(snake[i, 1], 0, image.shape[1] - 1)

        # Check convergence by computing average movement.
        # If the snake barely moves, stop early.
        movement = np.mean(np.sqrt(np.sum((snake - prev) ** 2, axis=1)))
        if movement < convergence:
            break

    # Return final contour after evolution.
    return snake


# Extract the largest connected component in a binary image
# Goal:
# Some bounding boxes included pieces of nearby buildings.
# These extra blobs confused the snake, causing it to lock onto the wrong structure.
# Keeping only the largest blob removes unwanted regions so the snake focuses on the intended building.
def extract_main_blob(binary_img):
    bw = (binary_img > 0).astype(np.uint8)

    labeled = manual_label(binary_img, connectivity=8)
    props = manual_regionprops(labeled)

    if len(props) == 0:
        return bw  # Return original if no blobs
    largest = max(props, key=lambda x: x['area'])
    main_mask = (labeled == largest['label']).astype(np.uint8)
    return main_mask


# Main processing pipeline
def main():
    # Load binary mask and convert to 0/1
    binary_mask = cv2.imread("binary_mask.png", cv2.IMREAD_GRAYSCALE)
    binary_mask = (binary_mask > 127).astype(np.uint8)

    # Load original RGB image for visualization
    img = cv2.imread("data/img_mosaic.tif", cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect individual building regions
    regions = find_building_regions(binary_mask, min_area=400)
    print(f"len(regions) = {len(regions)}")

    H, W = binary_mask.shape
    refined_mask = np.zeros((H, W), dtype=np.uint8)
    bbox_mask = np.zeros_like(refined_mask)
    pad = 5

    snake_outs = []
    mask_filleds = []

    for i, reg in enumerate(regions):
        y0, x0, y1, x1 = reg["bbox"]
        crop_mask = reg["mask_crop"]

        # Add padding and extract main connected component
        padded = np.pad(crop_mask, pad_width=pad, mode="constant", constant_values=0)
        padded = extract_main_blob(padded)

        # Smooth the region using Gaussian filter
        smooth = gaussian(padded.astype(float), sigma=4)
        h, w = smooth.shape

        # Initialize snake around the border of the padded region
        top = np.array([[0, c] for c in np.linspace(0, w - 1, w)])
        bottom = np.array([[h - 1, c] for c in np.linspace(0, w - 1, w)])
        left = np.array([[r, 0] for r in np.linspace(0, h - 1, h)])
        right = np.array([[r, w - 1] for r in np.linspace(0, h - 1, h)])
        init_snake = np.vstack([top, right, bottom[::-1], left[::-1]])

        # Run snake algorithm to refine contour
        snake_out = simple_active_contour(
            smooth, init_snake, alpha=0.8, beta=0.5, gamma=22.0,
            max_iterations=5000, convergence=0.08
        )
        snake_outs.append(snake_out)

        # Map snake coordinates back to original image
        snake_y = snake_out[:, 0] - pad + y0
        snake_x = snake_out[:, 1] - pad + x0

        # Create binary mask for the refined snake
        mask = np.zeros((H, W), dtype=np.uint8)
        rr, cc = np.round(snake_y).astype(int), np.round(snake_x).astype(int)
        rr = np.clip(rr, 0, H - 1)
        cc = np.clip(cc, 0, W - 1)
        mask[rr, cc] = 1
        mask_filled = cv2.fillPoly(np.zeros_like(mask), [np.vstack([cc, rr]).T.astype(np.int32)], 1)
        mask_filleds.append(mask_filled)

        # Update final refined mask and bounding box mask
        refined_mask = np.maximum(refined_mask, mask_filled)
        bbox_mask[y0:y1, x0:x1] = 1

        # Save individual snake visualization
        plt.figure(figsize=(5, 5))
        plt.imshow(mask_filled, cmap="gray")
        plt.axis("off")
        plt.title(f"Region_{i + 1}_snake")
        plt.savefig(os.path.join(individual_dir, f"Region_{i + 1}_snake.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # Save overall visualizations
        plt.figure(figsize=(6, 6))
        plt.imshow(binary_mask, cmap="gray")
        plt.axis("off")
        plt.title("Original Binary Mask")
        plt.savefig(os.path.join(output_dir, "Original_Binary_Mask.png"), dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.imshow(bbox_mask, cmap="Reds", alpha=0.4)
        plt.axis("off")
        plt.title("Bounding Boxes on Original Image")
        plt.savefig(os.path.join(output_dir, "Bounding_Boxes.png"), dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.imshow(refined_mask, cmap="gray")
        plt.axis("off")
        plt.title("Refined Segmentation (Snake)")
        plt.savefig(os.path.join(output_dir, "Refined_Segmentation.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # Save combined 3-panel figure
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(binary_mask, cmap="gray")
        plt.title("Original Binary Mask")

        plt.subplot(1, 3, 2)
        plt.imshow(img_rgb)
        plt.imshow(bbox_mask, cmap="Reds", alpha=0.4)
        plt.title("Bounding Boxes on Original Image")

        plt.subplot(1, 3, 3)
        plt.imshow(refined_mask, cmap="gray")
        plt.title("Refined Segmentation (Snake)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "02_refinement_results.png"), dpi=200)
        plt.show()

        # Save refined mask in original resolution
        refined_mask_path = os.path.join(output_dir, "Refined_Segmentation_OriginalSize.png")
        cv2.imwrite(refined_mask_path, (refined_mask * 255).astype(np.uint8))
        print(f"Refined mask (original size) saved at {refined_mask_path}")

    # Save refined mask as numpy array for further processing
    np.save(os.path.join(output_dir, "refined_mask.npy"), refined_mask)
    print(f"Refined mask saved at {os.path.join(output_dir, 'refined_mask.npy')}")


if __name__ == "__main__":
    main()
