import numpy as np
import cv2
import skimage
import scipy
import time
import matplotlib.pyplot as plt
import os

np.set_printoptions(suppress=True)

# Settings
INPUT_PATH = "02_refinement_results/Refined_Segmentation_OriginalSize.png"
MIN_AREA = 100
APPROX_EPS_RATIO = 0.01
ANGLE_TOL_DEG = 15
SNAP_ENABLED = True


# Utility Functions

def load_binary(path):
    # Load as grayscale and convert to a binary mask (0 or 1)
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, bw = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
    return (bw // 255).astype(np.uint8)


def find_contours_from_binary(binary):
    # Find continuous boundary curves in the binary mask
    bw8 = (binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(bw8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def approx_contour(cnt, eps_ratio=APPROX_EPS_RATIO):
    # Polygonal approximation
    perim = cv2.arcLength(cnt, True)
    eps = max(1.0, eps_ratio * perim)
    approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
    return approx.reshape(-1, 2)


def edge_angle(p0, p1):
    # Compute angle of an edge segment in degrees (0–180 range)
    dy = float(p1[1] - p0[1])
    dx = float(p1[0] - p0[0])
    ang = np.degrees(np.arctan2(dy, dx))
    if ang < 0:
        ang += 180
    return ang


def snap_polygon_to_axes(poly, angle_tol_deg=15):
    """
    Snap polygon edges to horizontal or vertical orientation
    if they are close to 0, 90, or 180 degrees.
    This regularizes building outlines.
    """
    poly = poly.copy().astype(float)
    n = len(poly)
    new_poly = poly.copy()

    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        ang = edge_angle(p0, p1)

        # Horizontal snapping (near 0 or 180 degrees)
        if min(abs(ang - 0), abs(ang - 180)) <= angle_tol_deg:
            y = int(round((p0[1] + p1[1]) / 2))
            new_poly[i, 1] = new_poly[(i + 1) % n, 1] = y

        # Vertical snapping (near 90 degrees)
        elif abs(ang - 90) <= angle_tol_deg:
            x = int(round((p0[0] + p1[0]) / 2))
            new_poly[i, 0] = new_poly[(i + 1) % n, 0] = x

    return np.round(new_poly).astype(int)


def polygon_area(poly):
    # Compute area via the standard shoelace formula
    if len(poly) < 3:
        return 0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def vectorize_refined(binary):
    """
    Convert refined binary segmentation into vector polygons:
    1) Find contours
    2) Approximate them
    3) Snap edges to horizontal/vertical
    4) Filter by area
    """
    contours = find_contours_from_binary(binary)
    polys = []

    for cnt in contours:
        # Skip small noisy components
        if cv2.contourArea(cnt) < MIN_AREA:
            continue

        # Polygon approximation
        approx = approx_contour(cnt)

        # axis-alignment
        if SNAP_ENABLED:
            approx = snap_polygon_to_axes(approx)

        # Reject degenerate polygons
        if polygon_area(approx) < MIN_AREA:
            continue

        polys.append(approx)

    return polys


#  Main Vectorization

def main_vectorize():
    bw = load_binary(INPUT_PATH)
    h, w = bw.shape

    # Convert refined segmentation to polygons
    polygons = vectorize_refined(bw)

    # 1) Create a blank canvas and draw polygons as white lines
    vector_canvas = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = poly.reshape((-1, 1, 2))
        cv2.polylines(vector_canvas, [pts], True, 255, 2)

    save_path = "vectorized_map.png"
    cv2.imwrite(save_path, vector_canvas)
    print("Vectorized map saved to:", save_path)

    # 2) Draw polygons over the original image if available
    if os.path.exists("data/img_mosaic.tif"):
        orig = cv2.imread("data/img_mosaic.tif")
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        if orig.shape[:2] != (h, w):
            orig = cv2.resize(orig, (w, h))
    else:
        orig = np.dstack([bw * 255] * 3)

    orig_overlay = orig.copy()
    for poly in polygons:
        pts = poly.reshape((-1, 1, 2))
        cv2.polylines(orig_overlay, [pts], True, (255, 0, 0), 3)

    # 3) Draw polygons over the refined segmentation mask
    refined = (bw * 255).astype(np.uint8)
    refined_rgb = np.dstack([refined] * 3)

    for poly in polygons:
        pts = poly.reshape((-1, 1, 2))
        cv2.polylines(refined_rgb, [pts], True, (255, 0, 0), 2)

    # 4) Display three results side by side
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(vector_canvas, cmap="gray")
    plt.title("Vector-Only Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(orig_overlay)
    plt.title("Vectors on Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(refined_rgb)
    plt.title("Vectors on Refined Segmentation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_vectorize()
