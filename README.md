# 🛰️  UAV Building Extraction via Classical Vision Methods

This project focuses on extracting building footprints from high-resolution UAV imagery using classical computer vision techniques, without machine learning. The pipeline includes spectral analysis, segmentation, connected components, active contour refinement, and vectorization, followed by IoU evaluation.

---

## Objective

Given a multi-band UAV image (NIR, Red, Green), the goal is:

* Segment building regions automatically
* Refine segmentation using energy-based active contours
* Convert raster output into vector polygons
* Evaluate performance using IoU

Results:

* IoU (refined): 0.8814
* IoU (vectorized): 0.8676
* Target: > 0.79

---

## Tech Stack

* Python 3.12
* OpenCV 4.11
* NumPy 2.3.3
* scikit-image
* matplotlib

Only classical image processing methods are used. No machine learning.

---

## Project Structure

```
.
├── data/
│   ├── img_mosaic.tif
│   ├── img_mosaic_label.tif
│   ├── img_mosaic_pred.tif
├── segmentation.py
├── energy_optimization.py
├── vectorization.py
├── README.md
```

---

## Method Overview

### 1. Initial Segmentation

* UAV image has NIR, Red, Green bands

* Histogram analysis (16-bin) used for understanding distributions

* Buildings detected using:

  * Normalized red threshold
  * Red–green dominance rule

* Morphological operations:

  * Opening → noise removal
  * Closing → gap filling

* K-means / clustering tested but unstable (fragmentation or merging issues)

Final choice: threshold-based segmentation

---

### 2. Refinement with Connected Components + Active Contours

* 8-connectivity used to extract components
* Small regions removed by area filtering
* Each building processed independently

Bounding box strategy:

* isolates buildings
* prevents overlap interference
* reduces computation

Active contour (snake):

* minimizes energy:

  * elasticity (smoothness)
  * curvature (shape regularity)
  * edge attraction

Contour evolves toward strong gradients.

---

### 3. Vectorization

* Contours extracted from refined mask
* Simplified using Douglas–Peucker
* Angle snapping:

  * 0°, 90°, 180° enforced (±15° tolerance)
* Small polygons removed

Output:

* Clean polygon-based building map

---

## Evaluation

IoU defined as:

IoU = (Prediction ∩ GroundTruth) / (Prediction ∪ GroundTruth)

Ground truth extracted from labeled mask.

Results:

* Vectorized IoU: 0.8676
* Refined IoU: 0.8814

---

## Key Insights

* Red vs green contrast is highly informative
* Morphological filtering improves spatial consistency
* Active contours improve boundary accuracy
* Vectorization enforces geometric regularity
* Classical pipeline achieves strong performance without ML

---

## Conclusion

A full classical vision pipeline successfully extracts building footprints from UAV imagery. The combination of spectral thresholds, morphological refinement, active contours, and vectorization produces accurate and geometrically consistent results, exceeding the required IoU threshold.
