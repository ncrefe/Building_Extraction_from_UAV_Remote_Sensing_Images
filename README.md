# 🛰️ Building Extraction from UAV Remote Sensing Images

This project focuses on **automatic building extraction** from a high-resolution UAV image using classical computer vision techniques. The workflow includes **segmentation, optimization-based refinement, and geometric vectorization** of building structures.

The task is designed to be solved without machine learning, relying only on traditional image processing and optimization methods.

---

## 🚀 Objective

Given a multi-band UAV image (NIR, Red, Green), the goal is to:

- Extract building regions automatically
- Refine segmentation using an energy-based optimization approach
- Convert final segmentation into vector polygons
- Achieve high overlap with ground truth (IoU evaluation)

Target performance:
- 🎯 IoU > 79%

---

## 🛠️ Tech Stack

- Python 3.12  
- NumPy 2.3.3  
- OpenCV 4.11  
- matplotlib  

> ⚠️ Only allowed course libraries are used  
> ⚠️ No machine learning methods are used

---

## 📂 Project Structure

    .
    ├── data/
    │   ├── img_mosaic.tif
    │   ├── img_mosaic_label.tif
    │   ├── img_mosaic_pred.tif
    ├── q1_segmentation.py
    ├── q2_energy_optimization.py
    ├── q3_vectorization.py
    ├── README.md

---

## ⚙️ Installation

    pip install numpy opencv-python matplotlib

---

## ▶️ Usage

Run each step separately:

    python q1_segmentation.py
    python q2_energy_optimization.py
    python q3_vectorization.py

---

## 🧪 Implemented Tasks

### 1. Initial Segmentation

- Load UAV image (NIR, Red, Green bands)
- Apply classical segmentation techniques:
  - Thresholding
  - Clustering (non-ML methods allowed)
  - Morphological operations
- Generate initial building mask similar to baseline result

---

### 2. Energy-Based Refinement

- Define a custom energy function **E**
- Optimize segmentation using iterative minimization
- Improve:
  - Boundary quality
  - Region consistency
  - Noise removal

#### 📉 Goal
- Minimization of energy → better building segmentation

---

### 3. Vectorization of Buildings

- Convert binary segmentation mask into polygons
- Extract geometric structures:
  - Straight lines
  - Right angles (≈90° constraint)
- Generate vector map representation of buildings

---

## 📊 Evaluation

- Compute **Intersection over Union (IoU)**:

\[
IoU = \frac{Prediction \cap GroundTruth}{Prediction \cup GroundTruth}
\]

- Compare predicted segmentation with:
  - `img_mosaic_label.tif`

---

## 📌 Notes

- Only classical image processing methods are used (no ML)
- Code must be clean, modular, and reproducible
- Final output must include:
  - Refined segmentation image (`img_mosaic_pred.tif`)
- Report should include visualizations similar to provided figures
- Data folder is excluded from submission due to size
