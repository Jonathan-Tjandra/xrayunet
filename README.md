# XRayUNet: Weakly Supervised Lung Nodule Detection from Synthetic X-rays

## Overview

This project explores lung nodule detection from chest X-rays without requiring real X-ray annotations, using a weakly supervised learning pipeline built entirely on synthetic projections from CT scans.

The core idea is to leverage differentiable ray-driven projection (DRR) to generate realistic X-ray-like images from CT volumes, and train a segmentation network that learns 2D detection consistent with 3D anatomical structure.

---

## Motivation

Annotated medical X-ray datasets are limited and expensive, while CT scans provide full 3D supervision.

This project bridges this gap by:
- Converting CT → synthetic X-rays using differentiable rendering (diffdrr)
- Training a U-Net on synthetic projections
- Enforcing 3D consistency through back-projection losses
- Evaluating generalization using external negative datasets (e.g., LUNA16-style negatives)

---

## Method Summary

### Synthetic X-ray Generation

We use differentiable rendering (diffdrr) to generate X-ray projections from CT volumes. Each CT can produce:
- Frontal view
- Lateral view
- Multiple randomized DRR angles

These projections form the training input distribution.

---

### Model Architecture

A standard U-Net is used:

- Input: single-view or dual-view X-ray images
- Output: pixel-wise nodule probability map
- Image-level score: max activation over segmentation map

image_score = max(sigmoid(segmentation_map))

This design ensures that even small localized nodules can trigger a strong global detection signal.

---

### Training Strategy

Training is based on three components:

(1) 2D segmentation supervision  
Synthetic masks are generated from CT projections.

(2) 3D consistency loss  
Predicted 2D masks are back-projected into 3D space and compared with CT-derived ground truth volumes.

(3) Multi-view learning  
Each CT is randomly sampled into multiple DRR views per epoch, with heavy augmentation:
- rotation / scaling
- brightness and contrast changes
- Gaussian noise
- geometric transforms (Kornia-based)

This encourages view-invariant representations.

---

### Negative Sampling

To ensure robust classification behavior, external negative samples are used:
- LUNA16-style non-nodule CT-derived DRRs
- clean synthetic projections without lesions

These are used during evaluation for ROC and F1 computation.

---

## Evaluation

The evaluation pipeline computes both detection and classification performance.

### Metrics computed:
- ROC curves (frontal, lateral, combined)
- AUC
- F1 score over threshold sweep
- Best F1 threshold
- Confusion matrix

### Scoring rule

Each image is reduced to a scalar score:

image_score = max(sigmoid(prediction_map))

This represents the strongest lesion activation in the predicted segmentation map.

---

## Outputs

Running evaluation produces:

analysis/
- roc_frontal.png
- roc_lateral.png
- roc_combined.png
- results.json

Additional logs may include activation statistics and threshold sensitivity behavior.

---

## Code Structure

training/
- train_multiview.py
- train_3d.py

analysis/
- eval.py
- metric_utils.py

data/
- generate_xrays.py
- build_3d_gt.py

inference/
- test.py

models/
- unet.py
- losses.py

utils/
- augmentation.py
- drr_utils.py

---

## Dataset Structure (Simplified)

Dual-view dataset:
- frontal/
- lateral/
- 3d_gt/

Each sample includes:
- DRR X-ray images (frontal + lateral)
- 2D segmentation masks
- CT volume path
- 3D ground truth tensor

---

## Technical Notes

- Model is segmentation-first, not classification-first
- Classification emerges from spatial activation maps
- Multi-view consistency improves robustness significantly
- 3D supervision acts as a strong regularizer against false positives
- Evaluation is threshold-based rather than probabilistic calibration

---

## Reproducibility & Research Note

This repository serves as a **research implementation** for an end-to-end pipeline. The framework integrates several complex objectives:
* **Medical Image Synthesis:** Bridging the domain gap between varied imaging modalities.
* **Weak Supervision:** Leveraging CT-derived labels for scalable training.
* **Multi-view Learning:** Optimizing feature extraction across disparate perspectives.
* **3D-Consistent Segmentation:** Maintaining spatial coherence during volumetric training.

The system is designed as an **extensible research testbed**. It prioritizes **modularity and iterative discovery**, reflecting a flexible architecture suited for active exploration rather than a static production pipeline.
