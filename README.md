# Improving Chest X-ray Nodule Segmentation via CT-Derived Multi-View Projections

📄 **[Read the full paper here](./x-ray-nodule-detection.pdf)**

## Overview
Detecting lung nodules in chest X-ray images is fundamentally challenging due to overlapping anatomical structures and the scarcity of pixel-level annotations. While Computed Tomography (CT) provides precise 3D nodule labels, it involves higher radiation exposure and is less accessible. 

This project bridges that gap by introducing a weakly supervised framework that generates synthetic chest X-ray images (DRRs) and corresponding segmentation masks directly from annotated CT volumes. By leveraging differentiable rendering, multi-view projections, and randomized online augmentations, this pipeline transfers 3D anatomical knowledge into a robust 2D detection system.

---

## 🔬 Research Note & Environment

This repository presents a consolidated implementation of a deep learning pipeline developed in a high-performance computing (HPC) research environment.

It is intended primarily as a **methodological and architectural reference**, highlighting key components including a U-Net segmentation model, synthetic DRR generation from CT volumes, multi-view projections, a 3D back-projection consistency loss, and geometric and photometric augmentation for robustness across imaging variations.

The codebase has been refactored from its original form into a more modular structure to improve clarity and presentation. As a result, some components differ from the original execution setup, particularly in data handling, file organization, and preprocessing. It is best understood as a reference implementation for training and evaluation rather than a fully integrated end-to-end pipeline.

Reproduction of results requires the NSCLC Radiogenomics (positive) and LIDC-IDRI (consensus-negative) datasets.

## Methodology

### 1. Synthetic X-ray Generation (DiffDRR)
To overcome the lack of annotated 2D radiographs, we utilize the `DiffDRR` rendering library to simulate X-ray formation from CT volumes. Each CT generates:
* Frontal and lateral views.
* Multiple randomized DRR angles to expose the model to diverse viewpoints.
* Paired 2D binary segmentation masks derived from the same ray-tracing procedure applied to 3D nodule annotations.

### 2. Network Architecture
The pipeline employs a **U-Net** architecture to predict pixel-wise nodule probability maps.
* **Input:** Single-view or multi-view synthetic X-ray projections.
* **Output:** Pixel-wise probability map.
* **Image-Level Scoring:** Maximum activation over the segmentation map to capture small nodules.

### 3. Training & Augmentation Strategy
Training combines **Dice Loss** and **Binary Cross-Entropy Loss**. To address limited data:
* **Enhanced Online Augmentation:** Using `Kornia` for dynamic geometric and photometric transformations.
* **Volumetric Consistency:** Optional 3D back-projection loss to enforce agreement across projections.

---

## Key Findings

Multi-view projections combined with strong augmentation significantly improve detection performance, particularly for anatomically ambiguous lateral X-rays. Increasing projection diversity proves more effective and computationally efficient than enforcing strict 3D reconstruction constraints.

For full quantitative results, ROC analyses, and qualitative comparisons across nodule sizes, see the [full paper](./x-ray-nodule-detection.pdf).

---

## Code Structure

* `training/` – Training loops, multi-view logic, and 3D consistency
* `analysis/` – ROC, F1 sweeps, and component-level evaluation
* `data/` – DRR generation and dataset construction
* `inference/` – Prediction scripts
* `models/` – U-Net and loss functions
* `utils/` – Augmentation and DiffDRR utilities