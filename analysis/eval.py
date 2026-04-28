"""
Computes:
- ROC curve (frontal, lateral, combined)
- AUC
- F1 vs threshold
- Best F1 threshold
- Confusion matrix
- Saves plots + JSON results

Scoring:
    image_score = max(sigmoid(prediction map))

Usage:

python analysis/eval.py \
    --model-path checkpoints/best.pt \
    --pos-frontal data/test/frontal_pos \
    --pos-lateral data/test/lateral_pos \
    --neg-frontal data/test/frontal_neg \
    --neg-lateral data/test/lateral_neg \
    --out-dir outputs
"""

import os
import argparse
import json
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt  # type: ignore

from models.unet import UNet
from analysis.metrics_utils import (
    confusion_matrix,
    precision_recall_f1,
    auc_trapz,
    best_threshold_by_f1,
)


# -----------------------------
# Dataset loader
# -----------------------------
class DRRFolder:
    def __init__(self, folder):
        self.files = sorted(glob(os.path.join(folder, "*.png")))
        self.tf = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def load(self, i):
        img = Image.open(self.files[i]).convert("L")
        return self.tf(img).unsqueeze(0)


# -----------------------------
# Model loader
# -----------------------------
def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt)

    in_ch = state["inc.net.0.weight"].shape[1]
    out_ch = state["outc.conv.weight"].shape[0]

    model = UNet(in_channels=in_ch, out_channels=out_ch).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# -----------------------------
# Scoring
# -----------------------------
def score_image(model, x):
    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0, 0]
        return float(pred.max().cpu().item())


def score_folder(model, folder, device):
    ds = DRRFolder(folder)
    scores = []

    for i in tqdm(range(len(ds)), desc=f"Scoring {folder}"):
        x = ds.load(i).to(device)
        scores.append(score_image(model, x))

    return np.array(scores)


# -----------------------------
# ROC
# -----------------------------
def compute_roc(pos, neg, thresholds):
    tpr, fpr = [], []

    for t in thresholds:
        TP = np.sum(pos >= t)
        FN = np.sum(pos < t)
        FP = np.sum(neg >= t)
        TN = np.sum(neg < t)

        tpr.append(TP / (TP + FN + 1e-8))
        fpr.append(FP / (FP + TN + 1e-8))

    fpr = np.array(fpr)
    tpr = np.array(tpr)

    order = np.argsort(fpr)
    return fpr[order], tpr[order]


# -----------------------------
# Plot ROC
# -----------------------------
def plot_roc(fpr, tpr, auc, title, path):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()

    plt.savefig(path)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.model_path, device)

    print("\nLoading data...")

    pos_f = score_folder(model, args.pos_frontal, device)
    pos_l = score_folder(model, args.pos_lateral, device)

    neg_f = score_folder(model, args.neg_frontal, device)
    neg_l = score_folder(model, args.neg_lateral, device)

    # combine views
    pos = np.maximum(pos_f, pos_l)
    neg = np.maximum(neg_f, neg_l)

    # sanity check
    print("\nScore sanity check:")
    print(f"pos mean: {pos.mean():.4f}")
    print(f"neg mean: {neg.mean():.4f}")

    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    y_score = np.concatenate([pos, neg])

    # better thresholds
    thresholds = np.unique(y_score)

    # -----------------------------
    # ROC
    # -----------------------------
    print("\nComputing ROC...")
    fpr, tpr = compute_roc(pos, neg, thresholds)
    auc = auc_trapz(fpr, tpr)

    # FPR @ TPR ≈ 0.9
    target_tpr = 0.9
    idx = np.argmin(np.abs(tpr - target_tpr))
    fpr_at_90_tpr = fpr[idx]

    # -----------------------------
    # F1 optimization
    # -----------------------------
    print("Computing F1...")
    best_t, best_f1 = best_threshold_by_f1(thresholds, y_true, y_score)

    y_pred = (y_score >= best_t).astype(int)

    TP, FP, TN, FN = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(TP, FP, FN)

    fpr_at_best = FP / (FP + TN + 1e-8)

    # -----------------------------
    # Print results
    # -----------------------------
    print("\n====================")
    print("FINAL RESULTS")
    print("====================")

    print(f"AUC: {auc:.4f}")
    print(f"Best threshold (F1): {best_t:.4f}")
    print(f"Best F1: {best_f1:.4f}")

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print(f"FPR @ best threshold: {fpr_at_best:.4f}")
    print(f"FPR @ TPR≈0.9: {fpr_at_90_tpr:.4f}")

    print(f"\nTP={TP}, FP={FP}, TN={TN}, FN={FN}")

    # -----------------------------
    # Save outputs
    # -----------------------------
    plot_roc(
        fpr, tpr, auc,
        "ROC Curve (Combined)",
        os.path.join(args.out_dir, "roc_combined.png")
    )

    results = {
        "auc": float(auc),
        "best_threshold": float(best_t),
        "best_f1": float(best_f1),
        "precision": float(precision),
        "recall": float(recall),
        "TP": int(TP),
        "FP": int(FP),
        "TN": int(TN),
        "FN": int(FN),
        "fpr_at_best": float(fpr_at_best),
        "fpr_at_90_tpr": float(fpr_at_90_tpr),
    }

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved outputs to:", args.out_dir)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", required=True)

    parser.add_argument("--pos-frontal", required=True)
    parser.add_argument("--pos-lateral", required=True)
    parser.add_argument("--neg-frontal", required=True)
    parser.add_argument("--neg-lateral", required=True)

    parser.add_argument("--out-dir", default="outputs")

    args = parser.parse_args()
    main(args)