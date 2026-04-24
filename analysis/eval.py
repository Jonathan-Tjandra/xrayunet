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

This matches training pipeline assumption:
UNet -> segmentation map -> lesion presence score

Basic usage:

python analyze_drr.py \
    --model-path checkpoints/best.pt \
    --pos-frontal data/test/frontal_pos \
    --pos-lateral data/test/lateral_pos \
    --neg-frontal data/test/frontal_neg \
    --neg-lateral data/test/lateral_neg
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
import matplotlib.pyplot as plt # type: ignore

from models.unet import UNet
from analysis.metrics_utils import (
    confusion_matrix,
    precision_recall_f1,
    auc_trapz,
    best_threshold_by_f1,
)


# -----------------------------
# Dataset loader (simple)
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
# model
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
# scoring function
# -----------------------------
def score_image(model, x):
    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0, 0]
        return float(pred.max().cpu().item())


def score_folder(model, folder, device):
    scores = []
    ds = DRRFolder(folder)

    for i in tqdm(range(len(ds))):
        x = ds.load(i).to(device)
        scores.append(score_image(model, x))

    return np.array(scores)


# -----------------------------
# ROC computation
# -----------------------------
def compute_roc(pos, neg, thresholds):
    tpr, fpr = [], []

    for t in thresholds:
        TP = np.sum(pos > t)
        FN = np.sum(pos <= t)
        FP = np.sum(neg > t)
        TN = np.sum(neg <= t)

        tpr.append(TP / (TP + FN + 1e-8))
        fpr.append(FP / (FP + TN + 1e-8))

    return np.array(fpr), np.array(tpr)


# -----------------------------
# plot
# -----------------------------
def plot_roc(fpr, tpr, title, out_path):
    plt.figure()
    plt.plot(fpr, tpr, label=title)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# main
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)

    print("\nLoading data...")

    pos_f = score_folder(model, args.pos_frontal, device)
    pos_l = score_folder(model, args.pos_lateral, device)

    neg_f = score_folder(model, args.neg_frontal, device)
    neg_l = score_folder(model, args.neg_lateral, device)

    # combine views (max pooling across views)
    pos = np.maximum(pos_f, pos_l)
    neg = np.maximum(neg_f, neg_l)

    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    y_score = np.concatenate([pos, neg])

    thresholds = np.linspace(0, 1, 200)

    print("\nComputing ROC...")
    fpr, tpr = compute_roc(pos, neg, thresholds)
    auc = auc_trapz(fpr, tpr)

    print("Computing F1...")
    best_t, best_f1 = best_threshold_by_f1(thresholds, y_true, y_score)

    y_pred = (y_score > best_t).astype(int)

    TP, FP, TN, FN = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(TP, FP, FN)

    print("\n====================")
    print("FINAL RESULTS")
    print("====================")
    print(f"AUC: {auc:.4f}")
    print(f"Best threshold: {best_t:.4f}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nConfusion Matrix:")
    print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")

    # save ROC plot
    os.makedirs("outputs", exist_ok=True)
    plot_roc(fpr, tpr, "ROC Curve", "outputs/roc_combined.png")

    # save JSON
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
    }

    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)


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

    args = parser.parse_args()
    main(args)