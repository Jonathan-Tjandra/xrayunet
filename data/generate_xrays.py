"""
Generate synthetic X-ray (DRR) dataset from CT volumes.

Modes:
  1. positive → CT + nodule mask (supervised signal)
  2. negative → CT only (no nodule signal)

Output:
  images/x-ray-<id>.png
  masks/nodule-<id>.png   (only for positive)

CLI:
  python generate_drr_dataset.py --ct-root ... --out-img ... --out-mask ... --mode positive
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert


# --------------------------
# CONFIG
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SDD = 1020
HEIGHT = 200
DELTX = 2.0
PAD = 4


# --------------------------
# DRR GENERATION
# --------------------------
def make_drr(ct_path, mask_path=None):
    subject = read(volume=ct_path, labelmap=mask_path)
    drr = DRR(subject, sdd=SDD, height=HEIGHT, delx=DELTX).to(DEVICE)

    rot = torch.tensor([[torch.pi / 2, 0.0, 0.0]], device=DEVICE)
    xyz = torch.tensor([[0.0, 850.0, 0.0]], device=DEVICE)

    pose = convert(rot, xyz, "euler_angles", "ZXY")
    img = drr(pose)[0, 0].detach().cpu().numpy()

    return img


# --------------------------
# UTIL
# --------------------------
def normalize(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return (img * 255).astype(np.uint8)


def pad(img):
    return np.pad(img, ((PAD, PAD), (PAD, PAD)), mode="constant")


def save(img, path):
    Image.fromarray(img).save(path)


# --------------------------
# PROCESS SINGLE CT
# --------------------------
def process(ct_path, out_img, out_mask, mode):
    ct_id = os.path.basename(ct_path).replace(".mhd", "")

    # ---------------- POSITIVE ----------------
    if mode == "positive":
        mask_path = ct_path.replace("CT", "Seg")  # adjust if needed

        try:
            img = make_drr(ct_path, mask_path)
            img = pad(normalize(img))

            save(img, os.path.join(out_img, f"x-ray-{ct_id}.png"))

            # optional mask export
            if out_mask is not None:
                mask = make_drr(ct_path, mask_path)
                mask = (mask > 0).astype(np.uint8)
                mask = pad(mask)
                save(mask * 255, os.path.join(out_mask, f"nodule-{ct_id}.png"))

            print(f"[POSITIVE] {ct_id}")

        except Exception as e:
            print(f"[FAIL POS] {ct_id}: {e}")

    # ---------------- NEGATIVE ----------------
    elif mode == "negative":
        try:
            img = make_drr(ct_path, None)
            img = pad(normalize(img))

            save(img, os.path.join(out_img, f"x-ray-{ct_id}.png"))

            print(f"[NEGATIVE] {ct_id}")

        except Exception as e:
            print(f"[FAIL NEG] {ct_id}: {e}")


# --------------------------
# DATASET LOOP
# --------------------------
def run(root, out_img, out_mask, mode):
    os.makedirs(out_img, exist_ok=True)
    if out_mask is not None:
        os.makedirs(out_mask, exist_ok=True)

    for sub in sorted(os.listdir(root)):
        sub_path = os.path.join(root, sub)

        if not os.path.isdir(sub_path):
            continue

        for f in os.listdir(sub_path):
            if f.endswith(".mhd") and f.startswith("ct"):
                process(os.path.join(sub_path, f), out_img, out_mask, mode)


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-root", required=True)
    parser.add_argument("--out-img", required=True)
    parser.add_argument("--out-mask", default=None)
    parser.add_argument("--mode", choices=["positive", "negative"], required=True)

    args = parser.parse_args()

    run(args.ct_root, args.out_img, args.out_mask, args.mode)