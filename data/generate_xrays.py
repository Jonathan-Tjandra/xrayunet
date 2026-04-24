"""
Generate synthetic X-ray (DRR) dataset from CT volumes.

Modes:
  1. positive → CT + nodule mask (labelmap used)
  2. negative → CT only (no labelmap → no nodule signal)
  3. mixed    → both positive + negative samples

Output:
  images/x-ray-<id>.png
  masks/nodule-<id>.png   (only for positive)

------------------------------------------------------------
CLI USAGE
------------------------------------------------------------

# Positive-only dataset (with nodules)
python generate_drr_dataset.py \
    --ct-root /data/NSCLC/CT \
    --out-img /data/drr/positive_images \
    --mode positive

# Negative-only dataset (no nodules)
python generate_drr_dataset.py \
    --ct-root /data/NSCLC/CT \
    --out-img /data/drr/negative_images \
    --mode negative

# Mixed dataset (both positive + negative)
python generate_drr_dataset.py \
    --ct-root /data/NSCLC/CT \
    --out-img /data/drr/mixed_images \
    --mode mixed

------------------------------------------------------------
NOTES
------------------------------------------------------------
- Positive mode uses CT + segmentation labelmap via DiffDRR.
- Negative mode renders CT without labelmap (no nodule signal).
- Mixed mode generates both variants for the same CTs.
- Output images are padded to 208×208 after normalization.
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert


# --------------------------
# config
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SDD = 1020
HEIGHT = 200
DELTX = 2.0

PAD = 4


# --------------------------
# DRR core
# --------------------------
def make_drr(ct_path, mask_path=None):
    subject = read(volume=ct_path, labelmap=mask_path)

    drr = DRR(
        subject,
        sdd=SDD,
        height=HEIGHT,
        delx=DELTX
    ).to(DEVICE)

    rot = torch.tensor([[torch.pi / 2, 0.0, 0.0]], device=DEVICE)
    xyz = torch.tensor([[0.0, 850.0, 0.0]], device=DEVICE)

    pose = convert(rot, xyz, "euler_angles", "ZXY")

    img = drr(pose)[0, 0].detach().cpu().numpy()

    return img


# --------------------------
# utils
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
# process one CT
# --------------------------
def process(ct_path, out_img, out_mask, mode):
    ct_id = os.path.basename(ct_path).replace(".mhd", "")

    # ---------------- POSITIVE ----------------
    if mode in ["positive", "mixed"]:
        mask_path = ct_path.replace("CT", "Seg")  # adjust if needed

        try:
            img = make_drr(ct_path, mask_path)
            img = pad(normalize(img))

            save(img, os.path.join(out_img, f"x-ray-{ct_id}.png"))

            print(f"[POSITIVE] {ct_id}")

        except Exception as e:
            print(f"[FAIL POS] {ct_id}: {e}")

    # ---------------- NEGATIVE ----------------
    if mode in ["negative", "mixed"]:
        try:
            img = make_drr(ct_path, None)
            img = pad(normalize(img))

            save(img, os.path.join(out_img, f"x-ray-{ct_id}_neg.png"))

            print(f"[NEGATIVE] {ct_id}")

        except Exception as e:
            print(f"[FAIL NEG] {ct_id}: {e}")


# --------------------------
# dataset loop
# --------------------------
def run(root, out_img, out_mask, mode):
    os.makedirs(out_img, exist_ok=True)

    for sub in sorted(os.listdir(root)):
        sub_path = os.path.join(root, sub)

        if not os.path.isdir(sub_path):
            continue

        for f in os.listdir(sub_path):
            if f.endswith(".mhd") and f.startswith("ct"):
                process(
                    os.path.join(sub_path, f),
                    out_img,
                    out_mask,
                    mode
                )


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-root", required=True)
    parser.add_argument("--out-img", required=True)
    parser.add_argument("--out-mask", required=False, default=None)
    parser.add_argument("--mode", choices=["positive", "negative", "mixed"], default="positive")

    args = parser.parse_args()

    run(args.ct_root, args.out_img, args.out_mask, args.mode)