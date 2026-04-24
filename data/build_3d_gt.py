"""
Generate synthetic X-ray (DRR) dataset from CT volumes.

Modes:
  1. positive → CT + nodule mask (labelmap used)
  2. negative → CT only (no labelmap → no nodule signal)
  3. mixed    → both positive + negative samples

Output:
  images/x-ray-<id>.png
  masks/nodule-<id>.png   (only for positive samples)

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
import torch
import nibabel as nib
from glob import glob
from tqdm import tqdm


# -----------------------------
# utils
# -----------------------------
def find_mask(seg_dir):
    files = glob(os.path.join(seg_dir, "*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No mask in {seg_dir}")
    return files[0]


def load_mask(path):
    data = nib.load(path).get_fdata()
    return torch.from_numpy((data > 0).astype(np.float32))  # (D,H,W)


# -----------------------------
# main build
# -----------------------------
def build(ct_dir, mask_dir, out_dir, start, end):
    os.makedirs(out_dir, exist_ok=True)

    ct_files = sorted(glob(os.path.join(ct_dir, "ct-*.nii.gz")))
    if not ct_files:
        raise ValueError(f"No CT files in {ct_dir}")

    # extract IDs
    nums = [
        os.path.basename(f).replace("ct-", "").replace(".nii.gz", "")
        for f in ct_files
    ]

    # filter range
    nums = [n for n in nums if start <= int(n) <= (end if end else int(n))]

    print(f"Processing {len(nums)} samples...\n")

    for num in tqdm(nums):
        ct_path = os.path.join(ct_dir, f"ct-{num}.nii.gz")
        seg_dir = os.path.join(mask_dir, f"seg-{num}")

        if not os.path.exists(ct_path) or not os.path.exists(seg_dir):
            continue

        try:
            mask_path = find_mask(seg_dir)
            mask = load_mask(mask_path)

            # save
            torch.save(mask, os.path.join(out_dir, f"{num}_gt.pt"))

            with open(os.path.join(out_dir, f"{num}_ct.txt"), "w") as f:
                f.write(ct_path)

            with open(os.path.join(out_dir, f"{num}_mask.txt"), "w") as f:
                f.write(mask_path)

        except Exception as e:
            print(f"Skip {num}: {e}")

    print(f"\nDone. Output: {out_dir}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ct-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--start-num", type=int, default=1)
    parser.add_argument("--end-num", type=int, default=None)

    args = parser.parse_args()

    build(
        ct_dir=args.ct_dir,
        mask_dir=args.mask_dir,
        out_dir=args.output_dir,
        start=args.start_num,
        end=args.end_num,
    )