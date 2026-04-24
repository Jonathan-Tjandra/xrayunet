"""
data/build_3d_gt.py

Convert NSCLC nodule segmentation masks (.nii.gz) into the 3D ground truth
format expected by train_3d.py.

For each CT sample, three files are written to --output-dir:
    {num}_gt.pt     — (D, H, W) binary float32 tensor of the nodule mask
    {num}_ct.txt    — absolute path to the CT NIfTI file
    {num}_mask.txt  — absolute path to the mask NIfTI file

Usage
-----
# Create 3D GT files:
python data/build_3d_gt.py \
    --ct-dir   /data/NSCLC/CT \
    --mask-dir /data/NSCLC/Nodule_Seg \
    --output-dir /data/gt_3d

# Preview what would be created without writing files:
python data/build_3d_gt.py ... --dry-run

# Verify an existing output directory:
python data/build_3d_gt.py --output-dir /data/gt_3d --verify
"""

import os
import argparse
import random
import numpy as np
import torch
import nibabel as nib
from glob import glob
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _find_mask_file(seg_dir: str) -> str:
    """Return the path to the first .nii.gz file found in seg_dir."""
    candidates = glob(os.path.join(seg_dir, "*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No .nii.gz files found in {seg_dir}")
    return candidates[0]


def _load_mask_tensor(mask_path: str) -> torch.Tensor:
    """
    Load a NIfTI segmentation mask and return a binary (D, H, W) float tensor.
    Any non-zero label is collapsed to 1.
    """
    data = nib.load(mask_path).get_fdata()
    return torch.from_numpy((data > 0).astype(np.float32))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(
    ct_dir: str,
    mask_dir: str,
    output_dir: str,
    start_num: int = 1,
    end_num: int | None = None,
    dry_run: bool = False,
):
    """
    Build 3D GT files for all CT samples found in ct_dir.

    CT files are expected as:  ct-{num}.nii.gz
    Mask dirs are expected as: seg-{num}/  (containing one .nii.gz inside)
    """
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    ct_files = sorted(glob(os.path.join(ct_dir, "ct-*.nii.gz")))
    if not ct_files:
        raise FileNotFoundError(f"No ct-*.nii.gz files found in {ct_dir}")

    # Parse sample numbers and apply range filter
    all_nums = [os.path.basename(f).replace("ct-", "").replace(".nii.gz", "") for f in ct_files]
    nums = [n for n in all_nums if start_num <= int(n) <= (end_num or int(n) + 1)]

    print(f"Found {len(ct_files)} CT files — processing {len(nums)} samples")
    if dry_run:
        print("DRY RUN — no files will be written\n")

    created = skipped = errors = 0

    for num_str in tqdm(nums, desc="Building 3D GT"):
        ct_path  = os.path.join(ct_dir,  f"ct-{num_str}.nii.gz")
        seg_dir  = os.path.join(mask_dir, f"seg-{num_str}")

        # Validate inputs
        if not os.path.exists(ct_path):
            print(f"  SKIP {num_str}: CT not found")
            skipped += 1
            continue
        if not os.path.exists(seg_dir):
            print(f"  SKIP {num_str}: seg directory not found")
            skipped += 1
            continue
        try:
            mask_path = _find_mask_file(seg_dir)
        except FileNotFoundError as e:
            print(f"  SKIP {num_str}: {e}")
            skipped += 1
            continue

        if dry_run:
            print(f"  {num_str}: CT={ct_path}  mask={mask_path}")
            print(f"    → {output_dir}/{num_str}_gt.pt, _ct.txt, _mask.txt")
            continue

        try:
            mask_tensor = _load_mask_tensor(mask_path)
            torch.save(mask_tensor, os.path.join(output_dir, f"{num_str}_gt.pt"))
            with open(os.path.join(output_dir, f"{num_str}_ct.txt"),   "w") as f:
                f.write(ct_path)
            with open(os.path.join(output_dir, f"{num_str}_mask.txt"), "w") as f:
                f.write(mask_path)
            created += 1

            if created <= 3:
                print(f"  OK {num_str}: shape={tuple(mask_tensor.shape)}  "
                      f"nodule_voxels={mask_tensor.sum().item():.0f}")
        except Exception as e:
            print(f"  ERROR {num_str}: {e}")
            errors += 1

    print(f"\nDone — created={created}  skipped={skipped}  errors={errors}")
    if not dry_run:
        print(f"Output: {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify(output_dir: str, num_samples: int = 5):
    """
    Spot-check a random subset of the output directory to confirm all three
    files exist and are loadable.
    """
    gt_files = sorted(glob(os.path.join(output_dir, "*_gt.pt")))
    if not gt_files:
        print(f"No *_gt.pt files found in {output_dir}")
        return

    print(f"Found {len(gt_files)} samples — verifying {min(num_samples, len(gt_files))}\n")
    all_ok = True

    for gt_path in random.sample(gt_files, min(num_samples, len(gt_files))):
        num_str = os.path.basename(gt_path).replace("_gt.pt", "")
        ct_txt  = os.path.join(output_dir, f"{num_str}_ct.txt")
        msk_txt = os.path.join(output_dir, f"{num_str}_mask.txt")

        print(f"{num_str}:")

        # GT tensor
        try:
            t = torch.load(gt_path)
            print(f"  GT tensor  : {tuple(t.shape)}  {t.sum().item():.0f} voxels")
        except Exception as e:
            print(f"  GT tensor  : ERROR — {e}")
            all_ok = False

        # CT and mask paths
        for label, txt_path in [("CT path  ", ct_txt), ("Mask path", msk_txt)]:
            if not os.path.exists(txt_path):
                print(f"  {label}: MISSING txt file")
                all_ok = False
                continue
            with open(txt_path) as f:
                p = f.read().strip()
            if os.path.exists(p):
                print(f"  {label}: OK  ({p})")
            else:
                print(f"  {label}: FILE NOT FOUND  ({p})")
                all_ok = False

    print("\nAll checks passed." if all_ok else "\nSome checks failed — review output above.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 3D ground truth for consistency loss training")

    parser.add_argument("--ct-dir",    type=str,
                        default="/data/vision/polina/scratch/jthntj/NSCLC/CT")
    parser.add_argument("--mask-dir",  type=str,
                        default="/data/vision/polina/scratch/jthntj/NSCLC/Nodule_Seg")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to write {num}_gt.pt, {num}_ct.txt, {num}_mask.txt")
    parser.add_argument("--start-num", type=int, default=1)
    parser.add_argument("--end-num",   type=int, default=None)
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--verify",    action="store_true",
                        help="Verify an existing output directory instead of building")

    args = parser.parse_args()

    if args.verify:
        verify(args.output_dir)
    else:
        build(
            ct_dir=args.ct_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            start_num=args.start_num,
            end_num=args.end_num,
            dry_run=args.dry_run,
        )