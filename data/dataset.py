"""
Two dataset classes for lung nodule detection from synthetic X-rays:

  DualViewDataset     – Loads paired frontal + lateral X-rays and their
                        2D masks, plus a 3D ground truth volume and CT path
                        for DRR-based 3D consistency training.

  MultiViewDataset    – Loads a set of DRR angle variants (feature indices)
                        for a single CT per item. Which variants are loaded
                        can be swapped each epoch via resample_features().

Expected on-disk layout
-----------------------
DualViewDataset:
  <root_frontal>/<split>/images/x_ray-<num>.png
  <root_frontal>/<split>/tensors/nodule-<num>.pt
  <root_lateral>/<split>/images/x_ray-<num>.png
  <root_lateral>/<split>/tensors/nodule-<num>.pt
  <root_3d_gt>/<num>_gt.pt          # 3D binary volume tensor
  <root_3d_gt>/<num>_ct.txt         # path to CT NIfTI
  <root_3d_gt>/<num>_mask.txt       # path to mask NIfTI

MultiViewDataset:
  <root>/<split>/images/x-ray-<num>-<feature_idx>.png
  <root>/<split>/tensors/nodule-<num>-<feature_idx>.pt
"""

import os
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def dual_view_collate(batch):
    """
    Collate for DualViewDataset. Stacks 2D tensors normally; keeps 3D GT
    volumes as a list because they can differ in spatial size across CTs.
    """
    fro_imgs, lat_imgs, fro_masks, lat_masks, vol_gts, ct_paths, mask_paths, nums = zip(*batch)
    return (
        torch.stack(fro_imgs),
        torch.stack(lat_imgs),
        torch.stack(fro_masks),
        torch.stack(lat_masks),
        list(vol_gts),          # variable size → keep as list
        list(ct_paths),
        list(mask_paths),
        list(nums),
    )


def multi_view_collate(batch):
    """
    Collate for MultiViewDataset.

    Each item is (img_dict, mask_dict, num) where the dicts map
    feature_idx → (1, H, W) tensor.

    Returns:
        imgs   : (B, num_feature, 1, H, W)
        masks  : (B, num_feature, 1, H, W)
        nums   : list of str
        feature_indices : sorted list of feature indices present
    """
    img_dicts, mask_dicts, nums = zip(*batch)
    feature_indices = sorted(img_dicts[0].keys())

    imgs  = torch.stack(
        [torch.stack([d[fi] for fi in feature_indices]) for d in img_dicts]
    )
    masks = torch.stack(
        [torch.stack([d[fi] for fi in feature_indices]) for d in mask_dicts]
    )
    return imgs, masks, list(nums), feature_indices


# ---------------------------------------------------------------------------
# DualViewDataset
# ---------------------------------------------------------------------------

class DualViewDataset(Dataset):
    """
    Loads paired frontal and lateral DRR X-rays with their 2D nodule masks
    and a 3D ground truth volume for 3D-consistency-loss training.
    """

    def __init__(self, root_frontal, root_lateral, root_3d_gt, split="train", transforms=None):
        self.f_img_dir  = os.path.join(root_frontal, split, "images")
        self.f_mask_dir = os.path.join(root_frontal, split, "tensors")
        self.l_img_dir  = os.path.join(root_lateral, split, "images")
        self.l_mask_dir = os.path.join(root_lateral, split, "tensors")
        self.root_3d_gt = root_3d_gt
        self.transforms = transforms or T.ToTensor()

        for d, label in [
            (self.f_img_dir, "frontal images"),
            (self.l_img_dir, "lateral images"),
        ]:
            if not os.path.isdir(d):
                raise ValueError(f"{label} folder not found: {d}")

        f_files = sorted(os.path.basename(p) for p in glob(os.path.join(self.f_img_dir, "x_ray-*.png")))
        l_files = sorted(os.path.basename(p) for p in glob(os.path.join(self.l_img_dir, "x_ray-*.png")))

        if not f_files:
            raise ValueError(f"No x_ray-*.png files found in {self.f_img_dir}")
        if f_files != l_files:
            raise AssertionError("Frontal and lateral image filenames do not match")

        self.img_files = f_files

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def _num_from_filename(fname):
        base = os.path.basename(fname)
        if base.startswith("x_ray-") and base.endswith(".png"):
            return base[len("x_ray-"):-len(".png")]
        return os.path.splitext(base)[0]

    def _load_mask(self, mask_dir, num):
        path = os.path.join(mask_dir, f"nodule-{num}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask not found: {path}")
        mask = torch.load(path, map_location="cpu")
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask.float()

    def _load_3d_gt(self, num):
        gt_path       = os.path.join(self.root_3d_gt, f"{num}_gt.pt")
        ct_path_file  = os.path.join(self.root_3d_gt, f"{num}_ct.txt")
        msk_path_file = os.path.join(self.root_3d_gt, f"{num}_mask.txt")

        for p, label in [
            (gt_path, "3D GT"), (ct_path_file, "CT path"), (msk_path_file, "mask path")
        ]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{label} not found: {p}")

        vol_gt = torch.load(gt_path, map_location="cpu")
        if isinstance(vol_gt, np.ndarray):
            vol_gt = torch.from_numpy(vol_gt)

        with open(ct_path_file)  as f: ct_path   = f.read().strip()
        with open(msk_path_file) as f: mask_path = f.read().strip()

        return vol_gt.float(), ct_path, mask_path

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        num   = self._num_from_filename(fname)

        fro_img = self.transforms(Image.open(os.path.join(self.f_img_dir, fname)).convert("L"))
        lat_img = self.transforms(Image.open(os.path.join(self.l_img_dir, fname)).convert("L"))
        fro_mask = self._load_mask(self.f_mask_dir, num)
        lat_mask = self._load_mask(self.l_mask_dir, num)
        vol_gt, ct_path, mask_path = self._load_3d_gt(num)

        return fro_img, lat_img, fro_mask, lat_mask, vol_gt, ct_path, mask_path, num


# ---------------------------------------------------------------------------
# MultiViewDataset
# ---------------------------------------------------------------------------

class MultiViewDataset(Dataset):
    """
    Loads multiple DRR angle/translation variants (feature indices) per CT
    sample. Which variants are active is controlled by resample_features(),
    which should be called at the start of each epoch.

    File naming: x-ray-<num>-<feature_idx>.png / nodule-<num>-<feature_idx>.pt
    """

    def __init__(self, data_root, split="train", transforms=None):
        self.img_dir  = os.path.join(data_root, split, "images")
        self.mask_dir = os.path.join(data_root, split, "tensors")
        self.transforms = transforms or T.ToTensor()

        if not os.path.isdir(self.img_dir):
            raise ValueError(f"Images folder not found: {self.img_dir}")

        all_files = sorted(glob(os.path.join(self.img_dir, "x-ray-*-*.png")))
        if not all_files:
            raise ValueError(f"No x-ray-*-*.png files found in {self.img_dir}")

        nums_set    = set()
        feature_set = set()
        for p in all_files:
            parts = os.path.basename(p).replace(".png", "").split("-")
            # Expected: ['x', 'ray', '<num>', '<feature_idx>']
            if len(parts) >= 4:
                nums_set.add(parts[2])
                feature_set.add(int(parts[3]))

        self.nums           = sorted(nums_set)
        self.total_features = max(feature_set)
        self.active_features = [1]  # updated each epoch via resample_features()

        print(f"  [{split}] {len(self.nums)} CTs, {self.total_features} feature variants")

    def resample_features(self, feature_indices: list):
        """Set which feature variants to load for the next epoch."""
        self.active_features = feature_indices

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, idx):
        num = self.nums[idx]
        img_dict  = {}
        mask_dict = {}

        for fi in self.active_features:
            stem = f"{num}-{fi}"
            img  = Image.open(os.path.join(self.img_dir, f"x-ray-{stem}.png")).convert("L")
            img  = self.transforms(img)   # (1, H, W)

            mask = torch.load(
                os.path.join(self.mask_dir, f"nodule-{stem}.pt"), map_location="cpu"
            )
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            img_dict[fi]  = img
            mask_dict[fi] = mask.float()

        return img_dict, mask_dict, num