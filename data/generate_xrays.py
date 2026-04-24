"""
Generate synthetic X-ray images (DRRs) from CT volumes using DiffDRR.

Two modes:
  1. Single view (AP/lateral) with optional mask overlay — for building the
     dual-view dataset used by train_3d.py.
  2. Multi-angle augmented views — for building the multi-view dataset used
     by train_multiview.py.

Typical usage
-------------
# Generate single AP view + mask overlay for all CTs:
python data/generate_xrays.py single \
    --ct-dir    /data/NSCLC/CT \
    --mask-dir  /data/NSCLC/Combined_Seg \
    --out-img   /data/x_rays/frontal \
    --out-overlay /data/x_rays/frontal_overlay

# Generate 40 augmented angle variants per CT:
python data/generate_xrays.py multiview \
    --ct-dir      /data/NSCLC/CT \
    --seg-root    /data/NSCLC/Nodule_Seg \
    --source-root /data/data_padded_distributed \
    --output-root /data/data_padded_distributed_aug
"""

import argparse
import re
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert, RigidTransform, euler_angles_to_matrix, make_matrix


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COLORS = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 165, 0),    # orange
    (255, 105, 180),  # pink
    (128, 0, 128),    # purple
    (0, 255, 255),    # cyan
]

# DRR renderer settings shared across all modes
DRR_SDD   = 1020    # source-to-detector distance (mm)
DRR_H     = 200     # detector height (pixels)
DRR_DELX  = 2.0     # pixel spacing (mm)

# Padding applied after rendering to reach 208×208
PAD_EACH  = 4       # 200 + 4 + 4 = 208


# ---------------------------------------------------------------------------
# DRR helpers
# ---------------------------------------------------------------------------

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_drr(ct_path: str, mask_path: str | None, device: torch.device) -> DRR:
    subject = read(volume=ct_path, labelmap=mask_path)
    return DRR(subject, sdd=DRR_SDD, height=DRR_H, delx=DRR_DELX).to(device)


def _ap_pose(device: torch.device):
    """Return a standard AP (frontal) pose."""
    rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    xyz = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Normalise a 2D float tensor to a uint8 numpy array."""
    arr = t.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr * 255).astype(np.uint8)


def _find_nodule_channel(img_masked: torch.Tensor) -> int:
    """Return the first non-background channel that contains nonzero pixels."""
    for c in range(1, img_masked.shape[1]):
        if (img_masked[0, c] > 0).sum().item() > 0:
            return c
    print("    WARNING: no nonzero mask channel found, defaulting to channel 1")
    return 1


# ---------------------------------------------------------------------------
# Padding helpers (200 → 208)
# ---------------------------------------------------------------------------

def _pad_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.pad(img, PAD_EACH, mode="constant", constant_values=0)
    return np.pad(img, ((PAD_EACH, PAD_EACH), (PAD_EACH, PAD_EACH), (0, 0)),
                  mode="constant", constant_values=0)


def _pad_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3:
        t = t.squeeze(0)
    return F.pad(t, (PAD_EACH, PAD_EACH, PAD_EACH, PAD_EACH), value=0)


# ---------------------------------------------------------------------------
# Mode 1 — Single-view DRR + optional mask overlay
# ---------------------------------------------------------------------------

def generate_single_view(
    ct_path: str,
    mask_path: str | None,
    out_img_path: str | None = None,
    out_overlay_path: str | None = None,
    threshold: float = 0.0,
    alpha: float = 0.4,
    overlay_sigma: float | None = None,
    device: torch.device | None = None,
):
    """
    Render one CT into an AP DRR grayscale image and an optional mask overlay.

    Args:
        ct_path:         Path to CT NIfTI file.
        mask_path:       Path to segmentation NIfTI (or None for no mask).
        out_img_path:    Save path for grayscale DRR PNG (or None to skip).
        out_overlay_path: Save path for overlay PNG (or None to skip).
        threshold:       Pixel threshold above which the DRR background mask = 1.
        alpha:           Overlay blending strength.
        overlay_sigma:   Gaussian smoothing sigma for overlay edges (None = sharp).
        device:          Torch device.

    Returns:
        img:        (1, 1, H, W) DRR intensity tensor.
        img_masked: (1, C, H, W) per-channel mask tensor, or None.
    """
    device = device or _get_device()
    drr    = _build_drr(ct_path, mask_path, device)
    pose   = _ap_pose(device)

    img        = drr(pose)
    img_masked = drr(pose, mask_to_channels=True) if mask_path else None

    if out_img_path:
        _save_grayscale(img, out_img_path)

    if out_overlay_path and img_masked is not None:
        _save_overlay(img, img_masked, out_overlay_path, alpha, overlay_sigma)

    return img, img_masked


def _save_grayscale(img: torch.Tensor, path: str):
    arr = _tensor_to_uint8(img[0, 0])
    Image.fromarray(arr).save(path)
    print(f"  Saved grayscale: {path}")


def _save_overlay(
    img: torch.Tensor,
    img_masked: torch.Tensor,
    path: str,
    alpha: float = 0.4,
    sigma: float | None = None,
    colors: list = DEFAULT_COLORS,
):
    """Blend each mask channel over the grayscale DRR and save as RGB PNG."""
    base = _tensor_to_uint8(img[0, 0])
    overlay = np.stack([base] * 3, axis=-1).astype(float)

    for j in range(1, img_masked.shape[1]):
        mask_np = img_masked[0, j].detach().cpu().numpy()
        mask_bin = (mask_np > 0).astype(float)

        if sigma is not None:
            mask_bin = gaussian_filter(mask_bin, sigma=sigma)
            mask_bin /= mask_bin.max() + 1e-8

        color = colors[(j - 1) % len(colors)]
        for c in range(3):
            overlay[..., c] = np.where(
                mask_bin > 0,
                (1 - alpha * mask_bin) * overlay[..., c] + alpha * mask_bin * color[c],
                overlay[..., c],
            )

    Image.fromarray(overlay.astype(np.uint8)).save(path)
    print(f"  Saved overlay: {path}")


def process_folder_single_view(
    ct_folder: str,
    mask_folder: str,
    out_img_folder: str,
    out_overlay_folder: str,
):
    """
    Batch-generate AP DRR images and overlay visualisations for a folder of CTs.

    Expected filenames:
        CT:   ct-{number}.nii.gz
        Mask: seg_comb-{number}.nii.gz
    """
    os.makedirs(out_img_folder, exist_ok=True)
    os.makedirs(out_overlay_folder, exist_ok=True)
    device = _get_device()

    for ct_file in sorted(os.listdir(ct_folder)):
        if not (ct_file.endswith(".nii.gz") and ct_file.startswith("ct-")):
            continue

        number    = ct_file.split("-")[1].split(".")[0]
        ct_path   = os.path.join(ct_folder, ct_file)
        mask_path = os.path.join(mask_folder, f"seg_comb-{number}.nii.gz")

        if not os.path.exists(mask_path):
            print(f"  Skipping {ct_file} — mask not found")
            continue

        generate_single_view(
            ct_path=ct_path,
            mask_path=mask_path,
            out_img_path=os.path.join(out_img_folder,     f"x_ray-{number}.png"),
            out_overlay_path=os.path.join(out_overlay_folder, f"x_ray_seg-{number}.png"),
            alpha=0.4,
            overlay_sigma=0.7,
            device=device,
        )


# ---------------------------------------------------------------------------
# Mode 2 — Multi-angle augmented DRR dataset
# ---------------------------------------------------------------------------

# Camera rotation angles and XYZ offsets defining each augmented view.
# Each combination produces one "feature variant" in the multi-view dataset.
_BASE_ROTS = [
    [torch.pi / 2 * 1 / 3, 0.0, 0.0],
    [torch.pi / 2 * 2 / 3, 0.0, 0.0],
    [torch.pi / 2 * 1 / 3, torch.pi / 2 * 1 / 3, 0.0],
    [torch.pi / 2 * 1 / 3, torch.pi / 2 * 2 / 3, 0.0],
    [torch.pi / 2 * 2 / 3, torch.pi / 2 * 1 / 3, 0.0],
    [torch.pi / 2 * 2 / 3, torch.pi / 2 * 2 / 3, 0.0],
]
_BASE_XYZ    = [0.0, 850.0, 0.0]
_XYZ_OFFSETS = [[0.0, 0.0, 0.0]]

AUGMENTATIONS = [
    (rot, [_BASE_XYZ[i] + off[i] for i in range(3)])
    for rot in _BASE_ROTS
    for off in _XYZ_OFFSETS
]


def _render_one_view(
    number: int,
    rot_list: list,
    xyz_list: list,
    ct_dir: Path,
    seg_root: Path,
    device: torch.device,
):
    """
    Render a single CT at one camera pose.

    Returns:
        drr_uint8_208  : (208, 208) uint8 numpy  — padded grayscale X-ray
        mask_bin_208   : (208, 208) torch.uint8  — padded binary nodule mask
        mask_img_208   : (208, 208) uint8 numpy  — padded mask as image
        nodule_channel : int
    """
    num_str  = str(number).zfill(3)
    ct_path  = ct_dir / f"ct-{num_str}.nii.gz"
    seg_dir  = seg_root / f"seg-{num_str}"

    candidates = sorted(seg_dir.glob("*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No mask found in {seg_dir}")
    mask_path = candidates[0]

    subject = read(volume=str(ct_path), labelmap=str(mask_path))
    drr_obj = DRR(subject, sdd=DRR_SDD, height=DRR_H, delx=DRR_DELX).to(device)

    rot_t = torch.tensor([rot_list], device=device, dtype=torch.float32)
    xyz_t = torch.tensor([xyz_list], device=device, dtype=torch.float32)
    rotmat        = euler_angles_to_matrix(rot_t, "ZXY")
    camera_center = torch.einsum("bij,bj->bi", rotmat, xyz_t)
    pose          = RigidTransform(make_matrix(rotmat, camera_center))

    source, target = drr_obj.detector(pose, None)
    raw      = drr_obj.render(drr_obj.density, source, target)
    raw_mask = drr_obj.render(drr_obj.density, source, target, mask_to_channels=True)

    result      = drr_obj.reshape_transform(raw,      batch_size=len(pose))  # (1,1,200,200)
    result_mask = drr_obj.reshape_transform(raw_mask, batch_size=len(pose))  # (1,C,200,200)

    drr_uint8     = _tensor_to_uint8(result[0, 0])
    drr_uint8_208 = _pad_image(drr_uint8)

    nodule_ch    = _find_nodule_channel(result_mask)
    mask_bin     = (result_mask[0, nodule_ch] > 0).to(torch.uint8)
    mask_bin_208 = _pad_tensor(mask_bin)
    mask_img_208 = (mask_bin_208.numpy() * 255).astype(np.uint8)

    return drr_uint8_208, mask_bin_208, mask_img_208, nodule_ch


def _discover_split_numbers(source_root: Path) -> dict:
    """Read train/val/test split from an existing dataset directory."""
    split_map = {}
    for split in ("train", "val", "test"):
        tensor_dir = source_root / split / "tensors"
        if not tensor_dir.exists():
            print(f"  WARNING: {tensor_dir} not found — skipping split '{split}'")
            split_map[split] = []
            continue
        nums = []
        for p in sorted(tensor_dir.glob("nodule-*.pt")):
            m = re.match(r"nodule-(\d+)\.pt", p.name)
            if m:
                nums.append(int(m.group(1)))
        split_map[split] = sorted(nums)
    return split_map


def generate_multiview_dataset(
    ct_dir: Path,
    seg_root: Path,
    source_root: Path,
    output_root: Path,
    start_aug_idx: int = 1,
):
    """
    Generate an augmented multi-angle DRR dataset.

    The train/val/test split is inherited from source_root (an existing
    single-view dataset). Each CT is rendered at every angle in AUGMENTATIONS.

    Output structure:
        output_root/<split>/images/             x-ray-{num}-{aug_idx}.png
        output_root/<split>/images_from_tensor/ x-ray-{num}-{aug_idx}.png
        output_root/<split>/tensors/            nodule-{num}-{aug_idx}.pt

    Args:
        ct_dir:       Directory containing ct-{num}.nii.gz files.
        seg_root:     Directory containing seg-{num}/ subdirectories.
        source_root:  Existing dataset whose split assignment is reused.
        output_root:  Where to write the new augmented dataset.
        start_aug_idx: First augmentation index to write (default 1). Useful
                       for resuming partial runs.
    """
    device = _get_device()
    print(f"Device: {device}")

    split_map = _discover_split_numbers(source_root)
    for split, nums in split_map.items():
        print(f"  {split}: {len(nums)} samples")

    # Create all output subdirectories
    for split in ("train", "val", "test"):
        for sub in ("images", "images_from_tensor", "tensors"):
            (output_root / split / sub).mkdir(parents=True, exist_ok=True)

    total_aug   = len(AUGMENTATIONS)
    total_cases = sum(len(v) for v in split_map.values())
    print(f"\n{total_aug} augmentations × {total_cases} CTs = {total_aug * total_cases} renders\n")

    for aug_idx, (rot_list, xyz_list) in enumerate(AUGMENTATIONS, start=start_aug_idx):
        print(f"\n{'='*60}")
        print(f"Augmentation {aug_idx}/{total_aug}  rot={rot_list}  xyz={xyz_list}")
        print(f"{'='*60}")

        for split, numbers in split_map.items():
            if not numbers:
                continue
            print(f"\n  [{split}]  ({len(numbers)} cases)")

            for number in numbers:
                num_str = str(number).zfill(3)
                try:
                    drr_img, bin_tensor, mask_img, ch = _render_one_view(
                        number, rot_list, xyz_list, ct_dir, seg_root, device
                    )
                    stem = f"{num_str}-{aug_idx}"
                    base = output_root / split
                    Image.fromarray(drr_img,  mode="L").save(base / "images"             / f"x-ray-{stem}.png")
                    Image.fromarray(mask_img, mode="L").save(base / "images_from_tensor" / f"x-ray-{stem}.png")
                    torch.save(bin_tensor,               base / "tensors"                / f"nodule-{stem}.pt")
                    print(f"    OK  {stem}  ch={ch}  nonzero={bin_tensor.sum().item()}")
                except FileNotFoundError as e:
                    print(f"    SKIP {num_str}  {e}")
                except Exception as e:
                    print(f"    ERROR {num_str}  {e}")
                    raise

    print(f"\nDone. Dataset written to: {output_root.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic X-ray DRR images from NSCLC CT volumes"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── single view ──────────────────────────────────────────────────────────
    p_single = sub.add_parser("single", help="Generate AP DRR + overlay for a folder of CTs")
    p_single.add_argument("--ct-dir",      required=True)
    p_single.add_argument("--mask-dir",    required=True)
    p_single.add_argument("--out-img",     required=True, help="Output folder for grayscale DRRs")
    p_single.add_argument("--out-overlay", required=True, help="Output folder for overlay images")

    # ── multi-view ───────────────────────────────────────────────────────────
    p_multi = sub.add_parser("multiview", help="Generate augmented multi-angle DRR dataset")
    p_multi.add_argument("--ct-dir",       required=True)
    p_multi.add_argument("--seg-root",     required=True)
    p_multi.add_argument("--source-root",  required=True,
                         help="Existing dataset whose train/val/test split to inherit")
    p_multi.add_argument("--output-root",  required=True)
    p_multi.add_argument("--start-aug-idx", type=int, default=1,
                         help="First augmentation index to write (for resuming partial runs)")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mode == "single":
        process_folder_single_view(
            ct_folder=args.ct_dir,
            mask_folder=args.mask_dir,
            out_img_folder=args.out_img,
            out_overlay_folder=args.out_overlay,
        )
    elif args.mode == "multiview":
        generate_multiview_dataset(
            ct_dir=Path(args.ct_dir),
            seg_root=Path(args.seg_root),
            source_root=Path(args.source_root),
            output_root=Path(args.output_root),
            start_aug_idx=args.start_aug_idx,
        )