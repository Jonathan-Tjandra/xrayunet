"""
Evaluate a trained UNet on the test set.
Supports single- and dual-channel checkpoints, optional 3D Dice evaluation
via backprojection, and saving overlay/prediction visualization.

Usage
-----
# 2D-only (fast)
python inference/test.py \
    --model-path saved_models/best.pt \
    --data-root-frontal data_frontal \
    --data-root-lateral data_lateral \
    --data-root-3d-gt   data_3d_gt \
    --no-3d

# Full 3D evaluation + save overlays
python inference/test.py \
    --model-path saved_models/best.pt \
    --data-root-frontal data_frontal \
    --data-root-lateral data_lateral \
    --data-root-3d-gt   data_3d_gt \
    --save-preds --save-pred-dir results/
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from models.unet import UNet
from models.losses import dice_coeff
from data.dataset import DualViewDataset
from utils.drr_utils import get_drr_geometry, backproject_mask_to_volume


# ---------------------------------------------------------------------------
# Eval-only metrics not in losses.py (IoU and 3-D variants)
# ---------------------------------------------------------------------------

def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    """Binary 2-D IoU (threshold at 0.5)."""
    pred   = (pred   > 0.5).float()
    target = (target > 0.5).float()
    inter  = (pred * target).sum(dim=(1, 2, 3))
    union  = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


def dice_3d(vol_probs: torch.Tensor, vol_gt: torch.Tensor, eps: float = 1e-6):
    """
    Volumetric Dice between a soft probability volume and binary GT.
    Resizes GT to match vol_probs if shapes differ.
    Returns (coeff, loss) where loss = 1 - coeff.
    """
    if vol_probs.shape != vol_gt.shape:
        vol_gt = torch.nn.functional.interpolate(
            vol_gt.unsqueeze(0).unsqueeze(0),
            size=vol_probs.shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        vol_gt = (vol_gt > 0.5).float()

    p     = vol_probs.contiguous().view(-1)
    t     = vol_gt.contiguous().view(-1).float()
    inter = (p * t).sum()
    coeff = (2.0 * inter + eps) / (p.sum() + t.sum() + eps)
    return coeff.item(), (1.0 - coeff).item()


def dice_3d_intersection(
    vol_f: torch.Tensor, vol_l: torch.Tensor, vol_gt: torch.Tensor, eps: float = 1e-7
) -> float:
    """
    Secondary diagnostic: Dice on the hard intersection of both backprojected
    views against the 3-D GT. Complements dice_3d by penalising over-projection.
    """
    vol_inter = ((vol_f > 0) & (vol_l > 0)).float()
    if vol_inter.shape != vol_gt.shape:
        vol_inter = torch.nn.functional.interpolate(
            vol_inter.unsqueeze(0).unsqueeze(0),
            size=vol_gt.shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        vol_inter = (vol_inter > 0.5).float()
    inter = (vol_inter * vol_gt).sum()
    union = vol_inter.sum() + vol_gt.sum()
    return ((2.0 * inter + eps) / (union + eps)).item()


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def _unwrap_state(checkpoint: dict) -> dict:
    return checkpoint.get("model_state", checkpoint)


def detect_channels(checkpoint: dict) -> tuple[int, int]:
    """
    Infer in_channels and out_channels from saved weight shapes.
    Reads inc.net.0.weight (in_channels) and outc.conv.weight (out_channels).
    """
    state = _unwrap_state(checkpoint)
    if "inc.net.0.weight" not in state or "outc.conv.weight" not in state:
        raise KeyError(
            "Cannot auto-detect channels: checkpoint is missing expected keys. "
            "Pass --in-channels and --out-channels manually."
        )
    in_ch  = state["inc.net.0.weight"].shape[1]
    out_ch = state["outc.conv.weight"].shape[0]
    return in_ch, out_ch


def load_model(checkpoint: dict, in_channels: int, out_channels: int, device: torch.device) -> UNet:
    """
    Build UNet and load weights, adapting channel mismatches if needed.
    A mismatch typically means a single-output checkpoint is loaded into a
    dual-output model (or vice versa); weights are tiled/averaged safely.
    """
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
    state = _unwrap_state(checkpoint)
    model_sd = model.state_dict()

    adapted = {}
    for name, param in state.items():
        if name not in model_sd:
            continue
        target = model_sd[name]
        if param.shape == target.shape:
            adapted[name] = param
            continue

        print(f"  [adapt] {name}: {tuple(param.shape)} → {tuple(target.shape)}")
        if param.ndim == 4:
            # Input channel expansion: [C_out, 1, kH, kW] → [C_out, 2, kH, kW]
            if param.shape[1] == 1 and target.shape[1] == 2:
                adapted[name] = param.repeat(1, 2, 1, 1) / 2.0
                continue
            # Output channel expansion: [1, C_in, kH, kW] → [2, C_in, kH, kW]
            if param.shape[0] == 1 and target.shape[0] == 2:
                adapted[name] = param.repeat(2, 1, 1, 1)
                continue
        if param.ndim == 1 and param.shape[0] == 1 and target.shape[0] == 2:
            adapted[name] = param.repeat(2)
            continue
        # Fallback: skip incompatible weight (keeps random init for that layer)
        print(f"  [skip]  {name}: shapes incompatible, keeping random init")

    model.load_state_dict(adapted, strict=False)
    return model


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def make_overlay(img: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blend grayscale X-ray with prediction (red) and GT mask (green).
    All inputs are uint8 [0, 255] 2-D arrays.
    """
    rgb = np.stack([img, img, img], axis=-1)
    rgb[:, :, 0] = np.maximum(rgb[:, :, 0], (pred > 127).astype(np.uint8) * 255)
    rgb[:, :, 1] = np.maximum(rgb[:, :, 1], (mask > 127).astype(np.uint8) * 255)
    return rgb


def save_sample(
    save_dir: str,
    idx: int,
    fro_img, lat_img,
    pred_f, pred_l,
    fro_mask, lat_mask,
) -> None:
    """Write per-sample images and overlays to disk."""
    out = os.path.join(save_dir, f"sample_{idx:03d}")
    os.makedirs(out, exist_ok=True)

    def _to_u8(t):
        return (t.squeeze().cpu().numpy() * 255).astype(np.uint8)

    fi, li   = _to_u8(fro_img),  _to_u8(lat_img)
    pf, pl   = _to_u8(pred_f),   _to_u8(pred_l)
    mf, ml   = _to_u8(fro_mask), _to_u8(lat_mask)

    Image.fromarray(fi).save(os.path.join(out, "img_frontal.png"))
    Image.fromarray(li).save(os.path.join(out, "img_lateral.png"))
    Image.fromarray(pf).save(os.path.join(out, "pred_frontal.png"))
    Image.fromarray(pl).save(os.path.join(out, "pred_lateral.png"))
    Image.fromarray(mf).save(os.path.join(out, "mask_frontal.png"))
    Image.fromarray(ml).save(os.path.join(out, "mask_lateral.png"))
    Image.fromarray(make_overlay(fi, pf, mf)).save(os.path.join(out, "overlay_frontal.png"))
    Image.fromarray(make_overlay(li, pl, ml)).save(os.path.join(out, "overlay_lateral.png"))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_3d  = not args.no_3d

    print(f"\n{'='*60}")
    print(f"  Testing UNet — Lung Nodule Detection")
    print(f"{'='*60}")
    print(f"  Model      : {args.model_path}")
    print(f"  Device     : {device}")
    print(f"  3-D eval   : {'yes' if eval_3d else 'no  (--no-3d)'}")
    print(f"{'='*60}\n")

    # ---- Load checkpoint ------------------------------------------------
    checkpoint = torch.load(args.model_path, map_location=device)

    if args.in_channels is not None and args.out_channels is not None:
        in_ch, out_ch = args.in_channels, args.out_channels
        print(f"  Channels (manual)    : in={in_ch}, out={out_ch}")
    else:
        in_ch, out_ch = detect_channels(checkpoint)
        print(f"  Channels (auto)      : in={in_ch}, out={out_ch}")

    model = load_model(checkpoint, in_ch, out_ch, device)
    epoch = checkpoint.get("epoch", "?") if isinstance(checkpoint, dict) else "?"
    print(f"  Loaded epoch         : {epoch}\n")
    model.eval()

    dual_input  = (in_ch  == 2)
    dual_output = (out_ch == 2)
    print(f"  Input mode  : {'frontal + lateral (concatenated)' if dual_input  else 'frontal only'}")
    print(f"  Output mode : {'dual head (frontal + lateral)'    if dual_output else 'single head'}\n")

    # ---- Dataset --------------------------------------------------------
    dataset = DualViewDataset(
        root_frontal=args.data_root_frontal,
        root_lateral=args.data_root_lateral,
        root_3d_gt=args.data_root_3d_gt,
        split="test",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"  Test samples : {len(dataset)}\n")

    # ---- Metric accumulators --------------------------------------------
    dice_f_list, dice_l_list = [], []
    iou_f_list,  iou_l_list  = [], []
    d3_coeff_list, d3_loss_list, d3_inter_list = [], [], []

    n_save    = args.num_samples_to_save if args.num_samples_to_save is not None else len(dataset)
    saved_idx = 0

    if args.save_preds:
        os.makedirs(args.save_pred_dir, exist_ok=True)

    # ---- Evaluation loop ------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            fro_img, lat_img, fro_mask, lat_mask, vol_gt, ct_paths, mask_paths, fnames = batch

            fro_img  = fro_img.to(device)
            lat_img  = lat_img.to(device)
            fro_mask = fro_mask.to(device)
            lat_mask = lat_mask.to(device)

            # Forward pass
            inp    = torch.cat([fro_img, lat_img], dim=1) if dual_input else fro_img
            logits = model(inp)

            pred_f = torch.sigmoid(logits[:, 0:1])
            pred_l = torch.sigmoid(logits[:, 1:2]) if dual_output else pred_f

            # 2-D metrics (dice_coeff imported from models.losses)
            dice_f_list.append(dice_coeff((pred_f > 0.5).float(), fro_mask).item())
            iou_f_list.append(iou_score(pred_f, fro_mask))
            dice_l_list.append(dice_coeff((pred_l > 0.5).float(), lat_mask).item())
            iou_l_list.append(iou_score(pred_l, lat_mask))

            # 3-D metrics (per sample — DRR loading is expensive)
            if eval_3d:
                for i in range(len(fnames)):
                    drr, (src_f, tgt_f), (src_l, tgt_l) = get_drr_geometry(
                        ct_paths[i], mask_paths[i], device
                    )
                    vol_gt_i = vol_gt[i].to(device)

                    vol_pred_f = backproject_mask_to_volume(pred_f[i, 0], src_f, tgt_f, drr, device)
                    vol_pred_l = backproject_mask_to_volume(pred_l[i, 0], src_l, tgt_l, drr, device)

                    vol_probs = torch.sigmoid(vol_pred_f + vol_pred_l)
                    coeff, loss = dice_3d(vol_probs, vol_gt_i)
                    inter       = dice_3d_intersection(vol_pred_f, vol_pred_l, vol_gt_i)

                    d3_coeff_list.append(coeff)
                    d3_loss_list.append(loss)
                    d3_inter_list.append(inter)

                    del drr, src_f, tgt_f, src_l, tgt_l
                    torch.cuda.empty_cache()

            # Save visualisations
            if args.save_preds:
                for i in range(len(fnames)):
                    if saved_idx >= n_save:
                        break
                    save_sample(
                        args.save_pred_dir, saved_idx,
                        fro_img[i], lat_img[i],
                        pred_f[i],  pred_l[i],
                        fro_mask[i], lat_mask[i],
                    )
                    saved_idx += 1

    # ---- Print summary --------------------------------------------------
    mean_df, std_df = np.mean(dice_f_list), np.std(dice_f_list)
    mean_dl, std_dl = np.mean(dice_l_list), np.std(dice_l_list)
    mean_iou_f      = np.mean(iou_f_list)
    mean_iou_l      = np.mean(iou_l_list)
    overall_dice    = (mean_df + mean_dl) / 2
    overall_iou     = (mean_iou_f + mean_iou_l) / 2

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"\n  Frontal  —  Dice: {mean_df:.4f} ± {std_df:.4f}   IoU: {mean_iou_f:.4f}")
    print(f"  Lateral  —  Dice: {mean_dl:.4f} ± {std_dl:.4f}   IoU: {mean_iou_l:.4f}")

    if eval_3d:
        mean_d3c, std_d3c = np.mean(d3_coeff_list), np.std(d3_coeff_list)
        mean_d3l, std_d3l = np.mean(d3_loss_list),  np.std(d3_loss_list)
        mean_d3i, std_d3i = np.mean(d3_inter_list), np.std(d3_inter_list)
        print(f"\n  3-D Dice coeff      : {mean_d3c:.4f} ± {std_d3c:.4f}  (higher = better)")
        print(f"  3-D Dice loss       : {mean_d3l:.4f} ± {std_d3l:.4f}  (lower  = better)")
        print(f"  3-D Dice intersect  : {mean_d3i:.4f} ± {std_d3i:.4f}  (secondary)")
    else:
        print("\n  3-D evaluation : skipped (--no-3d)")

    print(f"\n  Overall 2-D Dice : {overall_dice:.4f}")
    print(f"  Overall 2-D IoU  : {overall_iou:.4f}")
    if eval_3d:
        print(f"  Overall 3-D Dice : {mean_d3c:.4f}")
    print(f"{'='*60}\n")

    # ---- Write results.txt ----------------------------------------------
    if args.save_preds:
        results_path = os.path.join(args.save_pred_dir, "test_results.txt")
        with open(results_path, "w") as f:
            f.write("UNet — Lung Nodule Detection — Test Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"model        : {args.model_path}\n")
            f.write(f"in_channels  : {in_ch}   out_channels : {out_ch}\n")
            f.write(f"test samples : {len(dataset)}\n")
            f.write(f"3-D eval     : {'yes' if eval_3d else 'no'}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Frontal  Dice : {mean_df:.4f} +/- {std_df:.4f}\n")
            f.write(f"Frontal  IoU  : {mean_iou_f:.4f}\n")
            f.write(f"Lateral  Dice : {mean_dl:.4f} +/- {std_dl:.4f}\n")
            f.write(f"Lateral  IoU  : {mean_iou_l:.4f}\n\n")
            if eval_3d:
                f.write(f"3-D Dice coeff     : {mean_d3c:.4f} +/- {std_d3c:.4f}\n")
                f.write(f"3-D Dice loss      : {mean_d3l:.4f} +/- {std_d3l:.4f}\n")
                f.write(f"3-D Dice intersect : {mean_d3i:.4f} +/- {std_d3i:.4f}\n\n")
            f.write(f"Overall 2-D Dice : {overall_dice:.4f}\n")
            f.write(f"Overall 2-D IoU  : {overall_iou:.4f}\n")
            if eval_3d:
                f.write(f"Overall 3-D Dice : {mean_d3c:.4f}\n")
        print(f"  Results saved → {results_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained UNet on the lung-nodule DRR test set."
    )
    p.add_argument("--model-path",          type=str, required=True,
                   help="Path to .pt checkpoint")
    p.add_argument("--data-root-frontal",   type=str, required=True,
                   help="Root of the frontal DRR dataset (train/test/val structure)")
    p.add_argument("--data-root-lateral",   type=str, required=True,
                   help="Root of the lateral DRR dataset")
    p.add_argument("--data-root-3d-gt",     type=str, required=True,
                   help="Directory containing {num}_gt.pt / _ct.txt / _mask.txt files")
    p.add_argument("--batch-size",          type=int, default=1)
    p.add_argument("--num-workers",         type=int, default=4)
    p.add_argument("--no-3d",               action="store_true",
                   help="Skip 3-D Dice evaluation (much faster)")
    p.add_argument("--save-preds",          action="store_true",
                   help="Save prediction images and overlays")
    p.add_argument("--save-pred-dir",       type=str, default="test_results",
                   help="Output directory for saved predictions")
    p.add_argument("--num-samples-to-save", type=int, default=None,
                   help="Limit the number of samples saved (default: all)")
    p.add_argument("--in-channels",         type=int, default=None,
                   help="Override auto-detected in_channels")
    p.add_argument("--out-channels",        type=int, default=None,
                   help="Override auto-detected out_channels")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())