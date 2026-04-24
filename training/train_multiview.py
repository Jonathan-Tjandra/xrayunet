"""
Train a single-view UNet on many DRR angle/translation variants per CT.

Each epoch, a random subset of DRR feature variants (--num-feature) is
sampled without replacement, and each variant is independently augmented
--num-aug times with Kornia. Total forward passes per CT per epoch:
    num_feature × num_aug

The model takes a single-channel X-ray (B, 1, H, W) and produces a
single-channel mask (B, 1, H, W).

Usage
-----
python training/train_multiview.py \\
    --data-root  data/multiview \\
    --save-root  checkpoints/multiview \\
    --epochs 2500 --batch-size 4 \\
    --num-feature 16 --num-aug 25

Resume:
    ... --resume checkpoints/multiview/best1.pt
"""

import os
import gc
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import wandb

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from models.unet import UNet
from data.dataset import MultiViewDataset, multi_view_collate
from utils.augmentation import KorniaAugmentation
from training.trainer import (
    set_seed, init_weights_kaiming,
    compute_grad_norm, check_gradient_flow,
    next_available_index, save_checkpoint, load_checkpoint,
    init_wandb, build_scheduler, step_scheduler,
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(logits, targets, smooth=1.0):
    pred  = torch.sigmoid(logits)
    inter = (pred * targets).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + targets.sum() + smooth)


def combined_loss(logits, targets, bce_weight):
    bce  = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    total = bce_weight * bce + (1.0 - bce_weight) * dice
    return total, bce, dice


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_root = os.path.abspath(args.save_root or "checkpoints/multiview")
    os.makedirs(save_root, exist_ok=True)

    init_wandb(
        project="lung-nodule-segmentation",
        run_prefix="multiview",
        config=vars(args),
    )

    # ── Datasets & loaders ───────────────────────────────────────────────────
    transform    = T.ToTensor()
    train_ds     = MultiViewDataset(args.data_root, split="train", transforms=transform)
    val_ds       = MultiViewDataset(args.data_root, split="val",   transforms=transform)

    total_features = train_ds.total_features
    print(f"Total feature variants available: {total_features}")
    if args.num_feature > total_features:
        raise ValueError(
            f"--num-feature ({args.num_feature}) > total available ({total_features})"
        )

    val_batch    = args.val_batch_size or args.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=multi_view_collate,
    )
    val_loader   = DataLoader(
        val_ds, batch_size=val_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=multi_view_collate,
    )

    # ── Model, optimiser ─────────────────────────────────────────────────────
    aug_fn    = KorniaAugmentation(apply_aug=True).to(device)

    # Warm-up Kornia JIT compilation before training starts
    with torch.no_grad():
        _d = torch.zeros(1, 1, 64, 64, device=device)
        aug_fn(_d, _d)
        del _d

    model     = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(optimizer, args.lr_schedule, args.epochs, args.lr)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_dice = -1.0
    if args.resume:
        _, start_epoch, best_val_dice = load_checkpoint(
            args.resume, device, model, optimizer, scheduler
        )
        print(f"Resuming from epoch {start_epoch}")
    else:
        model.apply(init_weights_kaiming)
        print("Applied Kaiming weight initialisation")

    idx       = next_available_index(save_root)
    last_path = os.path.join(save_root, f"last{idx}.pt")
    best_path = os.path.join(save_root, f"best{idx}.pt")

    current_epoch = start_epoch - 1

    # ── Epoch loop ───────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch

            # Sample a fresh subset of feature variants each epoch
            sampled = random.sample(range(1, total_features + 1), args.num_feature)
            train_ds.resample_features(sampled)
            val_ds.resample_features(sampled)

            print(f"\nEpoch {epoch}/{args.epochs}  features={sampled}  num_aug={args.num_aug}")

            model.train()
            total_list, bce_list, dice_list, grad_list = [], [], [], []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

            for imgs, masks, nums, feature_indices in pbar:
                # imgs, masks: (B, num_feature, 1, H, W)
                imgs  = imgs.to(device)
                masks = masks.to(device)

                B          = imgs.shape[0]
                norm       = float(B * args.num_feature * args.num_aug)

                optimizer.zero_grad()
                batch_total = batch_bce = batch_dice = 0.0

                for fi in range(args.num_feature):
                    img_fi  = imgs[:, fi]   # (B, 1, H, W)
                    mask_fi = masks[:, fi]

                    for _ in range(args.num_aug):
                        aug_img, aug_mask = aug_fn(img_fi, mask_fi)
                        logits = model(aug_img)
                        loss, bce, dice = combined_loss(logits, aug_mask, args.bce_weight)
                        (args.w_2d * loss / norm).backward()

                        batch_total += (args.w_2d * loss / norm).item()
                        batch_bce   += bce.item()  / norm
                        batch_dice  += dice.item() / norm

                grad_norm = compute_grad_norm(model)
                grad_list.append(grad_norm)

                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()

                del imgs, masks
                torch.cuda.empty_cache()
                if len(total_list) % 5 == 0:
                    gc.collect()

                total_list.append(batch_total)
                bce_list.append(batch_bce)
                dice_list.append(batch_dice)

                pbar.set_postfix({
                    "total": f"{np.mean(total_list):.4f}",
                    "bce":   f"{np.mean(bce_list):.4f}",
                    "dice":  f"{np.mean(dice_list):.4f}",
                    "grad":  f"{grad_norm:.3e}",
                })
                wandb.log({"batch/total": batch_total, "batch/grad_norm": grad_norm, "epoch": epoch})

            # ── Validation ───────────────────────────────────────────────────
            model.eval()
            val_total_list, val_bce_list, val_dice_list = [], [], []

            with torch.no_grad():
                for imgs, masks, nums, feature_indices in val_loader:
                    imgs  = imgs.to(device)
                    masks = masks.to(device)

                    for fi in range(imgs.shape[1]):
                        logits = model(imgs[:, fi])
                        loss, bce, dice = combined_loss(logits, masks[:, fi], args.bce_weight)
                        val_total_list.append((args.w_2d * loss).item())
                        val_bce_list.append(bce.item())
                        val_dice_list.append(dice.item())

            # ── Epoch stats ───────────────────────────────────────────────────
            mean_train_total = float(np.mean(total_list))   if total_list   else 0.0
            mean_val_dice    = float(np.mean(val_dice_list)) if val_dice_list else 0.0
            mean_grad_norm   = float(np.mean(grad_list))    if grad_list    else 0.0

            # Dice coefficient (1 - dice_loss) — used for model selection
            val_dice_coeff = 1.0 - mean_val_dice
            current_lr     = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}: train={mean_train_total:.4f}  "
                f"val_dice_coeff={val_dice_coeff:.4f}  grad={mean_grad_norm:.3e}"
            )
            if mean_grad_norm < 1e-5:
                print(f"  WARNING: mean grad norm very small ({mean_grad_norm:.2e})")

            step_scheduler(scheduler, metric=val_dice_coeff)

            wandb.log({
                "epoch":             epoch,
                "train/total":       mean_train_total,
                "train/bce":         float(np.mean(bce_list)),
                "train/dice_loss":   float(np.mean(dice_list)),
                "val/total":         float(np.mean(val_total_list)) if val_total_list else 0.0,
                "val/bce":           float(np.mean(val_bce_list))   if val_bce_list   else 0.0,
                "val/dice_loss":     mean_val_dice,
                "val/dice_coeff":    val_dice_coeff,
                "train/grad_norm":   mean_grad_norm,
                "learning_rate":     current_lr,
            })

            if epoch % 10 == 0:
                dead = check_gradient_flow(model)
                if dead:
                    print(f"  Dead gradient layers at epoch {epoch}:")
                    for name, norm in dead:
                        print(f"    {name}: {norm:.2e}")
                    wandb.log({"epoch/dead_layer_count": len(dead), "epoch": epoch})

            if val_dice_coeff > best_val_dice:
                best_val_dice = val_dice_coeff
                save_checkpoint(best_path, epoch, model, optimizer, scheduler, best_val_dice)
                print(f"  Saved best: {best_path}  (val_dice_coeff={best_val_dice:.4f})")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        save_checkpoint(last_path, current_epoch, model, optimizer, scheduler, best_val_dice)
        print(f"Final checkpoint saved: {last_path}")
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-view UNet on DRR angle variants")

    parser.add_argument("--data-root",       type=str, required=True,
                        help="Root of multi-view dataset")
    parser.add_argument("--save-root",       type=str, default=None)

    parser.add_argument("--epochs",         type=int,   default=150)
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--val-batch-size", type=int,   default=None)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--bce-weight",     type=float, default=0.2,
                        help="Weight of BCE inside DiceBCE loss")
    parser.add_argument("--w-2d",           type=float, default=1.0,
                        help="Scalar weight on the 2D DiceBCE loss")
    parser.add_argument("--num-feature",    type=int,   default=5,
                        help="DRR angle variants sampled per epoch (without replacement)")
    parser.add_argument("--num-aug",        type=int,   default=1,
                        help="Kornia augmentation repeats per feature variant")
    parser.add_argument("--num-workers",    type=int,   default=4)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--resume",         type=str,   default=None)
    parser.add_argument("--grad-clip",      type=float, default=1.0,
                        help="Gradient clipping max norm. 0 to disable.")
    parser.add_argument("--lr-schedule",    type=str,   default="cosine",
                        choices=["cosine", "plateau", "none"])

    args = parser.parse_args()

    if args.num_feature < 1:
        parser.error("--num-feature must be >= 1")
    if args.num_aug < 1:
        parser.error("--num-aug must be >= 1")

    train(args)