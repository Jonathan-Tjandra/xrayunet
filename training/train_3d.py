"""
Train a dual-view UNet with a combined 2D DiceBCE + 3D backprojection
consistency loss.

The model takes frontal and lateral X-rays concatenated along the channel
dimension (B, 2, H, W) and outputs two-channel logits, one per view.

Usage
-----
python training/train_3d.py \\
    --data-root-frontal  data/frontal \\
    --data-root-lateral  data/lateral \\
    --data-root-3d-gt    data/gt_3d   \\
    --save-root          checkpoints/3d \\
    --epochs 150 --batch-size 4 \\
    --bce-weight 0.2 --w-2d 1.0 --w-3d 0.5

Resume:
    ... --resume checkpoints/3d/best1.pt
"""

import os
import gc
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import wandb

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from models.unet import UNet
from models.losses import CombinedLoss, dice_coeff
from data.dataset import DualViewDataset, dual_view_collate
from utils.augmentation import KorniaAugmentation
from utils.drr_utils import get_drr_geometry
from training.trainer import (
    set_seed, init_weights_kaiming,
    compute_grad_norm, check_gradient_flow,
    next_available_index, save_checkpoint, load_checkpoint,
    init_wandb, build_scheduler, step_scheduler,
)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # ✅ speed + stability

    save_root = os.path.abspath(args.save_root or "checkpoints/3d")
    os.makedirs(save_root, exist_ok=True)

    init_wandb(
        project="lung-nodule-3d-consistency",
        run_prefix="3d",
        config=vars(args),
    )

    # ── Datasets & loaders ───────────────────────────────────────────────────
    transform = T.ToTensor()

    train_ds = DualViewDataset(
        args.data_root_frontal, args.data_root_lateral, args.data_root_3d_gt,
        split="train", transforms=transform,
    )
    val_ds = DualViewDataset(
        args.data_root_frontal, args.data_root_lateral, args.data_root_3d_gt,
        split="val", transforms=transform,
    )

    val_batch = args.val_batch_size or args.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dual_view_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dual_view_collate,
    )

    # ── Model, optimiser, loss ───────────────────────────────────────────────
    aug_train = KorniaAugmentation(apply_aug=True).to(device)

    model = UNet(in_channels=2, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = CombinedLoss(
        bce_weight=args.bce_weight,
        w_2d=args.w_2d,
        w_3d=args.w_3d,
    ).to(device)

    scheduler = build_scheduler(
        optimizer, args.lr_schedule, args.epochs, args.lr
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_dice = -1.0

    if args.resume:
        _, start_epoch, best_val_dice = load_checkpoint(
            args.resume, device, model, optimizer, scheduler
        )
        print(f"Resuming from epoch {start_epoch}")
    else:
        model.apply(init_weights_kaiming)
        print("Applied Kaiming weight initialisation")

    idx = next_available_index(save_root)
    last_path = os.path.join(save_root, f"last{idx}.pt")
    best_path = os.path.join(save_root, f"best{idx}.pt")

    current_epoch = start_epoch - 1

    # ── Epoch loop ───────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            model.train()

            losses, losses_2d, losses_3d, grad_norms = [], [], [], []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

            for fro_img, lat_img, fro_mask, lat_mask, vol_gts, ct_paths, mask_paths, nums in pbar:

                fro_img = fro_img.to(device)
                lat_img = lat_img.to(device)
                fro_mask = fro_mask.to(device)
                lat_mask = lat_mask.to(device)

                optimizer.zero_grad()

                total_loss = total_2d = total_3d = 0.0

                for i in range(fro_img.shape[0]):
                    fro_i = fro_img[i:i+1]
                    lat_i = lat_img[i:i+1]
                    fro_m = fro_mask[i:i+1]
                    lat_m = lat_mask[i:i+1]

                    # ✅ SAFE GT handling
                    vol_gt_i = vol_gts[i]
                    if not isinstance(vol_gt_i, torch.Tensor):
                        vol_gt_i = torch.tensor(vol_gt_i)
                    vol_gt_i = vol_gt_i.float().to(device)

                    # DRR geometry
                    if args.w_3d > 0:
                        drr, (src_F, tgt_F), (src_L, tgt_L) = get_drr_geometry(
                            ct_paths[i], mask_paths[i], device
                        )
                    else:
                        drr = src_F = tgt_F = src_L = tgt_L = None

                    # ✅ FIXED normalization
                    norm = float(args.num_augs)

                    for _ in range(args.num_augs):
                        aug_fro, aug_fro_m = aug_train(fro_i, fro_m)
                        aug_lat, aug_lat_m = aug_train(lat_i, lat_m)

                        logits = model(torch.cat([aug_fro, aug_lat], dim=1))
                        logits_f = logits[:, 0:1]
                        logits_l = logits[:, 1:2]

                        loss, loss_2d, loss_3d = criterion(
                            logits_f, logits_l,
                            aug_fro_m, aug_lat_m,
                            src_F, tgt_F, src_L, tgt_L,
                            drr, vol_gt_i, device,
                        )

                        (loss / norm).backward()

                        total_loss += loss.item() / norm
                        total_2d += loss_2d.item() / norm
                        total_3d += loss_3d.item() / norm

                    # cleanup (lightweight)
                    if drr is not None:
                        del drr, src_F, tgt_F, src_L, tgt_L

                grad_norm = compute_grad_norm(model)
                grad_norms.append(grad_norm)

                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()

                losses.append(total_loss)
                losses_2d.append(total_2d)
                losses_3d.append(total_3d)

                if len(losses) % 5 == 0:
                    gc.collect()

                pbar.set_postfix({
                    "loss": f"{np.mean(losses):.4f}",
                    "2d": f"{np.mean(losses_2d):.4f}",
                    "3d": f"{np.mean(losses_3d):.4f}",
                    "grad": f"{grad_norm:.3e}",
                })

                wandb.log({
                    "batch/loss": total_loss,
                    "batch/grad_norm": grad_norm,
                    "epoch": epoch,
                })

            # ── Validation ───────────────────────────────────────────────────
            model.eval()
            val_dice_list = []

            with torch.no_grad():
                for fro_img, lat_img, fro_mask, lat_mask, *_ in val_loader:
                    fro_img = fro_img.to(device)
                    lat_img = lat_img.to(device)
                    fro_mask = fro_mask.to(device)
                    lat_mask = lat_mask.to(device)

                    logits = model(torch.cat([fro_img, lat_img], dim=1))
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    gt = torch.cat([fro_mask, lat_mask], dim=1)

                    try:
                        val_dice_list.append(float(dice_coeff(preds, gt)))
                    except Exception:
                        pass

            mean_loss = float(np.mean(losses))
            mean_val_dice = float(np.mean(val_dice_list)) if val_dice_list else -1.0
            mean_grad_norm = float(np.mean(grad_norms))
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}: loss={mean_loss:.4f}  "
                f"val_dice={mean_val_dice:.4f}  grad={mean_grad_norm:.3e}"
            )

            step_scheduler(scheduler, metric=mean_val_dice)

            wandb.log({
                "epoch": epoch,
                "train/loss": mean_loss,
                "train/loss_2d": float(np.mean(losses_2d)),
                "train/loss_3d": float(np.mean(losses_3d)),
                "val/dice": mean_val_dice,
                "train/grad_norm": mean_grad_norm,
                "learning_rate": current_lr,
            })

            if mean_val_dice > best_val_dice:
                best_val_dice = mean_val_dice
                save_checkpoint(best_path, epoch, model, optimizer, scheduler, best_val_dice)
                print(f"  Saved best: {best_path}  (dice={best_val_dice:.4f})")

            if epoch % 5 == 0:
                save_checkpoint(last_path, epoch, model, optimizer, scheduler, best_val_dice)
                print(f"  Saved checkpoint: {last_path}")

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
    parser = argparse.ArgumentParser(description="Dual-view UNet with 3D consistency loss")

    parser.add_argument("--data-root-frontal", type=str, required=True)
    parser.add_argument("--data-root-lateral", type=str, required=True)
    parser.add_argument("--data-root-3d-gt", type=str, required=True)
    parser.add_argument("--save-root", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bce-weight", type=float, default=0.2)
    parser.add_argument("--w-2d", type=float, default=1.0)
    parser.add_argument("--w-3d", type=float, default=0.5)
    parser.add_argument("--num-augs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"])

    args = parser.parse_args()
    train(args)