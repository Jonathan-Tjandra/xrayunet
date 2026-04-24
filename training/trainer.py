"""
Shared utilities used by all training scripts:
  - reproducibility (set_seed)
  - weight initialisation (init_weights_kaiming)
  - gradient diagnostics (grad_norm, check_gradient_flow)
  - checkpoint helpers (next_available_index, save_checkpoint)
  - W&B setup (init_wandb)
  - LR scheduler factory (build_scheduler)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights_kaiming(m):
    """
    Kaiming (He) initialisation for Conv layers; ones/zeros for normalisation.
    Call via model.apply(init_weights_kaiming) before training.
    Skip when resuming from a checkpoint — don't overwrite learned weights.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Gradient diagnostics
# ---------------------------------------------------------------------------

def compute_grad_norm(model: nn.Module) -> float:
    """Return the total L2 norm of all parameter gradients."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total ** 0.5


def check_gradient_flow(model: nn.Module) -> list:
    """
    Return a list of (layer_name, grad_norm) for any parameters whose
    gradient norm is below 1e-7 (effectively dead / vanishing).
    """
    dead = []
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.norm().item() < 1e-7:
            dead.append((name, param.grad.norm().item()))
    return dead


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def next_available_index(save_dir: str) -> int:
    """
    Find the smallest integer X ≥ 1 such that neither lastX.pt nor bestX.pt
    already exists in save_dir. Prevents accidental overwrites.
    """
    existing = [
        f for f in os.listdir(save_dir)
        if (f.startswith("last") or f.startswith("best")) and f.endswith(".pt")
    ]
    used = set()
    for f in existing:
        digits = "".join(filter(str.isdigit, f))
        if digits:
            try:
                used.add(int(digits))
            except ValueError:
                pass
    x = 1
    while x in used:
        x += 1
    return x


def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler, best_val_dice: float):
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_val_dice":   best_val_dice,
        },
        path,
    )


def load_checkpoint(path: str, device, model, optimizer=None, scheduler=None):
    """
    Load a checkpoint. Returns (checkpoint_dict, start_epoch, best_val_dice).
    Gracefully skips optimizer/scheduler state if shapes have changed.
    """
    ckpt = torch.load(path, map_location=device)

    try:
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded model_state from {path}")
    except Exception as e:
        print(f"  Could not load model_state: {e}")

    if optimizer is not None and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print("  Loaded optimizer_state")
        except Exception as e:
            print(f"  Could not load optimizer_state: {e}")

    if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"]:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
            print("  Loaded scheduler_state")
        except Exception as e:
            print(f"  Could not load scheduler_state: {e}")

    start_epoch   = ckpt.get("epoch", 0) + 1
    best_val_dice = ckpt.get("best_val_dice", -1.0)
    return ckpt, start_epoch, best_val_dice


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def init_wandb(project: str, run_prefix: str, config: dict):
    wandb.init(project=project, config=config)
    wandb.run.name = f"{run_prefix}_{wandb.run.id[:6]}"
    print(f"W&B run: {wandb.run.name}")


# ---------------------------------------------------------------------------
# LR scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, schedule: str, epochs: int, lr: float):
    """
    Build an LR scheduler from a string name.

    Options:
        "cosine"  – CosineAnnealingLR, decays to lr * 1e-2 over all epochs.
        "plateau" – ReduceLROnPlateau, halves LR when val dice stalls.
        "none"    – No scheduling (constant LR).
    """
    if schedule == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 1e-2
        )
    elif schedule == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, verbose=True
        )
    return None


def step_scheduler(scheduler, metric: float = None):
    """Step scheduler, passing metric for ReduceLROnPlateau if needed."""
    if scheduler is None:
        return
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()