import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def dice_coeff(preds, targets, eps=1e-6):
    """
    Compute mean Dice coefficient between binary prediction and target tensors.

    Supports shapes (C, H, W), (1, H, W), or (B, 1, H, W).
    """
    if preds.dim() == 4:
        preds   = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter   = (preds * targets).sum(1)
        union   = preds.sum(1) + targets.sum(1)
        return ((2.0 * inter + eps) / (union + eps)).mean()
    else:
        p     = preds.contiguous().view(-1).float()
        t     = targets.contiguous().view(-1).float()
        inter = (p * t).sum()
        return (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


# ---------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return 1.0 - dice_coeff(probs, targets, eps=self.eps)


class DiceBCELoss(nn.Module):
    """
    Combined BCEWithLogitsLoss (applied to raw logits) and Dice loss
    (applied to sigmoid probabilities).

    total = bce_weight * BCE + (1 - bce_weight) * Dice
    """
    def __init__(self, bce_weight=0.5, eps=1e-6):
        super().__init__()
        self.bce       = nn.BCEWithLogitsLoss()
        self.dice      = DiceLoss(eps=eps)
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return (
            self.bce_weight * self.bce(logits, targets)
            + (1.0 - self.bce_weight) * self.dice(logits, targets)
        )


# ---------------------------------------------------------------------------
# 3D consistency loss via backprojection
# ---------------------------------------------------------------------------

class Backprojection3DConsistencyLoss(nn.Module):
    """
    Backprojects 2D segmentation predictions from frontal and lateral views
    into a shared 3D voxel volume, overlays them, and supervises with a 3D
    binary ground truth using BCE loss.

    Overlay logic: frontal_volume + lateral_volume → values in {0, 1, 2},
    then sigmoid maps those to probabilities before BCE.
    """
    def __init__(self, sample_points=2000):
        super().__init__()
        self.sample_points = sample_points
        self.bce = nn.BCELoss()

    def _backproject(self, mask_2d, source, target, drr, device, threshold=0.5):
        """
        Cast rays from source through active (above-threshold) pixels of mask_2d
        and mark the intersected voxels in a 3D volume.

        Args:
            mask_2d: (H_mask, W_mask) predicted probability mask.
            source:  Ray source position tensor (from DRR).
            target:  Detector pixel positions tensor (from DRR).
            drr:     DRR renderer (provides volume shape and affine).
            device:  Torch device.
            threshold: Binarization threshold for mask_2d.

        Returns:
            volume: (D, H, W) binary voxel volume.
        """
        volume_shape = drr.density.shape
        volume = torch.zeros(volume_shape, device=device)

        # Normalise source shape → (3,)
        if source.dim() == 2:
            src = source[0]
        elif source.dim() == 3:
            src = source[0, 0]
        else:
            raise ValueError(f"Unexpected source shape: {source.shape}")

        # Normalise target shape → (H, W, 3)
        if target.dim() == 4:
            target_grid = target[0]
            H, W = target_grid.shape[0], target_grid.shape[1]
        elif target.dim() == 3:
            H = drr.detector.height
            W = drr.detector.width
            n_points = target.shape[1]
            if n_points != H * W:
                raise ValueError(
                    f"Cannot reshape target: n_points={n_points}, expected H*W={H * W}"
                )
            target_grid = target[0].reshape(H, W, 3)
        else:
            raise ValueError(f"Unexpected target shape: {target.shape}")

        # Resize mask to detector resolution if needed
        if mask_2d.shape[0] != H or mask_2d.shape[1] != W:
            mask_resized = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            mask_binary = (mask_resized > threshold).float()
        else:
            mask_binary = (mask_2d > threshold).float()

        active_pixels = torch.nonzero(mask_binary, as_tuple=False)
        if active_pixels.shape[0] == 0:
            return volume

        det_pixels = target_grid[active_pixels[:, 0], active_pixels[:, 1]]  # (N, 3)
        ray_dirs   = det_pixels - src.unsqueeze(0)
        ray_lengths = torch.norm(ray_dirs, dim=1, keepdim=True)
        ray_dirs    = ray_dirs / (ray_lengths + 1e-8)

        # Sample points along each ray
        t_values   = torch.linspace(0, 1, self.sample_points, device=device)
        t_values   = t_values.unsqueeze(0).unsqueeze(2)              # (1, S, 1)
        max_t      = ray_lengths.unsqueeze(1) * 2.5                  # (N, 1, 1)
        t_samples  = t_values * max_t                                 # (N, S, 1)
        world_pts  = (
            src.unsqueeze(0).unsqueeze(0)
            + ray_dirs.unsqueeze(1) * t_samples
        )  # (N, S, 3)

        # Map world coordinates → voxel indices
        flat_pts    = world_pts.reshape(-1, 3)
        voxel_coords = drr.affine_inverse(flat_pts.unsqueeze(0))[0]
        voxel_idx   = torch.round(voxel_coords).long()

        valid = (
            (voxel_idx[:, 0] >= 0) & (voxel_idx[:, 0] < volume_shape[0]) &
            (voxel_idx[:, 1] >= 0) & (voxel_idx[:, 1] < volume_shape[1]) &
            (voxel_idx[:, 2] >= 0) & (voxel_idx[:, 2] < volume_shape[2])
        )
        valid_idx = voxel_idx[valid]
        if valid_idx.shape[0] > 0:
            volume[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]] = 1.0

        return volume

    def forward(
        self,
        pred_frontal, pred_lateral,
        source_F, target_F,
        source_L, target_L,
        drr, vol_gt_3d, device,
    ):
        """
        Args:
            pred_frontal:  (B, 1, H, W) frontal sigmoid probabilities.
            pred_lateral:  (B, 1, H, W) lateral sigmoid probabilities.
            source_F/L:    Ray source tensors for frontal/lateral views.
            target_F/L:    Detector pixel tensors for frontal/lateral views.
            drr:           DRR renderer object.
            vol_gt_3d:     (D, H, W) 3D ground truth binary volume.
            device:        Torch device.

        Returns:
            Scalar BCE loss between backprojected combined volume and 3D GT.
        """
        B = pred_frontal.shape[0]
        total_loss = 0.0

        for i in range(B):
            vol_F = self._backproject(pred_frontal[i, 0], source_F, target_F, drr, device)
            vol_L = self._backproject(pred_lateral[i, 0], source_L, target_L, drr, device)

            vol_combined = torch.sigmoid(vol_F + vol_L)  # (D, H, W), values ∈ (0, 1)

            if vol_combined.shape != vol_gt_3d.shape:
                gt = F.interpolate(
                    vol_gt_3d.unsqueeze(0).unsqueeze(0),
                    size=vol_combined.shape,
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                gt = (gt > 0.5).float()
            else:
                gt = vol_gt_3d

            total_loss += self.bce(vol_combined, gt)

        return total_loss / B


class CombinedLoss(nn.Module):
    """
    Combines 2D DiceBCE loss on both views with a 3D backprojection
    consistency loss.

    total = w_2d * mean(DiceBCE_frontal, DiceBCE_lateral) + w_3d * BCE_3d
    """
    def __init__(self, bce_weight=0.2, w_2d=1.0, w_3d=0.5, sample_points=200):
        super().__init__()
        self.dice_bce      = DiceBCELoss(bce_weight=bce_weight)
        self.consistency3d = Backprojection3DConsistencyLoss(sample_points=sample_points)
        self.w_2d = w_2d
        self.w_3d = w_3d

    def forward(
        self,
        logits_frontal, logits_lateral,
        target_frontal, target_lateral,
        source_F, target_F,
        source_L, target_L,
        drr, vol_gt_3d, device,
    ):
        """
        Args:
            logits_frontal/lateral:  (B, 1, H, W) raw logits.
            target_frontal/lateral:  (B, 1, H, W) binary ground truth masks.
            source_F/L, target_F/L:  DRR view geometry.
            drr:                     DRR renderer.
            vol_gt_3d:               (D, H, W) 3D ground truth.
            device:                  Torch device.

        Returns:
            (total_loss, loss_2d, loss_3d)
        """
        loss_2d = (
            self.dice_bce(logits_frontal, target_frontal)
            + self.dice_bce(logits_lateral, target_lateral)
        ) / 2.0

        if self.w_3d > 0:
            loss_3d = self.consistency3d(
                torch.sigmoid(logits_frontal),
                torch.sigmoid(logits_lateral),
                source_F, target_F,
                source_L, target_L,
                drr, vol_gt_3d, device,
            )
        else:
            loss_3d = torch.tensor(0.0, device=device)

        total_loss = self.w_2d * loss_2d + self.w_3d * loss_3d
        return total_loss, loss_2d, loss_3d