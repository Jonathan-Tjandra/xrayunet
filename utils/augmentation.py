"""
Kornia-based spatial and photometric augmentation for image-mask pairs.
"""

import torch
import torch.nn as nn
import kornia.augmentation as KA


class KorniaAugmentation(nn.Module):
    """
    Apply consistent spatial and photometric augmentations to an image and
    its corresponding segmentation mask.

    When apply_aug=False the module acts as a no-op (useful for validation).
    The mask is re-binarised after each augmentation to counteract
    interpolation artefacts introduced by affine / rotation transforms.

    Expected input shapes: (B, 1, H, W) for both image and mask.
    """

    def __init__(self, apply_aug: bool = True):
        super().__init__()
        self.apply_aug = apply_aug

        if not apply_aug:
            return

        self.aug = KA.AugmentationSequential(
            KA.RandomHorizontalFlip(p=0.5, same_on_batch=False),
            KA.RandomVerticalFlip(p=0.3, same_on_batch=False),
            KA.RandomRotation(degrees=15.0, p=0.5, same_on_batch=False),
            KA.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=(-5, 5),
                p=0.5,
                same_on_batch=False,
            ),
            KA.RandomBrightness(brightness=(0.9, 1.1), p=0.3, same_on_batch=False),
            KA.RandomContrast(contrast=(0.9, 1.1), p=0.3, same_on_batch=False),
            data_keys=["input", "mask"],
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        if not self.apply_aug:
            return image, mask
        aug_image, aug_mask = self.aug(image, mask)
        aug_mask = (aug_mask > 0.5).float()
        return aug_image, aug_mask