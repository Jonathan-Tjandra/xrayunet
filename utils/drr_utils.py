"""
Helpers for setting up DiffDRR renderers and computing frontal/lateral
camera geometry from a CT volume.
"""

import torch
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, euler_angles_to_matrix, make_matrix
from diffdrr.data import read


def build_drr(ct_path: str, mask_path: str, device: torch.device) -> DRR:
    """
    Initialise a DiffDRR renderer from a CT volume and its labelmap.

    Args:
        ct_path:   Path to the CT NIfTI file.
        mask_path: Path to the nodule mask NIfTI file.
        device:    Torch device.

    Returns:
        drr: DRR renderer placed on device.
    """
    subject = read(volume=ct_path, labelmap=mask_path)
    drr = DRR(subject, sdd=1020, height=200, delx=2.0).to(device)
    return drr


def get_view_geometry(drr: DRR, rot_euler: torch.Tensor, xyz: torch.Tensor, device: torch.device):
    """
    Compute source and target tensors for a single DRR view.

    Args:
        drr:       DRR renderer.
        rot_euler: (1, 3) Euler angles in radians (ZXY convention).
        xyz:       (1, 3) Camera translation.
        device:    Torch device.

    Returns:
        source: Ray source positions.
        target: Detector pixel positions.
    """
    rotmat = euler_angles_to_matrix(rot_euler, "ZXY")
    camera_center = torch.einsum("bij,bj->bi", rotmat, xyz)
    matrix = make_matrix(rotmat, camera_center)
    pose   = RigidTransform(matrix)
    source, target = drr.detector(pose, None)
    return source, target


def get_drr_geometry(ct_path: str, mask_path: str, device: torch.device):
    """
    Build a DRR renderer and compute frontal + lateral view geometry.

    Returns:
        drr:              DRR renderer.
        (src_F, tgt_F):   Frontal view source and target tensors.
        (src_L, tgt_L):   Lateral view source and target tensors.
    """
    drr = build_drr(ct_path, mask_path, device)

    src_F, tgt_F = get_view_geometry(
        drr,
        rot_euler=torch.tensor([[0.0, 0.0, 0.0]], device=device),
        xyz=torch.tensor([[0.0, 850.0, 0.0]], device=device),
        device=device,
    )

    src_L, tgt_L = get_view_geometry(
        drr,
        rot_euler=torch.tensor([[torch.pi / 2, 0.0, 0.0]], device=device),
        xyz=torch.tensor([[0.0, 850.0, 0.0]], device=device),
        device=device,
    )

    return drr, (src_F, tgt_F), (src_L, tgt_L)