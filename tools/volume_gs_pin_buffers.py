#!/usr/bin/env python3
"""Build VOLUME-GS upload buffers (pid + particle SoA) for atmosphere.

Supports:
  - pin from voxels (one GS per active cell, scale=spacing/√2)
  - fitted particles from P1b (`volume_gs_fit_torch.py` npz)

Stencil half-extent is REACH_SIGMA (6σ) AABB in cell units, matching device
`ATM_VOLUME_GS_REACH_SIGMA` gate.

Usage:
  from volume_gs_pin_buffers import build_pin_from_voxels, attach_volume_gs_to_atmosphere_buffers
  pin = build_pin_from_voxels(ka, ks, origin, spacing)
  attach_volume_gs_to_atmosphere_buffers(buffers, pin)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

REACH_SIGMA = 6.0


def stencil_radius_cells(
    scales: np.ndarray,
    spacing: np.ndarray,
    reach_sigma: float = REACH_SIGMA,
) -> tuple[int, int, int]:
    """Cell half-widths covering per-axis reach_sigma·s_axis AABB (device gate)."""
    scales = np.asarray(scales, dtype=np.float64)
    spacing = np.maximum(np.asarray(spacing, dtype=np.float64).reshape(3), 1e-12)
    if scales.ndim == 1:
        scales = scales.reshape(1, 3)
    s_axis = scales.max(axis=0) if scales.size else np.zeros(3)
    reach = reach_sigma * s_axis
    rx = int(max(1, np.ceil(reach[0] / spacing[0])))
    ry = int(max(1, np.ceil(reach[1] / spacing[1])))
    rz = int(max(1, np.ceil(reach[2] / spacing[2])))
    return rx, ry, rz


def build_pin_from_voxels(
    ka: np.ndarray,
    ks: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    ke_eps: float = 1.0e-6,
    max_particles: int | None = None,
    reach_sigma: float = REACH_SIGMA,
) -> dict:
    ka = np.asarray(ka, dtype=np.float32)
    ks = np.asarray(ks, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    nz, ny, nx = ka.shape
    ke = ka + ks
    iz, iy, ix = np.where(ke > ke_eps)
    n = int(iz.size)
    if max_particles is not None and n > max_particles:
        step = int(np.ceil(n / max_particles))
        iz, iy, ix = iz[::step], iy[::step], ix[::step]
        n = int(iz.size)
    centers = np.stack(
        [
            origin[0] + (ix + 0.5) * spacing[0],
            origin[1] + (iy + 0.5) * spacing[1],
            origin[2] + (iz + 0.5) * spacing[2],
        ],
        axis=1,
    ).astype(np.float32)
    scales = np.repeat((spacing / np.sqrt(2.0))[None, :], n, axis=0).astype(np.float32)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    ka_i = ka[iz, iy, ix].astype(np.float32)
    ks_i = ks[iz, iy, ix].astype(np.float32)
    pid = np.full((nz, ny, nx), -1, dtype=np.int32)
    pid[iz, iy, ix] = np.arange(n, dtype=np.int32)
    rx, ry, rz = stencil_radius_cells(scales, spacing, reach_sigma=reach_sigma)
    return {
        "pid": pid,
        "center": centers,
        "scale": scales,
        "quat": quats,
        "ka": ka_i,
        "ks": ks_i,
        "rx": rx,
        "ry": ry,
        "rz": rz,
        "n": n,
        "ijk": np.stack([iz, iy, ix], axis=1),
        "origin": origin.astype(np.float64),
        "spacing": spacing.astype(np.float64),
        "reach_sigma": float(reach_sigma),
        "source": "pin",
    }


def build_upload_from_particles(
    centers: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    ka: np.ndarray,
    ks: np.ndarray,
    ijk: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
    origin: np.ndarray,
    spacing: np.ndarray,
    reach_sigma: float = REACH_SIGMA,
) -> dict:
    """Fitted (or pin) particles → AtmVolumeGs upload dict with 6σ stencil."""
    centers = np.asarray(centers, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    quats = np.asarray(quats, dtype=np.float32)
    ka = np.asarray(ka, dtype=np.float32).reshape(-1)
    ks = np.asarray(ks, dtype=np.float32).reshape(-1)
    ijk = np.asarray(ijk, dtype=np.int64)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    nz, ny, nx = (int(grid_shape_zyx[0]), int(grid_shape_zyx[1]), int(grid_shape_zyx[2]))
    n = int(centers.shape[0])
    if ijk.shape != (n, 3):
        raise ValueError(f"ijk shape {ijk.shape} != ({n}, 3)")
    pid = np.full((nz, ny, nx), -1, dtype=np.int32)
    pid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(n, dtype=np.int32)
    rx, ry, rz = stencil_radius_cells(scales, spacing, reach_sigma=reach_sigma)
    return {
        "pid": pid,
        "center": centers,
        "scale": scales,
        "quat": quats,
        "ka": ka,
        "ks": ks,
        "rx": rx,
        "ry": ry,
        "rz": rz,
        "n": n,
        "ijk": ijk.astype(np.int64),
        "origin": origin,
        "spacing": spacing,
        "reach_sigma": float(reach_sigma),
        "source": "fit",
    }


def save_volume_gs_npz(path: str, upload: dict, extra: Optional[dict] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "pid": upload["pid"],
        "center": upload["center"],
        "scale": upload["scale"],
        "quat": upload["quat"],
        "ka": upload["ka"],
        "ks": upload["ks"],
        "ijk": upload["ijk"],
        "origin": np.asarray(upload["origin"], dtype=np.float64),
        "spacing": np.asarray(upload["spacing"], dtype=np.float64),
        "rx": np.int32(upload["rx"]),
        "ry": np.int32(upload["ry"]),
        "rz": np.int32(upload["rz"]),
        "n": np.int32(upload["n"]),
        "reach_sigma": np.float64(upload.get("reach_sigma", REACH_SIGMA)),
    }
    if extra:
        for k, v in extra.items():
            payload[k] = np.asarray(v)
    np.savez_compressed(path, **payload)
    return path


def load_volume_gs_npz(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {
        "pid": z["pid"],
        "center": z["center"].astype(np.float32),
        "scale": z["scale"].astype(np.float32),
        "quat": z["quat"].astype(np.float32),
        "ka": z["ka"].astype(np.float32),
        "ks": z["ks"].astype(np.float32),
        "ijk": z["ijk"].astype(np.int64) if "ijk" in z.files else None,
        "origin": z["origin"].astype(np.float64) if "origin" in z.files else None,
        "spacing": z["spacing"].astype(np.float64) if "spacing" in z.files else None,
        "rx": int(z["rx"]),
        "ry": int(z["ry"]),
        "rz": int(z["rz"]),
        "n": int(z["n"]),
        "reach_sigma": float(z["reach_sigma"]) if "reach_sigma" in z.files else REACH_SIGMA,
        "source": "npz",
    }


def attach_volume_gs_to_atmosphere_buffers(buffers, upload: dict, device=None, majorant_boost: float = 4.0):
    """Fill AtmosphereBuffers volume_gs_* tensors from pin/fit upload dict."""
    import torch

    if device is None:
        device = buffers.voxel_ka.device
    buffers.volume_gs_pid = torch.as_tensor(upload["pid"], dtype=torch.int32, device=device).contiguous()
    buffers.volume_gs_center = torch.as_tensor(upload["center"], dtype=torch.float32, device=device).contiguous()
    buffers.volume_gs_scale = torch.as_tensor(upload["scale"], dtype=torch.float32, device=device).contiguous()
    buffers.volume_gs_quat = torch.as_tensor(upload["quat"], dtype=torch.float32, device=device).contiguous()
    buffers.volume_gs_ka = torch.as_tensor(upload["ka"], dtype=torch.float32, device=device).contiguous()
    buffers.volume_gs_ks = torch.as_tensor(upload["ks"], dtype=torch.float32, device=device).contiguous()
    buffers.volume_gs_kmax = torch.empty(0, dtype=torch.float32, device=device)
    buffers.volume_gs_rx = int(upload["rx"])
    buffers.volume_gs_ry = int(upload["ry"])
    buffers.volume_gs_rz = int(upload["rz"])
    buffers.volume_gs_majorant_boost = float(majorant_boost)
    return buffers


# Backward-compatible alias
attach_pin_to_atmosphere_buffers = attach_volume_gs_to_atmosphere_buffers
