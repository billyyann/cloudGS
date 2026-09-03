# Voxel cloud + 1D gas atmosphere on the copied PTIR tracer.
# Cloud k is the voxel field only. Gaussians/mesh are surfaces.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


SPECTRAL_SW = 0
SPECTRAL_LW = 1


def identity_ray_to_world(device: torch.device | str) -> torch.Tensor:
    m = torch.zeros(3, 4, dtype=torch.float32, device=device)
    m[0, 0] = 1.0
    m[1, 1] = 1.0
    m[2, 2] = 1.0
    return m


def build_brick_majorants(
    ka: np.ndarray,
    ks: np.ndarray,
    brick: int = 8,
) -> np.ndarray:
    """Per-brick k̂ = max(ks). Same majorant as GPU Woodcock (htrdr SW)."""
    nz, ny, nx = ka.shape
    b = max(int(brick), 1)
    nbz = (nz + b - 1) // b
    nby = (ny + b - 1) // b
    nbx = (nx + b - 1) // b
    kmax = np.zeros((nbz, nby, nbx), dtype=np.float32)
    ks_a = np.asarray(ks, dtype=np.float32)
    for bz in range(nbz):
        for by in range(nby):
            for bx in range(nbx):
                sl = ks_a[bz * b : min((bz + 1) * b, nz), by * b : min((by + 1) * b, ny), bx * b : min((bx + 1) * b, nx)]
                kmax[bz, by, bx] = float(sl.max()) if sl.size else 0.0
    return kmax


def _to_cuda_f32(value, shape, device) -> torch.Tensor:
    if value is None:
        return torch.empty(shape, dtype=torch.float32, device=device)
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32).contiguous()
    return torch.as_tensor(value, dtype=torch.float32, device=device).contiguous()


def _to_cuda_i32(value, shape, device) -> torch.Tensor:
    if value is None:
        return torch.empty(shape, dtype=torch.int32, device=device)
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.int32).contiguous()
    return torch.as_tensor(value, dtype=torch.int32, device=device).contiguous()


@dataclass
class AtmosphereBuffers:
    voxel_ka: torch.Tensor
    voxel_ks: torch.Tensor
    voxel_temp: torch.Tensor
    voxel_kmax: torch.Tensor
    voxel_origin: torch.Tensor
    voxel_spacing: torch.Tensor
    brick: int
    gas_z: torch.Tensor
    gas_ka: torch.Tensor
    gas_ks: torch.Tensor
    gas_temp: torch.Tensor
    mesh_verts: torch.Tensor
    mesh_idx: torch.Tensor
    mesh_albedo: torch.Tensor
    mesh_temp: torch.Tensor
    sun_dir: torch.Tensor
    sun_radiance: float = 1.0
    sun_cos_half: float = 1.0
    spectral_mode: int = SPECTRAL_SW
    wavelength_m: float = 5.0e-7
    mie_omega: float = 0.0
    mie_g: float = 0.85
    path_tmax: float = 1.0e5
    max_bounces: int = 8
    particle_density: torch.Tensor = field(default=None)  # type: ignore
    particle_material: torch.Tensor = field(default=None)  # type: ignore
    particle_radiance: torch.Tensor = field(default=None)  # type: ignore
    particle_shading_normal: torch.Tensor = field(default=None)  # type: ignore

    @classmethod
    def from_numpy(
        cls,
        device: torch.device | str = "cuda",
        voxel_ka: Optional[np.ndarray] = None,
        voxel_ks: Optional[np.ndarray] = None,
        voxel_temp: Optional[np.ndarray] = None,
        voxel_origin=(0.0, 0.0, 0.0),
        voxel_spacing=(1.0, 1.0, 1.0),
        brick: int = 8,
        gas_z: Optional[np.ndarray] = None,
        gas_ka: Optional[np.ndarray] = None,
        gas_ks: Optional[np.ndarray] = None,
        gas_temp: Optional[np.ndarray] = None,
        mesh_verts: Optional[np.ndarray] = None,
        mesh_idx: Optional[np.ndarray] = None,
        mesh_albedo: Optional[np.ndarray] = None,
        mesh_temp: Optional[np.ndarray] = None,
        sun_dir=(0.0, 0.0, 1.0),
        sun_radiance: float = 1.0,
        sun_cos_half: float = 1.0,
        spectral_mode: int = SPECTRAL_SW,
        wavelength_m: float = 5.0e-7,
        mie_omega: float = 0.0,
        mie_g: float = 0.85,
        path_tmax: float = 1.0e5,
        max_bounces: int = 8,
    ) -> "AtmosphereBuffers":
        if voxel_ka is not None:
            ka = np.asarray(voxel_ka, dtype=np.float32)
            ks = np.asarray(voxel_ks if voxel_ks is not None else np.zeros_like(ka), dtype=np.float32)
            vt = np.asarray(
                voxel_temp if voxel_temp is not None else np.full(ka.shape, 288.15, dtype=np.float32),
                dtype=np.float32,
            )
            kmax = build_brick_majorants(ka, ks, brick)
        else:
            ka = np.zeros((0, 0, 0), dtype=np.float32)
            ks = ka
            vt = ka
            kmax = np.zeros((0, 0, 0), dtype=np.float32)

        empty0 = torch.empty(0, dtype=torch.float32, device=device)
        return cls(
            voxel_ka=_to_cuda_f32(ka, (0, 0, 0), device),
            voxel_ks=_to_cuda_f32(ks, (0, 0, 0), device),
            voxel_temp=_to_cuda_f32(vt, (0, 0, 0), device),
            voxel_kmax=_to_cuda_f32(kmax, (0, 0, 0), device),
            voxel_origin=_to_cuda_f32(np.asarray(voxel_origin, dtype=np.float32), (3,), device),
            voxel_spacing=_to_cuda_f32(np.asarray(voxel_spacing, dtype=np.float32), (3,), device),
            brick=int(brick),
            gas_z=_to_cuda_f32(gas_z, (0,), device),
            gas_ka=_to_cuda_f32(gas_ka, (0,), device),
            gas_ks=_to_cuda_f32(gas_ks, (0,), device),
            gas_temp=_to_cuda_f32(gas_temp, (0,), device),
            mesh_verts=_to_cuda_f32(mesh_verts, (0, 3), device),
            mesh_idx=_to_cuda_i32(mesh_idx, (0, 3), device),
            mesh_albedo=_to_cuda_f32(mesh_albedo, (0, 3), device),
            mesh_temp=_to_cuda_f32(mesh_temp, (0,), device),
            sun_dir=_to_cuda_f32(np.asarray(sun_dir, dtype=np.float32), (3,), device),
            sun_radiance=float(sun_radiance),
            sun_cos_half=float(sun_cos_half),
            spectral_mode=int(spectral_mode),
            wavelength_m=float(wavelength_m),
            mie_omega=float(mie_omega),
            mie_g=float(mie_g),
            path_tmax=float(path_tmax),
            max_bounces=int(max_bounces),
            particle_density=torch.empty((0, 12), dtype=torch.float32, device=device),
            particle_material=torch.empty((0, 5), dtype=torch.float32, device=device),
            particle_radiance=empty0,
            particle_shading_normal=torch.empty((0, 3), dtype=torch.float32, device=device),
        )


def trace_atmosphere(tracer_wrapper, ray_ori: torch.Tensor, ray_dir: torch.Tensor, buffers: AtmosphereBuffers, frame_number: int = 0):
    """Launch the copied PTIR pipeline with atmosphere.enabled.

    Returns the same 13-tuple as OptixTracer.trace; ray_pbr is path-traced radiance.
    """
    device = ray_ori.device
    ray_to_world = identity_ray_to_world(device)
    return tracer_wrapper.trace_atmosphere(
        int(frame_number),
        ray_to_world,
        ray_ori.contiguous(),
        ray_dir.contiguous(),
        buffers.particle_density,
        buffers.particle_material,
        buffers.particle_radiance,
        buffers.particle_shading_normal,
        buffers.voxel_ka,
        buffers.voxel_ks,
        buffers.voxel_temp,
        buffers.voxel_kmax,
        buffers.voxel_origin,
        buffers.voxel_spacing,
        int(buffers.brick),
        buffers.gas_z,
        buffers.gas_ka,
        buffers.gas_ks,
        buffers.gas_temp,
        buffers.mesh_verts,
        buffers.mesh_idx,
        buffers.mesh_albedo,
        buffers.mesh_temp,
        buffers.sun_dir,
        float(buffers.sun_radiance),
        float(buffers.sun_cos_half),
        int(buffers.spectral_mode),
        float(buffers.wavelength_m),
        float(buffers.mie_omega),
        float(buffers.mie_g),
        float(buffers.path_tmax),
        int(buffers.max_bounces),
    )
