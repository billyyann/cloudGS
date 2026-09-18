# Voxel cloud + 1D gas atmosphere on the copied PTIR tracer.
# Cloud k is the voxel field only. Gaussians/mesh are surfaces.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


SPECTRAL_SW = 0
SPECTRAL_LW = 1
SPECTRAL_SW_CIE_XYZ = 2
MTL_SPECTRAL_LAMBERT = 0
MTL_RGB_LAMBERT = 1
MTL_SPECTRAL_SPECULAR = 2  # htrdr ssf_specular; same ρ(λ) LUT as Lambert

# Through-atmosphere mean solar-disk luminance (cd/m²).
SUN_LUMINANCE_ATM_CD_M2 = 1.44e9


def xenon_searchlight_params(
    pos,
    target,
    lumens: float = 8.0e4,
    diameter_m: float = 1.0,
    half_angle_deg: float | None = None,
    sun_luminance: float = SUN_LUMINANCE_ATM_CD_M2,
) -> tuple[np.ndarray, dict]:
    """One 2 kW-class xenon searchlight as a Lambert disk in a cone.

    Φ ≈ 8×10⁴ lm, D = 1 m. Half-angle covers the target if omitted.
    L = Φ / (π A sin²α) ≪ L_sun because the aperture and beam are both large.
    Returns (9-vector for C++, info dict).
    """
    pos = np.asarray(pos, np.float64).reshape(3)
    target = np.asarray(target, np.float64).reshape(3)
    delta = target - pos
    dist = float(np.linalg.norm(delta))
    if dist < 1.0:
        raise ValueError("searchlight pos too close to target")
    direction = delta / dist
    if half_angle_deg is None:
        # Cover a 747 (~70 m) with a little spill onto the cloud.
        half_angle_deg = float(np.degrees(np.arctan2(40.0, dist)))
        half_angle_deg = float(np.clip(half_angle_deg, 4.0, 10.0))
    radius = 0.5 * float(diameter_m)
    area = math.pi * radius * radius
    alpha = math.radians(float(half_angle_deg))
    sin2 = max(math.sin(alpha) ** 2, 1e-12)
    luminance = float(lumens) / (math.pi * area * sin2)
    scale = luminance / max(float(sun_luminance), 1.0)
    vec = np.array(
        [
            pos[0],
            pos[1],
            pos[2],
            direction[0],
            direction[1],
            direction[2],
            scale,
            math.cos(alpha),
            radius,
        ],
        dtype=np.float32,
    )
    info = {
        "lumens": float(lumens),
        "diameter_m": float(diameter_m),
        "radius_m": radius,
        "half_angle_deg": float(half_angle_deg),
        "dist_m": dist,
        "luminance_cd_m2": luminance,
        "sun_luminance_cd_m2": float(sun_luminance),
        "spot_sun_scale": scale,
        "I0_cd": luminance * area,
        "E_axis_lux": (luminance * area) / (dist * dist),
    }
    return vec, info


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
    use_extinction: bool = False,
) -> np.ndarray:
    """Per-brick k̂. SW Woodcock (htrdr): max(ks). LW: max(ka+ks)."""
    nz, ny, nx = ka.shape
    b = max(int(brick), 1)
    nbz = (nz + b - 1) // b
    nby = (ny + b - 1) // b
    nbx = (nx + b - 1) // b
    kmax = np.zeros((nbz, nby, nbx), dtype=np.float32)
    ks_a = np.asarray(ks, dtype=np.float32)
    src = ks_a
    if use_extinction:
        src = np.asarray(ka, dtype=np.float32) + ks_a
    for bz in range(nbz):
        for by in range(nby):
            for bx in range(nbx):
                sl = src[bz * b : min((bz + 1) * b, nz), by * b : min((by + 1) * b, ny), bx * b : min((bx + 1) * b, nx)]
                kmax[bz, by, bx] = float(sl.max()) if sl.size else 0.0
    return kmax


def _cie_xbar(lam):
    a = (lam - 442.0) * np.where(lam < 442.0, 0.0624, 0.0374)
    b = (lam - 599.8) * np.where(lam < 599.8, 0.0264, 0.0323)
    c = (lam - 501.1) * np.where(lam < 501.1, 0.0490, 0.0382)
    return 0.362 * np.exp(-0.5 * a * a) + 1.056 * np.exp(-0.5 * b * b) - 0.065 * np.exp(-0.5 * c * c)


def _cie_ybar(lam):
    a = (lam - 568.8) * np.where(lam < 568.8, 0.0213, 0.0247)
    b = (lam - 530.9) * np.where(lam < 530.9, 0.0613, 0.0322)
    return 0.821 * np.exp(-0.5 * a * a) + 0.286 * np.exp(-0.5 * b * b)


def _cie_zbar(lam):
    a = (lam - 437.0) * np.where(lam < 437.0, 0.0845, 0.0278)
    b = (lam - 459.0) * np.where(lam < 459.0, 0.0385, 0.0725)
    return 1.217 * np.exp(-0.5 * a * a) + 0.681 * np.exp(-0.5 * b * b)


def _trap_cmf(lo, hi, fn, n=None):
    if n is None:
        n = int(hi - lo) + 1
    n = max(int(n), 1)
    d = (hi - lo) / n
    x = lo + d * np.arange(n + 1, dtype=np.float64)
    f = fn(x)
    return float(np.sum((f[:-1] + f[1:]) * d * 0.5))


def build_cie_xyz_tables(lambda0_nm: float = 380.0, lambda1_nm: float = 780.0, nbins: int = 400):
    """Same CDF construction as htrdr_ran_wlen_cie_xyz (380-780 nm, 1 nm bins)."""
    band_len = (lambda1_nm - lambda0_nm) / float(nbins)
    fns = (_cie_xbar, _cie_ybar, _cie_zbar)
    pdf = np.zeros((3, nbins), dtype=np.float64)
    rcp = np.zeros(3, dtype=np.float32)
    for ch, fn in enumerate(fns):
        for i in range(nbins):
            lo = lambda0_nm + i * band_len
            hi = min(lo + band_len, lambda1_nm)
            pdf[ch, i] = _trap_cmf(lo, hi, fn)
        s = float(pdf[ch].sum())
        rcp[ch] = np.float32(1.0 / max(s, 1e-30))
        pdf[ch] /= max(s, 1e-30)
    cdf = np.cumsum(pdf, axis=1).astype(np.float32)
    cdf[:, -1] = 1.0
    return {
        "nbins": nbins,
        "lambda0_nm": float(lambda0_nm),
        "dlambda_nm": float(band_len),
        "rcp_integral": rcp,
        "cdf": cdf.reshape(-1),
    }


def htpp_xyz_to_srgb(xyz: np.ndarray, exposure: float = 0.2, white: float | None = None) -> np.ndarray:
    """htpp pp_image: filmic on XYZ, D65*XYZ → sRGB, sRGB gamma. xyz is (H,W,3).
    white: htpp image.white; None → 99.5% Y (htpp default)."""
    x = np.asarray(xyz, dtype=np.float64)
    y = x[..., 1]
    ys = np.sort(y.reshape(-1))
    imax = int(min(len(ys) - 1, max(0, round((len(ys) - 1) * 0.995))))
    ymax_auto = float(ys[imax]) if ys.size else 1.0
    ymax = float(white) if white is not None and white > 0 else ymax_auto
    ymax = max(ymax, 1e-12)
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30

    def tone(v):
        return ((v * (a * v + c * b) + d * e) / (v * (a * v + b) + d * f)) - e / f

    white = tone(ymax * exposure)
    t = tone(np.maximum(x * exposure, 0.0)) / white
    d65_x, d65_y = 0.31271, 0.32902
    d65 = np.array([d65_x / d65_y, 1.0, (1.0 - d65_x - d65_y) / d65_y], dtype=np.float64)
    xyz_d65 = t * d65
    mat = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    rgb = np.clip(xyz_d65 @ mat.T, 0.0, None)

    def gamma(v):
        return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(np.maximum(v, 0.0), 1.0 / 2.4) - 0.055)

    return np.clip(gamma(rgb), 0.0, 1.0)


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
    cie_cdf: torch.Tensor = field(default=None)  # type: ignore
    cie_rcp_integral: torch.Tensor = field(default=None)  # type: ignore
    cie_lambda0_nm: float = 380.0
    cie_dlambda_nm: float = 1.0
    sw_lambda_lo: torch.Tensor = field(default=None)  # type: ignore
    sw_lambda_hi: torch.Tensor = field(default=None)  # type: ignore
    sw_gas_ka: torch.Tensor = field(default=None)  # type: ignore
    sw_gas_ks: torch.Tensor = field(default=None)  # type: ignore
    brdf_lambda_nm: torch.Tensor = field(default=None)  # type: ignore
    brdf_R: torch.Tensor = field(default=None)  # type: ignore
    sw_nquad: torch.Tensor = field(default=None)  # type: ignore
    sw_quad_cdf: torch.Tensor = field(default=None)  # type: ignore
    mie_cs_band: torch.Tensor = field(default=None)  # type: ignore
    mie_ca_band: torch.Tensor = field(default=None)  # type: ignore
    mie_g_nm: torch.Tensor = field(default=None)  # type: ignore
    mie_cs_ref: float = 1.0
    mie_ca_ref: float = 1.0
    voxel_z_iface: torch.Tensor = field(default=None)  # type: ignore
    mesh_mat_id: torch.Tensor = field(default=None)  # type: ignore
    spot_params: torch.Tensor = field(default=None)  # type: ignore
    mie_g_lambda0_nm: float = 380.0
    mie_g_dlambda_nm: float = 0.0  # 0 = host derives 380–780 from LUT length
    lw_lambda_lo: torch.Tensor = field(default=None)  # type: ignore
    lw_lambda_hi: torch.Tensor = field(default=None)  # type: ignore
    lw_gas_ka: torch.Tensor = field(default=None)  # type: ignore
    lw_nquad: torch.Tensor = field(default=None)  # type: ignore
    lw_quad_cdf: torch.Tensor = field(default=None)  # type: ignore
    voxel_x_h2o: torch.Tensor = field(default=None)  # type: ignore
    sw_h2o_ntab: int = 0
    sw_h2o_n: torch.Tensor = field(default=None)  # type: ignore
    sw_h2o_x: torch.Tensor = field(default=None)  # type: ignore
    sw_h2o_ka: torch.Tensor = field(default=None)  # type: ignore
    sw_h2o_ks: torch.Tensor = field(default=None)  # type: ignore
    sw_h2o_ks_max: torch.Tensor = field(default=None)  # type: ignore

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
        voxel_z_iface: Optional[np.ndarray] = None,
    ) -> "AtmosphereBuffers":
        if voxel_ka is not None:
            ka = np.asarray(voxel_ka, dtype=np.float32)
            ks = np.asarray(voxel_ks if voxel_ks is not None else np.zeros_like(ka), dtype=np.float32)
            vt = np.asarray(
                voxel_temp if voxel_temp is not None else np.full(ka.shape, 288.15, dtype=np.float32),
                dtype=np.float32,
            )
            kmax = build_brick_majorants(
                ka, ks, brick, use_extinction=(int(spectral_mode) == SPECTRAL_LW)
            )
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
            cie_cdf=torch.empty(0, dtype=torch.float32, device=device),
            cie_rcp_integral=torch.zeros(3, dtype=torch.float32, device=device),
            cie_lambda0_nm=380.0,
            cie_dlambda_nm=1.0,
            sw_lambda_lo=torch.empty(0, dtype=torch.float32, device=device),
            sw_lambda_hi=torch.empty(0, dtype=torch.float32, device=device),
            sw_gas_ka=torch.empty((0, 0), dtype=torch.float32, device=device),
            sw_gas_ks=torch.empty((0, 0), dtype=torch.float32, device=device),
            brdf_lambda_nm=torch.empty(0, dtype=torch.float32, device=device),
            brdf_R=torch.empty(0, dtype=torch.float32, device=device),
            sw_nquad=torch.empty(0, dtype=torch.int32, device=device),
            sw_quad_cdf=torch.empty(0, dtype=torch.float32, device=device),
            mie_cs_band=torch.empty(0, dtype=torch.float32, device=device),
            mie_ca_band=torch.empty(0, dtype=torch.float32, device=device),
            mie_g_nm=torch.empty(0, dtype=torch.float32, device=device),
            mie_cs_ref=1.0,
            mie_ca_ref=1.0,
            voxel_z_iface=_to_cuda_f32(voxel_z_iface, (0,), device),
            lw_lambda_lo=torch.empty(0, dtype=torch.float32, device=device),
            lw_lambda_hi=torch.empty(0, dtype=torch.float32, device=device),
            lw_gas_ka=torch.empty((0, 0, 0), dtype=torch.float32, device=device),
            lw_nquad=torch.empty(0, dtype=torch.int32, device=device),
            lw_quad_cdf=torch.empty(0, dtype=torch.float32, device=device),
            voxel_x_h2o=torch.empty(0, dtype=torch.float32, device=device),
            sw_h2o_ntab=0,
            sw_h2o_n=torch.empty(0, dtype=torch.int32, device=device),
            sw_h2o_x=torch.empty(0, dtype=torch.float32, device=device),
            sw_h2o_ka=torch.empty(0, dtype=torch.float32, device=device),
            sw_h2o_ks=torch.empty(0, dtype=torch.float32, device=device),
            sw_h2o_ks_max=torch.empty(0, dtype=torch.float32, device=device),
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
        buffers.cie_cdf if buffers.cie_cdf is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.cie_rcp_integral if buffers.cie_rcp_integral is not None else torch.zeros(3, dtype=torch.float32, device=device),
        float(buffers.cie_lambda0_nm),
        float(buffers.cie_dlambda_nm),
        buffers.sw_lambda_lo if buffers.sw_lambda_lo is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.sw_lambda_hi if buffers.sw_lambda_hi is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.sw_gas_ka if buffers.sw_gas_ka is not None else torch.empty((0, 0), dtype=torch.float32, device=device),
        buffers.sw_gas_ks if buffers.sw_gas_ks is not None else torch.empty((0, 0), dtype=torch.float32, device=device),
        buffers.brdf_lambda_nm if buffers.brdf_lambda_nm is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.brdf_R if buffers.brdf_R is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.sw_nquad if buffers.sw_nquad is not None else torch.empty(0, dtype=torch.int32, device=device),
        buffers.sw_quad_cdf if buffers.sw_quad_cdf is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.mie_cs_band if buffers.mie_cs_band is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.mie_ca_band if buffers.mie_ca_band is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.mie_g_nm if buffers.mie_g_nm is not None else torch.empty(0, dtype=torch.float32, device=device),
        float(buffers.mie_cs_ref),
        float(buffers.mie_ca_ref),
        buffers.voxel_z_iface if buffers.voxel_z_iface is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.mesh_mat_id if buffers.mesh_mat_id is not None else torch.empty(0, dtype=torch.int32, device=device),
        buffers.spot_params if buffers.spot_params is not None else torch.empty(0, dtype=torch.float32, device=device),
        float(buffers.mie_g_lambda0_nm),
        float(buffers.mie_g_dlambda_nm),
        buffers.lw_lambda_lo if buffers.lw_lambda_lo is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.lw_lambda_hi if buffers.lw_lambda_hi is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.lw_gas_ka if buffers.lw_gas_ka is not None else torch.empty((0, 0, 0), dtype=torch.float32, device=device),
        buffers.lw_nquad if buffers.lw_nquad is not None else torch.empty(0, dtype=torch.int32, device=device),
        buffers.lw_quad_cdf if buffers.lw_quad_cdf is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.voxel_x_h2o if buffers.voxel_x_h2o is not None else torch.empty(0, dtype=torch.float32, device=device),
        int(buffers.sw_h2o_ntab),
        buffers.sw_h2o_n if buffers.sw_h2o_n is not None else torch.empty(0, dtype=torch.int32, device=device),
        buffers.sw_h2o_x if buffers.sw_h2o_x is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.sw_h2o_ka if buffers.sw_h2o_ka is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.sw_h2o_ks if buffers.sw_h2o_ks is not None else torch.empty(0, dtype=torch.float32, device=device),
        buffers.sw_h2o_ks_max if buffers.sw_h2o_ks_max is not None else torch.empty(0, dtype=torch.float32, device=device),
    )
