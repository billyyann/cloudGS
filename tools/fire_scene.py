#!/usr/bin/env python3
"""FIRE marine stratocumulus over seawater (host assets).

Clouds: L25.1.FIRE.012. Ground: sea_3m swell + JONSWAP chop on a 0.5 m
tile (blue_water Dirac in htrdr; GPU Fresnel+body). Camera is unchanged.
Does not write into scenes/mountain_gs/ or scenes/sea_gs/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_local_lib = os.path.join(ROOT, "local", "lib")
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _local_lib not in _ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = _local_lib + (":" + _ld if _ld else "")
sys.path.insert(0, os.path.dirname(__file__))

import mountain_gs_scene as mgs  # noqa: E402
import sea_scene as sea  # noqa: E402

SPK = mgs.SPK
DEFAULT_OUT = os.path.join(ROOT, "scenes", "fire_gs")
DEFAULT_HTCP = os.path.join(SPK, "clouds", "L25.1.FIRE.012.htcp")
DEFAULT_OBJ = sea.DEFAULT_OBJ
DEFAULT_WATER = os.path.join(SPK, "materials", "legacy", "blue_water.mrumtl")

# Pack L25_FIRE nadir: 25 km tile fills the frame (full Sc deck).
CAM_UP = np.array([0.0, 0.0, 1.0], np.float32)
CAM_FOV = 70.0
SUN_AZ_DEG = 340.0
SUN_EL_DEG = 25.0
ORTHO_CENTER = np.array([12500.0, 12500.0, 80000.0], np.float32)
ORTHO_HEIGHT = 25000.0

# Boat camera unchanged. JONSWAP tile is placed so this point sits inside it.
_CAM_LOOK_XY = 420.0
CAM_POS = np.array([1600.0, 5800.0, 18.0], np.float32)
CAM_TGT = np.array(
    [
        CAM_POS[0] + _CAM_LOOK_XY * math.cos(math.radians(SUN_AZ_DEG)),
        CAM_POS[1] + _CAM_LOOK_XY * math.sin(math.radians(SUN_AZ_DEG)),
        0.0,
    ],
    np.float32,
)

SEA_TILE_N = 4096
SEA_TILE_DX = 0.5
SEA_TILE_U10 = 8.0
SEA_TILE_LAM_MAX = 20.0
SEA_TILE_SEED = 1
# World origin of the periodic FFT tile; camera (1600, 5800) is inside.
SEA_TILE_ORIGIN = np.array([600.0, 4800.0], np.float64)
# Leaving body: dark blue, not 0.11 gray-green. Specular is Schlick in the kernel.
WATER_BODY_ALBEDO = np.array([0.012, 0.016, 0.055], np.float32)


def extend_voxels_y(voxels: dict, y1_world: float) -> dict:
    """Empty cells north of the LES so a camera there does not wrap into the deck."""
    ks = np.asarray(voxels["ks"])
    nz, ny, nx = ks.shape
    dy = float(voxels["spacing"][1])
    oy = float(voxels["origin"][1])
    ny2 = int(np.ceil((float(y1_world) - oy) / dy))
    if ny2 <= ny:
        return voxels
    ks2 = np.zeros((nz, ny2, nx), np.float32)
    ka2 = np.zeros_like(ks2)
    tmp = np.full((nz, ny2, nx), 290.0, np.float32)
    ks2[:, :ny, :] = voxels["ks"]
    ka2[:, :ny, :] = voxels["ka"]
    tmp[:, :ny, :] = voxels["temp"]
    out = dict(voxels)
    out["ks"], out["ka"], out["temp"] = ks2, ka2, tmp
    if "x_h2o" in voxels:
        x2 = np.full((nz, ny2, nx), -1.0, np.float32)
        x2[:, :ny, :] = voxels["x_h2o"]
        out["x_h2o"] = x2
    print(f"extend FIRE Y {ny} → {ny2}  y=[{oy:.0f},{oy + ny2 * dy:.0f}]")
    return out


def _bilinear_z(swell: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample sea_3m heightfield at world (x, y); clamp to the swell AABB."""
    x0 = float(swell[0, 0, 0])
    y0 = float(swell[0, 0, 1])
    dx = float(swell[0, 1, 0] - swell[0, 0, 0])
    dy = float(swell[1, 0, 1] - swell[0, 0, 1])
    ny, nx = swell.shape[0], swell.shape[1]
    fx = (x - x0) / dx
    fy = (y - y0) / dy
    ix0 = np.clip(np.floor(fx).astype(np.int32), 0, nx - 2)
    iy0 = np.clip(np.floor(fy).astype(np.int32), 0, ny - 2)
    tx = np.clip(fx - ix0, 0.0, 1.0)
    ty = np.clip(fy - iy0, 0.0, 1.0)
    z00 = swell[iy0, ix0, 2]
    z10 = swell[iy0, ix0 + 1, 2]
    z01 = swell[iy0 + 1, ix0, 2]
    z11 = swell[iy0 + 1, ix0 + 1, 2]
    return (1.0 - ty) * ((1.0 - tx) * z00 + tx * z10) + ty * ((1.0 - tx) * z01 + tx * z11)


def jonswap_chop(
    n: int,
    dx: float,
    u10: float,
    wind_az_deg: float,
    lam_min: float,
    lam_max: float,
    seed: int,
) -> np.ndarray:
    """Tessendorf IFFT of a JONSWAP/Phillips directional spectrum, band-limited."""
    g = 9.81
    phil_a = 0.0081
    gamma = 3.3
    kp = 0.86 * g / max(u10 * u10, 1e-6)
    omega_p = math.sqrt(g * kp)
    L = u10 * u10 / g
    wx = math.cos(math.radians(wind_az_deg))
    wy = math.sin(math.radians(wind_az_deg))
    k1d = 2.0 * math.pi * np.fft.fftfreq(n, d=dx).astype(np.float64)
    kx, ky = np.meshgrid(k1d, k1d, indexing="xy")
    k = np.hypot(kx, ky)
    k_safe = np.maximum(k, 1e-12)
    omega = np.sqrt(g * k_safe)
    sigma = np.where(k < kp, 0.07, 0.09)
    r = np.exp(-((omega - omega_p) ** 2) / (2.0 * sigma * sigma * omega_p * omega_p))
    phil = phil_a * np.exp(-1.0 / np.maximum(k_safe * L, 1e-12) ** 2) / (k_safe ** 4)
    spec = phil * (gamma ** r)
    mu = (kx * wx + ky * wy) / k_safe
    spec = spec * np.where(mu > 0.0, mu * mu, 0.0)
    kmin = 2.0 * math.pi / lam_max
    kmax = 2.0 * math.pi / lam_min
    spec = np.where((k >= kmin) & (k <= kmax), spec, 0.0)
    spec[0, 0] = 0.0
    dk = 2.0 * math.pi / (n * dx)
    amp = np.sqrt(np.maximum(spec, 0.0) * dk * dk / 2.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    gaus = rng.standard_normal((n, n)).astype(np.float32) + 1j * rng.standard_normal((n, n)).astype(np.float32)
    h = (amp * gaus).astype(np.complex64)
    ii = (-np.arange(n)) % n
    h = 0.5 * (h + np.conj(h[np.ix_(ii, ii)]))
    h[0, 0] = 0
    eta = np.fft.ifft2(h).real * float(n * n)
    hs = 4.0 * float(eta.std())
    print(
        f"JONSWAP chop n={n} dx={dx:g} U10={u10:g} λ=[{lam_min:g},{lam_max:g}] "
        f"Hs={hs:.3f} m  z=[{eta.min():.3f},{eta.max():.3f}]"
    )
    return eta.astype(np.float32)


def build_fire_sea_grid(obj_path: str) -> np.ndarray:
    """Native sea_3m swell on a 0.5 m JONSWAP tile that contains the boat camera."""
    swell = sea.load_sea_grid(obj_path)
    n = int(SEA_TILE_N)
    dx = float(SEA_TILE_DX)
    ox, oy = float(SEA_TILE_ORIGIN[0]), float(SEA_TILE_ORIGIN[1])
    xs = ox + np.arange(n, dtype=np.float64) * dx
    ys = oy + np.arange(n, dtype=np.float64) * dx
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    z = _bilinear_z(swell, xx, yy)
    del swell
    lam_min = 2.0 * dx
    z = z + jonswap_chop(n, dx, SEA_TILE_U10, SUN_AZ_DEG, lam_min, SEA_TILE_LAM_MAX, SEA_TILE_SEED)
    grid = np.empty((n, n, 3), np.float32)
    grid[:, :, 0] = xx.astype(np.float32)
    grid[:, :, 1] = yy.astype(np.float32)
    grid[:, :, 2] = z.astype(np.float32)
    print(
        f"FIRE sea tile origin=({ox:.0f},{oy:.0f}) L={n * dx:.0f} m  "
        f"z=[{float(grid[:,:,2].min()):.2f},{float(grid[:,:,2].max()):.2f}]"
    )
    return grid


def write_fire_sea_obj(path: str, verts: np.ndarray, idx: np.ndarray, usemtl: str = "air:blue_water") -> None:
    faces = np.ascontiguousarray(idx, dtype=np.int32) + 1
    with open(path, "w", buffering=1024 * 1024) as f:
        f.write(f"usemtl {usemtl}\n")
        np.savetxt(f, verts, fmt="v %.5f %.5f %.5f")
        np.savetxt(f, faces, fmt="f %d %d %d")
    print(f"wrote {path}  verts={verts.shape[0]} faces={idx.shape[0]}")


def ensure_assets(
    outdir: str = DEFAULT_OUT,
    rebuild: bool = False,
    htcp_path: str = DEFAULT_HTCP,
    ecrad_path: str = mgs.DEFAULT_ECRAD,
    obj_path: str = DEFAULT_OBJ,
) -> dict:
    os.makedirs(outdir, exist_ok=True)
    voxels_bin = os.path.join(outdir, "fire_voxels.bin")
    gas_npz = os.path.join(outdir, "ecrad_gas_1d.npz")
    mesh_npz = os.path.join(outdir, "sea_mesh.npz")
    scene = os.path.join(outdir, "scene.json")

    need_vox = rebuild or not os.path.isfile(voxels_bin)
    if not need_vox:
        voxels = mgs.load_voxel_bin(voxels_bin)
        zi = voxels.get("z_iface")
        if zi is None:
            need_vox = True
        else:
            zia = np.asarray(zi, np.float64)
            if (
                zia.size < 2
                or (not np.isfinite(zia).all())
                or float(zia[-1]) <= float(zia[0])
            ):
                print("FIRE voxels have invalid z_iface; rebuilding")
                need_vox = True
            elif int(voxels["ks"].shape[1]) <= 511:
                print("FIRE voxels lack north sea strip; rebuilding")
                need_vox = True
    if need_vox:
        voxels = extend_voxels_y(mgs.load_htcp_voxels(htcp_path), y1_world=30000.0)
        mgs.save_voxel_bin(voxels_bin, voxels)
        with open(os.path.join(outdir, "fire_voxels.mie.json"), "w") as f:
            json.dump({"Cs_source": "htmie", "htcp": os.path.basename(htcp_path)}, f)

    if rebuild or not os.path.isfile(gas_npz):
        gas = mgs.load_ecrad_gas_1d(ecrad_path)
        mgs.save_gas_npz(gas_npz, gas)
    else:
        gas = mgs.load_gas_npz(gas_npz)
        print(
            f"gas from cache {gas_npz}  nlayers={gas['ka'].shape[0]} "
            f"z=[{gas['z'][0]:.1f},{gas['z'][-1]:.1f}]"
        )

    ox, oy = float(voxels["origin"][0]), float(voxels["origin"][1])
    nx, ny = int(voxels["ks"].shape[2]), int(voxels["ks"].shape[1])
    dx, dy = float(voxels["spacing"][0]), float(voxels["spacing"][1])
    x1, y1 = ox + nx * dx, oy + ny * dy
    obj_out = os.path.join(outdir, "sea_chop.obj")
    need_mesh = rebuild or not os.path.isfile(mesh_npz)
    if not need_mesh:
        data = np.load(mesh_npz)
        mesh = {k: data[k] for k in ("verts", "idx", "albedo", "temp", "mat_id")}
        nvert = int(mesh["verts"].shape[0])
        nside = int(round(math.sqrt(nvert)))
        dx_mesh = float(mesh["verts"][1, 0] - mesh["verts"][0, 0]) if nvert > 1 else 99.0
        xmin = float(mesh["verts"][:, 0].min())
        if nside < 3000 or abs(dx_mesh - SEA_TILE_DX) > 0.1 or abs(xmin - float(SEA_TILE_ORIGIN[0])) > 50.0:
            print(
                f"sea mesh is not {SEA_TILE_N}² Δx={SEA_TILE_DX:g} tile "
                f"(n={nside} dx={dx_mesh:.3f} xmin={xmin:.1f}); rebuilding"
            )
            need_mesh = True
        else:
            print(
                f"sea mesh from cache {mesh_npz}  tris={mesh['idx'].shape[0]}  "
                f"dx={dx_mesh:.3f} zmax={float(mesh['verts'][:, 2].max()):.2f}"
            )
    if need_mesh:
        mesh = sea.heightfield_to_sea_mesh(build_fire_sea_grid(obj_path))
        mesh["albedo"][:] = WATER_BODY_ALBEDO
        np.savez(mesh_npz, **mesh)
        print(f"wrote {mesh_npz}  JONSWAP Δx={SEA_TILE_DX:g} + sea_3m swell, body albedo")
        write_fire_sea_obj(obj_out, mesh["verts"], mesh["idx"])
    else:
        if not np.allclose(mesh["albedo"][0], WATER_BODY_ALBEDO, atol=1e-4):
            mesh["albedo"][:] = WATER_BODY_ALBEDO
            np.savez(mesh_npz, **mesh)
            print(f"updated sea body albedo → {WATER_BODY_ALBEDO.tolist()}")
        if not os.path.isfile(obj_out):
            write_fire_sea_obj(obj_out, mesh["verts"], mesh["idx"])

    water = sea.load_mrumtl(DEFAULT_WATER)
    data = {
        "name": "fire_gs",
        "match": "FIRE full-deck ortho; boat look 340°, JONSWAP 0.5 m tile + sea_3m swell",
        "camera": {
            "pos": CAM_POS.tolist(),
            "tgt": CAM_TGT.tolist(),
            "up": CAM_UP.tolist(),
            "fov": CAM_FOV,
        },
        "ortho": {
            "center": ORTHO_CENTER.tolist(),
            "height": float(ORTHO_HEIGHT),
        },
        "sun": {"azimuth_deg": SUN_AZ_DEG, "elevation_deg": SUN_EL_DEG},
        "volume": {"path": "fire_voxels.bin", "primitive": "voxel"},
        "gas": {"path": "ecrad_gas_1d.npz"},
        "ground": {"path": "sea_mesh.npz", "material": "blue_water"},
    }
    with open(scene, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"wrote {scene}")

    ke = voxels["ks"] + voxels["ka"]
    zi = voxels.get("z_iface")
    if zi is not None:
        dz = np.diff(np.asarray(zi, np.float64))
        tau = (ke.reshape(ke.shape[0], -1) * dz[:, None]).sum(0)
    else:
        tau = ke.reshape(ke.shape[0], -1).sum(0) * float(voxels["spacing"][2])
    cloudy = tau > 0.3
    print(
        f"FIRE column τ mean={tau.mean():.2f}  frac(τ>0.3)={cloudy.mean():.3f}  "
        f"cloudy med/p90/max="
        f"{(np.median(tau[cloudy]) if cloudy.any() else 0):.1f}/"
        f"{(np.percentile(tau[cloudy], 90) if cloudy.any() else 0):.1f}/"
        f"{tau.max():.1f}"
    )
    return {
        "voxels_bin": voxels_bin,
        "gas_npz": gas_npz,
        "scene": scene,
        "voxels": voxels,
        "gas": gas,
        "mesh": mesh,
        "water": water,
        "xy": (ox, x1, oy, y1),
    }


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Build fire_gs FIRE Sc + seawater assets")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--htcp", default=DEFAULT_HTCP)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args(argv)
    ensure_assets(outdir=args.outdir, rebuild=args.rebuild, htcp_path=args.htcp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
