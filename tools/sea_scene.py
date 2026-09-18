#!/usr/bin/env python3
"""BOMEX sea-cloud assets for GPU path tracing (htrdr sea.sh).

Host data only: voxels, mesh, camera, sun, blue_water ρ(λ). The tracer is
scene-agnostic; Dirac specular is a material type (mat_id=2), not a “sea”
special case in the kernel.

Does not write into scenes/mountain_gs/.
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

SPK = mgs.SPK
DEFAULT_OUT = os.path.join(ROOT, "scenes", "sea_gs")
DEFAULT_HTCP = os.path.join(SPK, "clouds", "L12km.1.BOMEX.005.htcp")
DEFAULT_OBJ = os.path.join(SPK, "models", "sea_3m.obj")
DEFAULT_WATER = os.path.join(SPK, "materials", "legacy", "blue_water.mrumtl")
WATER_T_K = 290.0
# ATM_MTL_SPECTRAL_SPECULAR — append-only enum in atmosphere.h
MTL_SPECTRAL_SPECULAR = 2


def load_mrumtl(path: str) -> dict:
    """Parse htrdr mrumtl knots (lambertian or specular)."""
    lams = []
    rs = []
    kind = None
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("lambertian", "specular"):
                kind = parts[1]
                lams.append(float(parts[0]))
                rs.append(float(parts[2]))
    if not lams:
        raise RuntimeError(f"no BRDF knots in {path}")
    print(
        f"{os.path.basename(path)} {kind} {len(lams)} knots  "
        f"λ=[{lams[0]:.0f},{lams[-1]:.0f}] nm  R=[{min(rs):.4f},{max(rs):.4f}]"
    )
    if kind == "specular" and lams[-1] >= 1.0e5:
        print(
            "  note: pack zeros ρ after ~2.08 μm; not seawater Fresnel. "
            "SWIR/MWIR/LWIR windows must not reuse this LUT as-is."
        )
    return {
        "lambda_nm": np.asarray(lams, np.float32),
        "R": np.asarray(rs, np.float32),
        "kind": kind or "unknown",
    }


def load_sea_grid(path: str) -> np.ndarray:
    """sea_3m.obj puts `usemtl` before vertices; do not reuse mountain's early break."""
    pts = np.loadtxt(
        path,
        usecols=(1, 2, 3),
        comments=("u", "f", "#", "s", "o", "g", "m"),
        dtype=np.float32,
    )
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    n = int(round(math.sqrt(pts.shape[0])))
    if n * n != pts.shape[0]:
        raise RuntimeError(f"vertex count {pts.shape[0]} is not a square grid")
    print(f"loaded {n} x {n} heightfield from {path}")
    return pts.reshape(n, n, 3)


def heightfield_to_sea_mesh(grid: np.ndarray) -> dict:
    """Same grid triangulation as mountain, specular water at 290 K."""
    n = int(grid.shape[0])
    verts = np.ascontiguousarray(grid.reshape(-1, 3), dtype=np.float32)
    n1 = n - 1
    iy = np.arange(n1, dtype=np.int32)[:, None]
    ix = np.arange(n1, dtype=np.int32)[None, :]
    i00 = iy * n + ix
    i10 = i00 + 1
    i01 = i00 + n
    i11 = i01 + 1
    t0 = np.stack([i00, i10, i11], axis=-1)
    t1 = np.stack([i00, i11, i01], axis=-1)
    idx = np.stack([t0, t1], axis=2).reshape(-1, 3).astype(np.int32)
    alb = np.full((idx.shape[0], 3), 0.04, np.float32)
    temp = np.full((idx.shape[0],), WATER_T_K, np.float32)
    mat_id = np.full((idx.shape[0],), MTL_SPECTRAL_SPECULAR, np.int32)
    print(
        f"sea mesh {n}x{n} verts={verts.shape[0]} tris={idx.shape[0]} "
        f"z=[{verts[:, 2].min():.2f},{verts[:, 2].max():.2f}] T={WATER_T_K:.0f} K specular"
    )
    return {"verts": verts, "idx": idx, "albedo": alb, "temp": temp, "mat_id": mat_id}


def ensure_sea_mesh(outdir: str = DEFAULT_OUT, obj_path: str = DEFAULT_OBJ, rebuild: bool = False) -> dict:
    cache = os.path.join(outdir, "sea_mesh.npz")
    if (not rebuild) and os.path.isfile(cache):
        data = np.load(cache)
        mesh = {k: data[k] for k in ("verts", "idx", "albedo", "temp")}
        mesh["mat_id"] = data["mat_id"] if "mat_id" in data.files else np.full(
            (mesh["idx"].shape[0],), MTL_SPECTRAL_SPECULAR, np.int32
        )
        print(f"sea mesh from cache {cache}  verts={mesh['verts'].shape[0]} tris={mesh['idx'].shape[0]}")
        return mesh
    grid = load_sea_grid(obj_path)
    mesh = heightfield_to_sea_mesh(grid)
    os.makedirs(outdir, exist_ok=True)
    np.savez(cache, **mesh)
    print(f"wrote {cache}")
    return mesh


def ensure_assets(
    outdir: str = DEFAULT_OUT,
    rebuild: bool = False,
    htcp_path: str = DEFAULT_HTCP,
    ecrad_path: str = mgs.DEFAULT_ECRAD,
    obj_path: str = DEFAULT_OBJ,
) -> dict:
    os.makedirs(outdir, exist_ok=True)
    voxels_bin = os.path.join(outdir, "bomex_voxels.bin")
    gas_npz = os.path.join(outdir, "ecrad_gas_1d.npz")
    scene = os.path.join(outdir, "scene.json")

    need_vox = rebuild or not os.path.isfile(voxels_bin)
    if not need_vox:
        voxels = mgs.load_voxel_bin(voxels_bin)
        if voxels.get("z_iface") is None:
            need_vox = True
    if need_vox:
        voxels = mgs.load_htcp_voxels(htcp_path)
        mgs.save_voxel_bin(voxels_bin, voxels)
        with open(os.path.join(outdir, "bomex_voxels.mie.json"), "w") as f:
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

    mesh = ensure_sea_mesh(outdir=outdir, obj_path=obj_path, rebuild=rebuild)
    water = load_mrumtl(DEFAULT_WATER)
    if not os.path.isfile(scene) or rebuild:
        data = {
            "name": "sea_gs",
            "match": "sea.sh: BOMEX -r, sea_3m.obj -R, blue_water specular@290K",
            "camera": {
                "pos": [2224.91, 3684.71, 1492.95],
                "tgt": [8208.71, 4471.54, -393.82],
                "up": [0, 0, 1],
                "fov": 70,
            },
            "sun": {"azimuth_deg": 0, "elevation_deg": 50},
            "volume": {"path": "bomex_voxels.bin", "primitive": "voxel"},
            "gas": {"path": "ecrad_gas_1d.npz"},
            "ground": {"path": "sea_mesh.npz", "material": "blue_water"},
        }
        with open(scene, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"wrote {scene}")

    return {
        "voxels_bin": voxels_bin,
        "gas_npz": gas_npz,
        "scene": scene,
        "voxels": voxels,
        "gas": gas,
        "mesh": mesh,
        "water": water,
    }


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Build sea_gs BOMEX + sea_3m assets")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--htcp", default=DEFAULT_HTCP)
    p.add_argument("--obj", default=DEFAULT_OBJ)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args(argv)
    ensure_assets(outdir=args.outdir, rebuild=args.rebuild, htcp_path=args.htcp, obj_path=args.obj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
