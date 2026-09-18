#!/usr/bin/env python3
"""Dump FIRE shahed-camera water OBJ for htrdr MWIR 4000 nm (GPU dump λ).

Same camera/water as SWIR 1600. Spectral is LW, not CIE, not -s sw=.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import fire_scene as fire  # noqa: E402
import mountain_gs_scene as mgs  # noqa: E402
from render_fire_shahed_bands import (  # noqa: E402
    ORTHO_HEIGHT_M,
    ORTHO_Z_M,
    WATER_SIZE_M,
    WATER_T_K,
    pick_mean_column,
    water_quad,
)

OUT = os.path.join(ROOT, "scenes", "fire_gs", "mwir4000")
MRUMTL = os.path.join(OUT, "lambert_rho004.mrumtl")
GPU_LAMBDA_NM = 4000.0


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("usemtl air:water\n")
        for v in verts:
            f.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    with open(MRUMTL, "w") as f:
        f.write("wavelengths    4\n")
        f.write("   1600.0   lambertian  0.04\n")
        f.write("   3000.0   lambertian  0.04\n")
        f.write("   4000.0   lambertian  0.04\n")
        f.write("   5000.0   lambertian  0.04\n")
    vox = mgs.load_htcp_voxels(fire.DEFAULT_HTCP, wavelength_nm=550.0)
    place = pick_mean_column(vox)
    cx, cy = float(place["x"]), float(place["y"])
    water = water_quad(cx, cy, WATER_SIZE_M, (0.04, 0.04, 0.04), WATER_T_K)
    obj = os.path.join(OUT, "water.obj")
    mtls = os.path.join(OUT, "water.mtls")
    write_obj(obj, water["verts"], water["idx"])
    with open(mtls, "w") as f:
        f.write("air   none\n")
        f.write(f'water "{os.path.abspath(MRUMTL)}" {WATER_T_K:.0f}\n')
    rec = {
        "place": place,
        "camera": {
            "pos": [cx, cy, ORTHO_Z_M],
            "tgt": [cx, cy, 0.0],
            "up": [0.0, 1.0, 0.0],
            "height_m": ORTHO_HEIGHT_M,
            "film": [512, 512],
        },
        "sun": {"az_deg": fire.SUN_AZ_DEG, "el_deg": fire.SUN_EL_DEG, "used": False},
        "water": {"size_m": WATER_SIZE_M, "rho": 0.04, "T_K": WATER_T_K},
        "spectral": {
            "lw_nm": [GPU_LAMBDA_NM, GPU_LAMBDA_NM + 1.0],
            "gpu_lambda_nm": GPU_LAMBDA_NM,
            "note": "IR MWIR; htrdr -s lw=4000,4001; GPU SPECTRAL_LW dump 4000 nm; not CIE/SW",
        },
        "htcp": fire.DEFAULT_HTCP,
        "obj": obj,
        "mtls": mtls,
    }
    jpath = os.path.join(OUT, "place.json")
    with open(jpath, "w") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
    cam = os.path.join(OUT, "camera.env")
    with open(cam, "w") as f:
        f.write(f"cx={cx}\ncy={cy}\ncz={ORTHO_Z_M}\nheight={ORTHO_HEIGHT_M}\n")
    print(
        f"MWIR 4000 nm place xy=({cx:.1f},{cy:.1f})  tau550={place['tau']:.2f}  "
        f"cam z={ORTHO_Z_M:.0f} height={ORTHO_HEIGHT_M:.0f}  wrote {jpath}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
