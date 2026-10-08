#!/usr/bin/env python3
"""One FIRE SWIR stare clip: 10 s, 5 fps, 512x512, no supersampling.

The camera stays fixed. An-124 flies a straight 220 m/s chord that is still
inside the focal plane at 10 s. The HTCP box translates at the scene wind.
Each frame is a cloudy / clear pair. Clear keeps the 1D gas. Resumable.
Does not change the OptiX kernel.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "gpu"))
sys.path.insert(0, os.path.dirname(__file__))

_local_lib = os.path.join(ROOT, "local", "lib")
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _local_lib not in _ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = _local_lib + (":" + _ld if _ld else "")

import aircraft_mesh as acmesh  # noqa: E402
import fire_scene as fire  # noqa: E402
import mountain_gs_scene as mgs  # noqa: E402
import render_fire_shahed_bands as rf  # noqa: E402
import render_fire_swir_gate_four as gate  # noqa: E402

OUTDIR_DEFAULT = os.path.join(ROOT, "scenes", "fire_gs", "dit_stare10s")
FPS = 5.0
DURATION_S = 10.0
N_FRAMES = int(FPS * DURATION_S)
V_MPS = 220.0
MIN_CHORD_M = V_MPS * DURATION_S


def _crossings(theta: float, impact_m: float, half_m: float) -> tuple[float, float]:
    """Times (meters along the heading) where the line meets the film square."""
    c, s = math.cos(theta), math.sin(theta)
    ts = []
    if abs(c) > 1e-8:
        for x in (-half_m, half_m):
            t = (x + impact_m * s) / c
            y = impact_m * c + t * s
            if -half_m - 1e-4 <= y <= half_m + 1e-4:
                ts.append(t)
    if abs(s) > 1e-8:
        for y in (-half_m, half_m):
            t = (y - impact_m * c) / s
            x = -impact_m * s + t * c
            if -half_m - 1e-4 <= x <= half_m + 1e-4:
                ts.append(t)
    uniq = []
    for t in sorted(ts):
        if not uniq or abs(t - uniq[-1]) > 1e-3:
            uniq.append(t)
    if len(uniq) < 2:
        return 0.0, 0.0
    return uniq[0], uniq[-1]


def sample_pass(half_m: float, rng: np.random.Generator) -> dict:
    """Straight pass. Chord is at least 10 s of cruise, so the clip stays inside."""
    for _ in range(10000):
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        bmax = half_m * (abs(math.cos(theta)) + abs(math.sin(theta)))
        impact = float(rng.uniform(-bmax, bmax))
        t0, t1 = _crossings(theta, impact, half_m)
        if t1 - t0 >= MIN_CHORD_M:
            return {"heading_rad": theta, "impact_m": impact, "t_entry_m": t0, "t_exit_m": t1}
    raise RuntimeError("no chord of at least 2200 m")


def aircraft_xy(clip: dict, tau_s: float, center_xy: tuple[float, float]) -> tuple[float, float]:
    theta = clip["heading_rad"]
    c, s = math.cos(theta), math.sin(theta)
    along = clip["t_entry_m"] + V_MPS * tau_s
    x = center_xy[0] - clip["impact_m"] * s + along * c
    y = center_xy[1] + clip["impact_m"] * c + along * s
    return x, y


def project_rc(x: float, y: float, cx: float, cy: float, film: float, n: int) -> tuple[float, float]:
    col = (x - cx) / film * n + n / 2.0 - 0.5
    row = (0.5 - (y - cy) / film) * n - 0.5
    return row, col


def wind_vec() -> tuple[float, float]:
    az = math.radians(fire.SUN_AZ_DEG)
    return fire.SEA_TILE_U10 * math.cos(az), fire.SEA_TILE_U10 * math.sin(az)


def main() -> int:
    p = argparse.ArgumentParser(description="One 10 s FIRE SWIR stare clip, cloudy/clear pairs")
    p.add_argument("--outdir", default=OUTDIR_DEFAULT)
    p.add_argument("--spp", type=int, default=1024)
    p.add_argument("--fps", type=float, default=FPS)
    p.add_argument("--frames", type=int, default=N_FRAMES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--brick", type=int, default=8)
    p.add_argument("--max-bounces", type=int, default=256)
    p.add_argument("--track-only", action="store_true")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    geo = gate.stare_geometry()
    n = int(geo["nx"])
    film = float(geo["film_w_m"])
    half = 0.5 * film
    cx, cy = gate.PLACE_X, gate.PLACE_Y
    rng = np.random.default_rng(int(args.seed))
    clip = sample_pass(half, rng)
    wx, wy = wind_vec()
    rows = []
    for i in range(int(args.frames)):
        tau = i / float(args.fps)
        x, y = aircraft_xy(clip, tau, (cx, cy))
        row, col = project_rc(x, y, cx, cy, film, n)
        rows.append({
            "frame": i,
            "t_s": tau,
            "x_m": x,
            "y_m": y,
            "z_m": rf.AC_Z_M,
            "heading_rad": clip["heading_rad"],
            "row": row,
            "col": col,
            "cloud_dx_m": wx * tau,
            "cloud_dy_m": wy * tau,
        })
    meta = {
        "fps": float(args.fps),
        "n_frames": int(args.frames),
        "duration_s": (int(args.frames) - 1) / float(args.fps),
        "spp": int(args.spp),
        "nx": n,
        "ny": n,
        "ss": 1,
        "H_m": float(geo["H_m"]),
        "film_m": film,
        "stare_xy_m": [cx, cy],
        "v_mps": V_MPS,
        "heading_rad": clip["heading_rad"],
        "impact_m": clip["impact_m"],
        "chord_m": clip["t_exit_m"] - clip["t_entry_m"],
        "t_entry_m": clip["t_entry_m"],
        "t_exit_m": clip["t_exit_m"],
        "wind_mps": [wx, wy],
        "wind_note": "HTCP origin xy += wind * t. Scene U10 and sun azimuth. No kernel wrap.",
        "lambda_nm": 1600.0,
        "frames": rows,
    }
    with open(os.path.join(args.outdir, "clip.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(
        f"heading={math.degrees(clip['heading_rad']):.1f} deg  "
        f"chord={meta['chord_m']:.0f} m  frames={len(rows)}  "
        f"entry=({rows[0]['x_m']:.0f},{rows[0]['y_m']:.0f})  "
        f"last=({rows[-1]['x_m']:.0f},{rows[-1]['y_m']:.0f})"
    )
    if args.track_only:
        print("wrote", os.path.join(args.outdir, "clip.json"))
        return 0

    import torch

    lam = 1600.0
    mie = mgs.fetch_mie(lam)
    vox = mgs.load_htcp_voxels(fire.DEFAULT_HTCP, wavelength_nm=lam)
    gas = mgs.load_ecrad_gas_1d(wavelength_m=lam * 1.0e-9)
    sw_tab = {
        "vis": mgs.load_ecrad_sw_visible(),
        "mono": mgs.load_ecrad_sw_visible(lambda0_nm=lam, lambda1_nm=lam),
    }
    lw = mgs.load_ecrad_lw()
    origin0 = np.asarray(vox["origin"], np.float64).copy()
    band = gate.band_swir()
    band["spp"] = int(args.spp)
    az = math.radians(fire.SUN_AZ_DEG)
    el = math.radians(fire.SUN_EL_DEG)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    water = rf.water_quad(cx, cy, float(geo["water_m"]), rf.IR_WATER_ALBEDO, rf.WATER_T_K)
    cam = np.array([cx, cy, geo["H_m"]], np.float32)
    tracer, device = gate.load_plugin()
    ac0 = rf.apply_swir_airframe(
        rf.set_airframe_temps(acmesh.make_an124_coarse_mesh((cx, cy, rf.AC_Z_M), heading_rad=0.0))
    )
    mesh0 = mgs.concat_meshes(water, ac0)
    buf_cloud = rf.make_buffers(
        device, vox, gas, mesh0, band, sun, sw_tab, lw, None,
        args.brick, args.max_bounces, mie["g"], mie, no_clouds=False,
    )
    buf_clear = rf.make_buffers(
        device, vox, gas, mesh0, band, sun, sw_tab, lw, None,
        args.brick, args.max_bounces, mie["g"], mie, no_clouds=True,
    )
    frames_dir = os.path.join(args.outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    done = []
    for rec in rows:
        fdir = os.path.join(frames_dir, f"{rec['frame']:04d}")
        os.makedirs(fdir, exist_ok=True)
        cloudy_path = os.path.join(fdir, "cloudy.npy")
        clear_path = os.path.join(fdir, "clear.npy")
        if os.path.isfile(cloudy_path) and os.path.isfile(clear_path):
            print(f"frame {rec['frame']:04d} exists")
            done.append(rec)
            continue
        x, y = rec["x_m"], rec["y_m"]
        ac = rf.apply_swir_airframe(
            rf.set_airframe_temps(
                acmesh.make_an124_coarse_mesh((x, y, rf.AC_Z_M), heading_rad=clip["heading_rad"])
            )
        )
        mesh = mgs.concat_meshes(water, ac)
        shifted = origin0.copy()
        shifted[0] += rec["cloud_dx_m"]
        shifted[1] += rec["cloud_dy_m"]
        buf_cloud.voxel_origin.copy_(torch.tensor(shifted, dtype=torch.float32, device=device))
        rf.attach_mesh(buf_cloud, mesh, device)
        rf.attach_mesh(buf_clear, mesh, device)
        print(f"-- frame {rec['frame']:04d} t={rec['t_s']:.1f}s xy=({x:.0f},{y:.0f})")
        if not os.path.isfile(cloudy_path):
            img = rf.render_ortho(
                tracer, buf_cloud, cam, film, n, n, int(args.spp), device,
                max(int(args.chunk), 1), int(args.seed) + rec["frame"] * 2,
            )
            np.save(cloudy_path, img.astype(np.float32))
            del img
        if not os.path.isfile(clear_path):
            img = rf.render_ortho(
                tracer, buf_clear, cam, film, n, n, int(args.spp), device,
                max(int(args.chunk), 1), int(args.seed) + rec["frame"] * 2 + 1,
            )
            np.save(clear_path, img.astype(np.float32))
            del img
        torch.cuda.empty_cache()
        done.append(rec)
    with open(os.path.join(args.outdir, "frames.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    print("wrote", args.outdir, "frames", len(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
