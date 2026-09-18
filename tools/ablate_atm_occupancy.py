#!/usr/bin/env python3
"""Occupancy ablation harness: cloud crop spp=4 bounce=256.

Records optixLaunch, GPU/htrdr Y, and optional nvidia-smi power.
Does not overwrite spp32 npy keep files; restores crop npy after each run.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = "/home/szyj2025/miniconda3/envs/p311/bin/python"
OUT = os.path.join(ROOT, "scenes/mountain_gs")
KEEP = os.path.join(OUT, "gpu_xyz_crop_350_40_950_240_spp32.npy")
KEEP_FIRST = os.path.join(OUT, "gpu_xyz_first_crop_350_40_950_240_spp32.npy")
CROP = os.path.join(OUT, "gpu_xyz_crop_350_40_950_240.npy")
CROP_FIRST = os.path.join(OUT, "gpu_xyz_first_crop_350_40_950_240.npy")
LOG = os.path.join(OUT, "ablate_occupancy.log")

BASE_Y = 10.53
BASE_RATIO = 0.997
BASE_MS = 35.8


def restore_spp32():
    if os.path.isfile(KEEP):
        shutil.copy2(KEEP, CROP)
    if os.path.isfile(KEEP_FIRST):
        shutil.copy2(KEEP_FIRST, CROP_FIRST)


def parse(text: str) -> dict:
    d = {}
    m = re.search(r"optixLaunch=([0-9.]+)s", text)
    if m:
        d["optix_s"] = float(m.group(1))
    m = re.search(r"GPU Y mean/p50=([0-9.]+)/([0-9.]+)", text)
    if m:
        d["y_mean"] = float(m.group(1))
        d["y_p50"] = float(m.group(2))
    m = re.search(r"GPU/htrdr mean=([0-9.]+)", text)
    if m:
        d["ratio"] = float(m.group(1))
    m = re.search(r"stack computed trav=(\d+) state=(\d+) cont=(\d+)", text)
    if m:
        d["css_computed"] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"pipeline stack trav=(\d+) state=(\d+) cont=(\d+)", text)
    if m:
        d["css_used"] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"batch cloud n=(\d+) sky n=(\d+)", text)
    if m:
        d["n_cloud"] = int(m.group(1))
        d["n_sky"] = int(m.group(2))
    m = re.search(r"batch sky_ms=([0-9.]+) cloud_ms=([0-9.]+) total_ms=([0-9.]+)", text)
    if m:
        d["sky_ms"] = float(m.group(1))
        d["cloud_ms"] = float(m.group(2))
        d["total_ms"] = float(m.group(3))
    return d


def y_gate(d: dict) -> bool:
    if "y_mean" not in d or "ratio" not in d:
        return False
    if abs(d["y_mean"] - BASE_Y) / BASE_Y >= 0.02:
        return False
    return d["ratio"] >= 0.98


def run_one(name: str, extra: list[str], env_extra: dict | None = None, crop=None) -> dict:
    env = os.environ.copy()
    env["PTIR_PROFILE_RAYGEN"] = "1"
    env["PATH"] = (
        "/home/szyj2025/miniconda3/envs/p311/lib/python3.11/site-packages/slangtorch/bin:"
        + env.get("PATH", "")
    )
    ld = env.get("LD_LIBRARY_PATH", "")
    local = os.path.join(ROOT, "local/lib")
    env["LD_LIBRARY_PATH"] = local + (":" + ld if ld else "")
    if env_extra:
        env.update(env_extra)
    cmd = [
        PY,
        "-u",
        os.path.join(ROOT, "tools/render_mountain_gs_gpu.py"),
        "--camera",
        "mountain",
        "--ground",
        "mesh",
        "--spp",
        "4",
        "--max-bounces",
        "256",
        "--profile",
    ]
    if crop is None:
        cmd += ["--crop", "350", "40", "950", "240"]
    else:
        cmd += ["--crop"] + [str(c) for c in crop]
    cmd += extra
    print("====", name, "====", flush=True)
    print(" ", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    with open(LOG, "a") as f:
        f.write(f"\n\n======== {name} exit={proc.returncode} wall={dt:.1f}s ========\n")
        f.write(text)
    if proc.returncode != 0:
        print(text[-4000:])
        raise RuntimeError(f"{name} failed rc={proc.returncode}")
    d = parse(text)
    d["name"] = name
    d["wall_s"] = dt
    print(
        f"  optix={d.get('optix_s')} Y={d.get('y_mean')} ratio={d.get('ratio')} "
        f"gate={y_gate(d)} css={d.get('css_used')} cloud/sky={d.get('n_cloud')}/{d.get('n_sky')}",
        flush=True,
    )
    restore_spp32()
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    open(LOG, "w").write(f"ablate {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    restore_spp32()
    rows = []
    try:
        rows.append(run_one("A_batch", ["--batch-cloud"]))
    except RuntimeError as e:
        print("A_batch FAILED:", e)
        rows.append({"name": "A_batch", "error": str(e), "y_mean": 0.0, "ratio": 0.0})
        restore_spp32()
        print("abort remaining jobs after A_batch failure")
        for r in rows:
            print(r)
        return
    rows.append(run_one("A_sky_strip", ["--batch-cloud"], crop=(350, 0, 950, 40)))
    if not y_gate(rows[0]):
        print("A Y-gate FAIL; still running B/C as planned (wall-clock is observational)")
    rows.append(run_one("A+B_cont0", ["--batch-cloud", "--cont-stack", "0"]))
    try:
        rows.append(run_one("A+B_desert", ["--batch-cloud", "--cont-stack", "0"], crop=(350, 400, 950, 720)))
    except RuntimeError as e:
        print("A+B_desert FAILED:", e)
        rows.append({"name": "A+B_desert", "error": str(e), "y_mean": 0.0, "ratio": 0.0})
        restore_spp32()
    rows.append(run_one("A+B+C_brick4", ["--batch-cloud", "--brick", "4"]))
    for r in rows:
        print(r)
    print("Y gates:", {r["name"]: y_gate(r) for r in rows if "error" not in r})


if __name__ == "__main__":
    try:
        main()
    finally:
        restore_spp32()
