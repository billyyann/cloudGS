#!/usr/bin/env python3
"""Compare GPU SPECTRAL_LW L_λ at 10500 nm vs htrdr -s lw=10500,10501."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "scenes", "fire_gs", "lwir10500")
DLAMBDA_M = 1.0e-9  # lw=10500,10501 nm
GPU_LAMBDA_NM = 10500.0


def load_htrdr_xwave(path: str) -> np.ndarray:
    """dump_pixel_xwave: T_E T_SE L_E L_SE 0 0 time_E time_SE. L is W/m²/sr."""
    rows = []
    with open(path, "r") as f:
        header = f.readline().split()
        w, h = int(header[0]), int(header[1])
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            rows.append(float(parts[2]))
    arr = np.asarray(rows, np.float64)
    if arr.size != w * h:
        raise ValueError(f"{path}: expected {w*h} pixels, got {arr.size}")
    return arr.reshape(h, w)


def gpu_L_lambda(npy: str) -> np.ndarray:
    img = np.load(npy)
    if img.ndim == 3:
        return np.mean(img.astype(np.float64), axis=-1)
    return img.astype(np.float64)


def tone_linear(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((y - lo) / max(hi - lo, 1e-30), 0.0, 1.0)
    g = (t * 255.0 + 0.5).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def signed_rgb(d: np.ndarray, scale: float) -> np.ndarray:
    t = np.clip(d / max(float(scale), 1e-30), -1.0, 1.0)
    out = np.zeros(d.shape + (3,), np.float32)
    out[..., 0] = np.clip(t, 0.0, 1.0)
    out[..., 2] = np.clip(-t, 0.0, 1.0)
    out[..., 1] = 0.04 * (1.0 - np.abs(t))
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def stats(name: str, a: np.ndarray) -> dict:
    a = np.asarray(a, np.float64)
    return {
        "name": name,
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "cv": float(a.std() / max(a.mean(), 1e-30)),
        "p01": float(np.percentile(a, 1)),
        "p05": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "p995": float(np.percentile(a, 99.5)),
        "min": float(a.min()),
        "max": float(a.max()),
        "frac_zero": float(np.mean(a <= 0.0)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-npy", default="")
    p.add_argument("--htrdr-txt", default="")
    p.add_argument("--outdir", default=OUT)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    txt = args.htrdr_txt
    if not txt:
        cands = [
            os.path.join(args.outdir, n)
            for n in os.listdir(args.outdir)
            if n.startswith("htrdr_fire_lwir10500_") and n.endswith(".txt")
        ]
        cands.sort(key=os.path.getmtime)
        if not cands:
            print("no htrdr txt in", args.outdir, file=sys.stderr)
            return 1
        txt = cands[-1]
    gpu_npy = args.gpu_npy
    if not gpu_npy:
        cands = [
            os.path.join(args.outdir, n)
            for n in os.listdir(args.outdir)
            if n.startswith("gpu_lwir_nouav") and n.endswith(".npy")
        ]
        cands.sort(key=os.path.getmtime)
        if not cands:
            print("no GPU npy in", args.outdir, file=sys.stderr)
            return 1
        gpu_npy = cands[-1]
    if not os.path.isfile(gpu_npy):
        print("missing GPU npy", gpu_npy, file=sys.stderr)
        return 1
    args.gpu_npy = gpu_npy
    gpu = gpu_L_lambda(args.gpu_npy)
    htr = load_htrdr_xwave(txt)
    if gpu.shape != htr.shape:
        print(f"shape GPU {gpu.shape} vs htrdr {htr.shape}", file=sys.stderr)
        return 1
    htr_spec = htr / DLAMBDA_M
    ratio = float(gpu.mean() / max(htr_spec.mean(), 1e-30))
    rel = (gpu - htr_spec) / np.maximum(htr_spec, 1e-30)
    pcts = np.array([1, 5, 25, 50, 75, 95, 99], dtype=np.float64)
    qg = np.percentile(gpu, pcts)
    qh = np.percentile(htr_spec, pcts)
    rec = {
        "gpu_npy": args.gpu_npy,
        "htrdr_txt": txt,
        "dlambda_m": DLAMBDA_M,
        "lambda_nm": GPU_LAMBDA_NM,
        "unit": "GPU and htrdr/Δλ are L_λ W m^-2 sr^-1 m^-1",
        "gpu": stats("gpu_L_lambda", gpu),
        "htrdr_integrated": stats("htrdr_E", htr),
        "htrdr_L_lambda": stats("htrdr_E/dlambda", htr_spec),
        "mean_GPU_over_htrdr": ratio,
        "median_GPU_over_htrdr": float(np.median(gpu) / max(np.median(htr_spec), 1e-30)),
        "quantile_GPU_over_htrdr": {
            f"p{int(p)}": float(qg[i] / max(qh[i], 1e-30)) for i, p in enumerate(pcts)
        },
        "rel_mean": float(rel.mean()),
        "rel_p50": float(np.median(rel)),
        "rel_p95_abs": float(np.percentile(np.abs(rel), 95)),
        "corr": float(np.corrcoef(gpu.ravel(), htr_spec.ravel())[0, 1]),
    }
    print(
        f"LWIR 10500 nm  GPU L_λ mean={rec['gpu']['mean']:.4g}  "
        f"htrdr L_λ mean={rec['htrdr_L_lambda']['mean']:.4g}  "
        f"GPU/htrdr={ratio:.3f}  "
        f"htrdr ∫L dλ mean={rec['htrdr_integrated']['mean']:.4g}"
    )
    print(
        "quantile GPU/htrdr "
        + " ".join(f"p{int(p)}={rec['quantile_GPU_over_htrdr'][f'p{int(p)}']:.4f}" for p in pcts)
    )
    px_m = 400.0 / float(gpu.shape[0])
    sig_px = 20.0 / px_m
    g_s = gaussian_filter(gpu, sig_px, mode="nearest")
    h_s = gaussian_filter(htr_spec, sig_px, mode="nearest")
    rec["noise_std_over_mean_gpu"] = float((gpu - g_s).std() / max(gpu.mean(), 1e-30))
    rec["noise_std_over_mean_htrdr"] = float(
        (htr_spec - h_s).std() / max(htr_spec.mean(), 1e-30)
    )
    rec["corr_20m"] = float(np.corrcoef(g_s.ravel(), h_s.ravel())[0, 1])
    rec["smooth_rel_p95_abs"] = float(
        np.percentile(np.abs((g_s - h_s) / np.maximum(h_s, 1e-30)), 95)
    )
    jpath = os.path.join(args.outdir, "compare_lwir10500.json")
    with open(jpath, "w") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
    mid = 0.5 * (gpu.mean() + htr_spec.mean())
    lo_raw, hi_raw = mid * 0.97, mid * 1.03
    lo_s, hi_s = mid * 0.997, mid * 1.003
    g_rgb = tone_linear(gpu, lo_raw, hi_raw)
    h_rgb = tone_linear(htr_spec, lo_raw, hi_raw)
    gs_rgb = tone_linear(g_s, lo_s, hi_s)
    hs_rgb = tone_linear(h_s, lo_s, hi_s)
    d_rgb = signed_rgb(rel, 0.05)
    rel_s = (g_s - h_s) / np.maximum(h_s, 1e-30)
    ds_rgb = signed_rgb(rel_s, 0.005)
    hgt, wdt = gpu.shape
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    grid = Image.new("RGB", (wdt * 3 + 24, hgt * 2 + 72), (12, 12, 12))
    dr = ImageDraw.Draw(grid)
    dr.text(
        (8, 6),
        f"FIRE LWIR 10500 nm  GPU/htrdr mean={ratio:.3f}  "
        f"row1 raw ±3% of mean  (not 1–99.5% noise stretch)",
        fill=(220, 220, 220),
        font=font,
    )
    dr.text(
        (8, 24),
        f"row2 Gauss 20 m ±0.3%  σ_noise GPU={rec['noise_std_over_mean_gpu']:.2%} "
        f"htrdr={rec['noise_std_over_mean_htrdr']:.2%}  corr20m={rec['corr_20m']:.3f}",
        fill=(180, 180, 180),
        font=font,
    )
    grid.paste(Image.fromarray(h_rgb), (8, 48))
    grid.paste(Image.fromarray(g_rgb), (16 + wdt, 48))
    grid.paste(Image.fromarray(d_rgb), (24 + 2 * wdt, 48))
    grid.paste(Image.fromarray(hs_rgb), (8, 56 + hgt))
    grid.paste(Image.fromarray(gs_rgb), (16 + wdt, 56 + hgt))
    grid.paste(Image.fromarray(ds_rgb), (24 + 2 * wdt, 56 + hgt))
    png = os.path.join(args.outdir, "compare_lwir10500.png")
    grid.save(png)
    Image.fromarray(h_rgb).save(os.path.join(args.outdir, "htrdr_lwir10500.png"))
    Image.fromarray(g_rgb).save(os.path.join(args.outdir, "gpu_lwir10500.png"))
    print("wrote", png, jpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
