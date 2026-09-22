#!/usr/bin/env python3
"""SWIR host camera chain on MC radiance (after OptiX / htrdr-aligned RTE).

1) Optional supersample render at ss·N
2) Detector/optical MTF (Gaussian on the SS grid) — NOT atmospheric scatter
3) 2×2 box area-integrate down to detector N×N

Volume RTE is unchanged; MTF is sensor footprint only. Does not change the OptiX kernel.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DOWNSAMPLE_CSV = os.path.join(
    ROOT, "scenes", "fire_gs", "swir_video5s", "downsample_2x_box.csv"
)
DEFAULT_MTF_CSV = os.path.join(
    ROOT, "scenes", "fire_gs", "swir_video5s", "detector_mtf_gaussian.csv"
)

# Detector-plane Gaussian MTF FWHM in *detector* pixels (on the final N grid).
# On SS grid: sigma_ss = (FWHM_det * ss) / (2*sqrt(2*ln2)).
MTF_FWHM_DET_PX = 1.75

# Kept for old call sites; not used by the SS→detector path.
DEFAULT_PSF_CSV = os.path.join(ROOT, "scenes", "fire_gs", "swir_video5s", "psf_3x3.csv")


def build_box2x() -> np.ndarray:
    """2×2 box on the supersample grid: one detector pixel = mean of 4 SS cells."""
    return np.full((2, 2), 0.25, np.float64)


def write_downsample_csv(path: str = DEFAULT_DOWNSAMPLE_CSV) -> np.ndarray:
    k = build_box2x()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("# SWIR detector downsample kernel (supersample → nominal)\n")
        f.write("# role=area_integrate_ss_to_detector\n")
        f.write("# model=box_2x2\n")
        f.write("# ss_factor=2\n")
        f.write("# meaning: each detector pixel averages the aligned 2x2 SS block\n")
        f.write(f"# sum={float(k.sum()):.12g}  (energy conserving on uniform field)\n")
        f.write("# layout: di,dj in SS sub-pixel coords within one detector cell; w weight\n")
        w = csv.writer(f)
        w.writerow(["di", "dj", "w"])
        for di in (0, 1):
            for dj in (0, 1):
                w.writerow([di, dj, f"{k[di, dj]:.12g}"])
    return k


def write_mtf_csv(
    path: str = DEFAULT_MTF_CSV,
    fwhm_det_px: float = MTF_FWHM_DET_PX,
    ss: int = 2,
) -> dict:
    """Document detector/optical MTF params (not atmospheric scatter)."""
    sigma_det = fwhm_det_px / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    sigma_ss = sigma_det * float(ss)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("# SWIR detector / optical MTF (host post-process on SS radiance)\n")
        f.write("# role=pixel_optical_mtf\n")
        f.write("# NOT atmospheric path radiance or Rayleigh/aerosol scatter\n")
        f.write("# model=isotropic_gaussian\n")
        f.write(f"# fwhm_detector_px={fwhm_det_px:.6g}\n")
        f.write(f"# sigma_detector_px={sigma_det:.6g}\n")
        f.write(f"# ss_factor={ss}\n")
        f.write(f"# sigma_ss_px={sigma_ss:.6g}  (applied on supersample grid before box downsample)\n")
        f.write("# apply: gaussian_filter(L_ss, sigma=sigma_ss_px, mode=reflect) then box 2x2\n")
        w = csv.writer(f)
        w.writerow(["param", "value"])
        w.writerow(["fwhm_detector_px", f"{fwhm_det_px:.12g}"])
        w.writerow(["sigma_detector_px", f"{sigma_det:.12g}"])
        w.writerow(["ss_factor", f"{ss}"])
        w.writerow(["sigma_ss_px", f"{sigma_ss:.12g}"])
    return {
        "fwhm_detector_px": float(fwhm_det_px),
        "sigma_detector_px": float(sigma_det),
        "ss_factor": int(ss),
        "sigma_ss_px": float(sigma_ss),
    }


def load_mtf(path: str = DEFAULT_MTF_CSV) -> dict:
    if not os.path.isfile(path):
        return write_mtf_csv(path)
    vals: dict[str, float] = {}
    with open(path, newline="") as f:
        body = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    for row in csv.DictReader(body):
        vals[row["param"].strip()] = float(row["value"])
    need = ("fwhm_detector_px", "sigma_ss_px", "ss_factor")
    if not all(k in vals for k in need):
        raise RuntimeError(f"MTF CSV missing fields {need}: {path}")
    return {
        "fwhm_detector_px": vals["fwhm_detector_px"],
        "sigma_detector_px": vals.get(
            "sigma_detector_px",
            vals["fwhm_detector_px"] / (2.0 * math.sqrt(2.0 * math.log(2.0))),
        ),
        "ss_factor": int(vals["ss_factor"]),
        "sigma_ss_px": vals["sigma_ss_px"],
    }


def apply_mtf_ss(
    img: np.ndarray,
    mtf: dict | None = None,
    path: str = DEFAULT_MTF_CSV,
) -> np.ndarray:
    """Gaussian MTF on supersampled radiance (HW or HWC). Energy ~conserved."""
    meta = load_mtf(path) if mtf is None else mtf
    sig = float(meta["sigma_ss_px"])
    a = np.asarray(img, np.float64)
    if a.ndim == 2:
        return gaussian_filter(a, sigma=sig, mode="reflect").astype(np.float32, copy=False)
    if a.ndim == 3:
        chans = [
            gaussian_filter(a[..., c], sigma=sig, mode="reflect") for c in range(a.shape[-1])
        ]
        return np.stack(chans, axis=-1).astype(np.float32, copy=False)
    raise ValueError(f"img ndim must be 2 or 3, got {a.ndim}")


def load_downsample(path: str = DEFAULT_DOWNSAMPLE_CSV) -> np.ndarray:
    if not os.path.isfile(path):
        write_downsample_csv(path)
    rows: dict[tuple[int, int], float] = {}
    with open(path, newline="") as f:
        body = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    rdr = csv.DictReader(body)
    for row in rdr:
        rows[(int(row["di"]), int(row["dj"]))] = float(row["w"])
    if set(rows) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise RuntimeError(f"downsample CSV must define 2x2 di,dj in {{0,1}}: {path}")
    k = np.array(
        [[rows[(0, 0)], rows[(0, 1)]], [rows[(1, 0)], rows[(1, 1)]]],
        np.float64,
    )
    s = float(k.sum())
    if s <= 0.0 or np.any(k < 0.0):
        raise RuntimeError(f"bad downsample kernel in {path}")
    if abs(s - 1.0) > 1e-6:
        k = k / s
    return k


def downsample_ss(img: np.ndarray, kernel: np.ndarray | None = None, ss: int = 2) -> np.ndarray:
    """Area-integrate supersampled HW[C] down by ss (default 2) with ss×ss kernel."""
    if ss != 2:
        raise NotImplementedError("only ss=2 box / loaded 2x2 kernel implemented")
    k = build_box2x() if kernel is None else np.asarray(kernel, np.float64)
    if k.shape != (ss, ss):
        raise ValueError(f"kernel must be {ss}x{ss}, got {k.shape}")
    a = np.asarray(img, np.float64)
    if a.ndim == 2:
        h, w = a.shape
        if h % ss or w % ss:
            raise ValueError(f"SS shape must be divisible by {ss}, got {a.shape}")
        # weighted sum over 2x2 blocks
        out = (
            k[0, 0] * a[0::ss, 0::ss]
            + k[0, 1] * a[0::ss, 1::ss]
            + k[1, 0] * a[1::ss, 0::ss]
            + k[1, 1] * a[1::ss, 1::ss]
        )
        return out.astype(np.float32, copy=False)
    if a.ndim == 3:
        chans = [downsample_ss(a[..., c], kernel=k, ss=ss) for c in range(a.shape[-1])]
        return np.stack(chans, axis=-1)
    raise ValueError(f"img ndim must be 2 or 3, got {a.ndim}")


# --- legacy 3x3 PSF (optional debug); not the SS→detector path ---
def build_gaussian_3x3(sigma_px: float = 0.60) -> np.ndarray:
    xs = np.arange(-1, 2, dtype=np.float64)
    x, y = np.meshgrid(xs, xs, indexing="xy")
    k = np.exp(-(x * x + y * y) / (2.0 * sigma_px * sigma_px))
    return (k / float(k.sum())).astype(np.float64)


def load_psf(path: str = DEFAULT_PSF_CSV) -> np.ndarray:
    from scipy.ndimage import convolve  # local; only if legacy used

    if not os.path.isfile(path):
        k = build_gaussian_3x3()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            f.write("# legacy 3x3 gaussian — prefer downsample_2x_box.csv\n")
            w = csv.writer(f)
            w.writerow(["di", "dj", "w"])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    w.writerow([di, dj, f"{k[di + 1, dj + 1]:.12g}"])
        return k
    rows = {}
    with open(path, newline="") as f:
        body = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    for row in csv.DictReader(body):
        rows[(int(row["di"]), int(row["dj"]))] = float(row["w"])
    k = np.zeros((3, 3), np.float64)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            k[di + 1, dj + 1] = rows[(di, dj)]
    return k / float(k.sum())


def apply_psf(img: np.ndarray, kernel: np.ndarray | None = None, path: str = DEFAULT_PSF_CSV) -> np.ndarray:
    from scipy.ndimage import convolve

    k = load_psf(path) if kernel is None else np.asarray(kernel, np.float64)
    a = np.asarray(img, np.float64)
    if a.ndim == 2:
        return convolve(a, k, mode="reflect").astype(np.float32, copy=False)
    return np.stack(
        [convolve(a[..., c], k, mode="reflect") for c in range(a.shape[-1])],
        axis=-1,
    ).astype(np.float32, copy=False)


if __name__ == "__main__":
    k = write_downsample_csv()
    print("wrote", DEFAULT_DOWNSAMPLE_CSV)
    print(k)
    meta = write_mtf_csv()
    print("wrote", DEFAULT_MTF_CSV)
    print(meta)
