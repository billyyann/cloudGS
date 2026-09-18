#!/usr/bin/env python3
"""FIRE Sc nadir: An-124 under cloud, four spectral bands, with/without aircraft.

GPU OptiX only. Writes scenes/fire_gs/transport_bands/. Does not touch mountain_gs
or rewrite the FIRE sea path.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.utils.cpp_extension
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACER = os.path.join(ROOT, "gpu", "threedgptir_tracer")
sys.path.insert(0, os.path.join(ROOT, "gpu"))
sys.path.insert(0, os.path.dirname(__file__))

_local_lib = os.path.join(ROOT, "local", "lib")
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _local_lib not in _ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = _local_lib + (":" + _ld if _ld else "")

import aircraft_mesh as acmesh  # noqa: E402
import fire_scene as fire  # noqa: E402
import mountain_gs_scene as mgs  # noqa: E402
import sea_scene as sea  # noqa: E402
from render_mountain_gs_gpu import analog_kext_tr, make_conf, make_ortho_rays  # noqa: E402
from render_sea_gpu import bind_lw, bind_sw, bind_sw_h2o  # noqa: E402
from threedgptir_tracer.atmosphere import (  # noqa: E402
    MTL_RGB_LAMBERT,
    MTL_SPECTRAL_SPECULAR,
    SPECTRAL_LW,
    SPECTRAL_SW,
    SPECTRAL_SW_CIE_XYZ,
    AtmosphereBuffers,
    build_cie_xyz_tables,
    htpp_xyz_to_srgb,
    trace_atmosphere,
)

AC_Z_M = 100.0
ORTHO_Z_M = 2000.0
ORTHO_HEIGHT_M = 400.0
WATER_SIZE_M = 600.0
WATER_T_K = 290.0
BODY_T_K = 300.0
ENGINE_T_K = 400.0
BODY_R = 0.42
ENGINE_R = 0.08
# IR: painted airframe ε~0.90; nacelle hotter than the visible-mesh 400 K slot.
IR_BODY_R = 0.10
IR_ENGINE_R = 0.08
IR_ENGINE_T_K = 550.0
ORTHO_LOOK = np.array([0.0, 0.0, -1.0], np.float32)
ORTHO_UP = np.array([0.0, 1.0, 0.0], np.float32)
IR_WATER_ALBEDO = np.array([0.04, 0.04, 0.04], np.float32)
# Smooth ΔL/L at this scale so col3 shows the thermal halo, not pixel MC noise.
DELTA_SMOOTH_M = 20.0
# MWIR/LWIR: do not 99.5%-stretch. 64 spp 20 m p995 is MC noise (~0.07% / 0.003%).
# A real through-cloud halo at 2048 spp was ~0.1%; fixed ±0.5% keeps noise dark.
IR_DELTA_SCALE = 5.0e-3
# Seawater Fresnel ρ across IR windows. blue_water.mrumtl zeros after ~2.08 μm.
IR_WATER_LUT_NM = np.array([1400.0, 1800.0, 3700.0, 4800.0, 8000.0, 12000.0], np.float32)
IR_WATER_LUT_R = np.full(IR_WATER_LUT_NM.shape, 0.04, np.float32)

BANDS = (
    {
        "key": "vis",
        "label": "Visible 380-780 nm",
        "lambda_nm": 550.0,
        "window_lo_nm": 380.0,
        "window_hi_nm": 780.0,
        "mode": SPECTRAL_SW_CIE_XYZ,
        "spp": 128,
        "sun": True,
        "absorb_only": False,
    },
    {
        "key": "swir",
        "label": "SWIR 1600 nm",
        "lambda_nm": 1600.0,
        "window_lo_nm": 1600.0,
        "window_hi_nm": 1600.0,
        "mode": SPECTRAL_SW,
        "spp": 128,
        "sun": True,
        "absorb_only": False,
    },
    {
        "key": "mwir",
        "label": "MWIR 4000 nm",
        "lambda_nm": 4000.0,
        "window_lo_nm": 4000.0,
        "window_hi_nm": 4000.0,
        "mode": SPECTRAL_LW,
        "spp": 64,
        "sun": False,
        "absorb_only": True,
    },
    {
        "key": "lwir",
        "label": "LWIR 10500 nm",
        "lambda_nm": 10500.0,
        "window_lo_nm": 10500.0,
        "window_hi_nm": 10500.0,
        "mode": SPECTRAL_LW,
        "spp": 64,
        "sun": False,
        "absorb_only": True,
    },
)


def mie_ok(mie: dict) -> bool:
    cs, ca, g = float(mie["Cs"]), float(mie["Ca"]), float(mie["g"])
    return (
        np.isfinite(cs)
        and np.isfinite(ca)
        and np.isfinite(g)
        and cs > 0.0
        and ca >= 0.0
        and 0.0 <= g < 1.0
    )


def window_mean_planck(lo_nm: float, hi_nm: float, temperature_K: float, n: int = 48) -> float:
    """Uniform-λ mean of B_λ over a sensor window (same estimator as atm_lw_bind_path)."""
    lo = float(lo_nm) * 1.0e-9
    hi = float(hi_nm) * 1.0e-9
    if hi <= lo:
        return planck_B(lo if lo > 0.0 else hi, temperature_K)
    lams = np.linspace(lo, hi, int(n), dtype=np.float64)
    return float(np.mean([planck_B(float(l), temperature_K) for l in lams]))


def cloudy_mean_temp(voxels: dict) -> float:
    ka = np.asarray(voxels["ka"], np.float32)
    ks = np.asarray(voxels["ks"], np.float32)
    temp = np.asarray(voxels["temp"], np.float32)
    m = (ka + ks) > 1.0e-6
    if not np.any(m):
        return float(np.nanmean(temp))
    return float(temp[m].mean())


def planck_B(lambda_m: float, temperature_K: float) -> float:
    """Monochromatic Planck B_λ, same constants as atm_planck_monochromatic."""
    c = 299792458.0
    h = 6.62607015e-34
    k = 1.380649e-23
    if temperature_K <= 0.0 or lambda_m <= 0.0:
        return 0.0
    lam5 = lambda_m ** 5
    hc_lkT = (h * c) / (lambda_m * k * temperature_K)
    return (2.0 * h * c * c / lam5) / (math.expm1(hc_lkT))


def analog_column(
    ka,
    ks,
    origin,
    spacing,
    org,
    direction,
    tmax=3.0e3,
    z_iface=None,
    absorb_only=False,
) -> tuple[float, float]:
    """Return (τ, Tr) along a host analog column (diagnostic, not the GPU weight).

    GPU LW samples collisions with kext; it does not multiply analog Tr(Ka).
    ka-only is a cloud-absorption diagnostic.
    """
    ka = np.asarray(ka, np.float32)
    ks = np.zeros_like(ka) if absorb_only else np.asarray(ks, np.float32)
    tr = analog_kext_tr(ka, ks, origin, spacing, org, direction, tmax=tmax, z_iface=z_iface)
    if tr <= 0.0:
        return 80.0, 0.0
    tau = float(-math.log(max(tr, 1e-30)))
    return tau, float(tr)


def pick_mean_column(voxels: dict, edge: int = 12) -> dict:
    """Interior column whose 550 nm τ is closest to the domain mean (cloudy)."""
    ka = np.asarray(voxels["ka"])
    ks = np.asarray(voxels["ks"])
    zi = np.asarray(voxels["z_iface"], np.float64)
    dz = np.diff(zi)
    tau = ((ks.astype(np.float64) + ka.astype(np.float64)) * dz[:, None, None]).sum(0)
    ny, nx = tau.shape
    mean = float(tau.mean())
    e = min(int(edge), ny // 4, nx // 4)
    sl = tau[e : ny - e, e : nx - e]
    cloudy = sl > 0.3
    if not np.any(cloudy):
        cloudy = sl > 0.0
    err = np.where(cloudy, np.abs(sl - mean), np.inf)
    iy, ix = np.unravel_index(int(np.argmin(err)), sl.shape)
    iy = int(iy + e)
    ix = int(ix + e)
    ox, oy = float(voxels["origin"][0]), float(voxels["origin"][1])
    dx, dy = float(voxels["spacing"][0]), float(voxels["spacing"][1])
    x = ox + (ix + 0.5) * dx
    y = oy + (iy + 0.5) * dy
    print(
        f"place column ix,iy=({ix},{iy})  xy=({x:.1f},{y:.1f})  "
        f"τ={float(tau[iy, ix]):.2f}  mean={mean:.2f}  max={float(tau.max()):.2f}"
    )
    return {
        "ix": ix,
        "iy": iy,
        "x": x,
        "y": y,
        "tau": float(tau[iy, ix]),
        "tau_mean": mean,
        "tau_max": float(tau.max()),
    }


def water_quad(cx: float, cy: float, size: float, albedo, temp: float) -> dict:
    h = 0.5 * float(size)
    verts = np.array(
        [
            [cx - h, cy - h, 0.0],
            [cx + h, cy - h, 0.0],
            [cx + h, cy + h, 0.0],
            [cx - h, cy + h, 0.0],
        ],
        np.float32,
    )
    idx = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    alb = np.broadcast_to(np.asarray(albedo, np.float32).reshape(1, 3), (2, 3)).copy()
    return {
        "verts": verts,
        "idx": idx,
        "albedo": alb,
        "temp": np.full((2,), float(temp), np.float32),
        "mat_id": np.full((2,), MTL_RGB_LAMBERT, np.int32),
    }


_SWELL_GRID = None


def local_sea_mesh(
    cx: float,
    cy: float,
    size: float = WATER_SIZE_M,
    albedo=None,
    temp: float = WATER_T_K,
    max_tris: int = 64800,
) -> dict:
    """JONSWAP chop + sea_3m swell on a finite patch. ntri kept <=65536 so wrap stays off."""
    global _SWELL_GRID
    n1 = int(math.floor(math.sqrt(max(float(max_tris), 8.0) / 2.0)))
    n = n1 + 1
    dx = float(size) / float(n1)
    ox = float(cx) - 0.5 * float(size)
    oy = float(cy) - 0.5 * float(size)
    xs = ox + np.arange(n, dtype=np.float64) * dx
    ys = oy + np.arange(n, dtype=np.float64) * dx
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    if _SWELL_GRID is None:
        _SWELL_GRID = sea.load_sea_grid(fire.DEFAULT_OBJ)
    z = fire._bilinear_z(_SWELL_GRID, xx, yy)
    lam_min = 2.0 * dx
    lam_max = min(float(fire.SEA_TILE_LAM_MAX), 0.25 * float(size))
    z = z + fire.jonswap_chop(
        n, dx, fire.SEA_TILE_U10, fire.SUN_AZ_DEG, lam_min, lam_max, fire.SEA_TILE_SEED
    )
    grid = np.empty((n, n, 3), np.float32)
    grid[:, :, 0] = xx.astype(np.float32)
    grid[:, :, 1] = yy.astype(np.float32)
    grid[:, :, 2] = z.astype(np.float32)
    mesh = sea.heightfield_to_sea_mesh(grid)
    alb = fire.WATER_BODY_ALBEDO if albedo is None else np.asarray(albedo, np.float32)
    mesh["albedo"][:] = alb.reshape(1, 3)
    mesh["temp"][:] = float(temp)
    mesh["mat_id"][:] = MTL_SPECTRAL_SPECULAR
    ntri = int(mesh["idx"].shape[0])
    if not (32 < ntri <= 65536):
        raise RuntimeError(f"local sea tris={ntri} not in (32, 65536]")
    print(
        f"local sea n={n} dx={dx:.2f} m  tris={ntri}  "
        f"z=[{float(mesh['verts'][:, 2].min()):.2f},{float(mesh['verts'][:, 2].max()):.2f}]  "
        f"specular blue_water"
    )
    return mesh


def set_airframe_temps(ac: dict) -> dict:
    dark = np.max(ac["albedo"], axis=1) < 0.2
    ac["temp"] = np.where(dark, ENGINE_T_K, BODY_T_K).astype(np.float32)
    n_dark = int(np.count_nonzero(dark))
    print(
        f"An-124 T body={BODY_T_K:.0f} K  engine={ENGINE_T_K:.0f} K  "
        f"engine_tris={n_dark}/{ac['temp'].shape[0]}"
    )
    return ac


def apply_ir_airframe(ac: dict) -> dict:
    """Kirchhoff grey body for MWIR/LWIR: ρ=1-ε, not the visible paint albedo."""
    out = {
        "verts": np.asarray(ac["verts"], np.float32).copy(),
        "idx": np.asarray(ac["idx"], np.int32).copy(),
        "albedo": np.asarray(ac["albedo"], np.float32).copy(),
        "temp": np.asarray(ac["temp"], np.float32).copy(),
        "mat_id": np.asarray(ac["mat_id"], np.int32).copy(),
    }
    dark = np.max(out["albedo"], axis=1) < 0.2
    out["albedo"][:] = IR_BODY_R
    out["albedo"][dark] = IR_ENGINE_R
    out["temp"] = np.where(dark, IR_ENGINE_T_K, BODY_T_K).astype(np.float32)
    print(
        f"An-124 IR optics  body ρ={IR_BODY_R:.2f} T={BODY_T_K:.0f} K  "
        f"engine ρ={IR_ENGINE_R:.2f} T={IR_ENGINE_T_K:.0f} K  "
        f"engine_tris={int(np.count_nonzero(dark))}/{out['temp'].shape[0]}"
    )
    return out


def ac_film_mask(verts: np.ndarray, cx: float, cy: float, height: float, w: int, h: int, pad_m: float = 4.0):
    xy = np.asarray(verts, np.float64)[:, :2]
    x0, y0 = xy.min(0) - pad_m
    x1, y1 = xy.max(0) + pad_m
    js0 = (x0 - cx) / height + 0.5
    js1 = (x1 - cx) / height + 0.5
    is0 = 0.5 - (y1 - cy) / height
    is1 = 0.5 - (y0 - cy) / height
    c0 = int(np.floor(min(js0, js1) * w))
    c1 = int(np.ceil(max(js0, js1) * w))
    r0 = int(np.floor(min(is0, is1) * h))
    r1 = int(np.ceil(max(is0, is1) * h))
    c0 = max(c0, 0)
    r0 = max(r0, 0)
    c1 = min(c1, w)
    r1 = min(r1, h)
    mask = np.zeros((h, w), dtype=bool)
    if r1 > r0 and c1 > c0:
        mask[r0:r1, c0:c1] = True
    print(f"An-124 film mask rows {r0}:{r1} cols {c0}:{c1}  npix={int(mask.sum())}")
    return mask


def lum(img: np.ndarray, cie: bool) -> np.ndarray:
    if cie:
        return np.asarray(img[..., 1], np.float64)
    return np.mean(img, axis=-1).astype(np.float64)


def to_png(img: np.ndarray, cie: bool) -> np.ndarray:
    if cie:
        return (htpp_xyz_to_srgb(img, exposure=0.2) * 255.0 + 0.5).astype(np.uint8)
    y = lum(img, False)
    lo, hi = np.percentile(y, (1.0, 99.5))
    t = np.clip((y - lo) / max(hi - lo, 1e-20), 0.0, 1.0)
    g = (np.sqrt(t) * 255.0 + 0.5).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def beauty_tone(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((y - lo) / max(hi - lo, 1e-20), 0.0, 1.0)
    return np.sqrt(t)


def signed_delta_rgb(d: np.ndarray, scale: float) -> np.ndarray:
    s = max(float(scale), 1e-20)
    t = np.clip(d / s, -1.0, 1.0)
    out = np.zeros(d.shape + (3,), np.float32)
    out[..., 0] = np.clip(t, 0.0, 1.0)
    out[..., 2] = np.clip(-t, 0.0, 1.0)
    out[..., 1] = 0.04 * (1.0 - np.abs(t))
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def subtract_display_rgb(y_on: np.ndarray, y_off: np.ndarray) -> np.ndarray:
    """Kept for callers; prefer smooth_contrast_rgb for the compare grid."""
    lo, hi = np.percentile(y_off, (1.0, 99.5))
    d = beauty_tone(y_on, lo, hi) - beauty_tone(y_off, lo, hi)
    return signed_delta_rgb(d, 1.0)


def _gaussian_smooth(d: np.ndarray, sigma_px: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(np.asarray(d, np.float64), sigma=max(float(sigma_px), 0.5), mode="nearest")


def overlay_colorbar(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    bw = 14
    out = rgb.copy()
    out[:, w - bw - 2 : w - bw] = 50
    t = 1.0 - 2.0 * np.linspace(0.0, 1.0, h, dtype=np.float32)
    bar = signed_delta_rgb(t.reshape(h, 1), 1.0)
    out[:, w - bw :] = np.repeat(bar, bw, axis=1)
    return out


def overlay_aabb(rgb: np.ndarray, mask: np.ndarray, color=(210, 210, 210)) -> np.ndarray:
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    if rows.size == 0 or cols.size == 0:
        return rgb
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    out = rgb.copy()
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    return out


def smooth_contrast_rgb(
    y_on: np.ndarray,
    y_off: np.ndarray,
    px_m: float,
    mask: np.ndarray | None = None,
    smooth_m: float = DELTA_SMOOTH_M,
    scale: float | None = None,
    min_scale: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Gaussian-smoothed ΔL/L.

    Default scale is 99.5% of |smoothed|. For MWIR/LWIR pass min_scale=IR_DELTA_SCALE
    so 64 spp noise blobs are not stretched into a fake map.
    """
    c = (y_on - y_off) / max(abs(float(np.mean(y_off))), 1e-30)
    sm = _gaussian_smooth(c, smooth_m / max(float(px_m), 1e-9))
    auto = float(np.percentile(np.abs(sm), 99.5))
    if scale is None:
        scale = auto
    scale = max(float(scale), float(min_scale), 1e-20)
    rgb = signed_delta_rgb(sm, scale)
    rgb = overlay_colorbar(rgb)
    if mask is not None:
        rgb = overlay_aabb(rgb, mask)
    return rgb, scale


def write_compare_grid(
    outdir: str,
    stamp: str,
    bands: list,
    png_cells: dict,
    delta_captions: dict,
    place: dict,
    extra_note: str = "",
) -> str:
    rows = [b["key"] for b in bands]
    cols = (
        ("uav", "An-124 under cloud"),
        ("nouav", "cloud only"),
        ("delta", "smoothed ΔL/L"),
    )
    sample = next(iter(png_cells.values()))
    cw, ch = sample.shape[1], sample.shape[0] + 36
    grid = Image.new("RGB", (cw * 3 + 16, ch * len(rows) + 56), (12, 12, 12))
    dr = ImageDraw.Draw(grid)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
    )
    dr.text(
        (8, 6),
        f"FIRE An-124 nadir {ORTHO_HEIGHT_M:.0f} m  z_ac={AC_Z_M:.0f} m  "
        f"tau550={place['tau']:.2f}  spp={bands[0].get('spp', '') if bands else ''}  {stamp}",
        fill=(220, 220, 220),
        font=font,
    )
    note = (
        f"Col3 = ΔL/L Gaussian {DELTA_SMOOTH_M:.0f} m; "
        f"MWIR/LWIR fixed ±{100.0 * IR_DELTA_SCALE:.2f}% "
        f"(not 99.5% stretch). white box = airframe AABB; bar red=+ blue=-."
    )
    if extra_note:
        note = extra_note
    dr.text((8, 26), note, fill=(180, 180, 180), font=font)
    labels = {b["key"]: b["label"] for b in BANDS}
    for r, key in enumerate(rows):
        for c, (tag, clab) in enumerate(cols):
            rgb = png_cells.get((key, tag))
            if rgb is None:
                continue
            cap = delta_captions[key] if tag == "delta" else clab
            im = label_img(rgb, f"{labels.get(key, key)}  {cap}")
            grid.paste(im, (8 + c * cw, 48 + r * ch))
    gpath = os.path.join(outdir, f"compare_an124_bands_{stamp}.png")
    grid.save(gpath)
    print("wrote", gpath)
    return gpath


def compose_from_outdir(outdir: str) -> int:
    """Rebuild col3 + compare PNG from record.json npy paths. No GPU."""
    rec_path = os.path.join(outdir, "record.json")
    with open(rec_path, "r") as f:
        summary = json.load(f)
    place = summary["place"]
    w, h = int(summary["camera"]["film"][0]), int(summary["camera"]["film"][1])
    cx, cy = float(place["x"]), float(place["y"])
    size_m = summary.get("aircraft", {}).get("size_m") or [69.1, 73.3]
    ac_verts = np.array(
        [
            [cx - 0.5 * size_m[0], cy - 0.5 * size_m[1], AC_Z_M],
            [cx + 0.5 * size_m[0], cy + 0.5 * size_m[1], AC_Z_M],
        ],
        np.float32,
    )
    mask = ac_film_mask(ac_verts, cx, cy, ORTHO_HEIGHT_M, w, h)
    px_m = float(ORTHO_HEIGHT_M) / float(w)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png_cells = {}
    delta_captions = {}
    bands = []
    for br in summary["bands"]:
        key = br["band"]
        band = next(b for b in BANDS if b["key"] == key)
        bands.append(band)
        on = np.load(br["uav"]["npy"])
        off = np.load(br["nouav"]["npy"])
        is_cie = band["mode"] == SPECTRAL_SW_CIE_XYZ
        png_cells[(key, "uav")] = to_png(on, is_cie)
        png_cells[(key, "nouav")] = to_png(off, is_cie)
        y_on = lum(on, is_cie)
        y_off = lum(off, is_cie)
        ir_min = IR_DELTA_SCALE if band["mode"] == SPECTRAL_LW else 0.0
        d_rgb, c_scale = smooth_contrast_rgb(
            y_on, y_off, px_m, mask=mask, min_scale=ir_min
        )
        png_cells[(key, "delta")] = d_rgb
        if band["mode"] == SPECTRAL_LW:
            delta_captions[key] = (
                f"ΔL/L  Gauss {DELTA_SMOOTH_M:.0f} m  ±{100.0 * c_scale:.2f}%  "
                f"fixed"
            )
        else:
            delta_captions[key] = (
                f"ΔL/L  Gauss {DELTA_SMOOTH_M:.0f} m  ±{100.0 * c_scale:.3f}%"
            )
        print(f"  {key} col3 ±{100.0 * c_scale:.4f}%")
        dpng = os.path.join(outdir, f"delta_{key}_{stamp}.png")
        Image.fromarray(d_rgb).save(dpng)
    write_compare_grid(outdir, stamp, bands, png_cells, delta_captions, place)
    return 0


def label_img(rgb8, text: str) -> Image.Image:
    im = Image.fromarray(rgb8)
    bar = 36
    canvas = Image.new("RGB", (im.width, im.height + bar), (18, 18, 18))
    canvas.paste(im, (0, bar))
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
    )
    ImageDraw.Draw(canvas).text((8, 8), text, fill=(230, 230, 230), font=font)
    return canvas


def render_ortho(tracer, buffers, center, height, w, h, spp, device, chunk, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nchunk = (spp + chunk - 1) // chunk
    acc = np.zeros((h, w, 3), np.float64)
    t0 = time.perf_counter()
    for c in range(nchunk):
        n = min(chunk, spp - c * chunk)
        ori, drc = make_ortho_rays(w, h, center, height, ORTHO_UP, ORTHO_LOOK, n, device, rng=rng)
        torch.cuda.synchronize()
        out = trace_atmosphere(tracer, ori, drc, buffers, frame_number=c)
        torch.cuda.synchronize()
        rgb = out[10].float().cpu().numpy()
        acc += rgb.sum(axis=0)
        print(f"    chunk {c + 1}/{nchunk} mean={float(rgb.mean()):.4g}")
        del ori, drc, out, rgb
        torch.cuda.empty_cache()
    img = (acc / float(spp)).astype(np.float32)
    print(f"    done {time.perf_counter() - t0:.1f}s  mean={float(img.mean()):.4g}")
    return img


def attach_mesh(buf: AtmosphereBuffers, mesh: dict, device: torch.device) -> None:
    buf.mesh_verts = torch.from_numpy(np.ascontiguousarray(mesh["verts"])).to(device)
    buf.mesh_idx = torch.from_numpy(np.ascontiguousarray(mesh["idx"])).to(device)
    buf.mesh_albedo = torch.from_numpy(np.ascontiguousarray(mesh["albedo"])).to(device)
    buf.mesh_temp = torch.from_numpy(np.ascontiguousarray(mesh["temp"])).to(device)
    buf.mesh_mat_id = torch.from_numpy(np.ascontiguousarray(mesh["mat_id"])).to(device)


def bind_htrdr_cloud_cbar(
    buf: AtmosphereBuffers,
    table: dict,
    dump_mie: dict,
    dump_nm: float,
    device: torch.device,
) -> None:
    """Voxels are k=ql*C(λ_dump). Scale to htrdr C̄(Δν) of the RRTMG band that contains λ_dump."""
    cs_ref = float(dump_mie["Cs"])
    ca_ref = float(dump_mie["Ca"])
    if cs_ref <= 0.0 or ca_ref < 0.0:
        raise RuntimeError(f"dump Mie Cs/Ca unusable at {dump_nm:.0f} nm")
    if "mie_cs_band" not in table:
        raise RuntimeError("HTGOP table missing mie_cs_band (need band-average C)")
    buf.mie_cs_band = torch.as_tensor(table["mie_cs_band"], dtype=torch.float32, device=device)
    buf.mie_ca_band = torch.as_tensor(table["mie_ca_band"], dtype=torch.float32, device=device)
    buf.mie_cs_ref = cs_ref
    buf.mie_ca_ref = ca_ref if ca_ref > 0.0 else 1.0
    lo = np.asarray(table["lambda_lo_nm"], np.float64).reshape(-1)
    hi = np.asarray(table["lambda_hi_nm"], np.float64).reshape(-1)
    cs_b = np.asarray(table["mie_cs_band"], np.float64).reshape(-1)
    ca_b = np.asarray(table["mie_ca_band"], np.float64).reshape(-1)
    ib = -1
    w = float(dump_nm)
    for i in range(int(lo.size)):
        if w >= lo[i] and w < hi[i]:
            ib = i
            break
    if ib < 0 and lo.size:
        last = int(lo.size) - 1
        if w >= lo[last] and w <= hi[last]:
            ib = last
    if ib < 0:
        raise RuntimeError(f"no RRTMG band contains dump λ={dump_nm:.0f} nm")
    ks_s = float(cs_b[ib] / cs_ref)
    ka_s = float(ca_b[ib] / buf.mie_ca_ref)
    print(
        f"htrdr cloud Cbar  dump={dump_nm:.0f} nm  band={lo[ib]:.0f}-{hi[ib]:.0f} nm  "
        f"ks_scale={ks_s:.4f} ka_scale={ka_s:.4f}  (C(λ)/Cbar inverse of voxel dump)"
    )


def make_buffers(
    device,
    voxels,
    gas,
    mesh,
    band,
    sun,
    sw,
    lw,
    cie,
    brick,
    max_bounces,
    mie_g,
    mie,
    no_clouds: bool = False,
):
    mode = int(band["mode"])
    lam_nm = float(band["lambda_nm"])
    ka = np.asarray(voxels["ka"], np.float32)
    ks = np.asarray(voxels["ks"], np.float32)
    if no_clouds:
        ka = np.zeros_like(ka)
        ks = np.zeros_like(ks)
        print("no-clouds: voxel ka/ks zeroed, 1D gas kept")
    gka, gks = gas["ka"], gas["ks"]
    if mode == SPECTRAL_LW:
        gks = np.zeros_like(gka)
        ks_use = ks
        sun_radiance = 1.0 if band.get("sun") else 0.0
        mie_omega = 0.0
    elif mode == SPECTRAL_SW_CIE_XYZ:
        ks_use = ks
        sun_radiance = 24.0
        mie_omega = 1.0
    else:
        ks_use = ks
        sun_radiance = 1.0
        mie_omega = 1.0
    buf = AtmosphereBuffers.from_numpy(
        device=device,
        voxel_ka=ka,
        voxel_ks=ks_use,
        voxel_temp=voxels["temp"],
        voxel_origin=tuple(np.asarray(voxels["origin"], np.float32).tolist()),
        voxel_spacing=tuple(np.asarray(voxels["spacing"], np.float32).tolist()),
        brick=int(brick),
        gas_z=gas["z"],
        gas_ka=gka,
        gas_ks=gks,
        gas_temp=gas["temp"],
        sun_dir=sun,
        sun_radiance=sun_radiance,
        sun_cos_half=math.cos(4.6524e-3),
        spectral_mode=mode,
        wavelength_m=lam_nm * 1.0e-9,
        mie_omega=mie_omega,
        mie_g=float(mie_g),
        path_tmax=1.2e5,
        max_bounces=int(max_bounces),
        voxel_z_iface=voxels.get("z_iface"),
    )
    if mode == SPECTRAL_SW_CIE_XYZ:
        buf.cie_cdf = torch.as_tensor(cie["cdf"], dtype=torch.float32, device=device)
        buf.cie_rcp_integral = torch.as_tensor(cie["rcp_integral"], dtype=torch.float32, device=device)
        buf.cie_lambda0_nm = float(cie["lambda0_nm"])
        buf.cie_dlambda_nm = float(cie["dlambda_nm"])
        bind_sw(buf, sw["vis"], device)
    elif mode == SPECTRAL_SW:
        table = sw.get("mono") if isinstance(sw, dict) else None
        if table is None or "lambda_lo_nm" not in table:
            table = mgs.load_ecrad_sw_visible(lambda0_nm=lam_nm, lambda1_nm=lam_nm)
        ib = mgs.require_sw_band(table, lam_nm)
        lo = float(table["lambda_lo_nm"][ib])
        hi = float(table["lambda_hi_nm"][ib])
        nq = int(np.asarray(table["nquad"]).reshape(-1)[ib])
        bsun = mgs.planck_B_lambda(lam_nm * 1.0e-9, 5778.0)
        print(
            f"SPECTRAL_SW bind HTGOP {lo:.0f}-{hi:.0f} nm  nq={nq}  "
            f"L_sun=B_λ(5778 K)={bsun:.4g}  host placeholder={sun_radiance} (kernel overwrites)"
        )
        bind_sw(buf, table, device)
        bind_htrdr_cloud_cbar(buf, table, mie, lam_nm, device)
        x_h2o, table = mgs.require_sw_h2o(table, voxels)
        bind_sw_h2o(buf, table, x_h2o, device)
    else:
        bind_lw(buf, lw, device)
        bind_htrdr_cloud_cbar(buf, lw, mie, lam_nm, device)
    if mode == SPECTRAL_SW_CIE_XYZ:
        water_lut = sea.load_mrumtl(fire.DEFAULT_WATER)
        buf.brdf_lambda_nm = torch.as_tensor(water_lut["lambda_nm"], dtype=torch.float32, device=device)
        buf.brdf_R = torch.as_tensor(water_lut["R"], dtype=torch.float32, device=device)
    else:
        # IR windows: do not bind blue_water (ρ=0 after ~2.08 μm).
        buf.brdf_lambda_nm = torch.as_tensor(IR_WATER_LUT_NM, dtype=torch.float32, device=device)
        buf.brdf_R = torch.as_tensor(IR_WATER_LUT_R, dtype=torch.float32, device=device)
        print(
            f"IR seawater LUT ρ=0.04  λ=[{IR_WATER_LUT_NM[0]:.0f},{IR_WATER_LUT_NM[-1]:.0f}] nm  "
            f"(not blue_water)"
        )
    buf.mesh_verts = torch.from_numpy(np.ascontiguousarray(mesh["verts"])).to(device)
    buf.mesh_idx = torch.from_numpy(np.ascontiguousarray(mesh["idx"])).to(device)
    buf.mesh_albedo = torch.from_numpy(np.ascontiguousarray(mesh["albedo"])).to(device)
    buf.mesh_temp = torch.from_numpy(np.ascontiguousarray(mesh["temp"])).to(device)
    buf.mesh_mat_id = torch.from_numpy(np.ascontiguousarray(mesh["mat_id"])).to(device)
    return buf


def judge_spot(delta: np.ndarray, mask: np.ndarray, l_off: np.ndarray) -> dict:
    n = int(mask.sum())
    if n < 8:
        return {"spot": False, "C": 0.0, "reason": "mask too small", "n_mask": n}
    outside = ~mask
    d_on = float(delta[mask].mean())
    d_bg = float(delta[outside].mean()) if int(outside.sum()) > 32 else 0.0
    resid = d_on - d_bg
    l0 = float(l_off[mask].mean())
    c = resid / (abs(l0) + 1e-20)
    noise = float((delta[outside] - d_bg).std()) if int(outside.sum()) > 32 else float(delta.std())
    ratio = abs(resid) / (3.0 * max(noise, 1e-20))
    spot = bool(abs(resid) > 3.0 * max(noise, 1e-20) and abs(c) > 1e-3)
    reason = (
        f"Δmask={d_on:.4g} Δbg={d_bg:.4g} resid={resid:.4g}  C={c:.4g}  "
        f"3σ={3.0 * noise:.4g}  |resid|/(3σ)={ratio:.3f}"
    )
    if not spot:
        reason += "  -> no spot"
    else:
        reason += "  -> spot"
    return {
        "spot": spot,
        "C": c,
        "delta_mean": d_on,
        "delta_bg": d_bg,
        "resid": resid,
        "L_off": l0,
        "noise": noise,
        "n_mask": n,
        "reason": reason,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="FIRE An-124 four-band through-cloud GPU experiment")
    p.add_argument("--outdir", default=os.path.join(ROOT, "scenes", "fire_gs", "transport_bands"))
    p.add_argument("--spp", type=int, default=None, help="override spp for every band")
    p.add_argument("--film", nargs=2, type=int, default=[512, 512])
    p.add_argument("--max-bounces", type=int, default=256)
    p.add_argument("--brick", type=int, default=8)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="+", default=None, help="subset of vis swir mwir lwir")
    p.add_argument(
        "--compose-only",
        action="store_true",
        help="rebuild compare PNG from existing npy (no GPU)",
    )
    p.add_argument(
        "--no-uav",
        action="store_true",
        help="water+cloud only (htrdr SWIR 1600 / MWIR 4000 / LWIR 10500 compare; skip An-124)",
    )
    args = p.parse_args()

    if args.compose_only:
        return compose_from_outdir(args.outdir)

    os.makedirs(args.outdir, exist_ok=True)
    w, h = int(args.film[0]), int(args.film[1])
    bands = list(BANDS)
    if args.only:
        want = set(args.only)
        bands = [b for b in bands if b["key"] in want]
        if not bands:
            raise SystemExit(f"--only {sorted(want)} matched no band")
    if args.spp is not None:
        for b in bands:
            b["spp"] = int(args.spp)

    print("=== Mie LUT probe ===")
    mie_probe = {}
    for b in BANDS:
        mie = mgs.fetch_mie(b["lambda_nm"])
        ok = mie_ok(mie)
        mie_probe[b["key"]] = {
            "lambda_nm": b["lambda_nm"],
            "Cs": float(mie["Cs"]),
            "Ca": float(mie["Ca"]),
            "g": float(mie["g"]),
            "ok": ok,
        }
        print(f"  {b['key']} λ={b['lambda_nm']:.0f} nm  ok={ok}")
    with open(os.path.join(args.outdir, "mie_probe.json"), "w") as f:
        json.dump(mie_probe, f, indent=2)
        f.write("\n")

    print("=== FIRE native voxels @ 550 nm (no Y-pad) ===")
    vox550 = mgs.load_htcp_voxels(fire.DEFAULT_HTCP, wavelength_nm=550.0)
    place = pick_mean_column(vox550)
    place["z"] = AC_Z_M
    cx, cy = float(place["x"]), float(place["y"])
    ac = set_airframe_temps(acmesh.make_an124_coarse_mesh((cx, cy, AC_Z_M), heading_rad=0.0))
    ntri_ac = int(ac["idx"].shape[0])
    if ntri_ac < 32 or ntri_ac > 65536:
        raise RuntimeError(f"An-124 coarse mesh tris={ntri_ac} not in (32, 65536]")
    bb_ac0 = ac["verts"].min(0)
    bb_ac1 = ac["verts"].max(0)
    size_ac = (bb_ac1 - bb_ac0).astype(float)
    area_xy = float(size_ac[0] * size_ac[1])
    print(
        f"An-124 coarse  verts={ac['verts'].shape[0]} tris={ntri_ac}  "
        f"size={size_ac.tolist()} m  area_xy={area_xy:.0f} m²  "
        f"T=[{float(ac['temp'].min()):.0f},{float(ac['temp'].max()):.0f}] K  "
        f"albedo=[{float(ac['albedo'].min()):.3f},{float(ac['albedo'].max()):.3f}]"
    )
    mask = ac_film_mask(ac["verts"], cx, cy, ORTHO_HEIGHT_M, w, h)
    cam_center = np.array([cx, cy, ORTHO_Z_M], np.float32)
    az = math.radians(fire.SUN_AZ_DEG)
    el = math.radians(fire.SUN_EL_DEG)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    print(
        f"ortho center={cam_center.tolist()} height={ORTHO_HEIGHT_M}  "
        f"sun az/el={fire.SUN_AZ_DEG}/{fire.SUN_EL_DEG}"
    )

    cie_tables = build_cie_xyz_tables()
    sw_vis = mgs.load_ecrad_sw_visible()
    lw = mgs.load_ecrad_lw()
    sw = {"vis": sw_vis}

    device = torch.device("cuda")
    torch.zeros(1, device=device)
    conf = make_conf()
    from threedgptir_tracer.tracer import load_threedgptir_plugin
    import threedgptir_tracer.tracer as tracer_mod

    print("loading PTIR OptiX plugin...")
    load_threedgptir_plugin(conf)
    plugin = tracer_mod._threedgptir_plugin
    tracer = plugin.OptixTracer(
        TRACER,
        torch.utils.cpp_extension.CUDA_HOME,
        conf.render.pipeline_type,
        conf.render.backward_pipeline_type,
        conf.render.primitive_type,
        float(conf.render.particle_kernel_degree),
        float(conf.render.particle_kernel_min_response),
        bool(conf.render.particle_kernel_density_clamping),
        int(conf.render.particle_radiance_sph_degree),
        bool(conf.render.enable_normals),
        bool(conf.render.enable_hitcounts),
        bool(conf.render.enable_mis),
        bool(conf.render.enable_metallic),
        False,
        False,
        1,
    )
    print("OptixTracer ready")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    records = []
    png_cells = {}
    delta_captions = {}

    for band in bands:
        key = band["key"]
        lam = float(band["lambda_nm"])
        mie = {
            "Cs": mie_probe[key]["Cs"],
            "Ca": mie_probe[key]["Ca"],
            "g": mie_probe[key]["g"],
        }
        rec = {
            "band": key,
            "label": band["label"],
            "lambda_nm": lam,
            "window_lo_nm": float(band.get("window_lo_nm", lam)),
            "window_hi_nm": float(band.get("window_hi_nm", lam)),
            "mode": int(band["mode"]),
            "mie": mie,
            "lut_ok": bool(mie_probe[key]["ok"]),
            "place": place,
            "sun_on": bool(band["sun"]),
            "note_mwir": (
                "SPECTRAL_LW thermal only at dump λ=4000 nm; sun off (htrdr -s lw=4000,4001)"
                if key == "mwir"
                else None
            ),
            "note_lwir": (
                "SPECTRAL_LW thermal only at dump λ=10500 nm; sun off (htrdr -s lw=10500,10501)"
                if key == "lwir"
                else None
            ),
        }
        if not mie_probe[key]["ok"]:
            rec["aborted"] = True
            rec["reason"] = "Mie LUT did not return a usable Cs/Ca/g; not using 550 nm k as IR"
            rec["spot"] = False
            records.append(rec)
            print(f"ABORT {key}: {rec['reason']}")
            continue

        print(f"=== {band['label']} ===")
        if abs(lam - 550.0) < 1e-6:
            voxels = vox550
        else:
            voxels = mgs.load_htcp_voxels(fire.DEFAULT_HTCP, wavelength_nm=lam)
        if band["mode"] == SPECTRAL_LW:
            gas = mgs.load_ecrad_gas_1d_lw(wavelength_m=lam * 1.0e-9)
            ac_use = apply_ir_airframe(ac)
            t_eng = IR_ENGINE_T_K
            r_body, r_eng = IR_BODY_R, IR_ENGINE_R
        else:
            gas = mgs.load_ecrad_gas_1d(wavelength_m=lam * 1.0e-9)
            ac_use = ac
            t_eng = ENGINE_T_K
            r_body, r_eng = BODY_R, ENGINE_R
        water_alb = fire.WATER_BODY_ALBEDO if band["mode"] == SPECTRAL_SW_CIE_XYZ else IR_WATER_ALBEDO
        water = water_quad(cx, cy, WATER_SIZE_M, water_alb, WATER_T_K)
        mesh_on = mgs.concat_meshes(water, ac_use)
        ntri_on = int(mesh_on["idx"].shape[0])
        if ntri_on != 2 + ntri_ac:
            raise RuntimeError(f"concat mesh tris={ntri_on}, expected water 2 + An-124 {ntri_ac}")
        bb0 = mesh_on["verts"].min(0)
        bb1 = mesh_on["verts"].max(0)
        print(
            f"  mesh ac tris={ntri_on}  nouav tris={water['idx'].shape[0]}  "
            f"AABB z=[{float(bb0[2]):.3f},{float(bb1[2]):.3f}]  "
            f"T_ac=[{float(ac_use['temp'].min()):.0f},{float(ac_use['temp'].max()):.0f}]  "
            f"T_water={WATER_T_K:.0f} K"
        )
        rec["mesh"] = {
            "ntri_ac": ntri_on,
            "ntri_nouav": int(water["idx"].shape[0]),
            "ntri_aircraft": ntri_ac,
            "aabb_lo": bb0.astype(float).tolist(),
            "aabb_hi": bb1.astype(float).tolist(),
            "T_body_K": BODY_T_K,
            "T_engine_K": t_eng,
            "T_water_K": WATER_T_K,
            "B_body": planck_B(lam * 1.0e-9, BODY_T_K),
            "B_engine": planck_B(lam * 1.0e-9, t_eng),
            "B_water": planck_B(lam * 1.0e-9, WATER_T_K),
            "eps_body": 1.0 - r_body,
            "eps_engine": 1.0 - r_eng,
        }
        org = np.array([cx, cy, AC_Z_M], np.float64)
        tau, tr = analog_column(
            voxels["ka"],
            voxels["ks"],
            voxels["origin"],
            voxels["spacing"],
            org,
            (0.0, 0.0, 1.0),
            tmax=3.0e3,
            z_iface=voxels.get("z_iface"),
            absorb_only=False,
        )
        tau_ka, tr_ka = analog_column(
            voxels["ka"],
            voxels["ks"],
            voxels["origin"],
            voxels["spacing"],
            org,
            (0.0, 0.0, 1.0),
            tmax=3.0e3,
            z_iface=voxels.get("z_iface"),
            absorb_only=True,
        )
        rec["analog"] = {
            "org": org.tolist(),
            "dir": [0.0, 0.0, 1.0],
            "tmax_m": 3000.0,
            "gpu_walk": "kext=ka+ks (no analog Tr(Ka) on LW)",
            "tau": tau,
            "Tr": tr,
            "tau_ka_only": tau_ka,
            "Tr_ka_only": tr_ka,
        }
        print(
            f"  analog nadir kext τ={tau:.3f} Tr={tr:.4g}  "
            f"ka-only τ={tau_ka:.3f} Tr={tr_ka:.4g}  "
            f"B(T_body)={rec['mesh']['B_body']:.4g}  B(T_eng)={rec['mesh']['B_engine']:.4g}"
        )

        spp = int(band["spp"])
        buf = make_buffers(
            device,
            voxels,
            gas,
            water,
            band,
            sun,
            sw,
            lw,
            cie_tables,
            args.brick,
            args.max_bounces,
            mie["g"],
            mie,
        )
        tags = ((False, "nouav"),) if args.no_uav else ((False, "nouav"), (True, "uav"))
        for with_uav, tag in tags:
            mesh = mesh_on if with_uav else water
            attach_mesh(buf, mesh, device)
            print(
                f"-- {key} {tag} spp={spp} tris={mesh['idx'].shape[0]}  "
                f"Tmesh=[{float(mesh['temp'].min()):.0f},{float(mesh['temp'].max()):.0f}] K  "
                f"λ={lam:.0f} nm mode={int(band['mode'])} sun={'on' if band['sun'] else 'off'}"
            )
            img = render_ortho(
                tracer,
                buf,
                cam_center,
                ORTHO_HEIGHT_M,
                w,
                h,
                spp,
                device,
                max(int(args.chunk), 1),
                args.seed,
            )
            npy = os.path.join(args.outdir, f"gpu_{key}_{tag}_spp{spp}.npy")
            np.save(npy, img.astype(np.float32))
            rgb8 = to_png(img, band["mode"] == SPECTRAL_SW_CIE_XYZ)
            png = os.path.join(args.outdir, f"render_{key}_{tag}_spp{spp}_{stamp}.png")
            Image.fromarray(rgb8).save(png)
            print("wrote", npy, png)
            rec[tag] = {
                "npy": npy,
                "png": png,
                "mean": float(img.mean()),
                "L_mask": float(lum(img, band["mode"] == SPECTRAL_SW_CIE_XYZ)[mask].mean())
                if mask.any()
                else 0.0,
            }
            png_cells[(key, tag)] = rgb8
            del img
            torch.cuda.empty_cache()

        del buf
        torch.cuda.empty_cache()

        if args.no_uav:
            rec["aborted"] = False
            rec["no_uav"] = True
            records.append(rec)
            with open(os.path.join(args.outdir, f"record_{key}.json"), "w") as f:
                json.dump(rec, f, indent=2)
                f.write("\n")
            continue

        on = np.load(rec["uav"]["npy"])
        off = np.load(rec["nouav"]["npy"])
        is_cie = band["mode"] == SPECTRAL_SW_CIE_XYZ
        y_on = lum(on, is_cie)
        y_off = lum(off, is_cie)
        delta = y_on - y_off
        judge = judge_spot(delta, mask, y_off)
        rec.update(judge)
        rec["L_on"] = float(y_on[mask].mean()) if mask.any() else 0.0
        rec["frame_mean_ratio_uav_nouav"] = float(rec["uav"]["mean"] / max(abs(rec["nouav"]["mean"]), 1e-30))
        c = 32
        corner_on = float(
            0.25
            * (
                y_on[:c, :c].mean()
                + y_on[:c, -c:].mean()
                + y_on[-c:, :c].mean()
                + y_on[-c:, -c:].mean()
            )
        )
        corner_off = float(
            0.25
            * (
                y_off[:c, :c].mean()
                + y_off[:c, -c:].mean()
                + y_off[-c:, :c].mean()
                + y_off[-c:, -c:].mean()
            )
        )
        rec["corner_ratio_uav_nouav"] = corner_on / max(abs(corner_off), 1e-30)
        rec["corner_on"] = corner_on
        rec["corner_off"] = corner_off
        if band["mode"] == SPECTRAL_LW:
            t_c = cloudy_mean_temp(voxels)
            lo = float(band.get("window_lo_nm", lam))
            hi = float(band.get("window_hi_nm", lam))
            b_win = window_mean_planck(lo, hi, t_c)
            b_c = planck_B(lam * 1.0e-9, t_c)
            rec["T_cloud_K"] = t_c
            rec["B_window_cloud"] = b_win
            rec["B_center_cloud"] = b_c
            rec["L_cloud_mean"] = float(rec["nouav"]["mean"])
            print(
                f"  T_cloud={t_c:.1f} K  "
                f"B_λ-bar[{lo:.0f},{hi:.0f} nm]={b_win:.4g}  "
                f"B_λ({lam:.0f} nm)={b_c:.4g}  "
                f"L_nouav={rec['L_cloud_mean']:.4g}  "
                f"L/B_bar={rec['L_cloud_mean'] / max(b_win, 1e-30):.3f}"
            )
        if abs(rec["frame_mean_ratio_uav_nouav"] - 1.0) > 0.25:
            print(
                f"  WARNING whole-frame mean ratio "
                f"uav/nouav={rec['frame_mean_ratio_uav_nouav']:.3f}  "
                f"corner={rec['corner_ratio_uav_nouav']:.3f}  "
                f"(UAV footprint should not rescale the whole film)"
            )
        rec["aborted"] = False
        dpng = os.path.join(args.outdir, f"delta_{key}_{stamp}.png")
        px_m = float(ORTHO_HEIGHT_M) / float(w)
        ir_min = IR_DELTA_SCALE if band["mode"] == SPECTRAL_LW else 0.0
        d_rgb, c_scale = smooth_contrast_rgb(
            y_on, y_off, px_m, mask=mask, min_scale=ir_min
        )
        Image.fromarray(d_rgb).save(dpng)
        rec["delta_png"] = dpng
        if band["mode"] == SPECTRAL_LW:
            rec["delta_scale"] = (
                f"smoothed ΔL/L Gauss {DELTA_SMOOTH_M:.0f} m; "
                f"fixed ±{100.0 * c_scale:.2f}% "
                f"(not 99.5% stretch of MC noise)"
            )
            delta_captions[key] = (
                f"ΔL/L  Gauss {DELTA_SMOOTH_M:.0f} m  ±{100.0 * c_scale:.2f}%  "
                f"fixed"
            )
        else:
            rec["delta_scale"] = (
                f"smoothed ΔL/L Gauss {DELTA_SMOOTH_M:.0f} m; "
                f"colorbar ±{100.0 * c_scale:.4f}% of cloud mean"
            )
            delta_captions[key] = (
                f"ΔL/L  Gauss {DELTA_SMOOTH_M:.0f} m  ±{100.0 * c_scale:.3f}%"
            )
        rec["delta_C_scale"] = c_scale
        png_cells[(key, "delta")] = d_rgb
        print(f"  {key} {judge['reason']}")
        records.append(rec)
        with open(os.path.join(args.outdir, f"record_{key}.json"), "w") as f:
            json.dump(rec, f, indent=2)
            f.write("\n")

    summary = {
        "stamp": stamp,
        "place": place,
        "camera": {
            "center": cam_center.tolist(),
            "height_m": ORTHO_HEIGHT_M,
            "film": [w, h],
        },
        "aircraft": {
            "type": "An-124",
            "ntri": ntri_ac,
            "nvert": int(ac["verts"].shape[0]),
            "T_body_K": BODY_T_K,
            "T_engine_K": ENGINE_T_K,
            "albedo_body": BODY_R,
            "albedo_engine": ENGINE_R,
            "z_m": AC_Z_M,
            "size_m": size_ac.tolist(),
            "area_xy_m2": area_xy,
        },
        "water": {"T_K": WATER_T_K, "size_m": WATER_SIZE_M},
        "mesh_xy_wrap": "off for ntri<=65536 (finite water+aircraft; sea heightfield still -R)",
        "mie_probe": mie_probe,
        "bands": records,
        "spot_by_band": {r["band"]: r.get("spot") for r in records},
    }
    spath = os.path.join(args.outdir, "record.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print("wrote", spath)

    if png_cells and not args.no_uav:
        write_compare_grid(
            args.outdir, stamp, bands, png_cells, delta_captions, place
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
