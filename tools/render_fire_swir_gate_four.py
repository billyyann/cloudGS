#!/usr/bin/env python3
"""SWIR 1600 nm gate: render cases from unified manifest CSV(s).

One row per file, e.g. ``manifest1.csv`` (clear + aircraft), ``manifest2.csv``
(cloudy + aircraft). Switches:

  clouds=0/1         kernel no_clouds=(clouds==0)
  has_aircraft=0/1   water-only vs water+An-124
  cloud_mode         none | htcp | slab

Usage::

  python tools/render_fire_swir_gate_four.py --manifest scenes/fire_gs/swir_video5s/manifest1.csv
  python tools/render_fire_swir_gate_four.py --manifest .../manifest1.csv .../manifest2.csv

Does not change the OptiX kernel.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

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
import render_fire_shahed_bands as rf  # noqa: E402
import retrieve_fire_sc_rte as rte  # noqa: E402
import swir_camera_psf as cam_psf  # noqa: E402
import swir_manifest as swir_mf  # noqa: E402
from render_mountain_gs_gpu import make_conf  # noqa: E402
from threedgptir_tracer.tracer import load_threedgptir_plugin  # noqa: E402
import threedgptir_tracer.tracer as tracer_mod  # noqa: E402

PLACE_X = 13275.0
PLACE_Y = 19725.0
V_MPS = 220.0
T_S = 5.0
PAD_M = 40.0
AC_LEN_M = 69.1
HEADING_RAD = 0.0

# Shared stare optics (LWIR row decides H so last frame stays in FOV).
LWIR_NX = 640
LWIR_PITCH_M = 17.0e-6
LWIR_F_M = 14.0e-3
SWIR_NX = 512
SWIR_NY = 512
SWIR_SS = 2  # render ss*N, area-downsample to N
SWIR_PITCH_M = 15.0e-6
SWIR_F_M = 9.6e-3
SWIR_FNUM = 2.0
SWIR_FPS = 30.0
SWIR_NETD_K = 0.05
OUTDIR_DEFAULT = os.path.join(ROOT, "scenes", "fire_gs", "swir_video5s", "gate")
DOWNSAMPLE_CSV_DEFAULT = cam_psf.DEFAULT_DOWNSAMPLE_CSV
MTF_CSV_DEFAULT = cam_psf.DEFAULT_MTF_CSV
MANIFEST_DEFAULT = [swir_mf.MANIFEST1]


def stare_geometry(nx: int = SWIR_NX, ny: int = SWIR_NY) -> dict:
    half_need = V_MPS * T_S + 0.5 * AC_LEN_M + PAD_M
    ifov_lwir = LWIR_PITCH_M / LWIR_F_M
    h_cam = (2.0 * half_need) / (LWIR_NX * ifov_lwir)
    film_w = 2.0 * half_need
    film_h = film_w * (ny / float(nx))
    gsd_optics = (SWIR_PITCH_M / SWIR_F_M) * h_cam
    gsd_film = film_w / float(nx)  # actual ortho sample pitch
    return {
        "H_m": float(h_cam),
        "film_w_m": float(film_w),
        "film_h_m": float(film_h),
        "nx": int(nx),
        "ny": int(ny),
        "gsd_m": float(gsd_film),
        "gsd_optics_m": float(gsd_optics),
        "half_need_m": float(half_need),
        "water_m": float(film_w + 200.0),
    }


def band_swir() -> dict:
    for b in rf.BANDS:
        if b["key"] == "swir":
            return dict(b)
    raise RuntimeError("SWIR band missing")


def load_plugin():
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    conf = make_conf()
    print("loading PTIR OptiX plugin...")
    load_threedgptir_plugin(conf)
    plugin = tracer_mod._threedgptir_plugin
    tracer = plugin.OptixTracer(
        TRACER,
        __import__("torch.utils.cpp_extension", fromlist=["CUDA_HOME"]).CUDA_HOME,
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
    return tracer, device


def lum(img: np.ndarray) -> np.ndarray:
    return rf.lum(img, False)


def save_u16_png(path: str, y: np.ndarray, lo: float, scale: float) -> None:
    t = (np.asarray(y, np.float64) - lo) / max(scale, 1e-30)
    u = np.clip(np.rint(t * 65535.0), 0, 65535).astype(np.uint16)
    Image.fromarray(u).save(path)


def save_u8_display(path: str, y: np.ndarray, *, clear_with_ac: bool = False) -> None:
    """8-bit viewing only. Does not replace scientific I;16.

    clear_with_ac: linear stretch anchored on sea so body/sea≈2–4 stays mid-grey,
    not a binary crush (radiometry lives in the float npy / I;16).
    """
    y = np.asarray(y, np.float64)
    if clear_with_ac:
        h, w_ = y.shape
        sea = np.concatenate(
            [
                y[: max(h // 8, 8), : max(w_ // 8, 8)].ravel(),
                y[: max(h // 8, 8), -max(w_ // 8, 8) :].ravel(),
                y[-max(h // 8, 8) :, : max(w_ // 8, 8)].ravel(),
                y[-max(h // 8, 8) :, -max(w_ // 8, 8) :].ravel(),
            ]
        )
        sea_med = float(np.median(sea))
        lo = sea_med * 0.90
        hi = max(float(np.percentile(y, 99.5)), sea_med * 4.0)
        u8 = np.clip(np.rint((y - lo) / max(hi - lo, 1e-30) * 255.0), 0, 255).astype(np.uint8)
    else:
        lo = float(np.percentile(y, 0.5))
        hi = float(np.percentile(y, 99.5))
        u8 = np.clip(np.rint((y - lo) / max(hi - lo, 1e-30) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", path)



def main() -> int:
    p = argparse.ArgumentParser(description="SWIR gate from unified manifest CSV(s)")
    p.add_argument(
        "--manifest",
        nargs="+",
        default=None,
        help="one or more manifest CSV paths (default: manifest1.csv)",
    )
    p.add_argument("--outdir", default=OUTDIR_DEFAULT)
    p.add_argument("--spp", type=int, default=None, help="override manifest spp")
    p.add_argument("--no-ss", action="store_true", help="render at detector res, no supersample")
    p.add_argument("--no-mtf", action="store_true", help="skip detector MTF (RTE-only beauty)")
    p.add_argument("--max-bounces", type=int, default=256)
    p.add_argument("--brick", type=int, default=8)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="legacy: clear_off→manifest1, cloud_org→manifest2 (ignored if --manifest set)",
    )
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    man_paths: list[str] = []
    if args.manifest:
        man_paths = [os.path.abspath(x) for x in args.manifest]
    elif args.only:
        alias = {
            "clear_off": swir_mf.MANIFEST1,
            "cloud_org": swir_mf.MANIFEST2,
            "cloud_noac": swir_mf.MANIFEST3,
        }
        for key in args.only:
            if key not in alias:
                raise SystemExit(
                    f"--only {key!r} needs --manifest; known aliases: {sorted(alias)}"
                )
            man_paths.append(alias[key])
    else:
        man_paths = list(MANIFEST_DEFAULT)

    manifests = [swir_mf.load_manifest(p) for p in man_paths]
    for m in manifests:
        print(
            f"manifest {m['_path']}  case={m['case']}  "
            f"clouds={m['clouds']}  has_aircraft={m['has_aircraft']}  "
            f"cloud_mode={m['cloud_mode']}"
        )

    m0 = manifests[0]
    nx = int(m0.get("nx") or SWIR_NX)
    ny = int(m0.get("ny") or SWIR_NY)
    geo = stare_geometry(nx, ny)
    if m0.get("H_m"):
        geo["H_m"] = float(m0["H_m"])
    if m0.get("film_w_m"):
        geo["film_w_m"] = float(m0["film_w_m"])
        geo["film_h_m"] = float(m0.get("film_h_m") or m0["film_w_m"] * ny / float(nx))
        geo["gsd_m"] = geo["film_w_m"] / float(nx)
        geo["water_m"] = float(m0.get("water_size_m") or (geo["film_w_m"] + 200.0))
    w, h = int(geo["nx"]), int(geo["ny"])
    ss_cfg = int(m0.get("ss") or SWIR_SS)
    ss = 1 if args.no_ss else max(ss_cfg, 1)
    w_ss, h_ss = w * ss, h * ss
    film_h = geo["film_h_m"]
    cx = float(m0.get("place_x") if m0.get("place_x") is not None else PLACE_X)
    cy = float(m0.get("place_y") if m0.get("place_y") is not None else PLACE_Y)
    heading = float(m0.get("heading_rad") if m0.get("heading_rad") is not None else HEADING_RAD)
    ac_z = float(m0.get("ac_z_m") if m0.get("ac_z_m") is not None else rf.AC_Z_M)
    spp = int(args.spp if args.spp is not None else (m0.get("spp") or 1024))
    seed = int(m0.get("seed") if m0.get("seed") is not None else 0)
    sun_az = float(m0.get("sun_az_deg") if m0.get("sun_az_deg") is not None else fire.SUN_AZ_DEG)
    sun_el = float(m0.get("sun_el_deg") if m0.get("sun_el_deg") is not None else fire.SUN_EL_DEG)
    ds_path = swir_mf.resolve_path(m0.get("downsample_csv"), DOWNSAMPLE_CSV_DEFAULT)
    mtf_path = swir_mf.resolve_path(m0.get("mtf_csv"), MTF_CSV_DEFAULT)
    htcp_path = m0.get("htcp_path") or fire.DEFAULT_HTCP
    water_rho = float(
        m0.get("water_rho") if m0.get("water_rho") is not None else rf.IR_WATER_ALBEDO[0]
    )
    water_T = float(m0.get("water_T_K") if m0.get("water_T_K") is not None else rf.WATER_T_K)
    water_alb = np.array([water_rho, water_rho, water_rho], np.float32)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    band = band_swir()
    band["spp"] = spp
    lam = float(m0.get("lambda_nm") or band["lambda_nm"])
    ds_k = None
    mtf_meta = None
    if ss > 1:
        ds_k = cam_psf.load_downsample(ds_path)
        print(f"SS {ss}x → detector {w}x{h}; downsample {ds_path}\n{ds_k}")
    if not args.no_mtf and ss > 1:
        mtf_meta = cam_psf.load_mtf(mtf_path)
        print(
            f"MTF {mtf_path}  FWHM_det={mtf_meta['fwhm_detector_px']:.3g} px  "
            f"(pixel/optical, not atmosphere)"
        )

    print("=== stare geometry ===")
    print(
        f"  H={geo['H_m']:.1f} m  film={geo['film_w_m']:.1f}x{geo['film_h_m']:.1f} m  "
        f"detector={w}x{h}  render={w_ss}x{h_ss}  GSD={geo['gsd_m']:.3f} m"
    )

    print("=== Mie + HTCP ===")
    mie = mgs.fetch_mie(lam)
    if not rf.mie_ok(mie):
        raise RuntimeError("SWIR Mie unusable")
    Cs, Ca, g = float(mie["Cs"]), float(mie["Ca"]), float(mie["g"])
    om_dump = Cs / max(Cs + Ca, 1e-30)
    vox = mgs.load_htcp_voxels(htcp_path, wavelength_nm=lam)
    maps = rte.analog_column_maps(vox)
    slab = rte.make_slab_voxels(vox, maps, om_dump)
    gas = mgs.load_ecrad_gas_1d(wavelength_m=lam * 1.0e-9)
    sw_tab = {
        "vis": mgs.load_ecrad_sw_visible(),
        "mono": mgs.load_ecrad_sw_visible(lambda0_nm=lam, lambda1_nm=lam),
    }
    lw = mgs.load_ecrad_lw()
    water = rf.water_quad(cx, cy, geo["water_m"], water_alb, water_T)

    rho_body = float(m0.get("rho_body") if m0.get("rho_body") is not None else rf.SWIR_BODY_R)
    rho_wing = float(m0.get("rho_wing") if m0.get("rho_wing") is not None else rf.SWIR_WING_R)
    rho_eng = float(m0.get("rho_engine") if m0.get("rho_engine") is not None else rf.SWIR_ENGINE_R)
    old = (rf.SWIR_BODY_R, rf.SWIR_WING_R, rf.SWIR_ENGINE_R)
    rf.SWIR_BODY_R, rf.SWIR_WING_R, rf.SWIR_ENGINE_R = rho_body, rho_wing, rho_eng
    ac = rf.apply_swir_airframe(
        rf.set_airframe_temps(
            acmesh.make_an124_coarse_mesh((cx, cy, ac_z), heading_rad=heading)
        )
    )
    rf.SWIR_BODY_R, rf.SWIR_WING_R, rf.SWIR_ENGINE_R = old
    mesh_org = mgs.concat_meshes(water, ac)
    cam = np.array([cx, cy, geo["H_m"]], np.float32)
    az = math.radians(sun_az)
    el = math.radians(sun_el)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))

    def voxels_for(cloud_mode: str):
        if cloud_mode == "slab":
            return slab
        return vox

    tracer, device = load_plugin()
    imgs = {}
    meta = {}
    result_rows = []
    for man in manifests:
        key = str(man["case"])
        clouds = int(man["clouds"])
        has_ac = int(man["has_aircraft"])
        cloud_mode = str(man["cloud_mode"])
        mesh = mesh_org if has_ac else water
        voxels = voxels_for(cloud_mode)
        buf = rf.make_buffers(
            device, voxels, gas, mesh, band, sun, sw_tab, lw, None,
            args.brick, args.max_bounces, mie["g"], mie, no_clouds=(clouds == 0),
        )
        print(
            f"-- {key} spp={spp} tris={mesh['idx'].shape[0]} "
            f"clouds={clouds} has_aircraft={has_ac} mode={cloud_mode} "
            f"render={w_ss}x{h_ss}"
        )
        img = rf.render_ortho(
            tracer, buf, cam, film_h, w_ss, h_ss, spp, device,
            max(int(args.chunk), 1), seed,
        )
        npy_ss = os.path.join(args.outdir, f"gpu_swir_{key}_spp{spp}_ss{ss}.npy")
        np.save(npy_ss, img.astype(np.float32))
        if ss > 1:
            if mtf_meta is not None:
                img = cam_psf.apply_mtf_ss(img, mtf=mtf_meta)
            img = cam_psf.downsample_ss(img, kernel=ds_k, ss=ss)
        npy = os.path.join(args.outdir, f"gpu_swir_{key}_spp{spp}.npy")
        np.save(npy, img.astype(np.float32))
        print("wrote", npy_ss if ss > 1 else npy)
        print("wrote", npy)
        y = lum(img)
        imgs[key] = y
        meta[key] = {
            "manifest": man["_path"],
            "npy": npy,
            "npy_ss": npy_ss if ss > 1 else None,
            "cloud_mode": cloud_mode,
            "clouds": clouds,
            "has_aircraft": has_ac,
            "mean": float(y.mean()),
            "ss": int(ss),
            "swir_rho_body": rho_body if has_ac else None,
            "swir_rho_engine": rho_eng if has_ac else None,
            "water_rho": water_rho,
        }
        row = {k: man.get(k, "") for k in swir_mf.FIELDS}
        row["manifest_path"] = man["_path"]
        row["npy"] = npy
        row["L_mean"] = meta[key]["mean"]
        result_rows.append(row)
        del buf, img
        torch.cuda.empty_cache()

    case_order = [str(m["case"]) for m in manifests]
    stack = np.concatenate([imgs[k].ravel() for k in case_order if k in imgs])
    lo = 0.0
    hi = float(np.percentile(stack, 99.5))
    scale = max(hi - lo, 1e-30)

    z_bot = float(np.nanmean(maps["z_bot"]))
    z_top = float(np.nanmean(maps["z_top"]))
    t_skin = float(np.nanmean(maps["t_skin"]))
    tau_mean = float(maps["tau"].mean())

    for i, key in enumerate(case_order):
        y = imgs[key]
        png = os.path.join(args.outdir, f"swir_{key}_spp{spp}_{stamp}.png")
        png8 = os.path.join(args.outdir, f"swir_{key}_spp{spp}_{stamp}_u8.png")
        save_u16_png(png, y, lo, scale)
        save_u8_display(
            png8, y,
            clear_with_ac=(int(meta[key]["clouds"]) == 0 and int(meta[key]["has_aircraft"]) == 1),
        )
        print(f"wrote {png}  L_offset={lo:.6g} L_scale={scale:.6g}")
        print(f"wrote {png8}")
        meta[key]["png"] = png
        meta[key]["png_u8"] = png8
        result_rows[i]["png"] = png
        result_rows[i]["png_u8"] = png8
        result_rows[i]["L_scale"] = scale
        result_rows[i]["L_offset"] = lo
        if int(meta[key]["clouds"]) == 1:
            result_rows[i]["tau_mean"] = tau_mean
            result_rows[i]["z_bot_m"] = z_bot
            result_rows[i]["z_top_m"] = z_top
            result_rows[i]["t_skin_K"] = t_skin
            result_rows[i]["omega_dump"] = om_dump
            result_rows[i]["g"] = g

    out_fields = swir_mf.FIELDS + [
        "manifest_path", "npy", "png", "png_u8", "L_mean", "L_scale", "L_offset",
        "tau_mean", "z_bot_m", "z_top_m", "t_skin_K", "omega_dump", "g",
    ]
    for row in result_rows:
        base = os.path.splitext(os.path.basename(row["manifest_path"]))[0]
        out_csv = os.path.join(args.outdir, f"{base}_result.csv")
        with open(out_csv, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
            wri.writeheader()
            wri.writerow({k: row.get(k, "") for k in out_fields})
        print("wrote", out_csv)

    accept = {}
    if "clear_off" in imgs:
        y = imgs["clear_off"]
        mask = rf.ac_film_mask(ac["verts"], cx, cy, film_h, w, h, pad_m=2.0)
        c = 32
        sea_patches = [y[:c, :c], y[:c, -c:], y[-c:, :c], y[-c:, -c:]]
        sea_m = float(np.mean([p.mean() for p in sea_patches]))
        core = mask & (y > sea_m * 1.15)
        if not core.any() and mask.any():
            thr = float(np.percentile(y[mask], 60.0))
            core = mask & (y >= thr)
        body_m = float(y[core].mean()) if core.any() else float("nan")
        bbox_m = float(y[mask].mean()) if mask.any() else float("nan")
        contrast = body_m / max(sea_m, 1e-30)
        gy, gx = np.gradient(y.astype(np.float64))
        gmag = np.hypot(gx, gy)
        from scipy.ndimage import binary_dilation

        edge_band = binary_dilation(mask, iterations=2) & ~mask
        edge = gmag[edge_band] if edge_band.any() else gmag[mask]
        accept = {
            "body_core_mean": body_m,
            "body_bbox_mean": bbox_m,
            "sea_corner_mean": sea_m,
            "contrast_body_sea": float(contrast),
            "contrast_bbox_sea": float(bbox_m / max(sea_m, 1e-30)),
            "peak_sea": float(float(y.max()) / max(sea_m, 1e-30)),
            "shadow_p1": float(np.percentile(y, 1)),
            "edge_grad_mean": float(edge.mean()) if edge.size else float("nan"),
            "edge_grad_p95": float(np.percentile(edge, 95)) if edge.size else float("nan"),
            "core_npix": int(core.sum()),
            "target_contrast": [2.0, 4.0],
            "ok_contrast": bool(2.0 <= contrast <= 4.5),
            "ok_shadow_not_near_zero": bool(float(np.percentile(y, 1)) > 0.02 * sea_m),
        }
        print(
            f"accept clear_off  body_core/sea={contrast:.3g}  "
            f"ok_contrast={accept['ok_contrast']}"
        )
        with open(os.path.join(args.outdir, "accept_clear_off.json"), "w") as f:
            json.dump(accept, f, indent=2)
            f.write("\n")

    rec = {
        "stamp": stamp,
        "manifests": man_paths,
        "geometry": geo,
        "lambda_nm": lam,
        "spp": spp,
        "seed": seed,
        "L_offset": lo,
        "L_scale": scale,
        "L_calibration": "u16_shared_0_stack_p99.5; beauty=SS_MTF_box2x; RTE_pre_MTF_in_ss_npy",
        "downsample_csv": None if ss <= 1 else ds_path,
        "mtf_csv": None if mtf_meta is None else mtf_path,
        "ss": int(ss),
        "swir_rho_body": rho_body,
        "swir_rho_engine": rho_eng,
        "water_rho": water_rho,
        "note": "Config from unified manifest CSV. Kernel no_clouds=(clouds==0).",
        "detector_nx": int(w),
        "detector_ny": int(h),
        "cases": meta,
        "mie": {"Cs": Cs, "Ca": Ca, "g": g, "omega_dump": om_dump},
        "accept_clear_off": accept,
    }
    with open(os.path.join(args.outdir, "record.json"), "w") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
    print("wrote", os.path.join(args.outdir, "record.json"))
    for k, m in meta.items():
        print(
            f"  {k:10s} mean={m['mean']:.6g}  "
            f"clouds={m['clouds']}  has_aircraft={m['has_aircraft']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
