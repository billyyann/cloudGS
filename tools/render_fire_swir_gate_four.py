#!/usr/bin/env python3
"""SWIR 1600 nm gate: four stills before the 5 s stare video.

Cases:
  cloud_org  HTCP + An-124 at stare center
  cloud_off  HTCP, no aircraft
  clear_off  ka=ks=0 (no cloud) + An-124  — 无云有运输机
  slab_off   retrieve column slab, no aircraft

Writes 16-bit PNG + imaging/target/scene CSV. Does not change the kernel.
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
    p = argparse.ArgumentParser(description="SWIR gate four stills + CSV + 16-bit PNG")
    p.add_argument("--outdir", default=OUTDIR_DEFAULT)
    p.add_argument("--spp", type=int, default=1024)
    p.add_argument("--film", nargs=2, type=int, default=[SWIR_NX, SWIR_NY], help="detector nx ny")
    p.add_argument("--ss", type=int, default=SWIR_SS, help="supersample factor (render ss*N then downsample)")
    p.add_argument("--downsample-csv", default=DOWNSAMPLE_CSV_DEFAULT, help="ss×ss area kernel CSV")
    p.add_argument("--mtf-csv", default=MTF_CSV_DEFAULT, help="detector/optical MTF CSV (not atmosphere)")
    p.add_argument("--no-ss", action="store_true", help="render at detector res, no supersample")
    p.add_argument("--no-mtf", action="store_true", help="skip detector MTF (RTE-only beauty)")
    p.add_argument("--max-bounces", type=int, default=256)
    p.add_argument("--brick", type=int, default=8)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="subset of cloud_org cloud_off clear_off slab_off",
    )
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    geo = stare_geometry(int(args.film[0]), int(args.film[1]))
    w, h = geo["nx"], geo["ny"]
    ss = 1 if args.no_ss else max(int(args.ss), 1)
    w_ss, h_ss = w * ss, h * ss
    film_h = geo["film_h_m"]
    cx, cy = PLACE_X, PLACE_Y
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    band = band_swir()
    band["spp"] = int(args.spp)
    lam = float(band["lambda_nm"])
    ds_k = None
    mtf_meta = None
    if ss > 1:
        ds_k = cam_psf.load_downsample(args.downsample_csv)
        print(
            f"SS {ss}x → detector {w}x{h}; downsample {args.downsample_csv}  "
            f"k=\n{ds_k}"
        )
    if not args.no_mtf and ss > 1:
        mtf_meta = cam_psf.load_mtf(args.mtf_csv)
        print(
            f"MTF {args.mtf_csv}  FWHM_det={mtf_meta['fwhm_detector_px']:.3g} px  "
            f"sigma_ss={mtf_meta['sigma_ss_px']:.3g}  (pixel/optical, not atmosphere)"
        )

    print("=== stare geometry ===")
    print(
        f"  H={geo['H_m']:.1f} m  film={geo['film_w_m']:.1f}x{geo['film_h_m']:.1f} m  "
        f"detector={w}x{h}  render={w_ss}x{h_ss}  "
        f"GSD_film={geo['gsd_m']:.3f} m  GSD_optics={geo['gsd_optics_m']:.3f} m  "
        f"half_need={geo['half_need_m']:.1f} m"
    )

    print("=== Mie + HTCP ===")
    mie = mgs.fetch_mie(lam)
    if not rf.mie_ok(mie):
        raise RuntimeError("SWIR Mie unusable")
    Cs, Ca, g = float(mie["Cs"]), float(mie["Ca"]), float(mie["g"])
    om_dump = Cs / max(Cs + Ca, 1e-30)
    vox = mgs.load_htcp_voxels(fire.DEFAULT_HTCP, wavelength_nm=lam)
    maps = rte.analog_column_maps(vox)
    slab = rte.make_slab_voxels(vox, maps, om_dump)
    gas = mgs.load_ecrad_gas_1d(wavelength_m=lam * 1.0e-9)
    sw_tab = {
        "vis": mgs.load_ecrad_sw_visible(),
        "mono": mgs.load_ecrad_sw_visible(lambda0_nm=lam, lambda1_nm=lam),
    }
    lw = mgs.load_ecrad_lw()
    water = rf.water_quad(cx, cy, geo["water_m"], rf.IR_WATER_ALBEDO, rf.WATER_T_K)
    ac = rf.apply_swir_airframe(
        rf.set_airframe_temps(
            acmesh.make_an124_coarse_mesh((cx, cy, rf.AC_Z_M), heading_rad=HEADING_RAD)
        )
    )
    mesh_org = mgs.concat_meshes(water, ac)
    cam = np.array([cx, cy, geo["H_m"]], np.float32)
    az = math.radians(fire.SUN_AZ_DEG)
    el = math.radians(fire.SUN_EL_DEG)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))

    want = args.only or ["cloud_org", "cloud_off", "clear_off", "slab_off"]
    jobs = []
    if "cloud_org" in want:
        jobs.append(("cloud_org", vox, mesh_org, False, "htcp", 1))
    if "cloud_off" in want:
        jobs.append(("cloud_off", vox, water, False, "htcp", 0))
    if "clear_off" in want:
        # 无云有机：体素消光清零，运输机仍在（海面+An-124）
        jobs.append(("clear_off", vox, mesh_org, True, "none", 1))
    if "slab_off" in want:
        jobs.append(("slab_off", slab, water, False, "slab", 0))

    tracer, device = load_plugin()
    imgs = {}
    meta = {}
    for key, voxels, mesh, no_clouds, cloud_mode, has_ac in jobs:
        buf = rf.make_buffers(
            device, voxels, gas, mesh, band, sun, sw_tab, lw, None,
            args.brick, args.max_bounces, mie["g"], mie, no_clouds=no_clouds,
        )
        print(
            f"-- {key} spp={args.spp} tris={mesh['idx'].shape[0]} "
            f"no_clouds={int(no_clouds)} render={w_ss}x{h_ss}"
        )
        img = rf.render_ortho(
            tracer, buf, cam, film_h, w_ss, h_ss, int(args.spp), device,
            max(int(args.chunk), 1), args.seed,
        )
        npy_ss = os.path.join(args.outdir, f"gpu_swir_{key}_spp{args.spp}_ss{ss}.npy")
        np.save(npy_ss, img.astype(np.float32))
        if ss > 1:
            if mtf_meta is not None:
                img = cam_psf.apply_mtf_ss(img, mtf=mtf_meta)
            img = cam_psf.downsample_ss(img, kernel=ds_k, ss=ss)
        npy = os.path.join(args.outdir, f"gpu_swir_{key}_spp{args.spp}.npy")
        np.save(npy, img.astype(np.float32))
        print("wrote", npy_ss if ss > 1 else npy)
        print("wrote", npy)
        y = lum(img)
        imgs[key] = y
        meta[key] = {
            "npy": npy,
            "npy_ss": npy_ss if ss > 1 else None,
            "cloud_mode": cloud_mode,
            "has_aircraft": has_ac,
            "mean": float(y.mean()),
            "downsample_csv": None if ss <= 1 else os.path.abspath(args.downsample_csv),
            "mtf_csv": None if mtf_meta is None else os.path.abspath(args.mtf_csv),
            "ss": int(ss),
            "swir_rho_body": float(rf.SWIR_BODY_R) if has_ac else None,
            "swir_rho_engine": float(rf.SWIR_ENGINE_R) if has_ac else None,
            "water_rho": float(rf.IR_WATER_ALBEDO[0]),
        }
        del buf, img
        torch.cuda.empty_cache()

    # Pull sibling npys so --only still shares absolute calibration with the gate set.
    case_order = ["cloud_org", "cloud_off", "clear_off", "slab_off"]
    case_has = {"cloud_org": 1, "cloud_off": 0, "clear_off": 1, "slab_off": 0}
    case_mode = {"cloud_org": "htcp", "cloud_off": "htcp", "clear_off": "none", "slab_off": "slab"}
    for key in case_order:
        if key in imgs:
            continue
        npy = os.path.join(args.outdir, f"gpu_swir_{key}_spp{args.spp}.npy")
        if not os.path.isfile(npy):
            continue
        y = lum(np.load(npy))
        if y.shape != (h, w):
            print(f"cal skip {key}: shape {y.shape} != detector {(h, w)}")
            continue
        imgs[key] = y
        meta[key] = {
            "npy": npy,
            "cloud_mode": case_mode[key],
            "has_aircraft": case_has[key],
            "mean": float(y.mean()),
            "from_disk": True,
        }
        print(f"cal load {key} mean={meta[key]['mean']:.6g}")

    # Scientific I;16: shared absolute [0, stack p99.5] for radiometric compare.
    # Viewing: separate 8-bit L PNG (sea/shadow/aircraft tone map on clear_off).
    stack = np.concatenate([imgs[k].ravel() for k in case_order if k in imgs])
    lo = 0.0
    hi = float(np.percentile(stack, 99.5))
    scale = max(hi - lo, 1e-30)

    imaging_rows = []
    target_rows = []
    scene_rows = []
    size = np.asarray(ac["verts"], np.float64)
    sx = float(size[:, 0].max() - size[:, 0].min())
    sy = float(size[:, 1].max() - size[:, 1].min())
    sz = float(size[:, 2].max() - size[:, 2].min())
    z_bot = float(np.nanmean(maps["z_bot"]))
    z_top = float(np.nanmean(maps["z_top"]))
    t_skin = float(np.nanmean(maps["t_skin"]))
    tau_mean = float(maps["tau"].mean())

    for key in case_order:
        if key not in imgs:
            continue
        y = imgs[key]
        png = os.path.join(args.outdir, f"swir_{key}_spp{args.spp}_{stamp}.png")
        png8 = os.path.join(args.outdir, f"swir_{key}_spp{args.spp}_{stamp}_u8.png")
        save_u16_png(png, y, lo, scale)
        save_u8_display(png8, y, clear_with_ac=(key == "clear_off"))
        print(f"wrote {png}  L_offset={lo:.6g} L_scale={scale:.6g}")
        print(f"wrote {png8}")
        meta[key]["png"] = png
        meta[key]["png_u8"] = png8
        imaging_rows.append(
            {
                "case": key,
                "lambda_nm": lam,
                "H_m": geo["H_m"],
                "film_w_m": geo["film_w_m"],
                "film_h_m": geo["film_h_m"],
                "nx": w,
                "ny": h,
                "gsd_m": geo["gsd_m"],
                "gsd_optics_m": geo["gsd_optics_m"],
                "ss": int(ss),
                "downsample_csv": "" if ss <= 1 else os.path.abspath(args.downsample_csv),
                "mtf_csv": "" if mtf_meta is None else os.path.abspath(args.mtf_csv),
                "spp": int(args.spp),
                "seed": int(args.seed),
                "fps": SWIR_FPS,
                "f_m": SWIR_F_M,
                "pitch_m": SWIR_PITCH_M,
                "fnum": SWIR_FNUM,
                "netd_k": SWIR_NETD_K,
                "sun_az_deg": fire.SUN_AZ_DEG,
                "sun_el_deg": fire.SUN_EL_DEG,
                "L_scale": scale,
                "L_offset": lo,
                "png": png,
                "png_u8": png8,
                "npy": meta[key]["npy"],
                "L_mean": meta[key]["mean"],
            }
        )
        has_ac = int(meta[key]["has_aircraft"])
        target_rows.append(
            {
                "case": key,
                "has_aircraft": has_ac,
                "x_m": cx if has_ac else "",
                "y_m": cy if has_ac else "",
                "z_m": rf.AC_Z_M if has_ac else "",
                "heading_rad": HEADING_RAD if has_ac else "",
                "v_mps": V_MPS if has_ac else "",
                "ntri": int(ac["idx"].shape[0]) if has_ac else 0,
                "T_body_K": rf.BODY_T_K if has_ac else "",
                "T_engine_K": rf.ENGINE_T_K if has_ac else "",
                "rho_body": rf.SWIR_BODY_R if has_ac else "",
                "rho_engine": rf.SWIR_ENGINE_R if has_ac else "",
                "size_x_m": sx if has_ac else "",
                "size_y_m": sy if has_ac else "",
                "size_z_m": sz if has_ac else "",
            }
        )
        cm = meta[key]["cloud_mode"]
        scene_rows.append(
            {
                "case": key,
                "cloud_mode": cm,
                "htcp_path": fire.DEFAULT_HTCP,
                "omega_dump": om_dump if cm != "none" else "",
                "g": g if cm != "none" else "",
                "tau_mean": tau_mean if cm != "none" else 0.0,
                "z_bot_m": z_bot if cm != "none" else "",
                "z_top_m": z_top if cm != "none" else "",
                "t_skin_K": t_skin if cm != "none" else "",
                "water_T_K": rf.WATER_T_K,
                "water_rho": float(rf.IR_WATER_ALBEDO[0]),
                "water_size_m": geo["water_m"],
                "gas_1d": 1,
                "place_x": cx,
                "place_y": cy,
            }
        )

    write_csv(os.path.join(args.outdir, "imaging.csv"), imaging_rows)
    write_csv(os.path.join(args.outdir, "target.csv"), target_rows)
    write_csv(os.path.join(args.outdir, "scene.csv"), scene_rows)

    # clear_off acceptance: body-core / sea contrast + edge gradient (detector beauty)
    accept = {}
    if "clear_off" in imgs:
        y = imgs["clear_off"]
        mask = rf.ac_film_mask(ac["verts"], cx, cy, film_h, w, h, pad_m=2.0)
        c = 32
        sea_patches = [y[:c, :c], y[:c, -c:], y[-c:, :c], y[-c:, -c:]]
        sea_m = float(np.mean([p.mean() for p in sea_patches]))
        # body core: film bbox pixels clearly above sea (exclude mixed/MTF skirt)
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

        band = binary_dilation(mask, iterations=2) & ~mask
        edge = gmag[band] if band.any() else gmag[mask]
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
            f"bbox/sea={accept['contrast_bbox_sea']:.3g}  peak/sea={accept['peak_sea']:.3g}  "
            f"body={body_m:.4g} sea={sea_m:.4g}  "
            f"shadow_p1={accept['shadow_p1']:.4g}  "
            f"edge_grad_mean={accept['edge_grad_mean']:.4g}  "
            f"ok_contrast={accept['ok_contrast']}"
        )
        with open(os.path.join(args.outdir, "accept_clear_off.json"), "w") as f:
            json.dump(accept, f, indent=2)
            f.write("\n")

    for k in meta:
        meta[k]["png"] = os.path.join(args.outdir, f"swir_{k}_spp{args.spp}_{stamp}.png")
        meta[k]["png_u8"] = os.path.join(args.outdir, f"swir_{k}_spp{args.spp}_{stamp}_u8.png")
    rec = {
        "stamp": stamp,
        "geometry": geo,
        "lambda_nm": lam,
        "spp": int(args.spp),
        "seed": int(args.seed),
        "L_offset": lo,
        "L_scale": scale,
        "L_calibration": "u16_shared_0_stack_p99.5; beauty=SS_MTF_box2x; RTE_pre_MTF_in_ss_npy",
        "downsample_csv": None if ss <= 1 else os.path.abspath(args.downsample_csv),
        "mtf_csv": None if mtf_meta is None else os.path.abspath(args.mtf_csv),
        "ss": int(ss),
        "swir_rho_body": float(rf.SWIR_BODY_R),
        "swir_rho_engine": float(rf.SWIR_ENGINE_R),
        "water_rho": float(rf.IR_WATER_ALBEDO[0]),
        "note": "No sky pad. Sea rho=0.04 matches htrdr IR LUT. MTF is detector/optical only.",
        "detector_nx": int(w),
        "detector_ny": int(h),
        "cases": meta,
        "mie": {"Cs": Cs, "Ca": Ca, "g": g, "omega_dump": om_dump},
        "imaging_csv": os.path.join(args.outdir, "imaging.csv"),
        "target_csv": os.path.join(args.outdir, "target.csv"),
        "scene_csv": os.path.join(args.outdir, "scene.csv"),
        "accept_clear_off": accept,
    }
    with open(os.path.join(args.outdir, "record.json"), "w") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
    print("wrote", os.path.join(args.outdir, "record.json"))
    for k, m in meta.items():
        print(f"  {k:10s} mean={m['mean']:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
