#!/usr/bin/env python3
"""An-124 12-panel: vis/MWIR/LWIR × (clear+AC, cloud+AC, cloud only, Δ).

Turns the cloud off the same way as render_747_cloud / --no-clouds: zero voxel
ka/ks on the host, keep 1D gas. No kernel change.

Reuses existing MWIR/LWIR cloudy 2048 spp npy. Writes a new outdir.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
import sys

import numpy as np
import torch
import torch.utils.cpp_extension
from PIL import Image, ImageDraw, ImageFont

import render_fire_shahed_bands as rb
from render_fire_shahed_bands import (
    AC_Z_M,
    BANDS,
    BODY_R,
    BODY_T_K,
    DELTA_SMOOTH_M,
    ENGINE_R,
    ENGINE_T_K,
    IR_WATER_ALBEDO,
    ORTHO_HEIGHT_M,
    ORTHO_Z_M,
    ROOT,
    TRACER,
    WATER_SIZE_M,
    WATER_T_K,
    ac_film_mask,
    analog_column,
    apply_ir_airframe,
    attach_mesh,
    htpp_xyz_to_srgb,
    label_img,
    lum,
    make_buffers,
    mie_ok,
    planck_B,
    render_ortho,
    set_airframe_temps,
    smooth_contrast_rgb,
    to_png,
    water_quad,
)
import aircraft_mesh as acmesh
import fire_scene as fire
import mountain_gs_scene as mgs
from render_mountain_gs_gpu import make_conf
from threedgptir_tracer.atmosphere import SPECTRAL_LW, SPECTRAL_SW_CIE_XYZ, build_cie_xyz_tables
from threedgptir_tracer.tracer import load_threedgptir_plugin
import threedgptir_tracer.tracer as tracer_mod

REUSE_CLOUDY = os.path.join(ROOT, "scenes", "fire_gs", "transport_ir_spp2048")
REUSE_NPY = {
    ("mwir", "cloud_ac"): os.path.join(REUSE_CLOUDY, "gpu_mwir_uav_spp2048.npy"),
    ("mwir", "cloud"): os.path.join(REUSE_CLOUDY, "gpu_mwir_nouav_spp2048.npy"),
    ("lwir", "cloud_ac"): os.path.join(REUSE_CLOUDY, "gpu_lwir_uav_spp2048.npy"),
    ("lwir", "cloud"): os.path.join(REUSE_CLOUDY, "gpu_lwir_nouav_spp2048.npy"),
}

CASES = (
    ("nocloud_ac", True, True, "clear + An-124"),
    ("cloud_ac", False, True, "cloud + An-124"),
    ("cloud", False, False, "cloud only"),
)


def vis_png(xyz: np.ndarray, white: float) -> np.ndarray:
    return (htpp_xyz_to_srgb(xyz, exposure=0.2, white=white) * 255.0 + 0.5).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--outdir",
        default=os.path.join(ROOT, "scenes", "fire_gs", "an124_clear_cloud_spp2048"),
    )
    p.add_argument("--spp", type=int, default=2048)
    p.add_argument("--film", nargs=2, type=int, default=[512, 512])
    p.add_argument("--max-bounces", type=int, default=256)
    p.add_argument("--brick", type=int, default=8)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    w, h = int(args.film[0]), int(args.film[1])
    spp = int(args.spp)
    bands = [dict(b) for b in BANDS if b["key"] in ("vis", "mwir", "lwir")]
    for b in bands:
        b["spp"] = spp

    print("=== Mie LUT probe ===")
    mie_probe = {}
    for b in bands:
        mie = mgs.fetch_mie(b["lambda_nm"])
        mie_probe[b["key"]] = {
            "lambda_nm": b["lambda_nm"],
            "Cs": float(mie["Cs"]),
            "Ca": float(mie["Ca"]),
            "g": float(mie["g"]),
            "ok": mie_ok(mie),
        }
        print(f"  {b['key']} λ={b['lambda_nm']:.0f} nm  ok={mie_probe[b['key']]['ok']}")

    print("=== FIRE native voxels @ 550 nm ===")
    vox550 = mgs.load_htcp_voxels(fire.DEFAULT_HTCP, wavelength_nm=550.0)
    place = rb.pick_mean_column(vox550)
    cx, cy = float(place["x"]), float(place["y"])
    place["z"] = AC_Z_M
    ac = set_airframe_temps(acmesh.make_an124_coarse_mesh((cx, cy, AC_Z_M), heading_rad=0.0))
    ntri_ac = int(ac["idx"].shape[0])
    size_ac = ac["verts"].max(0) - ac["verts"].min(0)
    mask = ac_film_mask(ac["verts"], cx, cy, ORTHO_HEIGHT_M, w, h)
    cam_center = np.array([cx, cy, ORTHO_Z_M], np.float32)
    az = math.radians(fire.SUN_AZ_DEG)
    el = math.radians(fire.SUN_EL_DEG)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(el), math.sin(el))
    # sun dir: same convention as shahed script
    sun = (
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    )
    print(f"ortho center={cam_center.tolist()} height={ORTHO_HEIGHT_M}  An-124 ntri={ntri_ac}")

    cie_tables = build_cie_xyz_tables()
    sw = {"vis": mgs.load_ecrad_sw_visible()}
    lw = mgs.load_ecrad_lw()

    device = torch.device("cuda")
    torch.zeros(1, device=device)
    conf = make_conf()
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
    images = {}
    meta = {"stamp": stamp, "spp": spp, "place": place, "reuse": {}, "bands": []}

    for band in bands:
        key = band["key"]
        lam = float(band["lambda_nm"])
        mie = {k: mie_probe[key][k] for k in ("Cs", "Ca", "g")}
        print(f"=== {band['label']} ===")
        voxels = vox550 if abs(lam - 550.0) < 1e-6 else mgs.load_htcp_voxels(fire.DEFAULT_HTCP, wavelength_nm=lam)
        gas = (
            mgs.load_ecrad_gas_1d_lw(wavelength_m=lam * 1.0e-9)
            if band["mode"] == SPECTRAL_LW
            else mgs.load_ecrad_gas_1d(wavelength_m=lam * 1.0e-9)
        )
        water_alb = fire.WATER_BODY_ALBEDO if band["mode"] == SPECTRAL_SW_CIE_XYZ else IR_WATER_ALBEDO
        water = water_quad(cx, cy, WATER_SIZE_M, water_alb, WATER_T_K)
        ac_use = apply_ir_airframe(ac) if band["mode"] == SPECTRAL_LW else ac
        mesh_on = mgs.concat_meshes(water, ac_use)
        tau, tr = analog_column(
            voxels["ka"], voxels["ks"], voxels["origin"], voxels["spacing"],
            np.array([cx, cy, AC_Z_M], np.float64), (0.0, 0.0, 1.0),
            tmax=3.0e3, z_iface=voxels.get("z_iface"), absorb_only=False,
        )
        print(f"  analog cloudy kext τ={tau:.3f} Tr={tr:.4g}  B_body={planck_B(lam*1e-9, BODY_T_K):.4g}")

        band_imgs = {}
        for ckey, no_clouds, with_ac, _clab in CASES:
            npy = os.path.join(args.outdir, f"gpu_{key}_{ckey}_spp{spp}.npy")
            src = REUSE_NPY.get((key, ckey))
            if os.path.isfile(npy):
                img = np.load(npy)
                print("loaded", npy)
            elif src and os.path.isfile(src):
                shutil.copy2(src, npy)
                img = np.load(npy)
                meta["reuse"][f"{key}_{ckey}"] = src
                print("reused", src, "->", npy)
            else:
                mesh = mesh_on if with_ac else water
                buf = make_buffers(
                    device, voxels, gas, mesh, band, sun, sw, lw, cie_tables,
                    args.brick, args.max_bounces, mie["g"], mie, no_clouds=no_clouds,
                )
                print(
                    f"-- {key} {ckey} spp={spp} no_clouds={no_clouds} "
                    f"tris={mesh['idx'].shape[0]} λ={lam:.0f}"
                )
                img = render_ortho(
                    tracer, buf, cam_center, ORTHO_HEIGHT_M, w, h, spp, device,
                    max(int(args.chunk), 1), args.seed,
                )
                np.save(npy, img.astype(np.float32))
                print("wrote", npy)
                del buf
                torch.cuda.empty_cache()
            band_imgs[ckey] = img
            images[(key, ckey)] = img

        is_cie = band["mode"] == SPECTRAL_SW_CIE_XYZ
        y_on = lum(band_imgs["cloud_ac"], is_cie)
        y_off = lum(band_imgs["cloud"], is_cie)
        px_m = float(ORTHO_HEIGHT_M) / float(w)
        d_rgb, c_scale = smooth_contrast_rgb(y_on, y_off, px_m, mask=mask)
        dpng = os.path.join(args.outdir, f"delta_{key}_{stamp}.png")
        Image.fromarray(d_rgb).save(dpng)
        images[(key, "delta")] = d_rgb
        meta["bands"].append({
            "band": key,
            "analog_tau": tau,
            "analog_Tr": tr,
            "delta_C_scale": c_scale,
            "means": {ck: float(band_imgs[ck].mean()) for ck, *_ in CASES},
        })
        print(f"  {key} cloudy means ac={band_imgs['cloud_ac'].mean():.4g} "
              f"cloud={band_imgs['cloud'].mean():.4g}  ΔL/L scale={c_scale:.4g}")

    vis_white = float(np.percentile(images[("vis", "nocloud_ac")][..., 1], 99.5))
    print("vis htpp white from nocloud_ac Y p995", vis_white)

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    cols = (
        ("nocloud_ac", "clear + An-124"),
        ("cloud_ac", "cloud + An-124"),
        ("cloud", "cloud only"),
        ("delta", None),
    )
    png_cells = {}
    captions = {}
    for band in bands:
        key = band["key"]
        is_cie = band["mode"] == SPECTRAL_SW_CIE_XYZ
        for ckey, clab in cols:
            if ckey == "delta":
                rgb = images[(key, "delta")]
                scale = next(b["delta_C_scale"] for b in meta["bands"] if b["band"] == key)
                cap = f"ΔL/L  Gauss {DELTA_SMOOTH_M:.0f} m  ±{100.0 * scale:.3f}%"
            else:
                img = images[(key, ckey)]
                if is_cie:
                    rgb = vis_png(img, vis_white)
                else:
                    rgb = to_png(img, False)
                png = os.path.join(args.outdir, f"render_{key}_{ckey}_spp{spp}_{stamp}.png")
                Image.fromarray(rgb).save(png)
                cap = clab
            png_cells[(key, ckey)] = rgb
            captions[(key, ckey)] = cap

    rows = [b["key"] for b in bands]
    labels = {b["key"]: b["label"] for b in BANDS}
    sample = next(iter(png_cells.values()))
    cw, ch = sample.shape[1], sample.shape[0] + 36
    grid = Image.new("RGB", (cw * 4 + 20, ch * 3 + 56), (12, 12, 12))
    dr = ImageDraw.Draw(grid)
    dr.text(
        (8, 6),
        f"FIRE An-124 nadir {ORTHO_HEIGHT_M:.0f} m  z_ac={AC_Z_M:.0f} m  "
        f"tau550={place['tau']:.2f}  spp={spp}  {stamp}",
        fill=(220, 220, 220),
        font=font,
    )
    dr.text(
        (8, 26),
        "Col1 = voxel ka/ks zeroed, 1D gas kept (no kernel change).  "
        "VIS shares nocloud white.  Col4 = cloudy with minus without.",
        fill=(180, 180, 180),
        font=font,
    )
    for r, key in enumerate(rows):
        for c, (ckey, _clab) in enumerate(cols):
            im = label_img(png_cells[(key, ckey)], f"{labels[key]}  {captions[(key, ckey)]}")
            grid.paste(im, (8 + c * cw, 48 + r * ch))
    gpath = os.path.join(args.outdir, f"compare_an124_clear_cloud_12_{stamp}.png")
    grid.save(gpath)
    print("wrote", gpath)

    meta["vis_white"] = vis_white
    meta["aircraft"] = {
        "type": "An-124",
        "ntri": ntri_ac,
        "size_m": size_ac.tolist(),
        "T_body_K": BODY_T_K,
        "T_engine_K": ENGINE_T_K,
        "albedo_body": BODY_R,
        "albedo_engine": ENGINE_R,
    }
    with open(os.path.join(args.outdir, "record.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print("wrote", os.path.join(args.outdir, "record.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
