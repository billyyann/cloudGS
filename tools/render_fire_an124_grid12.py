#!/usr/bin/env python3
"""16-panel An-124: vis / SWIR / MWIR / LWIR × clear+ac / cloud+ac / cloud / delta.

Keeps existing spp2048 cloudy vis/MWIR/LWIR npy. Clear-sky and SWIR are
re-rendered with a finite JONSWAP sea patch (ntri<=65536, no wrap).
Voxel ka/ks zeroed for clear sky; 1D gas stays. GPU OptiX only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
import torch
import torch.utils.cpp_extension
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "gpu"))
sys.path.insert(0, os.path.dirname(__file__))

import fire_scene as fire  # noqa: E402
import mountain_gs_scene as mgs  # noqa: E402
import render_fire_shahed_bands as rf  # noqa: E402
from render_mountain_gs_gpu import make_conf  # noqa: E402
from threedgptir_tracer.atmosphere import SPECTRAL_LW, SPECTRAL_SW_CIE_XYZ  # noqa: E402
from threedgptir_tracer.tracer import load_threedgptir_plugin  # noqa: E402
import threedgptir_tracer.tracer as tracer_mod  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
GRID_BANDS = ("vis", "swir", "mwir", "lwir")
CLOUD_REUSE = {
    ("vis", "uav"): "gpu_vis_uav_spp2048.npy",
    ("vis", "nouav"): "gpu_vis_nouav_spp2048.npy",
    ("mwir", "uav"): "gpu_mwir_uav_spp2048.npy",
    ("mwir", "nouav"): "gpu_mwir_nouav_spp2048.npy",
    ("lwir", "uav"): "gpu_lwir_uav_spp2048.npy",
    ("lwir", "nouav"): "gpu_lwir_nouav_spp2048.npy",
}
# Flat 2-triangle water had no chop; re-render these with the JONSWAP patch.
FORCE_SEA = {
    ("vis", "nocloud_uav"),
    ("mwir", "nocloud_uav"),
    ("lwir", "nocloud_uav"),
}


def zero_cloud(voxels: dict) -> dict:
    out = dict(voxels)
    out["ka"] = np.zeros_like(voxels["ka"])
    out["ks"] = np.zeros_like(voxels["ks"])
    return out


def band_by_key(key: str) -> dict:
    for b in rf.BANDS:
        if b["key"] == key:
            return dict(b)
    raise KeyError(key)


def npy_path(outdir: str, key: str, tag: str, spp: int) -> str:
    return os.path.join(outdir, f"gpu_{key}_{tag}_spp{spp}.npy")


def save_png(img: np.ndarray, cie: bool, path: str) -> np.ndarray:
    rgb = rf.to_png(img, cie)
    Image.fromarray(rgb).save(path)
    return rgb


def render_one(tracer, buf, mesh, cam, w, h, spp, chunk, seed, device, tag: str):
    rf.attach_mesh(buf, mesh, device)
    print(
        f"-- {tag} spp={spp} tris={mesh['idx'].shape[0]}  "
        f"T=[{float(mesh['temp'].min()):.0f},{float(mesh['temp'].max()):.0f}] K"
    )
    return rf.render_ortho(tracer, buf, cam, rf.ORTHO_HEIGHT_M, w, h, spp, device, chunk, seed)


def assemble_grid(outdir: str, stamp: str, spp: int, place: dict, cells: dict, captions: dict) -> str:
    labels = {b["key"]: b["label"] for b in rf.BANDS}
    cols = (
        ("nocloud_uav", "no cloud + An-124"),
        ("uav", "cloud + An-124"),
        ("nouav", "cloud only"),
        ("delta", None),
    )
    sample = next(iter(cells.values()))
    cw, ch = sample.shape[1], sample.shape[0] + 36
    rows = list(GRID_BANDS)
    grid = Image.new("RGB", (cw * 4 + 20, ch * len(rows) + 56), (12, 12, 12))
    dr = ImageDraw.Draw(grid)
    font = ImageFont.truetype(FONT, 14)
    dr.text(
        (8, 6),
        f"FIRE An-124 nadir {rf.ORTHO_HEIGHT_M:.0f} m  z_ac={rf.AC_Z_M:.0f} m  "
        f"tau550={place['tau']:.2f}  spp={spp}  {stamp}",
        fill=(220, 220, 220),
        font=font,
    )
    dr.text(
        (8, 26),
        f"Col4 = cloudy ΔL/L Gauss {rf.DELTA_SMOOTH_M:.0f} m. "
        f"Clear sea = JONSWAP+swell, ntri≤65536, blue_water specular.",
        fill=(180, 180, 180),
        font=font,
    )
    for r, key in enumerate(rows):
        for c, (tag, clab) in enumerate(cols):
            rgb = cells.get((key, tag))
            if rgb is None:
                continue
            cap = captions.get((key, tag), clab)
            im = rf.label_img(rgb, f"{labels[key]}  {cap}")
            grid.paste(im, (8 + c * cw, 48 + r * ch))
    path = os.path.join(outdir, f"compare_an124_grid16_{stamp}.png")
    grid.save(path)
    print("wrote", path)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="An-124 16-panel vis/SWIR/MWIR/LWIR grid, spp 2048")
    p.add_argument(
        "--outdir",
        default=os.path.join(ROOT, "scenes", "fire_gs", "transport_ir_spp2048"),
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
    chunk = max(int(args.chunk), 1)

    print("=== Mie LUT probe ===")
    mie_probe = {}
    for b in rf.BANDS:
        mie = mgs.fetch_mie(b["lambda_nm"])
        mie_probe[b["key"]] = {
            "Cs": float(mie["Cs"]),
            "Ca": float(mie["Ca"]),
            "g": float(mie["g"]),
            "ok": rf.mie_ok(mie),
        }
        print(f"  {b['key']} λ={b['lambda_nm']:.0f} nm  ok={mie_probe[b['key']]['ok']}")

    print("=== FIRE native voxels @ 550 nm ===")
    vox550 = mgs.load_htcp_voxels(fire.DEFAULT_HTCP, wavelength_nm=550.0)
    place = rf.pick_mean_column(vox550)
    cx, cy = float(place["x"]), float(place["y"])
    place["z"] = rf.AC_Z_M
    ac = rf.set_airframe_temps(rf.acmesh.make_an124_coarse_mesh((cx, cy, rf.AC_Z_M), heading_rad=0.0))
    ntri_ac = int(ac["idx"].shape[0])
    if not (32 < ntri_ac <= 65536):
        raise RuntimeError(f"An-124 coarse mesh tris={ntri_ac} not in (32, 65536]")
    mask = rf.ac_film_mask(ac["verts"], cx, cy, rf.ORTHO_HEIGHT_M, w, h)
    cam_center = np.array([cx, cy, rf.ORTHO_Z_M], np.float32)
    az = math.radians(fire.SUN_AZ_DEG)
    el = math.radians(fire.SUN_EL_DEG)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    cie_tables = rf.build_cie_xyz_tables()
    sw_vis = mgs.load_ecrad_sw_visible()
    lw = mgs.load_ecrad_lw()
    sw = {"vis": sw_vis}

    device = torch.device("cuda")
    torch.zeros(1, device=device)
    conf = make_conf()
    print("loading PTIR OptiX plugin...")
    load_threedgptir_plugin(conf)
    plugin = tracer_mod._threedgptir_plugin
    tracer = plugin.OptixTracer(
        rf.TRACER,
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

    jobs = []
    for key in GRID_BANDS:
        jobs.append((key, False, "uav"))
        jobs.append((key, False, "nouav"))
        jobs.append((key, True, "nocloud_uav"))

    record = {"stamp": stamp, "spp": spp, "place": place, "kept": [], "rendered": []}
    hdr = {}

    for key, nocloud, tag in jobs:
        reuse_name = CLOUD_REUSE.get((key, tag))
        dest = npy_path(args.outdir, key, tag, spp)
        force = (key, tag) in FORCE_SEA or key == "swir"
        if reuse_name and not force:
            src = os.path.join(args.outdir, reuse_name)
            if not os.path.isfile(src):
                raise FileNotFoundError(f"keep current results: missing {src}")
            hdr[(key, tag)] = src
            record["kept"].append(src)
            print(f"keep {src}")
            continue
        if os.path.isfile(dest) and not force:
            hdr[(key, tag)] = dest
            record["kept"].append(dest)
            print(f"reuse {dest}")
            continue

        band = band_by_key(key)
        lam = float(band["lambda_nm"])
        if not mie_probe[key]["ok"]:
            raise RuntimeError(f"Mie LUT unusable for {key}")
        print(f"=== {band['label']}  {tag}  nocloud={nocloud} ===")
        voxels = vox550 if abs(lam - 550.0) < 1e-6 else mgs.load_htcp_voxels(fire.DEFAULT_HTCP, wavelength_nm=lam)
        if nocloud:
            voxels = zero_cloud(voxels)
            print("no-clouds: voxel ka/ks zeroed, 1D gas kept")
        gas = (
            mgs.load_ecrad_gas_1d_lw(wavelength_m=lam * 1.0e-9)
            if band["mode"] == SPECTRAL_LW
            else mgs.load_ecrad_gas_1d(wavelength_m=lam * 1.0e-9)
        )
        water = rf.local_sea_mesh(cx, cy, rf.WATER_SIZE_M, temp=rf.WATER_T_K)
        ac_use = rf.apply_ir_airframe(ac) if band["mode"] == SPECTRAL_LW else ac
        mesh_on = mgs.concat_meshes(water, ac_use)
        ntri_on = int(mesh_on["idx"].shape[0])
        if not (32 < ntri_on <= 65536):
            raise RuntimeError(f"sea+An-124 tris={ntri_on} not in (32, 65536]")
        mesh = mesh_on if tag in ("uav", "nocloud_uav") else water
        mie = mie_probe[key]
        buf = rf.make_buffers(
            device,
            voxels,
            gas,
            mesh,
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
        img = render_one(
            tracer, buf, mesh, cam_center, w, h, spp, chunk, args.seed, device, f"{key} {tag}"
        )
        np.save(dest, img.astype(np.float32))
        png = os.path.join(args.outdir, f"render_{key}_{tag}_spp{spp}_{stamp}.png")
        save_png(img, band["mode"] == SPECTRAL_SW_CIE_XYZ, png)
        print("wrote", dest, png)
        hdr[(key, tag)] = dest
        record["rendered"].append(dest)
        del img, buf
        torch.cuda.empty_cache()

    cells = {}
    captions = {}
    px_m = float(rf.ORTHO_HEIGHT_M) / float(w)
    for key in GRID_BANDS:
        band = band_by_key(key)
        cie = band["mode"] == SPECTRAL_SW_CIE_XYZ
        on = np.load(hdr[(key, "uav")])
        off = np.load(hdr[(key, "nouav")])
        clear = np.load(hdr[(key, "nocloud_uav")])
        cells[(key, "uav")] = rf.to_png(on, cie)
        cells[(key, "nouav")] = rf.to_png(off, cie)
        cells[(key, "nocloud_uav")] = rf.to_png(clear, cie)
        y_on = rf.lum(on, cie)
        y_off = rf.lum(off, cie)
        d_rgb, c_scale = rf.smooth_contrast_rgb(y_on, y_off, px_m, mask=mask)
        cells[(key, "delta")] = d_rgb
        captions[(key, "delta")] = f"ΔL/L  Gauss {rf.DELTA_SMOOTH_M:.0f} m  ±{100.0 * c_scale:.3f}%"
        Image.fromarray(d_rgb).save(os.path.join(args.outdir, f"delta_{key}_grid12_{stamp}.png"))
        Image.fromarray(cells[(key, "nocloud_uav")]).save(
            os.path.join(args.outdir, f"render_{key}_nocloud_uav_spp{spp}_{stamp}.png")
        )
        Image.fromarray(cells[(key, "uav")]).save(
            os.path.join(args.outdir, f"render_{key}_uav_spp{spp}_{stamp}.png")
        )
        Image.fromarray(cells[(key, "nouav")]).save(
            os.path.join(args.outdir, f"render_{key}_nouav_spp{spp}_{stamp}.png")
        )

    gpath = assemble_grid(args.outdir, stamp, spp, place, cells, captions)
    record["grid"] = gpath
    rpath = os.path.join(args.outdir, f"record_grid12_{stamp}.json")
    with open(rpath, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print("wrote", rpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
