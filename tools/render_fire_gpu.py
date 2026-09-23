#!/usr/bin/env python3
"""GPU path-trace FIRE marine Sc over seawater.

Writes only under scenes/fire_gs/. Does not touch mountain_gs or sea_gs.
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import time

import numpy as np
import torch
import torch.utils.cpp_extension
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACER = os.path.join(ROOT, "gpu", "threedgptir_tracer")
sys.path.insert(0, os.path.join(ROOT, "gpu"))
sys.path.insert(0, os.path.dirname(__file__))

_local_lib = os.path.join(ROOT, "local", "lib")
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _local_lib not in _ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = _local_lib + (":" + _ld if _ld else "")

import fire_scene as fire  # noqa: E402
from render_mountain_gs_gpu import (  # noqa: E402
    diagnose_cloud_sun_tr,
    load_htcp_bin,
    make_conf,
    make_ortho_rays,
    make_persp_rays,
)
from render_sea_gpu import bind_sw, htrdr_xyz  # noqa: E402
from threedgptir_tracer.atmosphere import (  # noqa: E402
    SPECTRAL_LW,
    SPECTRAL_SW,
    SPECTRAL_SW_CIE_XYZ,
    AtmosphereBuffers,
    build_cie_xyz_tables,
    htpp_xyz_to_srgb,
    trace_atmosphere,
)

CAM_POS = fire.CAM_POS
CAM_TGT = fire.CAM_TGT
CAM_UP = fire.CAM_UP
CAM_FOV = fire.CAM_FOV
ORTHO_CENTER = fire.ORTHO_CENTER
ORTHO_HEIGHT = fire.ORTHO_HEIGHT
ORTHO_LOOK = np.array([0.0, 0.0, -1.0], np.float32)
ORTHO_UP = np.array([0.0, 1.0, 0.0], np.float32)
SUN_AZ_DEG = fire.SUN_AZ_DEG
SUN_EL_DEG = fire.SUN_EL_DEG
HTRDR_ORTHO = "htrdr_ref_fire_ortho_512x512x256"
HTRDR_OBLIQUE = "htrdr_ref_fire_oblique_1280x720x256"


def write_compare(outdir: str, camera: str, gpu_xyz: np.ndarray, spp: int) -> str | None:
    stem = HTRDR_ORTHO if camera == "ortho" else HTRDR_OBLIQUE
    txt = os.path.join(outdir, f"{stem}.txt")
    png_l = os.path.join(outdir, f"{stem}.png")
    h, w = gpu_xyz.shape[:2]
    if not os.path.isfile(txt) or os.path.getsize(txt) < 4096:
        print(f"no finished htrdr txt yet ({txt}); skip compare")
        return None
    try:
        href = htrdr_xyz(txt, h, w)
    except ValueError as exc:
        print(f"htrdr txt not ready ({txt}): {exc}; skip compare")
        return None
    if href.size != h * w * 3:
        print(f"htrdr txt size mismatch ({txt}); skip compare")
        return None
    white = float(np.percentile(href[..., 1], 99.5))
    gpu_rgb = (htpp_xyz_to_srgb(gpu_xyz, exposure=0.2, white=white) * 255.0 + 0.5).astype(np.uint8)
    if os.path.isfile(png_l):
        left = Image.open(png_l).convert("RGB")
        if left.size != (w, h):
            left = left.resize((w, h), Image.NEAREST)
    else:
        left = Image.fromarray(
            (htpp_xyz_to_srgb(href, exposure=0.2, white=white) * 255.0 + 0.5).astype(np.uint8)
        )
    gy, hy = gpu_xyz[..., 1], href[..., 1]
    print(
        f"compare {camera}  htrdr Y mean/p50/p995={hy.mean():.4f}/{np.median(hy):.4f}/{white:.4f}  "
        f"GPU Y mean/p50/p995={gy.mean():.4f}/{np.median(gy):.4f}/{float(np.percentile(gy, 99.5)):.4f}  "
        f"meanY GPU/htrdr={gy.mean() / max(hy.mean(), 1e-12):.3f}"
    )
    from PIL import ImageDraw

    canvas = Image.new("RGB", (w * 2 + 24, h + 48), (18, 18, 18))
    canvas.paste(left, (8, 40))
    canvas.paste(Image.fromarray(gpu_rgb), (w + 16, 40))
    d = ImageDraw.Draw(canvas)
    view = "俯视 ortho 512×512" if camera == "ortho" else "斜视 云下看海 1280×720"
    d.text((8, 8), f"htrdr CPU  {view}  spp256", fill=(220, 220, 220))
    d.text((w + 16, 8), f"GPU  同相机  spp{spp}  (htpp white=htrdr Y p995)", fill=(220, 220, 220))
    out = os.path.join(outdir, f"compare_fire_{camera}_htrdr256_gpu{spp}.png")
    canvas.save(out)
    print("wrote", out)
    Image.fromarray(gpu_rgb).save(
        os.path.join(outdir, f"render_fire_{camera}_cie_spp{spp}_htpp_htrdrwhite.png")
    )
    return out


def render_xyz(
    tracer,
    buffers,
    camera: str,
    w: int,
    h: int,
    spp: int,
    device,
    chunk: int,
    seed: int,
    ortho_center,
    ortho_height: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nchunk = (spp + chunk - 1) // chunk
    sum_xyz = np.zeros((h, w, 3), np.float64)
    if camera == "ortho":
        print(
            f"GPU FIRE ortho {w}x{h} spp={spp} height={ORTHO_HEIGHT} "
            f"center={ORTHO_CENTER.tolist()} sun az/el={SUN_AZ_DEG}/{SUN_EL_DEG}"
        )
    else:
        print(
            f"GPU FIRE under-Sc persp {w}x{h} spp={spp} fov={CAM_FOV} "
            f"pos={CAM_POS.tolist()} tgt={CAM_TGT.tolist()} sun az/el={SUN_AZ_DEG}/{SUN_EL_DEG}"
        )
    t0 = time.perf_counter()
    for c in range(nchunk):
        n = min(chunk, spp - c * chunk)
        if camera == "ortho":
            ori, drc = make_ortho_rays(
                w, h, ORTHO_CENTER, ORTHO_HEIGHT, ORTHO_UP, ORTHO_LOOK, n, device, rng=rng
            )
        else:
            ori, drc = make_persp_rays(w, h, CAM_POS, CAM_TGT, CAM_UP, CAM_FOV, n, device, rng=rng)
        torch.cuda.synchronize()
        out = trace_atmosphere(tracer, ori, drc, buffers, frame_number=c)
        torch.cuda.synchronize()
        rgb = out[10].float().cpu().numpy()
        sum_xyz += rgb.sum(axis=0)
        print(f"  {camera} chunk {c + 1}/{nchunk} meanY={float(rgb[..., 1].mean()):.4f}")
        del ori, drc, out, rgb
        torch.cuda.empty_cache()
    img = (sum_xyz / float(spp)).astype(np.float32)
    print(
        f"  {camera} done {time.perf_counter() - t0:.1f}s  "
        f"Y mean/p50={float(img[..., 1].mean()):.4f}/{float(np.median(img[..., 1])):.4f}"
    )
    return img


def main() -> int:
    p = argparse.ArgumentParser(description="Render htrdr L25_FIRE.sh on the GPU atmosphere path")
    p.add_argument("--outdir", default=os.path.join(ROOT, "scenes", "fire_gs"))
    p.add_argument("--spp", type=int, default=4)
    p.add_argument("--film", nargs=2, type=int, metavar=("W", "H"), default=None)
    p.add_argument("--camera", choices=("oblique", "ortho", "both"), default="oblique")
    p.add_argument("--compare-only", action="store_true")
    p.add_argument("--no-compare", action="store_true")
    p.add_argument("--max-bounces", type=int, default=256)
    p.add_argument("--mie-g", type=float, default=0.865)
    p.add_argument("--brick", type=int, default=8)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--rebuild-scene", action="store_true")
    p.add_argument("--no-clouds", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cams = ("ortho", "oblique") if args.camera == "both" else (args.camera,)
    if args.compare_only:
        n = 0
        for cam in cams:
            npy = os.path.join(args.outdir, f"gpu_xyz_fire_{cam}_cie.npy")
            if not os.path.isfile(npy):
                print("missing", npy)
                continue
            write_compare(args.outdir, cam, np.load(npy), args.spp)
            n += 1
        return 0 if n else 1

    assets = fire.ensure_assets(outdir=args.outdir, rebuild=args.rebuild_scene)
    gas = assets["gas"]
    mesh = assets["mesh"]
    water = assets["water"]
    ox, x1, oy, y1 = assets["xy"]
    ortho_center = ORTHO_CENTER
    ortho_height = ORTHO_HEIGHT
    print(f"FIRE XY tile [{ox:.0f},{x1:.0f}]x[{oy:.0f},{y1:.0f}]  ortho match htrdr height={ortho_height}")

    ka, ks, temp, origin, spacing, z_iface, _xh = load_htcp_bin(assets["voxels_bin"])
    print(f"voxels {ks.shape} origin={origin} dx={spacing} z_iface={'yes' if z_iface is not None else 'no'}")
    if args.no_clouds:
        ka = np.zeros_like(ka)
        ks = np.zeros_like(ks)
        print("no-clouds: voxel ka/ks zeroed")

    az = math.radians(SUN_AZ_DEG)
    el = math.radians(SUN_EL_DEG)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    diagnose_cloud_sun_tr(ka, ks, origin, spacing, sun, z_iface=z_iface)

    device = torch.device("cuda")
    torch.zeros(1, device=device)
    conf = make_conf()
    from threedgptir_tracer.tracer import load_threedgptir_plugin
    import threedgptir_tracer.tracer as tracer_mod
    import mountain_gs_scene as mgs

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
    print("OptixTracer ready  ground: sea_3m blue_water specular (GS off)")

    mode = SPECTRAL_SW_CIE_XYZ
    buffers = AtmosphereBuffers.from_numpy(
        device=device,
        voxel_ka=ka,
        voxel_ks=ks,
        voxel_temp=temp,
        voxel_origin=tuple(origin.tolist()),
        voxel_spacing=tuple(np.asarray(spacing, np.float32).tolist()),
        brick=int(args.brick),
        gas_z=gas["z"],
        gas_ka=gas["ka"],
        gas_ks=gas["ks"],
        gas_temp=gas["temp"],
        sun_dir=sun,
        sun_radiance=24.0,
        sun_cos_half=math.cos(4.6524e-3),
        spectral_mode=mode,
        wavelength_m=5.5e-7,
        mie_omega=1.0,
        mie_g=args.mie_g,
        path_tmax=1.2e5,
        max_bounces=args.max_bounces,
        voxel_z_iface=z_iface,
    )
    cie = build_cie_xyz_tables()
    buffers.cie_cdf = torch.as_tensor(cie["cdf"], dtype=torch.float32, device=device)
    buffers.cie_rcp_integral = torch.as_tensor(cie["rcp_integral"], dtype=torch.float32, device=device)
    buffers.cie_lambda0_nm = float(cie["lambda0_nm"])
    buffers.cie_dlambda_nm = float(cie["dlambda_nm"])
    bind_sw(buffers, mgs.load_ecrad_sw_visible(), device)
    buffers.brdf_lambda_nm = torch.as_tensor(water["lambda_nm"], dtype=torch.float32, device=device)
    buffers.brdf_R = torch.as_tensor(water["R"], dtype=torch.float32, device=device)
    buffers.particle_density = torch.zeros((0, 12), dtype=torch.float32, device=device)
    buffers.particle_material = torch.zeros((0, 5), dtype=torch.float32, device=device)
    buffers.particle_radiance = torch.zeros((0, 1), dtype=torch.float32, device=device)
    buffers.particle_shading_normal = torch.zeros((0, 3), dtype=torch.float32, device=device)
    buffers.mesh_verts = torch.from_numpy(np.ascontiguousarray(mesh["verts"])).to(device)
    buffers.mesh_idx = torch.from_numpy(np.ascontiguousarray(mesh["idx"])).to(device)
    buffers.mesh_albedo = torch.from_numpy(np.ascontiguousarray(mesh["albedo"])).to(device)
    buffers.mesh_temp = torch.from_numpy(np.ascontiguousarray(mesh["temp"])).to(device)
    buffers.mesh_mat_id = torch.from_numpy(np.ascontiguousarray(mesh["mat_id"])).to(device)
    print(f"mesh tris={mesh['idx'].shape[0]}  voxel ks max/mean={float(np.max(ks)):.4g}/{float(np.mean(ks)):.4g}")

    spp = int(args.spp)
    chunk = max(int(args.chunk), 1)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for cam in cams:
        if args.film is not None:
            w, h = int(args.film[0]), int(args.film[1])
        elif cam == "ortho":
            w, h = 512, 512
        else:
            w, h = 1280, 720
        img = render_xyz(
            tracer, buffers, cam, w, h, spp, device, chunk, args.seed, ortho_center, ortho_height
        )
        npy = os.path.join(args.outdir, f"gpu_xyz_fire_{cam}_cie.npy")
        np.save(npy, img.astype(np.float32))
        print("wrote", npy)
        rgb8 = (htpp_xyz_to_srgb(img, exposure=0.2) * 255.0 + 0.5).astype(np.uint8)
        png = os.path.join(args.outdir, f"render_fire_{cam}_cie_spp{spp}_{stamp}.png")
        Image.fromarray(rgb8).save(png)
        print("wrote", png)
        if not args.no_compare:
            write_compare(args.outdir, cam, img, spp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
