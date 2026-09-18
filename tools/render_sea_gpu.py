#!/usr/bin/env python3
"""GPU path-trace htrdr sea.sh: BOMEX voxels + specular sea_3m.obj.

Writes only under scenes/sea_gs/. Does not touch mountain_gs keep files.
Default film is CIE 1931 XYZ → htpp sRGB. --lambda-nm is monochromatic SW.
--spectral lw binds HTGOP LW tables but does not emit a sensor-window image.
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

import sea_scene as sea  # noqa: E402
from render_mountain_gs_gpu import (  # noqa: E402
    diagnose_cloud_sun_tr,
    load_htcp_bin,
    make_conf,
    make_ortho_rays,
    make_persp_rays,
)
from threedgptir_tracer.atmosphere import (  # noqa: E402
    SPECTRAL_LW,
    SPECTRAL_SW,
    SPECTRAL_SW_CIE_XYZ,
    AtmosphereBuffers,
    build_cie_xyz_tables,
    htpp_xyz_to_srgb,
    trace_atmosphere,
)

CAM_POS = np.array([2224.91, 3684.71, 1492.95], np.float32)
CAM_TGT = np.array([8208.71, 4471.54, -393.82], np.float32)
CAM_UP = np.array([0.0, 0.0, 1.0], np.float32)
CAM_FOV = 70.0
ORTHO_CENTER = np.array([6400.0, 6400.0, 80000.0], np.float32)
ORTHO_LOOK = np.array([0.0, 0.0, -1.0], np.float32)
ORTHO_UP = np.array([0.0, 1.0, 0.0], np.float32)
ORTHO_HEIGHT = 12800.0
SUN_AZ_DEG = 0.0
SUN_EL_DEG = 50.0

HTRDR_ORTHO = "htrdr_ref_sea_ortho_512x512x256"
HTRDR_OBLIQUE = "htrdr_ref_sea_oblique_1280x720x256"


def bind_sw(buf: AtmosphereBuffers, sw: dict, device: torch.device) -> None:
    buf.sw_lambda_lo = torch.as_tensor(sw["lambda_lo_nm"], dtype=torch.float32, device=device)
    buf.sw_lambda_hi = torch.as_tensor(sw["lambda_hi_nm"], dtype=torch.float32, device=device)
    buf.sw_gas_ka = torch.as_tensor(sw["ka_quad"], dtype=torch.float32, device=device).contiguous()
    buf.sw_gas_ks = torch.as_tensor(sw["ks_quad"], dtype=torch.float32, device=device).contiguous()
    buf.sw_nquad = torch.as_tensor(sw["nquad"], dtype=torch.int32, device=device).contiguous()
    buf.sw_quad_cdf = torch.as_tensor(sw["quad_cdf"], dtype=torch.float32, device=device).contiguous()
    buf.mie_cs_band = torch.as_tensor(sw["mie_cs_band"], dtype=torch.float32, device=device)
    buf.mie_ca_band = torch.as_tensor(sw["mie_ca_band"], dtype=torch.float32, device=device)
    buf.mie_g_nm = torch.as_tensor(sw["mie_g_nm"], dtype=torch.float32, device=device)
    buf.mie_cs_ref = float(sw["mie_cs_ref"])
    buf.mie_ca_ref = float(sw["mie_ca_ref"])
    buf.mie_g_lambda0_nm = float(sw["mie_g_lambda0_nm"])
    buf.mie_g_dlambda_nm = float(sw["mie_g_dlambda_nm"])


def bind_sw_h2o(buf: AtmosphereBuffers, sw: dict, x_h2o: np.ndarray, device: torch.device) -> None:
    """SPECTRAL_SW only. CIE/LW must not call this (leave tensors empty)."""
    x = np.asarray(x_h2o, dtype=np.float32)
    ntab = int(np.asarray(sw["h2o_x"]).shape[-1])
    buf.voxel_x_h2o = torch.as_tensor(x, dtype=torch.float32, device=device).contiguous()
    buf.sw_h2o_ntab = ntab
    buf.sw_h2o_n = torch.as_tensor(sw["h2o_n"], dtype=torch.int32, device=device).contiguous()
    buf.sw_h2o_x = torch.as_tensor(sw["h2o_x"], dtype=torch.float32, device=device).contiguous()
    buf.sw_h2o_ka = torch.as_tensor(sw["h2o_ka"], dtype=torch.float32, device=device).contiguous()
    buf.sw_h2o_ks = torch.as_tensor(sw["h2o_ks"], dtype=torch.float32, device=device).contiguous()
    buf.sw_h2o_ks_max = torch.as_tensor(sw["h2o_ks_max"], dtype=torch.float32, device=device).contiguous()
    valid = x >= 0.0
    print(
        f"SPECTRAL_SW x_H2O bound  ntab={ntab}  valid={float(valid.mean()):.3f}  "
        f"mean={float(x[valid].mean()) if valid.any() else 0.0:.4g}"
    )


def bind_lw(buf: AtmosphereBuffers, lw: dict, device: torch.device) -> None:
    buf.lw_lambda_lo = torch.as_tensor(lw["lambda_lo_nm"], dtype=torch.float32, device=device)
    buf.lw_lambda_hi = torch.as_tensor(lw["lambda_hi_nm"], dtype=torch.float32, device=device)
    buf.lw_gas_ka = torch.as_tensor(lw["ka_quad"], dtype=torch.float32, device=device).contiguous()
    buf.lw_nquad = torch.as_tensor(lw["nquad"], dtype=torch.int32, device=device).contiguous()
    buf.lw_quad_cdf = torch.as_tensor(lw["quad_cdf"], dtype=torch.float32, device=device).contiguous()
    if "mie_cs_band" in lw:
        buf.mie_cs_band = torch.as_tensor(lw["mie_cs_band"], dtype=torch.float32, device=device)
        buf.mie_ca_band = torch.as_tensor(lw["mie_ca_band"], dtype=torch.float32, device=device)
        buf.mie_cs_ref = float(lw.get("mie_cs_ref", 1.0))
        buf.mie_ca_ref = float(lw.get("mie_ca_ref", 1.0))


def htrdr_xyz(txt_path: str, h: int, w: int) -> np.ndarray:
    href = np.loadtxt(txt_path, skiprows=1, dtype=np.float64)
    href = href.reshape(-1, 8)
    xyz = np.stack([href[:, 0], href[:, 2], href[:, 4]], axis=-1).reshape(h, w, 3)
    return xyz.astype(np.float32)


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
        left = Image.fromarray((htpp_xyz_to_srgb(href, exposure=0.2, white=white) * 255.0 + 0.5).astype(np.uint8))
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
    view = "俯视 ortho 512×512" if camera == "ortho" else "斜视 sea.sh 1280×720"
    d.text((8, 8), f"htrdr CPU  {view}  spp256", fill=(220, 220, 220))
    d.text((w + 16, 8), f"GPU  同相机  spp{spp}  (htpp white=htrdr Y p995)", fill=(220, 220, 220))
    out = os.path.join(outdir, f"compare_sea_{camera}_htrdr256_gpu{spp}.png")
    canvas.save(out)
    print("wrote", out)
    gpu_aligned = os.path.join(outdir, f"render_sea_{camera}_cie_spp{spp}_htpp_htrdrwhite.png")
    Image.fromarray(gpu_rgb).save(gpu_aligned)
    print("wrote", gpu_aligned)
    return out


def render_xyz(tracer, buffers, camera: str, w: int, h: int, spp: int, device, chunk: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nchunk = (spp + chunk - 1) // chunk
    sum_xyz = np.zeros((h, w, 3), np.float64)
    if camera == "ortho":
        print(
            f"GPU ortho {w}x{h} spp={spp} height={ORTHO_HEIGHT} "
            f"center={ORTHO_CENTER.tolist()} sun az/el={SUN_AZ_DEG}/{SUN_EL_DEG}"
        )
    else:
        print(
            f"GPU sea.sh persp {w}x{h} spp={spp} fov={CAM_FOV} "
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
    p = argparse.ArgumentParser(description="Render htrdr sea.sh on the GPU atmosphere path")
    p.add_argument("--outdir", default=os.path.join(ROOT, "scenes", "sea_gs"))
    p.add_argument("--spp", type=int, default=4)
    p.add_argument("--film", nargs=2, type=int, metavar=("W", "H"), default=None)
    p.add_argument(
        "--camera",
        choices=("oblique", "ortho", "both"),
        default="oblique",
        help="oblique = sea.sh fov=70; ortho = 12.8 km nadir; both = the two comparison views",
    )
    p.add_argument("--compare-only", action="store_true", help="rebuild side-by-side from saved npy + htrdr txt")
    p.add_argument("--max-bounces", type=int, default=256)
    p.add_argument("--mie-g", type=float, default=0.865)
    p.add_argument("--brick", type=int, default=8)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--rebuild-scene", action="store_true")
    p.add_argument("--no-clouds", action="store_true")
    p.add_argument(
        "--spectral",
        choices=("cie", "sw", "lw"),
        default="cie",
        help="cie = sea.sh visible image; sw = --lambda-nm; lw = bind LW k (no window png)",
    )
    p.add_argument("--lambda-nm", type=float, default=None, help="monochromatic λ (implies --spectral sw unless lw)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.lambda_nm is not None and args.spectral == "cie":
        args.spectral = "sw"

    os.makedirs(args.outdir, exist_ok=True)
    cams = ("ortho", "oblique") if args.camera == "both" else (args.camera,)
    if args.compare_only:
        n = 0
        for cam in cams:
            npy = os.path.join(args.outdir, f"gpu_xyz_sea_{cam}_cie.npy")
            if not os.path.isfile(npy):
                print("missing", npy)
                continue
            write_compare(args.outdir, cam, np.load(npy), args.spp)
            n += 1
        return 0 if n else 1
    assets = sea.ensure_assets(outdir=args.outdir, rebuild=args.rebuild_scene)
    gas = assets["gas"]
    mesh = assets["mesh"]
    water = assets["water"]
    htcp_bin = assets["voxels_bin"]

    ka, ks, temp, origin, spacing, z_iface, x_h2o = load_htcp_bin(htcp_bin)
    print(
        f"voxels {ks.shape} origin={origin} dx={spacing} "
        f"z_iface={'yes' if z_iface is not None else 'no'}"
    )
    if args.no_clouds:
        ka = np.zeros_like(ka)
        ks = np.zeros_like(ks)
        print("no-clouds: voxel ka/ks zeroed, 1D gas kept")

    az = math.radians(SUN_AZ_DEG)
    el = math.radians(SUN_EL_DEG)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    diagnose_cloud_sun_tr(ka, ks, origin, spacing, sun, z_iface=z_iface)

    device = torch.device("cuda")
    torch.zeros(1, device=device)
    conf = make_conf()
    from threedgptir_tracer.tracer import load_threedgptir_plugin
    import threedgptir_tracer.tracer as tracer_mod

    print("loading PTIR OptiX plugin (first compile can take several minutes)...")
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
    print("OptixTracer ready  ground: sea_3m specular mesh (GS off)")

    import mountain_gs_scene as mgs

    lam_nm = float(args.lambda_nm) if args.lambda_nm is not None else 550.0
    if args.spectral == "lw":
        mode = SPECTRAL_LW
        if args.lambda_nm is None:
            lam_nm = 10000.0
        print(f"SPECTRAL_LW λ={lam_nm:.0f} nm  (tables only; no 8–14 μm window image)")
    elif args.spectral == "sw":
        mode = SPECTRAL_SW
        print(f"SPECTRAL_SW λ={lam_nm:.0f} nm")
    else:
        mode = SPECTRAL_SW_CIE_XYZ
        lam_nm = 550.0
        print("SPECTRAL_SW_CIE_XYZ 380–780 nm (sea.sh visible)")

    gka, gks = gas["ka"], gas["ks"]
    mie_omega = 1.0
    sun_radiance = 24.0
    if mode == SPECTRAL_LW:
        gas = mgs.load_ecrad_gas_1d_lw(wavelength_m=lam_nm * 1.0e-9)
        gka = gas["ka"]
        gks = np.zeros_like(gka)
        mie_omega = 0.0
        sun_radiance = 0.0
    elif mode == SPECTRAL_SW:
        sun_radiance = 1.0

    buffers = AtmosphereBuffers.from_numpy(
        device=device,
        voxel_ka=ka,
        voxel_ks=ks,
        voxel_temp=temp,
        voxel_origin=tuple(origin.tolist()),
        voxel_spacing=tuple(np.asarray(spacing, np.float32).tolist()),
        brick=int(args.brick),
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
        mie_g=args.mie_g,
        path_tmax=1.2e5,
        max_bounces=args.max_bounces,
        voxel_z_iface=z_iface,
    )
    if mode == SPECTRAL_SW_CIE_XYZ:
        cie = build_cie_xyz_tables()
        buffers.cie_cdf = torch.as_tensor(cie["cdf"], dtype=torch.float32, device=device)
        buffers.cie_rcp_integral = torch.as_tensor(cie["rcp_integral"], dtype=torch.float32, device=device)
        buffers.cie_lambda0_nm = float(cie["lambda0_nm"])
        buffers.cie_dlambda_nm = float(cie["dlambda_nm"])
        sw = mgs.load_ecrad_sw_visible()
        bind_sw(buffers, sw, device)
    elif mode == SPECTRAL_SW:
        sw = mgs.load_ecrad_sw_visible(lambda0_nm=lam_nm, lambda1_nm=lam_nm)
        ib = mgs.require_sw_band(sw, lam_nm)
        lo = float(sw["lambda_lo_nm"][ib])
        hi = float(sw["lambda_hi_nm"][ib])
        nq = int(np.asarray(sw["nquad"]).reshape(-1)[ib])
        bsun = mgs.planck_B_lambda(lam_nm * 1.0e-9, 5778.0)
        print(
            f"SPECTRAL_SW bind HTGOP {lo:.0f}-{hi:.0f} nm  nq={nq}  "
            f"L_sun=B_λ(5778 K)={bsun:.4g}  host placeholder={sun_radiance} (kernel overwrites)"
        )
        bind_sw(buffers, sw, device)
        if x_h2o is None:
            print("voxel bin has no x_H2O; dumping HTCP RVT")
            ht = mgs.load_htcp_voxels(sea.DEFAULT_HTCP, wavelength_nm=lam_nm)
            x_h2o = ht.get("x_h2o")
        x_h2o, sw = mgs.require_sw_h2o(sw, {"ks": ks, "x_h2o": x_h2o})
        bind_sw_h2o(buffers, sw, x_h2o, device)
    else:
        bind_lw(buffers, mgs.load_ecrad_lw(), device)

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
    print(
        f"mesh verts={mesh['verts'].shape[0]} tris={mesh['idx'].shape[0]} "
        f"mat_id={int(mesh['mat_id'][0])}  water knots={water['lambda_nm'].shape[0]}  "
        f"voxel ks max/mean={float(np.max(ks)):.4g}/{float(np.mean(ks)):.4g}"
    )

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
            tracer, buffers, cam, w, h, spp, device, chunk, args.seed
        )
        tag = f"_{cam}"
        if args.no_clouds:
            tag += "_nocloud"
        if mode == SPECTRAL_SW:
            tag += f"_sw{int(lam_nm)}"
        elif mode == SPECTRAL_LW:
            tag += f"_lw{int(lam_nm)}"
        else:
            tag += "_cie"
        npy = os.path.join(args.outdir, f"gpu_xyz_sea{tag}.npy")
        np.save(npy, img.astype(np.float32))
        print("wrote", npy)
        if mode == SPECTRAL_SW_CIE_XYZ:
            rgb8 = (htpp_xyz_to_srgb(img, exposure=0.2) * 255.0 + 0.5).astype(np.uint8)
        else:
            y = np.maximum(img[..., 1], 0.0)
            scale = float(np.percentile(y, 99.5)) if y.size else 1.0
            scale = max(scale, 1e-12)
            t = np.clip(y / scale, 0.0, 1.0)
            rgb8 = (np.stack([t, t, t], axis=-1) * 255.0 + 0.5).astype(np.uint8)
        png = os.path.join(args.outdir, f"render_sea{tag}_spp{spp}_{stamp}.png")
        Image.fromarray(rgb8).save(png)
        print("wrote", png)
        if mode == SPECTRAL_SW_CIE_XYZ:
            write_compare(args.outdir, cam, img, spp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
