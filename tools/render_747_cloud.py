#!/usr/bin/env python3
"""747-in-cloud ablation: same cloud_top / cloud_base cameras, three scenes.

1. cloud + aircraft
2. cloud, no aircraft
3. no cloud + aircraft

Does not overwrite *_keep.png or gpu_xyz_crop_*_spp32.npy.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.utils.cpp_extension
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "gpu"))
sys.path.insert(0, os.path.dirname(__file__))

import aircraft_mesh as acmesh  # noqa: E402
import mountain_gs_scene as mgs  # noqa: E402
from render_mountain_gs_gpu import (  # noqa: E402
    diagnose_cloud_sun_tr,
    load_htcp_bin,
    make_conf,
    make_persp_rays,
)
from threedgptir_tracer.atmosphere import (  # noqa: E402
    MTL_RGB_LAMBERT,
    SPECTRAL_SW_CIE_XYZ,
    AtmosphereBuffers,
    htpp_xyz_to_srgb,
    trace_atmosphere,
    xenon_searchlight_params,
)


def camera_pair(place: dict, standoff_m: float = 400.0):
    c = np.asarray(place["center"], np.float64)
    z_top = float(place["z_top"])
    z_base = float(place["z_base"])
    stand = float(standoff_m)
    pos_top = np.array([c[0], c[1], z_top + stand], np.float32)
    z_cam_base = max(z_base - stand, 750.0)
    if z_cam_base >= c[2] - 0.25 * stand:
        z_cam_base = float(c[2] - stand)
    pos_base = np.array([c[0], c[1], z_cam_base], np.float32)
    tgt = c.astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    d_top = float(np.linalg.norm(pos_top - tgt))
    d_base = float(np.linalg.norm(pos_base - tgt))
    print(
        f"cameras  top z={pos_top[2]:.1f} dist={d_top:.1f} m   "
        f"base z={pos_base[2]:.1f} dist={d_base:.1f} m  tgt={tgt.tolist()}"
    )
    return {"top": (pos_top, tgt, up, d_top), "base": (pos_base, tgt, up, d_base)}


def render_xyz(tracer, buffers, pos, tgt, up, w, h, spp, device, chunk=4, seed=0):
    rng = np.random.default_rng(seed)
    sum_xyz = np.zeros((h, w, 3), np.float64)
    nchunk = (spp + chunk - 1) // chunk
    t0 = time.perf_counter()
    for c in range(nchunk):
        n = min(chunk, spp - c * chunk)
        ori, drc = make_persp_rays(w, h, pos, tgt, up, 30.0, n, device, rng=rng)
        torch.cuda.synchronize()
        out = trace_atmosphere(tracer, ori, drc, buffers, frame_number=c)
        torch.cuda.synchronize()
        rgb = out[10].float().cpu().numpy()
        sum_xyz += rgb.sum(axis=0)
        print(f"    chunk {c+1}/{nchunk} meanY={float(rgb[..., 1].mean()):.4f}")
        del ori, drc, out, rgb
        torch.cuda.empty_cache()
    img = (sum_xyz / float(spp)).astype(np.float32)
    print(
        f"    done {time.perf_counter()-t0:.1f}s  "
        f"Y mean/p50={float(img[...,1].mean()):.4f}/{float(np.median(img[...,1])):.4f}"
    )
    return img


def signed_delta_rgb(dY: np.ndarray, scale: float) -> np.ndarray:
    s = max(float(scale), 1e-8)
    t = np.clip(dY / s, -1.0, 1.0)
    out = np.zeros(dY.shape + (3,), np.float32)
    out[..., 0] = np.clip(t, 0.0, 1.0)
    out[..., 2] = np.clip(-t, 0.0, 1.0)
    out[..., 1] = 0.15 * (1.0 - np.abs(t))
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def label_img(rgb8, text: str) -> Image.Image:
    im = Image.fromarray(rgb8)
    canvas = Image.new("RGB", (im.width, im.height + 28), (18, 18, 18))
    canvas.paste(im, (0, 28))
    ImageDraw.Draw(canvas).text((8, 6), text, fill=(230, 230, 230))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=os.path.join(ROOT, "scenes/mountain_gs"))
    p.add_argument("--spp", type=int, default=32)
    p.add_argument("--film", nargs=2, type=int, default=[640, 640])
    p.add_argument("--max-bounces", type=int, default=256)
    p.add_argument("--mie-g", type=float, default=0.865)
    p.add_argument("--brick", type=int, default=8)
    p.add_argument("--chunk", type=int, default=4, help="spp per optixLaunch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="subset of: cloud_ac cloud nocloud_ac",
    )
    p.add_argument("--place", choices=("peak", "mid"), default="peak")
    p.add_argument(
        "--tag",
        default="",
        help="filename suffix, e.g. mid → gpu_xyz_747_*_mid.npy (does not overwrite peak spp256)",
    )
    p.add_argument(
        "--spot",
        action="store_true",
        help="2 kW-class xenon searchlight on the ground, aimed at the aircraft",
    )
    p.add_argument("--spot-lumens", type=float, default=8.0e4, help="searchlight lumens (default 8e4, xenon 2 kW)")
    p.add_argument("--spot-diameter", type=float, default=1.0, help="reflector diameter in meters")
    p.add_argument(
        "--aircraft",
        choices=("747", "shahed"),
        default="747",
        help="747 = procedural; shahed = CGTrader Shahed-136 OBJ in scenes/aircraft/shahed136/",
    )
    p.add_argument(
        "--mesh",
        default=None,
        help="Shahed OBJ/glTF/GLB/STL or zip (default: first mesh under scenes/aircraft/shahed136/)",
    )
    p.add_argument(
        "--ac-length",
        type=float,
        default=None,
        help="scale longest AABB edge to this many meters (shahed default 3.5)",
    )
    p.add_argument(
        "--standoff",
        type=float,
        default=None,
        help="camera distance past cloud top/base (747 default 400 m; shahed ~18 m)",
    )
    args = p.parse_args()
    if args.spot and not args.tag:
        args.tag = f"{args.place}_spot"
        print(f"--spot: default --tag {args.tag} (will not overwrite peak/mid)")
    os.makedirs(args.outdir, exist_ok=True)

    assets = mgs.ensure_assets(outdir=args.outdir, rebuild=False)
    ka, ks, temp, origin, spacing, z_iface, _xh = load_htcp_bin(assets["voxels_bin"])
    gas = assets["gas"]
    place = mgs.pick_b747_placement(
        ks, ka, origin, spacing, z_iface, tau_min=8.0, place=args.place
    )
    tag_sfx = f"_{args.tag}" if args.tag else ""
    ac_key = "747" if args.aircraft == "747" else "shahed"
    if args.aircraft == "shahed":
        length = float(args.ac_length) if args.ac_length else acmesh.SHAHED_LENGTH_M
        if args.mesh:
            ac = acmesh.make_aircraft_mesh(
                place["center"],
                place["heading"],
                path=args.mesh,
                length_m=length,
                name="Shahed-136",
            )
        else:
            ac = acmesh.make_shahed136_coarse_mesh(place["center"], place["heading"])
            length = float(ac["length_m"])
        standoff = float(args.standoff) if args.standoff else max(12.0, 5.0 * length)
    else:
        ac = mgs.make_b747_mesh(place["center"], place["heading"])
        standoff = float(args.standoff) if args.standoff else 400.0
    assert int(ac["mat_id"][0]) == MTL_RGB_LAMBERT
    mountain = mgs.ensure_mountain_mesh(outdir=args.outdir)
    both = mgs.concat_meshes(mountain, ac)
    spot_vec = None
    if args.spot:
        verts = np.asarray(mountain["verts"], np.float64)
        c = np.asarray(place["center"], np.float64)
        d2 = (verts[:, 0] - c[0]) ** 2 + (verts[:, 1] - c[1]) ** 2
        k = np.argpartition(d2, min(16, verts.shape[0] - 1))[:16]
        z_g = float(np.median(verts[k, 2])) + 2.0
        lamp = np.array([c[0], c[1], z_g], np.float64)
        spot_vec, spot_info = xenon_searchlight_params(
            lamp, c, lumens=args.spot_lumens, diameter_m=args.spot_diameter
        )
        print(
            f"searchlight  Φ={spot_info['lumens']:.0f} lm  D={spot_info['diameter_m']:.2f} m  "
            f"α={spot_info['half_angle_deg']:.2f}°  dist={spot_info['dist_m']:.1f} m"
        )
        print(
            f"  L={spot_info['luminance_cd_m2']:.3e} cd/m²   "
            f"Lsun={spot_info['sun_luminance_cd_m2']:.3e}  "
            f"L/Lsun={spot_info['spot_sun_scale']:.4g}"
        )
        print(
            f"  I0={spot_info['I0_cd']:.3e} cd  "
            f"E_axis@aircraft={spot_info['E_axis_lux']:.2f} lux  "
            f"(day sun ~1e5 lux; this is a real xenon flood, not a small sun)"
        )
        print(f"  lamp pos={lamp.tolist()}")
    print(
        "mesh mat_id counts",
        {int(k): int(v) for k, v in zip(*np.unique(both["mat_id"], return_counts=True))},
        "ntri",
        both["idx"].shape[0],
    )
    cams = camera_pair(place, standoff_m=standoff)
    w, h = int(args.film[0]), int(args.film[1])

    device = torch.device("cuda")
    torch.zeros(1, device=device)
    conf = make_conf()
    from threedgptir_tracer.tracer import load_threedgptir_plugin
    import threedgptir_tracer.tracer as tracer_mod

    print("loading PTIR OptiX plugin...")
    load_threedgptir_plugin(conf)
    plugin = tracer_mod._threedgptir_plugin
    tracer = plugin.OptixTracer(
        os.path.join(ROOT, "gpu", "threedgptir_tracer"),
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

    az = math.radians(225.0)
    el = math.radians(30.0)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    diagnose_cloud_sun_tr(ka, ks, origin, spacing, sun, z_iface=z_iface)

    from threedgptir_tracer.atmosphere import build_cie_xyz_tables

    cie = build_cie_xyz_tables()
    sw = mgs.load_ecrad_sw_visible()
    sand = mgs.load_sand_mrumtl()

    def make_buffers(no_clouds: bool, with_ac: bool) -> AtmosphereBuffers:
        ka_u = np.zeros_like(ka) if no_clouds else ka
        ks_u = np.zeros_like(ks) if no_clouds else ks
        gka, gks = gas["ka"], gas["ks"]
        sw_ka, sw_ks = sw["ka_quad"], sw["ks_quad"]
        if no_clouds:
            print("no-clouds: voxel ka/ks zeroed, 1D gas kept")
        mesh = both if with_ac else mountain
        buf = AtmosphereBuffers.from_numpy(
            device=device,
            voxel_ka=ka_u,
            voxel_ks=ks_u,
            voxel_temp=temp,
            voxel_origin=tuple(origin.tolist()),
            voxel_spacing=tuple(np.asarray(spacing, np.float32).tolist()),
            brick=int(args.brick),
            gas_z=gas["z"],
            gas_ka=gka,
            gas_ks=gks,
            gas_temp=gas["temp"],
            sun_dir=sun,
            sun_radiance=24.0,
            sun_cos_half=math.cos(4.6524e-3),
            spectral_mode=SPECTRAL_SW_CIE_XYZ,
            wavelength_m=5.5e-7,
            mie_omega=1.0,
            mie_g=args.mie_g,
            path_tmax=1.2e5,
            max_bounces=args.max_bounces,
            voxel_z_iface=z_iface,
        )
        buf.cie_cdf = torch.as_tensor(cie["cdf"], dtype=torch.float32, device=device)
        buf.cie_rcp_integral = torch.as_tensor(cie["rcp_integral"], dtype=torch.float32, device=device)
        buf.cie_lambda0_nm = float(cie["lambda0_nm"])
        buf.cie_dlambda_nm = float(cie["dlambda_nm"])
        buf.sw_lambda_lo = torch.as_tensor(sw["lambda_lo_nm"], dtype=torch.float32, device=device)
        buf.sw_lambda_hi = torch.as_tensor(sw["lambda_hi_nm"], dtype=torch.float32, device=device)
        buf.sw_gas_ka = torch.as_tensor(sw_ka, dtype=torch.float32, device=device).contiguous()
        buf.sw_gas_ks = torch.as_tensor(sw_ks, dtype=torch.float32, device=device).contiguous()
        buf.sw_nquad = torch.as_tensor(sw["nquad"], dtype=torch.int32, device=device).contiguous()
        buf.sw_quad_cdf = torch.as_tensor(sw["quad_cdf"], dtype=torch.float32, device=device).contiguous()
        buf.mie_cs_band = torch.as_tensor(sw["mie_cs_band"], dtype=torch.float32, device=device)
        buf.mie_ca_band = torch.as_tensor(sw["mie_ca_band"], dtype=torch.float32, device=device)
        buf.mie_g_nm = torch.as_tensor(sw["mie_g_nm"], dtype=torch.float32, device=device)
        buf.mie_cs_ref = float(sw["mie_cs_ref"])
        buf.mie_ca_ref = float(sw["mie_ca_ref"])
        buf.brdf_lambda_nm = torch.as_tensor(sand["lambda_nm"], dtype=torch.float32, device=device)
        buf.brdf_R = torch.as_tensor(sand["R"], dtype=torch.float32, device=device)
        buf.particle_density = torch.zeros((0, 12), dtype=torch.float32, device=device)
        buf.particle_material = torch.zeros((0, 5), dtype=torch.float32, device=device)
        buf.particle_radiance = torch.zeros((0, 1), dtype=torch.float32, device=device)
        buf.particle_shading_normal = torch.zeros((0, 3), dtype=torch.float32, device=device)
        buf.mesh_verts = torch.from_numpy(np.ascontiguousarray(mesh["verts"])).to(device)
        buf.mesh_idx = torch.from_numpy(np.ascontiguousarray(mesh["idx"])).to(device)
        buf.mesh_albedo = torch.from_numpy(np.ascontiguousarray(mesh["albedo"])).to(device)
        buf.mesh_temp = torch.from_numpy(np.ascontiguousarray(mesh["temp"])).to(device)
        mat_id = mesh.get("mat_id")
        if mat_id is None:
            mat_id = np.zeros((mesh["idx"].shape[0],), np.int32)
        buf.mesh_mat_id = torch.from_numpy(np.ascontiguousarray(mat_id)).to(device)
        if spot_vec is not None:
            buf.spot_params = torch.from_numpy(np.ascontiguousarray(spot_vec))
        return buf

    cases = (
        ("cloud_ac", False, True, "有云+飞机"),
        ("cloud", False, False, "有云"),
        ("nocloud_ac", True, True, "无云+飞机"),
    )
    if args.only:
        want = set(args.only)
        cases = tuple(c for c in cases if c[0] in want)
        if not cases:
            raise SystemExit(f"--only {sorted(want)} matched no case")
    xyz = {}
    for key, no_c, with_ac, label in cases:
        print(f"=== buffers {key} ({label}) ===")
        buf = make_buffers(no_c, with_ac)
        for cam_name, (pos, tgt, up, dist) in cams.items():
            tag = f"{key}_{cam_name}"
            print(f"-- render {tag} dist={dist:.1f}m spp={args.spp}")
            xyz[tag] = render_xyz(
                tracer, buf, pos, tgt, up, w, h, args.spp, device,
                chunk=max(int(args.chunk), 1), seed=int(args.seed),
            )
            np.save(os.path.join(args.outdir, f"gpu_xyz_{ac_key}_{tag}{tag_sfx}.npy"), xyz[tag])
        del buf
        torch.cuda.empty_cache()

    for key in ("cloud_ac", "cloud", "nocloud_ac"):
        for cam_name in ("top", "base"):
            tag = f"{key}_{cam_name}"
            if tag in xyz:
                continue
            npy = os.path.join(args.outdir, f"gpu_xyz_{ac_key}_{tag}{tag_sfx}.npy")
            if os.path.isfile(npy):
                xyz[tag] = np.load(npy)
                print("loaded", npy)

    white = float(np.percentile(xyz["nocloud_ac_top"][..., 1], 99.5))
    print("htpp white from nocloud_ac_top Y p995", white)
    png = {}
    for key in ("cloud_ac", "cloud", "nocloud_ac"):
        for cam_name in ("top", "base"):
            tag = f"{key}_{cam_name}"
            if tag not in xyz:
                raise SystemExit(
                    f"missing {tag} (render it or keep gpu_xyz_{ac_key}_{tag}{tag_sfx}.npy)"
                )
            rgb8 = (htpp_xyz_to_srgb(xyz[tag], exposure=0.2, white=white) * 255.0 + 0.5).astype(np.uint8)
            png[tag] = rgb8
            path = os.path.join(args.outdir, f"render_{ac_key}_{tag}{tag_sfx}.png")
            Image.fromarray(rgb8).save(path)
            print("wrote", path, "Y", float(xyz[tag][..., 1].mean()))

    cols = ("cloud_ac", "cloud", "nocloud_ac")
    col_labels = ("有云+飞机", "有云", "无云+飞机")
    row_names = (("top", "云顶"), ("base", "云底"))
    cell_w, cell_h = w, h + 28
    grid = Image.new("RGB", (cell_w * 3 + 16, cell_h * 2 + 40), (12, 12, 12))
    dr = ImageDraw.Draw(grid)
    dr.text(
        (8, 8),
        f"{ac_key} 1:1  place={place['place']}  z={place['center'][2]:.0f}m  "
        f"tau={place['tau']:.1f}  τ↑={place['tau_above']:.1f} τ↓={place['tau_below']:.1f}  "
        f"H={place['cloud_h']:.0f}m  spp={args.spp}"
        + (f"  xenon {args.spot_lumens:.0e}lm" if args.spot else ""),
        fill=(220, 220, 220),
    )
    for r, (cam, rlab) in enumerate(row_names):
        for c, (ck, clab) in enumerate(zip(cols, col_labels)):
            im = label_img(png[f"{ck}_{cam}"], f"{rlab}  {clab}")
            grid.paste(im, (8 + c * cell_w, 32 + r * cell_h))
    gpath = os.path.join(args.outdir, f"compare_{ac_key}_2x3{tag_sfx}.png")
    grid.save(gpath)
    print("wrote", gpath)

    def dump_delta(a, b, name, note):
        dY = xyz[a][..., 1] - xyz[b][..., 1]
        sc = float(np.percentile(np.abs(dY), 99.0))
        rgb = signed_delta_rgb(dY, sc)
        Image.fromarray(rgb).save(os.path.join(args.outdir, f"deltaY_{ac_key}_{name}{tag_sfx}.png"))
        np.save(os.path.join(args.outdir, f"deltaY_{ac_key}_{name}{tag_sfx}.npy"), dY.astype(np.float32))
        h0, w0 = dY.shape
        y0, x0 = h0 // 2 - 40, w0 // 2 - 40
        cen = dY[y0 : y0 + 80, x0 : x0 + 80]
        print(
            f"ΔY {name} ({note})  mean={float(dY.mean()):.4g}  "
            f"rms={float(np.sqrt((dY*dY).mean())):.4g}  "
            f"|dY| p50/p99={float(np.median(np.abs(dY))):.4g}/{sc:.4g}  "
            f"center80 mean/rms={float(cen.mean()):.4g}/{float(np.sqrt((cen*cen).mean())):.4g}"
        )

    dump_delta("cloud_ac_top", "cloud_top", "cloud_ac_minus_cloud_top", "有云 有飞机−无飞机 云顶")
    dump_delta("cloud_ac_base", "cloud_base", "cloud_ac_minus_cloud_base", "有云 有飞机−无飞机 云底")
    dump_delta("cloud_ac_top", "nocloud_ac_top", "cloud_minus_clear_top", "有飞机 有云−无云 云顶")
    dump_delta("cloud_ac_base", "nocloud_ac_base", "cloud_minus_clear_base", "有飞机 有云−无云 云底")
    print(f"{ac_key} ablate done")


if __name__ == "__main__":
    main()
