#!/usr/bin/env python3
"""Orthographic GPU PTIR atmosphere path (voxel Woodcock + GS surfaces + optional mesh).

RTE lives in copied __raygen__rg → traceAtmospherePath. Cloud k is voxels only.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACER = os.path.join(ROOT, "gpu", "threedgptir_tracer")
sys.path.insert(0, os.path.join(ROOT, "gpu"))
sys.path.insert(0, os.path.dirname(__file__))

_local_lib = os.path.join(ROOT, "local", "lib")
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _local_lib not in _ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = _local_lib + (":" + _ld if _ld else "")


def make_conf():
    render = SimpleNamespace(
        pipeline_type="reference",
        backward_pipeline_type="referenceBwd",
        primitive_type="instances",
        particle_kernel_degree=4,
        particle_kernel_min_response=0.0113,
        particle_kernel_min_alpha=1.0 / 255.0,
        particle_kernel_max_alpha=0.99,
        particle_kernel_density_clamping=True,
        particle_radiance_sph_degree=3,
        enable_normals=True,
        enable_hitcounts=True,
        enable_mis=True,
        enable_metallic=True,
        visualize_lights=False,
        enable_raygen_setup_profiling=False,
        raygen_ablation_stage=1,
        filter_type="none",
        max_consecutive_bvh_update=15,
        enable_kernel_timings=False,
        get=lambda self, k, d=None: getattr(self, k, d),
    )
    # SimpleNamespace has no .get bound correctly above; fix:
    def _get(k, d=None, _r=render):
        return getattr(_r, k, d)

    render.get = _get
    model = SimpleNamespace(get=lambda k, d=False: d)
    return SimpleNamespace(render=render, model=model)


def load_ply_gs(path: str, device: torch.device):
    """ASCII ply from tools/mountain_gs_scene.py (mountain.obj → GS)."""
    with open(path, "r") as f:
        nvert = 0
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("bad ply")
            if line.startswith("element vertex"):
                nvert = int(line.split()[-1])
            if line.startswith("end_header"):
                break
        data = np.loadtxt(f, dtype=np.float32, max_rows=nvert)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    pos = torch.from_numpy(data[:, 0:3]).to(device)
    nrm = torch.from_numpy(data[:, 3:6]).to(device)
    scl = torch.from_numpy(data[:, 6:9]).to(device)
    quat = torch.from_numpy(data[:, 9:13]).to(device)
    alb = torch.from_numpy(data[:, 13:16]).to(device)
    rgh = torch.from_numpy(data[:, 16:17]).to(device)
    met = torch.from_numpy(data[:, 17:18]).to(device)
    opa = torch.from_numpy(data[:, 19:20]).to(device)
    scl = scl * 2.05  # overlap neighbors (ply spacing ~39 m, stored scale ~21 m)
    # Surface-like: high density so Bernoulli(α) ≈ opaque Lambert ground.
    dns = torch.full((pos.shape[0], 1), 80.0, dtype=torch.float32, device=device)
    density = torch.cat([pos, dns, quat, scl, torch.zeros_like(dns)], dim=1).contiguous()
    material = torch.cat([alb, rgh, met], dim=1).contiguous()
    radiance = torch.zeros((pos.shape[0], 1), dtype=torch.float32, device=device)
    return {
        "pos": pos,
        "rot": quat,
        "scl": scl,
        "dns": dns,
        "density": density,
        "material": material,
        "radiance": radiance,
        "normal": nrm.contiguous(),
    }


def fbm_cumulus(nx, ny, nz, origin, spacing):
    """Same broken-Cu field as the CPU preview (6.4 km tile)."""

    def hash31(x, y, z):
        h = np.uint32(x * 374761393 + y * 668265263 + z * 1597334677)
        h = np.uint32((h ^ (h >> 13)) * 1274126177)
        return (int(h) & 0x00FFFFFF) / 16777216.0

    def vnoise(x, y, z):
        ix, iy, iz = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        fx, fy, fz = x - ix, y - iy, z - iz
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)
        fz = fz * fz * (3 - 2 * fz)

        def at(dx, dy, dz):
            return hash31(ix + dx, iy + dy, iz + dz)

        n000, n100 = at(0, 0, 0), at(1, 0, 0)
        n010, n110 = at(0, 1, 0), at(1, 1, 0)
        n001, n101 = at(0, 0, 1), at(1, 0, 1)
        n011, n111 = at(0, 1, 1), at(1, 1, 1)
        x00 = n000 + (n100 - n000) * fx
        x10 = n010 + (n110 - n010) * fx
        x01 = n001 + (n101 - n001) * fx
        x11 = n011 + (n111 - n011) * fx
        y0 = x00 + (x10 - x00) * fy
        y1 = x01 + (x11 - x01) * fy
        return y0 + (y1 - y0) * fz

    def fbm(x, y, z):
        a, w, s = 0.0, 0.5, 0.0
        for _ in range(5):
            a += w * vnoise(x, y, z)
            s += w
            w *= 0.5
            x *= 2.03
            y *= 2.03
            z *= 2.03
        return a / s

    ks = np.zeros((nz, ny, nx), np.float32)
    ka = np.zeros_like(ks)
    temp = np.full_like(ks, 278.0)
    tile = 6400.0
    ox, oy, oz = origin
    dx, dy, dz = spacing
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                x = ox + (ix + 0.5) * dx
                y = oy + (iy + 0.5) * dy
                z = oz + (iz + 0.5) * dz
                u = (x % tile) / tile
                v = (y % tile) / tile
                if u < 0:
                    u += 1
                if v < 0:
                    v += 1
                h = (z - 850.0) / 1400.0
                cells = 5.8
                xx = u * cells + 0.35 * (fbm(u * 3, v * 3, h * 2) - 0.5)
                yy = v * cells + 0.35 * (fbm(u * 3 + 2.1, v * 3, h * 2) - 0.5)
                ixc, iyc = int(math.floor(xx)), int(math.floor(yy))
                keep = 0.0
                for dj in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        cx, cy = ixc + di, iyc + dj
                        if hash31(cx, cy, 19) > 0.30:
                            continue
                        px = cx + 0.18 + 0.64 * hash31(cx, cy, 1)
                        py = cy + 0.18 + 0.64 * hash31(cx, cy, 2)
                        pz = 0.28 + 0.40 * hash31(cx, cy, 7)
                        rad = 0.32 + 0.38 * hash31(cx, cy, 3)
                        radz = 0.22 + 0.28 * hash31(cx, cy, 11)
                        ddx = (xx - px) / max(rad, 1e-3)
                        ddy = (yy - py) / max(rad, 1e-3)
                        ddz = (h - pz) / max(radz, 1e-3)
                        d = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                        w = max(0.0, 1.0 - d)
                        keep = max(keep, w * w * (1.0 - 0.25 * d))
                detail = fbm(u * 9, v * 9, h * 6)
                dens = keep * max(0.0, 0.65 + 0.55 * (detail - 0.5)) * 0.10
                ks[iz, iy, ix] = dens
                ka[iz, iy, ix] = dens * 0.001
    return ka, ks, temp


def load_htcp_bin(path: str):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=4)
        nx, ny, nz = int(hdr[0]), int(hdr[1]), int(hdr[2])
        origin = np.fromfile(f, dtype=np.float32, count=3)
        spacing = np.fromfile(f, dtype=np.float32, count=3)
        n = nx * ny * nz
        ks = np.fromfile(f, dtype=np.float32, count=n).reshape(nz, ny, nx)
        ka = np.fromfile(f, dtype=np.float32, count=n).reshape(nz, ny, nx)
        temp = np.fromfile(f, dtype=np.float32, count=n).reshape(nz, ny, nx)
    return ka, ks, temp, origin, spacing


def analog_kext_tr(ka, ks, origin, spacing, org, direction, tmax=2.0e4):
    """Host analog exp(-∫Kext) along a ray (same cell-step idea as GPU DDA)."""
    kext = np.asarray(ka, dtype=np.float32) + np.asarray(ks, dtype=np.float32)
    nz, ny, nx = kext.shape
    dx, dy, dz = [float(spacing[i]) for i in range(3)]
    ox, oy, oz = [float(origin[i]) for i in range(3)]
    d = np.asarray(direction, dtype=np.float64)
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        return 1.0
    d = d / nrm
    p = np.asarray(org, dtype=np.float64)
    t = 0.0
    tau = 0.0
    ds = 0.5 * min(dx, dy, dz)
    while t < tmax:
        ix = int(np.floor((p[0] - ox) / dx))
        iy = int(np.floor((p[1] - oy) / dy))
        iz = int(np.floor((p[2] - oz) / dz))
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            tau += float(kext[iz, iy, ix]) * ds
        p = p + d * ds
        t += ds
    if tau > 80.0:
        return 0.0
    return float(np.exp(-tau))


def diagnose_cloud_sun_tr(ka, ks, origin, spacing, sun):
    """Cloud-top point toward the sun: Tr should be in (0, 1), not identically 0."""
    kext = np.asarray(ka, dtype=np.float32) + np.asarray(ks, dtype=np.float32)
    iz, iy, ix = np.unravel_index(int(np.argmax(kext)), kext.shape)
    org = np.array(
        [
            float(origin[0]) + (ix + 0.5) * float(spacing[0]),
            float(origin[1]) + (iy + 0.5) * float(spacing[1]),
            float(origin[2]) + (iz + 0.5) * float(spacing[2]),
        ],
        dtype=np.float64,
    )
    tr = analog_kext_tr(ka, ks, origin, spacing, org, sun)
    print(
        f"diag analog Tr(Kext) at cloud cell ({ix},{iy},{iz}) z={org[2]:.1f} toward sun: {tr:.4g}"
    )
    if not (0.0 < tr < 1.0):
        print("warning: expected analog sun Tr in (0,1)")


def make_ortho_rays(w, h, center, height, up, look, spp, device):
    """htrdr orthographic: pos/tgt/up, height = vertical sensor extent (meters)."""
    look = look / np.linalg.norm(look)
    up = up - look * np.dot(up, look)
    up = up / np.linalg.norm(up)
    right = np.cross(look, up)
    right = right / np.linalg.norm(right)
    aspect = w / float(h)
    half_h = 0.5 * height
    half_w = half_h * aspect
    rng = np.random.default_rng(0)
    ori = np.zeros((spp, h, w, 3), np.float32)
    drc = np.zeros((spp, h, w, 3), np.float32)
    for s in range(spp):
        js = (np.arange(w, dtype=np.float32)[None, :] + rng.random((h, w))) / w
        is_ = (np.arange(h, dtype=np.float32)[:, None] + rng.random((h, w))) / h
        u = (js - 0.5) * 2.0 * half_w
        v = (0.5 - is_) * 2.0 * half_h
        o = center[None, None, :] + right[None, None, :] * u[..., None] + up[None, None, :] * v[..., None]
        ori[s] = o
        drc[s] = look[None, None, :]
    return torch.from_numpy(ori).to(device), torch.from_numpy(drc).to(device)


def tonemap(img, exposure=1.8):
    x = np.maximum(img * exposure, 0.0)
    x = x / (1.0 + x)
    x = np.power(x, 1.0 / 2.2)
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", default=os.path.join(ROOT, "scenes/mountain_gs/mountain_gs.ply"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "scenes/mountain_gs"))
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--htcp-bin", default=os.path.join(ROOT, "scenes/mountain_gs/dzvar_voxels.bin"))
    p.add_argument("--height", type=float, default=6400.0, help="ortho sensor height meters (htrdr DZVAR_ortho)")
    p.add_argument("--max-bounces", type=int, default=1024, help="volume walk budget (htrdr is unbounded)")
    p.add_argument("--mie-g", type=float, default=0.865, help="HG asymmetry (Mie LUT ~0.865 at 550 nm)")
    p.add_argument("--rebuild-scene", action="store_true", help="rebuild ply/voxels/gas in Python")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import mountain_gs_scene as mgs

    assets = mgs.ensure_assets(outdir=args.outdir, rebuild=args.rebuild_scene)
    args.ply = assets["ply"]
    args.htcp_bin = assets["voxels_bin"]
    gas = assets["gas"]

    device = torch.device("cuda")
    torch.zeros(1, device=device)  # force CUDA context before OptiX

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
    print("OptixTracer ready")

    gs = load_ply_gs(args.ply, device)
    print(f"GS particles: {gs['pos'].shape[0]}")
    tracer.build_bvh(gs["pos"], gs["rot"], gs["scl"], gs["dns"], 1, False)

    if os.path.isfile(args.htcp_bin):
        ka, ks, temp, origin, spacing = load_htcp_bin(args.htcp_bin)
        print(f"voxels from htcp dump {ks.shape} origin={origin} dx={spacing}")
    else:
        print("no htcp dump, using procedural Cu")
        nx, ny, nz = 128, 128, 40
        origin = np.array([0.0, 0.0, 850.0], np.float32)
        spacing = np.array([50.0, 50.0, 35.0], np.float32)
        ka, ks, temp = fbm_cumulus(nx, ny, nz, origin, spacing)

    from threedgptir_tracer.atmosphere import AtmosphereBuffers, SPECTRAL_SW

    az = math.radians(225.0)
    el = math.radians(30.0)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    diagnose_cloud_sun_tr(ka, ks, origin, spacing, sun)
    buffers = AtmosphereBuffers.from_numpy(
        device=device,
        voxel_ka=ka,
        voxel_ks=ks,
        voxel_temp=temp,
        voxel_origin=tuple(origin.tolist()),
        voxel_spacing=tuple(np.asarray(spacing, np.float32).tolist()),
        brick=8,
        gas_z=gas["z"],
        gas_ka=gas["ka"],
        gas_ks=gas["ks"],
        gas_temp=gas["temp"],
        sun_dir=sun,
        sun_radiance=24.0,
        sun_cos_half=1.0,
        spectral_mode=SPECTRAL_SW,
        wavelength_m=5.5e-7,
        mie_omega=1.0,
        mie_g=args.mie_g,
        path_tmax=1.2e5,
        max_bounces=args.max_bounces,
    )
    buffers.particle_density = gs["density"]
    buffers.particle_material = gs["material"]
    buffers.particle_radiance = gs["radiance"]
    buffers.particle_shading_normal = gs["normal"]
    # Sand plane mesh (same domain as htrdr desert ortho) fills GS holes; GS is closer relief.
    mesh_verts = torch.tensor(
        [[0.0, 0.0, 1.0], [6400.0, 0.0, 1.0], [6400.0, 6400.0, 1.0], [0.0, 6400.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    mesh_idx = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=device)
    mesh_alb = torch.tensor([[0.22, 0.16, 0.10], [0.22, 0.16, 0.10]], dtype=torch.float32, device=device)
    mesh_t = torch.tensor([300.0, 300.0], dtype=torch.float32, device=device)
    buffers.mesh_verts = mesh_verts
    buffers.mesh_idx = mesh_idx
    buffers.mesh_albedo = mesh_alb
    buffers.mesh_temp = mesh_t
    print(
        "voxel ks max/mean",
        float(np.max(ks)),
        float(np.mean(ks)),
        "kmax max",
        float(buffers.voxel_kmax.max().item()),
    )

    # Match htrdr DZVAR_ortho: pos=3200,3200,80000 tgt=3200,3200,0 up=0,1,0 height=6400
    center = np.array([3200.0, 3200.0, 80000.0], np.float32)
    look = np.array([0.0, 0.0, -1.0], np.float32)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    w = h = args.res
    acc = None
    chunk = min(args.spp, 4)
    nchunk = (args.spp + chunk - 1) // chunk
    print(f"GPU PTIR ortho {w}x{h} spp={args.spp} (chunks of {chunk})")
    from threedgptir_tracer.atmosphere import trace_atmosphere

    for c in range(nchunk):
        n = min(chunk, args.spp - c * chunk)
        ori, drc = make_ortho_rays(w, h, center, args.height, up, look, n, device)
        out = trace_atmosphere(tracer, ori, drc, buffers, frame_number=c)
        rgb = out[10]  # ray_pbr
        part = rgb.float().mean(dim=0).cpu().numpy()
        acc = part if acc is None else acc + part
        print(f"  chunk {c+1}/{nchunk} mean={part.mean():.4f}")
        if c == 0:
            hit = out[2][..., 0].float().cpu().numpy().reshape(-1)
            rad = rgb.float().mean(dim=-1).cpu().numpy().reshape(-1)
            dns = out[1][..., 0].float().cpu().numpy().reshape(-1)
            light = out[11].float().mean(dim=-1).cpu().numpy().reshape(-1)
            print(
                "  diag hitT p10/50/90",
                np.percentile(hit, [10, 50, 90]),
                "frac<76000",
                float((hit < 76000).mean()),
                "frac 76-79.2k (grid)",
                float(((hit >= 76000) & (hit < 79200)).mean()),
                "frac>79500 (ground)",
                float((hit > 79500).mean()),
            )
            grid_hit = (hit >= 76000) & (hit < 79200)
            gnd_hit = hit > 79500
            print(
                "  diag rad grid_hit",
                float(rad[grid_hit].mean()) if grid_hit.any() else None,
                "gnd",
                float(rad[gnd_hit].mean()) if gnd_hit.any() else None,
                "direct grid",
                float(light[grid_hit].mean()) if grid_hit.any() else None,
                "direct gnd",
                float(light[gnd_hit].mean()) if gnd_hit.any() else None,
                "density mean",
                float(dns.mean()),
            )
        del ori, drc, out, rgb
        torch.cuda.empty_cache()
    img = acc / float(nchunk)
    rgb8 = tonemap(img)
    from PIL import Image
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_path = os.path.join(args.outdir, f"render_ortho_gpu_{stamp}.png")
    Image.fromarray(rgb8).save(gpu_path)
    print("wrote", gpu_path)
    # Never overwrite frozen keep copies.
    keep_gpu = os.path.join(args.outdir, "render_ortho_gpu_keep.png")
    if not os.path.isfile(keep_gpu):
        Image.fromarray(rgb8).save(keep_gpu)

    orig = os.path.join(
        ROOT, "htrdr-atmosphere-spk/htrdr-Atmosphere-Starter-Pack-0.8.0/ortho_512x512x32.ppm"
    )
    if os.path.isfile(orig):
        oim = Image.open(orig).convert("RGB").resize((w, h), Image.NEAREST)
        canvas = Image.new("RGB", (w * 2 + 24, h + 48), (18, 18, 18))
        canvas.paste(oim, (8, 40))
        canvas.paste(Image.fromarray(rgb8), (w + 16, 40))
        try:
            from PIL import ImageDraw, ImageFont
            d = ImageDraw.Draw(canvas)
            d.text((8, 8), "htrdr ortho 512x512 spp32  (DZVAR + desert)", fill=(220, 220, 220))
            d.text((w + 16, 8), "GPU PTIR  Ks walk + Tr_abs + mesh-only volume NEE", fill=(220, 220, 220))
        except Exception:
            pass
        cmp = os.path.join(args.outdir, f"compare_ortho_{stamp}.png")
        canvas.save(cmp)
        print("wrote", cmp)
        keep_cmp = os.path.join(args.outdir, "compare_ortho_keep.png")
        if not os.path.isfile(keep_cmp):
            canvas.save(keep_cmp)


if __name__ == "__main__":
    main()
