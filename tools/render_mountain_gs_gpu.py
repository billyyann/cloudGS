#!/usr/bin/env python3
"""Orthographic GPU PTIR atmosphere path (voxel Woodcock + GS surfaces + optional mesh).

RTE lives in copied __raygen__rg → traceAtmospherePath. Cloud k is voxels only.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
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
        z_iface = None
        if int(hdr[3]) >= 2:
            z = np.fromfile(f, dtype=np.float32, count=nz + 1)
            if z.size == nz + 1:
                z_iface = z
    return ka, ks, temp, origin, spacing, z_iface


def analog_kext_tr(ka, ks, origin, spacing, org, direction, tmax=2.0e4, z_iface=None):
    """Host analog exp(-∫Kext) along a ray (same cell-step idea as GPU DDA)."""
    kext = np.asarray(ka, dtype=np.float32) + np.asarray(ks, dtype=np.float32)
    nz, ny, nx = kext.shape
    dx, dy, dz = [float(spacing[i]) for i in range(3)]
    ox, oy, oz = [float(origin[i]) for i in range(3)]
    zf = np.asarray(z_iface, dtype=np.float64) if z_iface is not None else None
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
        if zf is not None:
            iz = int(np.searchsorted(zf, p[2], side="right") - 1)
        else:
            iz = int(np.floor((p[2] - oz) / dz))
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            tau += float(kext[iz, iy, ix]) * ds
        p = p + d * ds
        t += ds
    if tau > 80.0:
        return 0.0
    return float(np.exp(-tau))


def diagnose_cloud_sun_tr(ka, ks, origin, spacing, sun, z_iface=None):
    """Cloud-top point toward the sun: Tr should be in (0, 1), not identically 0."""
    kext = np.asarray(ka, dtype=np.float32) + np.asarray(ks, dtype=np.float32)
    iz, iy, ix = np.unravel_index(int(np.argmax(kext)), kext.shape)
    if z_iface is not None and len(z_iface) == kext.shape[0] + 1:
        zc = 0.5 * (float(z_iface[iz]) + float(z_iface[iz + 1]))
    else:
        zc = float(origin[2]) + (iz + 0.5) * float(spacing[2])
    org = np.array(
        [
            float(origin[0]) + (ix + 0.5) * float(spacing[0]),
            float(origin[1]) + (iy + 0.5) * float(spacing[1]),
            zc,
        ],
        dtype=np.float64,
    )
    tr = analog_kext_tr(ka, ks, origin, spacing, org, sun, z_iface=z_iface)
    print(
        f"diag analog Tr(Kext) at cloud cell ({ix},{iy},{iz}) z={org[2]:.1f} toward sun: {tr:.4g}"
    )
    if not (0.0 < tr < 1.0):
        print("warning: expected analog sun Tr in (0,1)")


def persp_camera(w, h, pos, tgt, up, fov_y_deg):
    look = tgt - pos
    look = look / np.linalg.norm(look)
    up = up - look * np.dot(up, look)
    up = up / np.linalg.norm(up)
    right = np.cross(look, up)
    right = right / np.linalg.norm(right)
    half_v = math.tan(0.5 * math.radians(fov_y_deg))
    half_u = half_v * (w / float(h))
    return pos, look, right, up, half_u, half_v


def make_persp_rays(w, h, pos, tgt, up, fov_y_deg, spp, device, rng=None):
    """htrdr perspective: fov is vertical FOV in degrees (mountain.sh fov=30)."""
    pos, look, right, up, half_u, half_v = persp_camera(w, h, pos, tgt, up, fov_y_deg)
    if rng is None:
        rng = np.random.default_rng(0)
    ori = np.zeros((spp, h, w, 3), np.float32)
    drc = np.zeros((spp, h, w, 3), np.float32)
    for s in range(spp):
        js = (np.arange(w, dtype=np.float32)[None, :] + rng.random((h, w)).astype(np.float32)) / w
        is_ = (np.arange(h, dtype=np.float32)[:, None] + rng.random((h, w)).astype(np.float32)) / h
        u = (js - 0.5) * 2.0 * half_u
        v = (0.5 - is_) * 2.0 * half_v
        d = look[None, None, :] + right[None, None, :] * u[..., None] + up[None, None, :] * v[..., None]
        nrm = np.linalg.norm(d, axis=-1, keepdims=True)
        ori[s] = pos[None, None, :]
        drc[s] = (d / np.maximum(nrm, 1e-12)).astype(np.float32)
    return torch.from_numpy(ori).to(device), torch.from_numpy(drc).to(device)


def make_persp_rays_pixels(w, h, pos, tgt, up, fov_y_deg, ys, xs, rng):
    """One jittered camera ray per (y, x) film pixel. Returns (n, 3) float32."""
    pos, look, right, upv, half_u, half_v = persp_camera(w, h, pos, tgt, up, fov_y_deg)
    n = int(ys.size)
    js = (xs.astype(np.float32) + rng.random(n).astype(np.float32)) / float(w)
    is_ = (ys.astype(np.float32) + rng.random(n).astype(np.float32)) / float(h)
    u = (js - 0.5) * 2.0 * half_u
    v = (0.5 - is_) * 2.0 * half_v
    d = look[None, :] + right[None, :] * u[:, None] + upv[None, :] * v[:, None]
    nrm = np.linalg.norm(d, axis=-1, keepdims=True)
    ori = np.broadcast_to(pos.astype(np.float32), (n, 3)).copy()
    drc = (d / np.maximum(nrm, 1e-12)).astype(np.float32)
    return ori, drc


def make_ortho_rays(w, h, center, height, up, look, spp, device, rng=None):
    """htrdr orthographic: pos/tgt/up, height = vertical sensor extent (meters)."""
    look = look / np.linalg.norm(look)
    up = up - look * np.dot(up, look)
    up = up / np.linalg.norm(up)
    right = np.cross(look, up)
    right = right / np.linalg.norm(right)
    aspect = w / float(h)
    half_h = 0.5 * height
    half_w = half_h * aspect
    if rng is None:
        rng = np.random.default_rng(0)
    ori = np.zeros((spp, h, w, 3), np.float32)
    drc = np.zeros((spp, h, w, 3), np.float32)
    for s in range(spp):
        js = (np.arange(w, dtype=np.float32)[None, :] + rng.random((h, w)).astype(np.float32)) / w
        is_ = (np.arange(h, dtype=np.float32)[:, None] + rng.random((h, w)).astype(np.float32)) / h
        u = (js - 0.5) * 2.0 * half_w
        v = (0.5 - is_) * 2.0 * half_h
        o = center[None, None, :] + right[None, None, :] * u[..., None] + up[None, None, :] * v[..., None]
        ori[s] = o
        drc[s] = look[None, None, :]
    return torch.from_numpy(ori).to(device), torch.from_numpy(drc).to(device)


def make_ortho_rays_pixels(w, h, center, height, up, look, ys, xs, rng):
    look = look / np.linalg.norm(look)
    up = up - look * np.dot(up, look)
    up = up / np.linalg.norm(up)
    right = np.cross(look, up)
    right = right / np.linalg.norm(right)
    half_h = 0.5 * height
    half_w = half_h * (w / float(h))
    n = int(ys.size)
    js = (xs.astype(np.float32) + rng.random(n).astype(np.float32)) / float(w)
    is_ = (ys.astype(np.float32) + rng.random(n).astype(np.float32)) / float(h)
    u = (js - 0.5) * 2.0 * half_w
    v = (0.5 - is_) * 2.0 * half_h
    o = center[None, :] + right[None, :] * u[:, None] + up[None, :] * v[:, None]
    d = np.broadcast_to(look.astype(np.float32), (n, 3)).copy()
    return o.astype(np.float32), d


def pack_trace_pixels(tracer, buffers, ori_np, drc_np, frame, device, pack_w=256):
    from threedgptir_tracer.atmosphere import trace_atmosphere

    n = int(ori_np.shape[0])
    pack_h = (n + pack_w - 1) // pack_w
    pad = pack_h * pack_w
    ori = torch.zeros((1, pack_h, pack_w, 3), dtype=torch.float32, device=device)
    drc = torch.zeros((1, pack_h, pack_w, 3), dtype=torch.float32, device=device)
    ori.view(-1, 3)[:n].copy_(torch.from_numpy(np.ascontiguousarray(ori_np)))
    drc.view(-1, 3)[:n].copy_(torch.from_numpy(np.ascontiguousarray(drc_np)))
    out = trace_atmosphere(tracer, ori, drc, buffers, frame_number=int(frame))
    rgb = out[10].float().reshape(pad, 3)[:n].cpu().numpy()
    first = out[11].float().reshape(pad, 3)[:n].cpu().numpy()
    del out
    return rgb, first, n


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
    p.add_argument(
        "--film",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help="override perspective film size (default mountain/above=1280x720)",
    )
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--htcp-bin", default=os.path.join(ROOT, "scenes/mountain_gs/dzvar_voxels.bin"))
    p.add_argument("--height", type=float, default=6400.0, help="ortho sensor height meters (htrdr DZVAR_ortho)")
    p.add_argument(
        "--max-bounces",
        type=int,
        default=256,
        help="volume walk cap (htrdr is unbounded; 256 saturates this scene, 64 is ~9% dark)",
    )
    p.add_argument("--mie-g", type=float, default=0.865, help="HG asymmetry (Mie LUT ~0.865 at 550 nm)")
    p.add_argument("--rebuild-scene", action="store_true", help="rebuild ply/voxels/gas in Python")
    p.add_argument(
        "--no-clouds",
        action="store_true",
        help="zero voxel ka/ks and 1D gas so only GS+mesh ground is lit",
    )
    p.add_argument(
        "--camera",
        choices=("ortho", "mountain", "above"),
        default="ortho",
        help="ortho = desert_ortho nadir 6.4 km; mountain = mountain.sh (z=422 look up); "
        "above = z=4500 look down through Cu at the mountain",
    )
    p.add_argument(
        "--ground",
        choices=("mesh", "gs"),
        default="mesh",
        help="mesh = htrdr mountain.obj (default); gs = 3DGS + hole-fill quad (legacy 104659)",
    )
    p.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="trace a pixel window only (does not overwrite gpu_xyz_*.npy)",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="print host/OptiX wall times per chunk (set PTIR_PROFILE_RAYGEN=1 for CUDA events)",
    )
    p.add_argument(
        "--batch-cloud",
        action="store_true",
        help="compact sky vs cloud AABB into two optixLaunch (ATM_BATCH_CLOUD=1)",
    )
    p.add_argument(
        "--brick",
        type=int,
        default=8,
        help="Woodcock majorant brick edge (8 baseline; 4 is tighter kmax)",
    )
    p.add_argument(
        "--cont-stack",
        type=int,
        default=None,
        help="OptiX continuation stack bytes; 0=computed, default=16384 floor. Set before pipeline create.",
    )
    p.add_argument(
        "--adaptive",
        dest="adaptive",
        action="store_true",
        default=False,
        help="after warmup, spend remaining spp on high-Y-variance pixels",
    )
    p.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    p.add_argument(
        "--adaptive-warmup",
        type=int,
        default=32,
        help="uniform spp before variance allocation (default 32)",
    )
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.batch_cloud:
        os.environ["ATM_BATCH_CLOUD"] = "1"
    if args.cont_stack is not None:
        os.environ["ATM_CONT_STACK"] = str(int(args.cont_stack))

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

    use_mesh_ground = args.ground == "mesh"
    if use_mesh_ground:
        mountain_mesh = mgs.ensure_mountain_mesh(outdir=args.outdir)
        print("ground: htrdr mountain.obj mesh (GS off)")
    else:
        gs = load_ply_gs(args.ply, device)
        print(f"GS particles: {gs['pos'].shape[0]}")
        tracer.build_bvh(gs["pos"], gs["rot"], gs["scl"], gs["dns"], 1, False)

    z_iface = None
    if os.path.isfile(args.htcp_bin):
        ka, ks, temp, origin, spacing, z_iface = load_htcp_bin(args.htcp_bin)
        print(
            f"voxels from htcp dump {ks.shape} origin={origin} dx={spacing} "
            f"z_iface={'yes' if z_iface is not None else 'no'}"
        )
    else:
        print("no htcp dump, using procedural Cu")
        nx, ny, nz = 128, 128, 40
        origin = np.array([0.0, 0.0, 850.0], np.float32)
        spacing = np.array([50.0, 50.0, 35.0], np.float32)
        ka, ks, temp = fbm_cumulus(nx, ny, nz, origin, spacing)

    from threedgptir_tracer.atmosphere import (
        AtmosphereBuffers,
        SPECTRAL_SW,
        SPECTRAL_SW_CIE_XYZ,
        build_cie_xyz_tables,
        htpp_xyz_to_srgb,
    )

    az = math.radians(225.0)
    el = math.radians(30.0)
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    gas_ka, gas_ks = gas["ka"], gas["ks"]
    if args.no_clouds:
        ka = np.zeros_like(ka)
        ks = np.zeros_like(ks)
        gas_ka = np.zeros_like(gas_ka)
        gas_ks = np.zeros_like(gas_ks)
        print("no-clouds: voxel ka/ks and gas ka/ks zeroed")
    diagnose_cloud_sun_tr(ka, ks, origin, spacing, sun, z_iface=z_iface)
    buffers = AtmosphereBuffers.from_numpy(
        device=device,
        voxel_ka=ka,
        voxel_ks=ks,
        voxel_temp=temp,
        voxel_origin=tuple(origin.tolist()),
        voxel_spacing=tuple(np.asarray(spacing, np.float32).tolist()),
        brick=int(args.brick),
        gas_z=gas["z"],
        gas_ka=gas_ka,
        gas_ks=gas_ks,
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
    cie = build_cie_xyz_tables()
    buffers.cie_cdf = torch.as_tensor(cie["cdf"], dtype=torch.float32, device=device)
    buffers.cie_rcp_integral = torch.as_tensor(cie["rcp_integral"], dtype=torch.float32, device=device)
    buffers.cie_lambda0_nm = float(cie["lambda0_nm"])
    buffers.cie_dlambda_nm = float(cie["dlambda_nm"])
    sw = mgs.load_ecrad_sw_visible()
    if args.no_clouds:
        sw_ka = np.zeros_like(sw["ka_quad"])
        sw_ks = np.zeros_like(sw["ks_quad"])
    else:
        sw_ka, sw_ks = sw["ka_quad"], sw["ks_quad"]
    buffers.sw_lambda_lo = torch.as_tensor(sw["lambda_lo_nm"], dtype=torch.float32, device=device)
    buffers.sw_lambda_hi = torch.as_tensor(sw["lambda_hi_nm"], dtype=torch.float32, device=device)
    buffers.sw_gas_ka = torch.as_tensor(sw_ka, dtype=torch.float32, device=device).contiguous()
    buffers.sw_gas_ks = torch.as_tensor(sw_ks, dtype=torch.float32, device=device).contiguous()
    buffers.sw_nquad = torch.as_tensor(sw["nquad"], dtype=torch.int32, device=device).contiguous()
    buffers.sw_quad_cdf = torch.as_tensor(sw["quad_cdf"], dtype=torch.float32, device=device).contiguous()
    buffers.mie_cs_band = torch.as_tensor(sw["mie_cs_band"], dtype=torch.float32, device=device)
    buffers.mie_ca_band = torch.as_tensor(sw["mie_ca_band"], dtype=torch.float32, device=device)
    buffers.mie_g_nm = torch.as_tensor(sw["mie_g_nm"], dtype=torch.float32, device=device)
    buffers.mie_cs_ref = float(sw["mie_cs_ref"])
    buffers.mie_ca_ref = float(sw["mie_ca_ref"])
    sand = mgs.load_sand_mrumtl()
    buffers.brdf_lambda_nm = torch.as_tensor(sand["lambda_nm"], dtype=torch.float32, device=device)
    buffers.brdf_R = torch.as_tensor(sand["R"], dtype=torch.float32, device=device)
    print(
        "CIE XYZ",
        cie["nbins"],
        "bins  HTGOP SW bands",
        int(sw_ka.shape[0]),
        "nquad_max",
        int(sw_ka.shape[1]),
        "nlayers",
        int(sw_ka.shape[2]),
    )
    if use_mesh_ground:
        buffers.particle_density = torch.zeros((0, 12), dtype=torch.float32, device=device)
        buffers.particle_material = torch.zeros((0, 5), dtype=torch.float32, device=device)
        buffers.particle_radiance = torch.zeros((0, 1), dtype=torch.float32, device=device)
        buffers.particle_shading_normal = torch.zeros((0, 3), dtype=torch.float32, device=device)
        buffers.mesh_verts = torch.from_numpy(mountain_mesh["verts"]).to(device)
        buffers.mesh_idx = torch.from_numpy(mountain_mesh["idx"]).to(device)
        buffers.mesh_albedo = torch.from_numpy(mountain_mesh["albedo"]).to(device)
        buffers.mesh_temp = torch.from_numpy(mountain_mesh["temp"]).to(device)
    else:
        buffers.particle_density = gs["density"]
        buffers.particle_material = gs["material"]
        buffers.particle_radiance = gs["radiance"]
        buffers.particle_shading_normal = gs["normal"]
        if args.camera == "ortho":
            # Sand plane mesh (same domain as htrdr desert ortho) fills GS holes.
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

    rng = np.random.default_rng(0)
    do_adapt = bool(args.adaptive) and int(args.spp) > 1
    warmup = int(args.spp)
    if do_adapt:
        warmup = min(int(args.spp), max(int(args.adaptive_warmup), 1))
    chunk = min(warmup, 4)
    nchunk = (warmup + chunk - 1) // chunk
    from threedgptir_tracer.atmosphere import trace_atmosphere

    if args.camera == "ortho":
        w = h = args.res
        center = np.array([3200.0, 3200.0, 80000.0], np.float32)
        look = np.array([0.0, 0.0, -1.0], np.float32)
        up = np.array([0.0, 1.0, 0.0], np.float32)
        pos = tgt = None
        print(f"GPU PTIR ortho {w}x{h} spp={args.spp} warmup={warmup} adaptive={int(do_adapt)} (chunks of {chunk})")
    elif args.camera == "above":
        # ARMCu cloud ~1.02–2.27 km. Camera above cloud top, look down through
        # the slab at mountain.obj (z≤700). mountain.sh pose is unchanged.
        w, h = 1280, 720
        if args.film is not None:
            w, h = int(args.film[0]), int(args.film[1])
        pos = np.array([5000.0, 9000.0, 4500.0], np.float32)
        tgt = np.array([4000.0, 3200.0, 400.0], np.float32)
        up = np.array([0.0, 0.0, 1.0], np.float32)
        look = tgt - pos
        look = look / np.linalg.norm(look)
        print(
            f"GPU PTIR above-cloud persp {w}x{h} spp={args.spp} warmup={warmup} adaptive={int(do_adapt)} fov=30 "
            f"pos={pos.tolist()} tgt={tgt.tolist()} dir.z={float(look[2]):.3f} "
            f"(chunks of {chunk})"
        )
    else:
        w, h = 1280, 720
        if args.film is not None:
            w, h = int(args.film[0]), int(args.film[1])
        pos = np.array([10245.37, 615.02, 422.34], np.float32)
        tgt = np.array([-2977.39, 6829.40, 842.40], np.float32)
        up = np.array([0.0, 0.0, 1.0], np.float32)
        print(
            f"GPU PTIR mountain.sh persp {w}x{h} spp={args.spp} warmup={warmup} "
            f"adaptive={int(do_adapt)} fov=30 (chunks of {chunk})"
        )

    if args.crop is not None:
        crop_x0, crop_y0, crop_x1, crop_y1 = args.crop
        th, tw = crop_y1 - crop_y0, crop_x1 - crop_x0
    else:
        crop_x0, crop_y0 = 0, 0
        th, tw = h, w
    sum_xyz = np.zeros((th, tw, 3), np.float64)
    sum_first = np.zeros((th, tw, 3), np.float64)
    sum_y2 = np.zeros((th, tw), np.float64)
    count = np.zeros((th, tw), np.float64)

    t_loop0 = time.perf_counter()
    prof = {"rays": 0.0, "trace": 0.0, "d2h": 0.0, "optix_ms": 0.0, "nray": 0}
    acc = None
    acc_first = None
    for c in range(nchunk):
        n = min(chunk, warmup - c * chunk)
        t0 = time.perf_counter()
        if args.camera == "ortho":
            ori, drc = make_ortho_rays(w, h, center, args.height, up, look, n, device, rng=rng)
        else:
            ori, drc = make_persp_rays(w, h, pos, tgt, up, 30.0, n, device, rng=rng)
        if args.crop is not None:
            x0, y0, x1, y1 = args.crop
            ori = ori[:, y0:y1, x0:x1].contiguous()
            drc = drc[:, y0:y1, x0:x1].contiguous()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = trace_atmosphere(tracer, ori, drc, buffers, frame_number=c)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        optix_ms = 0.0
        if hasattr(tracer, "get_last_optix_launch_ms"):
            optix_ms = float(tracer.get_last_optix_launch_ms())
        rgb = out[10]  # ray_pbr (full CIE XYZ)
        first = out[11]  # ray_light: first volume NEE only
        rgb_np = rgb.float().cpu().numpy()
        first_np = first.float().cpu().numpy()
        part = rgb_np.mean(axis=0)
        part_first = first_np.mean(axis=0)
        t3 = time.perf_counter()
        sum_xyz += rgb_np.sum(axis=0)
        sum_first += first_np.sum(axis=0)
        sum_y2 += (rgb_np[..., 1] ** 2).sum(axis=0)
        count += float(n)
        acc = part if acc is None else acc + part
        acc_first = part_first if acc_first is None else acc_first + part_first
        nray = int(ori.shape[0] * ori.shape[1] * ori.shape[2])
        prof["rays"] += t1 - t0
        prof["trace"] += t2 - t1
        prof["d2h"] += t3 - t2
        prof["optix_ms"] += optix_ms
        prof["nray"] += nray
        extra = ""
        if args.profile or optix_ms > 0.0:
            extra = (
                f"  rays={t1-t0:.2f}s  optixLaunch={optix_ms/1000.0:.2f}s  "
                f"trace_wall={t2-t1:.2f}s  d2h={t3-t2:.2f}s  "
                f"{nray/max(t2-t1,1e-9):.0f} rays/s"
            )
        print(f"  chunk {c+1}/{nchunk} mean={part.mean():.4f}{extra}")
        if c == 0:
            hit = out[2][..., 0].float().cpu().numpy().reshape(-1)
            rad = rgb.float().mean(dim=-1).cpu().numpy().reshape(-1)
            dns = out[1][..., 0].float().cpu().numpy().reshape(-1)
            light = out[11].float().mean(dim=-1).cpu().numpy().reshape(-1)
            ev = out[8][..., 0].float().cpu().numpy()
            ev_flat = ev.reshape(-1)
            names = {
                0: "none",
                1: "absorb_vox",
                2: "scatter_vox",
                3: "absorb_gas",
                4: "scatter_gas",
                5: "surface",
                6: "escape",
            }
            ev_parts = []
            for k, name in names.items():
                ev_parts.append(f"{name}={float((ev_flat == k).mean()):.3f}")
            print("  diag first_event", " ".join(ev_parts))
            dz = drc[..., 2].float().cpu().numpy().reshape(-1)
            print(
                f"  diag cam dir.z down={float((dz < 0).mean()):.3f} "
                f"mean={float(dz.mean()):.3f} p10/p90={np.percentile(dz, [10, 90])}"
            )
            if args.camera == "mountain":
                ntop = max(1, ev.shape[-2] // 5)
                top_ev = ev[..., :ntop, :].reshape(-1)
                top_lum = rgb.float().mean(dim=-1).cpu().numpy()[..., :ntop, :].reshape(-1)
                print(
                    "  diag sky_top20 first scatter_gas",
                    float((top_ev == 4).mean()),
                    "escape",
                    float((top_ev == 6).mean()),
                    "lum<5",
                    float((top_lum < 5.0).mean()),
                    "lum_mean",
                    float(top_lum.mean()),
                )

                def _lum_by_event(label, ev_a, lum_a):
                    print(f"  diag {label} by first_event")
                    for k, name in ((4, "scatter_gas"), (6, "escape"), (2, "scatter_vox"), (5, "surface")):
                        m = ev_a == k
                        n = int(m.sum())
                        if n == 0:
                            print(f"    {name}: n=0")
                            continue
                        sl = lum_a[m]
                        print(
                            f"    {name}: n={n} frac={n / len(ev_a):.3f} "
                            f"lum_mean={float(sl.mean()):.4g} lum_p50={float(np.median(sl)):.4g} "
                            f"lum_p90={float(np.percentile(sl, 90)):.4g} "
                            f"frac<0.05={float((sl < 0.05).mean()):.3f}"
                        )
                    holes = (ev_a != 2) & (ev_a != 5)
                    print(
                        f"    non_cloud_ground: n={int(holes.sum())} "
                        f"scatter_gas={float((ev_a[holes] == 4).mean()) if holes.any() else 0.0} "
                        f"escape={float((ev_a[holes] == 6).mean()) if holes.any() else 0.0} "
                        f"lum_mean={float(lum_a[holes].mean()) if holes.any() else 0.0}"
                    )

                _lum_by_event("sky_top20", top_ev, top_lum)
                top_hit = out[2][..., 0].float().cpu().numpy()[..., :ntop, :].reshape(-1)
                for k, name in ((4, "scatter_gas"), (6, "escape"), (2, "scatter_vox")):
                    m = top_ev == k
                    if not m.any():
                        continue
                    ht = top_hit[m]
                    print(
                        f"    {name} hitT p10/50/90",
                        np.percentile(ht, [10, 50, 90]),
                        "z≈",
                        422.34 + 0.26 * np.percentile(ht, [10, 50, 90]),
                    )
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

    if do_adapt and int(args.spp) > warmup:
        mean_y = sum_xyz[..., 1] / np.maximum(count, 1.0)
        var_y = np.maximum(sum_y2 / np.maximum(count, 1.0) - mean_y * mean_y, 0.0)
        std = np.sqrt(var_y)
        floor = float(np.percentile(std, 10)) * 0.05 + 1e-6
        weight = std + floor
        weight = weight / max(float(weight.sum()), 1e-12)
        remain = int(args.spp - warmup) * th * tw
        extra = np.floor(remain * weight).astype(np.int32)
        cap = int(args.spp) - warmup
        extra = np.clip(extra, 0, cap)
        print(
            f"adaptive extra: target spp={args.spp} warmup={warmup} "
            f"extra min/mean/p50/max={int(extra.min())}/{float(extra.mean()):.1f}/"
            f"{int(np.median(extra))}/{int(extra.max())}  "
            f"at_cap={float((extra == cap).mean()):.3f}  at_0={float((extra == 0).mean()):.3f}  "
            f"total_extra={int(extra.sum())} ({100.0 * extra.sum() / max(th * tw * args.spp, 1):.1f}% of spp*{th}*{tw})"
        )
        wave = 0
        while True:
            if int(extra.max()) >= 4:
                k = 4
                mask = extra >= k
            else:
                k = 1
                mask = extra > 0
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                break
            max_pix = th * tw
            if ys.size > max_pix:
                key = extra[ys, xs]
                pick = np.argpartition(-key, max_pix)[:max_pix]
                ys, xs = ys[pick], xs[pick]
            if k > 1:
                ys_l = np.repeat(ys, k)
                xs_l = np.repeat(xs, k)
            else:
                ys_l, xs_l = ys, xs
            film_ys = ys_l + crop_y0
            film_xs = xs_l + crop_x0
            t0 = time.perf_counter()
            if args.camera == "ortho":
                ori_np, drc_np = make_ortho_rays_pixels(
                    w, h, center, args.height, up, look, film_ys, film_xs, rng
                )
            else:
                ori_np, drc_np = make_persp_rays_pixels(
                    w, h, pos, tgt, up, 30.0, film_ys, film_xs, rng
                )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            rgb_np, first_np, nray = pack_trace_pixels(
                tracer, buffers, ori_np, drc_np, 10000 + wave, device
            )
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            t3 = t2
            if k > 1:
                raw_rgb = rgb_np.reshape(-1, k, 3)
                raw_first = first_np.reshape(-1, k, 3)
                sum_xyz[ys, xs] += raw_rgb.sum(axis=1)
                sum_first[ys, xs] += raw_first.sum(axis=1)
                sum_y2[ys, xs] += (raw_rgb[..., 1] ** 2).sum(axis=1)
                count[ys, xs] += float(k)
            else:
                sum_xyz[ys, xs] += rgb_np
                sum_first[ys, xs] += first_np
                sum_y2[ys, xs] += rgb_np[:, 1] ** 2
                count[ys, xs] += 1.0
            extra[ys, xs] -= k
            optix_ms = 0.0
            if hasattr(tracer, "get_last_optix_launch_ms"):
                optix_ms = float(tracer.get_last_optix_launch_ms())
            prof["rays"] += t1 - t0
            prof["trace"] += t2 - t1
            prof["d2h"] += t3 - t2
            prof["optix_ms"] += optix_ms
            prof["nray"] += nray
            if wave % 8 == 0 or extra.max() == 0:
                extra_s = (
                    f"  rays={t1-t0:.2f}s  trace_wall={t2-t1:.2f}s  {nray/max(t2-t1,1e-9):.0f} rays/s"
                    if args.profile
                    else ""
                )
                print(
                    f"  adaptive wave {wave+1} n={nray} remain={int(extra.sum())} "
                    f"meanY={float((sum_xyz[...,1]/np.maximum(count,1)).mean()):.4f}{extra_s}"
                )
            wave += 1
            torch.cuda.empty_cache()

    loop_s = time.perf_counter() - t_loop0
    nray = max(prof["nray"], 1)
    print(
        f"profile loop={loop_s:.1f}s  rays={prof['rays']:.1f}s  "
        f"optixLaunch={prof['optix_ms']/1000.0:.1f}s  trace_wall={prof['trace']:.1f}s  "
        f"d2h={prof['d2h']:.1f}s  rays={nray}  "
        f"{nray/max(prof['trace'],1e-9):.0f} camera-rays/s  "
        f"{1e3*prof['trace']/nray:.3f} ms/camera-ray  "
        f"(CIE same-path XYZ; ATM_CIE_SPLIT=1 for legacy 3x)"
    )
    denom = np.maximum(count[..., None], 1.0)
    img = (sum_xyz / denom).astype(np.float32)
    img_first = (sum_first / denom).astype(np.float32)
    print(
        f"  spp/pixel min/mean/p50/max={float(count.min()):.0f}/{float(count.mean()):.1f}/"
        f"{float(np.median(count)):.0f}/{float(count.max()):.0f}"
    )
    if args.crop is not None:
        x0, y0, x1, y1 = args.crop
        print(
            f"crop {x0},{y0}-{x1},{y1} max_bounces={args.max_bounces} "
            f"GPU Y mean/p50={float(img[...,1].mean()):.3f}/{float(np.median(img[...,1])):.3f}"
        )
        htrdr_txt = None
        if args.camera == "mountain":
            htrdr_txt = os.path.join(args.outdir, "htrdr_ref_mountain_1280x720x256.txt")
        elif args.camera == "ortho":
            htrdr_txt = os.path.join(args.outdir, "htrdr_ref_ortho_512x512x256.txt")
        if htrdr_txt and os.path.isfile(htrdr_txt):
            href = np.loadtxt(htrdr_txt, skiprows=1, dtype=np.float64).reshape(-1, 8)
            hy = href[:, 2].reshape(h, w)[y0:y1, x0:x1]
            hx = href[:, 0].reshape(h, w)[y0:y1, x0:x1]
            gy = img[..., 1]
            fy = img_first[..., 1] if img_first is not None else np.zeros_like(gy)
            ry = gy - fy
            print(
                f"  htrdr crop Y mean/p50={float(hy.mean()):.3f}/{float(np.median(hy)):.3f}  "
                f"GPU/htrdr mean={float(gy.mean()/max(hy.mean(),1e-12)):.3f}  "
                f"p50={float(np.median(gy)/max(np.median(hy),1e-12)):.3f}  "
                f"Y/X htrdr={float(hy.mean()/max(hx.mean(),1e-12)):.3f} GPU={float(gy.mean()/max(img[...,0].mean(),1e-12)):.3f}"
            )
            print(
                f"  GPU first-vol-NEE Y mean={float(fy.mean()):.3f}  "
                f"later Y mean={float(ry.mean()):.3f}  "
                f"first/total={float(fy.mean()/max(gy.mean(),1e-12)):.3f}"
            )
            thr = float(np.percentile(hy, 70))
            bright = hy > thr
            dark = ~bright
            if bright.any():
                print(
                    f"  bright Y>p70 n={int(bright.sum())}  "
                    f"htrdr mean/p50={float(hy[bright].mean()):.3f}/{float(np.median(hy[bright])):.3f}  "
                    f"GPU mean/p50={float(gy[bright].mean()):.3f}/{float(np.median(gy[bright])):.3f}  "
                    f"ratio mean={float(gy[bright].mean()/hy[bright].mean()):.3f}  "
                    f"p50={float(np.median(gy[bright])/max(np.median(hy[bright]),1e-12)):.3f}"
                )
                print(
                    f"  bright first-vol-NEE/htrdr={float(fy[bright].mean()/hy[bright].mean()):.3f}  "
                    f"later/htrdr={float(ry[bright].mean()/hy[bright].mean()):.3f}  "
                    f"first/GPU={float(fy[bright].mean()/max(gy[bright].mean(),1e-12)):.3f}"
                )
                print(
                    f"  surround first-vol-NEE/htrdr={float(fy[dark].mean()/max(hy[dark].mean(),1e-12)):.3f}  "
                    f"later/htrdr={float(ry[dark].mean()/max(hy[dark].mean(),1e-12)):.3f}  "
                    f"first/GPU={float(fy[dark].mean()/max(gy[dark].mean(),1e-12)):.3f}"
                )
            crop_tag = f"crop_{x0}_{y0}_{x1}_{y1}"
            np.save(os.path.join(args.outdir, f"gpu_xyz_{crop_tag}.npy"), img.astype(np.float32))
            if img_first is not None:
                np.save(os.path.join(args.outdir, f"gpu_xyz_first_{crop_tag}.npy"), img_first.astype(np.float32))
                print("wrote", os.path.join(args.outdir, f"gpu_xyz_{crop_tag}.npy"), "and first-vol-NEE npy")
        return
    if buffers.spectral_mode == SPECTRAL_SW_CIE_XYZ and buffers.cie_cdf is not None and buffers.cie_cdf.numel() > 0:
        X, Y, Z = img[..., 0], img[..., 1], img[..., 2]
        y995 = float(np.percentile(Y, 99.5))
        print(
            "  XYZ mean",
            float(X.mean()),
            float(Y.mean()),
            float(Z.mean()),
            "Y/X",
            float(Y.mean() / max(X.mean(), 1e-12)),
            "Z/X",
            float(Z.mean() / max(X.mean(), 1e-12)),
            "Y p50/p995",
            float(np.median(Y)),
            y995,
        )
        htrdr_txt = None
        if args.camera == "mountain":
            htrdr_txt = os.path.join(args.outdir, "htrdr_ref_mountain_1280x720x256.txt")
        elif args.camera == "ortho":
            htrdr_txt = os.path.join(args.outdir, "htrdr_ref_ortho_512x512x256.txt")
        white = None
        if htrdr_txt and os.path.isfile(htrdr_txt):
            href = np.loadtxt(htrdr_txt, skiprows=1, dtype=np.float64)
            href = href.reshape(-1, 8)
            href_y995 = float(np.percentile(href[:, 2], 99.5))
            href_yx = float(href[:, 2].mean() / max(href[:, 0].mean(), 1e-12))
            href_zx = float(href[:, 4].mean() / max(href[:, 0].mean(), 1e-12))
            print(
                "  htrdr XYZ mean",
                float(href[:, 0].mean()),
                float(href[:, 2].mean()),
                float(href[:, 4].mean()),
                "Y/X",
                href_yx,
                "Z/X",
                href_zx,
                "Y p50/p995",
                float(np.median(href[:, 2])),
                href_y995,
            )
            # Same htpp operator as the frozen left png (its Y 99.5%). GPU's own
            # p995 is fireflies at low spp; using it is a different tone map.
            white = href_y995
            print("  htpp white = left png Y p995", white, "  GPU auto p995", y995)
        rgb8 = (htpp_xyz_to_srgb(img, exposure=0.2, white=white) * 255.0 + 0.5).astype(np.uint8)
        print("display: htpp filmic XYZ→sRGB exposure=0.2 white=left png")
    else:
        rgb8 = tonemap(img)
    from PIL import Image
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = ""
    if args.camera == "mountain":
        tag += "_mountain"
    elif args.camera == "above":
        tag += "_above"
    if args.ground == "mesh":
        tag += "_mesh"
    if args.no_clouds:
        tag += "_nocloud"
    if buffers.spectral_mode == SPECTRAL_SW_CIE_XYZ:
        np.save(os.path.join(args.outdir, f"gpu_xyz{tag}.npy"), img.astype(np.float32))
        print("wrote", os.path.join(args.outdir, f"gpu_xyz{tag}.npy"))
    gpu_path = os.path.join(args.outdir, f"render_ortho_gpu{tag}_{stamp}.png")
    Image.fromarray(rgb8).save(gpu_path)
    print("wrote", gpu_path)
    # Never overwrite frozen keep copies.
    keep_gpu = os.path.join(args.outdir, "render_ortho_gpu_keep.png")
    if args.camera == "ortho" and args.ground == "gs" and not args.no_clouds and not os.path.isfile(keep_gpu):
        Image.fromarray(rgb8).save(keep_gpu)

    spk = os.path.join(ROOT, "htrdr-atmosphere-spk/htrdr-Atmosphere-Starter-Pack-0.8.0")
    orig = None
    left_label = ""
    if args.camera == "mountain":
        orig = os.path.join(args.outdir, "htrdr_ref_mountain_1280x720x256.png")
        if not os.path.isfile(orig):
            orig = os.path.join(spk, "test_320x240x16.ppm")
        left_label = "htrdr mountain.sh 1280x720 spp256  (frozen png)"
    elif args.camera == "ortho":
        orig = os.path.join(args.outdir, "htrdr_ref_ortho_512x512x256.png")
        if not os.path.isfile(orig):
            orig = os.path.join(spk, "ortho_512x512x32.ppm")
        left_label = "htrdr ortho 512x512 spp256  (frozen png)"
    if orig and os.path.isfile(orig):
        oim = Image.open(orig).convert("RGB").resize((w, h), Image.NEAREST)
        canvas = Image.new("RGB", (w * 2 + 24, h + 48), (18, 18, 18))
        canvas.paste(oim, (8, 40))
        canvas.paste(Image.fromarray(rgb8), (w + 16, 40))
        try:
            from PIL import ImageDraw, ImageFont
            d = ImageDraw.Draw(canvas)
            d.text((8, 8), left_label, fill=(220, 220, 220))
            gpu_label = (
                "GPU PTIR  ground only (no voxel cloud, no gas)"
                if args.no_clouds
                else "GPU PTIR  Ks walk + Tr_abs + mesh-only volume NEE"
            )
            if args.ground == "mesh":
                gpu_label = "GPU PTIR  mountain.obj mesh" + ("  no clouds" if args.no_clouds else "")
            if args.camera == "mountain":
                gpu_label = "GPU PTIR  mountain.sh camera" + ("  no clouds" if args.no_clouds else "")
            gpu_label = f"{gpu_label}  spp{args.spp}"
            d.text((w + 16, 8), gpu_label, fill=(220, 220, 220))
        except Exception:
            pass
        cmp = os.path.join(args.outdir, f"compare_ortho{tag}_{stamp}.png")
        canvas.save(cmp)
        print("wrote", cmp)
        keep_cmp = os.path.join(args.outdir, "compare_ortho_keep.png")
        if args.camera == "ortho" and args.ground == "gs" and not args.no_clouds and not os.path.isfile(keep_cmp):
            canvas.save(keep_cmp)


if __name__ == "__main__":
    main()
