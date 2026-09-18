#!/usr/bin/env python3
"""Load a downloaded aircraft mesh (CGTrader OBJ/glTF) into OptiX meters.

Shahed-136 free low-poly (Royalty Free, no AI):
  https://www.cgtrader.com/free-3d-models/military/other/shahed-136-geran-2-low-poly-model
Drop the unzipped OBJ/glTF/GLB into scenes/aircraft/shahed136/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "gpu"))

from threedgptir_tracer.atmosphere import MTL_RGB_LAMBERT

SHAHED_DIR = os.path.join(ROOT, "scenes", "aircraft", "shahed136")
SHAHED_URL = (
    "https://www.cgtrader.com/free-3d-models/military/other/"
    "shahed-136-geran-2-low-poly-model"
)
SHAHED_LENGTH_M = 3.5
MESH_EXTS = (".obj", ".glb", ".gltf", ".stl")


def _box_mesh(center, half, R):
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    hx, hy, hz = (float(half[0]), float(half[1]), float(half[2]))
    verts = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )
    idx = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )
    alb = np.full((idx.shape[0], 3), float(R), np.float32)
    return verts, idx, alb


def _cyl_x_mesh(x0, x1, r, R, nseg=16):
    nseg = max(int(nseg), 6)
    t = np.linspace(0.0, 2.0 * np.pi, nseg, endpoint=False)
    ring = np.stack([np.zeros(nseg), r * np.cos(t), r * np.sin(t)], axis=1).astype(np.float32)
    v0 = ring + np.array([x0, 0.0, 0.0], np.float32)
    v1 = ring + np.array([x1, 0.0, 0.0], np.float32)
    c0 = np.array([[x0, 0.0, 0.0]], np.float32)
    c1 = np.array([[x1, 0.0, 0.0]], np.float32)
    verts = np.concatenate([v0, v1, c0, c1], axis=0)
    tris = []
    for i in range(nseg):
        j = (i + 1) % nseg
        tris.append([i, j, nseg + j])
        tris.append([i, nseg + j, nseg + i])
        tris.append([2 * nseg, j, i])
        tris.append([2 * nseg + 1, nseg + i, nseg + j])
    idx = np.asarray(tris, np.int32)
    alb = np.full((idx.shape[0], 3), float(R), np.float32)
    return verts, idx, alb


def _cone_x_mesh(x0, x1, r0, r1, R, nseg=16):
    nseg = max(int(nseg), 6)
    t = np.linspace(0.0, 2.0 * np.pi, nseg, endpoint=False)
    ring0 = np.stack([np.full(nseg, x0), r0 * np.cos(t), r0 * np.sin(t)], axis=1).astype(np.float32)
    ring1 = np.stack([np.full(nseg, x1), r1 * np.cos(t), r1 * np.sin(t)], axis=1).astype(np.float32)
    verts = np.concatenate([ring0, ring1], axis=0)
    tris = []
    for i in range(nseg):
        j = (i + 1) % nseg
        tris.append([i, j, nseg + j])
        tris.append([i, nseg + j, nseg + i])
    idx = np.asarray(tris, np.int32)
    alb = np.full((idx.shape[0], 3), float(R), np.float32)
    return verts, idx, alb


def _quad_slab(p00, p10, p11, p01, thick, R):
    p = np.stack([p00, p10, p11, p01], axis=0).astype(np.float64)
    n = np.cross(p[1] - p[0], p[3] - p[0])
    n = n / (np.linalg.norm(n) + 1e-12)
    h = 0.5 * float(thick)
    top = (p + h * n).astype(np.float32)
    bot = (p - h * n).astype(np.float32)
    verts = np.concatenate([top, bot], axis=0)
    idx = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )
    alb = np.full((idx.shape[0], 3), float(R), np.float32)
    return verts, idx, alb


def _join_parts(parts):
    verts, idx, alb = [], [], []
    base = 0
    for v, t, a in parts:
        verts.append(v)
        idx.append(t + base)
        alb.append(a)
        base += int(v.shape[0])
    return (
        np.concatenate(verts, axis=0).astype(np.float32),
        np.concatenate(idx, axis=0).astype(np.int32),
        np.concatenate(alb, axis=0).astype(np.float32),
    )


def _place_mesh(verts, idx, alb, center, heading_rad, name):
    c, s = float(np.cos(heading_rad)), float(np.sin(heading_rad))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)
    verts = verts @ rot.T
    verts = verts + np.asarray(center, np.float32).reshape(1, 3)
    temp = np.full((idx.shape[0],), 288.15, np.float32)
    mat_id = np.full((idx.shape[0],), MTL_RGB_LAMBERT, np.int32)
    bb0 = verts.min(axis=0)
    bb1 = verts.max(axis=0)
    print(
        f"{name} mesh verts={verts.shape[0]} tris={idx.shape[0]}  "
        f"AABB=[{bb0[0]:.2f},{bb0[1]:.2f},{bb0[2]:.2f}]-"
        f"[{bb1[0]:.2f},{bb1[1]:.2f},{bb1[2]:.2f}]  "
        f"size={(bb1 - bb0).tolist()} m  center={np.asarray(center, float).tolist()}"
    )
    return {
        "verts": np.ascontiguousarray(verts, np.float32),
        "idx": np.ascontiguousarray(idx, np.int32),
        "albedo": np.ascontiguousarray(alb, np.float32),
        "temp": temp,
        "mat_id": mat_id,
        "source": "procedural",
        "length_m": float(bb1[0] - bb0[0]),
    }


def make_shahed136_coarse_mesh(center, heading_rad: float = 0.0) -> dict:
    """White delta UAV from the product photo. Visual only, 1:1 meters.

    +X nose, +Z up. Length 3.5 m, span 2.5 m. Not a CAD / aero model.
    """
    r_w = 0.78
    r_k = 0.05
    fuse_r = 0.175
    nseg = 20
    parts = []
    # fuselage: rounded nose → cylinder → tail taper
    parts.append(_cone_x_mesh(1.22, 1.58, fuse_r, 0.11, r_w, nseg))
    parts.append(_cone_x_mesh(1.58, 1.75, 0.11, 0.018, r_w, 16))
    parts.append(_cyl_x_mesh(-1.32, 1.22, fuse_r, r_w, nseg))
    parts.append(_cone_x_mesh(-1.72, -1.32, 0.07, fuse_r, r_w, nseg))
    # black band just aft of the nose, as in the photo
    parts.append(_cyl_x_mesh(1.05, 1.16, fuse_r + 0.012, r_k, nseg))
    # delta wing: root at the band, tips at the tail with a short tip chord
    root_le = (0.52, fuse_r, 0.00)
    root_te = (-1.50, fuse_r, 0.00)
    tip_le = (-1.08, 1.25, 0.05)
    tip_te = (-1.52, 1.25, 0.05)
    parts.append(_quad_slab(root_le, tip_le, tip_te, root_te, 0.045, r_w))
    parts.append(
        _quad_slab(
            (root_le[0], -root_le[1], root_le[2]),
            (root_te[0], -root_te[1], root_te[2]),
            (tip_te[0], -tip_te[1], tip_te[2]),
            (tip_le[0], -tip_le[1], tip_le[2]),
            0.045,
            r_w,
        )
    )
    # wingtip vertical fins
    for ys in (1.25, -1.25):
        parts.append(_box_mesh((-1.36, ys, 0.24), (0.16, 0.012, 0.22), r_w))
    # dorsal intake (black oval on the spine)
    parts.append(_box_mesh((-0.18, 0.0, fuse_r + 0.03), (0.20, 0.07, 0.045), r_k))
    parts.append(_cyl_x_mesh(-0.08, 0.12, 0.055, r_k, 12))
    v, t, a = parts[-1]
    v[:, 2] += fuse_r + 0.01
    parts[-1] = (v, t, a)
    # pusher disc + hub (small; photo shows a dark tail rotor, not a face-on disc)
    parts.append(_cyl_x_mesh(-1.78, -1.762, 0.24, r_k, 16))
    parts.append(_cyl_x_mesh(-1.76, -1.68, 0.045, r_k, 10))
    verts, idx, alb = _join_parts(parts)
    return _place_mesh(verts, idx, alb, center, heading_rad, "Shahed-136 coarse")


def make_an124_coarse_mesh(center, heading_rad: float = 0.0) -> dict:
    """An-124 Ruslan low-poly. +X nose +Z up. 1:1 meters.

    Length 69.1 m, span 73.3 m, height ~20.8 m. High wing, T-tail, four
    underslung engines. Visual only, not CAD.
    """
    r_body = 0.42
    r_wing = 0.32
    r_eng = 0.08
    fuse_r = 3.35
    x_nose = 34.55
    x_tail = -34.55
    nseg = 12
    z_w = fuse_r + 0.25
    parts = []
    parts.append(_cyl_x_mesh(-26.0, 24.0, fuse_r, r_body, nseg))
    parts.append(_cone_x_mesh(24.0, x_nose, fuse_r, 0.95, r_body, nseg))
    parts.append(_cone_x_mesh(x_tail, -26.0, 2.4, fuse_r, r_body, nseg))
    parts.append(_box_mesh((1.0, 0.0, z_w), (10.0, 36.65, 0.70), r_wing))
    parts.append(_box_mesh((-8.0, 0.0, z_w - 0.08), (7.5, 18.0, 0.55), r_wing))
    z_vt = fuse_r + 8.4
    parts.append(_box_mesh((x_tail + 6.0, 0.0, z_vt), (5.2, 0.42, 7.1), r_body))
    parts.append(_box_mesh((x_tail + 5.5, 0.0, z_vt + 7.1), (3.4, 12.4, 0.38), r_body))
    for y in (-28.5, -16.8, 16.8, 28.5):
        parts.append(_cyl_x_mesh(-2.0, 8.8, 1.18, r_eng, 10))
        v, t, a = parts[-1]
        v[:, 1] += y
        v[:, 2] += z_w - 2.45
        parts[-1] = (v, t, a)
    verts, idx, alb = _join_parts(parts)
    return _place_mesh(verts, idx, alb, center, heading_rad, "An-124 coarse")


def _extract_zip(src: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)
    print(f"extracted {src} → {dest}")


def find_mesh_file(path: str | None = None, dest_dir: str = SHAHED_DIR) -> str:
    if path:
        path = os.path.abspath(os.path.expanduser(path))
        if path.lower().endswith(".zip"):
            _extract_zip(path, dest_dir)
            path = None
        elif os.path.isfile(path):
            return path
        else:
            raise FileNotFoundError(path)
    os.makedirs(dest_dir, exist_ok=True)
    hits = []
    for ext in MESH_EXTS:
        hits.extend(glob.glob(os.path.join(dest_dir, f"**/*{ext}"), recursive=True))
    hits = [h for h in hits if os.path.isfile(h)]
    if not hits:
        raise FileNotFoundError(
            "Shahed-136 mesh not found. Download the CGTrader free OBJ/glTF:\n"
            f"  {SHAHED_URL}\n"
            "Unzip into:\n"
            f"  {dest_dir}\n"
            "or pass --mesh /path/to/file.obj (zip also ok)."
        )
    hits.sort(key=lambda p: (0 if p.lower().endswith(".obj") else 1, len(p)))
    return hits[0]


def load_triangle_mesh(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    loaded = trimesh.load(path, force=None, process=True)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RuntimeError(f"no triangle geometry in {path}")
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"{path} is {type(mesh)}, need a triangle mesh")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise RuntimeError(f"{path} has no faces")
    verts = np.asarray(mesh.vertices, np.float32)
    idx = np.asarray(mesh.faces, np.int32)
    alb = None
    vis = getattr(mesh, "visual", None)
    if vis is not None and getattr(vis, "face_colors", None) is not None:
        fc = np.asarray(vis.face_colors)
        if fc.ndim == 2 and fc.shape[0] == idx.shape[0] and fc.shape[1] >= 3:
            alb = (fc[:, :3].astype(np.float32) / 255.0)
    if alb is None:
        alb = np.full((idx.shape[0], 3), 0.72, np.float32)
    print(
        f"loaded {path}  verts={verts.shape[0]} tris={idx.shape[0]}  "
        f"raw AABB={(verts.max(0) - verts.min(0)).tolist()}"
    )
    return verts, idx, alb


def orient_fuselage_x_up_z(verts: np.ndarray) -> np.ndarray:
    """PCA: longest axis → +X (pointy end = nose), shortest → +Z (fins up)."""
    v = np.asarray(verts, np.float64)
    v = v - v.mean(axis=0)
    _, _, vt = np.linalg.svd(v, full_matrices=False)
    v = v @ vt[:3].T
    ext = v.max(axis=0) - v.min(axis=0)
    order = np.argsort(ext)[::-1]
    v = v[:, order]
    det = np.linalg.det(vt[:3][order])
    if det < 0.0:
        v[:, 1] *= -1.0

    xmin, xmax = float(v[:, 0].min()), float(v[:, 0].max())
    span = xmax - xmin
    lo = v[:, 0] < xmin + 0.12 * span
    hi = v[:, 0] > xmax - 0.12 * span

    def slab_wh(mask: np.ndarray) -> float:
        sl = v[mask]
        if sl.shape[0] < 8:
            return 0.0
        return float((sl[:, 1].max() - sl[:, 1].min()) * (sl[:, 2].max() - sl[:, 2].min()))

    if slab_wh(lo) < slab_wh(hi):
        v[:, 0] *= -1.0
        v[:, 1] *= -1.0
    if abs(float(v[:, 2].max())) < abs(float(v[:, 2].min())):
        v[:, 2] *= -1.0
        v[:, 1] *= -1.0
    return v.astype(np.float32)


def make_aircraft_mesh(
    center,
    heading_rad: float = 0.0,
    path: str | None = None,
    length_m: float = SHAHED_LENGTH_M,
    dest_dir: str = SHAHED_DIR,
    name: str = "aircraft",
) -> dict:
    src = find_mesh_file(path, dest_dir=dest_dir)
    verts, idx, alb = load_triangle_mesh(src)
    verts = orient_fuselage_x_up_z(verts)
    span = float(verts[:, 0].max() - verts[:, 0].min())
    if span < 1e-6:
        raise RuntimeError(f"{src} has zero length after orient")
    verts = verts * (float(length_m) / span)
    c, s = float(np.cos(heading_rad)), float(np.sin(heading_rad))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)
    verts = verts @ rot.T
    verts = verts + np.asarray(center, np.float32).reshape(1, 3)
    temp = np.full((idx.shape[0],), 288.15, np.float32)
    mat_id = np.full((idx.shape[0],), MTL_RGB_LAMBERT, np.int32)
    bb0 = verts.min(axis=0)
    bb1 = verts.max(axis=0)
    print(
        f"{name} mesh verts={verts.shape[0]} tris={idx.shape[0]}  "
        f"AABB=[{bb0[0]:.2f},{bb0[1]:.2f},{bb0[2]:.2f}]-"
        f"[{bb1[0]:.2f},{bb1[1]:.2f},{bb1[2]:.2f}]  "
        f"size={(bb1 - bb0).tolist()} m  center={np.asarray(center, float).tolist()}"
    )
    return {
        "verts": np.ascontiguousarray(verts, np.float32),
        "idx": np.ascontiguousarray(idx, np.int32),
        "albedo": np.ascontiguousarray(alb, np.float32),
        "temp": temp,
        "mat_id": mat_id,
        "source": src,
        "length_m": float(length_m),
    }


def make_shahed136_mesh(center, heading_rad: float = 0.0, path: str | None = None) -> dict:
    if path:
        return make_aircraft_mesh(
            center,
            heading_rad=heading_rad,
            path=path,
            length_m=SHAHED_LENGTH_M,
            dest_dir=SHAHED_DIR,
            name="Shahed-136",
        )
    return make_shahed136_coarse_mesh(center, heading_rad=heading_rad)


def write_obj(path: str, verts: np.ndarray, idx: np.ndarray) -> None:
    with open(path, "w", encoding="ascii") as f:
        f.write("# shahed-136 coarse visual mesh, meters, +X nose +Z up\n")
        for v in verts:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in idx:
            f.write(f"f {int(t[0]) + 1} {int(t[1]) + 1} {int(t[2]) + 1}\n")


def _preview_png(path: str, verts: np.ndarray, idx: np.ndarray, alb: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(7.2, 6.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    tris = verts[idx]
    e0 = tris[:, 1] - tris[:, 0]
    e1 = tris[:, 2] - tris[:, 0]
    nrm = np.cross(e0, e1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    light = np.array([0.55, -0.35, 0.76], np.float64)
    light /= np.linalg.norm(light)
    ndl = np.clip(nrm @ light, 0.0, 1.0)
    cols = np.clip(alb * (0.28 + 0.72 * ndl[:, None]), 0.0, 1.0)
    mesh = Poly3DCollection(tris, linewidths=0.04, edgecolors=(0.55, 0.55, 0.55, 0.25))
    mesh.set_facecolor(cols)
    ax.add_collection3d(mesh)
    span = float(np.max(verts.max(0) - verts.min(0)))
    mid = 0.5 * (verts.max(0) + verts.min(0))
    r = 0.68 * span
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=38)
    ax.set_proj_type("ortho")
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Build / import Shahed-136 mesh")
    p.add_argument("--mesh", default=None, help="CGTrader OBJ/glTF/zip; omit = procedural coarse")
    p.add_argument("--length", type=float, default=SHAHED_LENGTH_M)
    args = p.parse_args()
    os.makedirs(SHAHED_DIR, exist_ok=True)
    if args.mesh:
        mesh = make_aircraft_mesh(
            np.zeros(3, np.float32),
            path=args.mesh,
            length_m=float(args.length),
            name="Shahed-136",
        )
    else:
        mesh = make_shahed136_coarse_mesh(np.zeros(3, np.float32))
    cache = os.path.join(SHAHED_DIR, "shahed136_mesh.npz")
    np.savez(
        cache,
        verts=mesh["verts"],
        idx=mesh["idx"],
        albedo=mesh["albedo"],
        temp=mesh["temp"],
        mat_id=mesh["mat_id"],
    )
    obj_path = os.path.join(SHAHED_DIR, "shahed136_coarse.obj")
    write_obj(obj_path, mesh["verts"], mesh["idx"])
    png = os.path.join(SHAHED_DIR, "shahed136_coarse_preview.png")
    _preview_png(png, mesh["verts"], mesh["idx"], mesh["albedo"])
    print(f"wrote {cache}")
    print(f"wrote {obj_path}")
    print(f"wrote {png}")
    print(f"source {mesh['source']}")


if __name__ == "__main__":
    main()
