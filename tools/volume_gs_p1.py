#!/usr/bin/env python3
"""P1 host-side VOLUME GS: unit probes for σ(x) and brick majorants.

No CUDA / atmosphere kernel changes. Conventions match
docs/plan_gs_replace_all_primitives.md (O1–O3).

Usage:
  python tools/volume_gs_p1.py unit
  python tools/volume_gs_p1.py fire [--voxels PATH] [--eps KE] [--max-particles N]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Repo root on path when run as tools/volume_gs_p1.py
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "tools"))

METALLIC_VOLUME = -1.0
REL_EPS = 1.0e-4
PROBE_EPS = 1.0e-12


def gaussian_G(x: np.ndarray, c: np.ndarray, scale: np.ndarray) -> float:
    """Isotropic/anisotropic axis-aligned Gaussian, G(c)=1.
    scale > 0 is per-axis std (r_i in plan's isotropic form when equal).
    """
    s = np.maximum(np.asarray(scale, dtype=np.float64), 1.0e-12)
    d = (np.asarray(x, dtype=np.float64) - np.asarray(c, dtype=np.float64)) / s
    return float(np.exp(-0.5 * np.dot(d, d)))


def sigma_at(x: np.ndarray, centers: np.ndarray, scales: np.ndarray, sigma_i: np.ndarray) -> float:
    """Brute-force σ(x) = sum_i σ_i G_i(x) (vectorized)."""
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    scales = np.maximum(np.asarray(scales, dtype=np.float64), 1.0e-12)
    sigma_i = np.asarray(sigma_i, dtype=np.float64)
    d = (x - centers) / scales
    g = np.exp(-0.5 * np.sum(d * d, axis=1))
    return float(np.sum(sigma_i * g))


def max_G_on_aabb(c: np.ndarray, scale: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    """Max of axis-aligned Gaussian on closed AABB: at closest point in box to c."""
    p = np.minimum(np.maximum(c, bmin), bmax)
    return gaussian_G(p, c, scale)


def build_majorants(
    centers: np.ndarray,
    scales: np.ndarray,
    sigma_i: np.ndarray,
    origin: np.ndarray,
    extent: np.ndarray,
    nb: tuple[int, int, int],
) -> np.ndarray:
    """σ̄_b = sum_i σ_i max_{x in b} G_i(x) over ALL particles (no reach cull).

    Reach culling underestimates the bound (tails still >0) and fails A1.3.
    """
    nbx, nby, nbz = nb
    origin = np.asarray(origin, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    scales = np.maximum(np.asarray(scales, dtype=np.float64), 1.0e-12)
    sigma_i = np.asarray(sigma_i, dtype=np.float64)
    db = extent / np.array([nbx, nby, nbz], dtype=np.float64)
    out = np.zeros((nbz, nby, nbx), dtype=np.float64)
    for iz in range(nbz):
        for iy in range(nby):
            for ix in range(nbx):
                bmin = origin + db * np.array([ix, iy, iz], dtype=np.float64)
                bmax = bmin + db
                p = np.minimum(np.maximum(centers, bmin), bmax)
                d = (p - centers) / scales
                gmax = np.exp(-0.5 * np.sum(d * d, axis=1))
                out[iz, iy, ix] = float(np.sum(sigma_i * gmax))
    return out


def sigma_at_near(
    x: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    sigma_i: np.ndarray,
    reach_sigma: float = 6.0,
) -> float:
    """Accelerated σ(x): only kernels whose reach AABB contains x.

    Default reach=6σ: G outside ≈ exp(-18) ~ 1.5e-8, enough for A1.2 REL_EPS.
    """
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    scales = np.maximum(np.asarray(scales, dtype=np.float64), 1.0e-12)
    sigma_i = np.asarray(sigma_i, dtype=np.float64)
    r = reach_sigma * scales
    mask = np.all(x >= centers - r, axis=1) & np.all(x <= centers + r, axis=1)
    if not np.any(mask):
        return 0.0
    c = centers[mask]
    s = scales[mask]
    d = (x - c) / s
    g = np.exp(-0.5 * np.sum(d * d, axis=1))
    return float(np.sum(sigma_i[mask] * g))


def validate_sigma_query(
    centers: np.ndarray,
    scales: np.ndarray,
    sigma_i: np.ndarray,
    probes: np.ndarray,
) -> dict:
    """A1.2: neighbor query vs full brute sum."""
    rels = []
    abs_err = 0.0
    for x in probes:
        s = sigma_at(x, centers, scales, sigma_i)
        s_q = sigma_at_near(x, centers, scales, sigma_i)
        err = abs(s_q - s)
        abs_err = max(abs_err, err)
        if s > 1.0e-6:
            rels.append(err / s)
        else:
            # Near-empty probes: absolute gate
            rels.append(0.0 if err <= REL_EPS * 1.0e-6 else 1.0)
    rels = np.asarray(rels, dtype=np.float64)
    return {
        "n": int(probes.shape[0]),
        "max_rel": float(rels.max()) if rels.size else 0.0,
        "max_abs": float(abs_err),
        "pass": bool(rels.size == 0 or rels.max() <= REL_EPS),
    }


def validate_majorants(
    centers: np.ndarray,
    scales: np.ndarray,
    sigma_i: np.ndarray,
    origin: np.ndarray,
    extent: np.ndarray,
    maj: np.ndarray,
    n_probes: int,
    rng: np.random.Generator,
) -> dict:
    """A1.3 validity + A1.4 tightness report."""
    nbz, nby, nbx = maj.shape
    origin = np.asarray(origin, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    db = extent / np.array([nbx, nby, nbz], dtype=np.float64)
    violations = 0
    max_viol_rel = 0.0
    ratios = []
    # Per-brick sample max σ estimate for tightness
    brick_max_est = np.zeros_like(maj)
    samples_per_brick = max(4, n_probes // max(maj.size, 1))
    for iz in range(nbz):
        for iy in range(nby):
            for ix in range(nbx):
                bmin = origin + db * np.array([ix, iy, iz], dtype=np.float64)
                bmax = bmin + db
                bar = float(maj[iz, iy, ix])
                local_max = 0.0
                for _ in range(samples_per_brick):
                    x = rng.uniform(bmin, bmax)
                    s = sigma_at(x, centers, scales, sigma_i)
                    local_max = max(local_max, s)
                    if s > bar * (1.0 + REL_EPS):
                        violations += 1
                        max_viol_rel = max(max_viol_rel, s / max(bar, PROBE_EPS) - 1.0)
                brick_max_est[iz, iy, ix] = local_max
                if local_max > PROBE_EPS:
                    ratios.append(bar / local_max)
    ratios = np.asarray(ratios, dtype=np.float64) if ratios else np.array([np.nan])
    return {
        "violations": int(violations),
        "max_viol_rel": float(max_viol_rel),
        "pass_validity": violations == 0,
        "tightness_median": float(np.nanmedian(ratios)),
        "tightness_p95": float(np.nanpercentile(ratios, 95)) if ratios.size and np.isfinite(ratios).any() else float("nan"),
        "n_ratio_bricks": int(np.isfinite(ratios).sum()),
    }


def make_unit_cloud(rng: np.random.Generator) -> dict:
    """Small synthetic VOLUME set (A1 unit probes)."""
    n = 8
    centers = rng.uniform(0.0, 10.0, size=(n, 3))
    scales = np.full((n, 3), 1.5, dtype=np.float64)
    ka = rng.uniform(0.01, 0.2, size=n)
    ks = rng.uniform(0.05, 1.0, size=n)
    sigma_i = ka + ks
    T = np.full(n, 280.0)
    g = np.full(n, 0.85)
    metallic = np.full(n, METALLIC_VOLUME)
    return {
        "centers": centers,
        "scales": scales,
        "sigma_i": sigma_i,
        "ka": ka,
        "ks": ks,
        "T": T,
        "g": g,
        "metallic": metallic,
        "origin": np.array([0.0, 0.0, 0.0]),
        "extent": np.array([10.0, 10.0, 10.0]),
        "nb": (4, 4, 4),
    }


def assert_volume_convention(cloud: dict) -> None:
    m = cloud["metallic"]
    if not np.all(m < 0.0):
        raise AssertionError("VOLUME metallic must be < 0")
    g = cloud["g"]
    if not np.all(np.abs(g) < 1.0):
        raise AssertionError("VOLUME roughness/g must satisfy |g| < 1")
    ke = cloud["ka"] + cloud["ks"]
    if not np.allclose(cloud["sigma_i"], ke, rtol=1e-5, atol=1e-8):
        raise AssertionError("density/σ_i must equal ka+ks")


def cmd_unit(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    cloud = make_unit_cloud(rng)
    assert_volume_convention(cloud)
    probes = rng.uniform(cloud["origin"], cloud["origin"] + cloud["extent"], size=(args.n_probes, 3))
    r_sigma = validate_sigma_query(cloud["centers"], cloud["scales"], cloud["sigma_i"], probes)
    maj = build_majorants(
        cloud["centers"],
        cloud["scales"],
        cloud["sigma_i"],
        cloud["origin"],
        cloud["extent"],
        cloud["nb"],
    )
    r_maj = validate_majorants(
        cloud["centers"],
        cloud["scales"],
        cloud["sigma_i"],
        cloud["origin"],
        cloud["extent"],
        maj,
        n_probes=args.n_probes,
        rng=rng,
    )
    print("=== P1 unit probes ===")
    print(f"particles={cloud['centers'].shape[0]} probes={args.n_probes}")
    print(f"A1.2 sigma max_rel={r_sigma['max_rel']:.3e} pass={r_sigma['pass']}")
    print(
        f"A1.3 majorant violations={r_maj['violations']} "
        f"max_viol_rel={r_maj['max_viol_rel']:.3e} pass={r_maj['pass_validity']}"
    )
    print(
        f"A1.4 tightness median={r_maj['tightness_median']:.4g} "
        f"p95={r_maj['tightness_p95']:.4g} (report only)"
    )
    ok = r_sigma["pass"] and r_maj["pass_validity"]
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def pin_from_voxels(
    ka: np.ndarray,
    ks: np.ndarray,
    temp: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    ke_eps: float,
    max_particles: int | None,
    g_default: float,
) -> dict:
    """O3 fire pin: one VOLUME GS per active voxel (plan §3)."""
    ka = np.asarray(ka, dtype=np.float64)
    ks = np.asarray(ks, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    nz, ny, nx = ka.shape
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    ke = ka + ks
    iz, iy, ix = np.where(ke > ke_eps)
    n = int(iz.size)
    if max_particles is not None and n > max_particles:
        # Deterministic subsample for smoke tests
        step = int(np.ceil(n / max_particles))
        iz, iy, ix = iz[::step], iy[::step], ix[::step]
        n = int(iz.size)
    centers = np.stack(
        [
            origin[0] + (ix + 0.5) * spacing[0],
            origin[1] + (iy + 0.5) * spacing[1],
            origin[2] + (iz + 0.5) * spacing[2],
        ],
        axis=1,
    )
    # scale ≈ cell size / sqrt(2) so half-cell ~ e^{-0.25} falloff; documented choice
    scales = np.repeat((spacing / np.sqrt(2.0))[None, :], n, axis=0)
    ka_i = ka[iz, iy, ix]
    ks_i = ks[iz, iy, ix]
    T_i = temp[iz, iy, ix]
    sigma_i = ka_i + ks_i
    # Domain for majorants: particle AABB + margin (critical when max_particles subsamples)
    margin = 6.0 * (spacing / np.sqrt(2.0))
    pmin = centers.min(axis=0) - margin
    pmax = centers.max(axis=0) + margin
    origin_m = pmin
    extent = np.maximum(pmax - pmin, spacing)
    # Brick count: ~8 cell sizes in world units
    brick_m = spacing * 8.0
    nb = tuple(int(max(1, np.ceil(extent[i] / max(brick_m[i], 1e-6)))) for i in range(3))
    # Cap brick count for host smoke runtime
    nb = tuple(min(int(v), 16) for v in nb)
    return {
        "centers": centers,
        "scales": scales,
        "sigma_i": sigma_i,
        "ka": ka_i,
        "ks": ks_i,
        "T": T_i,
        "g": np.full(n, g_default, dtype=np.float64),
        "metallic": np.full(n, METALLIC_VOLUME, dtype=np.float64),
        "origin": origin_m,
        "extent": extent,
        "nb": nb,
        "n_active_voxels": int(np.count_nonzero(ke > ke_eps)),
        "n_particles": n,
        "spacing": spacing.copy(),
        "voxel_origin": origin.copy(),
    }


def overlap_bias_report(cloud: dict, ka: np.ndarray, ks: np.ndarray, origin: np.ndarray, spacing: np.ndarray, rng: np.random.Generator, n: int = 500) -> dict:
    """Report σ_GS vs voxel ke at random active-ish probes (honest pin bias)."""
    ke = np.asarray(ka, dtype=np.float64) + np.asarray(ks, dtype=np.float64)
    nz, ny, nx = ke.shape
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    ratios = []
    for _ in range(n):
        ix = int(rng.integers(0, nx))
        iy = int(rng.integers(0, ny))
        iz = int(rng.integers(0, nz))
        x = origin + (np.array([ix, iy, iz], dtype=np.float64) + 0.5) * spacing
        kv = float(ke[iz, iy, ix])
        if kv <= PROBE_EPS:
            continue
        sg = sigma_at(x, cloud["centers"], cloud["scales"], cloud["sigma_i"])
        ratios.append(sg / kv)
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.size == 0:
        return {"n": 0}
    return {
        "n": int(ratios.size),
        "median_sigma_over_voxel_ke": float(np.median(ratios)),
        "p95_sigma_over_voxel_ke": float(np.percentile(ratios, 95)),
        "mean_sigma_over_voxel_ke": float(np.mean(ratios)),
    }


def cmd_fire(args: argparse.Namespace) -> int:
    import mountain_gs_scene as mgs

    voxels_path = args.voxels
    if not os.path.isfile(voxels_path):
        print(f"missing voxels: {voxels_path}", file=sys.stderr)
        return 2
    vox = mgs.load_voxel_bin(voxels_path)
    ka, ks, temp = vox["ka"], vox["ks"], vox["temp"]
    origin = np.asarray(vox["origin"], dtype=np.float64)
    spacing = np.asarray(vox["spacing"], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    cloud = pin_from_voxels(
        ka,
        ks,
        temp,
        origin,
        spacing,
        ke_eps=args.ke_eps,
        max_particles=args.max_particles,
        g_default=args.g,
    )
    assert_volume_convention(cloud)
    print("=== P1 fire pin ===")
    print(f"voxels={voxels_path}")
    print(
        f"grid={ka.shape[::-1]} (nx,ny,nz) active_ke>{args.ke_eps}: "
        f"{cloud['n_active_voxels']} particles={cloud['n_particles']} nb={cloud['nb']}"
    )
    probes = rng.uniform(cloud["origin"], cloud["origin"] + cloud["extent"], size=(args.n_probes, 3))
    r_sigma = validate_sigma_query(cloud["centers"], cloud["scales"], cloud["sigma_i"], probes)
    print(f"A1.2 sigma max_rel={r_sigma['max_rel']:.3e} pass={r_sigma['pass']}")

    # Majorant on subsampled particle set can be expensive; cap brick count for smoke
    maj = build_majorants(
        cloud["centers"],
        cloud["scales"],
        cloud["sigma_i"],
        cloud["origin"],
        cloud["extent"],
        cloud["nb"],
    )
    r_maj = validate_majorants(
        cloud["centers"],
        cloud["scales"],
        cloud["sigma_i"],
        cloud["origin"],
        cloud["extent"],
        maj,
        n_probes=min(args.n_probes, 2000),
        rng=rng,
    )
    print(
        f"A1.3 majorant violations={r_maj['violations']} "
        f"max_viol_rel={r_maj['max_viol_rel']:.3e} pass={r_maj['pass_validity']}"
    )
    print(
        f"A1.4 tightness median={r_maj['tightness_median']:.4g} "
        f"p95={r_maj['tightness_p95']:.4g} (report only)"
    )
    bias = overlap_bias_report(cloud, ka, ks, origin, spacing, rng, n=args.n_probes)
    print(f"A1.7 overlap bias vs voxel ke: {bias}")
    ok = r_sigma["pass"] and r_maj["pass_validity"]
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("unit", help="Synthetic unit probes (A1.2–A1.4 gate)")
    u.add_argument("--n-probes", type=int, default=1000)
    u.add_argument("--seed", type=int, default=0)
    u.set_defaults(func=cmd_unit)

    f = sub.add_parser("fire", help="Pin-extract from fire voxels (O3)")
    f.add_argument(
        "--voxels",
        default=os.path.join(_ROOT, "scenes/fire_gs/fire_voxels.bin"),
        help="Path to fire_voxels.bin",
    )
    f.add_argument("--ke-eps", type=float, default=1.0e-6)
    f.add_argument("--max-particles", type=int, default=800, help="Cap particles for smoke runtime")
    f.add_argument("--n-probes", type=int, default=500)
    f.add_argument("--g", type=float, default=0.85, help="Pin default HG g → roughness")
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_fire)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
