"""CPU checks of voxel Woodcock / 1D gas competing risks (same math as atmosphere.cuh)."""

from __future__ import annotations

import math
import random
import subprocess
import sys
from pathlib import Path

TRACER_ROOT = Path(__file__).resolve().parents[1]
INCLUDE = TRACER_ROOT / "include"
HOST_SRC = Path(__file__).resolve().parent / "test_atm_host.cpp"


def test_build_brick_majorants():
    try:
        import numpy as np
    except ImportError:
        print("skip test_build_brick_majorants (no numpy)")
        return

    sys.path.insert(0, str(TRACER_ROOT))
    from atmosphere import build_brick_majorants

    ka = np.zeros((8, 8, 8), dtype=np.float32)
    ks = np.zeros_like(ka)
    ks[1, 2, 3] = 4.5
    ks[7, 7, 7] = 1.25
    kmax = build_brick_majorants(ka, ks, brick=8)
    assert kmax.shape == (1, 1, 1)
    assert abs(float(kmax[0, 0, 0]) - 4.5) < 1e-5

    kmax4 = build_brick_majorants(ka, ks, brick=4)
    assert kmax4.shape == (2, 2, 2)
    assert abs(float(kmax4[0, 0, 0]) - 4.5) < 1e-5
    assert abs(float(kmax4[1, 1, 1]) - 1.25) < 1e-5


def test_competing_risk_formula():
    """Independent Exp(k) times: P(gas first) = k_gas / (k_gas + k_vox)."""
    rng = random.Random(0)
    k_g, k_v = 0.3, 0.1
    n = 20000
    n_gas = 0
    for _ in range(n):
        t_g = -math.log(max(rng.random(), 1e-12)) / k_g
        t_v = -math.log(max(rng.random(), 1e-12)) / k_v
        if t_g < t_v:
            n_gas += 1
    p = n_gas / n
    assert abs(p - k_g / (k_g + k_v)) < 0.02, p


def test_host_cuh():
    out = Path("/tmp/test_atm_host")
    cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        f"-I{INCLUDE}",
        str(HOST_SRC),
        "-o",
        str(out),
        "-lm",
    ]
    subprocess.check_call(cmd)
    subprocess.check_call([str(out)])


if __name__ == "__main__":
    test_build_brick_majorants()
    test_competing_risk_formula()
    test_host_cuh()
    print("test_atmosphere_unbiased: ok")
