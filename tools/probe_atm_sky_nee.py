#!/usr/bin/env python3
"""Dump gas layers and run host analog NEE probe (same atmosphere.cuh as GPU)."""
from __future__ import annotations

import os
import struct
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GAS = os.path.join(ROOT, "scenes/mountain_gs/ecrad_gas_1d.npz")
VOX = os.path.join(ROOT, "scenes/mountain_gs/dzvar_voxels.bin")
CPP = os.path.join(ROOT, "tools/probe_atm_sky_nee.cpp")
INC = os.path.join(ROOT, "gpu/threedgptir_tracer/include")


def main() -> int:
    gas = np.load(GAS)
    z = np.asarray(gas["z"], dtype=np.float32)
    ka = np.asarray(gas["ka"], dtype=np.float32)
    ks = np.asarray(gas["ks"], dtype=np.float32)
    temp = np.asarray(gas["temp"], dtype=np.float32)
    nlayers = int(ka.size)
    if z.size != nlayers + 1:
        raise SystemExit(f"gas z len {z.size} != nlayers+1 {nlayers + 1}")
    gas_bin = "/tmp/atm_gas_layers.bin"
    with open(gas_bin, "wb") as f:
        f.write(struct.pack("<i", nlayers))
        z.tofile(f)
        ka.tofile(f)
        ks.tofile(f)
        temp.tofile(f)
    exe = "/tmp/probe_atm_sky_nee"
    cmd = ["g++", "-O2", "-std=c++17", f"-I{INC}", CPP, "-o", exe, "-lm"]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    return subprocess.call([exe, VOX, gas_bin])


if __name__ == "__main__":
    sys.exit(main())
