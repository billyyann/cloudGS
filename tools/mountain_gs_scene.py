#!/usr/bin/env python3
"""Mountain GS scene assets in Python (no g++).

Builds:
  - mountain.obj heightfield → 3DGS ply + scene.json
  - DZVAR .htcp → native-layer voxel ka/ks + z_iface (ctypes libhtcp)
  - ecrad_opt_prop.txt → 1D SW gas layers (ctypes libhtgop)

Cloud optical truth stays voxels. GS/mesh are surfaces only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_size_t,
    c_void_p,
    cast,
    cdll,
)
from typing import Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_LIB = os.path.join(ROOT, "local", "lib")
SPK = os.path.join(ROOT, "htrdr-atmosphere-spk", "htrdr-Atmosphere-Starter-Pack-0.8.0")
DEFAULT_OBJ = os.path.join(SPK, "models", "mountain.obj")
DEFAULT_HTCP = os.path.join(SPK, "clouds", "DZVAR.1.ARMCU.008.diaKCL.htcp")
DEFAULT_ECRAD = os.path.join(SPK, "ecrad_opt_prop.txt")
DEFAULT_OUT = os.path.join(ROOT, "scenes", "mountain_gs")
RES_OK = 0
DRY_AIR_MOLAR_MASS = 0.0289644
GAS_CONSTANT = 8.3144598
DEFAULT_MIE = os.path.join(SPK, "Mie_LUT_Cloud-2-10-0.010.nc")


def _preload_local_libs() -> None:
    libdir = LOCAL_LIB
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if libdir not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = libdir + (":" + cur if cur else "")
    # ctypes does not re-read LD_LIBRARY_PATH; load deps by absolute path.
    for name in ("librsys.so", "libhtcp.so", "libhtgop.so", "libhtmie.so"):
        path = os.path.join(libdir, name)
        if os.path.isfile(path):
            cdll.LoadLibrary(path)


def _so(name: str):
    _preload_local_libs()
    path = os.path.join(LOCAL_LIB, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return cdll.LoadLibrary(path)


# --- libhtcp -----------------------------------------------------------------


class HtcpDesc(Structure):
    _fields_ = [
        ("pagesize", c_size_t),
        ("irregular_z", c_int),
        ("spatial_definition", c_size_t * 3),
        ("time_definition", c_size_t),
        ("lower", c_double * 3),
        ("upper", c_double * 3),
        ("vxsz_x", c_double),
        ("vxsz_y", c_double),
        ("vxsz_z", POINTER(c_double)),
        ("coord_z", POINTER(c_double)),
        ("RCT", POINTER(c_double)),
        ("RVT", POINTER(c_double)),
        ("PABST", POINTER(c_double)),
        ("T", POINTER(c_double)),
    ]


def _np_from_c_double(ptr, n: int) -> np.ndarray:
    if not ptr or n <= 0:
        return np.zeros(0, dtype=np.float64)
    addr = cast(ptr, c_void_p).value
    if not addr:
        return np.zeros(0, dtype=np.float64)
    return np.frombuffer((c_double * n).from_address(addr), dtype=np.float64, count=n).copy()


def fetch_mie(wavelength_nm: float = 550.0, mie_path: str = DEFAULT_MIE) -> dict:
    """Mass cross-sections (m^2/kg) and HG g from the starter-pack Mie LUT."""
    lib = _so("libhtmie.so")
    lib.htmie_create.argtypes = [c_void_p, c_void_p, c_int, POINTER(c_void_p)]
    lib.htmie_create.restype = c_int
    lib.htmie_load.argtypes = [c_void_p, c_char_p]
    lib.htmie_load.restype = c_int
    lib.htmie_ref_put.argtypes = [c_void_p]
    lib.htmie_ref_put.restype = c_int
    lib.htmie_fetch_xsection_scattering.argtypes = [c_void_p, c_double, c_int]
    lib.htmie_fetch_xsection_scattering.restype = c_double
    lib.htmie_fetch_xsection_absorption.argtypes = [c_void_p, c_double, c_int]
    lib.htmie_fetch_xsection_absorption.restype = c_double
    lib.htmie_fetch_asymmetry_parameter.argtypes = [c_void_p, c_double, c_int]
    lib.htmie_fetch_asymmetry_parameter.restype = c_double
    h = c_void_p()
    if lib.htmie_create(None, None, 0, byref(h)) != RES_OK:
        raise RuntimeError("htmie_create failed")
    if lib.htmie_load(h, mie_path.encode()) != RES_OK:
        lib.htmie_ref_put(h)
        raise RuntimeError(f"htmie_load failed: {mie_path}")
    w = float(wavelength_nm)
    Cs = float(lib.htmie_fetch_xsection_scattering(h, c_double(w), 0))
    Ca = float(lib.htmie_fetch_xsection_absorption(h, c_double(w), 0))
    g = float(lib.htmie_fetch_asymmetry_parameter(h, c_double(w), 0))
    lib.htmie_ref_put(h)
    print(f"Mie LUT {wavelength_nm:.0f} nm  Cs={Cs:.4g} Ca={Ca:.4g} g={g:.4f}")
    return {"Cs": Cs, "Ca": Ca, "g": g}


def load_htcp_voxels(
    htcp_path: str = DEFAULT_HTCP,
    nreg: int | None = None,
    z0: float | None = None,
    z1: float | None = None,
    wavelength_nm: float = 550.0,
    mie_path: str = DEFAULT_MIE,
) -> dict:
    """Native HTCP layers (nz, ny, nx). k = ρ_da · RCT · C_{s,a} (same as htsky).

    Irregular Z is kept: one cell per HTCP layer, z_iface[nz+1] for GPU query.
    nreg/z0/z1 are ignored (overlap bake onto uniform dz thinned cloud skins).
    """
    mie = fetch_mie(wavelength_nm, mie_path)
    Cs, Ca = mie["Cs"], mie["Ca"]
    lib = _so("libhtcp.so")
    lib.htcp_create.argtypes = [c_void_p, c_void_p, c_int, POINTER(c_void_p)]
    lib.htcp_create.restype = c_int
    lib.htcp_load.argtypes = [c_void_p, c_char_p]
    lib.htcp_load.restype = c_int
    lib.htcp_get_desc.argtypes = [c_void_p, POINTER(HtcpDesc)]
    lib.htcp_get_desc.restype = c_int
    lib.htcp_ref_put.argtypes = [c_void_p]
    lib.htcp_ref_put.restype = c_int

    cloud = c_void_p()
    if lib.htcp_create(None, None, 0, byref(cloud)) != RES_OK:
        raise RuntimeError("htcp_create failed")
    if lib.htcp_load(cloud, htcp_path.encode()) != RES_OK:
        lib.htcp_ref_put(cloud)
        raise RuntimeError(f"htcp_load failed: {htcp_path}")
    d = HtcpDesc()
    if lib.htcp_get_desc(cloud, byref(d)) != RES_OK:
        lib.htcp_ref_put(cloud)
        raise RuntimeError("htcp_get_desc failed")

    nx, ny, nz = (int(d.spatial_definition[0]), int(d.spatial_definition[1]), int(d.spatial_definition[2]))
    if nreg is not None or z0 is not None or z1 is not None:
        print("note: nreg/z0/z1 ignored; dumping native irregular-Z HTCP layers")
    nt = max(int(d.time_definition), 1)
    ncell = nx * ny * nz * nt
    rct = _np_from_c_double(d.RCT, ncell).reshape(nt, nz, ny, nx)
    temp_src = _np_from_c_double(d.T, ncell).reshape(nt, nz, ny, nx) if d.T else None
    pabst = _np_from_c_double(d.PABST, ncell).reshape(nt, nz, ny, nx) if d.PABST else None
    vxsz_z = _np_from_c_double(d.vxsz_z, nz) if d.vxsz_z else np.full(nz, (d.upper[2] - d.lower[2]) / max(nz, 1))
    if d.irregular_z and d.coord_z:
        zlo = _np_from_c_double(d.coord_z, nz)
    else:
        zlo = d.lower[2] + np.arange(nz, dtype=np.float64) * ((d.upper[2] - d.lower[2]) / max(nz, 1))
    zhi = zlo + vxsz_z
    z_iface = np.concatenate([zlo.astype(np.float32), np.float32([zhi[-1]])])

    ks = np.zeros((nz, ny, nx), np.float32)
    ka = np.zeros_like(ks)
    temp = np.full_like(ks, 280.0)
    rct0 = rct[0]
    for iz in range(nz):
        slab = rct0[iz]
        tk = temp_src[0, iz] if temp_src is not None else np.full_like(slab, 280.0)
        if pabst is not None:
            rho = (pabst[0, iz] * DRY_AIR_MOLAR_MASS) / np.maximum(tk * GAS_CONSTANT, 1e-6)
        else:
            rho = np.full_like(slab, 1.0)
        ql = rho * slab
        ks[iz] = (ql * Cs).astype(np.float32)
        ka[iz] = (ql * Ca).astype(np.float32)
        temp[iz] = tk.astype(np.float32)

    z0_out = float(zlo[0])
    z1_out = float(zhi[-1])
    origin = np.array([float(d.lower[0]), float(d.lower[1]), z0_out], np.float32)
    mean_dz = (z1_out - z0_out) / float(max(nz, 1))
    spacing = np.array([float(d.vxsz_x), float(d.vxsz_y), mean_dz], np.float32)
    lib.htcp_ref_put(cloud)
    print(
        f"htcp {nx}x{ny}x{nz} native irregular Z origin={origin} dx={spacing} "
        f"z=[{z0_out:.1f},{z1_out:.1f}] ks>0={(ks > 1e-6).mean():.3f} max_ks={ks.max():.4g} Cs={Cs:.4g}"
    )
    return {
        "ka": ka,
        "ks": ks,
        "temp": temp,
        "origin": origin,
        "spacing": spacing,
        "z_iface": z_iface,
        "mie_g": mie["g"],
    }


def save_voxel_bin(path: str, voxels: dict) -> None:
    ka, ks, temp = voxels["ka"], voxels["ks"], voxels["temp"]
    nz, ny, nx = ks.shape
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    z_iface = np.asarray(voxels["z_iface"], dtype=np.float32).reshape(-1)
    if z_iface.size != nz + 1:
        raise ValueError(f"z_iface length {z_iface.size} != nz+1={nz + 1}")
    with open(path, "wb") as f:
        np.array([nx, ny, nz, 2], dtype=np.int32).tofile(f)
        np.asarray(voxels["origin"], dtype=np.float32).tofile(f)
        np.asarray(voxels["spacing"], dtype=np.float32).tofile(f)
        ks.astype(np.float32).ravel().tofile(f)
        ka.astype(np.float32).ravel().tofile(f)
        temp.astype(np.float32).ravel().tofile(f)
        z_iface.tofile(f)
    print(f"wrote {path} (native z_iface {nz + 1})")


def load_voxel_bin(path: str) -> dict:
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=4)
        nx, ny, nz = int(hdr[0]), int(hdr[1]), int(hdr[2])
        origin = np.fromfile(f, dtype=np.float32, count=3)
        spacing = np.fromfile(f, dtype=np.float32, count=3)
        n = nx * ny * nz
        ks = np.fromfile(f, dtype=np.float32, count=n).reshape(nz, ny, nx)
        ka = np.fromfile(f, dtype=np.float32, count=n).reshape(nz, ny, nx)
        temp = np.fromfile(f, dtype=np.float32, count=n).reshape(nz, ny, nx)
        out = {"ka": ka, "ks": ks, "temp": temp, "origin": origin, "spacing": spacing}
        if int(hdr[3]) >= 2:
            z_iface = np.fromfile(f, dtype=np.float32, count=nz + 1)
            if z_iface.size == nz + 1:
                out["z_iface"] = z_iface
    return out


# --- libhtgop / ecrad 1D gas -------------------------------------------------


class HtgopLevel(Structure):
    _fields_ = [
        ("pressure", c_double),
        ("temperature", c_double),
        ("height", c_double),
    ]


class HtgopLayer(Structure):
    _fields_ = [
        ("x_h2o_nominal", c_double),
        ("x_h2o_tab", POINTER(c_double)),
        ("tab_length", c_size_t),
        ("lw_spectral_intervals_count", c_size_t),
        ("sw_spectral_intervals_count", c_size_t),
        ("htgop", c_void_p),
        ("data__", c_void_p),
    ]


class HtgopSwSpec(Structure):
    _fields_ = [
        ("ka_nominal", POINTER(c_double)),
        ("ks_nominal", POINTER(c_double)),
        ("quadrature_length", c_size_t),
        ("htgop", c_void_p),
        ("data__", c_void_p),
        ("layer__", c_void_p),
    ]


class HtgopSpectralInterval(Structure):
    _fields_ = [
        ("wave_numbers", c_double * 2),
        ("quadrature_pdf", POINTER(c_double)),
        ("quadrature_cdf", POINTER(c_double)),
        ("quadrature_length", c_size_t),
        ("htgop", c_void_p),
    ]


def load_ecrad_gas_1d(
    ecrad_path: str = DEFAULT_ECRAD,
    wavelength_m: float = 5.5e-7,
) -> dict:
    """Piecewise-constant SW ka/ks vs z from HTGOP (ecrad_opt_prop.txt)."""
    lib = _so("libhtgop.so")
    lib.htgop_create.argtypes = [c_void_p, c_void_p, c_int, POINTER(c_void_p)]
    lib.htgop_create.restype = c_int
    lib.htgop_load.argtypes = [c_void_p, c_char_p]
    lib.htgop_load.restype = c_int
    lib.htgop_ref_put.argtypes = [c_void_p]
    lib.htgop_ref_put.restype = c_int
    lib.htgop_get_layers_count.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.htgop_get_layers_count.restype = c_int
    lib.htgop_get_levels_count.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.htgop_get_levels_count.restype = c_int
    lib.htgop_get_level.argtypes = [c_void_p, c_size_t, POINTER(HtgopLevel)]
    lib.htgop_get_level.restype = c_int
    lib.htgop_get_layer.argtypes = [c_void_p, c_size_t, POINTER(HtgopLayer)]
    lib.htgop_get_layer.restype = c_int
    lib.htgop_find_sw_spectral_interval_id.argtypes = [c_void_p, c_double, POINTER(c_size_t)]
    lib.htgop_find_sw_spectral_interval_id.restype = c_int
    lib.htgop_get_sw_spectral_interval.argtypes = [c_void_p, c_size_t, POINTER(HtgopSpectralInterval)]
    lib.htgop_get_sw_spectral_interval.restype = c_int
    lib.htgop_layer_get_sw_spectral_interval.argtypes = [POINTER(HtgopLayer), c_size_t, POINTER(HtgopSwSpec)]
    lib.htgop_layer_get_sw_spectral_interval.restype = c_int

    gop = c_void_p()
    if lib.htgop_create(None, None, 0, byref(gop)) != RES_OK:
        raise RuntimeError("htgop_create failed")
    if lib.htgop_load(gop, ecrad_path.encode()) != RES_OK:
        lib.htgop_ref_put(gop)
        raise RuntimeError(f"htgop_load failed: {ecrad_path}")

    nlayers = c_size_t()
    nlevels = c_size_t()
    lib.htgop_get_layers_count(gop, byref(nlayers))
    lib.htgop_get_levels_count(gop, byref(nlevels))
    nl, nv = int(nlayers.value), int(nlevels.value)

    wn = 1.0e-2 / max(wavelength_m, 1e-12)  # m → cm^-1
    iband = c_size_t()
    if lib.htgop_find_sw_spectral_interval_id(gop, c_double(wn), byref(iband)) != RES_OK:
        lib.htgop_ref_put(gop)
        raise RuntimeError(f"no SW interval for {wavelength_m*1e9:.0f} nm ({wn:.1f} cm^-1)")

    spec = HtgopSpectralInterval()
    lib.htgop_get_sw_spectral_interval(gop, iband.value, byref(spec))
    nq = int(spec.quadrature_length)
    pdf = np.array([spec.quadrature_pdf[i] for i in range(nq)], dtype=np.float64)
    pdf = pdf / max(pdf.sum(), 1e-30)

    z = np.zeros(nv, np.float32)
    tlev = np.zeros(nv, np.float32)
    for i in range(nv):
        lev = HtgopLevel()
        lib.htgop_get_level(gop, i, byref(lev))
        z[i] = float(lev.height)
        tlev[i] = float(lev.temperature)

    ka = np.zeros(nl, np.float32)
    ks = np.zeros(nl, np.float32)
    temp = np.zeros(nl, np.float32)
    for i in range(nl):
        layer = HtgopLayer()
        lib.htgop_get_layer(gop, i, byref(layer))
        sw = HtgopSwSpec()
        lib.htgop_layer_get_sw_spectral_interval(byref(layer), iband.value, byref(sw))
        nq_l = int(sw.quadrature_length)
        w = pdf if nq_l == nq else np.full(nq_l, 1.0 / max(nq_l, 1))
        ka_i = sum(float(sw.ka_nominal[q]) * float(w[q]) for q in range(nq_l))
        ks_i = sum(float(sw.ks_nominal[q]) * float(w[q]) for q in range(nq_l))
        ka[i] = ka_i
        ks[i] = ks_i
        temp[i] = 0.5 * (tlev[i] + tlev[min(i + 1, nv - 1)])

    lib.htgop_ref_put(gop)
    tau_s = float(np.sum(ks * np.diff(z.astype(np.float64))))
    tau_a = float(np.sum(ka * np.diff(z.astype(np.float64))))
    print(
        f"ecrad gas {nl} layers  SW band {int(iband.value)} @ {wavelength_m*1e9:.0f} nm  "
        f"z=[{z[0]:.1f},{z[-1]:.1f}]  τ_s={tau_s:.4f} τ_a={tau_a:.4f}  "
        f"ks0={ks[0]:.4g} ka0={ka[0]:.4g}"
    )
    return {"z": z, "ka": ka, "ks": ks, "temp": temp}


def load_ecrad_sw_visible(
    ecrad_path: str = DEFAULT_ECRAD,
    lambda0_nm: float = 380.0,
    lambda1_nm: float = 780.0,
    mie_path: str = DEFAULT_MIE,
) -> dict:
    """HTGOP SW bands overlapping CIE 380-780 nm, plus htsky Mie band averages.

    Gas k is per quadrature point (htrdr `htsky_spectral_band_sample_quadrature`),
    not the pdf-weighted mean. Averaging ka in 625–778 nm made Tr_mean/Tr_true≈0.75
    and killed the red CIE-X wing (cyan + dark).
    """
    lib = _so("libhtgop.so")
    lib.htgop_create.argtypes = [c_void_p, c_void_p, c_int, POINTER(c_void_p)]
    lib.htgop_create.restype = c_int
    lib.htgop_load.argtypes = [c_void_p, c_char_p]
    lib.htgop_load.restype = c_int
    lib.htgop_ref_put.argtypes = [c_void_p]
    lib.htgop_ref_put.restype = c_int
    lib.htgop_get_layers_count.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.htgop_get_layers_count.restype = c_int
    lib.htgop_get_levels_count.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.htgop_get_levels_count.restype = c_int
    lib.htgop_get_level.argtypes = [c_void_p, c_size_t, POINTER(HtgopLevel)]
    lib.htgop_get_level.restype = c_int
    lib.htgop_get_layer.argtypes = [c_void_p, c_size_t, POINTER(HtgopLayer)]
    lib.htgop_get_layer.restype = c_int
    lib.htgop_get_sw_spectral_intervals_count.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.htgop_get_sw_spectral_intervals_count.restype = c_int
    lib.htgop_get_sw_spectral_interval.argtypes = [c_void_p, c_size_t, POINTER(HtgopSpectralInterval)]
    lib.htgop_get_sw_spectral_interval.restype = c_int
    lib.htgop_layer_get_sw_spectral_interval.argtypes = [POINTER(HtgopLayer), c_size_t, POINTER(HtgopSwSpec)]
    lib.htgop_layer_get_sw_spectral_interval.restype = c_int

    mie = _so("libhtmie.so")
    mie.htmie_create.argtypes = [c_void_p, c_void_p, c_int, POINTER(c_void_p)]
    mie.htmie_create.restype = c_int
    mie.htmie_load.argtypes = [c_void_p, c_char_p]
    mie.htmie_load.restype = c_int
    mie.htmie_ref_put.argtypes = [c_void_p]
    mie.htmie_ref_put.restype = c_int
    mie.htmie_fetch_xsection_scattering.argtypes = [c_void_p, c_double, c_int]
    mie.htmie_fetch_xsection_scattering.restype = c_double
    mie.htmie_fetch_xsection_absorption.argtypes = [c_void_p, c_double, c_int]
    mie.htmie_fetch_xsection_absorption.restype = c_double
    mie.htmie_fetch_asymmetry_parameter.argtypes = [c_void_p, c_double, c_int]
    mie.htmie_fetch_asymmetry_parameter.restype = c_double
    mie.htmie_compute_xsection_scattering_average.argtypes = [c_void_p, POINTER(c_double), c_int]
    mie.htmie_compute_xsection_scattering_average.restype = c_double
    mie.htmie_compute_xsection_absorption_average.argtypes = [c_void_p, POINTER(c_double), c_int]
    mie.htmie_compute_xsection_absorption_average.restype = c_double

    gop = c_void_p()
    if lib.htgop_create(None, None, 0, byref(gop)) != RES_OK:
        raise RuntimeError("htgop_create failed")
    if lib.htgop_load(gop, ecrad_path.encode()) != RES_OK:
        lib.htgop_ref_put(gop)
        raise RuntimeError(f"htgop_load failed: {ecrad_path}")

    hmie = c_void_p()
    if mie.htmie_create(None, None, 0, byref(hmie)) != RES_OK:
        lib.htgop_ref_put(gop)
        raise RuntimeError("htmie_create failed")
    if mie.htmie_load(hmie, mie_path.encode()) != RES_OK:
        mie.htmie_ref_put(hmie)
        lib.htgop_ref_put(gop)
        raise RuntimeError(f"htmie_load failed: {mie_path}")

    nlayers = c_size_t()
    nlevels = c_size_t()
    nint = c_size_t()
    lib.htgop_get_layers_count(gop, byref(nlayers))
    lib.htgop_get_levels_count(gop, byref(nlevels))
    lib.htgop_get_sw_spectral_intervals_count(gop, byref(nint))
    nl, nv, ni = int(nlayers.value), int(nlevels.value), int(nint.value)

    z = np.zeros(nv, np.float32)
    tlev = np.zeros(nv, np.float32)
    for i in range(nv):
        lev = HtgopLevel()
        lib.htgop_get_level(gop, i, byref(lev))
        z[i] = float(lev.height)
        tlev[i] = float(lev.temperature)

    raw = []
    for ib in range(ni):
        spec = HtgopSpectralInterval()
        lib.htgop_get_sw_spectral_interval(gop, ib, byref(spec))
        wn0, wn1 = float(spec.wave_numbers[0]), float(spec.wave_numbers[1])
        if wn0 <= 0.0 or wn1 <= 0.0:
            continue
        lam_hi = 1.0e7 / min(wn0, wn1)
        lam_lo = 1.0e7 / max(wn0, wn1)
        if lam_hi < lambda0_nm or lam_lo > lambda1_nm:
            continue
        nq = int(spec.quadrature_length)
        pdf = np.array([spec.quadrature_pdf[q] for q in range(nq)], dtype=np.float64)
        pdf = pdf / max(pdf.sum(), 1e-30)
        cdf = np.cumsum(pdf).astype(np.float32)
        cdf[-1] = 1.0
        ka_q = np.zeros((nq, nl), np.float32)
        ks_q = np.zeros((nq, nl), np.float32)
        for i in range(nl):
            layer = HtgopLayer()
            lib.htgop_get_layer(gop, i, byref(layer))
            sw = HtgopSwSpec()
            lib.htgop_layer_get_sw_spectral_interval(byref(layer), ib, byref(sw))
            nq_l = int(sw.quadrature_length)
            for q in range(min(nq, nq_l)):
                ka_q[q, i] = float(sw.ka_nominal[q])
                ks_q[q, i] = float(sw.ks_nominal[q])
        band = (c_double * 2)(lam_lo, lam_hi)
        cs_avg = float(mie.htmie_compute_xsection_scattering_average(hmie, band, 0))
        ca_avg = float(mie.htmie_compute_xsection_absorption_average(hmie, band, 0))
        raw.append((lam_lo, lam_hi, ka_q, ks_q, cdf, cs_avg, ca_avg, ib))

    cs_ref = float(mie.htmie_fetch_xsection_scattering(hmie, c_double(550.0), 0))
    ca_ref = float(mie.htmie_fetch_xsection_absorption(hmie, c_double(550.0), 0))
    g0, g1, gn = 380.0, 780.0, 401
    g_lut = np.zeros(gn, np.float32)
    for i in range(gn):
        lam = g0 + (g1 - g0) * i / float(gn - 1)
        g_lut[i] = float(mie.htmie_fetch_asymmetry_parameter(hmie, c_double(lam), 0))
    mie.htmie_ref_put(hmie)
    lib.htgop_ref_put(gop)
    if not raw:
        raise RuntimeError("no HTGOP SW bands in 380-780 nm")

    nquad = np.array([b[2].shape[0] for b in raw], np.int32)
    nqmax = int(nquad.max())
    nb = len(raw)
    lo = np.array([b[0] for b in raw], np.float32)
    hi = np.array([b[1] for b in raw], np.float32)
    ka_quad = np.zeros((nb, nqmax, nl), np.float32)
    ks_quad = np.zeros((nb, nqmax, nl), np.float32)
    cdf = np.ones((nb, nqmax), np.float32)
    ka_mean = np.zeros((nb, nl), np.float32)
    ks_mean = np.zeros((nb, nl), np.float32)
    mie_cs = np.zeros(nb, np.float32)
    mie_ca = np.zeros(nb, np.float32)
    for i, (lam_lo, lam_hi, ka_q, ks_q, cdf_i, cs_avg, ca_avg, _ib) in enumerate(raw):
        nq = ka_q.shape[0]
        ka_quad[i, :nq] = ka_q
        ks_quad[i, :nq] = ks_q
        cdf[i, :nq] = cdf_i
        cdf[i, nq:] = 1.0
        w = np.diff(np.concatenate([[0.0], cdf_i.astype(np.float64)]))
        ka_mean[i] = np.tensordot(w, ka_q.astype(np.float64), axes=(0, 0)).astype(np.float32)
        ks_mean[i] = np.tensordot(w, ks_q.astype(np.float64), axes=(0, 0)).astype(np.float32)
        mie_cs[i] = cs_avg
        mie_ca[i] = ca_avg
        print(
            f"  HTGOP band {lam_lo:.0f}-{lam_hi:.0f} nm  nq={nq}  "
            f"ks0_mean={ks_mean[i, 0]:.4g}  Cs/Cs550={cs_avg / max(cs_ref, 1e-30):.4f}"
        )
    print(
        f"ecrad SW visible {nb} bands  λ=[{lo.min():.0f},{hi.max():.0f}] nm  "
        f"layers={nl}  nquad_max={nqmax}  (per-quad k, not band mean)"
    )
    return {
        "z": z,
        "lambda_lo_nm": lo,
        "lambda_hi_nm": hi,
        "ka": ka_mean,
        "ks": ks_mean,
        "ka_quad": ka_quad,
        "ks_quad": ks_quad,
        "nquad": nquad,
        "quad_cdf": cdf,
        "mie_cs_band": mie_cs,
        "mie_ca_band": mie_ca,
        "mie_cs_ref": np.float32(cs_ref),
        "mie_ca_ref": np.float32(ca_ref),
        "mie_g_nm": g_lut,
        "mie_g_lambda0_nm": np.float32(g0),
        "mie_g_dlambda_nm": np.float32((g1 - g0) / float(gn - 1)),
    }


def load_sand_mrumtl(path: str | None = None) -> dict:
    """Parse htrdr sand.mrumtl lambertian knots (wavelength nm, R)."""
    if path is None:
        path = os.path.join(SPK, "materials", "legacy", "sand.mrumtl")
    lams = []
    rs = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "lambertian":
                lams.append(float(parts[0]))
                rs.append(float(parts[2]))
    if not lams:
        raise RuntimeError(f"no lambertian knots in {path}")
    print(f"sand.mrumtl {len(lams)} knots  λ=[{lams[0]:.0f},{lams[-1]:.0f}] nm  R(300)={rs[0]:.3f} R(600)={rs[1] if len(rs)>1 else rs[0]:.3f}")
    return {
        "lambda_nm": np.asarray(lams, np.float32),
        "R": np.asarray(rs, np.float32),
    }


def save_gas_npz(path: str, gas: dict) -> None:
    np.savez(path, z=gas["z"], ka=gas["ka"], ks=gas["ks"], temp=gas["temp"])
    print(f"wrote {path}")


def load_gas_npz(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in ("z", "ka", "ks", "temp")}


# --- mountain.obj → GS ply ---------------------------------------------------


def load_mountain_grid(path: str) -> np.ndarray:
    """Read `v x y z` until usemtl. Returns (N,N,3) heightfield."""
    pts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                pts.append((float(x), float(y), float(z)))
            elif line.startswith("usemtl"):
                break
    pts = np.asarray(pts, dtype=np.float32)
    n = int(round(math.sqrt(pts.shape[0])))
    if n * n != pts.shape[0]:
        raise RuntimeError(f"vertex count {pts.shape[0]} is not a square grid")
    print(f"loaded {n} x {n} heightfield from {path}")
    return pts.reshape(n, n, 3)


def _quat_from_normal(n: np.ndarray) -> np.ndarray:
    """Rotate +Z onto n. (w,x,y,z). n is (..., 3)."""
    z = n[..., 2]
    ax = -n[..., 1]
    ay = n[..., 0]
    az = np.zeros_like(z)
    q = np.stack([1.0 + z, ax, ay, az], axis=-1).astype(np.float32)
    near = (ax * ax + ay * ay) < 1e-12
    q[near & (z >= 0)] = np.array([1, 0, 0, 0], np.float32)
    q[near & (z < 0)] = np.array([0, 1, 0, 0], np.float32)
    qn = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(qn, 1e-12)


def downsample_to_gs(grid: np.ndarray, step: int = 8) -> dict:
    nsrc = grid.shape[0]
    nd = (nsrc - 1) // step + 1
    ii = np.minimum(np.arange(nd) * step, nsrc - 1)
    jj = np.minimum(np.arange(nd) * step, nsrc - 1)
    pos = grid[np.ix_(jj, ii)].reshape(-1, 3).copy()
    ip = np.minimum(ii + step, nsrc - 1)
    im = np.maximum(ii - step, 0)
    jp = np.minimum(jj + step, nsrc - 1)
    jm = np.maximum(jj - step, 0)
    # n = (p[j,ip]-p[j,im]) × (p[jp,i]-p[jm,i]) at each sample
    tng = grid[np.ix_(jj, ip)] - grid[np.ix_(jj, im)]  # (nd, nd, 3)
    btg = grid[np.ix_(jp, ii)] - grid[np.ix_(jm, ii)]
    n = np.cross(tng, btg)
    n[n[..., 2] < 0] *= -1.0
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(ln, 1e-12)
    n = n.reshape(-1, 3)
    slope = np.degrees(np.arccos(np.clip(n[:, 2], 0.0, 1.0)))
    s = np.clip(slope / 55.0, 0.0, 1.0)
    alb = np.stack(
        [0.22 * (1.0 - 0.22 * s), 0.16 * (1.0 - 0.18 * s), 0.10 * (1.0 - 0.12 * s)],
        axis=-1,
    ).astype(np.float32)
    dx = 10000.0 / (nd - 1)
    scale = np.full((pos.shape[0], 3), [dx * 0.55, dx * 0.55, dx * 0.18], np.float32)
    quat = _quat_from_normal(n.reshape(nd, nd, 3)).reshape(-1, 4)
    print(f"3DGS particles: {pos.shape[0]} (grid {nd} x {nd}, spacing {dx:.1f} m)")
    return {
        "pos": pos.astype(np.float32),
        "n": n.astype(np.float32),
        "scale": scale,
        "quat": quat,
        "albedo": alb,
        "roughness": np.ones((pos.shape[0],), np.float32),
        "metallic": np.zeros((pos.shape[0],), np.float32),
        "temperature": np.full((pos.shape[0],), 300.0, np.float32),
        "opacity": np.full((pos.shape[0],), 0.99, np.float32),
    }


def heightfield_to_mesh(grid: np.ndarray) -> dict:
    """Regular grid → triangle soup matching atm_mesh_grid_n (iy * n + ix)."""
    n = int(grid.shape[0])
    verts = np.ascontiguousarray(grid.reshape(-1, 3), dtype=np.float32)
    n1 = n - 1
    iy = np.arange(n1, dtype=np.int32)[:, None]
    ix = np.arange(n1, dtype=np.int32)[None, :]
    i00 = iy * n + ix
    i10 = i00 + 1
    i01 = i00 + n
    i11 = i01 + 1
    t0 = np.stack([i00, i10, i11], axis=-1)
    t1 = np.stack([i00, i11, i01], axis=-1)
    idx = np.stack([t0, t1], axis=2).reshape(-1, 3).astype(np.int32)
    v0 = verts[idx[:, 0]]
    e1 = verts[idx[:, 1]] - v0
    e2 = verts[idx[:, 2]] - v0
    nn = np.cross(e1, e2)
    nn[nn[:, 2] < 0] *= -1.0
    ln = np.linalg.norm(nn, axis=1, keepdims=True)
    nn = nn / np.maximum(ln, 1e-12)
    slope = np.degrees(np.arccos(np.clip(nn[:, 2], 0.0, 1.0)))
    s = np.clip(slope / 55.0, 0.0, 1.0)
    alb = np.stack(
        [0.22 * (1.0 - 0.22 * s), 0.16 * (1.0 - 0.18 * s), 0.10 * (1.0 - 0.12 * s)],
        axis=-1,
    ).astype(np.float32)
    temp = np.full((idx.shape[0],), 300.0, np.float32)
    print(f"mountain mesh {n}x{n} verts={verts.shape[0]} tris={idx.shape[0]} z=[{verts[:,2].min():.1f},{verts[:,2].max():.1f}]")
    return {"verts": verts, "idx": idx, "albedo": alb, "temp": temp}


def ensure_mountain_mesh(outdir: str = DEFAULT_OUT, obj_path: str = DEFAULT_OBJ, rebuild: bool = False) -> dict:
    cache = os.path.join(outdir, "mountain_mesh.npz")
    if (not rebuild) and os.path.isfile(cache):
        data = np.load(cache)
        mesh = {k: data[k] for k in ("verts", "idx", "albedo", "temp")}
        print(f"mountain mesh from cache {cache}  verts={mesh['verts'].shape[0]} tris={mesh['idx'].shape[0]}")
        return mesh
    grid = load_mountain_grid(obj_path)
    mesh = heightfield_to_mesh(grid)
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(cache, **mesh)
    print(f"wrote {cache}")
    return mesh


def write_ply(path: str, gs: dict) -> None:
    n = gs["pos"].shape[0]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\ncomment cyclic GS mountain from mountain.obj usemtl air:sand\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        f.write("property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
        f.write("property float albedo_r\nproperty float albedo_g\nproperty float albedo_b\n")
        f.write("property float roughness\nproperty float metallic\nproperty float temperature\n")
        f.write("property float opacity\nend_header\n")
        for i in range(n):
            p, nn, s, q, a = gs["pos"][i], gs["n"][i], gs["scale"][i], gs["quat"][i], gs["albedo"][i]
            f.write(
                f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {nn[0]:.5f} {nn[1]:.5f} {nn[2]:.5f} "
                f"{s[0]:.4f} {s[1]:.4f} {s[2]:.4f} {q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f} "
                f"{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} {gs['roughness'][i]:.4f} {gs['metallic'][i]:.4f} "
                f"{gs['temperature'][i]:.2f} {gs['opacity'][i]:.4f}\n"
            )
    print(f"wrote {path}")


def write_scene_json(path: str, ply_name: str, ngs: int) -> None:
    data = {
        "name": "mountain_gs",
        "match": "mountain.sh: cyclic ground -R, cyclic clouds -r, sand@300K, same camera/sun",
        "gaussians": {"path": ply_name, "count": ngs},
        "camera": {"pos": [10245.37, 615.02, 422.34], "tgt": [-2977.39, 6829.40, 842.40], "fov": 30},
        "sun": {"azimuth_deg": 225, "elevation_deg": 30},
        "volume": {"path": "dzvar_voxels.bin", "primitive": "voxel"},
        "gas": {"path": "ecrad_gas_1d.npz", "source": "ecrad_opt_prop.txt"},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")


def ensure_assets(
    outdir: str = DEFAULT_OUT,
    rebuild: bool = False,
    step: int = 8,
    obj_path: str = DEFAULT_OBJ,
    htcp_path: str = DEFAULT_HTCP,
    ecrad_path: str = DEFAULT_ECRAD,
) -> dict:
    """Create ply / voxels / 1D gas if missing. Returns paths."""
    os.makedirs(outdir, exist_ok=True)
    ply = os.path.join(outdir, "mountain_gs.ply")
    voxels_bin = os.path.join(outdir, "dzvar_voxels.bin")
    gas_npz = os.path.join(outdir, "ecrad_gas_1d.npz")
    scene = os.path.join(outdir, "scene.json")

    if rebuild or not os.path.isfile(ply):
        grid = load_mountain_grid(obj_path)
        gs = downsample_to_gs(grid, step=step)
        write_ply(ply, gs)
        write_scene_json(scene, "mountain_gs.ply", gs["pos"].shape[0])
    elif not os.path.isfile(scene):
        nvert = 0
        with open(ply) as f:
            for line in f:
                if line.startswith("element vertex"):
                    nvert = int(line.split()[-1])
                    break
        write_scene_json(scene, "mountain_gs.ply", nvert)

    need_vox = (
        rebuild
        or not os.path.isfile(voxels_bin)
        or not os.path.isfile(os.path.join(outdir, "dzvar_voxels.mie.json"))
    )
    if not need_vox:
        voxels = load_voxel_bin(voxels_bin)
        # overlap bake (hdr ver<2) smeared native skins; rebuild native z_iface.
        if voxels.get("z_iface") is None or int(voxels["ks"].shape[0]) < 160:
            need_vox = True
    if need_vox:
        voxels = load_htcp_voxels(htcp_path)
        save_voxel_bin(voxels_bin, voxels)
        with open(os.path.join(outdir, "dzvar_voxels.mie.json"), "w") as f:
            json.dump({"Cs_source": "htmie", "note": "k=rho*RCT*C"}, f)

    if rebuild or not os.path.isfile(gas_npz):
        gas = load_ecrad_gas_1d(ecrad_path)
        save_gas_npz(gas_npz, gas)
    else:
        gas = load_gas_npz(gas_npz)
        print(f"gas from cache {gas_npz}  nlayers={gas['ka'].shape[0]} z=[{gas['z'][0]:.1f},{gas['z'][-1]:.1f}]")

    return {"ply": ply, "voxels_bin": voxels_bin, "gas_npz": gas_npz, "scene": scene, "voxels": voxels, "gas": gas}


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Build mountain_gs scene assets (Python, no compile)")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--obj", default=DEFAULT_OBJ)
    p.add_argument("--htcp", default=DEFAULT_HTCP)
    p.add_argument("--ecrad", default=DEFAULT_ECRAD)
    p.add_argument("--step", type=int, default=8)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args(argv)
    ensure_assets(
        outdir=args.outdir,
        rebuild=args.rebuild,
        step=args.step,
        obj_path=args.obj,
        htcp_path=args.htcp,
        ecrad_path=args.ecrad,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
