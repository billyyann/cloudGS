#!/usr/bin/env python3
"""Unified SWIR stare / gate scene manifests (one CSV = one case).

Columns cover imaging + target + scene. Program reads these; do not split into
imaging/target/scene again.

  clouds=0/1          → kernel no_clouds=(clouds==0)
  has_aircraft=0/1    → mesh water-only vs water+An-124
  cloud_mode          → none | htcp | slab  (voxels source when clouds=1)
"""
from __future__ import annotations

import csv
import os
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_DIR = os.path.join(ROOT, "scenes", "fire_gs", "swir_video5s")
MANIFEST1 = os.path.join(MANIFEST_DIR, "manifest1.csv")
MANIFEST2 = os.path.join(MANIFEST_DIR, "manifest2.csv")

# Column order for all manifests (stable schema).
FIELDS = [
    "case",
    "clouds",
    "has_aircraft",
    "cloud_mode",
    "band",
    "lambda_nm",
    "nx",
    "ny",
    "ss",
    "H_m",
    "film_w_m",
    "film_h_m",
    "gsd_m",
    "gsd_optics_m",
    "f_m",
    "pitch_m",
    "fnum",
    "fps",
    "netd_k",
    "spp",
    "seed",
    "sun_az_deg",
    "sun_el_deg",
    "place_x",
    "place_y",
    "ac_z_m",
    "heading_rad",
    "v_mps",
    "T_s",
    "ac_type",
    "rho_body",
    "rho_wing",
    "rho_engine",
    "T_body_K",
    "T_engine_K",
    "water_T_K",
    "water_rho",
    "water_size_m",
    "htcp_path",
    "sky_pad",
    "downsample_csv",
    "mtf_csv",
    "mtf_fwhm_det_px",
    "note",
]

_INT = {
    "clouds",
    "has_aircraft",
    "nx",
    "ny",
    "ss",
    "spp",
    "seed",
    "sky_pad",
}
_FLOAT = {
    "lambda_nm",
    "H_m",
    "film_w_m",
    "film_h_m",
    "gsd_m",
    "gsd_optics_m",
    "f_m",
    "pitch_m",
    "fnum",
    "fps",
    "netd_k",
    "sun_az_deg",
    "sun_el_deg",
    "place_x",
    "place_y",
    "ac_z_m",
    "heading_rad",
    "v_mps",
    "T_s",
    "rho_body",
    "rho_wing",
    "rho_engine",
    "T_body_K",
    "T_engine_K",
    "water_T_K",
    "water_rho",
    "water_size_m",
    "mtf_fwhm_det_px",
}


def _coerce(key: str, val: str) -> Any:
    s = (val if val is not None else "").strip()
    if s == "":
        return None
    if key in _INT:
        return int(float(s))
    if key in _FLOAT:
        return float(s)
    return s


def load_manifest(path: str) -> dict:
    """Load first data row of a manifest CSV into a typed dict."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    raw = rows[0]
    out: dict[str, Any] = {"_path": path}
    for k in FIELDS:
        if k in raw:
            out[k] = _coerce(k, raw[k])
        else:
            out[k] = None
    # require switches
    if out.get("clouds") is None or out.get("has_aircraft") is None:
        raise RuntimeError(f"manifest needs clouds and has_aircraft: {path}")
    if out.get("case") is None:
        raise RuntimeError(f"manifest needs case: {path}")
    cm = out.get("cloud_mode") or ("none" if int(out["clouds"]) == 0 else "htcp")
    out["cloud_mode"] = cm
    if int(out["clouds"]) == 0:
        out["cloud_mode"] = "none"
    return out


def write_manifest(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow({k: row.get(k, "") if row.get(k) is not None else "" for k in FIELDS})


def resolve_path(p: str | None, default: str) -> str:
    if not p:
        return os.path.abspath(default)
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(ROOT, p))
