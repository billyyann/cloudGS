#!/usr/bin/env python3
"""Convert htrdr-atmosphere longwave radiance maps to payload DN.

Radiometric chain (extended source / calRad), no full-well truncation:

  F = f / d
  E = pi * L * t0 * ta * EE / (4 F^2)     # L: band-integrated [W/sr/m^2]
  P = E * dx^2
  N = P * IT * eta * lambda_hc + n_dark * dark_current_const * IT
  DN = N / FW * DNR + BC * DNR            # NOT clipped by FW

Outputs:
  *.dn.f32     float32 DN, row-major (exact linear DN; no FW truncation;
               may exceed DNR-1 and uint16)
  *_dn_16.png  16-bit grayscale PNG:
               - if all DN in [0, 65535]: store DN directly
               - else: min-max stretch DN -> [0, 65535] for display only
                 (still no FW truncation; exact values remain in .dn.f32)
"""

from __future__ import annotations

import argparse
import configparser
import math
import struct
import sys
from pathlib import Path


def load_ini(path: Path, section: str = "MIDDLELONGWAVE") -> dict:
    cp = configparser.ConfigParser()
    cp.optionxform = str
    text = path.read_text(encoding="utf-8")
    # Strip comment-only preamble until first [section]
    if f"[{section}]" not in text and not any(
        line.startswith("[") for line in text.splitlines()
    ):
        text = f"[{section}]\n" + text
    cp.read_string(text)
    if section not in cp:
        raise KeyError(f"section [{section}] not in {path}")
    raw = dict(cp[section])
    keys = (
        "t0",
        "ta",
        "EE",
        "d",
        "dx",
        "eta",
        "lambda_hc",
        "dark_current_const",
        "FW",
        "DNR",
        "BC",
        "n_dark",
        "f",
        "IT",
    )
    cfg = {k: float(raw[k]) for k in keys}
    cfg["lambdaC"] = float(raw.get("lambdaC", "nan"))
    cfg["w"] = float(raw.get("w", "nan"))
    return cfg


def load_htrdr_radiance(path: Path) -> tuple[int, int, list[float]]:
    """LW htrdr-image: BT_E BT_SE rad_E rad_SE ... → use rad_E (index 2)."""
    rads: list[float] = []
    with path.open() as fh:
        w, h = map(int, fh.readline().split())
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            rads.append(float(parts[2]))
    if len(rads) != w * h:
        raise ValueError(f"{path}: expected {w*h} pixels, got {len(rads)}")
    return w, h, rads


def cal_rad_dn(L: float, cfg: dict) -> float:
    """Linear DN from band-integrated radiance L [W/sr/m^2]. No FW clip."""
    F = cfg["f"] / cfg["d"]
    E = math.pi * L * cfg["t0"] * cfg["ta"] * cfg["EE"] / (4.0 * F * F)
    P = E * cfg["dx"] * cfg["dx"]
    N_sig = P * cfg["IT"] * cfg["eta"] * cfg["lambda_hc"]
    N_dark = cfg["n_dark"] * cfg["dark_current_const"] * cfg["IT"]
    N = N_sig + N_dark
    return N / cfg["FW"] * cfg["DNR"] + cfg["BC"] * cfg["DNR"]


def write_f32(path: Path, values: list[float]) -> None:
    with path.open("wb") as fh:
        fh.write(struct.pack(f"<{len(values)}f", *values))


def write_u16_png(path: Path, width: int, height: int, dns: list[float]) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow required: pip install pillow") from exc

    dmin = min(dns)
    dmax = max(dns)
    # Exact DN fits in uint16: store as-is. Otherwise display-stretch only.
    use_stretch = dmin < 0 or dmax > 65535
    raw = bytearray()
    if use_stretch:
        span = dmax - dmin if dmax > dmin else 1.0
        print(
            f"note: linear DN [{dmin:.1f}, {dmax:.1f}] exceeds uint16; "
            f"PNG is min-max display stretch (exact DN in .dn.f32; no FW clip)",
            file=sys.stderr,
        )
        for v in dns:
            u = int(round((v - dmin) / span * 65535.0))
            u = 0 if u < 0 else (65535 if u > 65535 else u)
            raw += struct.pack("<H", u)
    else:
        for v in dns:
            u = int(round(v))
            u = 0 if u < 0 else (65535 if u > 65535 else u)
            raw += struct.pack("<H", u)
    Image.frombytes("I;16", (width, height), bytes(raw)).save(path)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input",
        type=Path,
        help="htrdr-atmosphere LW ASCII image (*.txt)",
    )
    ap.add_argument(
        "-c",
        "--config",
        type=Path,
        default=repo / "config" / "payload" / "mlw.cfg.ini",
        help="payload ini (default: config/payload/mlw.cfg.ini)",
    )
    ap.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        default=None,
        help="output prefix (default: <input> without suffix)",
    )
    ap.add_argument(
        "--section",
        default="MIDDLELONGWAVE",
        help="ini section name",
    )
    args = ap.parse_args()

    cfg = load_ini(args.config, args.section)
    width, height, rads = load_htrdr_radiance(args.input)
    dns = [cal_rad_dn(L, cfg) for L in rads]

    prefix = args.output_prefix
    if prefix is None:
        prefix = args.input.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    f32_path = Path(str(prefix) + ".dn.f32")
    png_path = Path(str(prefix) + "_dn_16.png")
    write_f32(f32_path, dns)
    write_u16_png(png_path, width, height, dns)

    F = cfg["f"] / cfg["d"]
    print(f"config: {args.config} [{args.section}]")
    print(f"size:   {width}x{height}")
    print(f"F/#:    {F:.4f}")
    print(f"L:      {min(rads):.6g} .. {max(rads):.6g} W/sr/m^2")
    print(f"DN:     {min(dns):.4f} .. {max(dns):.4f}  (no FW truncation)")
    print(f"wrote:  {f32_path}")
    print(f"wrote:  {png_path}")


if __name__ == "__main__":
    main()
