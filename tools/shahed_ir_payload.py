#!/usr/bin/env python3
"""Map FIRE An-124 GPU path-tracer samples to a 2 km nadir IR payload.

SPP is Monte Carlo path count, not camera integration. A staring FPA already
integrates every physical path; dwell is set by 300 m halo contrast vs NETD
and by observer (50 m/s) or An-124 cruise (220 m/s) motion blur.

Does not touch atmosphere kernels or re-render. Reads
scenes/fire_gs/transport_bands/record.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RECORD = os.path.join(ROOT, "scenes", "fire_gs", "transport_bands", "record.json")

H_C_K = 0.0143877687750392  # m·K  (hc/k)
SNR_DETECT = 5.0
HALO_FWHM_M = 300.0
HALO_RADIUS_M = 0.5 * HALO_FWHM_M
BLUR_FRAC = 0.10  # stack window: blur < 10% of halo FWHM
EPS_MEAN = 0.5
A_AC_M2 = 69.1 * 73.3  # An-124 AABB fallback until record exists
T_SCENE_K = 280.0
H_OBS_M = 2000.0
V_CRUISE = 50.0
V_CROSS = 100.0
V_AN124 = 220.0
TAU_TH_S = 0.010

PAYLOADS = {
    "lwir": {
        "name": "主案 非制冷 LWIR VOx",
        "band": "lwir",
        "lambda_lo_um": 8.0,
        "lambda_hi_um": 12.0,
        "nx": 640,
        "ny": 512,
        "pitch_m": 17.0e-6,
        "f_m": 14.0e-3,
        "fnum": 1.0,
        "fps": 30.0,
        "netd_k": 0.050,
        "mass_g": (100.0, 400.0),
        "power_w": (1.0, 2.0),
        "cooled": False,
        "tau_th_s": TAU_TH_S,
    },
    "mwir": {
        "name": "备案 制冷 MWIR InSb/MCT",
        "band": "mwir",
        "lambda_lo_um": 3.7,
        "lambda_hi_um": 4.8,
        "nx": 640,
        "ny": 512,
        "pitch_m": 15.0e-6,
        "f_m": 9.6e-3,
        "fnum": 2.0,
        "fps": 30.0,
        "netd_k": 0.020,
        "mass_g": (1000.0, 2000.0),
        "power_w": (10.0, 20.0),
        "cooled": True,
        "tau_th_s": 1.0e-6,
    },
}


def target_area_m2(rec: dict) -> float:
    ac = rec.get("aircraft") or rec.get("shahed") or {}
    if "area_xy_m2" in ac:
        return float(ac["area_xy_m2"])
    size = ac.get("size_m")
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        return float(size[0]) * float(size[1])
    return float(A_AC_M2)


def dlnB_dT(lambda_m: float, temperature_K: float) -> float:
    """d ln B_λ / dT for Planck B_λ. ΔT ≈ C / this, if C = ΔL/L."""
    x = H_C_K / (lambda_m * temperature_K)
    ex = math.exp(x)
    return (x / temperature_K) * ex / (ex - 1.0)


def band_by_key(rec: dict, key: str) -> dict:
    for b in rec["bands"]:
        if b["band"] == key:
            return b
    raise KeyError(key)


def optics(p: dict) -> dict:
    ifov = p["pitch_m"] / p["f_m"]
    gsd = ifov * H_OBS_M
    fov_x = p["nx"] * gsd
    fov_y = p["ny"] * gsd
    d_ap = p["f_m"] / p["fnum"]
    n_halo = (math.pi * HALO_RADIUS_M**2) / max(gsd * gsd, 1e-12)
    return {
        "ifov_rad": ifov,
        "gsd_m": gsd,
        "fov_x_m": fov_x,
        "fov_y_m": fov_y,
        "D_m": d_ap,
        "n_halo_pix": n_halo,
        "halo_samples": HALO_FWHM_M / gsd,
    }


def contrast(band: dict, t_eff: float, area_m2: float) -> dict:
    a_halo = math.pi * HALO_RADIUS_M**2
    dilution = area_m2 / a_halo
    l_cloud = float(band["nouav"]["mean"])
    b_body = float(band["mesh"]["B_body"])
    lam = float(band["lambda_nm"]) * 1.0e-9
    dldt = dlnB_dT(lam, T_SCENE_K)
    delta_l = t_eff * EPS_MEAN * b_body * dilution
    c = delta_l / max(abs(l_cloud), 1e-30)
    dt = c / max(dldt, 1e-30)
    return {
        "T_eff": t_eff,
        "A_ac_m2": area_m2,
        "A_halo_m2": a_halo,
        "dilution": dilution,
        "eps": EPS_MEAN,
        "L_cloud": l_cloud,
        "B_body": b_body,
        "dlnB_dT": dldt,
        "delta_L": delta_l,
        "C": c,
        "dT_K": dt,
        "dT_mK": 1.0e3 * dt,
    }


def n_frames(dT_k: float, netd_k: float, snr: float = SNR_DETECT) -> float:
    if dT_k <= 0.0:
        return math.inf
    return (snr * netd_k / dT_k) ** 2


def motion(v_mps: float, t1: float, tau_th: float, fov_x: float) -> dict:
    s_max = BLUR_FRAC * HALO_FWHM_M
    t_max = s_max / v_mps
    return {
        "v_mps": v_mps,
        "s_frame_m": v_mps * t1,
        "s_thermal_m": v_mps * tau_th,
        "s_max_m": s_max,
        "t_max_s": t_max,
        "t_cross_halo_s": HALO_FWHM_M / v_mps,
        "t_cross_fov_s": fov_x / v_mps,
    }


def verdict(t_stack: float, t_cross_fov: float, t_max: float, dT_mk: float) -> str:
    if not math.isfinite(t_stack) or dT_mk < 1e-6:
        return "不可检出"
    if t_stack > t_cross_fov:
        return "不可检出（叠加长于越幕，目标飞出视场）"
    if t_stack > t_max:
        return "可检出（须短帧配准叠加，单帧不可拉长）"
    return "可检出（单次连续积分内）"


def case_row(band: dict, payload: dict, opt: dict, bound: str, t_eff: float, area_m2: float) -> dict:
    ctr = contrast(band, t_eff, area_m2)
    t1 = 1.0 / payload["fps"]
    nf = n_frames(ctr["dT_K"], payload["netd_k"])
    t_stack = nf / payload["fps"] if math.isfinite(nf) else math.inf
    mot50 = motion(V_CRUISE, t1, payload["tau_th_s"], opt["fov_x_m"])
    mot100 = motion(V_CROSS, t1, payload["tau_th_s"], opt["fov_x_m"])
    mot220 = motion(V_AN124, t1, payload["tau_th_s"], opt["fov_x_m"])
    return {
        "bound": bound,
        "payload": payload["name"],
        "band": band["band"],
        "contrast": ctr,
        "N_frame": nf,
        "t_stack_s": t_stack,
        "t1_s": t1,
        "fps": payload["fps"],
        "NETD_mK": 1.0e3 * payload["netd_k"],
        "motion_50": mot50,
        "motion_100": mot100,
        "motion_220": mot220,
        "verdict_50": verdict(t_stack, mot50["t_cross_fov_s"], mot50["t_max_s"], ctr["dT_mK"]),
        "verdict_100": verdict(t_stack, mot100["t_cross_fov_s"], mot100["t_max_s"], ctr["dT_mK"]),
        "verdict_220": verdict(t_stack, mot220["t_cross_fov_s"], mot220["t_max_s"], ctr["dT_mK"]),
    }


def fmt(x, nd=4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "inf"
    if isinstance(x, float) and (abs(x) >= 1.0e4 or 0.0 < abs(x) < 1.0e-3):
        return f"{x:.3g}"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def jsonable(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def print_report(rec: dict, rows: list[dict], optics_by: dict, area_m2: float) -> None:
    print("=== SPP 不是积分时间 ===")
    print(
        "GPU spp 是路径抽样次数。焦平面一帧已经积了物理上的全部路径；"
        "64–128 spp 只说明弹道事件还没抽到。相机瓶颈是晕衬比 ΔT 对 NETD，"
        "以及相对运动拖影能不能在叠加时间内配准。"
    )
    ac = rec.get("aircraft") or {}
    print(
        f"侦察机高度 {H_OBS_M:.0f} m 正视，目标 {ac.get('type', 'An-124')}  "
        f"晕 FWHM={HALO_FWHM_M:.0f} m，机体投影 {area_m2:.0f} m²，ε={EPS_MEAN}，SNR={SNR_DETECT:.0f}。"
    )
    print()

    print("=== 载荷光学（2 km 正视） ===")
    for key, p in PAYLOADS.items():
        o = optics_by[key]
        print(
            f"{p['name']}: {p['nx']}×{p['ny']}  pitch={p['pitch_m']*1e6:.0f} μm  "
            f"f={p['f_m']*1e3:.1f} mm  F/{p['fnum']:.1f}  D={o['D_m']*1e3:.1f} mm"
        )
        print(
            f"  FOV={o['fov_x_m']:.0f}×{o['fov_y_m']:.0f} m  GSD={o['gsd_m']:.2f} m  "
            f"晕上约 {o['halo_samples']:.0f} 像元  "
            f"质量 {p['mass_g'][0]:.0f}–{p['mass_g'][1]:.0f} g  "
            f"功耗 {p['power_w'][0]:.0f}–{p['power_w'][1]:.0f} W  "
            f"NETD={p['netd_k']*1e3:.0f} mK @ {p['fps']:.0f} Hz"
            f"{'  斯特林制冷' if p['cooled'] else '  非制冷 VOx'}"
        )
    print()

    print("=== 300 m 晕衬比与检出 ===")
    hdr = (
        f"{'波段':<6} {'透射界':<8} {'T_eff':>10} {'C':>10} {'ΔT mK':>10} "
        f"{'N帧':>8} {'t叠 s':>8} {'50 m/s':<22} {'220 m/s An-124':<22}"
    )
    print(hdr)
    for r in rows:
        print(
            f"{r['band']:<6} {r['bound']:<8} {fmt(r['contrast']['T_eff']):>10} "
            f"{fmt(r['contrast']['C']):>10} {fmt(r['contrast']['dT_mK']):>10} "
            f"{fmt(r['N_frame'], 3):>8} {fmt(r['t_stack_s']):>8} "
            f"{r['verdict_50']:<22} {r['verdict_220']:<22}"
        )
    print()

    print("=== 运动模糊（主案 LWIR 30 Hz） ===")
    lwir_opt = optics_by["lwir"]
    t1 = 1.0 / PAYLOADS["lwir"]["fps"]
    for v in (V_CRUISE, V_CROSS, V_AN124):
        m = motion(v, t1, TAU_TH_S, lwir_opt["fov_x_m"])
        print(
            f"v={v:.0f} m/s  单帧拖影={m['s_frame_m']:.2f} m  "
            f"热惯性={m['s_thermal_m']:.2f} m  "
            f"t_max(模糊<{m['s_max_m']:.0f} m)={m['t_max_s']:.2f} s  "
            f"晕过境={m['t_cross_halo_s']:.1f} s  "
            f"全幅越幕={m['t_cross_fov_s']:.1f} s"
        )
    print(
        "单帧 33 ms、热时间常数 10 ms 相对 300 m 晕可忽略。"
        "超过 t_max 必须短帧配准叠加，不能把微测辐射热计单帧拉到 0.6 s。"
    )
    print()
    lwir_ka = next(r for r in rows if r["band"] == "lwir" and r["bound"] == "ka")
    print("=== 判据 ===")
    print(
        f"吸收限（ka）：LWIR ΔT={lwir_ka['contrast']['dT_mK']:.1f} mK，"
        f"{lwir_ka['N_frame']:.0f} 帧 / {lwir_ka['t_stack_s']:.1f} s @ 30 Hz，"
        "50 m/s 须对移动 300 m 窗配准。"
    )
    print(
        "GPU kext：衬比仍受云柱 τ≈11 限制；若 ΔT 仍亚 mK 则叠加长于越幕，不可检出。"
    )
    print(
        "不要把渲染 2000 / 3e4 spp 做成曝光帧数；那是 400 m 画面上弹道蒙特卡罗的方差。"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="An-124 vs 2 km IR payload dwell")
    ap.add_argument("--record", default=DEFAULT_RECORD)
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "scenes", "fire_gs", "transport_bands", "payload_dwell.json"),
    )
    args = ap.parse_args()
    with open(args.record, encoding="utf-8") as f:
        rec = json.load(f)

    optics_by = {k: optics(p) for k, p in PAYLOADS.items()}
    area_m2 = target_area_m2(rec)
    rows = []
    for key, p in PAYLOADS.items():
        band = band_by_key(rec, p["band"])
        analog = band["analog"]
        rows.append(case_row(band, p, optics_by[key], "kext", float(analog["Tr"]), area_m2))
        rows.append(case_row(band, p, optics_by[key], "ka", float(analog["Tr_ka_only"]), area_m2))

    print_report(rec, rows, optics_by, area_m2)

    out = {
        "note": (
            "SPP is path-tracer sample count, not FPA dwell. "
            "Target is An-124 (not Shahed). Detection uses 300 m halo contrast vs NETD."
        ),
        "geometry": {
            "H_obs_m": H_OBS_M,
            "halo_fwhm_m": HALO_FWHM_M,
            "A_ac_m2": area_m2,
            "eps": EPS_MEAN,
            "T_scene_K": T_SCENE_K,
            "SNR": SNR_DETECT,
            "v_cruise_mps": V_CRUISE,
            "v_cross_mps": V_CROSS,
            "v_an124_mps": V_AN124,
        },
        "payloads": {
            k: {**p, "optics": optics_by[k], "mass_g": list(p["mass_g"]), "power_w": list(p["power_w"])}
            for k, p in PAYLOADS.items()
        },
        "cases": rows,
        "record": args.record,
        "stamp": rec.get("stamp"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(jsonable(out), f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
