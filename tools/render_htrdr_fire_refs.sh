#!/bin/bash
set -euo pipefail
# htrdr CPU FIRE marine Sc over seawater (sun 340,25):
#   1) nadir ortho: 25 km full deck
#   2) boat-height perspective: Sc filling the sky, glossy sea (clouds -r)
#
# Usage:
#   tools/render_htrdr_fire_refs.sh
#   ORTHO_SPP=256 PERS_SPP=256 tools/render_htrdr_fire_refs.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPK="${ROOT}/htrdr-atmosphere-spk/htrdr-Atmosphere-Starter-Pack-0.8.0"
OUT="${ROOT}/scenes/fire_gs"
export PATH="${ROOT}/local/bin:${PATH}"
export LD_LIBRARY_PATH="${ROOT}/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HTRDR_ATMOSPHERE_SPK="${SPK}"

atmosphere="${SPK}/ecrad_opt_prop.txt"
mie_lut="${SPK}/Mie_LUT_Cloud-2-10-0.010.nc"
clouds="${SPK}/clouds/L25.1.FIRE.012.htcp"
materials="${OUT}/ocean.mtls"
ground="${OUT}/sea_chop.obj"
cache="${OUT}/L25_FIRE_OCEAN_T1.cache"
sun_dir="340,25"
threads="${HTRDR_THREADS:-48}"
opthick="1"

ortho_def="${ORTHO_DEF:-512x512}"
ortho_spp="${ORTHO_SPP:-256}"
pers_def="${PERS_DEF:-1280x720}"
pers_spp="${PERS_SPP:-256}"
exposure="${HTPP_EXPOSURE:-0.2}"

mkdir -p "${OUT}"

run_one() {
    name="$1"
    camflag="$2"
    camera="$3"
    def="$4"
    spp="$5"
    wrap="${6:-1}"
    txt="${OUT}/htrdr_ref_fire_${name}_${def}x${spp}.txt"
    ppm="${OUT}/htrdr_ref_fire_${name}_${def}x${spp}.ppm"
    png="${OUT}/htrdr_ref_fire_${name}_${def}x${spp}.png"
    log="${OUT}/htrdr_ref_fire_${name}_${def}x${spp}.log"

    wrap_opt=""
    if [[ "${wrap}" == "1" ]]; then
        wrap_opt="-r"
        echo "==== fire ${name} ${def} spp=${spp} threads=${threads} clouds -r ===="
    else
        echo "==== fire ${name} ${def} spp=${spp} threads=${threads} clouds finite (no -r) ===="
    fi
    # shellcheck disable=SC2086
    htrdr-atmosphere -v \
        -t "${threads}" \
        -a "${atmosphere}" \
        -c "${clouds}" ${wrap_opt} \
        -m "${mie_lut}" \
        -g "${ground}" -R \
        -M "${materials}" \
        -D "${sun_dir}" \
        -i "def=${def}:spp=${spp}" \
        "${camflag}" "${camera}" \
        -T "${opthick}" \
        -O "${cache}" \
        -o "${txt}" \
        2>&1 | tee "${log}"

    htpp -f -i "exposure=${exposure}" -o "${ppm}" "${txt}"
    /home/szyj2025/miniconda3/envs/p311/bin/python - <<PY
from PIL import Image
Image.open("${ppm}").save("${png}")
print("wrote ${png}")
PY
}

ortho_cam="pos=12500,12500,80000"
ortho_cam="tgt=12500,12500,0:${ortho_cam}"
ortho_cam="up=0,1,0:${ortho_cam}"
ortho_cam="height=25000:${ortho_cam}"

pers_cam="pos=1600,5800,18"
pers_cam="tgt=1994.67,5656.35,0:${pers_cam}"
pers_cam="up=0,0,1:${pers_cam}"
pers_cam="fov=70:${pers_cam}"

if [[ "${SKIP_ORTHO:-0}" != "1" ]]; then
    run_one "ortho" -P "${ortho_cam}" "${ortho_def}" "${ortho_spp}" 1
fi
if [[ "${SKIP_OBLIQUE:-0}" != "1" ]]; then
    run_one "oblique" -C "${pers_cam}" "${pers_def}" "${pers_spp}" 1
fi
echo "done"
