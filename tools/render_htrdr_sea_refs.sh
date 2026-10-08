#!/bin/bash
set -euo pipefail
# htrdr CPU sea-cloud refs (BOMEX -r, sea_3m.obj -R, sun 0,50):
#   1) nadir ortho over the 12.8 km LES tile
#   2) sea.sh perspective
#
# Usage:
#   tools/render_htrdr_sea_refs.sh
#   ORTHO_SPP=256 PERS_SPP=256 tools/render_htrdr_sea_refs.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPK="${ROOT}/htrdr-atmosphere-spk/htrdr-Atmosphere-Starter-Pack-0.8.0"
OUT="${ROOT}/scenes/sea_gs"
export PATH="${ROOT}/local/bin:${PATH}"
export LD_LIBRARY_PATH="${ROOT}/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HTRDR_ATMOSPHERE_SPK="${SPK}"

atmosphere="${SPK}/ecrad_opt_prop.txt"
mie_lut="${SPK}/Mie_LUT_Cloud-2-10-0.010.nc"
clouds="${SPK}/clouds/L12km.1.BOMEX.005.htcp"
materials="${SPK}/materials/sea.mtls"
ground="${SPK}/models/sea_3m.obj"
cache="${OUT}/L12km_BOMEX_T1.cache"
sun_dir="0,50"
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
    txt="${OUT}/htrdr_ref_sea_${name}_${def}x${spp}.txt"
    ppm="${OUT}/htrdr_ref_sea_${name}_${def}x${spp}.ppm"
    png="${OUT}/htrdr_ref_sea_${name}_${def}x${spp}.png"
    log="${OUT}/htrdr_ref_sea_${name}_${def}x${spp}.log"

    echo "==== sea ${name} ${def} spp=${spp} threads=${threads} ===="
    htrdr-atmosphere -v \
        -t "${threads}" \
        -a "${atmosphere}" \
        -c "${clouds}" -r \
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

# 12.8 km BOMEX tile, nadir, same convention as desert_ortho.sh
ortho_cam="pos=6400,6400,80000"
ortho_cam="tgt=6400,6400,0:${ortho_cam}"
ortho_cam="up=0,1,0:${ortho_cam}"
ortho_cam="height=12800:${ortho_cam}"

# sea.sh
pers_cam="pos=2224.91,3684.71,1492.95"
pers_cam="tgt=8208.71,4471.54,-393.82:${pers_cam}"
pers_cam="up=0,0,1:${pers_cam}"
pers_cam="fov=70:${pers_cam}"

run_one "ortho" -P "${ortho_cam}" "${ortho_def}" "${ortho_spp}"
run_one "oblique" -C "${pers_cam}" "${pers_def}" "${pers_spp}"
echo "done"
