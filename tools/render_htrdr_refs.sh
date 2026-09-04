#!/bin/bash
set -euo pipefail
# High-spp htrdr CPU references for mountain.obj:
#   1) nadir ortho  (desert_ortho camera, mountain ground)
#   2) perspective  (mountain.sh camera)
#
# Usage:
#   tools/render_htrdr_refs.sh
#   ORTHO_SPP=256 PERS_SPP=256 tools/render_htrdr_refs.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPK="${ROOT}/htrdr-atmosphere-spk/htrdr-Atmosphere-Starter-Pack-0.8.0"
OUT="${ROOT}/scenes/mountain_gs"
export PATH="${ROOT}/local/bin:${PATH}"
export LD_LIBRARY_PATH="${ROOT}/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HTRDR_ATMOSPHERE_SPK="${SPK}"

atmosphere="${SPK}/ecrad_opt_prop.txt"
mie_lut="${SPK}/Mie_LUT_Cloud-2-10-0.010.nc"
clouds="${SPK}/clouds/DZVAR.1.ARMCU.008.diaKCL.htcp"
materials="${SPK}/materials/mountain.mtls"
ground="${SPK}/models/mountain.obj"
cache="${OUT}/DZVAR_T1.cache"
sun_dir="225,30"
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
    txt="${OUT}/htrdr_ref_${name}_${def}x${spp}.txt"
    ppm="${OUT}/htrdr_ref_${name}_${def}x${spp}.ppm"
    png="${OUT}/htrdr_ref_${name}_${def}x${spp}.png"
    log="${OUT}/htrdr_ref_${name}_${def}x${spp}.log"

    echo "==== ${name} ${def} spp=${spp} ===="
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

ortho_cam="pos=3200,3200,80000"
ortho_cam="tgt=3200,3200,0:${ortho_cam}"
ortho_cam="up=0,1,0:${ortho_cam}"
ortho_cam="height=6400:${ortho_cam}"

pers_cam="pos=10245.37,615.02,422.34"
pers_cam="tgt=-2977.39,6829.40,842.40:${pers_cam}"
pers_cam="up=0,0,1:${pers_cam}"
pers_cam="fov=30:${pers_cam}"

run_one "ortho" -P "${ortho_cam}" "${ortho_def}" "${ortho_spp}"
run_one "mountain" -C "${pers_cam}" "${pers_def}" "${pers_spp}"
echo "done"
