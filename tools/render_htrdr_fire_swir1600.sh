#!/bin/bash
set -euo pipefail
# FIRE shahed-camera SWIR: htrdr -s sw=1600,1601, Lambert ρ=0.04 water,
# same ortho as GPU SPECTRAL_SW (z=2000, height=400, 512², sun 340,25).
# No CIE. No blue_water. No ground -R. No cloud -r (film is inside the LES).
# New octree cache (CIE L25_FIRE_OCEAN_T1.cache is the wrong bands).
#
# Usage:
#   tools/prep_fire_swir1600_htrdr.py
#   SPP=32 tools/render_htrdr_fire_swir1600.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPK="${ROOT}/htrdr-atmosphere-spk/htrdr-Atmosphere-Starter-Pack-0.8.0"
OUT="${ROOT}/scenes/fire_gs/swir1600"
export PATH="${ROOT}/local/bin:${PATH}"
export LD_LIBRARY_PATH="${ROOT}/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HTRDR_ATMOSPHERE_SPK="${SPK}"

CAMENV="${OUT}/camera.env"
if [[ ! -f "${CAMENV}" ]]; then
    echo "missing ${CAMENV}; run tools/prep_fire_swir1600_htrdr.py first" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${CAMENV}"

atmosphere="${SPK}/ecrad_opt_prop.txt"
mie_lut="${SPK}/Mie_LUT_Cloud-2-10-0.010.nc"
clouds="${SPK}/clouds/L25.1.FIRE.012.htcp"
materials="${OUT}/water.mtls"
ground="${OUT}/water.obj"
cache="${OUT}/L25_FIRE_SWIR1600.cache"
sun_dir="340,25"
threads="${HTRDR_THREADS:-48}"
opthick="1"
def="${DEF:-512x512}"
spp="${SPP:-32}"
# 1 nm window so Planck CDF has nbands>=1. GPU is monochromatic 1600.0 nm.
spectral="sw=1600,1601"

txt="${OUT}/htrdr_fire_swir1600_${def}x${spp}.txt"
log="${OUT}/htrdr_fire_swir1600_${def}x${spp}.log"
cam="pos=${cx},${cy},${cz}:tgt=${cx},${cy},0:up=0,1,0:height=${height}"

echo "==== FIRE SWIR 1600 nm htrdr ${def} spp=${spp} cam=${cam} -s ${spectral} ===="
htrdr-atmosphere -v -f \
    -t "${threads}" \
    -a "${atmosphere}" \
    -c "${clouds}" \
    -m "${mie_lut}" \
    -g "${ground}" \
    -M "${materials}" \
    -D "${sun_dir}" \
    -s "${spectral}" \
    -i "def=${def}:spp=${spp}" \
    -P "${cam}" \
    -T "${opthick}" \
    -O "${cache}" \
    -o "${txt}" \
    2>&1 | tee "${log}"
echo "wrote ${txt}"
