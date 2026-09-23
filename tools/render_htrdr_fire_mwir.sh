#!/bin/bash
set -euo pipefail
# FIRE shahed-camera IR MWIR: htrdr -s lw=4000,4001 (compute_radiance_lw).
# Wavelength is GPU mwir dump λ=4000 nm. Not CIE, not -s sw=.
# Same ortho as GPU SPECTRAL_LW (z=2000, height=400, 512²).
# No ground -R. No cloud -r. New octree cache (SWIR cache is the wrong bands).
#
# Usage:
#   tools/prep_fire_mwir_htrdr.py
#   SPP=32 tools/render_htrdr_fire_mwir.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPK="${ROOT}/htrdr-atmosphere-spk/htrdr-Atmosphere-Starter-Pack-0.8.0"
OUT="${ROOT}/scenes/fire_gs/mwir4000"
export PATH="${ROOT}/local/bin:${PATH}"
export LD_LIBRARY_PATH="${ROOT}/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HTRDR_ATMOSPHERE_SPK="${SPK}"
# Singleton MPI without an unused NIC (some agent shells have no oob iface).
export OMPI_MCA_btl="${OMPI_MCA_btl:-self,vader}"
export OMPI_MCA_oob_tcp_if_include="${OMPI_MCA_oob_tcp_if_include:-lo}"

CAMENV="${OUT}/camera.env"
if [[ ! -f "${CAMENV}" ]]; then
    echo "missing ${CAMENV}; run tools/prep_fire_mwir_htrdr.py first" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${CAMENV}"

atmosphere="${SPK}/ecrad_opt_prop.txt"
mie_lut="${SPK}/Mie_LUT_Cloud-2-10-0.010.nc"
clouds="${SPK}/clouds/L25.1.FIRE.012.htcp"
materials="${OUT}/water.mtls"
ground="${OUT}/water.obj"
cache="${OUT}/L25_FIRE_MWIR4000.cache"
sun_dir="340,25"
threads="${HTRDR_THREADS:-48}"
opthick="1"
def="${DEF:-512x512}"
spp="${SPP:-32}"
# 1 nm window; GPU MWIR is monochromatic 4000.0 nm SPECTRAL_LW.
spectral="lw=4000,4001"

txt="${OUT}/htrdr_fire_mwir4000_${def}x${spp}.txt"
log="${OUT}/htrdr_fire_mwir4000_${def}x${spp}.log"
cam="pos=${cx},${cy},${cz}:tgt=${cx},${cy},0:up=0,1,0:height=${height}"

echo "==== FIRE MWIR 4000 nm htrdr ${def} spp=${spp} cam=${cam} -s ${spectral} ===="
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
