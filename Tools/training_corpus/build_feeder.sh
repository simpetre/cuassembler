#!/usr/bin/env bash
# Compile every .cu under this corpus for the target arch, disassemble with
# nvdisasm -hex -c (instruction + 128-bit encoding), strip sm_120 dataflow
# annotations, and concatenate into a single feeder file suitable for
# `Tools/build_sm120_repos.py`.
#
# Usage:
#   ./build_feeder.sh [ARCH] [OUTPUT_FEED]
#       ARCH         default: sm_120
#       OUTPUT_FEED  default: ./feed.txt
#
# Requires nvcc and nvdisasm on PATH (or set NVCC / NVDISASM env vars).

set -euo pipefail

ARCH="${1:-sm_120}"
OUT="${2:-./feed.txt}"
NVCC="${NVCC:-nvcc}"
CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"
# cuobjdump shells out to nvdisasm internally; make sure it can find it.
# Default to the CUDA toolkit's bin (where nvcc lives); override with NVDISASM_DIR.
_NVCC_BIN="$(dirname "$(command -v "${NVCC}" 2>/dev/null || echo /usr/local/cuda/bin/nvcc)")"
NVDISASM_DIR="${NVDISASM_DIR:-${_NVCC_BIN}}"
export PATH="${NVDISASM_DIR}:${PATH}"

CORPUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${CORPUS_DIR}/_build_${ARCH}"
mkdir -p "${BUILD_DIR}"

: > "${OUT}"
fail_count=0
ok_count=0

# Find all .cu files in deterministic (family-ordered) order
mapfile -t CU_FILES < <(find "${CORPUS_DIR}" -name '*.cu' | sort)

# Compile each file at multiple opt levels so the same source produces
# different SASS (e.g. -O3 fuses to FFMA where -O0 keeps separate FMUL+FADD).
# Each opt-level variant adds to the feeder, growing both shape coverage
# (different mnemonics) and operand-value variation (different reg allocs).
OPT_LEVELS=("-O3" "-O0 -G")

for cu in "${CU_FILES[@]}"; do
    rel="${cu#${CORPUS_DIR}/}"
    stem="${rel%.cu}"
    flat="${stem//\//_}"
    mkdir -p "${BUILD_DIR}"

    file_ok=0
    for opt in "${OPT_LEVELS[@]}"; do
        opt_tag="${opt// /}"
        cubin="${BUILD_DIR}/${flat}${opt_tag}.cubin"
        sass="${BUILD_DIR}/${flat}${opt_tag}.sass.txt"
        log="${cubin}.log"

        # shellcheck disable=SC2086
        if ! "${NVCC}" -arch="${ARCH}" -cubin -std=c++17 ${opt} -o "${cubin}" "${cu}" 2> "${log}"; then
            continue
        fi

        # cuobjdump --dump-sass emits the `.headerflags @"EF_CUDA_SMxxx ..."`
        # line the feeder uses to detect arch (raw nvdisasm doesn't). Each
        # instruction comes with its 128-bit encoding split across two
        # `/* 0x... */` comments. Strip the sm_120 dataflow hints (&wr=,
        # &req=, &rd=, ?transN, ?WAIT*_END_GROUP) so the parser sees a clean
        # instruction string.
        "${CUOBJDUMP}" --dump-sass "${cubin}" 2>/dev/null \
            | sed -E 's/[[:space:]]*&[A-Za-z_][A-Za-z_0-9]*=[^[:space:]]+//g;
                      s/[[:space:]]*\?[A-Za-z_0-9]+//g' \
            >> "${OUT}"
        file_ok=1
    done

    if [[ "${file_ok}" -eq 1 ]]; then
        echo "  OK    ${rel}"
        ok_count=$((ok_count + 1))
    else
        echo "  SKIP  ${rel}  (all opt levels failed)"
        fail_count=$((fail_count + 1))
    fi
done

echo
echo "compiled ${ok_count} / $((ok_count + fail_count)) files"
echo "feeder: ${OUT}  ($(wc -l < "${OUT}") lines)"
