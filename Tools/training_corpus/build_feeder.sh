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
NVDISASM_PATH="${NVDISASM_PATH:-/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/bin}"
export PATH="${NVDISASM_PATH}:${PATH}"

CORPUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${CORPUS_DIR}/_build_${ARCH}"
mkdir -p "${BUILD_DIR}"

: > "${OUT}"
fail_count=0
ok_count=0

# Find all .cu files in deterministic (family-ordered) order
mapfile -t CU_FILES < <(find "${CORPUS_DIR}" -name '*.cu' | sort)

for cu in "${CU_FILES[@]}"; do
    rel="${cu#${CORPUS_DIR}/}"
    stem="${rel%.cu}"
    cubin="${BUILD_DIR}/${stem//\//_}.cubin"
    sass="${BUILD_DIR}/${stem//\//_}.sass.txt"
    mkdir -p "$(dirname "${cubin}")"

    if ! "${NVCC}" -arch="${ARCH}" -cubin -std=c++17 -o "${cubin}" "${cu}" 2> "${cubin}.log"; then
        echo "  SKIP  ${rel}  (nvcc failed — see ${cubin}.log)"
        fail_count=$((fail_count + 1))
        continue
    fi

    # cuobjdump --dump-sass emits the `.headerflags @"EF_CUDA_SMxxx ..."`
    # line the feeder uses to detect arch (raw nvdisasm doesn't). Each
    # instruction comes with its 128-bit encoding split across two
    # `/* 0x... */` comments. Strip the sm_120 dataflow hints (&wr=, &req=,
    # ?transN, ?WAIT*_END_GROUP) so the parser sees a clean instruction.
    "${CUOBJDUMP}" --dump-sass "${cubin}" 2>/dev/null \
        | sed -E 's/[[:space:]]*&[A-Za-z_][A-Za-z_0-9]*=[^[:space:]]+//g;
                  s/[[:space:]]*\?[A-Za-z_0-9]+//g' \
        > "${sass}"

    cat "${sass}" >> "${OUT}"
    echo "  OK    ${rel}"
    ok_count=$((ok_count + 1))
done

echo
echo "compiled ${ok_count} / $((ok_count + fail_count)) files"
echo "feeder: ${OUT}  ($(wc -l < "${OUT}") lines)"
