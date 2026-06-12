#!/usr/bin/env bash
# Build a pinned llama.cpp (with CUDA) and expose the llama-server binary.
#
# The study needs a single, recorded build: the same commit decodes every condition, and
# its SHA goes into the run manifest (SPEC §7, §12). This script clones/updates llama.cpp,
# checks out a chosen ref, builds with CUDA, and prints the resolved commit to pin.
#
# Usage:
#   scripts/build_llamacpp.sh                      # build LLAMA_CPP_REF (default: master)
#   LLAMA_CPP_REF=b6789 scripts/build_llamacpp.sh  # build a specific tag/commit
#   CUDA_ARCH=120 scripts/build_llamacpp.sh        # RTX 5070 Ti (Blackwell, sm_120)
#
# After the first successful build, record the printed commit in your config's
# `server.llama_cpp_commit` so later runs are reproducible.
set -euo pipefail

REPO_URL="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
# Default checkout location lives under vendor/ (gitignored); override with LLAMA_CPP_DIR.
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-vendor/llama.cpp}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
# "native" lets nvcc detect the local GPU; set CUDA_ARCH=120 explicitly for Blackwell.
CUDA_ARCH="${CUDA_ARCH:-native}"
# CPU backend ISA. Default ON = -march=native (fine on a single box). On a heterogeneous
# cluster set GGML_NATIVE=OFF for a portable baseline: a binary built with -march=native on a
# new CPU (e.g. saxa's Xeon) SIGILLs on older nodes (the 2080 Ti damnii boxes). GPU does the
# heavy compute, so the generic CPU build's only cost is slightly slower sampling/tokenization.
GGML_NATIVE="${GGML_NATIVE:-ON}"
JOBS="${JOBS:-$(nproc)}"

echo ">> llama.cpp build"
echo "   repo : ${REPO_URL}"
echo "   dir  : ${LLAMA_CPP_DIR}"
echo "   ref  : ${LLAMA_CPP_REF}"
echo "   arch : sm_${CUDA_ARCH} (CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH})"

if [[ ! -d "${LLAMA_CPP_DIR}/.git" ]]; then
    echo ">> cloning..."
    git clone "${REPO_URL}" "${LLAMA_CPP_DIR}"
fi

git -C "${LLAMA_CPP_DIR}" fetch --tags --force origin
git -C "${LLAMA_CPP_DIR}" checkout "${LLAMA_CPP_REF}"
# Fast-forward only when on a branch (a detached tag/commit has no upstream to pull).
if git -C "${LLAMA_CPP_DIR}" symbolic-ref -q HEAD >/dev/null; then
    git -C "${LLAMA_CPP_DIR}" pull --ff-only
fi

echo ">> configuring (CUDA on)..."
cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_CPP_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE="${GGML_NATIVE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DLLAMA_CURL=ON

echo ">> building llama-server (-j ${JOBS})..."
cmake --build "${LLAMA_CPP_DIR}/build" --target llama-server -j "${JOBS}"

SERVER_BIN="$(readlink -f "${LLAMA_CPP_DIR}/build/bin/llama-server")"
COMMIT="$(git -C "${LLAMA_CPP_DIR}" rev-parse HEAD)"

echo
echo ">> done."
echo "   binary : ${SERVER_BIN}"
echo "   commit : ${COMMIT}"
echo
echo "   Pin this build:"
echo "     - set server.llama_cpp_commit: ${COMMIT}  in your config"
echo "     - export LLAMA_SERVER_BIN=${SERVER_BIN}    so the runner finds it"
