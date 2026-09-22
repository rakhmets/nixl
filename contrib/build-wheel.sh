#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PYTHON_VERSION="3.12"
ARCH=$(uname -m)
WHL_PLATFORM="manylinux_2_39_$ARCH"
UCX_PLUGINS_DIR="/usr/lib64/ucx"
NIXL_PLUGINS_DIR="/usr/local/nixl/lib/$ARCH-linux-gnu/plugins"
OUTPUT_DIR="dist"
BUILD_NIXL_EP="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --python-version)
            PYTHON_VERSION=$2
            shift
            shift
            ;;
        --platform)
            WHL_PLATFORM=$2
            shift
            shift
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift
            shift
            ;;
        --ucx-plugins-dir)
            UCX_PLUGINS_DIR=$2
            shift
            shift
            ;;
        --nixl-plugins-dir)
            NIXL_PLUGINS_DIR=$2
            shift
            shift
            ;;
        --help)
            echo "Usage: $0 [--python-version <python-version>] [--platform <platform>] [--output-dir <output-dir>] [--ucx-plugins-dir <ucx-plugins-dir>] [--nixl-plugins-dir <nixl-plugins-dir>]"
            echo "  --python-version: Python version to build the wheel for (default: $PYTHON_VERSION)"
            echo "  --platform: Platform to build the wheel for (default: $WHL_PLATFORM)"
            echo "  --output-dir: Directory to output the wheel to (default: $OUTPUT_DIR)"
            echo "  --ucx-plugins-dir: Directory to find UCX plugins in (default: $UCX_PLUGINS_DIR)"
            echo "  --nixl-plugins-dir: Directory to find NIXL plugins in (default: $NIXL_PLUGINS_DIR)"
            echo "  --build-nixl-ep: Build wheel with nixl_ep package included (requires a CUDA sm_90 or newer target environment)"
            echo "  --help: Show this help message"
            echo ""
            echo "Must be executed from the root of the NIXL repository."
            exit 0
            ;;
        --build-nixl-ep)
            BUILD_NIXL_EP="true"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

set -e
set -x

TMP_DIR=$(mktemp -d)

CUDA_MAJOR=$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 | cut -d'.' -f1)
# Must be 12 or 13
if [ "$CUDA_MAJOR" -ne 12 ] && [ "$CUDA_MAJOR" -ne 13 ]; then
    echo "Invalid CUDA_MAJOR: '$CUDA_MAJOR'"
    exit 1
fi
AUDITWHEEL_EXCLUDES="--exclude libcuda* --exclude libcufile* --exclude libcuobjclient* --exclude libssl* --exclude libcrypto* --exclude libefa* --exclude libhwloc* --exclude libfabric* --exclude libtorch* --exclude libc10* --exclude libdoca* --exclude libred_client* --exclude libred_async* --exclude liblz4*"

PKG_NAME="nixl-cu${CUDA_MAJOR}"
CU_TAG="cu$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 | tr -d .)"
./contrib/tomlutil.py --wheel-name $PKG_NAME pyproject.toml

TORCH_STABLE_INDEX="https://download.pytorch.org/whl/${CU_TAG}"
TORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/${CU_TAG}"

# Build deps for the per-iteration venv; torch is installed separately.
BUILD_DEPS=(
    "meson"
    "meson-python"
    "pybind11"
    "patchelf"
    "pyyaml"
    "types-PyYAML"
    "setuptools>=80.9.0"
)

# Slugify a dotted version (e.g. "3.10" -> "310") so it can be used
# unambiguously as a path component.
slug() { echo "${1//./}"; }

# Path for the per-iteration build venv. Lives in /workspace, not /tmp, so
# it inherits the image's UV_CACHE_DIR layout and is visible to debugging.
venv_path() {
    echo "/workspace/venv-py$(slug "$PYTHON_VERSION")"
}

# Install the latest torch available from the cu index, isolated from
# PyPI: with PyPI as a fallback its plain stable release would beat cu
# nightly's `X.Y.0.dev*+cuXX` (PEP 440: final > pre-release). Falls back to
# the nightly index if no stable torch is available yet for this
# Python/CUDA combo.
install_torch() {
    local VENV_PATH=$1
    uv pip install --python "$VENV_PATH/bin/python" --index-url "$TORCH_STABLE_INDEX" torch 2>/dev/null || \
    uv pip install --python "$VENV_PATH/bin/python" --index-url "$TORCH_NIGHTLY_INDEX" --pre torch
}

# Build the wheel for the current PYTHON_VERSION. Each iteration uses a
# fresh venv so torch's dependencies (nvidia-* wheels, triton, sympy, …) do
# not leak across iterations.
build_wheel() {
    local OUT_DIR=$1

    local VENV_PATH
    VENV_PATH=$(venv_path)
    rm -rf "$VENV_PATH"
    uv venv "$VENV_PATH" --python "$PYTHON_VERSION"

    echo "=== Provisioning ${VENV_PATH} (python ${PYTHON_VERSION}) ==="
    uv pip install --python "$VENV_PATH/bin/python" "${BUILD_DEPS[@]}"

    if [ "$BUILD_NIXL_EP" = "true" ]; then
        install_torch "$VENV_PATH" || {
            echo "ERROR: no torch wheel available for Python ${PYTHON_VERSION} + ${CU_TAG} on $(uname -m)" >&2
            exit 1
        }
    fi

    # Activate so meson's `find_installation('python3')` resolves to this
    # venv's interpreter (which has torch, if installed above).
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"

    local BUILD_ARGS=(
        --wheel
        --no-build-isolation
        --out-dir "$OUT_DIR"
        --python "$VENV_PATH/bin/python"
    )
    if [ "$BUILD_NIXL_EP" = "true" ]; then
        BUILD_ARGS+=(
            -Csetup-args=-Dbuild_nixl_ep=true
            -Csetup-args=-Dbuild_examples=true
        )
    fi
    uv build "${BUILD_ARGS[@]}"

    deactivate
    # torch + nvidia-* in the venv is several GB; tear down so the docker
    # layer does not get too large across the python-version matrix.
    rm -rf "$VENV_PATH"
}

repair_wheel() {
    local IN_DIR=$1
    local OUT_DIR=$2
    mkdir -p "$OUT_DIR"
    auditwheel repair $AUDITWHEEL_EXCLUDES "$IN_DIR"/nixl*.whl --plat "$WHL_PLATFORM" --wheel-dir "$OUT_DIR"
    ./contrib/wheel_add_ucx_plugins.py --ucx-plugins-dir "$UCX_PLUGINS_DIR" --nixl-plugins-dir "$NIXL_PLUGINS_DIR" "$OUT_DIR"/*.whl
}

# Echo the path of the single .whl in $1, or exit if the count is not 1.
get_wheel_path() {
    local dir=$1 wheels
    shopt -s nullglob
    wheels=("$dir"/*.whl)
    shopt -u nullglob
    if [ ${#wheels[@]} -ne 1 ]; then
        echo "expected 1 wheel in $dir, got ${#wheels[@]}: ${wheels[*]}" >&2
        exit 1
    fi
    echo "${wheels[0]}"
}

build_wheel "$TMP_DIR"
repair_wheel "$TMP_DIR" "$TMP_DIR/dist"
cp "$(get_wheel_path "$TMP_DIR/dist")" "$OUTPUT_DIR"

# Clean up
rm -rf "$TMP_DIR"
