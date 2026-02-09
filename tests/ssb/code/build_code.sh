#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="Release"
TARGET="ssb_code_all"
JOBS=""
CONFIGURE_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  ./build_code.sh [options]

Options:
  -b, --build-dir <dir>      Build directory (default: ./build)
  -t, --build-type <type>    Build type (default: Release)
  -j, --jobs <n>             Number of parallel threads (default: auto-detected)
  --target <name>            Build target (default: ssb_code_all)
  --configure-only           Only configure with CMake, do not build
  -h, --help                 Display help

Examples:
  ./build_code.sh
  ./build_code.sh -t Debug
  ./build_code.sh --target ssb_code_gpu_bp -j 32
  ./build_code.sh --configure-only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    -t|--build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    -j|--jobs)
      JOBS="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --configure-only)
      CONFIGURE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v cmake >/dev/null 2>&1; then
  echo "Error: cmake not found. Please install CMake first." >&2
  exit 1
fi

if [[ -z "${JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS="8"
  fi
fi

echo "[1/2] CMake Configuration"
echo "  Source directory: ${SCRIPT_DIR}"
echo "  Build directory: ${BUILD_DIR}"
echo "  Build type: ${BUILD_TYPE}"

cmake \
  -S "${SCRIPT_DIR}" \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

if [[ "${CONFIGURE_ONLY}" -eq 1 ]]; then
  echo "Configuration completed (configure-only)."
  exit 0
fi

echo "[2/2] Starting build"
echo "  Target: ${TARGET}"
echo "  Parallel jobs: ${JOBS}"

cmake --build "${BUILD_DIR}" --target "${TARGET}" --parallel "${JOBS}"

echo "Build completed: ${BUILD_DIR}"
