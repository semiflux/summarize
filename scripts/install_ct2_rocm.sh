#!/usr/bin/env bash
# Install the official CTranslate2 ROCm wheel (AMD GPU) over the CUDA-only one,
# so faster-whisper can use an AMD GPU. Safe to re-run; nothing is built.
#
# Usage:
#   PYTHON=.venv/bin/python scripts/install_ct2_rocm.sh          # latest pinned
#   PYTHON=.venv/bin/python scripts/install_ct2_rocm.sh 4.8.3    # explicit version
#
# Roll back with:  .venv/bin/pip install --force-reinstall ctranslate2
set -euo pipefail

VERSION="${1:-${CT2_VERSION:-4.8.3}}"
PYTHON="${PYTHON:-python3}"

command -v "$PYTHON" >/dev/null 2>&1 || { echo "python not found: $PYTHON" >&2; exit 1; }
[ -d /opt/rocm ] || { echo "ROCm not found at /opt/rocm -- install ROCm first." >&2; exit 1; }

PY_TAG="$("$PYTHON" -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
ARCH="$(uname -m)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

URL="https://github.com/OpenNMT/CTranslate2/releases/download/v${VERSION}/rocm-python-wheels-Linux.zip"
echo "Downloading CTranslate2 v${VERSION} ROCm wheels (~280 MB)…"
curl -fL --retry 3 -o "$TMP/ct2.zip" "$URL"

# Match ctranslate2-<ver>-<pytag>-<pytag>-manylinux...x86_64.whl (excludes the
# free-threaded "-cp314t-" variants, which are a different ABI).
PAT="ctranslate2-${VERSION}-${PY_TAG}-${PY_TAG}-.*manylinux.*${ARCH}\\.whl\$"
WHEEL="$(unzip -l "$TMP/ct2.zip" | awk '{print $NF}' | grep -E "$PAT" | head -1 || true)"
if [ -z "$WHEEL" ]; then
    echo "No ${PY_TAG} wheel for v${VERSION}. Wheels in the archive:" >&2
    unzip -l "$TMP/ct2.zip" | grep '\.whl' >&2
    exit 1
fi

echo "Installing $(basename "$WHEEL")…"
unzip -o -j "$TMP/ct2.zip" "$WHEEL" -d "$TMP" >/dev/null
"$PYTHON" -m pip install --force-reinstall --no-deps "$TMP/$(basename "$WHEEL")"

echo "Verifying…"
"$PYTHON" - <<'PY'
import ctranslate2
n = ctranslate2.get_cuda_device_count()
print(f"ctranslate2 {ctranslate2.__version__}: {n} GPU device(s) visible")
if n == 0:
    print("0 devices: check that /dev/kfd is accessible and that your user is in "
          "the 'video' and 'render' groups (re-login after adding).")
PY
