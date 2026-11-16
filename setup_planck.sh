#!/usr/bin/env bash
# Download the Planck 2018 likelihood code and data, build the clik library and
# its Python bindings, and run a self-check.
#
# Usage:  ./setup_planck.sh            (installs into ./planck_2018)
#         PLANCK_DIR=/some/path ./setup_planck.sh
#         PYTHON=/path/to/python3 ./setup_planck.sh
#
# Requirements: a C compiler, gfortran, cfitsio, and a Python 3 with numpy and
# cython (see README.md). Tested on macOS (Apple Silicon, Homebrew).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANCK_DIR="${PLANCK_DIR:-$REPO/planck_2018}"
PYTHON="${PYTHON:-python3}"
CC="${CC:-cc}"

PLA="https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.FILE_ID="
CODE_TGZ="COM_Likelihood_Code-v3.0_R3.10.tar.gz"
DATA_TGZ="COM_Likelihood_Data-baseline_R3.00.tar.gz"
BANDPOWERS="COM_PowerSpect_CMB-TT-full_R3.01.txt"

mkdir -p "$PLANCK_DIR"
cd "$PLANCK_DIR"
PLANCK_DIR="$(pwd)"
PLC="$PLANCK_DIR/code/plc_3.0/plc-3.1"

fetch() {
  if [ ! -s "$1" ]; then
    echo "Downloading $1 ..."
    curl -fL --retry 3 -o "$1.part" "$PLA$1"
    mv "$1.part" "$1"
  fi
}

# ── 1. download ──────────────────────────────────────────────────────────────
fetch "$CODE_TGZ"
fetch "$DATA_TGZ"
fetch "$BANDPOWERS"

# ── 2. unpack ────────────────────────────────────────────────────────────────
[ -d "$PLC" ]                  || tar xzf "$CODE_TGZ"
[ -d baseline/plc_3.0/hi_l ]   || tar xzf "$DATA_TGZ"

# ── 3. build libclik ─────────────────────────────────────────────────────────
cd "$PLC"

# waf hard-codes "-arch x86_64" on macOS; use the native architecture instead.
"$PYTHON" - <<'EOF'
p = "waf_tools/mbits.py"
s = open(p).read()
s = s.replace('mopt += ["-arch", "x86_64"]',
              'mopt += ["-arch", __import__("platform").machine()]')
open(p, "w").write(s)
EOF

CFITSIO_PREFIX="${CFITSIO_PREFIX:-}"
if [ -z "$CFITSIO_PREFIX" ] && command -v brew >/dev/null; then
  CFITSIO_PREFIX="$(brew --prefix cfitsio)"
fi
CFG_ARGS=()
[ -n "$CFITSIO_PREFIX" ] && CFG_ARGS+=(--cfitsio_prefix="$CFITSIO_PREFIX")

# -std=gnu89: plc's C sources predate C99/C23 defaults of current compilers.
# On macOS, BLAS/LAPACK come from Apple's Accelerate framework by default.
# --no_pytools: waf's Python support is broken on Python >= 3.12; step 4 builds it instead.
CFLAGS="-std=gnu89" "$PYTHON" waf configure --no_pytools "${CFG_ARGS[@]}" ${WAF_CONFIGURE_ARGS:-}
CFLAGS="-std=gnu89" "$PYTHON" waf install --no_pytools

# ── 4. build the Python bindings ─────────────────────────────────────────────
SITE="$PLC/lib/python/site-packages"
mkdir -p "$SITE/clik"
cp src/python/clik/*.py "$SITE/clik/"
NUMPY_INC="$("$PYTHON" -c 'import numpy; print(numpy.get_include())')"
PY_INC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
EXT="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

if [ "$(uname)" = Darwin ]; then
  # make libclik relocatable and let the extensions find it relative to themselves
  install_name_tool -id @rpath/libclik.dylib "$PLC/lib/libclik.dylib"
  LINK=(-bundle -undefined dynamic_lookup -Wl,-rpath,@loader_path/../../..)
else
  LINK=(-shared -fPIC -Wl,-rpath,'$ORIGIN/../../..')
fi

build_ext() {  # name  pyx-source
  "$PYTHON" -m cython -3 --output-file "build/$1.c" -I src/python/clik "$2"
  "$CC" -O2 -fPIC -I"$NUMPY_INC" -I"$PY_INC" \
    -Isrc -Isrc/python/clik -Isrc/plik -Iinclude -Isrc/minipmc \
    -DHAS_LAPACK -DLAPACK_CLIK -DNOHEALPIX -DCLIK_LENSING \
    "build/$1.c" -L"$PLC/lib" -lclik "${LINK[@]}" \
    -o "$SITE/clik/$1$EXT"
}
for m in lkl lkl_lensing parametric; do
  build_ext "$m" "src/python/clik/$m.pyx"
done
build_ext rel2015 src/plik/component_plugin/rel2015/rel2015.pyx

# ── 5. self-check ────────────────────────────────────────────────────────────
cd "$REPO"
PLANCK_DIR="$PLANCK_DIR" "$PYTHON" planck_paths.py

echo
echo "Planck likelihood installed in $PLANCK_DIR"
