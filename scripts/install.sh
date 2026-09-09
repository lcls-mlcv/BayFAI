#!/bin/bash
# Install BayFAI and register a Jupyter kernel that can run
# notebooks/manual_calibration.ipynb.
#
#   ./scripts/install.sh                # psana2 (LCLS-II), the default
#   ./scripts/install.sh --psana1       # psana1 (LCLS-I)
#   ./scripts/install.sh --name BayFAI-dev
#
# Why a dedicated kernel: LCLSGeom is not installable from PyPI (it needs psana
# and PSCalib), and the copy bundled in the psana conda envs lags behind the
# maintained one in /sdf/group. We therefore point PYTHONPATH at the maintained
# checkout and bake that into the kernel, exactly as LUTE does for its BayFAI
# tasks (see lute/managed_tasks.py).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LCLSGEOM_SRC="/sdf/group/lcls/ds/tools/LCLSGeom/src"
PSCONDA1="/sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh"
PSCONDA2="/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh"

PSANA_VERSION=2
KERNEL_NAME="BayFAI"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --psana1) PSANA_VERSION=1; shift ;;
        --psana2) PSANA_VERSION=2; shift ;;
        --name)   KERNEL_NAME="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "$PSANA_VERSION" == "1" ]]; then
    PSCONDA="$PSCONDA1"
else
    PSCONDA="$PSCONDA2"
fi

if [[ ! -f "$PSCONDA" ]]; then
    echo "ERROR: cannot find psana setup script at $PSCONDA" >&2
    exit 1
fi
if [[ ! -d "$LCLSGEOM_SRC" ]]; then
    echo "ERROR: cannot find LCLSGeom at $LCLSGEOM_SRC" >&2
    exit 1
fi

echo ">>> Sourcing psana$PSANA_VERSION environment"
# psconda.sh is not written for `set -u`.
set +u
# shellcheck disable=SC1090
source "$PSCONDA"
set -u

echo ">>> Using python: $(command -v python)"

echo ">>> Installing bayfai (editable, no deps)"
# --no-deps: numpy, pyFAI, h5py, matplotlib, scikit-learn, mpi4py, panel, bokeh
# and psana all ship with the psana env. Letting pip resolve them would pull
# incompatible PyPI wheels over the curated env.
python -m pip install -e "$REPO_ROOT" --no-deps --no-build-isolation --user

echo ">>> Verifying imports"
PYTHONPATH="$LCLSGEOM_SRC${PYTHONPATH:+:$PYTHONPATH}" python - <<'PYCHECK'
import LCLSGeom.converter, LCLSGeom.manager  # noqa: F401
import bayfai.optimization  # noqa: F401
import bayfai.fake_detector  # noqa: F401
print("    LCLSGeom and bayfai import cleanly")
PYCHECK

echo ">>> Registering Jupyter kernel '$KERNEL_NAME'"
python -m ipykernel install --user --name "$KERNEL_NAME" \
    --display-name "$KERNEL_NAME (psana$PSANA_VERSION)"

# ipykernel lower-cases the kernelspec name on disk, so ask Jupyter where it
# actually landed rather than assuming the directory matches $KERNEL_NAME.
KERNEL_DIR="$(KERNEL_NAME="$KERNEL_NAME" python - <<'PYDIR'
import os
from jupyter_client.kernelspec import KernelSpecManager

print(KernelSpecManager().get_kernel_spec(os.environ["KERNEL_NAME"].lower()).resource_dir)
PYDIR
)"

echo ">>> Baking the psana environment into the kernel"
# psana itself is supplied via PYTHONPATH by psconda.sh, so the kernel must
# inherit that too -- LCLSGeom goes in front of it, not instead of it.
KERNEL_PYTHONPATH="$LCLSGEOM_SRC${PYTHONPATH:+:$PYTHONPATH}"

# A Jupyter kernel inherits the SERVER's environment and then applies its own
# `env` on top. Relying on inheritance would mean the kernel only works when the
# server itself was launched from psconda.sh -- and psana1 fails with an
# unhelpful "expected str ... not NoneType" from Detector/dir_root.py when
# SIT_ROOT is unset. So capture the variables psana actually needs, now, while
# the environment is sourced, and write them into the spec.
KERNEL_JSON="$KERNEL_DIR/kernel.json" \
KERNEL_PYTHONPATH="$KERNEL_PYTHONPATH" \
python - <<'PYPATCH'
import json
import os

# Captured from the sourced psana environment. SIT_ROOT/SIT_ARCH/SIT_DATA are
# required by PSCalib and the psana1 Detector layer; SIT_PSDM_DATA is how psana
# locates experiment data; HDF5_USE_FILE_LOCKING must be FALSE on Lustre.
PSANA_VARS = (
    "SIT_ROOT",
    "SIT_ARCH",
    "SIT_DATA",
    "SIT_PSDM_DATA",
    "TESTRELDIR",
    "EPICS_BASE",
    "PYEPICS_LIBCA",
    "HDF5_USE_FILE_LOCKING",
    "OPENBLAS_NUM_THREADS",
)

path = os.environ["KERNEL_JSON"]
with open(path) as f:
    spec = json.load(f)

env = spec.setdefault("env", {})
env["PYTHONPATH"] = os.environ["KERNEL_PYTHONPATH"]

carried = []
for var in PSANA_VARS:
    value = os.environ.get(var)
    if value:
        env[var] = value
        carried.append(var)

# Match the thread limits LUTE sets for the BayFAI tasks.
env.setdefault("NUMEXPR_MAX_THREADS", "16")
env.setdefault("NUMEXPR_NUM_THREADS", "16")

with open(path, "w") as f:
    json.dump(spec, f, indent=1)
print(f"    patched {path}")
print(f"    carried psana vars: {', '.join(carried)}")
PYPATCH

cat <<EOF

Done. Next steps:
  1. Open a Jupyter session on S3DF OnDemand.
  2. Open notebooks/manual_calibration.ipynb.
  3. Select the "$KERNEL_NAME (psana$PSANA_VERSION)" kernel.

Kernel spec: $KERNEL_DIR/kernel.json
Kernel id:   $(echo "$KERNEL_NAME" | tr '[:upper:]' '[:lower:]')
EOF
