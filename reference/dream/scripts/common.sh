#!/usr/bin/env bash
# Shared paths for the reference DREAM build. Sourced by the other scripts.
set -euo pipefail
ROOT="${PIXI_PROJECT_ROOT:?run through pixi}"
# PETSc is shared by the data sets; each set has its own DREAM checkout and build.
export PETSC_DIR="$ROOT/petsc"
SET_DIR="${SET_DIR:?the directory of the data set, e.g. \$PIXI_PROJECT_ROOT/dream-v26.5}"
export DREAM_DIR="$SET_DIR/DREAM"
export DREAM_BUILD="$SET_DIR/build"
NJOBS="${NJOBS:-$(nproc)}"
# conda-forge c-compiler >= 2 provides plain gcc/g++ without exporting CC/CXX.
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"
