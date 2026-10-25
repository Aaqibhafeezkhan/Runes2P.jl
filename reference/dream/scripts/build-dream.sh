#!/usr/bin/env bash
# Configure and build DREAM against the local PETSc. The build step downloads
# ADAS atomic data through tools/get_adas.py on first run.
source "$(dirname "$0")/common.sh"
cmake -S "$DREAM_DIR" -B "$DREAM_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DPETSC_DIR="$PETSC_DIR" -DPETSC_ARCH="$PETSC_ARCH" \
    -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    -DHDF5_ROOT="$CONDA_PREFIX" -DGSL_ROOT_DIR="$CONDA_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build "$DREAM_BUILD" -j"$NJOBS"
# The DREAM Python interface looks for $DREAMPATH/build/iface/dreami.
ln -sfn "$DREAM_BUILD" "$DREAM_DIR/build"
