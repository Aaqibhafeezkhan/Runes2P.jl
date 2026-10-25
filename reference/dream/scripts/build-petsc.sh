#!/usr/bin/env bash
# PETSc without MPI, with MKL for BLAS/LAPACK and PARDISO, as recommended by
# DREAM (doc/sphinx/source/compiling.rst). Compilers and MKL come from the
# pixi environment.
source "$(dirname "$0")/common.sh"
if [[ -f "$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so" ]]; then
    echo "PETSc already built: $PETSC_DIR/$PETSC_ARCH"
    exit 0
fi
cd "$PETSC_DIR"
OPT="-O3 -march=native -mtune=native"
# PETSc reads compiler flags from its own options; the conda activation
# variables would otherwise be ignored with a warning or conflict.
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u FFLAGS \
./configure \
    PETSC_ARCH="$PETSC_ARCH" \
    --with-cc="$CC" --with-cxx="$CXX" --with-fc=0 \
    --with-debugging=0 --with-mpi=0 --with-x=0 --with-shared-libraries=1 \
    --COPTFLAGS="$OPT" --CXXOPTFLAGS="$OPT" \
    --LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib" \
    --with-blaslapack-dir="$CONDA_PREFIX" \
    --with-mkl_pardiso-dir="$CONDA_PREFIX"
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" -j"$NJOBS" all
