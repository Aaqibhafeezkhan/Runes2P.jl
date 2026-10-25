#!/usr/bin/env bash
# Build the shared PETSc and the two DREAM versions of the reference data sets, from the
# directory that holds pixi.toml. Usage: bash build_all.sh   (NJOBS sets the compile jobs)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export NJOBS="${NJOBS:-8}"
timed() {  # timed <log name> <command...>
    local start=$(date +%s)
    "${@:2}" > "logs/$1.log" 2>&1
    echo "$1: $(( $(date +%s) - start )) s" | tee -a logs/build-times.txt
}
build_set() {  # build_set <set name> <DREAM ref> <DREAM commit>
    local set_dir="$PWD/$1"
    mkdir -p "$set_dir"
    ln -sfn ../petsc "$set_dir/petsc"   # the case scripts read ROOT/petsc/include/petscversion.h
    SET_DIR="$set_dir" DREAM_REF="$2" DREAM_COMMIT="$3" timed "build-$1" nice -n 10 pixi run --frozen build-dream
}
SET_DIR="$PWD/dream-v26.5" timed build-petsc nice -n 10 pixi run --frozen build-petsc
build_set dream-v26.5 v26.5 ecdd5e146537c77602c9d7cc76b36100200e4b9a
build_set dream-boltzmann-eqterm-93d27e17 boltzmann-eqterm 93d27e1763827e19dafb1fe790e99c2688ad6d62
