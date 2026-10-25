#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
if [[ -d "$PETSC_DIR/.git" ]]; then
    echo "PETSc already present at $PETSC_DIR ($(git -C "$PETSC_DIR" describe --tags))"
    exit 0
fi
git clone --depth 1 -b "$PETSC_VERSION" https://gitlab.com/petsc/petsc.git "$PETSC_DIR"
