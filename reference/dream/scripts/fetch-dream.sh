#!/usr/bin/env bash
# Clone DREAM at the ref DREAM_REF (a tag or a branch) and check out DREAM_COMMIT, which
# must be the commit of that ref when it was used; the data records it as dream_commit.
source "$(dirname "$0")/common.sh"
: "${DREAM_REF:?}" "${DREAM_COMMIT:?}"
if [[ -d "$DREAM_DIR/.git" ]]; then
    head=$(git -C "$DREAM_DIR" rev-parse HEAD)
    echo "DREAM already present at $DREAM_DIR (${head:0:8})"
    [[ "$head" == "$DREAM_COMMIT" ]] || { echo "ERROR: checkout is not at $DREAM_COMMIT" >&2; exit 1; }
    exit 0
fi
git clone -b "$DREAM_REF" https://github.com/chalmersplasmatheory/DREAM.git "$DREAM_DIR"
git -C "$DREAM_DIR" checkout --detach "$DREAM_COMMIT"
git -C "$DREAM_DIR" submodule update --init --recursive
