#!/usr/bin/env bash
# Run DREAM's C++ unit tests, including the knock-on tests of the branch.
source "$(dirname "$0")/common.sh"
cd "$DREAM_BUILD"
./tests/cxx/dreamtests all
