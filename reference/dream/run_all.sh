#!/usr/bin/env bash
# Regenerate the public reference data sets of Runes2P from the case scripts in cases/ and
# the GSL program in ../gsl/.
#
# Usage: bash run_all.sh [<data directory>] [set ...]
#   sets: dream-v26.5, dream-boltzmann-eqterm-93d27e17, gsl-2.8 (default: all three)
#   The data directory defaults to the folder of this script. A first argument that is not
#   the name of a set is taken as the data directory.
#   ONLY="<script> ..." in the environment runs only those case scripts of the DREAM sets.
#
# The data directory holds pixi.toml, pixi.lock and scripts/ of this folder, and what
# build_all.sh builds there: petsc/ and one directory per DREAM set with DREAM/ and build/.
# The case scripts are copied into <set>/cases/, because they find dreami, the DREAM checkout
# and PETSc relative to their own folder. The outputs go to out/<set>/, the working files to
# work/<set>/ and the logs to logs/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SET_NAMES=(dream-v26.5 dream-boltzmann-eqterm-93d27e17 gsl-2.8)
DATA_DIR="$HERE"
if [[ $# -gt 0 && " ${SET_NAMES[*]} " != *" $1 "* ]]; then
    DATA_DIR="$(cd "$1" && pwd)"
    shift
fi
SETS=("$@")
[[ ${#SETS[@]} -gt 0 ]] || SETS=("${SET_NAMES[@]}")

# cases of each DREAM set: <script> <output file> [arguments]; runaway_rates before
# dreicer_kinetic, which reads the fields of dream_runaway_rates.h5 from the same set.
V265_CASES=(
    "collision_frequencies dream_collisions.h5"
    "cold_energy_fluid dream_cold_energy.h5"
    "cooling_positivity dream_cooling_positivity.h5 np=200 nxi=5 nt=1000 t_final=10 t_end=1e-3 reltol=1e-10 save=10 variants=centred,tcdf"
    "disruption_kinetic dream_disruption_small.h5"
    "runaway_rates dream_runaway_rates.h5"
    "dreicer_kinetic dream_dreicer_kinetic.h5"
    "example_runaway dream_example_runaway.h5"
    "flux_limiters dream_flux_limiters.h5"
    "ion_energy_fluid dream_ion_energy.h5"
    "ion_rates_fluid dream_ion_rates.h5"
    # kinetic_ionization sets absolute tolerances for j_ohm, j_tot, I_p and psi_p: these hold
    # round-off at E = 0, and without the tolerances DREAM's Newton iteration can stall on them
    "kinetic_ionization dream_kinetic_ionization.h5"
    "kinetic_operator dream_kinetic_operator.h5"
    "pmax_boundary dream_pmax_boundary.h5"
    "radial_transport dream_radial_transport.h5"
    "rp_avalanche dream_rp_avalanche.h5"
    "self_consistent_field dream_self_consistent_field.h5"
)
BOLTZMANN_CASES=(
    "knockon_operator dream_knockon_operator.h5"
    "avalanche_kinetic dream_avalanche_kinetic.h5 source=boltzmann pmax=200 tmax=0.08 nt=192 windows=4 pcut=0.13 store=test"
    "example_avalanche dream_example_avalanche.h5"
    "time_stepping dream_time_stepping.h5"
    "time_stepping_adaptive dream_time_stepping_adaptive.h5"
    "two_grids dream_two_grids.h5"
)
# the test-size disruption of test/test_disruption_dream.jl, with the MKL linear solver
DISRUPTION_ARGS="--nr 2 --np1 10 --np2 20 --nxi 4 --nt-ioniz 10 --t-ioniz 4e-7 --nt-quench 10 --t-quench 1.4e-6 --linear-solver mkl"

run_dream_set() {  # run_dream_set <set> <commit> <branch label> <case lines...>
    local set=$1 commit=$2 branch=$3
    shift 3
    local set_dir="$DATA_DIR/$set" out="$DATA_DIR/out/$set" work="$DATA_DIR/work/$set"
    mkdir -p "$set_dir/cases" "$out" "$work" "$DATA_DIR/logs"
    cp "$HERE"/cases/*.py "$set_dir/cases/"
    [[ "$(git -C "$set_dir/DREAM" rev-parse HEAD)" == "$commit" ]] || { echo "DREAM of $set is not at $commit" >&2; exit 1; }
    for line in "$@"; do
        read -r script file args <<< "$line"
        [[ -z ${ONLY:-} || " $ONLY " == *" $script "* ]] || continue
        local extra=()
        if [[ $script == disruption_kinetic ]]; then
            read -r -a extra <<< "$DISRUPTION_ARGS"
            extra+=(--outdir "$work/disruption_small" --test-data "$out/$file")
        else
            extra=("$out/$file")
            [[ -z ${args:-} ]] || { local a; read -r -a a <<< "$args"; extra+=("${a[@]}"); }
        fi
        local start=$(date +%s)
        (cd "$work" && nice -n 10 pixi run --frozen --manifest-path "$DATA_DIR/pixi.toml" \
            env DREAMPATH="$set_dir/DREAM" PYTHONPATH="$set_dir/DREAM/py" \
                DREAM_COMMIT="$commit" DREAM_BRANCH="$branch" \
            python "$set_dir/cases/$script.py" "${extra[@]}") \
            > "$DATA_DIR/logs/run-$set-$script.log" 2>&1 \
            || { echo "$set $script FAILED (logs/run-$set-$script.log)"; continue; }
        echo "$set $script: $(( $(date +%s) - start )) s"
    done
}

for set in "${SETS[@]}"; do
    case $set in
        dream-v26.5)
            run_dream_set "$set" ecdd5e146537c77602c9d7cc76b36100200e4b9a v26.5 "${V265_CASES[@]}" ;;
        dream-boltzmann-eqterm-93d27e17)
            run_dream_set "$set" 93d27e1763827e19dafb1fe790e99c2688ad6d62 boltzmann-eqterm "${BOLTZMANN_CASES[@]}" ;;
        gsl-2.8)
            mkdir -p "$DATA_DIR/out/$set" "$DATA_DIR/work/$set"
            pixi run --frozen --manifest-path "$DATA_DIR/pixi.toml" bash -c 'set -euo pipefail
                test "$(gsl-config --version)" = 2.8
                gcc -O0 -I"$CONDA_PREFIX/include" "$0" -L"$CONDA_PREFIX/lib" -lgsl -lgslcblas -lm -o "$1/bicubic_reference"
                LD_LIBRARY_PATH="$CONDA_PREFIX/lib" "$1/bicubic_reference" > "$2/gsl_bicubic_reference.txt"' \
                "$HERE/../gsl/bicubic_reference.c" "$DATA_DIR/work/$set" "$DATA_DIR/out/$set"
            echo "$set: done" ;;
        *) echo "unknown set $set" >&2; exit 1 ;;
    esac
done
