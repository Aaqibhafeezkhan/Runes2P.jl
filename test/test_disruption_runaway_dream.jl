# The disruption case with DREAM's runaway grid against DREAM: DREAM's examples/Disruptions
# generate.py --kinetic --cylindrical, scenario 1, at the size of the tests (2 radii, a hot-tail
# grid of 30 × 4 cells) with the case script's runaway grid
# (reference/dream/cases/disruption_kinetic.py --runaway-grid: 32 × 8 cells from the hot-tail p_max
# 0.8 to 20). The window of dream_disruption_runaway_small.h5 is DREAM's state at step 170 of the
# quench, 1.31e-4 s after the start of the scenario, where the runaway current is 5.9 % of the
# plasma current, and the 10 steps of 6.53e-7 s after it. The system of example_disruption.jl with
# `runaway = true` starts from that state and runs the 10 steps with DREAM's settings: the flux
# through the interface of the two grids into f_re and n_re, the kinetic avalanche on f_re and the
# fluid one in n_re above the hot-tail p_max, the runaway current as the moment of f_re, the kinetic
# ionization by f_re and the heating of the cold electrons by it, with every term of the case
# without a runaway grid. At the first state, DREAM's j_re is checked against the moment of its f_re
# on the built grid, and n_re against ∫f_re (n_re exceeds it by the runaways beyond the grid's
# p_max). The budgets of the assembled system and of the runaway grid are asserted independently of
# DREAM, and the window runs once more with Runes2P's defaults (compat = runes2p), where only the
# budgets are checked. The allocation of jacobian! at the first state is bounded, and so is that
# of the plasma columns of each term that adds them. Needs the ADF11 cache.
using HDF5
include(joinpath(@__DIR__, "example_disruption.jl"))

# the largest relative difference of each quantity over the window: about ten times the larger
# of the two measured at -O0 and at the default optimisation level (in the comment), so that a
# change of the method shows while a change of the rounding does not
#! format: off
const WINDOW_TOLERANCES = Dict("f_hot" => 5e-8,      # 4.9e-9
                               "f_re" => 2e-8,       # 1.1e-9
                               "j_re" => 2e-8,       # 1.2e-9
                               "n_re" => 2e-8,       # 1.1e-9
                               "n_hot" => 6e-8,      # 5.9e-9
                               "E_field" => 5e-8,    # 4.8e-9
                               "j_hot" => 1e-9,      # 8.1e-11
                               "n_i" => 3e-7,        # 2.2e-8
                               "j_ohm" => 4e-8,      # 3.8e-9
                               "n_cold" => 5e-8,     # 4.5e-9
                               "T_cold" => 1e-8,     # 9.8e-10
                               "W_cold" => 5e-8,     # 4.6e-9
                               "W_i" => 1e-8,        # 9.1e-10
                               "I_p" => 3e-11,       # 2.8e-12
                               "j_tot" => 4e-11,     # 3.5e-12
                               "psi_p" => 2e-11,     # 1.4e-12
                               "n_tot" => 1e-14)     # 2.5e-16, round-off of the quasi-neutral sum
# the closures of the runaway grid over the window (`runaway_budgets`), at least 44 times the
# larger of the values measured at -O0 and -O2 (compat = dream / runes2p): the particles through
# the interface to round-off, and the balance of n_re − ∫f_re against the avalanche beyond the
# grid and the outflow there
const WINDOW_RUNAWAY_BUDGETS = (interface = 2e-14,   # 2.8e-16 / 4.5e-16
                                avalanche = 1e-11)   # 1.3e-13 / 9.0e-14
#! format: on

# an assertion on allocations holds for optimised code alone: it is skipped, and reported
# as broken, in a development run at -O0 or -O1
const OPTIMIZED = Base.JLOptions().opt_level >= 2
# the bytes a call of jacobian! of the window allocates after a first call, and those of the
# plasma columns of each term that adds them: about twice those measured at -O2 in this
# testset (in the comment). A loop that inserts the partials of a dual number whose type is
# picked at run time boxes every entry: it exceeds the bound of jacobian! and those of the
# Fokker-Planck terms several times over, but on this small window the interface terms box
# fewer entries than the margin of their bound.
const JACOBIAN_ALLOCATION = 2.5e6           # 1.19e6
const PLASMA_COLUMNS_ALLOCATION = 1.6e5     # 7.8e4 (largest), 6.5e4 (f_re of the Fokker-Planck term)

# the largest relative residual of the algebraic rows `names` at the state `u` of `case`
function algebraic_mismatch(case, system, u, t, dt, names)
    residual = zeros(length(u))
    residual!(residual, u, u, system, t, dt)
    return maximum(maximum(abs(residual[i]) / abs(u[i])
                   for i in unknown_range(case.layout, unknown_id(case.layout, name), ir))
    for name in names, ir in 1:case.nr)
end

# the window from DREAM's state at its start, at `compat`
function run_window(start, compat)
    case = build(start; compat)
    u0 = initial_state(case, start)
    times = collect(start["t"])
    system = EquationSystem(case.layout, case.equations;
        time_scale = times[2] - times[1], initial_state = u0)
    solution = integrate(system, u0, times; stepper = FixedSteps())
    return case, system, u0, solution
end

@testset "the disruption case with a runaway grid against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the disruption case with a runaway grid against DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        file = reference_file("dream_disruption_runaway_small.h5")
        start = h5open(h5 -> read_dream_state(h5["window"]), file, "r")
        times = collect(start["t"])
        @test haskey(start, "f_re")
        case, system, u0, solution = run_window(start, :dream)
        # DREAM's first state: the runaway current is the moment of f_re, and n_re holds ∫f_re
        # and the runaways beyond the grid's p_max, 3e-6 and 8e-6 of n_re at the two radii
        @test algebraic_mismatch(case, system, u0, times[1], times[2] - times[1],
            (:runaway_current_density,)) < 1e-13   # 1.1e-15
        n_re = collect(unknown_values(u0, case.layout, unknown_id(case.layout, :runaway_density)))
        on_grid = density(case.runaway_grid,
            kinetic_array(u0, case.layout, unknown_id(case.layout, :f_re)))
        @test all(0 .< (n_re .- on_grid) ./ n_re .< 1e-4)
        @test all(solution.statistics.accepted)
        errors = phase_differences(case, solution, start)
        @test Set(keys(WINDOW_TOLERANCES)) == Set(keys(errors))
        for name in sort!(collect(keys(errors)))
            @info "the disruption case with a runaway grid against DREAM: $name $(errors[name])"
            @test errors[name] < WINDOW_TOLERANCES[name]
        end
        for (_, value, bound) in budget_checks(case, solution, "window, compat = dream", DRIFT["quench"])
            @test value < bound
        end
        for (_, value, bound) in runaway_budget_checks(
            case, solution, "window, compat = dream",
            WINDOW_RUNAWAY_BUDGETS)
            @test value < bound
        end
        # the plasma columns insert their partials without a box per entry
        J = jacobian_prototype(system)
        jacobian!(J, system, u0, times[1])
        allocated = @allocated jacobian!(J, system, u0, times[1])
        @info "the disruption case with a runaway grid against DREAM: jacobian! $allocated B"
        @test allocated < JACOBIAN_ALLOCATION skip = !OPTIMIZED
        @test sort!([string(nameof(typeof(b.term))) for b in system.column_terms]) ==
              ["FokkerPlanckTerm", "FokkerPlanckTerm", "InterfaceParticleFluxTerm",
            "MomentumInterfaceTerm", "MomentumInterfaceTerm"]
        for b in system.column_terms
            Runes2P.add_plasma_columns!(J, b.term, u0, case.layout, b.row, times[1])
            allocated = @allocated Runes2P.add_plasma_columns!(
                J, b.term, u0, case.layout, b.row, times[1])
            @info "the disruption case with a runaway grid against DREAM: plasma columns of the $(nameof(typeof(b.term))) of $(case.layout.unknowns[b.row].name) $allocated B"
            @test allocated < PLASMA_COLUMNS_ALLOCATION skip = !OPTIMIZED
        end
        # Runes2P's defaults: the budgets alone, since the physics departs from DREAM's
        case, _, _, solution = run_window(start, :runes2p)
        @test all(solution.statistics.accepted)
        for (_, value, bound) in budget_checks(case, solution, "window, compat = runes2p", DRIFT["quench"])
            @test value < bound
        end
        for (_, value, bound) in runaway_budget_checks(
            case, solution, "window, compat = runes2p",
            WINDOW_RUNAWAY_BUDGETS)
            @test value < bound
        end
    end
end
