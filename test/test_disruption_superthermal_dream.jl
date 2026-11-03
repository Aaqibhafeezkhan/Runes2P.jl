# The disruption case in DREAM's superthermal mode against DREAM, end to end, at the size of
# test_disruption_dream.jl: DREAM's chain of examples/Disruptions/generate.py --superthermal
# --cylindrical, scenario 1, runs 0 to 4 (reference/dream/cases/disruption_kinetic.py --mode
# superthermal --nr 2 --np2 20 --nxi 4 --nt-ioniz 10 --t-ioniz 4e-7 --nt-quench 10 --t-quench
# 1.4e-6, dream_disruption_superthermal_small.h5). The system of example_disruption.jl with
# mode = :superthermal: the initialization with the full collision frequencies from a Maxwellian
# of 0.9999 of the free electrons, without DREAM's corrected conductivity; then the phases with the
# superthermal frequencies, all of f as hot electrons, Ohm's law for the cold electrons, no particle
# source, the heating of the cold electrons by the friction of f (FrictionEnergyTransferTerm), and
# the cold temperature restarted at 1 eV. Each run of the initialization and each phase is compared
# with DREAM's over its steps, and the state handed to the ionization phase with DREAM's first
# state of that phase. The runaway density and current are compared over the radii: at the second
# radius, where n_re is 4e-7 of the first, they differ by 0.22 at the end of run 1 (0.23 m⁻³
# there, from the far tail of f) and by 4e-4 from run 2 on. With
# Runes2P's defaults (compat = runes2p) the chain runs as well and the budgets are asserted; the
# drift of ∫f − n_cold − n_hot is not checked, since the superthermal mode has no particle source
# that would keep it fixed. Needs the ADF11 cache.
using HDF5
include(joinpath(@__DIR__, "example_disruption.jl"))

# the largest relative difference of each quantity over the steps of runs 1 and 2, about ten
# times the larger of the two measured at the default optimisation level (in the comment)
#! format: off
const SUPERTHERMAL_INITIALIZATION = Dict("E_field" => 2e-8,    # 1.3e-9, run 2 (the rescale)
                                         "I_p" => 2e-8,        # 1.3e-9, run 1
                                         "j_ohm" => 2e-8,      # 1.5e-9
                                         "j_hot" => 2e-8,      # 1.5e-9
                                         "psi_p" => 1e-8,      # 9.0e-10
                                         "n_re" => 5e-9,       # 4.0e-10
                                         "j_re" => 5e-9,       # 4.0e-10
                                         "n_hot" => 2e-9,      # 1.4e-10
                                         "f_hot" => 2e-10,     # 1.9e-11
                                         "n_cold" => 5e-13,    # 3.2e-14
                                         "W_cold" => 5e-14,    # 3.1e-15
                                         "T_cold" => 1e-14,    # round-off of the prescribed profile
                                         "n_i" => 1e-13,       # round-off of the D background
                                         "n_tot" => 1e-14)     # 0
# the same over the steps of both phases
const SUPERTHERMAL_PHASES = Dict("n_i" => 5e-5,       # 3.5e-6
                                 "n_cold" => 3e-5,    # 2.9e-6
                                 "W_cold" => 3e-5,    # 2.8e-6
                                 "j_ohm" => 1e-5,     # 9.8e-7
                                 "T_cold" => 1e-5,    # 6.4e-7
                                 "f_hot" => 1e-5,     # 6.1e-7
                                 "n_hot" => 3e-6,     # 2.4e-7
                                 "E_field" => 2e-6,   # 1.2e-7
                                 "j_hot" => 2e-6,     # 1.8e-7
                                 "W_i" => 3e-7,       # 2.7e-8
                                 "psi_p" => 5e-9,     # 4.4e-10, from the initialization
                                 "n_re" => 2e-9,      # 1.7e-10
                                 "j_re" => 2e-9,      # 1.7e-10
                                 "I_p" => 1e-11,      # 8.9e-13
                                 "n_tot" => 1e-14)    # 3.8e-16, round-off of the quasi-neutral sum
#! format: on
# the state handed to the ionization phase against DREAM's first state of that phase, per unknown
# relative to its largest value: 1.3e-8 measured (the cold density, the free electrons less those of
# f, 1e-4 of them)
const SUPERTHERMAL_HANDOVER = 2e-7

# the superthermal chain on the grids and the times of DREAM's runs
function run_superthermal_chain(dream; compat)
    init = initialize(dream["init"]; compat, mode = :superthermal)
    phases = Pair{String, Any}[]
    u0 = nothing
    for phase in ("ionization", "quench")
        start = dream[phase]
        case = build(start; compat, mode = :superthermal)
        if phase == "ionization"
            u0 = initial_state_after_initialization(case, init.second.case,
                init.second.solution.states[:, end])
        end
        times = collect(start["t"])
        system = EquationSystem(case.layout, case.equations;
            time_scale = times[2] - times[1], initial_state = u0)
        solution = integrate(system, u0, times; stepper = FixedSteps())
        push!(phases, phase => (; case, solution, start = u0))
        u0 = solution.states[:, end]
    end
    return (; init, phases)
end

@testset "the disruption case in the superthermal mode against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the superthermal case against DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        file = reference_file("dream_disruption_superthermal_small.h5")
        dream = Dict(run => h5open(h5 -> read_dream_state(h5[run]), file, "r")
        for run in ("init", "init_rescaled", "ionization", "quench"))
        chain = run_superthermal_chain(dream; compat = :dream)
        for (name, run) in (("init", chain.init.first), (
            "init_rescaled", chain.init.second))
            @test all(run.solution.statistics.accepted)
            @test run.solution.times == dream[name]["t"]
            errors = phase_differences(run.case, run.solution, dream[name]; per_radius = ())
            for quantity in sort!(collect(keys(errors)))
                @info "the superthermal case against DREAM, run $name: $quantity $(errors[quantity])"
                @test errors[quantity] < SUPERTHERMAL_INITIALIZATION[quantity]
            end
        end
        for (phase, (; case, solution, start)) in chain.phases
            @test all(solution.statistics.accepted)
            if phase == "ionization"
                theirs = initial_state(case, dream[phase])
                for unknown in case.layout.unknowns
                    entries = _entries(case.layout, unknown.name)
                    reference = maximum(abs, theirs[entries])
                    difference = maximum(abs, start[entries] .- theirs[entries])
                    relative = difference / max(reference, floatmin())
                    @info "the superthermal hand-over to the ionization phase: $(unknown.name) $relative"
                    @test difference <= SUPERTHERMAL_HANDOVER * reference
                end
            end
            errors = phase_differences(case, solution, dream[phase]; per_radius = ())
            for name in sort!(collect(keys(errors)))
                @info "the superthermal case against DREAM, phase $phase: $name $(errors[name])"
                @test errors[name] < SUPERTHERMAL_PHASES[name]
            end
            for (_, value, bound) in budget_checks(case, solution,
                "superthermal, $phase, compat = dream", nothing)
                @test value < bound
            end
        end
        # Runes2P's defaults: the budgets alone, since the physics departs from DREAM's
        chain = run_superthermal_chain(dream; compat = :runes2p)
        @test all(chain.init.first.solution.statistics.accepted)
        @test all(chain.init.second.solution.statistics.accepted)
        for (phase, (; case, solution)) in chain.phases
            @test all(solution.statistics.accepted)
            for (_, value, bound) in budget_checks(case, solution,
                "superthermal, $phase, compat = runes2p", nothing)
                @test value < bound
            end
        end
    end
end
