# The disruption case end to end against DREAM, at the size of test_disruption_dream.jl: DREAM's
# chain of examples/Disruptions/generate.py --kinetic --cylindrical, scenario 1, runs 0 to 4
# (reference/dream/cases/disruption_kinetic.py --nr 2 --np1 10 --np2 20 --nxi 4 --nt-ioniz 10
# --t-ioniz 4e-7 --nt-quench 10 --t-quench 1.4e-6, dream_disruption_chain_small.h5). Runes2P runs
# the chain on its own: its initialization (`initialize` of example_disruption.jl, compared in
# test_disruption_initialization_dream.jl), the ionization phase from the state that the
# initialization hands over with the ions of the settings, and the quench from Runes2P's own end
# state of the ionization phase. Each phase is compared with DREAM's run over its steps, as in
# test_disruption_dream.jl, where each phase starts from DREAM's state; the differences of the
# initialization carry into the phases. The runaway density and current are compared over the
# radii, and per radius at the first radius. At the second radius, where n_re is 2e-15 of the
# first, DREAM's state at the start of the ionization phase holds n_re = 7/1024 m⁻³ where the end
# of its initialization has 6.66e-3, three to four significant bits (see test_disruption_dream.jl),
# and the chain carries Runes2P's value, 2.6e-2 apart. There the growth of n_re over the quench, the
# positive branch of the external avalanche, is compared instead; its change over the ionization
# phase, 4e-7 of its value, and j_re, which DREAM's state there holds at 0.95 to 1.04 times
# e c n_re, are below what the rounded start resolves. With Runes2P's defaults (compat = runes2p)
# the chain runs as well, from its own initialization, and the budgets are asserted. Needs the
# ADF11 cache.
using HDF5
include(joinpath(@__DIR__, "example_disruption.jl"))

# the largest relative difference of each quantity over the steps of both phases of the chain:
# about ten times the larger of the two measured at the default optimisation level (in the
# comment)
#! format: off
const CHAIN_TOLERANCES = Dict("f_hot" => 5e-4,      # 4.4e-5
                              "E_field" => 5e-6,    # 4.1e-7
                              "n_hot" => 2e-6,      # 1.9e-7
                              "j_hot" => 5e-7,      # 4.9e-8
                              "j_ohm" => 2e-7,      # 1.3e-8
                              "n_i" => 2e-7,        # 1.9e-8
                              "n_cold" => 5e-8,     # 4.9e-9
                              "T_cold" => 2e-8,     # 1.8e-9
                              "psi_p" => 3e-9,      # 2.8e-10, from the initialization
                              "W_cold" => 2e-9,     # 1.6e-10
                              "n_re" => 5e-10,      # 3.6e-11, over the radii
                              "j_re" => 5e-10,      # 3.6e-11, over the radii
                              "W_i" => 5e-10,       # 4.8e-11
                              "I_p" => 2e-10,       # 1.3e-11, from the initialization
                              "n_tot" => 1e-14)     # 3.8e-16, round-off of the quasi-neutral sum
# n_re and j_re at the first radius, per radius as in test_disruption_dream.jl (3.6e-11 measured),
# and the growth of n_re over the quench at the second radius (1.9e-3)
const FIRST_RADIUS = 5e-10
const GROWTH = 2e-2
#! format: on

# the chain on the grids and the times of DREAM's runs: the initialization, then the two phases,
# each from the end state of the run before it. Returns the initialization (`initialize`) and
# (phase => (; case, solution)) in the order of the chain.
function run_chain(dream; compat)
    init = initialize(dream["init"]; compat)
    phases = Pair{String, Any}[]
    u0 = nothing
    for phase in ("ionization", "quench")
        start = dream[phase]
        case = build(start; compat)
        if phase == "ionization"
            u0 = initial_state_after_initialization(case, init.second.case,
                init.second.solution.states[:, end])
        end
        times = collect(start["t"])
        system = EquationSystem(case.layout, case.equations;
            time_scale = times[2] - times[1], initial_state = u0)
        solution = integrate(system, u0, times; stepper = FixedSteps())
        push!(phases, phase => (; case, solution))
        u0 = solution.states[:, end]
    end
    return (; init, phases)
end

# the runaway density and current per radius at the radius `ir`, relative to DREAM's largest
# value there over the phase
function runaways_at(case, solution, dream, ir)
    layout = case.layout
    errors = Dict{String, Float64}()
    for (name, ours) in ("n_re" => :runaway_density, "j_re" => :runaway_current_density)
        id = unknown_id(layout, ours)
        errors[name] = maximum(abs(unknown_values(solution.states[:, m], layout, id)[ir] -
                                   dream[name][ir, m]) / maximum(abs, dream[name][ir, :])
        for m in 2:length(solution.times))
    end
    return errors
end

# the growth of the runaway density over the phase at the radius `ir`, against DREAM's, relative
# to DREAM's largest growth
function runaway_growth(case, solution, dream, ir)
    layout = case.layout
    id = unknown_id(layout, :runaway_density)
    ours = [unknown_values(state, layout, id)[ir] for state in eachcol(solution.states)]
    theirs = dream["n_re"][ir, :]
    return maximum(abs, (ours .- ours[1]) .- (theirs .- theirs[1])) /
           maximum(abs, theirs .- theirs[1])
end

@testset "the disruption case end to end against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the disruption chain against DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        file = reference_file("dream_disruption_chain_small.h5")
        dream = Dict(run => h5open(h5 -> read_dream_state(h5[run]), file, "r")
        for run in ("init", "ionization", "quench"))
        chain = run_chain(dream; compat = :dream)
        for (phase, (; case, solution)) in chain.phases
            @test all(solution.statistics.accepted)
            errors = phase_differences(case, solution, dream[phase]; per_radius = ())
            for name in sort!(collect(keys(errors)))
                @info "the disruption chain against DREAM, phase $phase: $name $(errors[name])"
                @test errors[name] < CHAIN_TOLERANCES[name]
            end
            for (name, error) in runaways_at(case, solution, dream[phase], 1)
                @info "the disruption chain against DREAM, phase $phase: $name at the first radius $error"
                @test error < FIRST_RADIUS
            end
            if phase == "quench"
                growth = runaway_growth(case, solution, dream[phase], 2)
                @info "the disruption chain against DREAM, quench: growth of n_re at the second radius $growth"
                @test growth < GROWTH
            end
            for (_, value, bound) in budget_checks(
                case, solution, "chain, $phase, compat = dream", DRIFT[phase])
                @test value < bound
            end
        end
        # Runes2P's defaults: the budgets alone, since the physics departs from DREAM's
        chain = run_chain(dream; compat = :runes2p)
        @test all(chain.init.first.solution.statistics.accepted)
        @test all(chain.init.second.solution.statistics.accepted)
        for (phase, (; case, solution)) in chain.phases
            @test all(solution.statistics.accepted)
            for (_, value, bound) in budget_checks(
                case, solution, "chain, $phase, compat = runes2p", DRIFT[phase])
                @test value < bound
            end
        end
    end
end
