# The coupled system of the disruption case against DREAM, at the size
# of a test: DREAM's examples/Disruptions/generate.py --kinetic --cylindrical, scenario 1, on 2
# radii and a hot-tail grid of 30 × 4 cells, 10 steps of 4e-8 s of the ionization phase and 10
# of 1e-7 s of the quench (reference/dream/cases/disruption_kinetic.py --nr 2 --np1 10 --np2 20
# --nxi 4 --nt-ioniz 10 --t-ioniz 4e-7 --nt-quench 10 --t-quench 1.4e-6,
# dream_disruption_small.h5). Each phase starts from DREAM's state at its start and runs the system
# of example_disruption.jl with DREAM's settings: every term of the case
# together, the field and the current, the temperature and the ion energies, the charge states
# with the kinetic ionization, the particle source and both avalanche terms. The runaway density
# and current are compared per radius: in the quench the external avalanche is on its positive
# branch, the matched rate, at the second radius, where n_re is 2e-15 of the first radius.
# DREAM's state at t = 0 of that radius has three to four significant bits (n_re = 7/1024
# m⁻³ in both phases, j_re 0.95 and 1.04 times e c n_re), far below DREAM's absolute tolerance
# of n_re; both codes start from it, and the comparison runs from the second time.
# The budgets of the assembled system are asserted independently of DREAM (`budgets` of
# example_disruption.jl), and both phases run once more with Runes2P's defaults (compat = runes2p),
# where only the budgets are checked. Needs the ADF11 cache.
using HDF5
include(joinpath(@__DIR__, "example_disruption.jl"))

# the largest relative difference of each quantity over the steps of both phases: about ten
# times the larger of the two measured at the default optimisation level (in the comment),
# so that a change of the method shows while a change of the rounding does not
#! format: off
const TOLERANCES = Dict("f_hot" => 5e-4,      # 4.4e-5
                        "j_tot" => 2e-11,     # 1.9e-12
                        "n_hot" => 2e-6,      # 1.9e-7
                        "E_field" => 5e-6,    # 4.1e-7
                        "j_hot" => 5e-7,      # 4.6e-8
                        "n_i" => 2e-7,        # 1.5e-8
                        "j_ohm" => 1e-7,      # 1.0e-8
                        "n_cold" => 5e-8,     # 4.1e-9
                        "T_cold" => 2e-8,     # 1.7e-9
                        "W_cold" => 1e-9,     # 7.4e-11
                        "W_i" => 3e-10,       # 2.4e-11
                        "n_re" => 2e-10,      # 1.5e-11 at the second radius, per radius (5.5e-12 at the first), at -O0
                        "j_re" => 2e-10,      # 1.4e-11 at the second radius, per radius (5.5e-12 at the first), at -O0
                        "I_p" => 2e-11,       # 1.3e-12
                        "psi_p" => 7e-12,     # 6.8e-13
                        "n_tot" => 1e-14)     # 5.0e-16, round-off of the quasi-neutral sum
#! format: on

@testset "the disruption case against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the disruption case against DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        file = reference_file("dream_disruption_small.h5")
        for phase in ("ionization", "quench")
            start = h5open(h5 -> read_dream_state(h5[phase]), file, "r")
            case = build(start)
            u0 = initial_state(case, start)
            times = collect(start["t"])
            system = EquationSystem(case.layout, case.equations;
                time_scale = times[2] - times[1], initial_state = u0)
            solution = integrate(system, u0, times; stepper = FixedSteps())
            @test all(solution.statistics.accepted)
            errors = phase_differences(case, solution, start)
            for name in sort!(collect(keys(errors)))
                @info "the disruption case against DREAM, phase $phase: $name $(errors[name])"
                @test errors[name] < TOLERANCES[name]
            end
            for (_, value, bound) in budget_checks(case, solution, "$phase, compat = dream", DRIFT[phase])
                @test value < bound
            end
        end
        # Runes2P's defaults: the budgets alone, since the physics departs from DREAM's
        for phase in ("ionization", "quench")
            start = h5open(h5 -> read_dream_state(h5[phase]), file, "r")
            case = build(start; compat = :runes2p)
            u0 = initial_state(case, start)
            times = collect(start["t"])
            system = EquationSystem(case.layout, case.equations;
                time_scale = times[2] - times[1], initial_state = u0)
            solution = integrate(system, u0, times; stepper = FixedSteps())
            @test all(solution.statistics.accepted)
            for (_, value, bound) in budget_checks(case, solution, "$phase, compat = runes2p", DRIFT[phase])
                @test value < bound
            end
        end
    end
end
