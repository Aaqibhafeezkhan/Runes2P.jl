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

# the closures of the assembled system (`budgets`), independent of DREAM: round-off bounds, at
# least 50 times the values measured at -O0 (compat = dream / runes2p, ionization and quench)
const BUDGETS = (species=1e-13,     # 3.2e-16 / 4.7e-16, 3.2e-16 / 3.2e-16
                 exchange=1e-13,    # 2.7e-15 / 1.3e-15, 5.1e-16 / 1.3e-15
                 diffusion=1e-13,   # 7.0e-17 / 7.3e-17, 0 / 7.3e-17
                 closure=1e-12,     # 7.9e-15 / 1.1e-14, 4.4e-15 / 4.2e-15, the row of W_cold against its channels
                 field=1e-8)        # 1.8e-10 / 1.2e-11, 2.7e-12 / 1.3e-11 of a change that is 3e-6 and 3e-5 of the energy
#! format: on
# the drift of ∫f − n_cold − n_hot, relative to n_cold, per phase: the deposition of the ionization
# phase moves it at the Newton level of the particle-source row (2.7e-8 measured); in the quench it
# is round-off (7.9e-16 / 6.9e-16), and the bound pins the knock-on number of the kinetic source
# against the ProductTerm of the particle source, 3e-9 of n_cold over the phase, to 3e-5
const DRIFT = Dict("ionization" => 1e-6, "quench" => 1e-13)

# the budgets of the assembled system against their bounds, the drift against the bound of the phase
function check_budgets(case, solution, label, drift_bound)
    b = budgets(case, solution)
    for name in (:species, :exchange, :diffusion, :closure, :field)
        @info "the disruption case, $label: $name $(getfield(b, name))"
        @test getfield(b, name) < getfield(BUDGETS, name)
    end
    @info "the disruption case, $label: drift $(b.drift)"
    @test b.drift < drift_bound
    # the magnetic energy falls over the phase: the field works on the currents
    @test b.field_energy_change < 0
    return nothing
end

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
            layout = case.layout
            # the largest difference over the radii and the steps, relative to the largest value of
            # DREAM over the radii at each step
            #! format: off
            difference(name, ours) = maximum(maximum(abs, unknown_values(solution.states[:, m], layout, unknown_id(layout, ours)) .-
                                                     start[name][:, m]) / maximum(abs, start[name][:, m])
                                             for m in 2:length(times))
            # per radius, relative to the largest value of DREAM at that radius over the phase: the
            # runaway density differs by 15 orders of magnitude between the radii
            difference_per_radius(name, ours) = maximum(abs(unknown_values(solution.states[:, m], layout, unknown_id(layout, ours))[ir] -
                                                            start[name][ir, m]) / maximum(abs, start[name][ir, :])
                                                        for ir in 1:case.nr, m in 2:length(times))
            errors = Dict(name => difference(name, ours)
                          for (name, ours) in ("T_cold" => :cold_electron_temperature, "E_field" => :electric_field,
                                               "n_cold" => :cold_electron_density, "n_hot" => :hot_electron_density,
                                               "n_tot" => :total_electron_density,
                                               "j_ohm" => :ohmic_current_density, "j_hot" => :hot_electron_current_density,
                                               "psi_p" => :poloidal_flux, "W_cold" => :cold_thermal_energy))
            for (name, ours) in ("n_re" => :runaway_density, "j_re" => :runaway_current_density)
                errors[name] = difference_per_radius(name, ours)
            end
            f_id, ions_id = unknown_id(layout, :f), unknown_id(layout, :ion_densities)
            energy_id = unknown_id(layout, :ion_thermal_energy)
            errors["f_hot"] = maximum(maximum(abs, kinetic_view(solution.states[:, m], layout, f_id, ir) .- start["f_hot"][:, :, ir, m]) /
                                      maximum(abs, start["f_hot"][:, :, ir, m]) for ir in 1:case.nr, m in 2:length(times))
            #! format: on
            errors["n_i"] = maximum(maximum(abs,
                                        radial_view(solution.states[:, m], layout, ions_id, ir) .-
                                        start["n_i"][ir, :, m]) /
                                    maximum(abs, start["n_i"][ir, :, m])
            for ir in 1:case.nr, m in 2:length(times))
            errors["W_i"] = maximum(maximum(abs,
                                        radial_view(solution.states[:, m], layout, energy_id, ir) .-
                                        start["W_i"][ir, :, m]) /
                                    maximum(abs, start["W_i"][ir, :, m])
            for ir in 1:case.nr, m in 2:length(times))
            current = only(unknown_range(layout, unknown_id(layout, :plasma_current)))
            errors["I_p"] = maximum(abs(solution.states[current, m] - start["I_p"][m]) /
                                    abs(start["I_p"][m]) for m in 2:length(times))
            for name in sort!(collect(keys(errors)))
                @info "the disruption case against DREAM, phase $phase: $name $(errors[name])"
                @test errors[name] < TOLERANCES[name]
            end
            check_budgets(case, solution, "$phase, compat = dream", DRIFT[phase])
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
            check_budgets(case, solution, "$phase, compat = runes2p", DRIFT[phase])
        end
    end
end
