# The coupled system of the disruption case against DREAM, at the size
# of a test: DREAM's examples/Disruptions/generate.py --kinetic --cylindrical, scenario 1, on 2
# radii and a hot-tail grid of 30 × 4 cells, 10 steps of 4e-8 s of the ionization phase and 10
# of 1e-7 s of the quench (reference/dream/cases/disruption_kinetic.py --nr 2 --np1 10 --np2 20
# --nxi 4 --nt-ioniz 10 --t-ioniz 4e-7 --nt-quench 10 --t-quench 1.4e-6, test/data/
# dream_disruption_small.h5). Each phase starts from DREAM's state at its start and runs the system
# of example_disruption.jl with DREAM's settings: every term of the case
# together, the field and the current, the temperature and the ion energies, the charge states
# with the kinetic ionization, the particle source and both avalanche terms. Needs the ADF11 cache.
using HDF5
include(joinpath(@__DIR__, "example_disruption.jl"))

# the largest relative difference of each quantity over the steps of both phases: about ten
# times the larger of the two measured at the default optimisation level (2026-10-12, in the
# comment), so that a change of the method shows while a change of the rounding does not
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
                        "n_re" => 6e-11,      # 5.5e-12
                        "j_re" => 6e-11,      # 5.5e-12
                        "I_p" => 2e-11,       # 1.3e-12
                        "psi_p" => 7e-12,     # 6.8e-13
                        "n_tot" => 1e-14)     # 5.0e-16, round-off of the quasi-neutral sum

@testset "the disruption case against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the disruption case against DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        file = joinpath(@__DIR__, "data", "dream_disruption_small.h5")
        for phase in ("ionization", "quench")
            start = h5open(h5 -> read_dream_state(h5[phase]), file, "r")
            case = build(start)
            u0 = initial_state(case, start)
            times = collect(start["t"])
            system = EquationSystem(case.layout, case.equations; time_scale=times[2] - times[1], initial_state=u0)
            solution = integrate(system, u0, times; stepper=FixedSteps())
            @test all(solution.statistics.accepted)
            layout = case.layout
            # the largest difference over the radii and the steps, relative to the largest value of
            # DREAM over the radii at each step
            difference(name, ours) = maximum(maximum(abs, radial_values(solution.states[:, m], layout, unknown_id(layout, ours)) .-
                                                     start[name][:, m]) / maximum(abs, start[name][:, m])
                                             for m in 2:length(times))
            errors = Dict(name => difference(name, ours)
                          for (name, ours) in ("T_cold" => :cold_electron_temperature, "E_field" => :electric_field,
                                               "n_cold" => :cold_electron_density, "n_hot" => :hot_electron_density,
                                               "n_re" => :runaway_density, "n_tot" => :total_electron_density,
                                               "j_ohm" => :ohmic_current_density, "j_hot" => :hot_electron_current_density,
                                               "j_re" => :runaway_current_density, "psi_p" => :poloidal_flux,
                                               "W_cold" => :cold_thermal_energy))
            f_id, ions_id = unknown_id(layout, :f), unknown_id(layout, :ion_densities)
            energy_id = unknown_id(layout, :ion_thermal_energy)
            errors["f_hot"] = maximum(maximum(abs, kinetic_view(solution.states[:, m], layout, f_id, ir) .- start["f_hot"][:, :, ir, m]) /
                                      maximum(abs, start["f_hot"][:, :, ir, m]) for ir in 1:case.nr, m in 2:length(times))
            errors["n_i"] = maximum(maximum(abs, radial_view(solution.states[:, m], layout, ions_id, ir) .- start["n_i"][ir, :, m]) /
                                    maximum(abs, start["n_i"][ir, :, m]) for ir in 1:case.nr, m in 2:length(times))
            errors["W_i"] = maximum(maximum(abs, radial_view(solution.states[:, m], layout, energy_id, ir) .- start["W_i"][ir, :, m]) /
                                    maximum(abs, start["W_i"][ir, :, m]) for ir in 1:case.nr, m in 2:length(times))
            current = only(unknown_range(layout, unknown_id(layout, :plasma_current)))
            errors["I_p"] = maximum(abs(solution.states[current, m] - start["I_p"][m]) / abs(start["I_p"][m]) for m in 2:length(times))
            for name in sort!(collect(keys(errors)))
                @info "the disruption case against DREAM, phase $phase: $name $(errors[name])"
                @test errors[name] < TOLERANCES[name]
            end
        end
    end
end
