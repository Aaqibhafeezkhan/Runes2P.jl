# The heating of the cold electrons by a runaway grid in a whole system against DREAM: the case
# `heating` of reference/dream/cases/initial_state.py (dream_runaway_heating.h5, set
# dream-v26.5-initial-state), the runaway grid of DREAM's examples/init_f_re with T_cold
# self-consistent, built by `runaway_heating_example` (example_cases.jl) and run over the 20
# steps of the case. At compat = dream T_cold, W_cold and f_re are compared with DREAM's; at
# compat = runes2p the two forms of the heating are compared over the run. Needs the ADF11 cache.
#
# Two differences of method remain at compat = dream, each checked by its mechanism:
# - DREAM's semi-implicit steps (LINEAR_IMPLICIT) take j_ohm of the start of the step as the
#   coefficient of the ohmic heating and the field as its unknown
#   (include/DREAM/Equations/Fluid/OhmicHeatingTerm.hpp:10-14, DREAM v26.5), and that j_ohm holds
#   the conductivity of the step before: the heating of the step from state m is σ(T_{m-1}) E²,
#   two states behind the end of the step (the saved Tcold_ohmic is σ(T_m) E²). Runes2P's
#   Newton steps take σ(T_{m+1}) E², which raises T_cold by about 1.4e-6 of its value per step.
#   The Fokker-Planck term of f_re takes the plasma of the start of the step (lagged = :all), as
#   DREAM's; the other terms have no lag. The test compares T_cold with DREAM's steps replayed
#   with the ohmic heating at the end of the step; the rest is the lower f_re (below) and the
#   heating by f_re at the end of the step.
# - The operator of Runes2P carries electrons out through p = 0 with the friction
#   lim p³ν_s (src/operators/advection_diffusion.jl:487-495, src/operators/kinetic_terms.jl:77-78,
#   p3_slowing_down_frequency_at_zero). DREAM computes the same limit
#   (src/Equations/Kinetic/SlowingDownTerm.cpp:54-56, fvm/Equation/AdvectionTerm.set.cpp:98-105),
#   but multiplies it by the interpolation coefficients of the face p = 0, which its Dirichlet
#   condition sets to zero (fvm/Equation/AdvectionTerm.cpp:36,
#   fvm/Equation/AdvectionInterpolationCoefficient.cpp:438-442): its density of f_re stays 1e16
#   to round-off, while Runes2P's falls by 1.5e-4 per step. The test checks the cause on the
#   first step of f_re alone and the loss of the run against Runes2P's flux through p = 0.
using HDF5

# the largest differences relative to DREAM's value, about ten times the value measured (in the
# comment)
const HEATING_RUN_TEMPERATURE = 3e-4    # 2.6e-5 of T_cold and 2.7e-5 of W_cold at the last step
const HEATING_RUN_REPLAY = 1e-5         # 1.1e-6 of T_cold from the replay of DREAM's steps
const HEATING_RUN_DISTRIBUTION = 3e-2   # 2.9e-3 of the largest f_re, at the last step
const HEATING_RUN_FIRST_STEP = 1e-14    # 2.4e-16 without the flux through p = 0, round-off
const HEATING_RUN_LOSS = 1e-12          # 1.2e-13 of the loss, the density lost against the flux through p = 0

@testset "the heating by a runaway grid in a run against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the run against DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        data = h5open(reference_file("dream_runaway_heating.h5"), "r") do h5
            group = h5["heating"]
            Dict(name => read(group[name])
            for name in ("grid/t", "grid/runaway/p1_f", "grid/runaway/p2_f", "eqsys/T_cold",
                "eqsys/W_cold", "eqsys/n_cold", "eqsys/f_re", "other/fluid/Tcold_ohmic",
                "other/fluid/Tcold_radiation", "other/fluid/Tcold_fre_coll"))
        end
        times = data["grid/t"]
        f_dream = data["eqsys/f_re"]   # (p, ξ, radius, time)
        T_dream = data["eqsys/T_cold"][1, :]
        W_dream = data["eqsys/W_cold"][1, :]
        steps = length(times) - 1
        value(u, layout, name) = only(unknown_values(u, layout, unknown_id(layout, name)))

        @testset "compat = dream" begin
            case = runaway_heating_example()
            (; layout, grid) = case
            m = momentum_grid(grid)
            @test m.p_faces == data["grid/runaway/p1_f"]
            @test m.xi_faces ≈ data["grid/runaway/p2_f"] rtol = 1e-14
            @test case.times ≈ times rtol = 1e-14
            id = unknown_id(layout, :runaway_distribution)
            distribution(u) = kinetic_view(u, layout, id, 1)
            largest_difference(u, k) = maximum(abs, distribution(u) .-
                                                    f_dream[:, :, 1, k]) /
                                       maximum(abs, f_dream[:, :, 1, k])
            @test largest_difference(case.initial_state, 1) < 2e-15

            solution = integrate(case.system, case.initial_state, times)
            @test all(solution.statistics.accepted)
            states = solution.states
            T = [value(states[:, k], layout, :cold_electron_temperature)
                 for k in eachindex(times)]
            W = [value(states[:, k], layout, :cold_thermal_energy)
                 for k in eachindex(times)]
            temperature = maximum(abs, (T .- T_dream) ./ T_dream)
            energy = maximum(abs, (W .- W_dream) ./ W_dream)
            distributions = [largest_difference(states[:, k], k) for k in eachindex(times)]
            @info "the run against DREAM: T_cold $temperature, W_cold $energy, f_re $(maximum(distributions))"
            @test temperature < HEATING_RUN_TEMPERATURE
            @test energy < HEATING_RUN_TEMPERATURE
            @test maximum(distributions) < HEATING_RUN_DISTRIBUTION

            # DREAM's steps replayed with the ohmic heating at the end of the step, as Runes2P's,
            # and DREAM's radiation and heating by f_re of the step (saved after it, the sum of a
            # row of DREAM is zero)
            plasma_at(temperature) = PlasmaState(;
                cold_electron_temperature = [temperature],
                ions = case.prescribed.ions, electric_field = [0.0], magnetic_field = [5.0])
            field = value(case.initial_state, layout, :electric_field)
            losses = data["other/fluid/Tcold_radiation"][1, :] .+
                     data["other/fluid/Tcold_fre_coll"][1, :]
            capacity = THERMAL_ENERGY_FACTOR * data["eqsys/n_cold"][1, 1]
            replayed = copy(T_dream)
            for k in 1:steps
                dt = times[k + 1] - times[k]
                x = replayed[k]
                for _ in 1:20
                    ohmic = braams_karney_conductivity(plasma_at(x), 1) * field^2
                    x = replayed[k] + dt * (ohmic - losses[k]) / capacity
                end
                replayed[k + 1] = x
            end
            replay = maximum(abs, (T .- replayed) ./ T_dream)
            @info "T_cold against DREAM's steps with the ohmic heating at the end of the step: $replay"
            @test replay < HEATING_RUN_REPLAY

            # the cause of the difference in f_re: the first step of f_re alone, with the
            # coefficients at the plasma of the start as DREAM's step takes them, matches DREAM's
            # without the flux through p = 0
            first_step = map((true, false)) do sink
                plasma = PlasmaState(; cold_electron_temperature = [T_dream[1]],
                    cold_electron_density = [data["eqsys/n_cold"][1, 1]],
                    ions = case.prescribed.ions, electric_field = [field],
                    magnetic_field = [5.0])
                operator = fokker_planck_operator(
                    grid, plasma, case.fokker_planck.settings;
                    synchrotron = false, scheme = :upwind, p_upper_boundary = :constant_flux)
                if !sink
                    operator.advection_p_times_p2_at_zero .= 0
                end
                alone = UnknownLayout(grid.radial_grid,
                    [KineticUnknown(:runaway_distribution, grid; scale = 1e16)])
                system = EquationSystem(alone,
                    [Equation(:runaway_distribution, TimeDerivative(),
                        FokkerPlanckTerm(:runaway_distribution, operator))])
                u0 = zeros(length(alone))
                kinetic_view(u0, alone, 1, 1) .= f_dream[:, :, 1, 1]
                u = integrate(system, u0, times[1:2]).states[:, 2]
                maximum(abs, kinetic_view(u, alone, 1, 1) .- f_dream[:, :, 1, 2]) /
                maximum(abs, f_dream[:, :, 1, 2])
            end
            # a test of the cause, not an option of Runes2P: the coefficient of the flux through
            # p = 0 set to zero by hand
            @info "the first step of f_re against DREAM, with and without the flux through p = 0: $first_step"
            @test first_step[1] ≈ distributions[2] rtol = 1e-6
            @test first_step[2] < HEATING_RUN_FIRST_STEP

            # the electrons that Runes2P's run loses are those that leave through p = 0 (and
            # p_max), with the coefficients of each step at the plasma of its start, as the
            # lagged Fokker-Planck term takes them
            densities = [density(grid, reshape(Array(distribution(states[:, k])), size(grid)))[1]
                         for k in eachindex(times)]
            through_boundaries = zeros(length(times))
            for k in 1:steps
                u = states[:, k]
                plasma = PlasmaState(;
                    cold_electron_temperature = [T[k]],
                    cold_electron_density = [value(u, layout, :cold_electron_density)],
                    ions = case.prescribed.ions, electric_field = [field], magnetic_field = [5.0])
                operator = fokker_planck_operator(
                    grid, plasma, case.fokker_planck.settings;
                    synchrotron = false, scheme = :upwind, p_upper_boundary = :constant_flux)
                f = reshape(Array(distribution(states[:, k + 1])), size(grid))
                net = momentum_flux(operator, f, 1) -
                      momentum_flux(operator, f, momentum_cell_count(grid) + 1)
                through_boundaries[k + 1] = through_boundaries[k] +
                                            (times[k + 1] - times[k]) * net
            end
            loss = densities .- densities[1]
            @info "the density of f_re lost in the run: $(loss[end] / densities[1]) of the start, " *
                  "the flux through the boundaries: $(through_boundaries[end] / densities[1])"
            mismatch = maximum(abs, loss .- through_boundaries) / abs(loss[end])
            @info "the density lost against the flux through the boundaries: $mismatch"
            @test loss[end] < -1e-3 * densities[1]
            @test mismatch < HEATING_RUN_LOSS
        end

        @testset "compat = runes2p" begin
            case = runaway_heating_example(; compat = :runes2p)
            (; layout, grid) = case
            solution = integrate(case.system, case.initial_state, times)
            @test all(solution.statistics.accepted)
            row = unknown_id(layout, :cold_electron_temperature)
            moment = FrictionEnergyTransferTerm(
                :runaway_distribution, grid, PlasmaFromUnknowns(case.prescribed))
            heating(term, k) = only(unknown_values(
                Runes2P.rhs!(zeros(size(solution.states, 1)), term, solution.states[:, k],
                    layout, row, times[k]),
                layout, row))
            energy_change = [heating(case.heating, k) for k in eachindex(times)]
            moments = [heating(moment, k) for k in eachindex(times)]
            @info "the heating by f_re at compat = runes2p over the run [W/m³]: the energy change " *
                  "$(extrema(energy_change)), DREAM's moment $(extrema(moments))"
            T = [value(solution.states[:, k], layout, :cold_electron_temperature)
                 for k in eachindex(times)]
            @info "T_cold at compat = runes2p against DREAM's: $(maximum(abs, (T .- T_dream) ./ T_dream))"
            # Measured: the energy change of the operator rises from 0 to 472 W/m³, DREAM's
            # moment on the same states is 7.65e4 to 7.81e4 W/m³. Nearly all of f_re sits in
            # the first momentum cell, below which there is no cell for the friction to carry it
            # to, so the energy change of the operator is small; the moment counts the friction
            # of that cell in full, an energy that f_re does not lose. T_cold ends 2.9e-4 below
            # DREAM's. The run takes the superthermal frequencies, since the electron-electron
            # operator has no ultra-relativistic form (runaway_heating_example).
            @test energy_change[1] == 0
            @test all(>=(0), energy_change)
            @test maximum(energy_change) < 1e-2 * minimum(moments)
            @test all(moments .> 7e4)
        end
    end
end
