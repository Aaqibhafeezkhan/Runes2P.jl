# The heating of the cold electrons by a runaway grid in a whole system against DREAM: the
# case `heating` of reference/dream/cases/initial_state.py (dream_runaway_heating.h5, set
# dream-v26.5-initial-state), the runaway grid of DREAM's examples/init_f_re with T_cold
# self-consistent, built by `runaway_heating_example` (example_cases.jl) and run over the 20
# steps of the case. At compat = dream T_cold, W_cold and f_re are compared with DREAM's. At
# compat = runes2p the run differs from DREAM's in the collision mode of the operator of
# f_re, the lag of its plasma and the form of the heating (runaway_heating_example), and
# the two forms of the heating are compared over the run.
# Needs the ADF11 cache.
#
# The operator of f_re takes no flux through p = 0 at both compat values
# (p_zero_boundary = :zero_flux), as DREAM's runaway grid: DREAM computes the friction's
# limit lim p³ν_s (src/Equations/Kinetic/SlowingDownTerm.cpp:54-56,
# fvm/Equation/AdvectionTerm.set.cpp:98-118), but multiplies it by the interpolation
# coefficients of the face p = 0, which its Dirichlet condition sets to zero
# (fvm/Equation/AdvectionTerm.cpp:36, fvm/Equation/AdvectionInterpolationCoefficient.cpp:438-442).
# That condition holds on DREAM's runaway grid only: f_re has no internal boundary condition
# (src/Settings/Equations/f_re.cpp:87), while on the hot-tail grid the lower boundary of p
# is mirrored (src/Settings/Equations/distribution.cpp:234-237), so that f_hot does carry
# the flux. The operator's default (p_zero_boundary = :outflow, DREAM's hot-tail grid)
# carries electrons out through p = 0; the test measures its size on the first step of f_re
# alone.
#
# One difference of method remains at compat = dream, checked by its mechanism:
# DREAM's semi-implicit steps (LINEAR_IMPLICIT) take j_ohm of the start of the step as the
# coefficient of the ohmic heating and the field as its unknown
# (include/DREAM/Equations/Fluid/OhmicHeatingTerm.hpp:10-14, DREAM v26.5), and that j_ohm
# holds the conductivity of the step before: the heating of the step from state m is
# σ(T_{m-1}) E², two states behind the end of the step (the saved Tcold_ohmic is
# σ(T_m) E²). Runes2P's Newton steps take σ(T_{m+1}) E², which raises T_cold by about
# 1.4e-6 of its value per step. The Fokker-Planck term of f_re takes the plasma of the
# start of the step (lagged = :all), as DREAM's; the other terms have no lag. The test
# compares T_cold with DREAM's steps replayed with the ohmic heating at the end of the
# step; the rest is the heating by f_re at the end of the step.
using HDF5

# the largest differences relative to DREAM's value, about ten times the measured value
# (in the comment above each), or round-off
# 2.7e-5 of T_cold and 2.7e-5 of W_cold at the last step
const HEATING_RUN_TEMPERATURE = 3e-4
# 6.9e-9 of T_cold from the replay of DREAM's steps
const HEATING_RUN_REPLAY = 1e-7
# 9.8e-7 of the largest f_re, the largest over the steps
const HEATING_RUN_DISTRIBUTION = 1e-5
# 2.4e-16 without the flux through p = 0, round-off
const HEATING_RUN_FIRST_STEP = 1e-14
# 4.0e-16 (compat = dream) and 6.0e-16 (compat = runes2p) of the density of f_re at the
# start, round-off
const HEATING_RUN_DENSITY = 1e-14
# with the operator's default flux through p = 0 the first step departs from DREAM's by
# 1.06e-4 of the largest value and loses 1.55e-4 of the density: at least half of each
const HEATING_RUN_DEFAULT_FIRST_STEP = 5e-5

@testset "the heating by a runaway grid in a run against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): " *
              "the run against DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        data = h5open(reference_file("dream_runaway_heating.h5"), "r") do h5
            group = h5["heating"]
            Dict(name => read(group[name])
            for name in (
                "grid/t", "grid/runaway/p1_f", "grid/runaway/p2_f", "eqsys/T_cold",
                "eqsys/W_cold", "eqsys/n_cold", "eqsys/f_re", "other/fluid/Tcold_radiation",
                "other/fluid/Tcold_fre_coll"))
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
            @info "the run against DREAM: T_cold $temperature, W_cold $energy, " *
                  "f_re $(maximum(distributions))"
            @test temperature < HEATING_RUN_TEMPERATURE
            @test energy < HEATING_RUN_TEMPERATURE
            @test maximum(distributions) < HEATING_RUN_DISTRIBUTION

            # DREAM's steps replayed with the ohmic heating at the end of the step, as
            # Runes2P's, and DREAM's radiation and heating by f_re of the step (saved after
            # it, the sum of a row of DREAM is zero)
            plasma_at(temperature) = PlasmaState(;
                cold_electron_temperature = [temperature],
                ions = case.prescribed.ions, electric_field = [0.0],
                magnetic_field = case.prescribed.magnetic_field)
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
            @info "T_cold against DREAM's steps with the ohmic heating at the end of the " *
                  "step: $replay"
            @test replay < HEATING_RUN_REPLAY

            # the first step of f_re alone, with the coefficients at the plasma of the start
            # as DREAM's step takes them: DREAM's without the flux through p = 0, and the
            # departure of the operator's default (:outflow, for a hot-tail grid), which
            # carries electrons out there
            first_step = map((:zero_flux, :outflow)) do boundary
                plasma = PlasmaState(; cold_electron_temperature = [T_dream[1]],
                    cold_electron_density = [data["eqsys/n_cold"][1, 1]],
                    ions = case.prescribed.ions, electric_field = [field],
                    magnetic_field = case.prescribed.magnetic_field)
                operator = fokker_planck_operator(
                    grid, plasma, case.fokker_planck.settings; synchrotron = false,
                    scheme = :upwind, p_upper_boundary = :constant_flux,
                    p_zero_boundary = boundary)
                alone = UnknownLayout(grid.radial_grid,
                    [KineticUnknown(:runaway_distribution, grid; scale = 1e16)])
                system = EquationSystem(alone,
                    [Equation(:runaway_distribution, TimeDerivative(),
                        FokkerPlanckTerm(:runaway_distribution, operator))])
                u0 = zeros(length(alone))
                kinetic_view(u0, alone, 1, 1) .= f_dream[:, :, 1, 1]
                u = integrate(system, u0, times[1:2]).states[:, 2]
                f_start = reshape(Array(kinetic_view(u0, alone, 1, 1)), size(grid))
                f_end = reshape(Array(kinetic_view(u, alone, 1, 1)), size(grid))
                n_start = density(grid, f_start)[1]
                difference = maximum(abs, f_end .- f_dream[:, :, 1, 2]) /
                             maximum(abs, f_dream[:, :, 1, 2])
                (difference, (density(grid, f_end)[1] - n_start) / n_start)
            end
            @info "the first step of f_re against DREAM, without and with the flux " *
                  "through p = 0 (the difference, the change of the density): $first_step"
            @test first_step[1][1] < HEATING_RUN_FIRST_STEP
            @test abs(first_step[1][2]) < HEATING_RUN_DENSITY
            # the operator's default: f_re departs from DREAM's by 1.06e-4 of its largest
            # value after the first step, and its density falls by 1.55e-4
            @test first_step[2][1] > HEATING_RUN_DEFAULT_FIRST_STEP
            @test first_step[2][2] < -HEATING_RUN_DEFAULT_FIRST_STEP

            # the density of f_re is conserved: no flux through p = 0, and f_re is negligible at
            # p_max
            distribution_array(u) = reshape(Array(distribution(u)), size(grid))
            densities = [density(grid, distribution_array(states[:, k]))[1]
                         for k in eachindex(times)]
            drift = maximum(abs, densities .- densities[1]) / densities[1]
            @info "the density of f_re over the run against its start: $drift"
            @test drift < HEATING_RUN_DENSITY
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
                rhs!(zeros(size(solution.states, 1)), term, solution.states[:, k],
                    layout, row, times[k]),
                layout, row))
            energy_change = [heating(case.heating, k) for k in eachindex(times)]
            moments = [heating(moment, k) for k in eachindex(times)]
            @info "the heating by f_re at compat = runes2p over the run [W/m³]: the " *
                  "energy change $(extrema(energy_change)), DREAM's moment " *
                  "$(extrema(moments)), the ratio at each step " *
                  "$(extrema(energy_change ./ moments))"
            T = [value(solution.states[:, k], layout, :cold_electron_temperature)
                 for k in eachindex(times)]
            temperature = maximum(abs, (T .- T_dream) ./ T_dream)
            @info "T_cold at compat = runes2p against DREAM's: $temperature, at the last " *
                  "step $((T[end] - T_dream[end]) / T_dream[end])"
            id = unknown_id(layout, :runaway_distribution)
            distribution_array(u) = reshape(
                Array(kinetic_view(u, layout, id, 1)), size(grid))
            distributions = [maximum(abs,
                                 distribution_array(solution.states[:, k]) .-
                                 f_dream[:, :, 1, k]) / maximum(abs, f_dream[:, :, 1, k])
                             for k in eachindex(times)]
            densities = [density(grid, distribution_array(solution.states[:, k]))[1]
                         for k in eachindex(times)]
            drift = maximum(abs, densities .- densities[1]) / densities[1]
            @info "f_re at compat = runes2p against DREAM's, after the first step and at " *
                  "the last: $(distributions[2]), $(distributions[end]); the density of " *
                  "f_re over the run against its start: $drift"
            # Measured: the energy change of the operator falls from 1.39e5 to 1.36e5 W/m³,
            # DREAM's moment on the same states falls from 7.81e4 to 7.68e4 W/m³ over the steps,
            # and the energy change is 1.772 to 1.779 times the moment at each step. Nearly all of
            # f_re sits in the first momentum cell (p = 0.25). The field of 1.5 V/m beats the
            # friction at the face p = 0.5 above it, so the momentum faces take the upwind side
            # of the whole advection there, the value of the first cell. The friction on the
            # free electrons then carries that cell's electrons down through the face, and the
            # energy change counts that flux with the energy step between the first two cells.
            # At the start that is 2.13 times the friction moment at the cell centre with the
            # collision settings of the operator (6.53e4 W/m³), which is 0.837 of DREAM's moment
            # above; the test set "the heating on refined momentum grids" below refines it.
            # The run takes lagged = :ions in place of :all and the superthermal
            # frequencies, since the electron-electron operator has no ultra-relativistic
            # form (runaway_heating_example). At p = 0.25 the superthermal frequencies give
            # ν_s 17 times and ν_D 4.1 times the ultra-relativistic ones
            # (_collisional_slowing_down and _deflection_frequency of
            # src/physics/collision_frequencies.jl), so f_re differs from
            # DREAM's by 0.46 of its largest value after the first step and 0.76 at the
            # last. T_cold ends 2.8e-4 above DREAM's: the form of the heating gives about
            # 2.5e-4 (the difference of the two heatings over the 5 µs, about 0.30 J/m³,
            # over the heat capacity 1.5 e n_cold of 12 J/(m³ eV)), and the ohmic heating at
            # the end of the step adds about 3e-5, as at compat = dream.
            # The bounds: the energy change is never negative; it is 1.7 to 1.85 times the
            # moment at every step; the smallest moment is 7.68e4 W/m³, above 7e4 W/m³; T_cold
            # at the last step is 2.77e-4 above DREAM's, between 2e-4 and 3.5e-4.
            @test all(>=(0), energy_change)
            @test all(1.7 .< energy_change ./ moments .< 1.85)
            @test all(moments .> 7e4)
            @test 2e-4 < (T[end] - T_dream[end]) / T_dream[end] < 3.5e-4
            # no flux through p = 0, as at compat = dream: the density of f_re changes by
            # 6.0e-16 of its start over the run, round-off
            @test drift < HEATING_RUN_DENSITY
            # f_re against DREAM's, relative to DREAM's largest value: 0.458 after the first
            # step and 0.756 at the last, the difference of the collision frequencies
            @test 0.4 < distributions[2] < 0.5
            @test 0.7 < distributions[end] < 0.8
        end

        @testset "the heating on refined momentum grids" begin
            # The free-electron part of the energy change at the start of the case at
            # compat = runes2p (T = 100 eV, 1.5 V/m), with f = 1 in the momentum cells below
            # p = 0.5 of the last pitch cell and zero elsewhere, divided by the friction moment
            # at the cell centres: FrictionEnergyTransferTerm with the collision settings of
            # the operator of f_re, on the same grid. Only the width of the momentum cells
            # varies. The shared side is the term; "own sides" is the moment of the operator
            # applied to f, each face on the side of its own advection.
            # Measured, for the widths 0.5, 0.25, 0.125 and 0.0625:
            # - shared side: 2.127, 1.543, 1.267, 1.133;
            # - own sides: 0.000, 0.486, 0.740, 0.869.
            # Both are of first order: their distances from 1 fall by 2.08, 2.03 and 2.01 and by
            # 1.95, 1.97 and 1.99 per halving. Their mean is of second order: its distance from
            # 1 falls by 4.41, 4.10 and 4.03. With f = 1 in every
            # pitch cell below p = 0.5 the shared side gives 0.978 at the width 0.5,
            # since the field beats the friction at p = 0.5 in some pitch cells only.
            function heating_ratios(cells; every_pitch = false)
                case = runaway_heating_example(; compat = :runes2p, momentum_cells = cells)
                (; layout, grid) = case
                u = copy(case.initial_state)
                id = unknown_id(layout, :runaway_distribution)
                f = kinetic_view(u, layout, id, 1)
                m = momentum_grid(grid)
                f .= 0
                for i in 1:momentum_cell_count(m)
                    m.p_centers[i] < 0.5 || continue
                    if every_pitch
                        f[i, :] .= 1
                    else
                        f[i, end] = 1
                    end
                end
                row = unknown_id(layout, :cold_electron_temperature)
                row_value(term) = only(unknown_values(
                    rhs!(zeros(length(u)), term, u, layout, row, 0.0), layout, row))
                friction = FrictionEnergyTransferTerm(:runaway_distribution, grid,
                    PlasmaFromUnknowns(case.prescribed); settings = case.fokker_planck.settings)
                reference = row_value(friction)
                term = case.heating
                plasma = Runes2P._transfer_plasma(term, u, layout, 0.0)
                change = Runes2P._loss_change(Runes2P._loss_operators(term, grid, plasma),
                    kinetic_array(u, layout, id))
                weights = Runes2P._loss_weights(term, grid, 1, u, layout, 0.0)
                own_sides = -term.coefficient * sum(weights .* view(change, :, :, 1))
                return (shared = row_value(term) / reference, own = own_sides / reference)
            end
            results = [heating_ratios(cells) for cells in (100, 200, 400, 800)]
            shared = [r.shared for r in results]
            own = [r.own for r in results]
            mean = (shared .+ own) ./ 2
            every_pitch = heating_ratios(100; every_pitch = true).shared
            @info "the heating over the friction moment at the widths 0.5 to 0.0625: shared " *
                  "side $shared, own sides $own; every pitch cell at 0.5: $every_pitch"
            @test all(shared .> 1 .> own)
            @test shared[1] ≈ 2.127 atol = 1e-3
            @test own[1] < 1e-12
            # first order: the error halves with the width; the mean falls by about 4
            @test all(1.8 .< (shared[1:3] .- 1) ./ (shared[2:4] .- 1) .< 2.2)
            @test all(1.8 .< (1 .- own[1:3]) ./ (1 .- own[2:4]) .< 2.2)
            @test all(3.5 .< (mean[1:3] .- 1) ./ (mean[2:4] .- 1) .< 4.6)
            @test every_pitch < 1
        end

        @testset "a step across the change of side of a face" begin
            # At compat = runes2p the heating by f_re jumps where a momentum face changes its
            # upwind side, and the side follows T_cold through the friction. Near the field at
            # which the whole advection of the face p = 0.5 of the last pitch cell vanishes at
            # the initial state, the first step from T_old can have no solution: Newton cycles
            # between the two sides until its 30 iterations end, and the step control halves the
            # step. Measured over 16 fields from (1 + 3e-7) to (1 + 1e-5) times that field, 6
            # runs have 1 to 6 rejected steps, the same under upwind and under TCDF and at -O0
            # and -O2, and every run finishes. At (1 + 1e-6) times that field the first attempt
            # fails and the run then takes two half steps, whose states are those of fixed half
            # steps; at (1 − 1e-6) times it no step is rejected.
            case = runaway_heating_example(; compat = :runes2p)
            whole = Runes2P._operator_at(case.fokker_planck, case.initial_state, case.layout, 0.0)
            pitch = size(case.grid)[2]
            @test momentum_grid(case.grid).p_faces[2] == 0.5
            flip = -whole.advection_p[2, pitch, 1] /
                   whole.unit_field_advection_p[2, pitch, 1]
            @info "the face p = 0.5 of the last pitch cell changes its side at $flip V/m"
            @test 0.1 < flip < 0.2
            steps = times[1:3]
            above = runaway_heating_example(; compat = :runes2p, field = flip * (1 + 1e-6))
            adaptive = integrate(above.system, above.initial_state, steps)
            statistics = adaptive.statistics
            @info "a step across the change of side: statuses $(statistics.status), Newton " *
                  "iterations $(statistics.newton_iterations), steps $(statistics.time_step)"
            @test adaptive.times[end] == steps[end]
            @test statistics.status[1] === :max_iterations
            @test !statistics.accepted[1]
            @test statistics.time_step[2] == statistics.time_step[1] / 2
            @test all(statistics.accepted[2:end])
            halves = collect(range(steps[1], steps[2]; length = 3))
            fixed = integrate(above.system, above.initial_state, halves; stepper = FixedSteps())
            @test all(fixed.statistics.accepted)
            row = unknown_id(above.layout, :cold_electron_temperature)
            T_adaptive = only(unknown_values(adaptive.states[:, 2], above.layout, row))
            T_fixed = only(unknown_values(fixed.states[:, end], above.layout, row))
            @test T_adaptive ≈ T_fixed rtol = 1e-12
            below = runaway_heating_example(; compat = :runes2p, field = flip * (1 - 1e-6))
            @test all(integrate(below.system, below.initial_state, steps).statistics.accepted)
        end
    end
end
