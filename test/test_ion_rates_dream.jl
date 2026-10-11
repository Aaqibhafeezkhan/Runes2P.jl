# The rate equations of the ion charge states against DREAM in its fluid mode
# (reference/dream/cases/ion_rates_fluid.py): three radii at prescribed temperatures of
# 20, 8 and 3 eV, deuterium singly ionized at 2.6e19 m⁻³, and neutral deuterium and argon
# deposited at the start, each at the background density (scenario 1 of DREAM's
# examples/Disruptions/generate.py). The cold density is the
# free-electron density of the ions. Backward Euler on DREAM's 200 steps over 20 µs. The same
# method in both codes: the charge states agree to the Newton tolerances. DREAM's deposited
# deuterium is a species of its own, which Runes2P adds to the neutrals of the background
# deuterium: the rates are linear in the densities, so the sums agree. Needs the ADF11 cache
# (scripts/fetch_adas.py).
using HDF5

@testset "ion rates against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the comparison with DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        data = h5open(joinpath(@__DIR__, "data", "dream_ion_rates.h5"), "r") do h5
            Dict(name => read(h5[name]) for name in keys(h5))
        end
        # DREAM stores (time, state, radius) transposed in HDF5: (radius, state, time) here
        n_dream = data["n_i"]
        times = data["t"]
        nr = size(n_dream, 1)
        radial = RadialGrid(uniform_r_faces(nr, data["minor_radius"]))
        @test radial.r_centers ≈ data["r_grid"] rtol = 1e-12
        background = data["background"]
        composition = IonComposition(["D" => 1, "Ar" => 18])
        layout = UnknownLayout(radial, [ion_densities_unknown(composition),
                                        RadialUnknown(:cold_electron_density; scale=1e19),
                                        RadialUnknown(:cold_electron_temperature; scale=10.0)])
        equations = [Equation(:ion_densities, TimeDerivative(), IonRateTerm(composition)),
                     population_equations(composition)...,
                     Equation(:cold_electron_temperature, Algebraic(), PrescribedValueTerm(data["temperature"]),
                              LinearTerm(:cold_electron_temperature; coefficient=-1))]
        u0 = zeros(length(layout))
        set_ion_densities!(u0, layout, [IonSpecies("D", 1, 1, fill(background, nr)),
                                        IonSpecies("Ar", 18, zeros(19, nr))])
        deposit_neutrals!(u0, layout, composition, "D" => data["deposit_deuterium"] * background,
                          "Ar" => data["deposit_argon"] * background)
        radial_values(u0, layout, 2) .= background
        radial_values(u0, layout, 3) .= data["temperature"]
        system = EquationSystem(layout, equations; time_scale=1e-7, initial_state=u0)
        solution = integrate(system, u0, times; stepper=FixedSteps())
        @test run_status(solution) === :finished
        # DREAM's single step per interval: a halved step would be another method
        @test all(solution.statistics.accepted) && length(solution.statistics.accepted) == length(times) - 1
        # Runes2P's states per time: deuterium as the sum of DREAM's two species
        states(k) = ion_densities(solution.states[:, k], layout, composition)
        for k in (2, 11, 51, length(times))
            ions = states(k)
            deuterium = n_dream[:, 1:2, k] .+ n_dream[:, 3:4, k]
            argon = n_dream[:, 5:23, k]
            @test permutedims(ions[1].charge_state_densities) ≈ deuterium rtol = 1e-8
            # each argon state on the scale of the argon density
            @test maximum(abs, permutedims(ions[2].charge_state_densities) .- argon) <=
                  1e-8 * data["deposit_argon"] * background
            @test radial_values(solution.states[:, k], layout, 2) ≈ data["n_cold"][:, k] rtol = 1e-8
        end
    end
end
