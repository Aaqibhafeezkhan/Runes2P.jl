# The ion charge states and the energy of the cold electrons together against DREAM in its
# fluid mode (reference/dream/cases/ion_energy_fluid.py): the case of test_cold_energy_dream.jl
# with dynamic ions, deuterium singly ionized and neutral argon deposited at the start. The
# radiated power, the binding energy, the conductivity of the ohmic heating and the rate
# equations all read the charge states of the run, and the cold density is the free-electron
# density of the ions. The temperature falls from 50 to about
# 3 eV in 100 µs. Backward Euler on DREAM's time points: the same method in both codes, so
# the states agree to the Newton tolerances. Needs the ADF11 cache (scripts/fetch_adas.py).
using HDF5

@testset "ions and cold energy against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the comparison with DREAM is skipped"
        @test_skip isdir(adas_cache_directory())
    else
        data = h5open(reference_file("dream_ion_energy.h5"), "r") do h5
            Dict(name => read(h5[name]) for name in keys(h5))
        end
        # DREAM stores (time, ..., radius) transposed in HDF5: radius first here
        T_dream, n_dream, ions_dream = data["T_cold"], data["n_cold"], data["n_i"]
        times = data["t"]
        nr = size(T_dream, 1)
        radial = RadialGrid(uniform_r_faces(nr, data["minor_radius"]))
        @test radial.r_centers ≈ data["r_grid"] rtol = 1e-12
        composition = IonComposition(["D" => 1, "Ar" => 18])
        base = PlasmaState(; cold_electron_temperature = data["temperature"],
            ions = [IonSpecies("D", 1, 1, fill(data["deuterium"], nr)),
                IonSpecies("Ar", 18, zeros(19, nr))],
            electric_field = zeros(nr), magnetic_field = zeros(nr))
        minus(name) = LinearTerm(name; coefficient = -1)
        layout = UnknownLayout(radial,
            [ion_densities_unknown(composition; scale = 1e20),
                RadialUnknown(:electric_field),
                RadialUnknown(:ohmic_current_density; scale = 1e6),
                RadialUnknown(:cold_electron_temperature; scale = 10.0, nonnegative = true),
                RadialUnknown(:cold_electron_density; scale = 1e20),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e21)])
        radiation = RadiatedPowerTerm(base; radiation = :dream,
            bremsstrahlung_corrections = :dream, ions = :ion_densities)
        conductivity = ConductivityTerm(:electric_field,
            ConductivityModel(PlasmaFromUnknowns(base; ions = :ion_densities)))
        equations = [
            Equation(:ion_densities, TimeDerivative(), IonRateTerm(composition)),
            Equation(:electric_field, Algebraic(),
                PrescribedValueTerm(fill(data["field"], nr)), minus(:electric_field)),
            population_equations(composition)...,
            Equation(:ohmic_current_density, Algebraic(), conductivity, minus(:ohmic_current_density)),
            thermal_energy_equation(),
            Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy),
                ohmic_heating_term(), radiation,
                HeatTransportTerm(; diffusion = data["diffusivity"]))]
        u0 = zeros(length(layout))
        set_ion_densities!(u0, layout, base)
        deposit_neutrals!(u0, layout, composition, "Ar" => data["argon"])
        unknown_values(u0, layout, 2) .= data["field"]
        unknown_values(u0, layout, 4) .= T_dream[:, 1]
        unknown_values(u0, layout, 5) .= n_dream[:, 1]
        system = EquationSystem(layout, equations; time_scale = 1e-6, initial_state = u0)
        solution = integrate(system, u0, times; stepper = FixedSteps())
        @test run_status(solution) === :finished
        # DREAM's single step per interval
        @test all(solution.statistics.accepted) &&
              length(solution.statistics.accepted) == length(times) - 1
        for k in (2, 11, 51, length(times))
            state = solution.states[:, k]
            @test unknown_values(state, layout, 4) ≈ T_dream[:, k] rtol = 1e-7
            @test unknown_values(state, layout, 5) ≈ n_dream[:, k] rtol = 1e-8
            ions = ion_densities(state, layout, composition)
            @test permutedims(ions[1].charge_state_densities) ≈ ions_dream[:, 1:2, k] rtol = 1e-8
            @test maximum(abs, permutedims(ions[2].charge_state_densities) .-
                               ions_dream[:, 3:21, k]) <=
                  1e-7 * data["argon"]
        end
    end
end
