# The energy equation of the cold electrons against DREAM in its fluid mode
# (reference/dream/cases/cold_energy_fluid.py): three radii, deuterium and singly ionized
# argon prescribed, a prescribed field, ohmic heating with DREAM's conductivity, the radiated
# power and binding energy of the ADAS tables in DREAM's default (recombination radiation
# off, Gould's bremsstrahlung), prescribed heat diffusion with a zero flux at the edge, and
# backward Euler on DREAM's time points. The same method in both codes: the temperatures agree
# to the Newton tolerances. Needs the ADF11 cache (scripts/fetch_adas.py).
using HDF5

@testset "cold energy against DREAM" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the comparison with DREAM is skipped"
    else
        data = h5open(joinpath(@__DIR__, "data", "dream_cold_energy.h5"), "r") do h5
            Dict(name => read(h5[name]) for name in keys(h5))
        end
        # DREAM stores (time, radius) transposed in HDF5: radius first here
        T_dream = data["T_cold"]
        times = data["t"]
        nr = size(T_dream, 1)
        radial = RadialGrid(uniform_r_faces(nr, data["minor_radius"]))
        @test radial.r_centers ≈ data["r_grid"] rtol = 1e-12
        n_D, n_Ar = data["deuterium"], data["argon"]
        argon = zeros(19, nr)
        argon[2, :] .= n_Ar
        n_free = n_D + n_Ar
        @test data["n_cold"][:, 1] ≈ fill(n_free, nr) rtol = 1e-12
        base = PlasmaState(; cold_electron_temperature=T_dream[:, 1], ions=[IonSpecies("D", 1, 1, fill(n_D, nr)),
                                                                          IonSpecies("Ar", 18, argon)],
                           electric_field=zeros(nr), magnetic_field=zeros(nr))
        minus(name) = LinearTerm(name; coefficient=-1)
        layout = UnknownLayout(radial, [RadialUnknown(:electric_field), RadialUnknown(:ohmic_current_density; scale=1e6),
                                        RadialUnknown(:cold_electron_temperature; scale=10.0, nonnegative=true),
                                        RadialUnknown(:cold_electron_density; scale=1e20),
                                        RadialUnknown(:cold_thermal_energy; scale=THERMAL_ENERGY_FACTOR * 1e21)])
        radiation = RadiatedPowerTerm(base; radiation=:dream)
        transport = HeatTransportTerm(; diffusion=data["diffusivity"])
        equations = [
            Equation(:electric_field, Algebraic(), PrescribedValueTerm(fill(data["field"], nr)), minus(:electric_field)),
            Equation(:cold_electron_density, Algebraic(), PrescribedValueTerm(fill(n_free, nr)), minus(:cold_electron_density)),
            Equation(:ohmic_current_density, Algebraic(),
                     ConductivityTerm(:electric_field, ConductivityModel(PlasmaFromUnknowns(base; density=nothing))),
                     minus(:ohmic_current_density)),
            thermal_energy_equation(),
            Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), ohmic_heating_term(), radiation,
                     transport)]
        u0 = zeros(length(layout))
        radial_values(u0, layout, 1) .= data["field"]
        radial_values(u0, layout, 3) .= T_dream[:, 1]
        radial_values(u0, layout, 4) .= n_free
        system = EquationSystem(layout, equations; time_scale=1e-5, initial_state=u0)
        solution = integrate(system, u0, times; stepper=FixedSteps())
        T = reduce(hcat, radial_values(solution.states[:, k], layout, 3) for k in eachindex(times))
        # the channels at the end of the first step (DREAM: the sum of the row is zero, so a
        # heating is negative)
        state = solution.states[:, 2]
        channel(term) = radial_values(Runes2P.rhs!(zeros(length(state)), term, state, layout, 3, times[2]), layout, 3)
        parts = radiated_power(radiation, state, layout, times[2])
        @test channel(ohmic_heating_term()) ≈ -data["Tcold_ohmic"][:, 1] rtol = 1e-8
        @test parts.radiation ≈ data["Tcold_radiation"][:, 1] rtol = 1e-8
        @test parts.binding ≈ data["Tcold_binding_energy"][:, 1] rtol = 1e-8
        @test channel(transport) ≈ -data["Tcold_transport"][:, 1] rtol = 1e-6
        # the temperature over the run, from 50 to 6 eV at the centre
        @test T ≈ T_dream rtol = 1e-7
    end
end
