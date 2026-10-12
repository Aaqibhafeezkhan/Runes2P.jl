# The system of the disruption case in Runes2P, from DREAM's state at the start of a phase:
# the builder of the system that test_disruption_dream.jl runs. The case is DREAM's
# examples/Disruptions/generate.py; the settings that make it DREAM's method are those of
# compat = dream below.
using Runes2P, HDF5

const WALL_RADIUS = 0.55            # ASDEXU.py:24
const DIFFUSIVITY = 4000.0          # m²/s, generate.py:502 (scenario 1)
const P_CUT = 0.01                  # generate.py:329
const THRESHOLD = 5.0               # generate.py:234
const COMPOSITION = IonComposition(["D" => 1, "D_inj" => 1, "Ar" => 18])

# DREAM's output of a run, an HDF5 file or group with DREAM's groups eqsys and grid: (name =>
# array) with the time last, as HDF5 gives DREAM's arrays to Julia
function read_dream_state(h5)
    data = Dict{String,Any}()
    for name in ("T_cold", "n_cold", "n_hot", "n_re", "n_tot", "E_field", "psi_p", "psi_edge", "psi_wall", "I_p",
                 "j_ohm", "j_hot", "j_re", "j_tot", "W_cold", "n_i", "W_i", "f_hot")
        haskey(h5["eqsys"], name) && (data[name] = read(h5["eqsys"][name]))
    end
    data["t"] = read(h5["grid/t"])
    data["r_f"] = read(h5["grid/r_f"])
    data["p1_f"] = read(h5["grid/hottail/p1_f"])
    data["p2_f"] = read(h5["grid/hottail/p2_f"])
    data["B"] = 2.5     # ASDEXU.py:25, uniform with R0 = ∞
    return data
end

collision_settings(compat) = compat === :dream ?
    CollisionSettings(; mode=:full, screening=:partial, coulomb_logarithm=:energy_dependent, screening_window=:dream,
                      ion_size_table=:dream, bremsstrahlung=:stopping_power, bremsstrahlung_diffusion=:dream,
                      bremsstrahlung_constant=:dream) :
    CollisionSettings(; mode=:full, screening=:partial, coulomb_logarithm=:energy_dependent)

function build(start; scheme::Symbol=:tcdf, compat::Symbol=:dream, diffusivity::Real=DIFFUSIVITY)
    radial = RadialGrid(start["r_f"])
    grid = cylindrical_grid(MomentumGrid(start["p1_f"], start["p2_f"]), radial)
    nr = radial_cell_count(radial)
    p_max = last(start["p1_f"])
    # the prescribed part: the magnetic field and the composition (the values are unknowns)
    background = [IonSpecies(name, Z, zeros(Z + 1, nr)) for (name, Z) in zip(COMPOSITION.names, COMPOSITION.nuclear_charges)]
    prescribed = PlasmaState(; cold_electron_temperature=fill(10.0, nr), ions=background, electric_field=zeros(nr),
                             magnetic_field=fill(start["B"], nr))
    settings = collision_settings(compat)
    region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor=THRESHOLD))
    plasma = PlasmaFromUnknowns(prescribed; ions=:ion_densities, temperature_floor=0.01)
    operator = AdvectionDiffusion(grid; scheme, p_upper_boundary=:f_zero)
    fp = FokkerPlanckTerm(:f, operator; plasma, settings, synchrotron=compat === :runes2p, electric_field=:electric_field,
                          lagged=compat === :dream ? () : :ions)
    minus(name) = LinearTerm(name; coefficient=-1)
    current(; kwargs...) = CurrentMomentTerm(:f, grid; pitch_rule=:cell_center, kwargs...)
    conductivity = ConductivityTerm(:electric_field,
                                    ConductivityModel(PlasmaFromUnknowns(prescribed; ions=:ion_densities,
                                                                         temperature_floor=0.01); kinetic=:dream_fit))
    kinetic = KineticIonizationTerm(COMPOSITION, :f, grid; region)
    fluid_ions = (ions=:ion_densities,)
    # the exchange among the ions: DREAM's logarithm with T_e and DREAM's masses for compat =
    # dream, the NRL logarithm and the standard atomic weights for runes2p
    exchange = compat === :dream ? (ion_coulomb_logarithm=:dream, mass_table=:dream) : NamedTuple()
    # Gould's relativistic corrections of the bremsstrahlung: DREAM's for compat = dream
    gould = compat === :dream ? (bremsstrahlung_corrections=:dream,) : NamedTuple()
    split = compat === :dream ? NamedTuple() : (kinetic=kinetic,)
    external = ExternalAvalancheTerm(prescribed; p_cutoff=p_max, settings, fluid_ions...)
    internal = rosenbluth_putvinski_rate(P_CUT, p_max)
    nuclear(c) = Float64[Z for Z in c.nuclear_charges for _ in 0:Z]
    unknowns = vcat(induction_unknowns(radial; electric_field_scale=1.0, current_density_scale=1e6), [
        ion_densities_unknown(COMPOSITION),
        RadialUnknown(:ion_thermal_energy; components=length(COMPOSITION), scale=1e4, nonnegative=true),
        RadialUnknown(:cold_electron_density; scale=1e19), RadialUnknown(:hot_electron_density; scale=1e17),
        RadialUnknown(:runaway_density; scale=1e15), RadialUnknown(:total_electron_density; scale=1e20),
        RadialUnknown(:free_electron_density; scale=1e19), RadialUnknown(:particle_source; scale=1e24),
        RadialUnknown(:cold_electron_temperature; scale=100.0),
        RadialUnknown(:cold_thermal_energy; scale=1e4),
        RadialUnknown(:ohmic_current_density; scale=1e6), RadialUnknown(:hot_electron_current_density; scale=1e5),
        RadialUnknown(:runaway_current_density; scale=1e5),
        KineticUnknown(:f, grid; scale=1e19)])
    layout = UnknownLayout(radial, unknowns)
    equations = vcat(induction_equations(radial; wall_radius=WALL_RADIUS, loop_voltage_at_wall=0.0), [
        # DREAM's fluid ionization takes the whole ADAS rate next to the kinetic one, so the tail of
        # the Maxwellian above the threshold ionizes twice (ions.cpp:280); Runes2P's split takes the
        # cold electrons below the threshold alone (IonRateTerm with kinetic)
        Equation(:ion_densities, TimeDerivative(), IonRateTerm(COMPOSITION; split...), kinetic),
        population_equations(COMPOSITION; hot_density=:hot_electron_density, runaway_density=:runaway_density)...,
        Equation(:hot_electron_density, Algebraic(), DensityMomentTerm(:f, grid; region), minus(:hot_electron_density)),
        Equation(:total_electron_density, Algebraic(), WeightedSumTerm(:ion_densities, nuclear(COMPOSITION)),
                 minus(:total_electron_density)),
        Equation(:free_electron_density, Algebraic(), WeightedSumTerm(:ion_densities, free_electron_weights(COMPOSITION)),
                 minus(:free_electron_density)),
        # DREAM's explicit particle source: dn_free/dt = S + internal + external avalanche
        Equation(:particle_source, TimeDerivative(:free_electron_density), LinearTerm(:particle_source),
                 ProductTerm(:total_electron_density, :runaway_density; coefficient=internal), external),
        Equation(:f, TimeDerivative(), fp, ParticleSourceTerm(:particle_source, grid, :cold_electron_temperature;
                                                              shape=:lowest_cell),
                 RosenbluthPutvinskiSourceTerm(:f, grid; p_cutoff=P_CUT)),
        Equation(:runaway_density, TimeDerivative(), MomentumBoundaryFluxTerm(:f, fp), external),
        Equation(:runaway_current_density, Algebraic(),
                 LinearTerm(:runaway_density; coefficient=Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT),
                 minus(:runaway_current_density)),
        Equation(:hot_electron_current_density, Algebraic(), current(; region), minus(:hot_electron_current_density)),
        Equation(:ohmic_current_density, Algebraic(), current(), minus(:hot_electron_current_density), conductivity,
                 minus(:ohmic_current_density)),
        Equation(:total_current_density, Algebraic(), LinearTerm(:ohmic_current_density),
                 LinearTerm(:hot_electron_current_density), LinearTerm(:runaway_current_density),
                 minus(:total_current_density)),
        thermal_energy_equation(),
        Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy), ohmic_heating_term(),
                 RadiatedPowerTerm(prescribed; radiation=:dream, fluid_ions..., split..., gould...),
                 HeatTransportTerm(; diffusion=diffusivity),
                 RunawayFluidEnergyTransferTerm(prescribed; fluid_ions...),
                 IonEnergyExchangeTerm(prescribed; side=:electrons, fluid_ions..., exchange...)),
        Equation(:ion_thermal_energy, TimeDerivative(), IonEnergyExchangeTerm(prescribed; side=:ions, fluid_ions..., exchange...))])
    return (; layout, equations, grid, radial, nr)
end

# DREAM's first state of the phase as the initial state of Runes2P
function initial_state(case, start)
    layout = case.layout
    u = zeros(length(layout))
    first_of(a) = selectdim(a, ndims(a), 1)
    names = Dict("T_cold" => :cold_electron_temperature, "n_cold" => :cold_electron_density,
                 "n_hot" => :hot_electron_density, "n_re" => :runaway_density, "n_tot" => :total_electron_density,
                 "E_field" => :electric_field, "psi_p" => :poloidal_flux, "j_ohm" => :ohmic_current_density,
                 "j_hot" => :hot_electron_current_density, "j_re" => :runaway_current_density,
                 "j_tot" => :total_current_density, "W_cold" => :cold_thermal_energy)
    for (dream, ours) in names
        haskey(start, dream) && (radial_values(u, layout, unknown_id(layout, ours)) .= first_of(start[dream]))
    end
    for (dream, ours) in ("psi_edge" => :edge_poloidal_flux, "psi_wall" => :wall_poloidal_flux, "I_p" => :plasma_current)
        u[only(unknown_range(layout, unknown_id(layout, ours)))] = first_of(start[dream])[1]
    end
    states = first_of(start["n_i"])   # (radius, state)
    energies = first_of(start["W_i"])  # (radius, species)
    ions_id, energy_id = unknown_id(layout, :ion_densities), unknown_id(layout, :ion_thermal_energy)
    for ir in 1:case.nr
        radial_view(u, layout, ions_id, ir) .= states[ir, :]
        radial_view(u, layout, energy_id, ir) .= energies[ir, :]
        kinetic_view(u, layout, unknown_id(layout, :f), ir) .= start["f_hot"][:, :, ir, 1]
    end
    radial_values(u, layout, unknown_id(layout, :free_electron_density)) .=
        [sum(free_electron_weights(COMPOSITION) .* states[ir, :]) for ir in 1:case.nr]
    return u
end

