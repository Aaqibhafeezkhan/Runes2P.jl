# The system of Runes2P for DREAM's disruption case (reference/dream/cases/disruption_kinetic.py:
# DREAM's examples/Disruptions/generate.py, scenario 1, with examples/Disruptions/ASDEXU.py),
# from DREAM's state at the start of one of the two phases of the case (the ionization of
# the injected D and Ar, or the thermal and current quench): the builder of
# test_disruption_dream.jl. With compat = dream it takes DREAM's settings of the case, with
# compat = runes2p the defaults of Runes2P.
using Runes2P, HDF5

const WALL_RADIUS = 0.55            # ASDEXU.py:24
const DIFFUSIVITY = 4000.0          # m²/s, generate.py:502 (scenario 1)
const P_CUT = 0.01                  # generate.py:329
const THRESHOLD = 5.0               # generate.py:234
const COMPOSITION = IonComposition(["D" => 1, "D_inj" => 1, "Ar" => 18])

# DREAM's output of a run, an HDF5 file or group with DREAM's groups eqsys and grid: (name =>
# array) with the time last, as HDF5 gives DREAM's arrays to Julia
function read_dream_state(h5)
    data = Dict{String, Any}()
    for name in ("T_cold", "n_cold", "n_hot", "n_re", "n_tot",
        "E_field", "psi_p", "psi_edge", "psi_wall", "I_p",
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

function collision_settings(compat)
    compat === :dream ?
    CollisionSettings(; mode = :full, screening = :partial,
        coulomb_logarithm = :energy_dependent, screening_window = :dream,
        ion_size_table = :dream, bremsstrahlung = :stopping_power, bremsstrahlung_diffusion = :dream,
        bremsstrahlung_constant = :dream) :
    CollisionSettings(; mode = :full, screening = :partial, coulomb_logarithm = :energy_dependent)
end

function build(start; scheme::Symbol = :tcdf, compat::Symbol = :dream, diffusivity::Real = DIFFUSIVITY)
    radial = RadialGrid(start["r_f"])
    grid = cylindrical_grid(MomentumGrid(start["p1_f"], start["p2_f"]), radial)
    nr = radial_cell_count(radial)
    p_max = last(start["p1_f"])
    # the prescribed part: the magnetic field and the composition (the values are unknowns)
    background = [IonSpecies(name, Z, zeros(Z + 1, nr))
                  for (name, Z) in zip(COMPOSITION.names, COMPOSITION.nuclear_charges)]
    prescribed = PlasmaState(; cold_electron_temperature = fill(10.0, nr),
        ions = background, electric_field = zeros(nr),
        magnetic_field = fill(start["B"], nr))
    settings = collision_settings(compat)
    # the hot region: DREAM's 5 thermal momenta for compat = dream; for runes2p, which splits the
    # ionization with CHIANTI's cross sections, the minimum of that split, 7
    factor = compat === :dream ? THRESHOLD :
             Runes2P.MINIMUM_SPLIT_THRESHOLD_FACTORS[:chianti]
    region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor))
    plasma = PlasmaFromUnknowns(prescribed; ions = :ion_densities, temperature_floor = 0.01)
    operator = AdvectionDiffusion(grid; scheme, p_upper_boundary = :f_zero)
    fp = FokkerPlanckTerm(
        :f, operator; plasma, settings, synchrotron = compat === :runes2p,
        electric_field = :electric_field,
        lagged = compat === :dream ? () : :ions)
    minus(name) = LinearTerm(name; coefficient = -1)
    current(; kwargs...) = CurrentMomentTerm(:f, grid; pitch_rule = :cell_center, kwargs...)
    conductivity = ConductivityTerm(:electric_field,
        ConductivityModel(
            PlasmaFromUnknowns(prescribed; ions = :ion_densities,
                temperature_floor = 0.01);
            kinetic = :dream_fit))
    # the cross sections of the kinetic ionization: DREAM's form and fits for compat = dream,
    # CHIANTI's continued by the Bethe theory for runes2p
    kinetic = KineticIonizationTerm(COMPOSITION, :f, grid; region,
        cross_section_form = compat === :dream ? :dream : :chianti)
    fluid_ions = (ions = :ion_densities,)
    # the exchange among the ions: DREAM's logarithm with T_e and DREAM's masses for compat =
    # dream, the NRL logarithm and the standard atomic weights for runes2p
    exchange = compat === :dream ? (ion_coulomb_logarithm = :dream, mass_table = :dream) :
               NamedTuple()
    # Gould's relativistic corrections of the bremsstrahlung: DREAM's for compat = dream
    gould = compat === :dream ? (bremsstrahlung_corrections = :dream,) : NamedTuple()
    split = compat === :dream ? NamedTuple() : (kinetic = kinetic,)
    external = ExternalAvalancheTerm(prescribed; p_cutoff = p_max, settings, fluid_ions...)
    internal = rosenbluth_putvinski_rate(P_CUT, p_max)
    nuclear(c) = Float64[Z for Z in c.nuclear_charges for _ in 0:Z]
    unknowns = vcat(
        induction_unknowns(radial; electric_field_scale = 1.0, current_density_scale = 1e6),
        [
            ion_densities_unknown(COMPOSITION),
            RadialUnknown(:ion_thermal_energy; components = length(COMPOSITION),
                scale = 1e4, nonnegative = true),
            RadialUnknown(:cold_electron_density; scale = 1e19), RadialUnknown(
                :hot_electron_density; scale = 1e17),
            RadialUnknown(:runaway_density; scale = 1e15), RadialUnknown(:total_electron_density; scale = 1e20),
            RadialUnknown(:free_electron_density; scale = 1e19), RadialUnknown(:particle_source; scale = 1e24),
            RadialUnknown(:cold_electron_temperature; scale = 100.0),
            RadialUnknown(:cold_thermal_energy; scale = 1e4),
            RadialUnknown(:ohmic_current_density; scale = 1e6), RadialUnknown(
                :hot_electron_current_density; scale = 1e5),
            RadialUnknown(:runaway_current_density; scale = 1e5),
            KineticUnknown(:f, grid; scale = 1e19)])
    layout = UnknownLayout(radial, unknowns)
    equations = vcat(
        induction_equations(radial; wall_radius = WALL_RADIUS, loop_voltage_at_wall = 0.0),
        [
            # DREAM's fluid ionization takes the whole ADAS rate next to the kinetic one, so the tail of
            # the Maxwellian above the threshold ionizes twice (ions.cpp:280); Runes2P's split takes the
            # cold electrons below the threshold alone (IonRateTerm with kinetic)
            Equation(:ion_densities, TimeDerivative(), IonRateTerm(COMPOSITION; split...), kinetic),
            population_equations(COMPOSITION; hot_density = :hot_electron_density,
                runaway_density = :runaway_density)...,
            Equation(:hot_electron_density, Algebraic(),
                DensityMomentTerm(:f, grid; region), minus(:hot_electron_density)),
            Equation(:total_electron_density, Algebraic(),
                WeightedSumTerm(:ion_densities, nuclear(COMPOSITION)),
                minus(:total_electron_density)),
            Equation(:free_electron_density, Algebraic(),
                WeightedSumTerm(:ion_densities, free_electron_weights(COMPOSITION)),
                minus(:free_electron_density)),
            # DREAM's explicit particle source: dn_free/dt = S + internal + external avalanche
            Equation(:particle_source, TimeDerivative(:free_electron_density),
                LinearTerm(:particle_source),
                ProductTerm(:total_electron_density, :runaway_density; coefficient = internal), external),
            Equation(:f,
                TimeDerivative(),
                fp,
                ParticleSourceTerm(:particle_source, grid, :cold_electron_temperature;
                    shape = :lowest_cell),
                RosenbluthPutvinskiSourceTerm(:f, grid; p_cutoff = P_CUT)),
            Equation(:runaway_density, TimeDerivative(), MomentumBoundaryFluxTerm(:f, fp), external),
            # j_re = e c n_re; DREAM's takes the sign of the field (j_re.cpp:95), which is positive at
            # every radius of both phases of the case
            Equation(:runaway_current_density, Algebraic(),
                LinearTerm(:runaway_density;
                    coefficient = Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT),
                minus(:runaway_current_density)),
            Equation(:hot_electron_current_density, Algebraic(),
                current(; region), minus(:hot_electron_current_density)),
            Equation(:ohmic_current_density, Algebraic(), current(),
                minus(:hot_electron_current_density), conductivity,
                minus(:ohmic_current_density)),
            Equation(
                :total_current_density, Algebraic(), LinearTerm(:ohmic_current_density),
                LinearTerm(:hot_electron_current_density), LinearTerm(:runaway_current_density),
                minus(:total_current_density)),
            thermal_energy_equation(),
            Equation(:cold_electron_temperature,
                TimeDerivative(:cold_thermal_energy), ohmic_heating_term(),
                RadiatedPowerTerm(
                    prescribed; radiation = :dream, fluid_ions..., split..., gould...),
                HeatTransportTerm(; diffusion = diffusivity),
                RunawayFluidEnergyTransferTerm(prescribed; fluid_ions...),
                IonEnergyExchangeTerm(prescribed; side = :electrons, fluid_ions..., exchange...)),
            Equation(:ion_thermal_energy,
                TimeDerivative(),
                IonEnergyExchangeTerm(prescribed; side = :ions, fluid_ions..., exchange...))])
    return (; layout, equations, grid, radial, nr)
end

# DREAM's first state of the phase as the initial state of Runes2P
function initial_state(case, start)
    layout = case.layout
    u = zeros(length(layout))
    first_of(a) = selectdim(a, ndims(a), 1)
    names = Dict(
        "T_cold" => :cold_electron_temperature, "n_cold" => :cold_electron_density,
        "n_hot" => :hot_electron_density, "n_re" => :runaway_density, "n_tot" =>
            :total_electron_density,
        "E_field" => :electric_field, "psi_p" => :poloidal_flux, "j_ohm" =>
            :ohmic_current_density,
        "j_hot" => :hot_electron_current_density, "j_re" => :runaway_current_density,
        "j_tot" => :total_current_density, "W_cold" => :cold_thermal_energy)
    for (dream, ours) in names
        haskey(start, dream) &&
            (unknown_values(u, layout, unknown_id(layout, ours)) .= first_of(start[dream]))
    end
    for (dream, ours) in ("psi_edge" => :edge_poloidal_flux,
        "psi_wall" => :wall_poloidal_flux, "I_p" => :plasma_current)
        u[only(unknown_range(layout, unknown_id(layout, ours)))] = first_of(start[dream])[1]
    end
    states = first_of(start["n_i"])   # (radius, state)
    energies = first_of(start["W_i"])  # (radius, species)
    ions_id, energy_id = unknown_id(layout, :ion_densities),
    unknown_id(layout, :ion_thermal_energy)
    for ir in 1:case.nr
        radial_view(u, layout, ions_id, ir) .= states[ir, :]
        radial_view(u, layout, energy_id, ir) .= energies[ir, :]
        kinetic_view(u, layout, unknown_id(layout, :f), ir) .= start["f_hot"][:, :, ir, 1]
    end
    unknown_values(u,
        layout,
        unknown_id(layout, :free_electron_density)) .= [sum(free_electron_weights(COMPOSITION) .*
                                                            states[ir, :])
                                                        for ir in 1:case.nr]
    return u
end

# the values of a term of a radial row at the state u, per radius
function _term_values(term, u, layout, row, t)
    out = Runes2P.rhs!(zeros(length(u)), term, u, layout, row, t)
    return collect(unknown_values(out, layout, row))
end

"""
    budgets(case, solution)

The closures of the system of `case` over `solution`, independent of DREAM (checked by
`test_disruption_dream.jl`): each ion species conserved (no ion transport), the exchange of
energy between the electrons and the ions antisymmetric, the heat diffusion without a net
flux (no flux at the edge), the particle source keeping `∫f − (n_cold + n_hot)` fixed, the
row of `W_cold` closed by its channels, and the energy of the poloidal field.
`W = Σ V'Δr (ψ − ψ_wall) j_tot` changes in a step by `4π dt Σ V'Δr E (j_new + j_old)/2`
when the map from `ψ` to `j` is symmetric (backward Euler, `∂ψ/∂t = 2πE`), so the magnetic
energy `−W/4π` falls by the work of the field on all currents (a perfectly conducting wall:
no flux through it). A balance of the total energy does not close
in this model: `f` covers the bulk as well, and the energy that `f` takes from the field or gives
to the bulk through collisions is not in the row of `W_cold` (DREAM's energy equation).

Returns a named tuple of the worst relative values over the steps: `species` (the change of the
density of a species, relative to the largest density), `exchange` (`Q_i + Q_e`, relative to the
largest electron side), `diffusion` (the net flux of the heat diffusion, relative to its absolute
sum), `closure` (the row of `W_cold` against its channels, relative to `dW/dt`), `drift` (of
`∫f − n_cold − n_hot`, relative to `n_cold`) and `field` (the change of the field energy against
the work of the field, relative to the change); and `channels` with `labels`, the energy of each
channel of the row of `W_cold` over the solution (volume mean, J/m³).
"""
function budgets(case, solution)
    layout = case.layout
    weights = case.radial.spatial_jacobian .* case.radial.r_widths        # V' Δr
    row_T = unknown_id(layout, :cold_electron_temperature)
    T_equation = only(e for e in case.equations if e.unknown === :cold_electron_temperature)
    W_i_equation = only(e for e in case.equations if e.unknown === :ion_thermal_energy)
    labels = ["E j_ohm", "radiation and binding", "heat diffusion",
        "e c E_c n_re", "exchange (electrons)"]
    channels = zeros(length(labels))
    ions_id, energy_id, f_id = unknown_id(layout, :ion_densities),
    unknown_id(layout, :ion_thermal_energy), unknown_id(layout, :f)
    species(u) = [sum(radial_view(u, layout, ions_id, ir)[species_components(COMPOSITION, s)])
                  for s in 1:length(COMPOSITION),
    ir in 1:case.nr]
    totals0 = species(solution.states[:, 1])
    worst = Dict(:species => 0.0, :exchange => 0.0, :diffusion => 0.0, :closure => 0.0)
    ψ_id, E_id, j_id = unknown_id(layout, :poloidal_flux),
    unknown_id(layout, :electric_field), unknown_id(layout, :total_current_density)
    wall = only(unknown_range(layout, unknown_id(layout, :wall_poloidal_flux)))
    field_energy(u) = sum(weights .* (unknown_values(u, layout, ψ_id) .- u[wall]) .*
                          unknown_values(u, layout, j_id))
    field_work = 0.0
    cold(u) = unknown_values(u, layout, unknown_id(layout, :cold_electron_density))
    gap(u) = density(case.grid, kinetic_array(u, layout, f_id)) .- cold(u) .-
             unknown_values(u, layout, unknown_id(layout, :hot_electron_density))
    gap0 = gap(solution.states[:, 1])
    drift = 0.0
    for m in 2:length(solution.times)
        u, t = solution.states[:, m], solution.times[m]
        dt = t - solution.times[m - 1]
        # the row of the temperature holds the time derivative of the thermal energy
        parts = [_term_values(term, u, layout, row_T, t) for term in T_equation.terms]
        channels .+= dt .* [sum(weights .* p) for p in parts] ./ sum(weights)
        worst[:species] = max(worst[:species], maximum(abs, species(u) .- totals0) /
                                               maximum(totals0))
        ion_side = Runes2P.rhs!(
            zeros(length(u)), only(W_i_equation.terms), u, layout, energy_id, t)
        Q_i = [sum(radial_view(ion_side, layout, energy_id, ir)) for ir in 1:case.nr]
        exchange = parts[findfirst(t -> t isa IonEnergyExchangeTerm, T_equation.terms)]
        diffusion = parts[findfirst(t -> t isa HeatTransportTerm, T_equation.terms)]
        worst[:exchange] = max(worst[:exchange], maximum(abs, Q_i .+ exchange) /
                                                 maximum(abs, exchange))
        worst[:diffusion] = max(worst[:diffusion],
            abs(sum(weights .* diffusion)) / sum(weights .* abs.(diffusion)))
        W = unknown_values(u, layout, unknown_id(layout, :cold_thermal_energy))
        W_old = unknown_values(solution.states[:, m - 1], layout, unknown_id(layout, :cold_thermal_energy))
        closure = (W .- W_old) ./ dt .- sum(parts)
        worst[:closure] = max(worst[:closure], maximum(abs, closure) /
                                               maximum(abs, (W .- W_old) ./ dt))
        drift = max(drift, maximum(abs, gap(u) .- gap0) / maximum(cold(u)))
        j_mid = (unknown_values(u, layout, j_id) .+
                 unknown_values(solution.states[:, m - 1], layout, j_id)) ./ 2
        field_work += 4π * dt * sum(weights .* unknown_values(u, layout, E_id) .* j_mid)
    end
    ΔW = field_energy(solution.states[:, end]) - field_energy(solution.states[:, 1])
    return (; species = worst[:species], exchange = worst[:exchange],
        diffusion = worst[:diffusion], closure = worst[:closure],
        drift, field = abs(ΔW - field_work) / abs(ΔW),
        field_energy_start = -field_energy(solution.states[:, 1]) / (4π * sum(weights)),
        field_energy_change = -ΔW / (4π * sum(weights)), channels, labels)
end
