# The system of Runes2P for DREAM's disruption case (reference/dream/cases/disruption_kinetic.py:
# DREAM's examples/Disruptions/generate.py, scenario 1, with examples/Disruptions/ASDEXU.py):
# the builder of test_disruption_dream.jl, test_disruption_initialization_dream.jl and
# test_disruption_chain_dream.jl, with the comparison of a phase with DREAM (`phase_differences`)
# and the budgets of the assembled system with their bounds (`budgets`, `check_budgets`).
# `build` gives the system of one of the two phases of the case (the ionization of the
# injected D and Ar, or the thermal and current quench), started from DREAM's state at its
# start (`initial_state`) or from Runes2P's own initialization (`initialize`, then
# `initial_state_after_initialization`). With compat = dream it takes DREAM's settings of the
# case, with compat = runes2p the defaults of Runes2P.
using Runes2P, HDF5

const WALL_RADIUS = 0.55            # ASDEXU.py:24
const DIFFUSIVITY = 4000.0          # m²/s, generate.py:502 (scenario 1)
const P_CUT = 0.01                  # generate.py:329
const THRESHOLD = 5.0               # generate.py:234
const COMPOSITION = IonComposition(["D" => 1, "D_inj" => 1, "Ar" => 18])

# the initialization of the case, generate.py getBaseline, parts 1 and 2 (lines 141-303)
const MINOR_RADIUS = 0.5            # ASDEXU.py:23
const MAGNETIC_FIELD = 2.5          # T, ASDEXU.py:25, uniform with R0 = ∞
const PLASMA_CURRENT = 800e3        # A, ASDEXU.py:27
const DENSITY = 2.6e19              # m⁻³, ASDEXU.py:30, uniform
const CENTRAL_TEMPERATURE = 5.8e3   # eV, ASDEXU.py:31
const TEMPERATURE_WIDTH = 0.4653    # ASDEXU.py:96, getInitialTemperature
const INITIALIZATION_TIME = 1.0     # s, generate.py:270
const INITIALIZATION_STEPS = 3      # generate.py:271
# the injected densities of scenario 1 as fractions of DENSITY, fracDinj and fracAr
# (generate.py:502)
const INJECTED_DEUTERIUM = 1.0
const INJECTED_ARGON = 1.0
const INJECTED_TEMPERATURE = 1.0    # eV, generate.py:439-440
# the ions of the initialization: the D background alone, generate.py:159
const BACKGROUND = IonComposition(["D" => 1])

"""
    initial_temperature(r)

ASDEXU.getInitialTemperature, `T0 exp(-((r/a)/0.4653)²)` on 100 points from 0 to `a`, at
the radii `r` by linear interpolation of those points, as DREAM takes the profile of
`T_cold.setPrescribedData` (examples/Disruptions/ASDEXU.py:83-97, generate.py:155-156).
"""
function initial_temperature(r)
    points = collect(range(0, MINOR_RADIUS; length = 100))
    values = @. CENTRAL_TEMPERATURE * exp(-((points / MINOR_RADIUS) / TEMPERATURE_WIDTH)^2)
    function interpolate(x)
        k = clamp(searchsortedlast(points, x), 1, length(points) - 1)
        weight = (x - points[k]) / (points[k + 1] - points[k])
        return values[k] + weight * (values[k + 1] - values[k])
    end
    return interpolate.(r)
end

# ASDEXU.getCurrentDensity (ASDEXU.py:116-131), (1 − (r/a)⁴)^1.5, at the cell centres, to be
# scaled to I_p
current_profile(r) = @. (1 - (r / MINOR_RADIUS)^4)^1.5

# the nuclear charge of each charge state of a composition, the weights of the total density
charge_state_nuclear_charges(c) = Float64[Z for Z in c.nuclear_charges for _ in 0:Z]

# the fully ionized D background at the uniform density (generate.py:159), per radius
function background_ions(nr)
    densities = zeros(2, nr)
    densities[2, :] .= DENSITY
    return [IonSpecies("D", 1, densities)]
end

"""
    initial_field(radial)

Run 0 of the case and the field of the prescribed current (generate.py:143-184, 247-254):
the conductivity of the initial plasma at the temperature of `initial_temperature` and the D
background, that of Braams and Karney, which is DREAM's of Sauter in a cylinder (the
neoclassical correction is 1 without trapped particles: `evaluateSauterElectricConductivity`
of src/Equations/RunawayFluid.cpp, DREAM v26.5), and the field
`E0 = j0 j(r) / σ(r)` with the profile of `current_profile` scaled so that `Σ 2π r Δr j0 j`
is the plasma current. Returns `(; conductivity, electric_field)`.
"""
function initial_field(radial)
    nr = radial_cell_count(radial)
    plasma = PlasmaState(;
        cold_electron_temperature = initial_temperature(radial.r_centers),
        ions = background_ions(nr), electric_field = zeros(nr),
        magnetic_field = fill(MAGNETIC_FIELD, nr))
    conductivity = [braams_karney_conductivity(plasma, ir) for ir in 1:nr]
    j = current_profile(radial.r_centers)
    j0 = PLASMA_CURRENT / sum(Runes2P.cross_section_weights(radial) .* j)
    return (; conductivity, electric_field = j0 .* j ./ conductivity)
end

"""
    build_initialization(start, electric_field; scheme=:tcdf, compat=:dream)

The system of runs 1 and 2 of the case (generate.py:193-303) on the grids of `start`, with
the field `electric_field` per radius: the hot-tail distribution with the full collision
frequencies and the D background alone, the field and the temperature held at their
prescribed values, DREAM's corrected conductivity, the fluid ionization of D, the explicit
particle source with DREAM's default Maxwellian shape, and the flux through `p_max` into the
runaway density, without avalanche or kinetic ionization (set in the phases only). The field
and the temperature stay unknowns, held by algebraic rows, since the kinetic terms read them
from the state. Ampère's law, the plasma current and the flux at the edge and the wall stay
as in the phases: DREAM computes them with the field prescribed, and the ionization phase
starts from their values.
"""
function build_initialization(start, electric_field; scheme::Symbol = :tcdf,
        compat::Symbol = :dream)
    radial = RadialGrid(start["r_f"])
    grid = cylindrical_grid(MomentumGrid(start["p1_f"], start["p2_f"]), radial)
    nr = radial_cell_count(radial)
    prescribed = PlasmaState(; cold_electron_temperature = fill(10.0, nr),
        ions = background_ions(nr), electric_field = zeros(nr),
        magnetic_field = fill(start["B"], nr))
    (; region, fp, conductivity) = kinetic_terms(grid, prescribed; scheme, compat)
    minus(name) = LinearTerm(name; coefficient = -1)
    current(; kwargs...) = CurrentMomentTerm(:f, grid; pitch_rule = :cell_center, kwargs...)
    # the initial values of the moments and the closures come from their rows
    derived(name, terms...) = Equation(name, Algebraic(), terms...;
        initial_value = :from_equation)
    unknowns = vcat(
        induction_unknowns(radial; electric_field_scale = 1.0, current_density_scale = 1e6),
        [
            ion_densities_unknown(BACKGROUND),
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
    # Faraday's law gives way to the prescribed field; the other induction rows stay
    induction = [eq
                 for eq in induction_equations(radial; wall_radius = WALL_RADIUS,
                         loop_voltage_at_wall = 0.0)
                 if eq.unknown !== :electric_field]
    population = only(population_equations(BACKGROUND; hot_density = :hot_electron_density,
        runaway_density = :runaway_density))
    equations = vcat(induction,
        [
            Equation(:electric_field, Algebraic(), PrescribedValueTerm(electric_field),
                minus(:electric_field)),
            Equation(:cold_electron_temperature, Algebraic(),
                PrescribedValueTerm(initial_temperature(radial.r_centers)),
                minus(:cold_electron_temperature)),
            Equation(:ion_densities, TimeDerivative(), IonRateTerm(BACKGROUND)),
            derived(:cold_electron_density, population.terms...),
            derived(:hot_electron_density, DensityMomentTerm(:f, grid; region),
                minus(:hot_electron_density)),
            derived(:total_electron_density,
                WeightedSumTerm(:ion_densities, charge_state_nuclear_charges(BACKGROUND)),
                minus(:total_electron_density)),
            derived(:free_electron_density,
                WeightedSumTerm(:ion_densities, free_electron_weights(BACKGROUND)),
                minus(:free_electron_density)),
            # DREAM's explicit particle source without avalanche, dn_free/dt = S, with the default
            # Maxwellian shape (py/DREAM/Settings/Equations/HotElectronDistribution.py:55-56)
            Equation(:particle_source, TimeDerivative(:free_electron_density),
                LinearTerm(:particle_source)),
            Equation(:f, TimeDerivative(), fp,
                ParticleSourceTerm(:particle_source, grid, :cold_electron_temperature;
                    shape = :relativistic_maxwellian)),
            Equation(:runaway_density, TimeDerivative(), MomentumBoundaryFluxTerm(:f, fp)),
            derived(:runaway_current_density,
                LinearTerm(:runaway_density;
                    coefficient = Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT),
                minus(:runaway_current_density)),
            derived(:hot_electron_current_density, current(; region),
                minus(:hot_electron_current_density)),
            derived(
                :ohmic_current_density, current(), minus(:hot_electron_current_density),
                conductivity, minus(:ohmic_current_density)),
            derived(:total_current_density, LinearTerm(:ohmic_current_density),
                LinearTerm(:hot_electron_current_density), LinearTerm(:runaway_current_density),
                minus(:total_current_density)),
            thermal_energy_equation()])
    return (; layout, equations, grid, radial, nr, electric_field)
end

# the given part of the state at the start of runs 1 and 2: the Maxwellian of the free
# electrons at the temperature of the case (generate.py:231), the D background and the
# prescribed field
function initialization_state(case)
    layout = case.layout
    u = zeros(length(layout))
    temperature = initial_temperature(case.radial.r_centers)
    unknown_values(u, layout, unknown_id(layout, :cold_electron_temperature)) .= temperature
    unknown_values(u, layout, unknown_id(layout, :electric_field)) .= case.electric_field
    densities = only(background_ions(case.nr)).charge_state_densities
    ions_id = unknown_id(layout, :ion_densities)
    for ir in 1:case.nr
        radial_view(u, layout, ions_id, ir) .= densities[:, ir]
    end
    fill_maxwellian!(u, layout, :f, DENSITY, temperature)
    return u
end

# the rest of the state at the start: the moments and the closures from their rows, and the
# flux of Ampère's law from the current
function complete_initialization_state!(u, case, system)
    layout = case.layout
    fill_initial_values!(u, system, 0.0)
    j_tot = unknown_values(u, layout, unknown_id(layout, :total_current_density))
    flux = initial_poloidal_flux(case.radial, collect(j_tot); wall_radius = WALL_RADIUS)
    unknown_values(u, layout, unknown_id(layout, :poloidal_flux)) .= flux.poloidal_flux
    for name in (:plasma_current, :edge_poloidal_flux, :wall_poloidal_flux)
        u[only(unknown_range(layout, unknown_id(layout, name)))] = getfield(flux, name)
    end
    return u
end

# one run of the initialization with the field `electric_field`: 1 s in 3 steps
function run_initialization(start, electric_field; kwargs...)
    case = build_initialization(start, electric_field; kwargs...)
    times = collect(range(0, INITIALIZATION_TIME; length = INITIALIZATION_STEPS + 1))
    u0 = initialization_state(case)
    system = EquationSystem(case.layout, case.equations; time_scale = times[2] - times[1],
        initial_state = u0)
    complete_initialization_state!(u0, case, system)
    solution = integrate(system, u0, times; stepper = FixedSteps())
    return (; case, solution)
end

"""
    initialize(start; scheme=:tcdf, compat=:dream)

Runes2P's initialization of the case on the grids of `start`, as DREAM's runs 0 to 2
(generate.py:141-303): the field of the prescribed current from the conductivity of the
initial plasma (`initial_field`), run 1 with that field from the Maxwellian, then run 2,
again from the Maxwellian, with the field of run 1 rescaled by `I_p / I_p(end)` of run 1
(generate.py:295-303), so that its current ends at the plasma current. Returns
`(; field, first, second)`, with `field` of `initial_field` and the case and the solution of
each run.
"""
function initialize(start; kwargs...)
    field = initial_field(RadialGrid(start["r_f"]))
    first = run_initialization(start, field.electric_field; kwargs...)
    layout = first.case.layout
    current = only(unknown_range(layout, unknown_id(layout, :plasma_current)))
    final = first.solution.states[:, end]
    rescaled = unknown_values(final, layout, unknown_id(layout, :electric_field)) .*
               PLASMA_CURRENT ./ final[current]
    second = run_initialization(start, collect(rescaled); kwargs...)
    return (; field, first, second)
end

# the entries of the unknown `name` in a state of `layout`, over all radii
function _entries(layout, name)
    reduce(vcat, collect.(Runes2P._unknown_ranges(layout, unknown_id(layout, name))))
end

"""
    initial_state_after_initialization(case, initialization, u)

The state at the start of the ionization phase of `case` (`build`) from the state `u` at the
end of the initialization of the case `initialization` (`build_initialization`), as DREAM's
`fromOutput(..., ignore = ['n_i', 'N_i', 'W_i'])` (generate.py:311, 394): every unknown of the
initialization by name but the ions and the free-electron density and the particle source,
which follow from them, and the ions from the settings (generate.py:159, 439-440), the fully
ionized D background at the uniform density and the temperature of the case, the injected D
and Ar neutral at 1 eV. The free-electron density follows from the ions; the particle source
starts at zero. The cold and total densities of the initialization are kept, as in DREAM,
and their rows set them from the new ions in the first step.
"""
function initial_state_after_initialization(case, initialization, u)
    layout, source = case.layout, initialization.layout
    state = zeros(length(layout))
    for unknown in source.unknowns
        name = unknown.name
        if name in (:ion_densities, :free_electron_density, :particle_source)
            continue
        end
        state[_entries(layout, name)] .= u[_entries(source, name)]
    end
    temperature = initial_temperature(case.radial.r_centers)
    ions_id = unknown_id(layout, :ion_densities)
    energy_id = unknown_id(layout, :ion_thermal_energy)
    injected = Dict("D_inj" => INJECTED_DEUTERIUM, "Ar" => INJECTED_ARGON)
    for ir in 1:case.nr
        states = radial_view(state, layout, ions_id, ir)
        energies = radial_view(state, layout, energy_id, ir)
        for (s, name) in enumerate(COMPOSITION.names)
            components = species_components(COMPOSITION, s)
            if name == "D"
                states[components[2]] = DENSITY
                energies[s] = THERMAL_ENERGY_FACTOR * DENSITY * temperature[ir]
            else
                states[components[1]] = injected[name] * DENSITY
                energies[s] = THERMAL_ENERGY_FACTOR * injected[name] * DENSITY *
                              INJECTED_TEMPERATURE
            end
        end
    end
    free = unknown_values(state, layout, unknown_id(layout, :free_electron_density))
    for ir in 1:case.nr
        free[ir] = sum(free_electron_weights(COMPOSITION) .*
                       radial_view(state, layout, ions_id, ir))
    end
    return state
end

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
    data["B"] = MAGNETIC_FIELD
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

# the kinetic part shared by the phases (`build`) and the initialization
# (`build_initialization`) on `grid`, with the ions of `prescribed` as unknowns: the collision
# settings, the hot region, the Fokker-Planck term of f with the field of the unknown and DREAM's
# corrected conductivity of the ohmic current
function kinetic_terms(grid, prescribed; scheme, compat)
    settings = collision_settings(compat)
    # the hot region: DREAM's 5 thermal momenta for compat = dream; for runes2p, which splits the
    # ionization with CHIANTI's cross sections, the minimum of that split, 7
    factor = if compat === :dream
        THRESHOLD
    else
        Runes2P.MINIMUM_SPLIT_THRESHOLD_FACTORS[:chianti]
    end
    region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor))
    plasma = PlasmaFromUnknowns(prescribed; ions = :ion_densities, temperature_floor = 0.01)
    operator = AdvectionDiffusion(grid; scheme, p_upper_boundary = :f_zero)
    fp = FokkerPlanckTerm(
        :f, operator; plasma, settings, synchrotron = compat === :runes2p,
        electric_field = :electric_field,
        lagged = compat === :dream ? () : :ions)
    conductivity = ConductivityTerm(:electric_field,
        ConductivityModel(
            PlasmaFromUnknowns(prescribed; ions = :ion_densities,
                temperature_floor = 0.01);
            kinetic = :dream_fit))
    return (; settings, region, fp, conductivity)
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
    (; settings, region, fp, conductivity) = kinetic_terms(grid, prescribed; scheme, compat)
    minus(name) = LinearTerm(name; coefficient = -1)
    current(; kwargs...) = CurrentMomentTerm(:f, grid; pitch_rule = :cell_center, kwargs...)
    # the cross sections of the kinetic ionization: DREAM's form and fits for compat = dream,
    # CHIANTI's continued by the Bethe theory for runes2p
    kinetic = KineticIonizationTerm(COMPOSITION, :f, grid; region,
        ionization_cross_section_form = compat === :dream ? :dream : :chianti)
    fluid_ions = (ions = :ion_densities,)
    # the exchange among the ions: DREAM's logarithm with T_e and DREAM's masses for compat =
    # dream, the NRL logarithm and the standard atomic weights for runes2p
    exchange = compat === :dream ? (ion_coulomb_logarithm = :dream, mass_table = :dream) :
               NamedTuple()
    # Gould's relativistic corrections of the bremsstrahlung: DREAM's for compat = dream
    gould = compat === :dream ? (bremsstrahlung_corrections = :dream,) : NamedTuple()
    split = compat === :dream ? NamedTuple() : (kinetic_ionization = kinetic,)
    external = ExternalAvalancheTerm(prescribed; p_cutoff = p_max, settings, fluid_ions...)
    internal = rosenbluth_putvinski_rate(P_CUT, p_max)
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
            # cold electrons below the threshold alone (IonRateTerm with kinetic_ionization)
            Equation(:ion_densities, TimeDerivative(), IonRateTerm(COMPOSITION; split...), kinetic),
            population_equations(COMPOSITION; hot_density = :hot_electron_density,
                runaway_density = :runaway_density)...,
            Equation(:hot_electron_density, Algebraic(),
                DensityMomentTerm(:f, grid; region), minus(:hot_electron_density)),
            Equation(:total_electron_density, Algebraic(),
                WeightedSumTerm(:ion_densities, charge_state_nuclear_charges(COMPOSITION)),
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

# the quantities of the comparison with DREAM per radius, (DREAM's name => Runes2P's unknown)
const COMPARED = ("T_cold" => :cold_electron_temperature, "E_field" => :electric_field,
    "n_cold" => :cold_electron_density, "n_hot" => :hot_electron_density,
    "n_tot" => :total_electron_density, "j_ohm" => :ohmic_current_density,
    "j_hot" => :hot_electron_current_density, "psi_p" => :poloidal_flux,
    "W_cold" => :cold_thermal_energy, "n_re" => :runaway_density,
    "j_re" => :runaway_current_density)

"""
    phase_differences(case, solution, start; per_radius=("n_re", "j_re"))

The largest relative difference of each quantity of a phase of `case` against DREAM's state
`start` (`read_dream_state`) over the steps of `solution` after the first: of the radial
quantities the largest difference over the radii relative to DREAM's largest value over the
radii at each step, of `f_hot`, `n_i` and `W_i` the same over the cells, charge states or
species of each radius, and of `I_p` the difference relative to DREAM's value. The quantities
of `per_radius` are compared at each radius relative to DREAM's largest value at that radius
over the phase: the runaway density differs by 15 orders of magnitude between the radii.
Returns a `Dict` of DREAM's names.
"""
function phase_differences(case, solution, start; per_radius = ("n_re", "j_re"))
    layout = case.layout
    steps = 2:length(solution.times)
    ours_at(m, ours) = unknown_values(solution.states[:, m], layout, unknown_id(layout, ours))
    errors = Dict{String, Float64}()
    for (name, ours) in COMPARED
        errors[name] = if name in per_radius
            maximum(abs(ours_at(m, ours)[ir] - start[name][ir, m]) /
                    maximum(abs, start[name][ir, :])
            for ir in 1:case.nr, m in steps)
        else
            maximum(maximum(abs, ours_at(m, ours) .- start[name][:, m]) /
                    maximum(abs, start[name][:, m])
            for m in steps)
        end
    end
    f_id, ions_id = unknown_id(layout, :f), unknown_id(layout, :ion_densities)
    energy_id = unknown_id(layout, :ion_thermal_energy)
    errors["f_hot"] = maximum(maximum(abs,
                                  kinetic_view(solution.states[:, m], layout, f_id, ir) .-
                                  start["f_hot"][:, :, ir, m]) /
                              maximum(abs, start["f_hot"][:, :, ir, m])
    for ir in 1:case.nr, m in steps)
    errors["n_i"] = maximum(maximum(abs,
                                radial_view(solution.states[:, m], layout, ions_id, ir) .-
                                start["n_i"][ir, :, m]) /
                            maximum(abs, start["n_i"][ir, :, m])
    for ir in 1:case.nr, m in steps)
    errors["W_i"] = maximum(maximum(abs,
                                radial_view(solution.states[:, m], layout, energy_id, ir) .-
                                start["W_i"][ir, :, m]) /
                            maximum(abs, start["W_i"][ir, :, m])
    for ir in 1:case.nr, m in steps)
    current = only(unknown_range(layout, unknown_id(layout, :plasma_current)))
    errors["I_p"] = maximum(abs(solution.states[current, m] - start["I_p"][m]) /
                            abs(start["I_p"][m]) for m in steps)
    return errors
end

# the closures of the assembled system (`budgets`), independent of DREAM: round-off bounds, at
# least 50 times the values measured at -O0 in the phases started from DREAM's state
# (test_disruption_dream.jl; compat = dream / runes2p, ionization and quench). The chain of
# test_disruption_chain_dream.jl stays below them as well (closure 1.1e-14, field 5.5e-11 at most,
# at the default optimisation level).
#! format: off
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
