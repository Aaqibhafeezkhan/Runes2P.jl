# Setups built from DREAM's examples, shared by the tests.

"""
    avalanche_example(; momentum_cells=200, pitch_cells=29,
                      momentum_cells_below_split=round(Int, 40 momentum_cells/200),
                      radial_cells=1, scheme=:upwind, over_critical=10.0)

Grid and plasma of DREAM's examples/avalanche/generateCylindrical.py (Ola Embréus,
2026-03-05): 10 eV, E = 10 E_c,tot, 1e19 m⁻³ ionized and 5e18 m⁻³ neutral D, 5e18 m⁻³
neutral Ar, superthermal collisions with partial screening and the energy-dependent
Coulomb logarithm, B = 1e-6 T; momentum grid to p = 30 split at p = 0.7 (40 of 200 cells
below), 30 pitch faces with geometric growth (exponent 1.5), knock-on cutoff p = 0.1,
time step 2 ms (100 steps to 0.2 s). The cell counts scale the example's grid for
resolution scans (examples/convergence/run.py halves and doubles them); `over_critical`
scales the field.

Differences from the example: the example uses the TCDF flux limiter, and `scheme` sets
the interpolation here (upwind by default); the cold-electron density is prescribed at the
ionized deuterium density; the particle source and the Rosenbluth-Putvinski source from
`n_re` are left out.

Returns a named tuple with the kinetic `grid`, the `plasma`, the Fokker-Planck operator
`fokker_planck`, the knock-on `cutoff`, the `total_electron_density` (free plus bound),
the `field`, the example's `time_step` and `initial_state(layout, id; density=1,
temperature=1e5)`, a state vector with a Maxwell-Jüttner distribution in kinetic unknown
`id` (the example's initial distribution: 1 m⁻³ at 100 keV).
"""
function avalanche_example(; momentum_cells::Integer = 200, pitch_cells::Integer = 29,
        momentum_cells_below_split::Integer = round(Int, 40 * momentum_cells / 200),
        radial_cells::Integer = 1, scheme::Symbol = :upwind, over_critical::Real = 10.0)
    temperature = 10.0
    ionized_deuterium, neutral_deuterium, neutral_argon = 1e19, 5e18, 5e18
    total_electron_density = ionized_deuterium + neutral_deuterium + 18 * neutral_argon
    free_electron_density = neutral_deuterium   # as in the example's lnΛ formula
    lnlambda = 14.9 - 0.5 * log(free_electron_density / 1e20) + log(temperature / 1e3)
    e, eps0 = Constants.ELEMENTARY_CHARGE, Constants.VACUUM_PERMITTIVITY
    me, c = Constants.ELECTRON_MASS, Constants.SPEED_OF_LIGHT
    critical_total = total_electron_density * lnlambda * e^3 / (4pi * eps0^2 * me * c^2)
    field = over_critical * critical_total

    xi_faces = reverse(1 .- 2 .* range(0, 1; length = pitch_cells + 1) .^ 1.5)
    p_faces = biuniform_p_faces(momentum_cells, momentum_cells_below_split, 0.0, 0.7, 30.0)
    mom = MomentumGrid(p_faces, xi_faces)
    nr = radial_cells
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(nr, 0.1)))
    deuterium = zeros(2, nr)
    deuterium[1, :] .= neutral_deuterium
    deuterium[2, :] .= ionized_deuterium
    argon = zeros(19, nr)
    argon[1, :] .= neutral_argon
    plasma = PlasmaState(cold_electron_temperature = fill(temperature, nr),
        cold_electron_density = fill(ionized_deuterium, nr),
        ions = [IonSpecies("D", 1, deuterium), IonSpecies("Ar", 18, argon)],
        electric_field = fill(field, nr), magnetic_field = fill(1e-6, nr))
    settings = CollisionSettings(mode = :superthermal, screening = :partial,
        coulomb_logarithm = :energy_dependent, ion_size_table = :dream,
        screening_window = :dream, bremsstrahlung = :neglect)
    fokker_planck = fokker_planck_operator(g, plasma, settings; scheme, synchrotron = false)
    function initial_state(layout, id; density = 1.0, temperature = 1e5)
        theta = temperature / Constants.ELECTRON_REST_ENERGY
        f = [exp(-(sqrt(1 + p^2) - 1) / theta)
             for p in mom.p_centers, _ in 1:pitch_cell_count(mom), _ in 1:nr]
        f .*= density ./ reshape(Runes2P.density(g, f), 1, 1, nr)
        u = zeros(length(layout))
        for ir in 1:nr
            kinetic_view(u, layout, id, ir) .= f[:, :, ir]
        end
        return u
    end
    return (; grid = g, plasma, fokker_planck, cutoff = 0.1, total_electron_density, field,
        time_step = 2e-3, initial_state)
end

"""
    runaway_heating_term(compat, source, grid, plasma)

The heating of the cold electrons by the electrons of a runaway grid, for the row of their
temperature, with `source` the [`FokkerPlanckTerm`](@ref) of the runaway distribution on
`grid` and `plasma` the plasma of the heating. `compat = :dream` gives DREAM's moment
`FrictionEnergyTransferTerm` (`src/Settings/Equations/T_cold.cpp:341-354`, DREAM v26.5), and
`compat = :runes2p` Runes2P's default, the energy change of the electron-electron operator of
`source` (`free_electron_energy_transfer_term`), which needs the collision mode `:full` or
`:superthermal` in `source`.
"""
function runaway_heating_term(compat::Symbol, source::FokkerPlanckTerm, grid, plasma)
    if compat === :dream
        return FrictionEnergyTransferTerm(source.unknown, grid, plasma)
    elseif compat === :runes2p
        return free_electron_energy_transfer_term(source)
    else
        throw(ArgumentError("compat must be :dream or :runes2p, not $compat"))
    end
end

"""
    runaway_heating_example(; compat=:dream, lagged=compat === :dream ? :all : :ions)

The case `heating` of reference/dream/cases/initial_state.py: the runaway grid of DREAM's
examples/init_f_re/generate.py (Mathias Hoppe, 2024-11-22) on one radius with the cold
temperature self-consistent, in the 20 steps of the case over 5 µs. The self-consistent
temperature, the forward start of `f_re` and the run of 5 µs are choices of this package.
Deuterium fully ionized at 5e19 m⁻³ and 100 eV at the start, the prescribed field of
1.5 V/m, B = 5 T, a = 0.22 m; the runaway grid alone (no hot-tail grid) from p = 0 to 50
with 100 uniform momentum cells and 40 pitch cells uniform in θ in two regions, 32 cells up
to θ = 0.5; the ultra-relativistic collision frequencies with partial screening and the
thermal Coulomb logarithm, no bremsstrahlung and no synchrotron loss; upwind interpolation
and DREAM's `BC_PHI_CONST` at `p_max`; `f_re` started from `n_re = 1e16 m⁻³` in the forward
shape ([`fill_runaway_distribution!`](@ref)). The row of the temperature holds the ohmic
heating with DREAM's conductivity, the radiated power of DREAM's mode (the bremsstrahlung of
Gould with DREAM's corrections for the fully ionized deuterium) and the heating by `f_re` of
[`runaway_heating_term`](@ref); the cold density is the free-electron density minus the
density of `f_re`, as DREAM's `n_cold`.

`compat` switches the heating by `f_re`: DREAM's moment for `:dream`, Runes2P's energy change
of the electron-electron operator for `:runes2p`. That operator has no ultra-relativistic
form, so `:runes2p` takes the superthermal frequencies, of which the ultra-relativistic ones
are the limit `p ≫ 1` and which DREAM's moment takes too. Every other setting is DREAM's in
both. `lagged` goes to the [`FokkerPlanckTerm`](@ref) of `f_re`: `:all` takes the collision
frequencies at the plasma of the start of the step, as DREAM's semi-implicit steps
(`LINEAR_IMPLICIT`) do; `:ions`, the default at `:runes2p`, is Runes2P's default treatment
of the plasma in the term, with only the ion densities taken at the start of the step.

Needs the ADF11 tables of deuterium. Returns a named tuple with the `system`, its `layout`,
the `initial_state`, the `times` of the case, the kinetic `grid`, the Fokker-Planck term
`fokker_planck` of `f_re`, the `heating` term and the `prescribed` part of the plasma.
"""
function runaway_heating_example(; compat::Symbol = :dream,
        lagged = compat === :dream ? :all : :ions)
    compat in (:dream, :runes2p) ||
        throw(ArgumentError("compat must be :dream or :runes2p, not $compat"))
    # reference/dream/cases/initial_state.py, after examples/init_f_re/generate.py
    field = 1.5                  # V/m, initial_state.py:67 (generate.py:35)
    deuterium = 5e19             # m⁻³, initial_state.py:68 (generate.py:36)
    temperature = 100.0          # eV, initial_state.py:69 (generate.py:37), the start (:181)
    runaway_density = 1e16       # m⁻³, initial_state.py:70 (generate.py:57)
    magnetic_field, minor_radius = 5.0, 0.22   # initial_state.py:74 (generate.py:71-72)
    # 100 × 40 cells to p = 50 (initial_state.py:71, generate.py:61-63), the pitch grid
    # biuniform in θ with 0.8 of the 40 cells below θ = 0.5 (initial_state.py:125, generate.py:64)
    nr = 1                       # initial_state.py:177
    radial = RadialGrid(uniform_r_faces(nr, minor_radius))
    grid = cylindrical_grid(
        MomentumGrid(uniform_p_faces(100, 0.0, 50.0), biuniform_theta_xi_faces(40, 32, 0.5)),
        radial)
    # 20 steps (initial_state.py:72, 183) over 5 µs, a choice of this package
    # (initial_state.py:73, 182)
    times = collect(range(0.0, 5e-6; length = 21))
    # the collisions: partial screening and the ultra-relativistic mode (initial_state.py:109-110,
    # generate.py:31-32), DREAM's thermal Coulomb logarithm and no bremsstrahlung, the defaults of
    # CollisionHandler (py/DREAM/Settings/CollisionHandler.py:30-32); the ion sizes and the
    # window of the screening are DREAM's and have no effect on fully ionized deuterium
    settings = CollisionSettings(;
        mode = compat === :dream ? :ultra_relativistic : :superthermal,
        screening = :partial, coulomb_logarithm = :thermal, bremsstrahlung = :neglect,
        ion_size_table = :dream, screening_window = :dream)
    # deuterium prescribed and fully ionized (initial_state.py:118); the temperature, the cold
    # density and the field are unknowns
    ions = [IonSpecies("D", 1, 1, fill(deuterium, nr))]
    prescribed = PlasmaState(; cold_electron_temperature = fill(temperature, nr), ions,
        electric_field = zeros(nr), magnetic_field = fill(magnetic_field, nr))
    plasma = PlasmaFromUnknowns(prescribed)
    # upwind interpolation (initial_state.py:126, generate.py:68), DREAM's default BC_PHI_CONST
    # at p_max and no synchrotron loss, DREAM's default; the field of the unknown
    operator = fokker_planck_operator(grid, prescribed, settings; synchrotron = false,
        scheme = :upwind, p_upper_boundary = :constant_flux, field_in_coefficients = false)
    fokker_planck = FokkerPlanckTerm(:runaway_distribution, operator; plasma, settings,
        synchrotron = false, electric_field = :electric_field, lagged)
    heating = runaway_heating_term(compat, fokker_planck, grid, plasma)
    minus(name) = LinearTerm(name; coefficient = -1)
    layout = UnknownLayout(radial,
        [RadialUnknown(:electric_field),
            RadialUnknown(:ohmic_current_density; scale = 1e6),
            RadialUnknown(:cold_electron_temperature; scale = 100.0),
            RadialUnknown(:cold_electron_density; scale = 1e19),
            RadialUnknown(:cold_thermal_energy; scale = 1e3),
            KineticUnknown(:runaway_distribution, grid; scale = 1e16)])
    # the self-consistent temperature, a choice of this package (initial_state.py:178, 180):
    # the ohmic heating, the radiated power and the heating by f_re of DREAM's row
    # (src/Settings/Equations/T_cold.cpp:175-176, 187-190, 341-354), with DREAM's radiation:
    # fully ionized deuterium radiates only bremsstrahlung
    equations = [
        Equation(:electric_field, Algebraic(),
            PrescribedValueTerm(fill(field, nr)), minus(:electric_field)),
        Equation(:ohmic_current_density, Algebraic(),
            ConductivityTerm(:electric_field,
                ConductivityModel(PlasmaFromUnknowns(prescribed; density = nothing))),
            minus(:ohmic_current_density)),
        Equation(:cold_electron_density, Algebraic(),
            PrescribedValueTerm(fill(deuterium, nr)),
            DensityMomentTerm(:runaway_distribution; coefficient = -1),
            minus(:cold_electron_density)),
        thermal_energy_equation(),
        Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy),
            ohmic_heating_term(),
            RadiatedPowerTerm(prescribed; radiation = :dream, bremsstrahlung_corrections = :dream),
            heating),
        Equation(:runaway_distribution, TimeDerivative(), fokker_planck)]
    u0 = zeros(length(layout))
    unknown_values(u0, layout, unknown_id(layout, :electric_field)) .= field
    unknown_values(u0, layout, unknown_id(layout, :cold_electron_temperature)) .= temperature
    unknown_values(u0, layout, unknown_id(layout, :cold_electron_density)) .= deuterium -
                                                                              runaway_density
    # INIT_FORWARD, a choice of this package (initial_state.py:178-179), from n_re
    # (initial_state.py:120)
    fill_runaway_distribution!(u0, layout, :runaway_distribution, runaway_density;
        shape = :forward, electric_field = field)
    unknown_values(u0, layout, unknown_id(layout, :ohmic_current_density)) .= field .*
                                                                              ConductivityModel(prescribed)(0.0)
    system = EquationSystem(layout, equations; time_scale = 1e-6, initial_state = u0)
    return (; system, layout, initial_state = u0, times, grid, fokker_planck, heating,
        prescribed)
end
