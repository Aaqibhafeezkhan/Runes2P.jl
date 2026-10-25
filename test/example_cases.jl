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
