# Avalanche multiplication of a seed of fast electrons by knock-on collisions, on the
# setup of DREAM's examples/avalanche/generateCylindrical.py (Ola Embréus): 10 eV,
# E = `e_over_ec` times the critical field of the total electron density, 1e19 m⁻³
# ionized and 5e18 m⁻³ neutral deuterium, 5e18 m⁻³ neutral argon, superthermal
# collisions with partial screening and the energy-dependent Coulomb logarithm, momentum
# grid to p = 30 split at p = 0.7 (40 of 200 cells below), 29 pitch cells with geometric
# growth, knock-on cutoff p = 0.1, time step 2 ms. The seed is a Maxwell-Jüttner
# distribution at 100 keV with 1 m⁻³. Switches beside those of common.jl:
# `treatment=explicit_primary|explicit|implicit` (the primary distribution at the previous
# time level, as DREAM and the default; the whole term there; or the term in the Jacobian,
# which needs `solver=gmres`), `cutoff=`, `np=`, `np_low=`, `nxi=`,
# `e_over_ec=`. The report gives the density growth rate per second over the run.
#
#   julia --project=examples examples/avalanche.jl
#   julia --project=examples examples/avalanche.jl treatment=implicit solver=gmres dt=8e-3
#   julia --project=examples examples/avalanche.jl scheme=tcdf solver=gmres np=400
#   julia --project=examples examples/avalanche.jl cutoff=0.2 e_over_ec=30
include(joinpath(@__DIR__, "common.jl"))

nr = 1
temperature = 10.0
ionized_deuterium, neutral_deuterium, neutral_argon = 1e19, 5e18, 5e18
total_electron_density = ionized_deuterium + neutral_deuterium + 18 * neutral_argon
# the example's Coulomb logarithm and critical field of the total density
lnlambda = 14.9 - 0.5 * log(neutral_deuterium / 1e20) + log(temperature / 1e3)
e, eps0 = Constants.ELEMENTARY_CHARGE, Constants.VACUUM_PERMITTIVITY
me, c = Constants.ELECTRON_MASS, Constants.SPEED_OF_LIGHT
critical_total = total_electron_density * lnlambda * e^3 / (4pi * eps0^2 * me * c^2)
field = number("e_over_ec", 10.0) * critical_total

np, nxi = integer("np", 200), integer("nxi", 29)
np_low = integer("np_low", round(Int, 40 * np / 200))
xi_faces = reverse(1 .- 2 .* range(0, 1; length=nxi + 1) .^ 1.5)
mom = MomentumGrid(biuniform_p_faces(np, np_low, 0.0, 0.7, 30.0), xi_faces)
grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(nr, 0.1)))
deuterium = zeros(2, nr)
deuterium[1, :] .= neutral_deuterium
deuterium[2, :] .= ionized_deuterium
argon = zeros(19, nr)
argon[1, :] .= neutral_argon
plasma = PlasmaState(cold_electron_temperature=fill(temperature, nr), cold_electron_density=fill(ionized_deuterium, nr),
                     ions=[IonSpecies("D", 1, deuterium), IonSpecies("Ar", 18, argon)],
                     electric_field=fill(field, nr), magnetic_field=fill(1e-6, nr))
settings = CollisionSettings(mode=symbol("mode", :superthermal), screening=symbol("screening", :partial),
                             coulomb_logarithm=symbol("lnlambda", :energy_dependent), ion_size_table=symbol("ion_size_table", :dream))
schemes = scheme_switches()
op = fokker_planck_operator(grid, plasma, settings; schemes...)

cutoff = number("cutoff", 0.1)
knockon = KnockOnOperator(grid, grid, cutoff)
treatment = symbol("treatment", :explicit_primary)
layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid; scale=1.0)])
system = EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op; jacobian=jacobian_switch()),
                                          KnockOnTerm(:f, knockon; target_density=fill(total_electron_density, nr), treatment))])
solver = solver_from_switches(solver=treatment === :implicit ? :gmres : :lu)
println("scheme p/ξ: ", op.p_scheme, "/", op.xi_scheme, ", knock-on ", treatment, " with cutoff p = ", cutoff,
        ", E/E_c,tot = ", round(field / critical_total; digits=2), ", ", np, " × ", nxi, " cells")

f0 = maxwellian(grid, 1e5, 1.0, nr)
times, save_times = times_from_switches(dt=2e-3, t_end=0.2, saves=11)

integrate(system, vec(f0), times[1:min(3, end)]; solver)   # compile
wall = @elapsed solution = integrate(system, vec(f0), times; solver, save_times)
report(solution, grid; wall)
n_start = density(grid, reshape(solution.states[:, 1], size(grid)))[1]
n_end = density(grid, reshape(solution.states[:, end], size(grid)))[1]
@printf("mean growth rate over the run: %.4e /s (density ×%.3f over %.3f s)\n", log(n_end / n_start) / last(times),
        n_end / n_start, last(times))
