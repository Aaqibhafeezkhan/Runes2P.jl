# A cooling plasma (a thermal quench in the slide-away regime): the
# temperature decays exponentially from `T` to `T_final` over `t_end`, the collision
# coefficients follow it (PrescribedFokkerPlanckTerm), the field is `e_over_ec` times the
# critical field at the initial temperature. With `nr=` > 1 the radii are coupled by
# Rechester-Rosenbluth radial diffusion of strength `dbb=` (δB/B, with the major radius
# folded in as in DREAM's examples/SPI/SPI_ITER.py; `dbb=0` decouples them) and f = 0 at
# the wall. The report shows where a scheme goes negative as the bulk cools.
#
#   julia --project=examples examples/cooling.jl
#   julia --project=examples examples/cooling.jl scheme=central                (goes negative)
#   julia --project=examples examples/cooling.jl scheme=van_albada solver=gmres
#   julia --project=examples examples/cooling.jl grid=biuniform_270 T_final=10
#   julia --project=examples examples/cooling.jl nr=11 dbb=1e-3 solver=gmres restart=60
#   julia --project=examples examples/cooling.jl checkpoint=cooling checkpoint_dir=run resume=true wall_budget=60 slack=5
#
# Values: the plasma of reference/dream/cases/cooling_positivity.py, deuterium at 1e20 m⁻³
# from 15 keV to 100 eV over 2 ms, E = 20 E_c at 15 keV (slide-away), B = 5.3 T; 200
# momentum cells to p = 2 and 15 pitch cells, dt = 1e-5 s (a choice of this package; see
# the docs page *Choices of this package*); radial values from the SPI example: R = 6.2 m,
# minor radius 2 m.
include(joinpath(@__DIR__, "common.jl"))

const MAJOR_RADIUS = 6.2
const MINOR_RADIUS = 2.0

nr = integer("nr", 1)
mom = momentum_grid_from_switches(np = 200, nxi = 15, p_max = 2.0)
grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(nr, nr == 1 ? 1.0 : MINOR_RADIUS)))
T0, T_final = number("T", 15e3), number("T_final", 100.0)
times, save_times = times_from_switches(dt = 1e-5, t_end = 2e-3)
t_end = last(times)
n, B = number("n", 1e20), number("B", 5.3)
field = has_argument("E") ? number("E", 0.0) :
        number("e_over_ec", 20.0) * critical_field(n, thermal_coulomb_logarithm(T0, n))
temperature(t) = T0 * (T_final / T0)^(t / t_end)
function plasma_at(t)
    PlasmaState(cold_electron_temperature = fill(temperature(t), nr),
        cold_electron_density = fill(n, nr),
        ions = [IonSpecies("D", 1, 1, fill(n, nr))], electric_field = fill(field, nr),
        magnetic_field = fill(B, nr))
end
settings = collision_settings_from_switches()
schemes = scheme_switches()
op = AdvectionDiffusion(grid; schemes...)
terms = EquationTerm[PrescribedFokkerPlanckTerm(
    :f, op, plasma_at; settings, jacobian = jacobian_switch())]

# radial diffusion D_rr = π R (δB/B)² |v_∥| (DREAM RechesterRosenbluthTransport, cylindrical
# form with q = 1), a linear term of its own so the rebuild of the collision
# coefficients leaves it alone
dbb = number("dbb", 1e-3)
if nr > 1 && dbb > 0
    radial = AdvectionDiffusion(grid; scheme = :central, r_boundary = :absorbing)
    for k in axes(radial.diffusion_r_r, 3), j in axes(radial.diffusion_r_r, 2),
        i in axes(radial.diffusion_r_r, 1)
        p, xi = mom.p_centers[i], mom.xi_centers[j]
        radial.diffusion_r_r[i, j, k] = pi * MAJOR_RADIUS * dbb^2 *
                                        Constants.SPEED_OF_LIGHT * abs(p * xi) /
                                        sqrt(1 + p^2)
    end
    push!(terms, FokkerPlanckTerm(:f, radial))
end
println("scheme p/ξ: ", op.p_scheme, "/", op.xi_scheme,
    ", Jacobian ", jacobian_switch(), ", ", nr, " radii",
    nr > 1 ? " (δB/B = $dbb)" : "", ", T from ", T0, " to ", T_final, " eV over ", t_end, " s, E/E_c(T0) = ",
    round(field / critical_field(n, thermal_coulomb_logarithm(T0, n)); digits = 2))

f0 = maxwellian(grid, T0, n, nr)
layout = UnknownLayout(grid.radial_grid,
    [KineticUnknown(:f, grid; scale = maximum(f0), nonnegative = boolean("nonnegative", false))])
system = EquationSystem(layout, [Equation(:f, TimeDerivative(), terms...)])
solver = solver_from_switches()

integrate(system, vec(f0), times[1:min(3, end)]; solver)   # compile
description = Dict(
    "example" => "cooling", "initial_state" => "Maxwellian at $T0 eV and $n m^-3",
    "delta_b_over_b" => dbb, "major_radius" => MAJOR_RADIUS, "minor_radius" => MINOR_RADIUS)
wall = @elapsed solution = integrate(
    system, vec(f0), times; solver, save_times, description, run_options()...)
report(solution, grid; wall)
for (k, t) in enumerate(solution.times)
    f = reshape(solution.states[:, k], size(grid))
    negative = count(<(0), f)
    negative == 0 && continue
    i_first = minimum(idx[1] for idx in findall(<(0), f))
    @printf("t = %9.3e s (T = %8.1f eV): %d negative cells, first at momentum cell %d (p = %.3f), min f/max f = %.2e\n",
        t, temperature(t), negative, i_first, mom.p_centers[i_first],
        minimum(f) / maximum(f))
end
finish(solution)
