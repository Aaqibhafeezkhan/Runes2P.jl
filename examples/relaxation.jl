# Hot electrons under a fixed plasma and electric field: a Maxwellian of temperature `T0`
# (the plasma's by default) relaxes in a deuterium plasma at `T`, `n` and a field `E` or
# `e_over_ec`, with the momentum grid, scheme, collision and solver switches of
# common.jl. The report lists the Newton statistics, the negative entries and the
# density at each save; `p_star=` adds the density above that momentum (the runaway or
# hot-tail fraction of the run).
#
#   julia --project=examples examples/relaxation.jl
#   julia --project=examples examples/relaxation.jl scheme=van_albada solver=gmres np=400
#   julia --project=examples examples/relaxation.jl grid=production e_over_ec=10 T=1e3 t_end=2e-3
#   julia --project=examples examples/relaxation.jl scheme=tcdf jacobian=upwind solver=gmres operator=stored
#   julia --project=examples examples/relaxation.jl mode=superthermal screening=partial n_ar=1e19
#
# Values: deuterium at 1 keV and 1e20 m⁻³ with E = 5 E_c (the Dreicer regime), 200
# momentum cells to p = 2 and 15 pitch cells, dt = 1e-5 s to 1 ms.
include(joinpath(@__DIR__, "common.jl"))

nr = 1
mom = momentum_grid_from_switches(np=200, nxi=15, p_max=2.0)
grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(nr, 1.0)))
plasma = plasma_from_switches(nr; T=1e3, n=1e20, e_over_ec=5.0)
settings = collision_settings_from_switches()
schemes = scheme_switches()
op = fokker_planck_operator(grid, plasma, settings; schemes..., synchrotron=boolean("synchrotron", false))
println("scheme p/ξ: ", op.p_scheme, "/", op.xi_scheme, ", Jacobian ", jacobian_switch(), ", p_max boundary ", op.p_upper_boundary,
        ", collisions ", settings.mode, ", E/E_c = ",
        round(plasma.electric_field[1] / critical_field(plasma.cold_electron_density[1],
              thermal_coulomb_logarithm(plasma.cold_electron_temperature[1], plasma.cold_electron_density[1])); digits=2))

f0 = maxwellian(grid, number("T0", plasma.cold_electron_temperature[1]), plasma.cold_electron_density[1], nr)
layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid; scale=maximum(f0), nonnegative=boolean("nonnegative", false))])
system = EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op; jacobian=jacobian_switch()))])
solver = solver_from_switches()
times, save_times = times_from_switches(dt=1e-5, t_end=1e-3)

integrate(system, vec(f0), times[1:min(3, end)]; solver)   # compile
wall = @elapsed solution = integrate(system, vec(f0), times; solver, save_times)
report(solution, grid; wall)

p_star = number("p_star", 0.0)
if p_star > 0
    weights = phase_space_jacobian(grid, 1) .* mom.p_widths .* mom.xi_widths'
    above = mom.p_centers .> p_star
    for (k, t) in enumerate(solution.times)
        f = reshape(solution.states[:, k], size(grid))
        fraction = sum(weights[above, :] .* f[above, :, 1]) / sum(weights .* f[:, :, 1])
        @printf("t = %9.3e s: fraction above p = %.2f: %.4e\n", t, p_star, fraction)
    end
end
