# The child process of test_threads_over_radii.jl: two systems on 5 radii, evaluated and stepped
# at the number of Julia threads of this process, their results written into the directory of
# the first argument as raw bytes, which the test compares between thread counts.
#
# The ionization system reaches the four loops that run on threads: the operator of the
# Fokker-Planck term, its coefficients in Float64 and in dual numbers, its plasma columns, and
# the Jacobian of the kinetic ionization with an envelope that follows the temperature. The
# second system takes a full Newton step that makes the free-electron density negative, so a
# trial state throws a DomainError inside a threaded loop and the solver recovers by a smaller
# step.

using Runes2P
using Runes2P: population_equations

const OUT = only(ARGS)
const RADII = 5

function save(name, values)
    write(joinpath(OUT, name), values)
    return nothing
end

statuses(solution) = join(string.(solution.statistics.status), ",")

function hold(name, value)
    Equation(name, Algebraic(), PrescribedValueTerm(value),
        LinearTerm(name; coefficient = -1))
end

function ionization_case()
    composition = IonComposition(["D" => 1, "Ar" => 18])
    hot = cylindrical_grid(
        MomentumGrid(uniform_p_faces(24, 0.0, 0.05), uniform_xi_faces(3)),
        RadialGrid(uniform_r_faces(RADII, 0.5)))
    temperatures = [5.0, 4.5, 4.0, 3.5, 3.0]
    argon = zeros(19, RADII)
    argon[1, :] .= [2e17, 3e17, 4e17, 3e17, 2e17]
    argon[2, :] .= [5e17, 4e17, 3e17, 4e17, 5e17]
    argon[3, :] .= [1e17, 0.0, 1e17, 0.0, 1e17]
    deuterium = [fill(1e18, RADII)'; fill(4e19, RADII)']
    ions = [IonSpecies("D", 1, deuterium), IonSpecies("Ar", 18, argon)]
    prescribed = PlasmaState(;
        cold_electron_temperature = temperatures, ions, electric_field = zeros(RADII),
        magnetic_field = fill(5.0, RADII))
    layout = UnknownLayout(hot.radial_grid,
        [ion_densities_unknown(composition),
            RadialUnknown(:cold_electron_density; scale = 1e19),
            RadialUnknown(:cold_electron_temperature; scale = 10.0),
            KineticUnknown(:f, hot; scale = 1e19)])
    settings = CollisionSettings(;
        mode = :full, screening = :partial, bound_friction = :ionization_split,
        ionization_threshold = 5.0)
    fp = FokkerPlanckTerm(
        :f, AdvectionDiffusion(hot; p_upper_boundary = :f_zero);
        plasma = PlasmaFromUnknowns(prescribed; ions = :ion_densities), settings)
    region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
    equations = [
        Equation(:ion_densities, TimeDerivative(),
            KineticIonizationTerm(composition, :f, hot; region)),
        population_equations(composition)...,
        hold(:cold_electron_temperature, temperatures),
        Equation(:f, TimeDerivative(), fp)]
    u = zeros(length(layout))
    set_ion_densities!(u, layout, ions)
    unknown_values(u, layout, 2) .= [free_electron_density(ions, ir) for ir in 1:RADII]
    unknown_values(u, layout, 3) .= temperatures
    for ir in 1:RADII
        theta = temperatures[ir] / Constants.ELECTRON_REST_ENERGY
        m = momentum_grid(hot, ir)
        kinetic_view(u, layout, 4, ir) .= [1e19 * exp(-p^2 / (2theta)) * (1 + 0.1xi)
                                           for p in m.p_centers, xi in m.xi_centers]
    end
    return EquationSystem(layout, equations; time_scale = 1e-7, initial_state = u), u
end

# a sink of the second charge state that drives it below zero within one full Newton step
struct KneeSink <: EquationTerm
    unknown::Symbol
    rate::Float64
    bottom::Float64
    top::Float64
end
Runes2P.dependencies(term::KneeSink) = [term.unknown]
function Runes2P.rhs!(out, term::KneeSink, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        value = u[unknown_range(layout, id, ir)[2]]
        out[unknown_range(layout, row, ir)[2]] -= term.rate *
                                                  clamp(
            (value - term.bottom) /
            (term.top - term.bottom), 0, 1)
    end
    return out
end
function Runes2P.sparsity!(entries, term::KneeSink, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        push!(entries, (
            unknown_range(layout, row, ir)[2], unknown_range(layout, id, ir)[2]))
    end
    return entries
end

function domain_error_case()
    radial = RadialGrid(uniform_r_faces(RADII, 0.5))
    momentum = MomentumGrid(uniform_p_faces(12, 0.0, 0.5), uniform_xi_faces(3))
    grid = cylindrical_grid(momentum, radial)
    densities = [fill(1e18, RADII)'; fill(1e19, RADII)']
    prescribed = PlasmaState(;
        cold_electron_temperature = fill(50.0, RADII), ions = [IonSpecies("D", 1, densities)],
        electric_field = zeros(RADII), magnetic_field = ones(RADII),
        cold_electron_density = fill(1e19, RADII))
    layout = UnknownLayout(radial,
        [RadialUnknown(:ion_densities; components = 2, scale = 1e19, groups = [1:2]),
            RadialUnknown(:cold_electron_temperature; scale = 10.0),
            RadialUnknown(:cold_electron_density; scale = 1e19), KineticUnknown(:f, grid; scale = 1e16)])
    plasma = PlasmaFromUnknowns(prescribed; ions = :ion_densities,
        density = :cold_electron_density, temperature_floor = 0.01)
    fp = FokkerPlanckTerm(:f, AdvectionDiffusion(grid; scheme = :central); plasma,
        settings = CollisionSettings(; coulomb_logarithm = :energy_dependent),
        synchrotron = false, lagged = ())
    dt = 1e-6
    equations = [
        Equation(:ion_densities, TimeDerivative(),
            KneeSink(:ion_densities, 2e19 / dt, 0.2e19, 0.8e19)),
        hold(:cold_electron_temperature, 50.0), hold(:cold_electron_density, 1e19),
        Equation(:f, TimeDerivative(), fp)]
    u = zeros(length(layout))
    set_ion_densities!(u, layout, [IonSpecies("D", 1, densities)])
    unknown_values(u, layout, 2) .= 50.0
    unknown_values(u, layout, 3) .= 1e19
    for ir in 1:RADII
        kinetic_view(u, layout, 4, ir) .= [1e16 * exp(-p / 0.1)
                                           for p in momentum.p_centers, _ in 1:3]
    end
    return EquationSystem(layout, equations; time_scale = dt, initial_state = u), u, dt
end

write(joinpath(OUT, "threads.txt"), string(Threads.nthreads()))

system, u = ionization_case()
du = zeros(length(u))
rhs!(du, u, system, 0.0)
save("ionization_rhs.bin", du)
J = jacobian_prototype(system)
jacobian!(J, system, u, 0.0)
save("ionization_jacobian.bin", J.nzval)
# a second state, so that the coefficients and the envelopes are rebuilt
v = u .* (1 .+ 1e-3 .* sin.(1:length(u)))
rhs!(du, v, system, 0.0)
save("ionization_rhs_2.bin", du)
jacobian!(J, system, v, 0.0)
save("ionization_jacobian_2.bin", J.nzval)
solution = integrate(system, u, [0.0, 1e-7, 2e-7]; stepper = FixedSteps())
save("ionization_states.bin", solution.states)
write(joinpath(OUT, "ionization_status.txt"), statuses(solution))

system, u, dt = domain_error_case()
solution = integrate(system, u, [0.0, dt]; stepper = FixedSteps())
save("domain_error_states.bin", solution.states)
write(joinpath(OUT, "domain_error_status.txt"), statuses(solution))
