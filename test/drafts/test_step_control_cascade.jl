# Draft, outside the suite.
# The cascade case of the step control: a prescribed cooling whose rate
# spans the decades from 1e-12 to 1e-3 s, as the injection cascade of a disruption run
# does (legs of 100 to 1 000 steps per decade from
# dt = 1e-12 s to 1e-6 s, set by hand). The temperature follows a power law in time,
# T = T0 (1 + t/τ)^(-α) with τ = 1e-12 s, so its relative rate α / (t + τ) falls as 1/t and
# a relative tolerance on the error asks for a step proportional to t. The controller
# must produce that ramp without hand-set legs and match the reference of the legs.
using HDF5

const CASCADE_T0 = 15e3          # eV
const CASCADE_TAU = 1e-12        # s
const CASCADE_ALPHA = 0.35       # T(1e-3 s) = T0 (1e9)^-0.35 = 10.9 eV
const CASCADE_T_END = 1e-3

cascade_temperature(t) = CASCADE_T0 * (1 + t / CASCADE_TAU)^(-CASCADE_ALPHA)

# hand-set legs as in a disruption run, scaled: `per_decade` steps per decade of dt from
# 1e-12 s to 1e-6 s, then 1e-6 s to the end
function cascade_legs(per_decade)
    times = [0.0]
    dt = 1e-12
    while last(times) < 1e-3 * (1 - 1e-12)
        t_leg_end = min(dt * per_decade + last(times), CASCADE_T_END)
        dt_leg = min(dt, 1e-6)
        while last(times) < t_leg_end * (1 - 1e-12)
            push!(times, min(last(times) + dt_leg, t_leg_end))
        end
        dt = min(10 * dt, 1e-6)
    end
    return times
end

function cascade_case(; np=40, nxi=6)
    mom = MomentumGrid(biuniform_p_faces(np, np ÷ 2, 0.0, 0.3, 2.0), uniform_xi_faces(nxi))
    grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
    n = 1e20
    plasma_at(t) = PlasmaState(cold_electron_temperature=[cascade_temperature(t)], cold_electron_density=[n],
                               ions=[IonSpecies("D", 1, 1, [n])], electric_field=[0.5], magnetic_field=[5.0])
    operator = AdvectionDiffusion(grid; scheme=:exponential_fitting, p_upper_boundary=:f_zero)
    layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid; scale=n)])
    system = EquationSystem(layout, [Equation(:f, TimeDerivative(),
                                              PrescribedFokkerPlanckTerm(:f, operator, plasma_at;
                                                                         settings=CollisionSettings(mode=:full)))])
    theta = CASCADE_T0 / Constants.ELECTRON_REST_ENERGY
    f0 = [n * exp(-(sqrt(1 + p^2) - 1) / theta) for p in mom.p_centers, _ in 1:nxi, _ in 1:1]
    f0 .*= n / density(grid, f0)[1]
    return (; grid, system, layout, u0=vec(f0), plasma_at)
end

@testset "cascade case of the step control" begin
    case = cascade_case()
    save_times = [0.0, 1e-9, 1e-7, 1e-5, 1e-4, CASCADE_T_END]
    # the reference: the legs at 300 steps per decade
    legs = cascade_legs(300)
    @test length(legs) > 2000
    reference = integrate(case.system, case.u0, legs; save_times)
    @test all(reference.statistics.accepted)

    @testset "predictor controller" begin
        stepper = StepController(initial_step=1e-12, relative_tolerance=1e-3)
        solution = integrate(case.system, case.u0, [0.0, CASCADE_T_END]; save_times, stepper)
        stats = solution.statistics
        accepted = findall(stats.accepted)
        @test solution.times == save_times
        # the ramp: the step grows with the time, at most by the cap of 2 per step, and
        # the count per decade of dt stays far below the legs
        t_acc, h_acc = stats.time[accepted], stats.time_step[accepted]
        per_decade = [count(k -> 10.0^d <= h_acc[k] < 10.0^(d + 1), eachindex(h_acc)) for d in -12:-7]
        @test all(per_decade .<= 40)
        @test length(accepted) < length(legs) / 5
        @test all(h_acc[2:end] ./ h_acc[1:end-1] .<= 2 * (1 + 1e-12))
        @test all(stats.error[accepted] .<= 1)
        # the states at the save times against the legs, relative to the scale of f
        for (m, s) in enumerate(save_times)
            a, b = reference.states[:, m], solution.states[:, m]
            @test maximum(abs, a - b) / maximum(abs, a) < 1e-2
        end
        @test count(!, stats.accepted) < length(accepted) / 2      # rejections stay a minority
    end

    @testset "step doubling" begin
        stepper = StepDoubling(initial_step=1e-12, relative_tolerance=1e-3)
        solution = integrate(case.system, case.u0, [0.0, CASCADE_T_END]; save_times, stepper)
        stats = solution.statistics
        accepted = findall(stats.accepted)
        @test solution.times == save_times
        @test length(accepted) < length(legs)
        for (m, s) in enumerate(save_times)
            a, b = reference.states[:, m], solution.states[:, m]
            @test maximum(abs, a - b) / maximum(abs, a) < 1e-2
        end
    end
end
