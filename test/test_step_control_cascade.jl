# The cascade case of the step control: a prescribed cooling
# whose rate spans the decades from 1e-12 to 1e-3 s, as the injection cascade of a
# disruption does. Legs of hand-set fixed steps typical of such a cascade (1 000 steps at
# dt = 1e-12, 3e-12, 7e-12 and 1e-11 s, then 100 per decade to 1e-6 s and a top-up to
# 1e-3 s) are the reference. The temperature follows a power law in time,
# T = T0 (1 + t/τ)^(-α) with τ = 1e-12 s, so its relative rate α / (t + τ) falls as 1/t and
# a relative tolerance on the error asks for a step proportional to t. Runes2P has no
# ionization yet, so the stiffness of the cascade is this prescribed
# cooling, from 15 keV to 10.6 eV in 1 ms. Measured values in the comments below.
const CASCADE_T0 = 15e3          # eV
const CASCADE_TAU = 1e-12        # s
const CASCADE_ALPHA = 0.35       # T(1e-3 s) = T0 (1e9)^-0.35 = 10.6 eV
const CASCADE_T_END = 1e-3
const CASCADE_DENSITY = 1e20

cascade_temperature(t) = CASCADE_T0 * (1 + t / CASCADE_TAU)^(-CASCADE_ALPHA)

# hand-set legs of an injection cascade as (duration, steps), the top-up to 1e-3 s appended
const CASCADE_LEGS = [(1e-9, 1000), (3e-9, 1000), (7e-9, 1000), (1e-8, 1000), (1e-8, 100),
                      (1e-7, 100), (1e-6, 100), (1e-5, 100), (1e-4, 100)]

# the time points of the legs with `factor` times their steps, and the end of each leg
function cascade_leg_times(factor=1)
    legs = vcat(CASCADE_LEGS, [(CASCADE_T_END - sum(first.(CASCADE_LEGS)), 1000)])
    times, ends = [0.0], Float64[]
    for (duration, count) in legs
        start, n = last(times), count * factor
        for k in 1:n-1
            push!(times, start + k * duration / n)
        end
        push!(times, start + duration)
        push!(ends, start + duration)
    end
    return times, ends
end

function cascade_case(; np=40, nxi=6)
    mom = MomentumGrid(biuniform_p_faces(np, np ÷ 2, 0.0, 0.3, 2.0), uniform_xi_faces(nxi))
    grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
    n = CASCADE_DENSITY
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
    return (; grid, mom, system, u0=vec(f0), shape=size(f0))
end

# the steps of a run: the indices of the accepted and the rejected steps, the half steps
function cascade_steps(statistics)
    steps = findall(==(:step), statistics.kind)
    accepted = [k for k in steps if statistics.accepted[k]]
    return (; accepted, rejected=length(steps) - length(accepted), half=count(==(:half_step), statistics.kind))
end

# the largest difference of two solutions at the save times, relative to the largest value
cascade_differences(a, b) = [maximum(abs, a.states[:, m] - b.states[:, m]) / maximum(abs, a.states[:, m])
                             for m in axes(a.states, 2)]

# the density at the save times relative to the initial one
cascade_densities(case, solution) = [density(case.grid, reshape(solution.states[:, m], case.shape))[1] / CASCADE_DENSITY
                                     for m in axes(solution.states, 2)]

# the density above the momentum `p_cut` at the save times, relative to that of `reference`
# minus one: the hot tail, which the error measure of the controller does not see where f lies
# below its absolute floor `relative × scale`
function cascade_tail_differences(case, solution, reference, p_cut)
    mask = [p > p_cut for p in case.mom.p_centers, _ in 1:case.shape[2], _ in 1:1]
    tail(u) = density(case.grid, reshape(u, case.shape) .* mask)[1]
    return [tail(solution.states[:, m]) / tail(reference.states[:, m]) - 1 for m in axes(solution.states, 2)]
end

# the growth ratios of consecutive accepted steps, without the steps that landed on a
# ceiling, after which the controller resumes its planned length
function cascade_growth_ratios(statistics, accepted, ends)
    h, t = statistics.time_step[accepted], statistics.time[accepted]
    on_ceiling = [any(e -> abs(s - e) <= 1e-9 * e, ends) for s in t]
    return [h[k+1] / h[k] for k in 1:length(h)-1 if !on_ceiling[k]]
end

@testset "cascade case of the step control" begin
    case = cascade_case()
    times, ends = cascade_leg_times()
    @test length(times) - 1 == 5500
    @test last(ends) ≈ CASCADE_T_END
    reference = integrate(case.system, case.u0, times; save_times=ends)
    @test all(reference.statistics.accepted)
    @test reference.times == ends
    @test all(d -> abs(d - 1) < 1e-5, cascade_densities(case, reference))
    # the time error of the legs themselves: three times their steps (measured 1.3e-3)
    finer_times, _ = cascade_leg_times(3)
    @test length(finer_times) - 1 == 16500
    finer = integrate(case.system, case.u0, finer_times; save_times=ends)
    leg_error = maximum(cascade_differences(finer, reference))
    @test leg_error < 3e-3
    # the tail of the legs is converged too: the density above p = 1 within 2.3e-3 of the finer legs
    @test maximum(abs, cascade_tail_differences(case, finer, reference, 1.0)) < 1e-2
    # the ramp of the legs: steps until dt reaches 1e-6 s (4 401)
    ramp_of_legs = findfirst(>=(1e-6), reference.statistics.time_step)
    @test ramp_of_legs > 4000

    @testset "predictor controller" begin
        # measured: 226 accepted steps, no rejection, 118 steps for the ramp, 7.2e-3 against the legs
        stepper = StepController(initial_step=1e-12, relative_tolerance=1e-3)
        solution = integrate(case.system, case.u0, [0.0, last(ends)]; save_times=ends, stepper)
        stats = solution.statistics
        steps = cascade_steps(stats)
        @test solution.times == ends
        @test length(steps.accepted) <= 300
        @test steps.rejected <= 5
        @test steps.half == 2                               # the doubling of the first step
        h = stats.time_step[steps.accepted]
        ramp = findfirst(>=(1e-6), h)
        @test ramp <= 160
        @test ramp < ramp_of_legs / 20
        per_decade = [count(k -> 10.0^d <= h[k] < 10.0^(d + 1), eachindex(h)) for d in -12:-7]
        @test all(per_decade .<= 60)
        @test maximum(cascade_growth_ratios(stats, steps.accepted, ends)) <= 2 * (1 + 1e-12)
        @test all(stats.error[steps.accepted] .<= 1)
        differences = cascade_differences(reference, solution)
        @test maximum(differences) < 2e-2
        @test maximum(differences) > leg_error                # the controller's error, not the legs'
        @test all(d -> abs(d - 1) < 1e-5, cascade_densities(case, solution))
        # the hot tail: the density above p = 0.3 within 7.5e-3 of the legs, the one
        # above p = 1 (6.9e-4 n at 1 ms) within 6.9e-5 during the cascade and 4.3e-2 at 1 ms. The
        # floor of the error measure, 1e17 m⁻³, lies 50 times above f there, so those cells are
        # not controlled at the late, long steps.
        @test maximum(abs, cascade_tail_differences(case, solution, reference, 0.3)) < 2e-2
        tail = cascade_tail_differences(case, solution, reference, 1.0)
        @test maximum(abs, tail[1:8]) < 1e-3                 # the save times up to 1.1e-5 s
        @test maximum(abs, tail) < 1e-1

        # the default tolerance: half the steps, twice the difference (measured 107 steps, 54 for
        # the ramp, 1.6e-2 against the legs)
        looser = integrate(case.system, case.u0, [0.0, last(ends)]; save_times=ends,
                           stepper=StepController(initial_step=1e-12))
        steps_looser = cascade_steps(looser.statistics)
        @test length(steps_looser.accepted) <= 150
        @test length(steps_looser.accepted) < length(steps.accepted)
        @test maximum(cascade_differences(reference, looser)) < 4e-2
        @test maximum(abs, cascade_tail_differences(case, looser, reference, 1.0)) < 3e-1   # measured 1.2e-1
    end

    @testset "step doubling" begin
        # DREAM's rule has no growth cap: the first step of 1e-12 s changes f by 1e-7 of the
        # tolerance (collisions at 15 keV are slow), and the next step is 1e-6 s. Measured: 85
        # accepted and 13 rejected steps with 196 half steps, 2.0e-2 against the legs.
        stepper = StepDoubling(initial_step=1e-12, relative_tolerance=1e-3)
        solution = integrate(case.system, case.u0, [0.0, last(ends)]; save_times=ends, stepper)
        stats = solution.statistics
        steps = cascade_steps(stats)
        @test solution.times == ends
        @test length(steps.accepted) <= 120
        @test steps.rejected <= 30
        @test steps.half == 2 * (length(steps.accepted) + steps.rejected)
        @test maximum(cascade_growth_ratios(stats, steps.accepted, ends)) > 100
        @test all(stats.error[steps.accepted] .<= 1)
        @test maximum(cascade_differences(reference, solution)) < 4e-2
        @test all(d -> abs(d - 1) < 1e-5, cascade_densities(case, solution))
        @test maximum(abs, cascade_tail_differences(case, solution, reference, 1.0)) < 4e-1   # measured 1.9e-1

        # at the default tolerance the rule jumps to steps of 1e-5 s and misses the cascade:
        # both solves of a step relax to the same Maxwellian, so the doubling estimate stays
        # under the tolerance while the states at the save times, interpolated across the
        # steps, differ from the legs by up to 1.1 of the largest value (measured 16 steps).
        default = integrate(case.system, case.u0, [0.0, last(ends)]; save_times=ends,
                            stepper=StepDoubling(initial_step=1e-12))
        steps_default = cascade_steps(default.statistics)
        @test length(steps_default.accepted) < 30
        @test all(default.statistics.error[steps_default.accepted] .<= 1)
        @test maximum(cascade_differences(reference, default)) > 0.3
        # at the final time, computed by the rule itself: 2.4e-2 of the largest value in the bulk and
        # 6.2e-1 in the density above p = 1
        @test cascade_differences(reference, default)[end] > 1e-2
        @test abs(cascade_tail_differences(case, default, reference, 1.0)[end]) > 0.3
    end
end
