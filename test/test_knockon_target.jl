# The knock-on term with a target density that is an unknown and changes during the run:
# the lagged primary (:explicit_primary) against the
# implicit term, with an error of first order in the step; the kernel applied once per
# start state; a resume inside an interval of the step controller; the SciML path without
# the stored Jacobian of a lagged term.
using SciMLBase: SciMLBase
using HDF5

# the case of test_checkpoint.jl with the target density an unknown that decays in 1e-4 s
function _target_case(; treatment)
    nr = 2
    mom = MomentumGrid(biuniform_p_faces(16, 8, 0.0, 0.2, 1.2), uniform_xi_faces(4))
    grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(nr, 1.0)))
    plasma = PlasmaState(cold_electron_temperature=[1000.0, 2000.0], cold_electron_density=fill(1e20, nr),
                         ions=[IonSpecies("D", 1, 1, fill(1e20, nr))], electric_field=[1.0, 0.8],
                         magnetic_field=fill(5.0, nr))
    op = AdvectionDiffusion(grid; scheme=:exponential_fitting, p_upper_boundary=:f_zero)
    knockon = KnockOnOperator(grid, grid, 0.45)
    knock = KnockOnTerm(:f, knockon; target_density=:target, treatment)
    layout = UnknownLayout(grid.radial_grid, [KineticUnknown(:f, grid; scale=1e20), RadialUnknown(:target; scale=1e20)])
    equations = [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op; plasma, settings=CollisionSettings(mode=:full)), knock),
                 Equation(:target, TimeDerivative(), LinearTerm(:target; coefficient=-1e4))]
    system = EquationSystem(layout, equations)
    theta = 1000.0 / Constants.ELECTRON_REST_ENERGY
    u0 = zeros(length(layout))
    for ir in 1:nr
        kinetic_view(u0, layout, 1, ir) .= [1e20 * exp(-(sqrt(1 + p^2) - 1) / theta) + 1e16 * exp(-(p - 0.8)^2 / 0.01)
                                            for p in mom.p_centers, _ in 1:4]
        radial_view(u0, layout, 2, ir) .= 1e20
    end
    return system, layout, u0, knock
end

@testset "knock-on with an unknown target density" begin
    @testset "lagged primary against implicit: first order in the step" begin
        t_end = 2e-4
        final(treatment, steps) = begin
            system, layout, u0, _ = _target_case(; treatment)
            # the implicit term is matrix free: Newton-Krylov with tight tolerances for both
            solver = NewtonSolver(linear_solver=GMRES(relative_tolerance=1e-12))
            integrate(system, u0, collect(range(0.0, t_end; length=steps + 1)); stepper=FixedSteps(), solver).states[:, end]
        end
        difference(steps) = maximum(abs, final(:explicit_primary, steps) - final(:implicit, steps))
        coarse, fine = difference(10), difference(20)
        @test coarse > 0
        @test 1.6 < coarse / fine < 2.4
    end

    @testset "the kernel once per start state" begin
        system, layout, u0, knock = _target_case(; treatment=:explicit_primary)
        F = zeros(length(u0))
        u = copy(u0)
        before = knock.preparations[]
        residual!(F, u, u0, system, 1e-6, 1e-6)
        residual!(F, u .* 1.01, u0, system, 1e-6, 1e-6)
        residual!(F, u .* 0.99, u0, system, 1e-6, 1e-6)
        @test knock.preparations[] == before + 1
        # a new start state prepares the source again, and its residual is that of a fresh term
        u1 = u0 .* 1.02
        residual!(F, u, u1, system, 1e-6, 1e-6)
        @test knock.preparations[] == before + 2
        # the same residual as with a source prepared anew
        knock.prepared[] = false
        G = zeros(length(u0))
        residual!(G, u, u1, system, 1e-6, 1e-6)
        @test G == F
        # an integration prepares once per step attempt
        before = knock.preparations[]
        solution = integrate(system, u0, collect(range(0.0, 1e-4; length=6)); stepper=FixedSteps())
        attempts = length(solution.statistics.accepted)
        @test attempts <= knock.preparations[] - before <= attempts + 1
    end

    @testset "a resume inside an interval of the step controller" begin
        system, layout, u0, _ = _target_case(; treatment=:explicit_primary)
        times = [0.0, 1e-4, 2e-4]
        stepper = StepController(; relative_tolerance=1e-3, initial_step=1e-6)
        whole = integrate(system, u0, times; stepper)
        @test count(whole.statistics.accepted) > 3
        checkpoint = CheckpointSettings("target"; directory=mktempdir(), interval=0)
        first_segment = integrate(system, u0, times; stepper, checkpoint, stop=StopCondition(wall_budget=1e-9, slack=0))
        @test run_status(first_segment) === :stopped
        second_segment = integrate(system, u0, times; stepper, checkpoint, resume=true)
        @test run_status(second_segment) === :finished
        @test second_segment.states[:, end] == whole.states[:, end]
    end

    @testset "the SciML path without the stored Jacobian of a lagged term" begin
        lagged, _, _, knock = _target_case(; treatment=:explicit_primary)
        @test lags_in_ode_form(knock)
        f = ode_function(lagged)
        @test f.jac === nothing && f.jvp === nothing
        implicit, _, _, implicit_knock = _target_case(; treatment=:implicit)
        @test !lags_in_ode_form(implicit_knock)
        # the right-hand side of the ODE form is that of the system at the state itself
        _, _, u0, _ = _target_case(; treatment=:explicit_primary)
        du = zeros(length(u0))
        f(du, u0, nothing, 0.0)
        @test du == rhs!(zeros(length(u0)), u0, lagged, 0.0)
    end
end
