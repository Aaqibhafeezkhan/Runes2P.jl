# FluxCorrectionTerm: a modified/quasi-Newton strategy, the residual of one flux scheme
# solved with another, cheaper scheme's Jacobian (DREAM's
# AD_INTERP_JACOBIAN_UPWIND, TCDF in the residual with the upwind Jacobian). Independent
# of DREAM: this is solver machinery, not new physics.
using ForwardDiff: ForwardDiff
using Random: Xoshiro

_flux_correction_plasma(temperature=100.0; density=1e20, field=5.0, nr=1) =
    PlasmaState(cold_electron_temperature=fill(temperature, nr), cold_electron_density=fill(density, nr),
               ions=[IonSpecies("D", 1, 1, fill(density, nr))], electric_field=fill(field, nr),
               magnetic_field=fill(5.0, nr))

@testset "FluxCorrectionTerm" begin
    mom = MomentumGrid(uniform_p_faces(20, 0.0, 1.0), uniform_xi_faces(6))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
    settings = CollisionSettings(mode=:full)
    plasma_at(t) = _flux_correction_plasma(100.0)
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1.0)])

    op_upwind = AdvectionDiffusion(g; scheme=:upwind, p_upper_boundary=:f_zero)
    op_tcdf = AdvectionDiffusion(g; scheme=:tcdf, p_upper_boundary=:f_zero)
    combo_system = EquationSystem(layout, [Equation(:f, TimeDerivative(),
                                                    PrescribedFokkerPlanckTerm(:f, op_upwind, plasma_at; settings),
                                                    FluxCorrectionTerm(:f, op_tcdf, op_upwind, plasma_at; settings))])

    op_tcdf_ref = AdvectionDiffusion(g; scheme=:tcdf, p_upper_boundary=:f_zero)
    reference_system = EquationSystem(layout, [Equation(:f, TimeDerivative(),
                                                        PrescribedFokkerPlanckTerm(:f, op_tcdf_ref, plasma_at; settings))])

    u = rand(Xoshiro(1), length(layout))
    t = 1e-3

    @testset "residual equals the full (TCDF) operator's alone" begin
        du_combo = zeros(length(layout))
        Runes2P.rhs!(du_combo, u, combo_system, t)
        du_reference = zeros(length(layout))
        Runes2P.rhs!(du_reference, u, reference_system, t)
        @test du_combo ≈ du_reference rtol = 1e-8
    end

    @testset "Jacobian action equals the base (upwind) operator's matrix alone" begin
        J = jacobian_prototype(combo_system)
        jacobian!(J, combo_system, u, t)
        op_upwind_fresh = AdvectionDiffusion(g; scheme=:upwind, p_upper_boundary=:f_zero)
        set_fokker_planck_coefficients!(op_upwind_fresh, plasma_at(t), settings)
        M_upwind = operator_matrix(op_upwind_fresh)
        @test Matrix(J) ≈ Matrix(M_upwind) rtol = 1e-12

        v = rand(Xoshiro(2), length(layout))
        Jop = jacobian_operator(J, combo_system, u, t)
        @test Jop * v ≈ M_upwind * v rtol = 1e-12
        # not the true (TCDF) Jacobian: TCDF is nonlinear in f, so its own Jacobian (dense
        # ForwardDiff of apply!, as check_stencil_pattern uses for nonlinear schemes) is
        # generally different from the upwind one used here
        residual_tcdf = x -> (y = zeros(eltype(x), size(g)); apply!(y, op_tcdf_ref, reshape(x, size(g))); vec(y))
        M_tcdf = ForwardDiff.jacobian(residual_tcdf, u)
        @test !isapprox(Matrix(J), M_tcdf; rtol=1e-3)
    end

    @testset "backward-Euler step matches a direct TCDF solve" begin
        # a linear-in-f scheme pair (upwind base, TCDF residual with GMRES) converges to
        # the same nonlinear solution as solving the TCDF equation directly, just with a
        # different (approximate) Jacobian driving the Newton iteration, at a step small
        # enough for the approximation to stay in the convergence basin (dt = 1e-4 here
        # needs 25-30 Newton iterations and stalls without truly converging, a property of
        # the modified-Newton method itself, not of this implementation: 1e-6, a shorter
        # step, needs only 9). A smooth, physical initial state, not
        # pure random noise.
        theta = 0.05
        u0 = [exp(-mom.p_centers[i]^2 / (2theta)) for i in 1:momentum_cell_count(g), _ in 1:pitch_cell_count(g),
                                                       _ in 1:1] |> vec
        times = [0.0, 1e-6]
        combo_solution = integrate(combo_system, u0, times; solver=NewtonSolver(linear_solver=GMRES()))
        reference_solution = integrate(reference_system, u0, times)
        @test combo_solution.states[:, end] ≈ reference_solution.states[:, end] rtol = 1e-8
        @test all(combo_solution.statistics.accepted)
    end

    @testset "validate requires base_operator's term to run first (documented, not enforced structurally)" begin
        term = FluxCorrectionTerm(:f, op_tcdf, op_upwind, plasma_at; settings)
        @test dependencies(term) == [:f]
        @test coupling(term) === :matrix_free
        wrong_layout = UnknownLayout(g.radial_grid, [ScalarUnknown(:f)])
        @test_throws ArgumentError Runes2P.validate(term, wrong_layout, 1)
    end
end
