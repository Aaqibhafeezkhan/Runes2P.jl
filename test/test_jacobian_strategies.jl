# Jacobian strategies for the flux limiters: the `jacobian` option of the
# Fokker-Planck terms (exact AD, frozen limiter, a linear scheme's Jacobian; DREAM's
# AD_INTERP_JACOBIAN_FULL/LINEAR/UPWIND), the exact Newton operator of GMRES over an
# approximate stored block, the Newton rules that depend on them (no residual-floor waiver
# for a quasi-Newton iteration, one-iteration acceptance of linear systems). Solver
# machinery, independent of DREAM.
using ForwardDiff: ForwardDiff
using SparseArrays: nnz
using Random: Xoshiro

_strategies_plasma(temperature=100.0; density=1e20, field=5.0, nr=1) =
    PlasmaState(cold_electron_temperature=fill(temperature, nr), cold_electron_density=fill(density, nr),
               ions=[IonSpecies("D", 1, 1, fill(density, nr))], electric_field=fill(field, nr),
               magnetic_field=fill(5.0, nr))

@testset "Jacobian strategies" begin
    mom = MomentumGrid(uniform_p_faces(20, 0.0, 1.0), uniform_xi_faces(6))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
    settings = CollisionSettings(mode=:full)
    plasma = _strategies_plasma(100.0)
    plasma_at(t) = plasma
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1.0)])
    t = 1e-3
    # a smooth positive state: the limiter's slope ratios are well defined
    theta = 0.05
    u0 = [exp(-mom.p_centers[i]^2 / (2theta)) * (1 + 0.3 * mom.xi_centers[j])
          for i in 1:momentum_cell_count(g), j in 1:pitch_cell_count(g), _ in 1:1] |> vec
    function system(scheme, jacobian)
        op = AdvectionDiffusion(g; scheme, p_upper_boundary=:f_zero)
        return EquationSystem(layout, [Equation(:f, TimeDerivative(),
                                                PrescribedFokkerPlanckTerm(:f, op, plasma_at; settings, jacobian))])
    end
    function dense_jacobian(scheme; frozen=false)
        op = AdvectionDiffusion(g; scheme, p_upper_boundary=:f_zero)
        set_fokker_planck_coefficients!(op, plasma, settings)
        state = reshape(u0, size(g))
        residual = x -> (y = zeros(eltype(x), size(g));
                         apply!(y, op, reshape(x, size(g)); limiter_state=frozen ? state : nothing); vec(y))
        return ForwardDiff.jacobian(residual, u0)
    end
    stored(sys) = Matrix(jacobian!(jacobian_prototype(sys), sys, u0, t))

    exact, frozen, upwind, fitted = (system(:tcdf, j) for j in (:exact, :frozen_limiter, :upwind, :exponential_fitting))
    J_tcdf = dense_jacobian(:tcdf)
    scale = maximum(abs, J_tcdf)

    @testset "modes and their stored blocks" begin
        @test jacobian_kind(exact.sparse_terms[1].term) === :exact
        @test all(jacobian_kind(s.sparse_terms[1].term) === :approximate for s in (frozen, upwind, fitted))
        @test exact.stored_jacobian_exact && !frozen.stored_jacobian_exact && !upwind.stored_jacobian_exact
        @test isempty(exact.correction_terms) && length(frozen.correction_terms) == 1
        # the residual is the operator's own in every mode
        residuals = map((exact, frozen, upwind, fitted)) do s
            r = zeros(length(layout))
            Runes2P.rhs!(r, u0, s, t)
            r
        end
        @test all(r == residuals[1] for r in residuals)
        # exact: the derivative with ψ′; frozen: the weights at u0 as constants (differs from
        # the exact one); a scheme name: that linear scheme's matrix, with its own pattern
        @test stored(exact) ≈ J_tcdf rtol = 1e-12
        J_frozen = dense_jacobian(:tcdf; frozen=true)
        @test stored(frozen) ≈ J_frozen rtol = 1e-12
        @test maximum(abs, J_frozen - J_tcdf) > 1e-3 * scale
        @test stored(upwind) ≈ Matrix(operator_matrix(with_scheme(exact.sparse_terms[1].term.inner.operator, :upwind))) rtol = 1e-12
        @test stored(fitted) ≈ dense_jacobian(:exponential_fitting) rtol = 1e-12
        @test nnz(jacobian_sparsity(upwind)) == nnz(jacobian_sparsity(system(:upwind, :exact))) <
              nnz(jacobian_sparsity(exact)) == nnz(jacobian_sparsity(frozen))
        @test jacobian_colors(upwind) < jacobian_colors(exact) == jacobian_colors(frozen)
        @test_throws ArgumentError FokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme=:tcdf); jacobian=:muscl)
        @test_throws ArgumentError FokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme=:tcdf); jacobian=:frozen)
        # a linear scheme: the frozen limiter is exact (no buffers), another scheme's
        # Jacobian is not (a central residual with the upwind Jacobian, DREAM #126)
        term = FokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme=:exponential_fitting); jacobian=:frozen_limiter)
        @test jacobian_kind(term) === :exact && isempty(term.values)
        @test jacobian_kind(FokkerPlanckTerm(:f, AdvectionDiffusion(g; scheme=:central); jacobian=:central)) === :exact
        fitted_upwind = system(:exponential_fitting, :upwind)
        @test jacobian_kind(fitted_upwind.sparse_terms[1].term) === :approximate
        # the residual is linear, the stored matrix is not its derivative
        @test fitted_upwind.linear && !NewtonWorkspace(fitted_upwind, NewtonSolver()).linear
        @test NewtonWorkspace(fitted_upwind, NewtonSolver(linear_solver=GMRES())).linear
        @test stored(fitted_upwind) ≈ stored(upwind) rtol = 1e-12
        J_fitted = dense_jacobian(:exponential_fitting)
        Jv = jacobian!(jacobian_prototype(fitted_upwind), fitted_upwind, u0, t)
        v = rand(Xoshiro(5), length(layout))
        @test jacobian_operator(Jv, fitted_upwind, u0, t) * v ≈ J_fitted * v rtol = 1e-12
        @test dependencies(frozen.sparse_terms[1].term) == [:f]
    end

    @testset "with_scheme shares the coefficients" begin
        op = AdvectionDiffusion(g; scheme=:van_albada, p_upper_boundary=:f_zero)
        twin = with_scheme(op, :upwind)
        @test twin.advection_p === op.advection_p && twin.diffusion_p_p === op.diffusion_p_p
        @test twin.p_upper_boundary === :f_zero && twin.r_scheme === op.r_scheme
        @test (twin.p_scheme, twin.xi_scheme) == (:upwind, :upwind)
        set_fokker_planck_coefficients!(op, plasma, settings)
        @test maximum(abs, twin.advection_p) > 0
        @test_throws ArgumentError with_scheme(op, :nonsense)
    end

    @testset "the exact operator over an approximate stored block" begin
        v = rand(Xoshiro(3), length(layout))
        for s in (frozen, upwind, fitted)
            J = jacobian!(jacobian_prototype(s), s, u0, t)
            @test jacobian_operator(J, s, u0, t) * v ≈ J_tcdf * v rtol = 1e-12
            @test !isapprox(J * v, J_tcdf * v; rtol=1e-6)
            correction = zeros(length(layout))
            jacobian_correction_product!(correction, s.sparse_terms[1].term, u0, v, layout, 1, t)
            @test J * v + correction ≈ J_tcdf * v rtol = 1e-12
        end
        # the correction is allocation free after the first call
        term = frozen.sparse_terms[1].term
        correction = zeros(length(layout))
        jacobian_correction_product!(correction, term, u0, v, layout, 1, t)
        @test @allocated(jacobian_correction_product!(correction, term, u0, v, layout, 1, t)) == 0
    end

    @testset "quasi-Newton on the stored block, exact Newton with GMRES" begin
        # dt small enough for the quasi-Newton iteration's convergence basin (a step of
        # 1e-4 stalls); the same state whatever the strategy
        times = [0.0, 1e-6]
        reference = integrate(exact, u0, times)
        @test reference.statistics.convergence_order[1] > 1.2 || reference.statistics.newton_iterations[1] <= 3
        for s in (frozen, upwind)
            for solver in (NewtonSolver(), NewtonSolver(linear_solver=GMRES(operator=:stored)))
                quasi = integrate(s, u0, times; solver)
                @test quasi.states[:, end] ≈ reference.states[:, end] rtol = 1e-8
                @test all(quasi.statistics.accepted)
                # linear convergence, and never accepted through the residual-floor waiver
                @test quasi.statistics.convergence_order[1] ≈ 1 atol = 0.15
                @test quasi.statistics.newton_iterations[1] > reference.statistics.newton_iterations[1]
                @test !any(quasi.statistics.accepted_at_residual_floor)
                @test !NewtonWorkspace(s, solver).exact_jacobian
            end
            # the exact operator: Newton as with the exact stored Jacobian, the stored
            # block only preconditions
            ws = NewtonWorkspace(s, NewtonSolver(linear_solver=GMRES()))
            @test ws.exact_jacobian
            newton = integrate(s, u0, times; solver=NewtonSolver(linear_solver=GMRES()))
            @test newton.states[:, end] ≈ reference.states[:, end] rtol = 1e-8
            @test newton.statistics.newton_iterations[1] <= reference.statistics.newton_iterations[1] + 1
            @test newton.statistics.krylov_iterations[1] > newton.statistics.newton_iterations[1]
        end
        @test_throws ArgumentError NewtonWorkspace(frozen, NewtonSolver(linear_solver=GMRES(operator=:sometimes)))
    end

    @testset "a linear system is accepted on the update test alone" begin
        linear = system(:exponential_fitting, :exact)
        @test linear.linear && !exact.linear
        # a stiff step (round-off floor of the residual above the tolerance), where the
        # residual test would need a third iteration to establish the stall
        times = [0.0, 1e-3, 2e-3, 3e-3]
        few = integrate(linear, u0, times)
        many = integrate(linear, u0, times; solver=NewtonSolver(linear_acceptance=false))
        @test all(few.statistics.accepted_as_linear) && all(==(2), few.statistics.newton_iterations)
        @test !any(many.statistics.accepted_as_linear)
        @test all(many.statistics.newton_iterations .>= few.statistics.newton_iterations)
        @test few.states[:, end] ≈ many.states[:, end] rtol = 1e-12
        # an accurate Krylov solve qualifies, a loose one does not
        accurate = integrate(linear, u0, times; solver=NewtonSolver(linear_solver=GMRES()))
        @test all(accurate.statistics.accepted_as_linear)
        @test accurate.states[:, end] ≈ many.states[:, end] rtol = 1e-9
        loose = integrate(linear, u0, times; solver=NewtonSolver(linear_solver=GMRES(relative_tolerance=1e-6)))
        @test !any(loose.statistics.accepted_as_linear)
        # a nonlinear scheme, or an approximate Jacobian of one, is never a linear step
        @test !any(integrate(exact, u0, [0.0, 1e-6]).statistics.accepted_as_linear)
        @test !NewtonWorkspace(upwind, NewtonSolver()).linear
        # the knock-on term is linear with a prescribed target density, bilinear with an unknown
        knockon = KnockOnOperator(g, g, 0.3)
        @test is_linear(KnockOnTerm(:f, knockon; target_density=[1e20]))
        @test !is_linear(KnockOnTerm(:f, knockon; target_density=:n, treatment=:implicit))
        # DREAM's form, the default: the target density times a source fixed for the step
        @test is_linear(KnockOnTerm(:f, knockon; target_density=:n))
        @test is_linear(LinearTerm(:f)) && is_linear(DensityMomentTerm(:f))
    end
end
