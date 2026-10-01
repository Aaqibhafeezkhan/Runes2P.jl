# Four-point advective schemes and flux limiters, independent of DREAM:
# orders by manufactured solutions at small and large Péclet number (the limiters solved by
# Newton), the Jacobian by colored AD against dense ForwardDiff, the frozen limiter, the
# refusals, the p = 0 face, and positivity in a relaxation run of the plasma of
# thermalquench/iter.py on the grid of runaway/basic.py (the setup of
# test_exponential_fitting.jl).

using ForwardDiff: ForwardDiff
using Random: Xoshiro

const FOUR_POINT = (:quick, :upwind_second_order, :tcdf, :muscl, :smart, :ospre, :van_albada)

# the 1D manufactured step u - L(u) = f_e - L_e f_e of test_manufactured_solutions.jl, with
# the advection scaled by `strength`, solved by Newton
function limiter_manufactured_error(cells, scheme, strength)
    ms = ManufacturedSolution
    faces = biuniform_p_faces(cells, cells ÷ 5, 0.0, 0.7, 30.0)
    advection(p) = -strength * (1 + 0.1p)
    flux(p) = advection(p) * ms.g(p) - ms.diffusion_p_p(p, 0.0) * ms.dg(p)
    mom = MomentumGrid(faces, [-1.0, 1.0])
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
    op = AdvectionDiffusion(g; p_scheme=scheme, xi_scheme=:central, p_upper_boundary=:f_zero)
    for (k, p) in enumerate(mom.p_faces)
        op.advection_p[k, 1, 1] = advection(p)
        op.diffusion_p_p[k, 1, 1] = ms.diffusion_p_p(p, 0.0)
    end
    Vf = phase_space_jacobian_p_faces(g, 1)[:, 1]
    V = phase_space_jacobian(g, 1)[:, 1]
    exact = [-(Vf[i+1] * flux(faces[i+1]) - Vf[i] * flux(faces[i])) / (V[i] * mom.p_widths[i]) for i in 1:cells]
    f_exact = ms.g.(mom.p_centers)
    layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1.0)])
    system = EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op))])
    u = copy(f_exact)
    result = newton_solve!(u, f_exact .- exact, NewtonWorkspace(system, NewtonSolver(max_iterations=60)), 1.0, 1.0)
    weights = V .* mom.p_widths
    return result.status, sum(abs.(u .- f_exact) .* weights) / sum(abs.(f_exact) .* weights)
end

@testset "flux limiters" begin
    @testset "orders by manufactured solutions" begin
        # observed L1 orders from 800 to 1600 cells: 1.97-2.13 at strength 1; at strength 100
        # (Péclet number above 2 on the coarse grids) TCDF 2.89, SMART 2.97, QUICK 2.67,
        # the others 1.90-2.08
        for strength in (1.0, 100.0), scheme in FOUR_POINT
            (status1, e1), (status2, e2) = (limiter_manufactured_error(n, scheme, strength) for n in (800, 1600))
            @test status1 === status2 === :converged
            @test log2(e1 / e2) > 1.8
            strength == 100.0 && scheme in (:tcdf, :smart) && @test log2(e1 / e2) > 2.5
        end
    end

    mom = MomentumGrid(biuniform_p_faces(30, 10, 0.0, 0.7, 5.0), uniform_xi_faces(6))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 1.0)))
    plasma = PlasmaState(cold_electron_temperature=[100.0, 200.0], cold_electron_density=[1e20, 1e20],
                         ions=[IonSpecies("D", 1, 1, [1e20, 1e20])], electric_field=[1.0, 2.0],
                         magnetic_field=[5.0, 5.0])

    @testset "Jacobian and stencil" begin
        for scheme in FOUR_POINT
            op = fokker_planck_operator(g, plasma, CollisionSettings(mode=:full); scheme)
            @test check_stencil_pattern(op)
            layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1e18)])
            system = EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op))])
            u = 1e18 .* (0.5 .+ rand(Xoshiro(3), length(layout)))
            J = jacobian!(jacobian_prototype(system), system, u, 0.0)
            dense = ForwardDiff.jacobian(x -> rhs!(similar(x), x, system, 0.0), u)
            @test Matrix(J) == dense
        end
        # finite derivatives where f underflows (the far tail of a Maxwellian)
        for scheme in (:tcdf, :muscl, :smart, :ospre, :van_albada)
            face(y) = Runes2P._four_point_face_value(scheme, 0.5, 1.0, 0.0, -1.2, y[1], y[2], y[3])
            @test all(isfinite, ForwardDiff.gradient(face, [1e-300, 2e-305, 1e-310]))
        end
    end

    @testset "frozen limiter" begin
        op = fokker_planck_operator(g, plasma, CollisionSettings(mode=:full); scheme=:tcdf)
        f = 1e18 .* (0.5 .+ rand(Xoshiro(4), size(g)...))
        state = 1e18 .* (0.5 .+ rand(Xoshiro(5), size(g)...))
        full, frozen_at_f = zeros(size(g)), zeros(size(g))
        apply!(full, op, f)
        apply!(frozen_at_f, op, f; limiter_state=f)
        @test frozen_at_f ≈ full rtol = 1e-12
        # frozen at another state, the operator is linear in f
        a, b, ab = zeros(size(g)), zeros(size(g)), zeros(size(g))
        apply!(a, op, f; limiter_state=state)
        apply!(b, op, 2 .* state; limiter_state=state)
        apply!(ab, op, f .+ 2 .* state; limiter_state=state)
        @test ab ≈ a .+ b rtol = 1e-12
    end

    @testset "refusals" begin
        op = fokker_planck_operator(g, plasma, CollisionSettings(mode=:full); scheme=:tcdf)
        @test_throws ArgumentError operator_matrix(op)
        # four-point schemes in r are tested in test_radial_transport.jl
        @test AdvectionDiffusion(g; scheme=:central, r_scheme=:tcdf).r_scheme === :tcdf
        @test_throws ArgumentError AdvectionDiffusion(g; scheme=:central, r_scheme=:superbee)
        @test_throws ArgumentError AdvectionDiffusion(g; scheme=:superbee)
        # a scheme per direction; the radial default of a four-point scheme is central
        mixed = AdvectionDiffusion(g; scheme=:tcdf, xi_scheme=:central)
        @test (mixed.p_scheme, mixed.xi_scheme, mixed.r_scheme) == (:tcdf, :central, :central)
        @test operator_matrix(fokker_planck_operator(g, plasma, CollisionSettings(mode=:full); scheme=:quick)) isa
              Runes2P.SparseMatrixCSC
    end

    @testset "p = 0 face" begin
        # With the ghost cell mirrored at p = 0, the limiters give f₁ on the p = 0 face and
        # QUICK its quadratic value there; the flux through p = 0 carries it (superthermal
        # mode, where p²Aᵖ does not vanish at p = 0; DREAM's AdvectionTerm)
        superthermal = CollisionSettings(mode=:superthermal)
        f = 1e18 .* (0.5 .+ rand(Xoshiro(6), size(g)...))
        flux(scheme) = momentum_flux(fokker_planck_operator(g, plasma, superthermal; scheme), f, 1)
        @test flux(:central) != 0
        @test flux(:tcdf) ≈ flux(:central) rtol = 1e-12
        @test flux(:upwind) ≈ flux(:central) rtol = 1e-12
        @test !isapprox(flux(:quick), flux(:central); rtol=1e-6)
    end

    @testset "positivity in a relaxation run" begin
        # a 1 keV Maxwellian relaxing on a 100 eV plasma with E = 1 V/m, 40 steps of 50 μs
        # (fixed temperature, one radius): TCDF, OSPRE and van Albada stay non-negative,
        # QUICK does not (observed 855 negative values over the saved states). MUSCL and
        # SMART go slightly negative far in the tail (2 and 116 values, down to 2e-66 of the
        # maximum). This run is not a proof of positivity: TCDF is not positivity preserving
        # in general, and a cooling plasma is the real test.
        basic_grid = MomentumGrid(uniform_p_faces(300, 0.0, 1.0), uniform_xi_faces(20))
        function relax(scheme)
            g, op = iter_plasma_operator(basic_grid, 100.0, 1.0; scheme)
            layout = UnknownLayout(g.radial_grid, [KineticUnknown(:hot_tail_distribution, g; scale=1e20)])
            system = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(),
                                                      FokkerPlanckTerm(:hot_tail_distribution, op))])
            theta = 1e3 / Constants.ELECTRON_REST_ENERGY
            f0 = [exp(-(sqrt(1 + p^2) - 1) / theta) for p in basic_grid.p_centers, _ in 1:20, _ in 1:1]
            f0 .*= 1e20 / Runes2P.density(g, f0)[1]
            solution = integrate(system, vec(f0), collect(range(0, 2e-3; length=41)))
            return count(<(0), solution.states)
        end
        for scheme in (:tcdf, :ospre, :van_albada)
            @test relax(scheme) == 0
        end
        @test relax(:quick) > 0
    end
end
