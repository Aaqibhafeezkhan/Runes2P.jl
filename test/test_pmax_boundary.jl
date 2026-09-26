# p_max boundary conditions constant_flux (DREAM BC_PHI_CONST), extrapolated_flux (DREAM
# BC_DPHI_CONST, ported as-is, including its apparent Aᵖ index and its shared last-face
# 𝒱') and extrapolated_flux_consistent (a Runes2P departure matching DREAM's own stated
# definition, Φ = 𝒱'Γᵖ, with every face's own 𝒱' and Γᵖ), independent of DREAM.
# The DREAM comparison is test_pmax_boundary_dream.jl.
using ForwardDiff: ForwardDiff
using Random
using SparseArrays: nnz

# smooth f, A and D on [0, 2], non-negligible at p = 2 (the manufactured p_max boundary
# flux test below)
module ManufacturedFlux
g(p) = exp(-p / 2) * (1 + p)
dg(p) = exp(-p / 2) * (0.5 - p / 2)
advection(p) = -(1 + 0.3p)
diffusion(p) = 0.4 + 0.1p
end

_random_op(g, rng, p_upper_boundary; p_scheme=:central) = begin
    op = AdvectionDiffusion(g; p_scheme, xi_scheme=:central, p_upper_boundary)
    for A in (op.advection_p, op.advection_xi, op.diffusion_p_xi, op.diffusion_xi_p)
        A .= randn(rng, size(A))
    end
    for D in (op.diffusion_p_p, op.diffusion_xi_xi)
        D .= rand(rng, size(D))
    end
    op
end

@testset "p_max boundary" begin
    rng = Xoshiro(4)
    mom = MomentumGrid(uniform_p_faces(8, 0.0, 2.0), uniform_xi_faces(5))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 1.0)))
    np = momentum_cell_count(g)

    @testset "grid too small" begin
        tiny = cylindrical_grid(MomentumGrid(uniform_p_faces(1, 0.0, 1.0), uniform_xi_faces(2)),
                                RadialGrid(uniform_r_faces(1, 1.0)))
        @test_throws ArgumentError AdvectionDiffusion(tiny; p_upper_boundary=:constant_flux)
        @test_throws ArgumentError AdvectionDiffusion(tiny; p_upper_boundary=:extrapolated_flux)
        # f_zero on one momentum cell is the outflow boundary (used by the pitch-only tests)
        outflow, f_zero = (AdvectionDiffusion(tiny; p_upper_boundary=b) for b in (:outflow, :f_zero))
        for op in (outflow, f_zero)
            op.advection_p .= 0.7
            op.diffusion_p_p .= 0.3
        end
        f_tiny = reshape([1.0, 2.0], 1, 2, 1)
        @test apply!(zeros(1, 2, 1), f_zero, f_tiny) == apply!(zeros(1, 2, 1), outflow, f_tiny)
        two =cylindrical_grid(MomentumGrid(uniform_p_faces(2, 0.0, 1.0), uniform_xi_faces(2)),
                               RadialGrid(uniform_r_faces(1, 1.0)))
        @test AdvectionDiffusion(two; p_upper_boundary=:constant_flux) isa AdvectionDiffusion
        @test_throws ArgumentError AdvectionDiffusion(two; p_upper_boundary=:extrapolated_flux)
        @test_throws ArgumentError AdvectionDiffusion(two; p_upper_boundary=:extrapolated_flux_consistent)
    end

    @testset "constant_flux: zero net p-divergence at the last cell" for scheme in (:upwind, :central, :exponential_fitting)
        op = _random_op(g, rng, :constant_flux; p_scheme=scheme)
        f = rand(rng, size(g)...)

        # with p transport alone (no xi or r coefficients), the last cell's divergence is
        # exactly zero: influx equals outflux
        op_p_only = AdvectionDiffusion(g; p_scheme=scheme, xi_scheme=:central, p_upper_boundary=:constant_flux)
        op_p_only.advection_p .= op.advection_p
        op_p_only.diffusion_p_p .= op.diffusion_p_p
        df = zeros(size(g))
        apply!(df, op_p_only, f)
        @test maximum(abs, df[np, :, :]) < 1e-12

        # momentum_flux at the boundary equals the flux through the last interior face
        for ir in 1:radial_cell_count(g)
            @test momentum_flux(op, f, np + 1, ir) ≈ momentum_flux(op, f, np, ir)
        end
    end

    @testset "extrapolated_flux matches DREAM's literal formula" begin
        # the shared 𝒱' of the last interior face and both advective terms multiplied by
        # that face's own Aᵖ (DREAM's PXiExternalLoss.cpp, ported as-is including the
        # apparent index inconsistency); the second term's
        # interpolation direction, weights and diffusion coefficient are the
        # second-to-last face's own
        op = _random_op(g, rng, :extrapolated_flux)
        f = rand(rng, size(g)...)
        for ir in 1:radial_cell_count(g), j in 1:pitch_cell_count(g)
            gamma_last = Runes2P._p_interior_gamma(op, f, np, j, ir, nothing)
            A_last = op.advection_p[np, j, ir]
            A_prev = op.advection_p[np-1, j, ir]
            f_left, f_right = f[np-2, j, ir], f[np-1, j, ir]
            distance_prev = mom.p_centers[np-1] - mom.p_centers[np-2]
            weight_left = (mom.p_centers[np-1] - mom.p_faces[np-1]) / distance_prev
            face_value_prev = Runes2P._face_value(:central, A_prev, f_left, f_right, weight_left)
            gamma_prev_dream = A_last * face_value_prev -
                                op.diffusion_p_p[np-1, j, ir] * (f_right - f_left) / distance_prev
            V_shared = op.jacobian_p_faces[np, j, ir]
            dd = mom.p_widths[np] / mom.p_widths[np-1]
            expected = V_shared * ((1 + dd) * gamma_last - dd * gamma_prev_dream)
            @test Runes2P._p_flux(op, f, np + 1, j, ir) ≈ expected rtol = 1e-12
        end
    end

    @testset "extrapolated_flux_consistent matches the corrected formula, and differs from the literal port" begin
        # every face's own 𝒱' and Γᵖ throughout: 𝒱'Γᵖ at each of the last two interior
        # faces (_p_flux, which already uses each face's own jacobian_p_faces), linearly
        # extrapolated with no shared 𝒱' (unlike DREAM's code)
        op = _random_op(g, rng, :extrapolated_flux_consistent)
        f = rand(rng, size(g)...)
        for ir in 1:radial_cell_count(g), j in 1:pitch_cell_count(g)
            phi_last = Runes2P._p_flux(op, f, np, j, ir)
            phi_prev = Runes2P._p_flux(op, f, np - 1, j, ir)
            dd = mom.p_widths[np] / mom.p_widths[np-1]
            expected = (1 + dd) * phi_last - dd * phi_prev
            @test Runes2P._p_flux(op, f, np + 1, j, ir) ≈ expected rtol = 1e-12
        end

        # the two boundaries disagree in general: they share coefficients but not the
        # index that multiplies the second term's Aᵖ, nor whether 𝒱' is each face's own
        # or the last interior face's shared between both terms
        op_literal = AdvectionDiffusion(g; p_upper_boundary=:extrapolated_flux)
        op_literal.advection_p .= op.advection_p
        op_literal.diffusion_p_p .= op.diffusion_p_p
        df_consistent, df_literal = zeros(size(g)), zeros(size(g))
        apply!(df_consistent, op, f)
        apply!(df_literal, op_literal, f)
        @test maximum(abs, df_consistent[np, :, :] - df_literal[np, :, :]) > 1e-3 * maximum(abs, df_literal[np, :, :])
    end

    @testset "order of the p_max flux by a manufactured solution" begin
        # smooth f, A and D, non-negligible at p_max (unlike ManufacturedSolution of
        # test_manufactured_solutions.jl, chosen there to vanish at p_max so the f = 0
        # ghost of :f_zero is consistent with it); the exact boundary flux is
        # Vf(p_max) * (A g - D g')(p_max)
        mg = ManufacturedFlux
        function boundary_flux_error(n, boundary)
            faces = uniform_p_faces(n, 0.0, 2.0)
            mom1 = MomentumGrid(faces, [-1.0, 1.0])
            g1 = cylindrical_grid(mom1, RadialGrid(uniform_r_faces(1, 1.0)))
            op1 = AdvectionDiffusion(g1; p_scheme=:central, xi_scheme=:central, p_upper_boundary=boundary)
            for k in 1:n+1
                op1.advection_p[k, 1, 1] = mg.advection(faces[k])
                op1.diffusion_p_p[k, 1, 1] = mg.diffusion(faces[k])
            end
            f = [mg.g(p) for p in mom1.p_centers, _ in 1:1, _ in 1:1]
            Vf = phase_space_jacobian_p_faces(g1, 1)[end, 1]
            exact = Vf * (mg.advection(2.0) * mg.g(2.0) - mg.diffusion(2.0) * mg.dg(2.0))
            computed = Runes2P._p_flux(op1, f, n + 1, 1, 1)
            return abs(computed - exact)
        end
        for (boundary, min_order) in ((:extrapolated_flux_consistent, 1.8), (:extrapolated_flux, 0.8))
            errors = [boundary_flux_error(n, boundary) for n in (50, 100, 200, 400)]
            orders = log2.(errors[1:end-1] ./ errors[2:end])
            # observed (50 to 400 cells): :extrapolated_flux_consistent orders 2.03, 2.01,
            # 2.01 (every face its own 𝒱' and Γᵖ, second order); :extrapolated_flux
            # (DREAM's literal, shared 𝒱', wrong-face Aᵖ) 0.99, 0.99, 1.00, first order, a
            # DREAM-independent accuracy statement about BC_DPHI_CONST itself
            @test all(>(min_order), orders)
        end
    end

    @testset "order of the p_max flux with dd != 1 (last cell twice the previous one's width)" begin
        # every other test here has a uniform grid, where dd = 1 everywhere and an
        # inverted ratio (1/dd for dd) would still pass; this grid's last cell is about
        # twice the second-to-last one's width, so a correct dd = 2 is needed for order 2
        mg = ManufacturedFlux
        function boundary_flux_error_dd(n)
            h = 2.0 / (n + 1)
            faces = [collect(0:h:(n-1)*h); 2.0]
            mom1 = MomentumGrid(faces, [-1.0, 1.0])
            g1 = cylindrical_grid(mom1, RadialGrid(uniform_r_faces(1, 1.0)))
            op1 = AdvectionDiffusion(g1; p_scheme=:central, xi_scheme=:central,
                                     p_upper_boundary=:extrapolated_flux_consistent)
            for k in 1:n+1
                op1.advection_p[k, 1, 1] = mg.advection(faces[k])
                # D = 0: a pure advective face value, still exact for a linear f at any
                # cell-width ratio, isolates dd from a diffusive Dᵖᵖ/Δp sensitivity to it
            end
            f = [mg.g(p) for p in mom1.p_centers, _ in 1:1, _ in 1:1]
            Vf = phase_space_jacobian_p_faces(g1, 1)[end, 1]
            exact = Vf * mg.advection(2.0) * mg.g(2.0)
            computed = Runes2P._p_flux(op1, f, n + 1, 1, 1)
            return abs(computed - exact)
        end
        errors = [boundary_flux_error_dd(n) for n in (50, 100, 200, 400)]
        orders = log2.(errors[1:end-1] ./ errors[2:end])
        # a throwaway check with dd inverted (p_widths[np-1]/p_widths[np]) dropped this to
        # 0.98-0.99, so the order bound below does discriminate an inverted ratio
        @test all(>(1.8), orders)
    end

    @testset "check_stencil_pattern" for boundary in (:constant_flux, :extrapolated_flux, :extrapolated_flux_consistent),
                                          scheme in (:upwind, :central, :tcdf)
        op = _random_op(g, rng, boundary; p_scheme=scheme)
        @test check_stencil_pattern(op)
    end

    @testset "operator_matrix reproduces apply!" for boundary in (:extrapolated_flux, :extrapolated_flux_consistent)
        op = _random_op(g, rng, boundary)
        M = operator_matrix(op)
        f = rand(rng, size(g)...)
        df = zeros(size(g))
        apply!(df, op, f)
        @test M * vec(f) ≈ vec(df)
        @test nnz(M) <= 14 * cell_count(g)
    end

    @testset "EquationSystem Jacobian" for boundary in (:extrapolated_flux, :extrapolated_flux_consistent),
                                            scheme in (:central, :tcdf)
        # FokkerPlanckTerm.sparsity! duplicates stencil_pattern(op); check_stencil_pattern
        # only exercises the latter, so this compares the colored-AD stored Jacobian
        # against a dense ForwardDiff Jacobian of the same sparse terms (as
        # test_equation_system.jl does), which would catch entries missing from
        # sparsity! that stencil_pattern already has
        op = _random_op(g, rng, boundary; p_scheme=scheme)
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale=1.0)])
        system = EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op))])
        u = rand(rng, length(layout))
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        function sparse_part(x)
            out = zeros(eltype(x), length(x))
            for bound in system.sparse_terms
                Runes2P.rhs!(out, bound.term, x, layout, bound.row, 0.0)
            end
            return out
        end
        dense = ForwardDiff.jacobian(sparse_part, u)
        @test Matrix(J) ≈ dense rtol = 1e-10
        @test all(jacobian_sparsity(system)[findall(!iszero, dense)])
    end

    @testset "conservation with random coefficients" for boundary in (:constant_flux, :extrapolated_flux,
                                                                        :extrapolated_flux_consistent)
        op = _random_op(g, rng, boundary)
        op.advection_r .= randn(rng, size(op.advection_r))
        op.diffusion_r_r .= rand(rng, size(op.diffusion_r_r))
        f = rand(rng, size(g)...)
        df = zeros(size(g))
        apply!(df, op, f)
        loss = _conservation_residual(g, df)
        # the flux-form divergence telescopes to the p_max face alone: r is :zero_flux and
        # p_min = 0 has no p²Aᵖ term set
        expected = sum(mom.xi_widths[j] * g.radial_grid.r_widths[ir] * Runes2P._p_flux(op, f, np + 1, j, ir)
                       for j in 1:pitch_cell_count(g), ir in 1:radial_cell_count(g))
        @test loss ≈ -expected rtol = 1e-10
    end
end
