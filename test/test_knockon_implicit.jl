# Implicit (matrix-free) knock-on operator: equal to the dense Jacobian block assembled
# from the kernels, usable in sums with sparse operators and in a Krylov solve.
using Krylov: gmres
using LinearAlgebra: I, mul!
using LinearOperators: LinearOperator
using Random: Xoshiro

# Dense ∂C/∂f from the kernels: C_ij = n_tot/𝒱'_ij Σ_kl S_ik Δp_k Δξ_l 𝒱'_kl δ_jl(i, k) f_kl
function knockon_dense(op::KnockOnOperator, n_tot)
    gK, gP = op.knockon, op.primary
    LK, LP = LinearIndices(size(gK)), LinearIndices(size(gP))
    A = zeros(cell_count(gK), cell_count(gP))
    D = zeros(pitch_cell_count(gK), pitch_cell_count(gP))
    for ir in 1:radial_cell_count(gK), i in 1:momentum_cell_count(gK),
        k in 1:momentum_cell_count(gP)
        s = energy_kernel(op, ir).rates[i, k]
        s == 0 && continue
        pitch_kernel_matrix!(D, knockon_kernels(op, ir), i, k)
        for j in 1:pitch_cell_count(gK), l in 1:pitch_cell_count(gP)

            W = momentum_grid(gP).p_widths[k] * momentum_grid(gP).xi_widths[l] *
                phase_space_jacobian(gP, ir)[k, l]
            A[LK[i, j, ir], LP[k, l, ir]] += n_tot[ir] * s * W * D[j, l] /
                                             phase_space_jacobian(gK, ir)[i, j]
        end
    end
    return A
end

@testset "implicit knock-on operator" begin
    rad = RadialGrid(uniform_r_faces(2, 0.5))
    g = cylindrical_grid(MomentumGrid(biuniform_p_faces(18, 6, 0.0, 0.6, 3.0), uniform_xi_faces(7)), rad)
    n_tot = [2e19, 4e19]
    rng = Xoshiro(1)

    @testset "equals the dense Jacobian block ($method)" for method in (:loop, :blas)
        op = KnockOnOperator(g, g, 0.1; method)
        L = knockon_linear_operator(op, n_tot)
        A = knockon_dense(op, n_tot)
        @test size(L) == (cell_count(g), cell_count(g))
        @test Matrix(L) ≈ A rtol = 1e-13
        # signed input (a perturbation, not a distribution)
        v = randn(rng, cell_count(g))
        @test L * v ≈ A * v rtol = 1e-13
        # 5-argument mul!: β = 0 ignores NaN in the output, β ≠ 0 accumulates
        res = fill(NaN, cell_count(g))
        mul!(res, L, v, 2.0, 0.0)
        @test res ≈ 2 .* (A * v) rtol = 1e-13
        res2 = copy(res)
        mul!(res2, L, v, -1.0, 0.5)
        @test maximum(abs, res2) < 1e-13 * maximum(abs, res)
        # n_tot is referenced: updating it in place rescales the operator
        nt = copy(n_tot)
        Lr = knockon_linear_operator(op, nt; scale = -1)
        nt .*= 3
        @test Lr * v ≈ -3 .* (A * v) rtol = 1e-13
    end

    @testset "rectangular: knock-ons on another grid" begin
        gP = cylindrical_grid(MomentumGrid(uniform_p_faces(9, 1.0, 10.0), uniform_xi_faces(5)), rad)
        op = KnockOnOperator(g, gP, 0.1)
        L = knockon_linear_operator(op, n_tot)
        @test size(L) == (cell_count(g), cell_count(gP))
        @test Matrix(L) ≈ knockon_dense(op, n_tot) rtol = 1e-13
    end

    @testset "backward-Euler step with a sparse operator, GMRES" begin
        # (I - Δt (M + K)) f1 = f0 with M a sparse advection-diffusion operator and K the
        # matrix-free knock-on term, against a dense direct solve
        ad = AdvectionDiffusion(g; scheme = :central)
        ad.diffusion_p_p .= 0.3
        ad.diffusion_xi_xi .= 0.2
        ad.advection_p .= -0.5
        M = operator_matrix(ad)
        op = KnockOnOperator(g, g, 0.1)
        K = knockon_linear_operator(op, n_tot)
        A = knockon_dense(op, n_tot)
        f0 = [exp(-p / 0.8) * (1.1 + ξ)
              for p in momentum_grid(g).p_centers, ξ in momentum_grid(g).xi_centers,
        _ in 1:2]
        # scale the knock-on term to O(1) against the unit identity
        Δt = 0.5 / maximum(abs, A)
        n = cell_count(g)
        B = LinearOperator(Matrix{Float64}(I, n, n)) - Δt * (LinearOperator(M) + K)
        @test B * vec(f0) ≈ vec(f0) - Δt * (M * vec(f0) + A * vec(f0)) rtol = 1e-13
        f1, stats = gmres(B, vec(f0); rtol = 1e-12, atol = 0.0)
        @test stats.solved
        @test f1 ≈ (I - Δt * (Matrix(M) + A)) \ vec(f0) rtol = 1e-10
    end

    @testset "explicit and implicit forms agree on one application" begin
        op = KnockOnOperator(g, g, 0.1)
        f = rand(rng, size(g)...)
        C = zeros(size(g))
        apply!(C, op, f, n_tot)
        @test knockon_linear_operator(op, n_tot) * vec(f) ≈ vec(C) rtol = 1e-14
        @test_throws DimensionMismatch knockon_linear_operator(op, [1.0])
    end
end
