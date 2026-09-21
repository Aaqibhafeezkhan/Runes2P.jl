# Checks of the finite-volume discretization that do not go through DREAM: matrices
# written out by hand, boundary fluxes, discrete eigenvalues against analytic ones, and
# structural properties (symmetry, sign pattern, allocations).
using LinearAlgebra: eigvals
using SparseArrays: findnz

_cell_weights(g) = [g.Vp[i, j, ir] * g.mom.dp[i] * g.mom.dxi[j] * g.rad.dr[ir]
                    for i in 1:n_p(g), j in 1:n_xi(g), ir in 1:n_r(g)]

@testset "discretization" begin
    @testset "hand-assembled p advection-diffusion matrix" begin
        # Uniform p grid from 0, constant Dᵖᵖ and Aᵖ, zero flux at both p ends.
        n1, D, A = 7, 0.8, -0.3
        mom = PXiGrid(uniform_p_faces(n1, 0.0, 1.4), uniform_xi_faces(1))
        g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
        p, pf, h = mom.p, mom.p_f, mom.dp[1]
        for scheme in (:central, :upwind)
            op = AdvectionDiffusion(g; scheme)
            op.Dpp .= D
            op.Ap[2:n1, :, :] .= A  # boundary faces stay closed
            M = zeros(n1, n1)
            for i in 1:n1, (k, s) in ((i, -1), (i + 1, 1))  # faces below and above cell i
                (k == 1 || k == n1 + 1) && continue
                w = s * pf[k]^2 / (p[i]^2 * h)  # 𝒱'_face / (𝒱'_i Δp), sign of the divergence
                # Γ = A f_face - D (f_k - f_{k-1}) / h
                if scheme === :central
                    M[i, k-1] -= w * A / 2
                    M[i, k] -= w * A / 2
                else  # A < 0: upwind value from the cell above the face
                    M[i, k] -= w * A
                end
                M[i, k] += w * D / h
                M[i, k-1] -= w * D / h
            end
            @test Matrix(operator_matrix(op)) ≈ M rtol = 1e-13
        end
    end

    @testset "hand-assembled radial diffusion matrix with absorbing wall" begin
        n3, D, a = 5, 0.4, 2.0
        g = cylindrical_grid(PXiGrid(uniform_p_faces(1, 0.5, 1.0), uniform_xi_faces(1)),
                             RadialGrid(uniform_r_faces(n3, a)))
        r, rf, h = g.rad.r, g.rad.r_f, g.rad.dr[1]
        op = AdvectionDiffusion(g; rbc=:absorbing)
        op.Drr .= D
        M = zeros(n3, n3)
        for i in 1:n3
            if i > 1
                M[i, i-1] += rf[i] * D / h^2 / r[i]
                M[i, i] -= rf[i] * D / h^2 / r[i]
            end
            if i < n3
                M[i, i+1] += rf[i+1] * D / h^2 / r[i]
                M[i, i] -= rf[i+1] * D / h^2 / r[i]
            else  # f = 0 on the wall face, half a cell away
                M[i, i] -= rf[i+1] * D / (h / 2) / h / r[i]
            end
        end
        @test Matrix(operator_matrix(op)) ≈ M rtol = 1e-13
    end

    @testset "particle budget: loss only through p = 0" begin
        ions = [IonSpecies("D", 1, 1, [4e19, 5e19]),
                IonSpecies("Ar", 18, [Z0 == 2 ? 1e19 : 1e17 for Z0 in 0:18, _ in 1:2])]
        plasma = PlasmaState(T_cold=[50.0, 800.0], ions=ions, E=[3.0, -2.0], B=[5.0, 5.0])
        mom = PXiGrid(biuniform_p_faces(30, 12, 0.0, 0.3, 4.0), uniform_xi_faces(9))
        g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(2, 0.5)))
        rng = Xoshiro(4)
        for settings in (CollisionSettings(mode=:superthermal),
                         CollisionSettings(mode=:full, screening=:non))
            op = fokker_planck_operator(g, plasma, settings; synchrotron=true)
            cm = Runes2P.CollisionModel(plasma, settings)
            f = rand(rng, size(g)...)
            f[end, :, :] .= 0  # nothing leaves through p_max
            df = zeros(size(g))
            apply!(df, op, f)
            change = sum(_cell_weights(g) .* df)
            expected = -sum(2π * g.rad.VpVol[ir] * mom.dxi[j] * g.rad.dr[ir] *
                            p3nu_s_at_zero(cm, ir) * f[1, j, ir]
                            for j in 1:n_xi(g), ir in 1:n_r(g))
            if settings.mode === :superthermal
                @test expected < 0
            end
            @test change ≈ expected rtol = 1e-12 atol = 1e-13 * sum(abs, _cell_weights(g) .* df)
        end
    end

    @testset "conservation with random coefficients, several draws" begin
        mom = PXiGrid(biuniform_p_faces(15, 5, 0.1, 0.6, 3.0), uniform_xi_faces(6))
        g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(4, 1.0)))
        rng = Xoshiro(5)
        W = _cell_weights(g)
        for scheme in (:upwind, :central), _ in 1:5
            op = AdvectionDiffusion(g; scheme, rbc=:zero_flux)
            for A in (op.Ap, op.Axi, op.Ar, op.Dpxi, op.Dxip)
                A .= randn(rng, size(A))
            end
            for D in (op.Dpp, op.Dxixi, op.Drr)
                D .= rand(rng, size(D))
            end
            op.Ap[end, :, :] .= 0
            df = zeros(size(g))
            apply!(df, op, rand(rng, size(g)...))
            @test abs(sum(W .* df)) < 1e-13 * sum(abs, W .* df)
        end
    end

    @testset "pitch-angle scattering: symmetry and Legendre eigenvalues" begin
        # (1/2) ∂_ξ (1 - ξ²) ∂_ξ P_l = -l(l+1)/2 P_l. On a uniform ξ grid the three-point
        # stencil with face coefficients (1 - ξ_f²)/2 maps degree-l polynomials to degree l
        # with the exact leading coefficient, so the low eigenvalues are exact.
        for n2 in (10, 40)
            g = cylindrical_grid(PXiGrid(uniform_p_faces(1, 0.5, 1.0), uniform_xi_faces(n2)),
                                 RadialGrid(uniform_r_faces(1, 1.0)))
            op = AdvectionDiffusion(g)
            for k in 1:n2+1
                op.Dxixi[1, k, 1] = (1 - g.mom.xi_f[k]^2) / 2
            end
            M = Matrix(operator_matrix(op))
            WM = vec(_cell_weights(g)) .* M
            @test maximum(abs, WM - WM') < 1e-14 * maximum(abs, WM)
            λ = sort(real(eigvals(M)); rev=true)
            @test all(abs(λ[l+1] + l * (l + 1) / 2) < 1e-11 for l in 0:4)
        end
    end

    @testset "radial diffusion: Bessel decay rate with absorbing wall" begin
        # ∂_t f = (1/r) ∂_r (r D ∂_r f), f(a) = 0: slowest rate D j₀₁² / a²
        j01 = 2.404825557695773
        D, a = 0.7, 1.3
        errs = Float64[]
        for n3 in (20, 40, 80)
            g = cylindrical_grid(PXiGrid(uniform_p_faces(1, 0.5, 1.0), uniform_xi_faces(1)),
                                 RadialGrid(uniform_r_faces(n3, a)))
            op = AdvectionDiffusion(g; rbc=:absorbing)
            op.Drr .= D
            λ = maximum(real, eigvals(Matrix(operator_matrix(op))))
            push!(errs, abs(-λ / (D * j01^2 / a^2) - 1))
        end
        @test errs[end] < 1e-3
        @test all(log2.(errs[1:end-1] ./ errs[2:end]) .> 1.8)
    end

    @testset "upwind Fokker-Planck matrix keeps f non-negative" begin
        # Non-negative off-diagonals make I - Δt M an M-matrix, so backward Euler
        # preserves positivity for any Δt.
        ions = [IonSpecies("D", 1, 1, [1e20]), IonSpecies("Ne", 10, 2, [1e19])]
        plasma = PlasmaState(T_cold=[20.0], ions=ions, E=[10.0], B=[5.0])
        mom = PXiGrid(biuniform_p_faces(40, 15, 0.0, 0.3, 5.0), uniform_xi_faces(12))
        g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
        op = fokker_planck_operator(g, plasma, CollisionSettings(); synchrotron=true,
                                    scheme=:upwind)
        I_, J_, V_ = findnz(operator_matrix(op))
        @test all(V_[k] >= 0 for k in eachindex(V_) if I_[k] != J_[k])
    end

    @testset "apply! does not allocate" begin
        mom = PXiGrid(biuniform_p_faces(20, 8, 0.0, 0.5, 3.0), uniform_xi_faces(10))
        g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(3, 1.0)))
        for scheme in (:upwind, :central)
            op = AdvectionDiffusion(g; scheme, pmax_bc=:f_zero, rbc=:absorbing)
            op.Dpxi .= 0.1
            f = rand(size(g)...)
            df = zeros(size(g))
            apply!(df, op, f)
            @test (@allocated apply!(df, op, f)) == 0
        end
    end
end
