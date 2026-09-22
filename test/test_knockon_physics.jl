# Physics limits of the knock-on operator: total production against n_tot ∫ v σ_tot f,
# the mean knock-on pitch, and the Rosenbluth-Putvinski limit of the DREAM theory notes
# (doc/notes/theory.tex, "Chiu-Harvey and Rosenbluth-Putvinski approximations").
using QuadGK: quadgk

@testset "knock-on physics" begin
    scale = 2π * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT
    rad = RadialGrid(uniform_r_faces(1, 0.5))
    Vp_vol = rad.spatial_jacobian[1]
    n_tot = [3e19]

    @testset "total production against n_tot ∫ v σ_tot f" begin
        # f = g(p)(1 + ξ/2), linear in ξ so the pitch midpoint rule is exact. Knock-ons
        # produced on the knock-on grid plus those above it equal the continuous rate
        # n_tot ∫ 2π p1² g(p1) ∫(1 + ξ/2)dξ R(p1) dp1, R = moller_total_rate(p1, p_cut),
        # up to the midpoint rule in p1 (second order).
        g(p) = exp(-(p - 3)^2 / 2)
        p_cut = 0.15
        kink = sqrt((2 * sqrt(1 + p_cut^2) - 1)^2 - 1)  # R(p1) = 0 below
        ref = n_tot[1] * 2 * quadgk(p1 -> 2π * p1^2 * g(p1) * moller_total_rate(p1, p_cut),
                                    kink, 12.0; rtol=1e-13)[1]
        knock = cylindrical_grid(MomentumGrid(biuniform_p_faces(40, 20, 0.0, 1.0, 2.0),
                                         uniform_xi_faces(8)), rad)
        errs = map((60, 120, 240)) do n
            prim = cylindrical_grid(MomentumGrid(uniform_p_faces(n, 0.0, 12.0), uniform_xi_faces(6)), rad)
            f = [g(p) * (1 + ξ / 2)
                 for p in momentum_grid(prim).p_centers, ξ in momentum_grid(prim).xi_centers, _ in 1:1]
            op = KnockOnOperator(knock, prim, p_cut)
            C = zeros(size(knock))
            apply!(C, op, f, n_tot)
            above = n_tot[1] * sum(phase_space_jacobian(prim, 1)[k, l] * momentum_grid(prim).p_widths[k] *
                                   momentum_grid(prim).xi_widths[l] * f[k, l, 1] *
                                   energy_kernel(op).rates_above_grid[k] for k in 1:n, l in 1:6) / Vp_vol
            abs((density(knock, C)[1] + above) / ref - 1)
        end
        @test errs[end] < 1e-4
        @test errs[1] / errs[3] > 20  # second order; measured 31
    end

    @testset "isotropic primaries give isotropic knock-ons" begin
        # Rotational invariance: an isotropic primary population produces knock-ons whose
        # pitch distribution is isotropic at every momentum. With the primary averaged over
        # its cell (default) this holds to the quadrature accuracy for every tabulated and
        # interpolated slice; DREAM's cell-centre primary misplaces ~10% in some cells.
        g = cylindrical_grid(MomentumGrid(biuniform_p_faces(30, 10, 0.0, 0.5, 5.0),
                                     -cos.(range(0, π; length=17))), rad)
        f = [exp(-p / 1.2) for p in momentum_grid(g).p_centers, _ in momentum_grid(g).xi_centers, _ in 1:1]
        dxi = momentum_grid(g).xi_widths
        aniso(C) = maximum(maximum(abs, C[i, :, 1] ./ (sum(dxi .* C[i, :, 1]) / 2) .- 1)
                           for i in 1:momentum_cell_count(g) if any(>(0), C[i, :, 1]))
        res = map(((primary_pitch=:centre,), (primary_pitch=:split, primary_pitch_points=4),
                   (primary_pitch=:split, primary_pitch_points=10))) do kw
            C = zeros(size(g))
            apply!(C, KnockOnOperator(g, g, 0.1; kw...), f, n_tot)
            aniso(C)
        end
        @test res[1] > 0.05   # measured 0.11
        @test res[2] < 1e-3   # measured 3.2e-4
        @test res[3] < 1e-5   # measured 1.1e-6
    end

    @testset "mean knock-on pitch" begin
        # a primary at pitch ξ₁ produces knock-ons with mean pitch ξ* ξ₁ (gyrophase
        # average); the discrete first pitch moment of C converges to it
        prim = cylindrical_grid(MomentumGrid([4.9, 5.1], [0.6, 0.62]), rad)  # one cell, ξ₁ = 0.61
        f = ones(1, 1, 1)
        errs = map((16, 64, 256)) do nξ
            knock = cylindrical_grid(MomentumGrid(uniform_p_faces(10, 0.0, 2.0), uniform_xi_faces(nξ)), rad)
            op = KnockOnOperator(knock, prim, 0.1; primary_pitch_points=1)
            C = zeros(size(knock))
            apply!(C, op, f, n_tot)
            err = 0.0
            for i in 1:momentum_cell_count(knock)
                w = [phase_space_jacobian(knock, 1)[i, j] * momentum_grid(knock).xi_widths[j] * C[i, j, 1]
                     for j in 1:nξ]
                sum(w) == 0 && continue
                pL = max(momentum_grid(knock).p_faces[i], 0.1)
                pU = min(momentum_grid(knock).p_faces[i+1], max_knockon_momentum(5.0))
                ξs = xi_star((pL + pU) / 2, 5.0)
                err = max(err, abs(sum(w .* momentum_grid(knock).xi_centers) / sum(w) - ξs * 0.61))
            end
            err
        end
        @test errs[end] < 1e-4
        @test errs[1] / errs[3] > 50  # cell-centre moment: O(Δξ²)
    end

    @testset "Rosenbluth-Putvinski limit" begin
        # Primaries at p1 → ∞ along the field (ξ₁ → 1), density n_RE. The notes give
        #   C_ij = n_tot n_RE V'/𝒱'_ij 2πr₀²c/(Δp_i Δξ_j) [1/(γ(ρ₁)-1) - 1/(γ(ρ₂)-1)]
        # in the pitch cell j of the knock-on angle, with ρ₁ = max(p_{i-1/2}, p_c),
        # ρ₂ = p_{i+1/2}. The momentum part of the finite-p1 operator approaches it like
        # 1/p1². The pitch part is a point mass per column; linear interpolation between
        # two tabulated slices splits it over two cells, so the placement is checked
        # with a fine table.
        knock = cylindrical_grid(MomentumGrid(biuniform_p_faces(30, 10, 0.0, 0.5, 3.0),
                                         uniform_xi_faces(20)), rad)
        p_cut = 0.12
        γ(p) = sqrt(1 + p^2)
        n_RE = 1e16
        h = 1e-9
        errs = map((1e2, 1e3, 1e4)) do p1
            prim = cylindrical_grid(MomentumGrid([p1 - 1, p1 + 1], [-1.0, 1 - 2h, 1.0]), rad)
            f = zeros(1, 2, 1)
            cell_volume = phase_space_jacobian(prim, 1)[1, 2] * momentum_grid(prim).p_widths[1] *
                          momentum_grid(prim).xi_widths[2]
            f[1, 2, 1] = n_RE * Vp_vol / cell_volume
            C = zeros(size(knock))
            apply!(C, KnockOnOperator(knock, prim, p_cut; table_size=4000), f, n_tot)
            e_mom, e_pitch = 0.0, 0.0
            for i in 1:momentum_cell_count(knock)
                ρ1, ρ2 = max(momentum_grid(knock).p_faces[i], p_cut), momentum_grid(knock).p_faces[i+1]
                ρ2 > ρ1 || continue
                rp = n_tot[1] * n_RE * scale /
                     (2π * momentum_grid(knock).p_centers[i]^2 * momentum_grid(knock).p_widths[i]) *
                     (1 / (γ(ρ1) - 1) - 1 / (γ(ρ2) - 1))
                row = [momentum_grid(knock).xi_widths[j] * C[i, j, 1] for j in 1:pitch_cell_count(knock)]
                e_mom = max(e_mom, abs(sum(row) / rp - 1))
                jstar = searchsortedlast(momentum_grid(knock).xi_faces, xi_star((ρ1 + ρ2) / 2, p1))
                e_pitch = max(e_pitch, 1 - row[jstar] / sum(row))
            end
            (e_mom, e_pitch)
        end
        @test errs[end][1] < 1e-7
        @test errs[1][1] / errs[3][1] > 5e3  # O(1/p1²): measured 2e4
        @test all(e -> e[2] < 1e-12, errs)
    end
end
