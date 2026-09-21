# Knock-on operator in its default configuration (primary averaged over its pitch cell,
# ξ* at the midpoint of the receiving interval) against a reference built from adaptive
# quadrature only: the energy part from the Møller cross section, the pitch part from
# the gyrophase fraction integrated over the primary cell. The DREAM comparison covers
# the DREAM-compatible configuration only; the loop, BLAS and dense forms share the
# kernels.
using QuadGK: quadgk

@testset "knock-on operator against quadrature" begin
    rad = RadialGrid(uniform_r_faces(1, 0.5))
    mom = PXiGrid(uniform_p_faces(14, 0.0, 4.0), -cos.(range(0, π; length=7)))
    g = cylindrical_grid(mom, rad)
    p_cut = 0.15
    np, nξ = n_p(g), n_xi(g)
    β(p) = p / sqrt(1 + p^2)

    # receiving interval of pair (i, k)
    interval(i, k) = (max(mom.p_f[i], p_cut), min(mom.p_f[i+1], max_knockon_momentum(mom.p[k])))
    pairs = [(i, k) for i in 1:np, k in 1:np if >(reverse(interval(i, k))...)]

    # Δξ_j δ_jl: gyrophase fraction reaching cell j, averaged over primary cell l. The
    # fraction has square-root kinks in ξ₁ where an edge ξ₁ξ* ± √((1-ξ₁²)(1-ξ*²)) of the
    # knock-on pitch range meets a face t; squaring gives ξ₁ = tξ* ± √((1-t²)(1-ξ*²)).
    # Without these breakpoints quadgk misses narrow kink regions (1e-5 errors while
    # reporting 1e-14).
    function fraction(j, l, ξs)
        lo, hi = mom.xi_f[l], mom.xi_f[l+1]
        cuts = [ξs * t + s * sqrt((1 - t^2) * (1 - ξs^2)) for t in mom.xi_f for s in (-1, 1)]
        pts = sort!([lo; filter(x -> lo < x < hi, cuts); hi])
        return quadgk(ξ1 -> pitch_cell_fraction(mom.xi_f[j], mom.xi_f[j+1], ξ1, ξs), pts...;
                      rtol=1e-12, atol=1e-14)[1] / mom.dxi[l]
    end

    @testset "pitch columns of the default kernel" begin
        # entries Δξ_j δ_jl against the reference; the coarse cells near ξ = 0 of this
        # cosine grid are the hardest case (measured 7.9e-4 with 4 points per piece, the
        # default, and 3.3e-7 with 12)
        col = zeros(nξ)
        errs = map((4, 12)) do pts
            nodes, weights = Runes2P.gauss(pts)
            err = 0.0
            for ξs in (0.2, 0.45, 0.7, 0.93), l in 1:nξ
                delta_column_split!(col, mom.xi_f, ξs; cell=(mom.xi_f[l], mom.xi_f[l+1]), nodes, weights)
                err = max(err, maximum(abs(mom.dxi[j] * col[j] - fraction(j, l, ξs)) for j in 1:nξ))
            end
            err
        end
        @test errs[1] < 1e-3
        @test errs[2] < 1e-6
    end

    # reference operator: C_ij = Σ_k Σ_l W_kl β₁ ∫ βΣ dp · fraction(j, l, ξ*_mid) / (𝒱'_ij Δp_i Δξ_j)
    f = [exp(-p / 0.9) * exp(2ξ) for p in mom.p, ξ in mom.xi, _ in 1:1]
    Cref = zeros(size(g))
    for (i, k) in pairs
        pL, pU = interval(i, k)
        p1 = mom.p[k]
        rate = β(p1) * quadgk(p -> β(p) * moller_cross_section(p, p1), pL, pU; rtol=1e-12)[1]
        ξs = xi_star((pL + pU) / 2, p1)
        for l in 1:nξ
            W = mom.dp[k] * mom.dxi[l] * g.Vp[k, l, 1] * f[k, l, 1]
            for j in 1:nξ
                Cref[i, j, 1] += rate * W * fraction(j, l, ξs) /
                                 (g.Vp[i, j, 1] * mom.dp[i] * mom.dxi[j])
            end
        end
    end
    wts = [g.Vp[i, j, 1] * mom.dp[i] * mom.dxi[j] for i in 1:np, j in 1:nξ, _ in 1:1]
    L1(C) = sum(wts .* abs.(C .- Cref)) / sum(wts .* abs.(Cref))

    @testset "operator: table interpolation converging" begin
        # with 12 points per piece the remaining difference is the linear interpolation
        # in ξ* between table slices
        errs = map((20, 80, 320)) do n_tab
            C = zeros(size(g))
            apply!(C, KnockOnOperator(g, g, p_cut; n_tab, primary_points=12), f, [1.0])
            L1(C)
        end
        @test errs[1] > 3 * errs[2] > 9 * errs[3]  # measured 9.2e-5, 5.0e-6, 2.8e-7
        C = zeros(size(g))
        apply!(C, KnockOnOperator(g, g, p_cut), f, [1.0])
        # default configuration (4 points per piece, 80 slices): the primary quadrature
        # dominates on this coarse grid (measured 5.2e-5)
        @test L1(C) < 2e-4
        # production is exact for every table size
        C = zeros(size(g))
        apply!(C, KnockOnOperator(g, g, p_cut; n_tab=20), f, [1.0])
        @test density(g, C) ≈ density(g, Cref) rtol = 1e-10
    end

    @testset "one ξ* per momentum pair" begin
        # The pitch kernel of pair (i, k) is evaluated at a single ξ*, at the midpoint of
        # the receiving interval, while ξ*(p) varies across it. Reference: the kernel
        # integrated over the interval with the Møller weight βΣ (composite Gauss in p,
        # 12-point columns). Isotropic primaries are unaffected (every column gives the
        # same isotropic result); anisotropic ones are not, and the error falls with the
        # momentum cell width.
        nodes, weights = Runes2P.gauss(12)
        xg, wg = Runes2P.gauss(8)
        function collapse_error(np, f)
            m = PXiGrid(uniform_p_faces(np, 0.0, 4.0), mom.xi_f)
            gg = cylindrical_grid(m, rad)
            col = zeros(nξ)
            D = zeros(nξ, nξ)
            function columns!(D, ξs)
                for l in 1:nξ
                    delta_column_split!(col, m.xi_f, ξs; cell=(m.xi_f[l], m.xi_f[l+1]), nodes, weights)
                    D[:, l] .= col
                end
                return D
            end
            Cmid, Cint = zeros(np, nξ), zeros(np, nξ)
            for i in 1:np, k in 1:np
                p1 = m.p[k]
                pL, pU = max(m.p_f[i], p_cut), min(m.p_f[i+1], max_knockon_momentum(p1))
                pU > pL || continue
                W = [m.dp[k] * m.dxi[l] * gg.Vp[k, l, 1] * f(m.p[k], m.xi[l]) for l in 1:nξ]
                rate = β(p1) * (moller_flux(pU, p1) - moller_flux(pL, p1))
                Cmid[i, :] .+= rate .* (columns!(D, xi_star((pL + pU) / 2, p1)) * W)
                # composite Gauss over 16 sub-intervals of the receiving interval
                h = (pU - pL) / 16
                for q in 0:15, (x, w) in zip(xg, wg)
                    p = pL + h * (q + (x + 1) / 2)
                    wt = w * h / 2 * β(p1) * β(p) * moller_cross_section(p, p1)
                    Cint[i, :] .+= wt .* (columns!(D, xi_star(p, p1)) * W)
                end
            end
            return sum(abs.(Cmid .- Cint)) / sum(abs.(Cint))  # C·𝒱'ΔpΔξ, so no division
        end
        # (16 and 64 sub-intervals give the same reference to four digits)
        @test collapse_error(14, (p, ξ) -> exp(-p)) < 1e-6
        e28 = collapse_error(28, (p, ξ) -> exp(-p) * exp(3ξ))
        e56 = collapse_error(56, (p, ξ) -> exp(-p) * exp(3ξ))
        # measured 2.3e-2 and 6.1e-3 (1.5e-3 at 112 cells): second order once Δp is
        # small, and far above the table error of the previous test set (5e-6 at 80
        # slices), so the single ξ* dominates the pitch error of the operator
        @test e28 > 3 * e56
        @test e56 > 1e-3
    end
end
