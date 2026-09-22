# Knock-on operator in its default configuration (primary averaged over its pitch cell,
# ξ* integrated over the receiving interval) against a reference built from
# quadrature only: the energy part from the Møller cross section, the pitch part from the
# gyrophase fraction integrated over the primary cell. The DREAM comparison covers the
# DREAM-compatible configuration only; the loop, BLAS and dense forms share the kernels.
using QuadGK: quadgk

@testset "knock-on operator against quadrature" begin
    rad = RadialGrid(uniform_r_faces(1, 0.5))
    mom = MomentumGrid(uniform_p_faces(14, 0.0, 4.0), -cos.(range(0, π; length=7)))
    g = cylindrical_grid(mom, rad)
    p_cut = 0.15
    np, nξ = momentum_cell_count(g), pitch_cell_count(g)
    β(p) = p / sqrt(1 + p^2)

    # receiving interval of pair (i, k)
    interval(i, k) = (max(mom.p_faces[i], p_cut),
                      min(mom.p_faces[i+1], max_knockon_momentum(mom.p_centers[k])))
    pairs = [(i, k) for i in 1:np, k in 1:np if >(reverse(interval(i, k))...)]

    # Δξ_j δ_jl: gyrophase fraction reaching cell j, averaged over primary cell l. The
    # fraction has square-root kinks in ξ₁ where an edge ξ₁ξ* ± √((1-ξ₁²)(1-ξ*²)) of the
    # knock-on pitch range meets a face t; squaring gives ξ₁ = tξ* ± √((1-t²)(1-ξ*²)).
    # Without these breakpoints quadgk misses narrow kink regions (1e-5 errors while
    # reporting 1e-14).
    function fraction(j, l, ξs)
        lo, hi = mom.xi_faces[l], mom.xi_faces[l+1]
        cuts = [ξs * t + s * sqrt((1 - t^2) * (1 - ξs^2)) for t in mom.xi_faces for s in (-1, 1)]
        pts = sort!([lo; filter(x -> lo < x < hi, cuts); hi])
        return quadgk(ξ1 -> pitch_cell_fraction(mom.xi_faces[j], mom.xi_faces[j+1], ξ1, ξs), pts...;
                      rtol=1e-12, atol=1e-14)[1] / mom.xi_widths[l]
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
                pitch_kernel_column_split!(col, mom.xi_faces, ξs;
                                           primary_cell=(mom.xi_faces[l], mom.xi_faces[l+1]), nodes, weights)
                err = max(err, maximum(abs(mom.xi_widths[j] * col[j] - fraction(j, l, ξs)) for j in 1:nξ))
            end
            err
        end
        @test errs[1] < 1e-3
        @test errs[2] < 1e-6
    end

    # reference operator: the Møller flux integrated over the receiving interval of each
    # pair against the pitch kernel at ξ*(p), instead of one ξ* per pair,
    # C_ij = Σ_k Σ_l W_kl β₁ ∫ β(p) Σ(p, p1) δ_jl(ξ*(p, p1)) dp / (𝒱'_ij Δp_i).
    # The columns use the split primary quadrature with 12 points, accurate to 1e-6
    # against `fraction` by the test set above.
    f = [exp(-p / 0.9) * exp(2ξ) for p in mom.p_centers, ξ in mom.xi_centers, _ in 1:1]
    nodes12, weights12 = Runes2P.gauss(12)
    column, D = zeros(nξ), zeros(nξ, nξ)
    function columns!(D, ξs)
        for l in 1:nξ
            pitch_kernel_column_split!(column, mom.xi_faces, ξs;
                                       primary_cell=(mom.xi_faces[l], mom.xi_faces[l+1]),
                                       nodes=nodes12, weights=weights12)
            D[:, l] .= column
        end
        return D
    end
    xg, wg = Runes2P.gauss(8)
    function reference(subintervals)
        C = zeros(size(g))
        for (i, k) in pairs
            pL, pU = interval(i, k)
            p1 = mom.p_centers[k]
            W = [mom.p_widths[k] * mom.xi_widths[l] * phase_space_jacobian(g, 1)[k, l] * f[k, l, 1]
                 for l in 1:nξ]
            h = (pU - pL) / subintervals
            for q in 0:subintervals-1, (x, w) in zip(xg, wg)
                p = pL + h * (q + (x + 1) / 2)
                weight = w * h / 2 * β(p1) * β(p) * moller_cross_section(p, p1)
                contribution = weight .* (columns!(D, xi_star(p, p1)) * W)
                for j in 1:nξ
                    # columns! returns δ (per unit ξ), so only 𝒱' Δp_i divides here
                    C[i, j, 1] += contribution[j] / (phase_space_jacobian(g, 1)[i, j] * mom.p_widths[i])
                end
            end
        end
        return C
    end
    Cref = reference(8)
    wts = [phase_space_jacobian(g, 1)[i, j] * mom.p_widths[i] * mom.xi_widths[j]
           for i in 1:np, j in 1:nξ, _ in 1:1]
    L1(C) = sum(wts .* abs.(C .- Cref)) / sum(wts .* abs.(Cref))

    @testset "operator against the integrated reference" begin
        # the reference is converged in the momentum quadrature
        @test L1(reference(16)) < 1e-7  # measured 6.0e-9
        # default configuration: 4 primary points per piece, 80 slices, ξ* integrated over
        # the receiving interval (measured 3.2e-5, the primary quadrature and the table of
        # this coarse pitch grid)
        C = zeros(size(g))
        apply!(C, KnockOnOperator(g, g, p_cut), f, [1.0])
        @test L1(C) < 2e-4
        # one ξ* per pair, at the midpoint of the receiving interval (an earlier Runes2P):
        # 600 times worse on the same grid (measured 1.9e-2, and 3.9e-2 with DREAM's
        # cell-centre ξ*)
        Cmid = zeros(size(g))
        apply!(Cmid, KnockOnOperator(g, g, p_cut; xi_star_mode=:midpoint), f, [1.0])
        @test L1(Cmid) > 100 * L1(C)
        # production is exact in both modes and for every table size
        for kw in ((;), (xi_star_mode=:midpoint,), (table_size=20,))
            Cp = zeros(size(g))
            apply!(Cp, KnockOnOperator(g, g, p_cut; kw...), f, [1.0])
            @test density(g, Cp) ≈ density(g, Cref) rtol = 1e-10
        end
    end

    @testset "table and primary quadrature converging" begin
        # with the ξ* integration the remaining errors are the pitch table and the primary
        # pitch quadrature; refining both converges to the reference
        # measured 8.8e-5, 3.2e-5, 5.9e-6, 3.6e-7
        errs = map(((20, 4), (80, 4), (80, 12), (320, 12))) do (table_size, primary_pitch_points)
            C = zeros(size(g))
            apply!(C, KnockOnOperator(g, g, p_cut; table_size, primary_pitch_points), f, [1.0])
            L1(C)
        end
        @test errs[1] > errs[2] > errs[3] > errs[4]
    end
end
