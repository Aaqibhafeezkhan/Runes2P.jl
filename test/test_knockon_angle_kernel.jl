# Knock-on pitch kernel in cylindrical geometry. DREAM's cylindrical test
# (KnockOn.cpp, CheckCylindricalDeltaCalculation) compares its functions with each
# other; the references here are the gyrophase average itself, the first pitch moment
# ⟨ξ⟩ = ξ₁ξ*, and ξ ↔ ξ₁ reciprocity.
using QuadGK: quadgk

@testset "knock-on angle kernel" begin
    @testset "gyrophase average" begin
        # fraction of the gyrophase with a + b cos φ in [t1, t2], sampled directly
        function sampled(t1, t2, ξ1, ξs; n=400_000)
            a = ξ1 * ξs
            b = sqrt((1 - ξ1^2) * (1 - ξs^2))
            return count(φ -> t1 <= a + b * cos(φ), (k - 0.5) * 2π / n for k in 1:n) / n -
                   count(φ -> t2 < a + b * cos(φ), (k - 0.5) * 2π / n for k in 1:n) / n
        end
        for (ξ1, ξs) in ((0.3, 0.6), (-0.8, 0.2), (0.95, 0.9)), (t1, t2) in ((-1.0, -0.2), (-0.2, 0.4), (0.4, 1.0))
            @test pitch_cell_fraction(t1, t2, ξ1, ξs) ≈ sampled(t1, t2, ξ1, ξs) atol = 2e-5
        end
        # fractions over a partition of [-1, 1] sum to 1
        xi_f = uniform_xi_faces(13)
        for ξ1 in (-0.9, 0.0, 0.5), ξs in (0.1, 0.7, 0.99)
            @test sum(pitch_cell_fraction(xi_f[j], xi_f[j+1], ξ1, ξs) for j in 1:13) ≈ 1 atol = 1e-14
        end
    end

    @testset "primary along the field: point mass at ξ = ξ*" begin
        xi_f = uniform_xi_faces(10)  # faces at multiples of 0.2
        col = zeros(10)
        delta_column!(col, xi_f, 0.5; cell=(1.0, 1.0))
        @test col[8] ≈ 1 / 0.2 && count(!iszero, col) == 1   # 0.5 ∈ (0.4, 0.6]
        # ξ* exactly on a face goes to exactly one cell
        delta_column!(col, xi_f, 0.6; cell=(1.0, 1.0))
        @test count(!iszero, col) == 1 && sum(0.2 .* col) ≈ 1
        delta_column!(col, xi_f, 1.0; cell=(-1.0, -1.0))   # knock-on at ξ = -1
        @test col[1] ≈ 1 / 0.2 && sum(0.2 .* col) ≈ 1
    end

    knock = PXiGrid(uniform_p_faces(30, 0.0, 6.0), uniform_xi_faces(24))
    prim = PXiGrid(uniform_p_faces(20, 0.0, 6.0), uniform_xi_faces(16))
    p_cut = 0.2
    K = MollerAngleKernel(knock, prim, p_cut)

    @testset "normalization of tabulated and interpolated columns" begin
        @test all(abs(sum(knock.dxi .* K.table[:, l, m]) - 1) < 1e-13
                  for l in 1:n_xi(prim), m in eachindex(K.xs))
        D = zeros(n_xi(knock), n_xi(prim))
        ok = true
        for i in 1:n_p(knock), k in 1:n_p(prim)
            delta_matrix!(D, K, i, k)
            ok &= all(abs(sum(knock.dxi .* D[:, l]) - 1) < 1e-13 for l in 1:n_xi(prim))
        end
        @test ok
        @test all(>=(0), K.table)
    end

    @testset "first pitch moment ⟨ξ⟩ = ξ₁ξ*" begin
        # The gyro-averaged knock-on pitch has mean ξ₁ξ*. The density has inverse-square-root
        # edges, so the cell-centre moment converges like Δξ^(3/2) (observed error·n^1.5 of
        # 0.25 to 0.52, depending on where the sampled ξ* fall).
        ns = (50, 200, 800)
        errs = map(ns) do n2
            fine = PXiGrid(uniform_p_faces(30, 0.0, 6.0), uniform_xi_faces(n2))
            Kf = MollerAngleKernel(fine, prim, p_cut)
            maximum(abs(sum(fine.dxi .* fine.xi .* Kf.table[:, l, m]) - prim.xi[l] * Kf.xs[m])
                    for m in (1, 40, 80), l in (2, 8, 15))
        end
        @test all(errs .* ns .^ 1.5 .< 0.6)
        @test errs[end] < 3e-5
    end

    @testset "reciprocity with the primary averaged over its cell" begin
        # A[j, l] = ∫_j ∫_l g(ξ | ξ₁) dξ dξ₁ is symmetric because the gyro-averaged density
        # is symmetric in ξ ↔ ξ₁. DREAM evaluates the primary at its cell centre, which
        # breaks the symmetry at O(Δξ²); averaging with Gauss points restores it.
        pg = PXiGrid(uniform_p_faces(4, 0.5, 6.0), uniform_xi_faces(12))
        ξs = 0.55
        A(K) = pg.dxi .* K .* pg.dxi'
        function kernel(points)
            T = zeros(12, 12)
            nodes, weights = points == 1 ? ([0.0], [2.0]) : Runes2P.gauss(points)
            for l in 1:12
                cell = points == 1 ? (pg.xi[l], pg.xi[l]) : (pg.xi_f[l], pg.xi_f[l+1])
                delta_column!(view(T, :, l), pg.xi_f, ξs; cell, nodes, weights)
            end
            return T
        end
        # reference: adaptive quadrature over the primary cell
        ref = [quadgk(ξ1 -> pitch_cell_fraction(pg.xi_f[j], pg.xi_f[j+1], ξ1, ξs),
                      pg.xi_f[l], pg.xi_f[l+1]; rtol=1e-12)[1] for j in 1:12, l in 1:12]
        @test maximum(abs, ref - ref') < 1e-10
        asym(K) = maximum(abs, A(K) - A(K)') / maximum(abs, A(K))
        err(K) = maximum(abs, A(K) - ref) / maximum(abs, ref)
        @test asym(kernel(1)) > 1e-3             # cell-centre primary (DREAM)
        @test asym(kernel(16)) < asym(kernel(1)) / 20
        @test err(kernel(16)) < err(kernel(1)) / 20
    end

    @testset "ξ* range: kernel support versus DREAM's criterion" begin
        # On a grid below γ₁ ≈ 3, DREAM's range misses contributing pairs, which are then
        # clamped to the last tabulated slice. Runes2P's table covers the ξ* of every pair
        # that produces knock-ons.
        g2 = PXiGrid(uniform_p_faces(30, 0.0, 2.0), uniform_xi_faces(8))
        pc = 0.1
        interval(i, k) = (max(g2.p_f[i], pc), min(g2.p_f[i+1], max_knockon_momentum(g2.p[k])))
        contributing = [(i, k) for i in 1:30, k in 1:30 if >(reverse(interval(i, k))...)]
        Ks = MollerAngleKernel(g2, g2, pc)
        Kd = MollerAngleKernel(g2, g2, pc; dream=true)
        xs_runes2p = [xi_star(sum(interval(i, k)) / 2, g2.p[k]) for (i, k) in contributing]
        xs_dream = [xi_star(g2.p[i], g2.p[k]) for (i, k) in contributing]
        outside(K, xs) = count(x -> x < first(K.xs) || x > last(K.xs), xs)
        @test outside(Ks, xs_runes2p) == 0
        @test outside(Kd, xs_dream) > length(contributing) ÷ 4
        @test last(Kd.xs) < 0.52
    end

    @testset "interpolation in ξ* converges with the table size" begin
        # δ_j(ξ*) has square-root kinks where the edges a ± b of the gyro-averaged
        # distribution cross a cell face: typical entries converge like Δξ*², the worst
        # ones like Δξ*^(1/2..1). Errors are measured as fractions of the column mass Δξ δ.
        function interp_errs(n_tab)
            Kt = MollerAngleKernel(knock, prim, p_cut; n_tab)
            D = zeros(n_xi(knock), n_xi(prim))
            exact = similar(D)
            errs = Float64[]
            for i in 1:n_p(knock), k in 1:n_p(prim)
                min(knock.p_f[i+1], max_knockon_momentum(prim.p[k])) > max(knock.p_f[i], p_cut) || continue
                delta_matrix!(D, Kt, i, k)
                pL = max(knock.p_f[i], p_cut)
                pU = min(knock.p_f[i+1], max_knockon_momentum(prim.p[k]))
                xs = xi_star((pL + pU) / 2, prim.p[k])
                for l in 1:n_xi(prim)
                    delta_column!(view(exact, :, l), knock.xi_f, xs; cell=(prim.xi[l], prim.xi[l]))
                end
                push!(errs, maximum(abs, (D - exact) .* knock.dxi))
            end
            sort!(errs)
            return errs[end], errs[end÷2]
        end
        max80, med80 = interp_errs(80)
        max320, med320 = interp_errs(320)
        @test med320 < med80 / 10       # observed 19
        @test max320 < max80 / 2        # observed 2.4
        @test max80 < 0.05 && med80 < 0.01
    end

    @testset "gyro-average keeps all knock-ons on the pitch grid (unnormalized norm = 1)" begin
        # a ± b = cos(θ₁ ∓ θ*) lies in [-1, 1], so the column integral before normalization
        # is 1 in cylindrical geometry; the normalization cannot hide an error in a or b
        col = zeros(n_xi(knock))
        nodes, weights = Runes2P.gauss(6)
        ok = true
        for ξs in range(0.0, 1.0; length=41), l in 1:n_xi(prim)
            ok &= abs(delta_column!(col, knock.xi_f, ξs; cell=(prim.xi[l], prim.xi[l])) - 1) < 1e-13
            ok &= abs(delta_column!(col, knock.xi_f, ξs; cell=(prim.xi_f[l], prim.xi_f[l+1]),
                                    nodes, weights) - 1) < 1e-13
        end
        @test ok
    end

    @testset "knock-on grid coarser than the primary grid" begin
        # knock-on cell [0, 1] receives knock-ons from primaries with p1 < 0.5, whose
        # momentum is below the cell centre; ξ* at the cell centre would exceed 1
        kn = PXiGrid(uniform_p_faces(3, 0.0, 3.0), uniform_xi_faces(6))
        pr = PXiGrid(uniform_p_faces(30, 0.0, 3.0), uniform_xi_faces(6))
        @test xi_star(kn.p[1], pr.p[3]) > 1
        Kc = MollerAngleKernel(kn, pr, 0.1)
        @test all(isfinite, Kc.table) && last(Kc.xs) < 1
    end

    @testset "cutoff above the knock-on grid" begin
        Ke = MollerAngleKernel(knock, prim, 10.0)
        @test Ke.empty
        @test all(isfinite, Ke.table)
        @test_throws ArgumentError MollerAngleKernel(knock, prim, 0.2; n_tab=1)
    end
end
