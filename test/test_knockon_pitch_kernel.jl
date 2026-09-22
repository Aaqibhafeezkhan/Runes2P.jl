# Knock-on pitch kernel in cylindrical geometry. DREAM's cylindrical test
# (KnockOn.cpp, CheckCylindricalDeltaCalculation) compares its functions with each
# other; the references here are the gyrophase average itself, the first pitch moment
# ⟨ξ⟩ = ξ₁ξ*, and ξ ↔ ξ₁ reciprocity.
using QuadGK: quadgk

@testset "knock-on pitch kernel" begin
    @testset "gyrophase average" begin
        # fraction of the gyrophase with a + b cos φ in [t1, t2], sampled directly
        function sampled(t1, t2, ξ1, ξs; n=400_000)
            a = ξ1 * ξs
            b = sqrt((1 - ξ1^2) * (1 - ξs^2))
            return count(φ -> t1 <= a + b * cos(φ), (k - 0.5) * 2π / n for k in 1:n) / n -
                   count(φ -> t2 < a + b * cos(φ), (k - 0.5) * 2π / n for k in 1:n) / n
        end
        for (ξ1, ξs) in ((0.3, 0.6), (-0.8, 0.2), (0.95, 0.9)),
            (t1, t2) in ((-1.0, -0.2), (-0.2, 0.4), (0.4, 1.0))
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
        pitch_kernel_column!(col, xi_f, 0.5; primary_cell=(1.0, 1.0))
        @test col[8] ≈ 1 / 0.2 && count(!iszero, col) == 1   # 0.5 ∈ (0.4, 0.6]
        # ξ* exactly on a face goes to exactly one cell
        pitch_kernel_column!(col, xi_f, 0.6; primary_cell=(1.0, 1.0))
        @test count(!iszero, col) == 1 && sum(0.2 .* col) ≈ 1
        pitch_kernel_column!(col, xi_f, 1.0; primary_cell=(-1.0, -1.0))   # knock-on at ξ = -1
        @test col[1] ≈ 1 / 0.2 && sum(0.2 .* col) ≈ 1
    end

    knock = MomentumGrid(uniform_p_faces(30, 0.0, 6.0), uniform_xi_faces(24))
    prim = MomentumGrid(uniform_p_faces(20, 0.0, 6.0), uniform_xi_faces(16))
    p_cut = 0.2
    K = MollerPitchKernel(knock, prim, p_cut)

    @testset "normalization of tabulated and interpolated columns" begin
        @test all(abs(sum(knock.xi_widths .* K.table[:, l, m]) - 1) < 1e-13
                  for l in 1:pitch_cell_count(prim), m in eachindex(K.xi_star_values))
        D = zeros(pitch_cell_count(knock), pitch_cell_count(prim))
        ok = true
        for i in 1:momentum_cell_count(knock), k in 1:momentum_cell_count(prim)
            pitch_kernel_matrix!(D, K, i, k)
            ok &= all(abs(sum(knock.xi_widths .* D[:, l]) - 1) < 1e-13 for l in 1:pitch_cell_count(prim))
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
            fine = MomentumGrid(uniform_p_faces(30, 0.0, 6.0), uniform_xi_faces(n2))
            Kf = MollerPitchKernel(fine, prim, p_cut)
            maximum(abs(sum(fine.xi_widths .* fine.xi_centers .* Kf.table[:, l, m]) -
                        prim.xi_centers[l] * Kf.xi_star_values[m])
                    for m in (1, 40, 80), l in (2, 8, 15))
        end
        @test all(errs .* ns .^ 1.5 .< 0.6)
        @test errs[end] < 3e-5
    end

    @testset "reciprocity with the primary averaged over its cell" begin
        # A[j, l] = ∫_j ∫_l g(ξ | ξ₁) dξ dξ₁ is symmetric because the gyro-averaged density
        # is symmetric in ξ ↔ ξ₁. DREAM evaluates the primary at its cell centre, which
        # breaks the symmetry at O(Δξ²); averaging with Gauss points restores it.
        pg = MomentumGrid(uniform_p_faces(4, 0.5, 6.0), uniform_xi_faces(12))
        ξs = 0.55
        A(K) = pg.xi_widths .* K .* pg.xi_widths'
        function kernel(points)
            T = zeros(12, 12)
            nodes, weights = points == 1 ? ([0.0], [2.0]) : Runes2P.gauss(points)
            for l in 1:12
                primary_cell = points == 1 ? (pg.xi_centers[l], pg.xi_centers[l]) :
                               (pg.xi_faces[l], pg.xi_faces[l+1])
                pitch_kernel_column!(view(T, :, l), pg.xi_faces, ξs; primary_cell, nodes, weights)
            end
            return T
        end
        # reference: adaptive quadrature over the primary cell
        ref = [quadgk(ξ1 -> pitch_cell_fraction(pg.xi_faces[j], pg.xi_faces[j+1], ξ1, ξs),
                      pg.xi_faces[l], pg.xi_faces[l+1]; rtol=1e-12)[1] for j in 1:12, l in 1:12]
        @test maximum(abs, ref - ref') < 1e-10
        asym(K) = maximum(abs, A(K) - A(K)') / maximum(abs, A(K))
        err(K) = maximum(abs, A(K) - ref) / maximum(abs, ref)
        @test asym(kernel(1)) > 1e-3             # cell-centre primary (DREAM)
        @test asym(kernel(16)) < asym(kernel(1)) / 20
        @test err(kernel(16)) < err(kernel(1)) / 20
    end

    @testset "primary averaged over its cell, split at the kinks (default)" begin
        # Exact cell averages satisfy isotropy (Σ_l Δξ_l δ_jl = 1: isotropic primaries give
        # isotropic knock-ons) and reciprocity (Δξ_j δ_jl Δξ_l symmetric). The split
        # quadrature converges geometrically in the points per piece; the cell-centre
        # primary (DREAM) and plain Gauss over the cell do not.
        function errors(n, ξs, mode, pts)
            g = MomentumGrid([0.0, 1.0], uniform_xi_faces(n))
            d = g.xi_widths
            T = zeros(n, n)
            nodes, weights = Runes2P.gauss(pts)
            for l in 1:n
                primary_cell = (g.xi_faces[l], g.xi_faces[l+1])
                if mode === :centre
                    pitch_kernel_column!(view(T, :, l), g.xi_faces, ξs;
                                         primary_cell=(g.xi_centers[l], g.xi_centers[l]))
                elseif mode === :gauss
                    pitch_kernel_column!(view(T, :, l), g.xi_faces, ξs; primary_cell, nodes, weights)
                else
                    pitch_kernel_column_split!(view(T, :, l), g.xi_faces, ξs; primary_cell, nodes, weights)
                end
            end
            A = d .* T .* d'
            return maximum(abs, T * d .- 1), maximum(abs, A - A') / maximum(abs, A)
        end
        for n in (12, 48), ξs in (0.3, 0.55, 0.9)
            iso_c, asym_c = errors(n, ξs, :centre, 1)
            @test iso_c > 0.05 && asym_c > 0.1
            iso_g, _ = errors(n, ξs, :gauss, 16)
            iso4, asym4 = errors(n, ξs, :split, 4)
            iso12, asym12 = errors(n, ξs, :split, 12)
            @test iso4 < 2e-4 && asym4 < 5e-4      # measured ≤ 1e-4, 3.3e-4
            @test iso12 < 1e-8 && asym12 < 5e-8    # measured ≤ 6e-9, 2.7e-8
            @test iso4 < iso_g / 10
        end
        # ξ* = 1: the knock-on moves along with the primary, so the averaged kernel is the
        # identity (jumps at the faces are split points)
        g = MomentumGrid([0.0, 1.0], uniform_xi_faces(10))
        T = zeros(10, 10)
        nodes, weights = Runes2P.gauss(2)
        for l in 1:10
            pitch_kernel_column_split!(view(T, :, l), g.xi_faces, 1.0;
                                       primary_cell=(g.xi_faces[l], g.xi_faces[l+1]), nodes, weights)
        end
        @test T .* g.xi_widths ≈ [j == l ? 1.0 : 0.0 for j in 1:10, l in 1:10] atol = 1e-12
        @test_throws ArgumentError MollerPitchKernel(knock, prim, p_cut; primary_pitch=:mean)
        @test_throws ArgumentError pitch_kernel_column_split!(zeros(10), g.xi_faces, 0.5;
                                                              primary_cell=(0.1, 0.1), nodes, weights)
    end

    @testset "ξ* range: kernel support versus DREAM's criterion" begin
        # On a grid below γ₁ ≈ 3, DREAM's range misses contributing pairs, which are then
        # clamped to the last tabulated slice. Runes2P's table covers the ξ* of every pair
        # that produces knock-ons.
        g2 = MomentumGrid(uniform_p_faces(30, 0.0, 2.0), uniform_xi_faces(8))
        pc = 0.1
        interval(i, k) = (max(g2.p_faces[i], pc), min(g2.p_faces[i+1], max_knockon_momentum(g2.p_centers[k])))
        contributing = [(i, k) for i in 1:30, k in 1:30 if >(reverse(interval(i, k))...)]
        Ks = MollerPitchKernel(g2, g2, pc)
        Kd = MollerPitchKernel(g2, g2, pc; dream_compatible=true)
        xs_runes2p = [xi_star(sum(interval(i, k)) / 2, g2.p_centers[k]) for (i, k) in contributing]
        xs_dream = [xi_star(g2.p_centers[i], g2.p_centers[k]) for (i, k) in contributing]
        outside(K, xs) = count(x -> x < first(K.xi_star_values) || x > last(K.xi_star_values), xs)
        @test outside(Ks, xs_runes2p) == 0
        @test outside(Kd, xs_dream) > length(contributing) ÷ 4
        @test last(Kd.xi_star_values) < 0.52
    end

    @testset "interpolation in ξ* converges with the table size" begin
        # δ_j(ξ*) has square-root kinks where the edges a ± b of the gyro-averaged
        # distribution cross a cell face: typical entries converge like Δξ*², the worst
        # ones like Δξ*^(1/2..1). Errors are measured as fractions of the column mass Δξ δ.
        function interp_errs(table_size)
            Kt = MollerPitchKernel(knock, prim, p_cut; table_size, primary_pitch=:centre)
            D = zeros(pitch_cell_count(knock), pitch_cell_count(prim))
            exact = similar(D)
            errs = Float64[]
            for i in 1:momentum_cell_count(knock), k in 1:momentum_cell_count(prim)
                min(knock.p_faces[i+1], max_knockon_momentum(prim.p_centers[k])) >
                    max(knock.p_faces[i], p_cut) || continue
                pitch_kernel_matrix!(D, Kt, i, k)
                pL = max(knock.p_faces[i], p_cut)
                pU = min(knock.p_faces[i+1], max_knockon_momentum(prim.p_centers[k]))
                xs = xi_star((pL + pU) / 2, prim.p_centers[k])
                for l in 1:pitch_cell_count(prim)
                    pitch_kernel_column!(view(exact, :, l), knock.xi_faces, xs;
                                         primary_cell=(prim.xi_centers[l], prim.xi_centers[l]))
                end
                push!(errs, maximum(abs, (D - exact) .* knock.xi_widths))
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
        col = zeros(pitch_cell_count(knock))
        nodes, weights = Runes2P.gauss(6)
        ok = true
        for ξs in range(0.0, 1.0; length=41), l in 1:pitch_cell_count(prim)
            ok &= abs(pitch_kernel_column!(col, knock.xi_faces, ξs;
                                           primary_cell=(prim.xi_centers[l], prim.xi_centers[l])) - 1) < 1e-13
            ok &= abs(pitch_kernel_column!(col, knock.xi_faces, ξs;
                                           primary_cell=(prim.xi_faces[l], prim.xi_faces[l+1]),
                                           nodes, weights) - 1) < 1e-13
        end
        @test ok
    end

    @testset "knock-on grid coarser than the primary grid" begin
        # knock-on cell [0, 1] receives knock-ons from primaries with p1 < 0.5, whose
        # momentum is below the cell centre; ξ* at the cell centre would exceed 1
        kn = MomentumGrid(uniform_p_faces(3, 0.0, 3.0), uniform_xi_faces(6))
        pr = MomentumGrid(uniform_p_faces(30, 0.0, 3.0), uniform_xi_faces(6))
        @test xi_star(kn.p_centers[1], pr.p_centers[3]) > 1
        Kc = MollerPitchKernel(kn, pr, 0.1)
        @test all(isfinite, Kc.table) && last(Kc.xi_star_values) < 1
    end

    @testset "cutoff above the knock-on grid" begin
        Ke = MollerPitchKernel(knock, prim, 10.0)
        @test Ke.no_contributing_pairs
        @test all(isfinite, Ke.table)
        @test_throws ArgumentError MollerPitchKernel(knock, prim, 0.2; table_size=1)
    end

    @testset "knock-on pitch grid must span [-1, 1]" begin
        # 44% of the knock-ons of a primary at ξ₁ = -0.1 (ξ* = 0.3) fall below ξ = -0.2;
        # the column normalization would put them back onto the grid
        partial = MomentumGrid(uniform_p_faces(30, 0.0, 6.0), collect(range(-0.2, 1.0; length=13)))
        col = zeros(12)
        @test pitch_kernel_column!(col, partial.xi_faces, 0.3; primary_cell=(-0.1, -0.1)) < 0.6
        @test_throws ArgumentError MollerPitchKernel(partial, prim, p_cut)
        # a primary grid covering part of [-1, 1] is fine
        partial_primary = MomentumGrid(uniform_p_faces(20, 0.0, 6.0), [0.6, 0.8, 1.0])
        @test MollerPitchKernel(knock, partial_primary, p_cut) isa MollerPitchKernel
    end
end

@testset "slice weights of each momentum pair" begin
    # The weights of a pair must sum to its production rate: the hat functions of the table
    # are a partition of unity over the pair's ξ* range, so the sum is a quadrature of the
    # same integral as the analytic Møller flux. This holds pair by pair, which the density
    # moment of the operator (summed over pairs) cannot show. The coarse cells just above
    # the cutoff have the widest ξ* range and the steepest Møller weight.
    rad = RadialGrid(uniform_r_faces(1, 0.5))
    grids = (MomentumGrid(uniform_p_faces(8, 0.0, 4.0), uniform_xi_faces(6)),
             MomentumGrid(biuniform_p_faces(20, 5, 0.0, 0.4, 12.0), -cos.(range(0, π; length=9))),
             MomentumGrid(uniform_p_faces(30, 0.0, 2.2), uniform_xi_faces(12)))
    for mom in grids, p_cut in (0.05, 0.3), table_size in (8, 80, 320)
        g = cylindrical_grid(mom, rad)
        kernels = knockon_kernels(KnockOnOperator(g, g, p_cut; table_size))
        rates = kernels.energy.rates
        worst = 0.0
        for k in axes(rates, 2), i in axes(rates, 1)
            rates[i, k] == 0 && continue
            total = sum(kernels.entry_weight[t] for t in kernels.pair_start[i, k]:kernels.pair_stop[i, k])
            worst = max(worst, abs(total / rates[i, k] - 1))
            # every entry is a share of a positive production
            @test all(kernels.entry_weight[t] > 0
                      for t in kernels.pair_start[i, k]:kernels.pair_stop[i, k])
        end
        @test worst < 1e-14
    end

    @testset "cutoff just below a momentum face" begin
        # The cell below the face keeps a sliver [p_cut, face] that receives knock-ons. Its
        # analytic flux is a difference of two nearly equal values of S and has lost its
        # digits; the weights converge against themselves instead, and the sliver adds
        # nothing measurable to the operator.
        mom = MomentumGrid(biuniform_p_faces(20, 5, 0.0, 0.4, 12.0), -cos.(range(0, π; length=9)))
        g = cylindrical_grid(mom, rad)
        face = mom.p_faces[3]
        f = [exp(-p) * (1 + 0.5xi) for p in mom.p_centers, xi in mom.xi_centers, _ in 1:1]
        at_face = zeros(size(g))
        apply!(at_face, KnockOnOperator(g, g, face), f, [1.0])
        for p_cut in (prevfloat(face), face - 4eps(face), face * (1 - 1e-12), face * (1 - 1e-8))
            op = KnockOnOperator(g, g, p_cut)
            kernels = knockon_kernels(op)
            rates = kernels.energy.rates
            for k in axes(rates, 2), i in axes(rates, 1)
                rates[i, k] == 0 && continue
                total = sum(kernels.entry_weight[t] for t in kernels.pair_start[i, k]:kernels.pair_stop[i, k])
                @test total ≈ rates[i, k] rtol = 1e-14
            end
            C = zeros(size(g))
            apply!(C, op, f, [1.0])
            @test maximum(abs, C - at_face) < 1e-7 * maximum(abs, at_face)
        end
    end
end
