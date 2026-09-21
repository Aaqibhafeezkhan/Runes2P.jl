# Knock-on operator application: DREAM's MollerBoltzmannOperator element by element
# (reference/dream/cases/knockon_operator.py), the per-pair interpolated-matrix reference
# path, and the production moment.
using HDF5

const DREAM_KNOCKON_DATA = joinpath(@__DIR__, "data", "dream_knockon_operator.h5")

# C[:, j, r] from the interpolated matrix of each momentum pair (reference path)
function knockon_reference(op::KnockOnOperator, f, n_tot)
    gK, gP = op.knockon, op.primary
    C = zeros(size(gK))
    D = zeros(n_xi(gK), n_xi(gP))
    for ir in 1:n_r(gK), i in 1:n_p(gK), k in 1:n_p(gP)
        s = energy_kernel(op, ir).S[i, k]
        s == 0 && continue
        delta_matrix!(D, angle_kernel(op, ir), i, k)
        for j in 1:n_xi(gK), l in 1:n_xi(gP)
            W = gP.mom.dp[k] * gP.mom.dxi[l] * gP.Vp[k, l, ir] * f[k, l, ir]
            C[i, j, ir] += n_tot[ir] * s * W * D[j, l] / gK.Vp[i, j, ir]
        end
    end
    return C
end

function max_rel_diff(a, b; floor=1e-12)
    cut = floor * maximum(abs, a)
    return maximum(abs(a[k] - b[k]) / abs(a[k]) for k in eachindex(a) if abs(a[k]) > cut)
end

@testset "knock-on operator" begin
    h5open(DREAM_KNOCKON_DATA) do h5
        @test read(attributes(h5)["dream_branch"]) == "boltzmann-eqterm"
        @test startswith(read(attributes(h5)["dream_commit"]), "93d27e1")
    end

    @testset "vs DREAM: $variant" for variant in ("pmax5", "pmax2p2")
        h5open(DREAM_KNOCKON_DATA) do h5
            grp = h5[variant]
            mom = PXiGrid(read(grp["grid/p1_f"]), read(grp["grid/p2_f"]))
            g = cylindrical_grid(mom, RadialGrid(read(grp["grid/r_f"])))
            # DREAM's cylindrical Jacobian is 4π² times Runes2P's; only 𝒱'_P/𝒱'_K enters C,
            # so the factor cancels and C is compared without rescaling
            @test all(r -> isapprox(r, 4π^2; rtol=1e-14), read(grp["grid/Vprime"]) ./ g.Vp)
            p_cut = read(grp["p_cut"])
            f = read(grp["f_hot"])          # (np, nξ, nr, nt + 1)
            C_dream = read(grp["C_boltz"])  # (np, nξ, nr, nt); C_boltz[n] from f_hot[n]
            n_tot = read(grp["n_tot"])      # (nr, nt + 1), constant here

            for method in (:loop, :blas)
                op = KnockOnOperator(g, g, p_cut; method, dream=true, primary_points=1, n_tab=80)
                for n in axes(C_dream, 4)
                    C = zeros(size(g))
                    apply!(C, op, f[:, :, :, n], n_tot[:, n])
                    @test max_rel_diff(C_dream[:, :, :, n], C) < 1e-13
                end
            end

            # Runes2P default: ξ* over the receiving part of the cell, full contributing range
            fn = f[:, :, :, 1]
            C_dream1 = C_dream[:, :, :, 1]
            C_def = zeros(size(g))
            apply!(C_def, KnockOnOperator(g, g, p_cut), fn, n_tot[:, 1])
            # same production (columns normalized in both); the pitch placement differs,
            # strongly where DREAM clamps pairs (p_max < 3)
            @test density(g, C_def) ≈ density(g, C_dream1) rtol = 1e-12
            diff_def = max_rel_diff(C_dream1, C_def; floor=1e-6)
            if variant == "pmax2p2"
                @test diff_def > 0.1
            else
                @test diff_def < 0.1
            end
        end
    end

    knock = PXiGrid(biuniform_p_faces(24, 8, 0.0, 0.8, 4.0), uniform_xi_faces(10))
    prim = PXiGrid(uniform_p_faces(16, 0.0, 4.0), -cos.(range(0, π; length=9)))
    rad = RadialGrid(uniform_r_faces(2, 0.5))
    gK, gP = cylindrical_grid(knock, rad), cylindrical_grid(prim, rad)
    fP = [exp(-p / 0.7) * (1.2 + ξ) * (1 + r) for p in prim.p, ξ in prim.xi, r in rad.r]
    n_tot = [2e19, 5e19]

    @testset "loop and BLAS forms vs per-pair interpolated matrices" begin
        for kw in ((dream=true,), (), (primary_points=3, n_tab=20), (n_tab=2,))
            ref = knockon_reference(KnockOnOperator(gK, gP, 0.1; kw...), fP, n_tot)
            for method in (:loop, :blas)
                C = zeros(size(gK))
                apply!(C, KnockOnOperator(gK, gP, 0.1; method, kw...), fP, n_tot)
                # summation order differs; measured differences are below 1e-14
                @test max_rel_diff(ref, C) < 1e-13
            end
        end
        # a knock-on grid different from the primary grid in both directions
        gK2 = cylindrical_grid(PXiGrid(uniform_p_faces(7, 0.0, 4.0), uniform_xi_faces(3)), rad)
        for (a, b) in ((gK2, gP), (gP, gK2))
            fb = ones(size(b))
            ref = knockon_reference(KnockOnOperator(a, b, 0.1), fb, n_tot)
            C = zeros(size(a))
            apply!(C, KnockOnOperator(a, b, 0.1; method=:blas), fb, n_tot)
            @test max_rel_diff(ref, C) < 1e-13
        end
        @test_throws ArgumentError KnockOnOperator(gK, gP, 0.1; method=:dense)
    end

    @testset "production, linearity, allocations" begin
        op = KnockOnOperator(gK, gP, 0.1)
        C = zeros(size(gK))
        apply!(C, op, fP, n_tot)
        # production onto the grid: n_tot Σ_kl (𝒱' Δp Δξ f / V') total[k]
        for ir in 1:2
            prod = n_tot[ir] * sum(gP.Vp[k, l, ir] * prim.dp[k] * prim.dxi[l] * fP[k, l, ir] *
                                   energy_kernel(op).total[k] for k in 1:n_p(prim), l in 1:n_xi(prim)) /
                   rad.VpVol[ir]
            @test density(gK, C)[ir] ≈ prod rtol = 1e-13
        end
        # linear in f and n_tot; α scales and accumulates
        C2 = copy(C)
        apply!(C2, op, 2 .* fP, n_tot; α=-0.5)
        @test maximum(abs, C2) < 1e-13 * maximum(abs, C)
        C3 = zeros(size(gK))
        apply!(C3, op, fP, 3 .* n_tot)
        @test C3 ≈ 3 .* C rtol = 1e-14
        for method in (:loop, :blas)
            opm = KnockOnOperator(gK, gP, 0.1; method)
            apply!(C, opm, fP, n_tot)
            apply!(C, opm, fP, n_tot)
            @test (@allocated apply!(C, opm, fP, n_tot)) == 0
        end
        # large cutoffs (knock-on support shrinking to a few cells near ξ* → 1) build
        # without empty columns; this is the regime a time-dependent p_c sweeps
        for pc in (0.5, 1.0, 1.5), pts in (1, 4)
            opc = KnockOnOperator(gK, gP, pc; primary_points=pts)
            Cc = zeros(size(gK))
            apply!(Cc, opc, fP, n_tot)
            @test all(isfinite, Cc)
        end
        # cutoff above the knock-on grid: no pair produces knock-ons on the grid
        for method in (:loop, :blas)
            op0 = KnockOnOperator(gK, gP, 10.0; method)
            @test angle_kernel(op0).empty
            C0 = zeros(size(gK))
            apply!(C0, op0, fP, n_tot)
            @test all(iszero, C0)
        end
        @test_throws DimensionMismatch apply!(zeros(size(gP)), op, fP, n_tot)
        @test_throws ArgumentError KnockOnOperator(gK, cylindrical_grid(prim, RadialGrid(uniform_r_faces(3, 0.5))), 0.1)
    end
end
