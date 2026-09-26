# Knock-on cutoff that follows the plasma: the window k p_Te ≲ p_c < p_crit, the choice
# of a grid face with hysteresis, per-radius kernels with sharing, and the insensitivity
# of production above a reference momentum to p_c.
@testset "knock-on cutoff" begin
    @testset "window" begin
        @test thermal_momentum(Constants.ELECTRON_REST_ENERGY / 2) ≈ 1
        @test critical_momentum(2.0, 1.0) ≈ 1
        @test critical_momentum(1.0, 1.0) == Inf && critical_momentum(0.5, 1.0) == Inf
        # E_c = n e³ lnΛ/(4π ε₀² mₑ c²): about 0.076 V/m at 1e20 m⁻³ and lnΛ = 15
        @test critical_field(1e19, 15.0) ≈
              1e19 * 15 * Constants.ELEMENTARY_CHARGE^3 /
              (4π * Constants.VACUUM_PERMITTIVITY^2 * Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT^2)
        @test 0.07 < critical_field(1e20, 15.0) < 0.08
        # p_crit²/p_Te² → E_D/(2E) for E ≫ E_c, with E_D = E_c mₑc²/T
        T, Ec = 20.0, 1.0
        E = 1e3 * Ec
        ED = Ec * Constants.ELECTRON_REST_ENERGY / T
        @test (critical_momentum(E, Ec) / thermal_momentum(T))^2 ≈ ED / (2E) rtol = 2e-3
        w = cutoff_window(0.01, 0.2; thermal_factor=4)
        @test w.lower ≈ 0.04 && w.upper == 0.2
    end

    @testset "choice on grid faces" begin
        p_f = uniform_p_faces(50, 0.0, 1.0)  # faces every 0.02
        c = choose_cutoff(p_f, 0.05, 0.5)
        @test c.inside_window && c.p_cutoff in p_f && 0.05 <= c.p_cutoff < 0.5
        @test c.p_cutoff ≈ 0.16  # face closest to √(0.05·0.5) ≈ 0.158
        # hysteresis: a current value inside the window is kept
        @test choose_cutoff(p_f, 0.1, 0.5; current_cutoff=0.12) == (p_cutoff=0.12, inside_window=true)
        @test choose_cutoff(p_f, 0.13, 0.5; current_cutoff=0.12).p_cutoff != 0.12
        # a current value that is not a face (another grid's cutoff) is replaced by a face
        c = choose_cutoff(p_f, 0.1, 0.5; current_cutoff=0.13)
        @test c.inside_window && c.p_cutoff in p_f && c.p_cutoff != 0.13
        # no upper bound (E ≤ E_c): smallest face above the lower bound
        @test choose_cutoff(p_f, 0.05, Inf) == (p_cutoff=0.06, inside_window=true)
        # closed window: largest face below p_crit, flagged
        c = choose_cutoff(p_f, 0.3, 0.25)
        @test !c.inside_window && c.p_cutoff ≈ 0.24
        # window narrower than a cell
        c = choose_cutoff(p_f, 0.101, 0.119)
        @test !c.inside_window && c.p_cutoff ≈ 0.1
        # below the first positive face
        @test choose_cutoff(p_f, 0.001, 0.01) == (p_cutoff=0.02, inside_window=false)
    end

    rad = RadialGrid(uniform_r_faces(3, 0.5))
    g = cylindrical_grid(MomentumGrid(biuniform_p_faces(40, 20, 0.0, 0.5, 4.0), uniform_xi_faces(8)), rad)
    f = [exp(-p / 0.9) * (1.3 + ξ) * (2 + r)
         for p in momentum_grid(g).p_centers, ξ in momentum_grid(g).xi_centers, r in rad.r_centers]
    n_tot = [1e19, 2e19, 3e19]
    pa, pb = momentum_grid(g).p_faces[5], momentum_grid(g).p_faces[9]  # 0.1, 0.2

    @testset "per-radius kernels and caching" begin
        op = KnockOnOperator(g, g, [pa, pb, pa])
        @test energy_kernel(op, 1) === energy_kernel(op, 3)
        @test energy_kernel(op, 1) !== energy_kernel(op, 2)
        C = zeros(size(g))
        apply!(C, op, f, n_tot)
        for (ir, pc) in ((1, pa), (2, pb), (3, pa))
            Cr = zeros(size(g))
            apply!(Cr, KnockOnOperator(g, g, pc), f, n_tot)
            @test C[:, :, ir] == Cr[:, :, ir]
        end
        # unchanged or already present values build nothing; new values build once
        @test set_cutoff!(op, [pa, pb, pa]) == 0
        @test set_cutoff!(op, [pb, pb, pb]) == 0
        @test length(op.cache) == 1
        @test set_cutoff!(op, [pa, pb, momentum_grid(g).p_faces[13]]) == 2
        @test op.p_cutoff == [pa, pb, momentum_grid(g).p_faces[13]]
        # the matrix-free form follows the new kernels
        L = knockon_linear_operator(op, n_tot)
        set_cutoff!(op, pb)
        Cb = zeros(size(g))
        apply!(Cb, op, f, n_tot)
        @test L * vec(f) ≈ vec(Cb) rtol = 1e-14
        @test_throws DimensionMismatch set_cutoff!(op, [pa, pb])
        @test_throws ArgumentError set_cutoff!(op, 0.0)
        # an invalid cutoff at a later radius leaves the operator as it was
        before = (copy(op.p_cutoff), copy(op.kernels), length(op.cache))
        @test_throws ArgumentError set_cutoff!(op, [pa, momentum_grid(g).p_faces[13], -1.0])
        @test op.p_cutoff == before[1] && length(op.cache) == before[3]
        @test all(k -> op.kernels[k] === before[2][k], eachindex(op.kernels))
    end

    @testset "production above a reference momentum does not depend on p_c" begin
        # Cells fully above p_c have p_c-independent S and ξ*; only the pitch table range
        # moves, and columns stay normalized. So the knock-ons produced above any
        # reference momentum p_ref ≥ p_c are the same for all p_c ≤ p_ref.
        iref = 13  # p_ref = p_f[13] = 0.3
        above(C) = sum(phase_space_jacobian(g, ir)[i, j] * momentum_grid(g).p_widths[i] *
                       momentum_grid(g).xi_widths[j] * C[i, j, ir]
                       for i in iref:momentum_cell_count(g), j in 1:pitch_cell_count(g), ir in 1:3)
        vals = map((3, 5, 9, 13)) do ic
            C = zeros(size(g))
            apply!(C, KnockOnOperator(g, g, momentum_grid(g).p_faces[ic]), f, n_tot)
            (above(C), density(g, C))
        end
        @test all(v -> isapprox(v[1], vals[1][1]; rtol=1e-13), vals)
        # the total production does depend on p_c (it diverges as p_c → 0)
        @test vals[1][2][1] > 1.5 * vals[end][2][1]
    end

    @testset "cutoff following a disruption" begin
        # T falls from 5 keV to 10 eV (thermal quench) while E/E_c rises to 50 and relaxes
        # (current quench). The cutoff stays inside the window whenever a face fits, and
        # the hysteresis keeps rebuilds to a few.
        ts = range(0, 1; length=400)
        T(t) = 5000 * exp(-t / 0.03) + 10
        EoverEc(t) = 0.3 + 49.7 * (1 - exp(-t / 0.05)) * exp(-t / 0.6)
        gd = cylindrical_grid(MomentumGrid(biuniform_p_faces(120, 60, 0.0, 0.6, 10.0), uniform_xi_faces(6)),
                              RadialGrid(uniform_r_faces(1, 0.5)))
        op = KnockOnOperator(gd, gd, 1.0)
        pc = nothing
        builds, outside = 0, 0
        for t in ts
            w = cutoff_window(thermal_momentum(T(t)), critical_momentum(EoverEc(t), 1.0))
            c = choose_cutoff(momentum_grid(gd).p_faces, w.lower, w.upper; current_cutoff=pc)
            fits = any(p -> w.lower <= p < w.upper, momentum_grid(gd).p_faces)
            @test c.inside_window == fits
            c.inside_window || (outside += 1)
            builds += set_cutoff!(op, c.p_cutoff)
            pc = c.p_cutoff
        end
        @test builds <= 20  # measured 17 over 400 steps
        @test outside < length(ts) ÷ 10
    end
end
