# ξ-weighted moments: the pitch weights of the piecewise-linear
# reconstruction against the cell-centre rule, the current density built on them and the
# CurrentMomentTerm of the equation system.
using ForwardDiff: ForwardDiff
using Random: MersenneTwister

@testset "pitch-weighted moments" begin
    random_faces = sort(unique(vcat(-1.0, 1.0, 2 .* rand(MersenneTwister(4), 12) .- 1)))
    @testset "weights are exact for f linear in ξ on any grid" begin
        for (label, faces) in (("uniform 2", uniform_xi_faces(2)), ("uniform 3", uniform_xi_faces(3)),
                               ("uniform 8", uniform_xi_faces(8)),
                               ("production biuniform θ", biuniform_theta_xi_faces(15, 10, 0.5)),
                               ("random", random_faces))
            m = MomentumGrid(uniform_p_faces(2, 0.0, 1.0), faces)
            ξ = m.xi_centers
            w_linear = pitch_moment_weights(m; pitch_rule=:linear)
            w_center = pitch_moment_weights(m; pitch_rule=:cell_center)
            @test w_center == ξ .* m.xi_widths
            # ∫ ξ dξ = 0 (the reconstruction keeps the density of every cell)
            @test abs(sum(w_linear)) < 1e-14
            @test abs(sum(w_center)) < 1e-14
            # ∫ ξ² dξ = 2/3: exact with the reconstruction on every grid ...
            @test sum(w_linear .* ξ) ≈ 2 / 3 atol = 1e-14
            # ... and short by Δξ²/4 with the cell-centre rule on a uniform grid
            if startswith(label, "uniform")
                nxi = pitch_cell_count(m)
                @test sum(w_center .* ξ) ≈ 2 / 3 * (1 - (2 / nxi)^2 / 4) atol = 1e-14
            else
                @test sum(w_center .* ξ) < 2 / 3 - 1e-3
            end
        end
        # one cell: no slope, both rules give the zero moment of the full range
        m1 = MomentumGrid(uniform_p_faces(2, 0.0, 1.0), uniform_xi_faces(1))
        @test pitch_moment_weights(m1; pitch_rule=:linear) == [0.0]
        @test pitch_moment_weights(m1; pitch_rule=:cell_center) == [0.0]
        @test_throws ArgumentError pitch_moment_weights(m1; pitch_rule=:trapezoid)
        @test pitch_moment_weights(m1) == pitch_moment_weights(m1; pitch_rule=:linear)
    end

    @testset "second order for a curved f, ahead of the cell-centre rule" begin
        # f = e^{κξ}: ∫ ξ f dξ = [e^{κξ} (κξ − 1) / κ²], ∫ f dξ = [e^{κξ} / κ]. The midpoint
        # rule's leading error is (Δξ²/24) ∫ (ξ f)'' = (Δξ²/24) (2 ∫ f' + ∫ ξ f''); the
        # reconstruction removes the first part (the error for f linear in ξ) and keeps the
        # second, so the ratio of the two errors tends to κ² I₁ / (2κ I₀ + κ² I₁) with
        # I₀ = ∫ f and I₁ = ∫ ξ f (0.135 at κ = 1).
        κ = 1.0
        I0 = (exp(κ) - exp(-κ)) / κ
        I1 = (exp(κ) * (κ - 1) - exp(-κ) * (-κ - 1)) / κ^2
        errors = Dict(rule => Float64[] for rule in (:linear, :cell_center))
        for nxi in (8, 16, 32, 64)
            m = MomentumGrid(uniform_p_faces(2, 0.0, 1.0), uniform_xi_faces(nxi))
            f = exp.(κ .* m.xi_centers)
            for rule in (:linear, :cell_center)
                push!(errors[rule], abs(sum(pitch_moment_weights(m; pitch_rule=rule) .* f) - I1) / I1)
            end
        end
        for rule in (:linear, :cell_center)
            ratios = errors[rule][1:end-1] ./ errors[rule][2:end]
            @test all(3.8 .< ratios .< 4.2)
        end
        @test all(errors[:linear] .< errors[:cell_center])
        @test errors[:linear][end] / errors[:cell_center][end] ≈ κ^2 * I1 / (2κ * I0 + κ^2 * I1) rtol = 0.02
    end

    @testset "current density of a Spitzer-like distribution" begin
        rad = RadialGrid(uniform_r_faces(2, 0.5))
        mom = MomentumGrid(biuniform_p_faces(50, 30, 0.0, 0.05, 0.6), biuniform_theta_xi_faces(15, 10, 0.5))
        g = cylindrical_grid(mom, rad)
        θ = 0.02
        a = [0.3, -0.2]
        f = [exp(-(sqrt(1 + p^2) - 1) / θ) * (1 + a[ir] * ξ) for p in mom.p_centers, ξ in mom.xi_centers, ir in 1:2]
        for ir in 1:2
            # the p sum is common to both rules, the ξ factor is a ∫ ξ² dξ
            p_sum = sum(phase_space_jacobian(g, ir)[i, 1] * mom.p_widths[i] * mom.p_centers[i] /
                        sqrt(1 + mom.p_centers[i]^2) * exp(-(sqrt(1 + mom.p_centers[i]^2) - 1) / θ)
                        for i in 1:momentum_cell_count(mom)) / rad.spatial_jacobian[ir]
            scale = Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT * p_sum * a[ir]
            @test current_density(g, f)[ir] ≈ scale * 2 / 3 rtol = 1e-13
            @test current_density(g, f; pitch_rule=:cell_center)[ir] ≈
                  scale * sum(mom.xi_centers .^ 2 .* mom.xi_widths) rtol = 1e-13
            # DREAM's sign: electrons skewed towards ξ = 1 carry a positive current
            @test sign(current_density(g, f)[ir]) == sign(a[ir])
        end
        @test_throws DimensionMismatch current_density(g, f[:, :, 1:1])
        @test_throws ArgumentError current_density(g, f; pitch_rule=:trapezoid)
    end

    @testset "CurrentMomentTerm" begin
        rad = RadialGrid(uniform_r_faces(2, 0.5))
        # different pitch grids at the two radii
        grids = [MomentumGrid(uniform_p_faces(6, 0.0, 0.6), uniform_xi_faces(4)),
                 MomentumGrid(uniform_p_faces(6, 0.0, 0.6), biuniform_theta_xi_faces(7, 4, 0.6))]
        ragged = cylindrical_grid(grids, rad)
        @test !has_shared_momentum_grid(ragged)
        for rule in (:linear, :cell_center)
            layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, ragged; scale=1e16),
                                         RadialUnknown(:hot_electron_current; scale=1e3)])
            term = CurrentMomentTerm(:hot_tail_distribution, ragged; pitch_rule=rule)
            @test is_linear(term)
            system = EquationSystem(layout, [
                Equation(:hot_tail_distribution, TimeDerivative(),
                         LinearTerm(:hot_tail_distribution; coefficient=-1.0)),
                Equation(:hot_electron_current, Algebraic(), term,
                         LinearTerm(:hot_electron_current; coefficient=-1))])
            f_id, j_id = unknown_id(layout, :hot_tail_distribution), unknown_id(layout, :hot_electron_current)
            u = zeros(length(layout))
            for ir in 1:2
                m = momentum_grid(ragged, ir)
                kinetic_view(u, layout, f_id, ir) .= [1e16 * exp(-p / 0.2) * (1 + 0.4ξ + 0.1ξ^2)
                                                      for p in m.p_centers, ξ in m.xi_centers]
                radial_view(u, layout, j_id, ir) .= 1e3 * ir
            end
            du = zeros(length(layout))
            rhs!(du, u, system, 0.0)
            for ir in 1:2
                # the same moment on a single-radius grid with that radius' momentum grid
                single = cylindrical_grid(grids[ir], RadialGrid(rad.r_faces[ir:ir+1]))
                f = reshape(kinetic_view(u, layout, f_id, ir), momentum_cell_count(grids[ir]),
                            pitch_cell_count(grids[ir]), 1)
                expected = current_density(single, f; pitch_rule=rule)[1]
                @test radial_view(du, layout, j_id, ir)[1] ≈ expected - 1e3 * ir rtol = 1e-13
            end
            # the stored Jacobian of the linear term is exact
            J = jacobian_prototype(system)
            jacobian!(J, system, u, 0.0)
            dense = ForwardDiff.jacobian(v -> rhs!(similar(v), v, system, 0.0), u)
            @test Matrix(J) ≈ dense rtol = 1e-12
            # validation
            wrong = cylindrical_grid(reverse(grids), rad)
            @test_throws DimensionMismatch EquationSystem(layout, [
                Equation(:hot_electron_current, Algebraic(),
                         CurrentMomentTerm(:hot_tail_distribution, wrong),
                         LinearTerm(:hot_electron_current; coefficient=-1)),
                Equation(:hot_tail_distribution, TimeDerivative(),
                         LinearTerm(:hot_tail_distribution; coefficient=-1.0))])
            @test_throws ArgumentError EquationSystem(layout, [
                Equation(:hot_tail_distribution, Algebraic(),
                         CurrentMomentTerm(:hot_tail_distribution, ragged),
                         LinearTerm(:hot_tail_distribution; coefficient=-1)),
                Equation(:hot_electron_current, TimeDerivative(),
                         LinearTerm(:hot_electron_current; coefficient=-1.0))])
        end
    end
end
