# Knock-on pairings between the hot-tail and the runaway grid: hot-tail from hot-tail,
# hot-tail from runaway and runaway from runaway primaries. Every knock-on above the
# cutoff is counted once, on one of the two grids.
using Random: Xoshiro

# production rate per volume of a knock-on source: (1/V') Σ 𝒱' Δp Δξ C at radius ir
function source_density(grid, source, ir)
    m = momentum_grid(grid, ir)
    V = phase_space_jacobian(grid, ir)
    return sum(V[i, j] * m.p_widths[i] * m.xi_widths[j] * source[i, j]
    for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)) /
           grid.radial_grid.spatial_jacobian[ir]
end

# the same per momentum cell
function source_per_momentum_cell(grid, source, ir)
    m = momentum_grid(grid, ir)
    V = phase_space_jacobian(grid, ir)
    return [sum(V[i, j] * m.p_widths[i] * m.xi_widths[j] * source[i, j]
            for j in 1:pitch_cell_count(m))
            for i in 1:momentum_cell_count(m)]
end

@testset "knock-on pairings of two grids" begin
    rng = Xoshiro(55)
    rad = RadialGrid(uniform_r_faces(2, 0.5))
    hot_p_faces = biuniform_p_faces(30, 12, 0.0, 0.6, 2.5)
    runaway_p_faces = uniform_p_faces(16, 2.5, 42.5)
    hot = cylindrical_grid(MomentumGrid(hot_p_faces, uniform_xi_faces(10)), rad)
    runaway = runaway_grid(hot, runaway_p_faces[2:end])
    hot_m, runaway_m = momentum_grid(hot), momentum_grid(runaway)
    cutoff = hot_p_faces[5]
    density = [1e20, 3e19]

    @testset "energy kernels" begin
        from_runaway_on_hot = MollerEnergyKernel(hot_m, runaway_m, cutoff)
        from_runaway_on_runaway = MollerEnergyKernel(runaway_m, runaway_m, cutoff)
        for k in 1:momentum_cell_count(runaway_m)
            p1 = runaway_m.p_centers[k]
            on_grids = from_runaway_on_hot.total_rates[k] +
                       from_runaway_on_runaway.total_rates[k]
            above = from_runaway_on_runaway.rates_above_grid[k]
            # all knock-ons above the cutoff: on the hot-tail grid, on the runaway grid, or
            # above the runaway grid
            @test on_grids + above ≈ moller_total_rate(p1, cutoff) rtol = 1e-12
            # what leaves the hot-tail grid is what the runaway grid and the region above it get
            @test from_runaway_on_hot.rates_above_grid[k] ≈
                  from_runaway_on_runaway.total_rates[k] + above rtol = 1e-12
        end
        @test any(>(0), from_runaway_on_hot.total_rates)
        @test any(>(0), from_runaway_on_runaway.total_rates)
        # a knock-on takes at most half of the kinetic energy of the primary: primaries on
        # the hot-tail grid produce nothing on the runaway grid, and the pairing is not built
        @test all(p -> max_knockon_momentum(p) < p, hot_m.p_centers)
        @test max_knockon_momentum(last(hot_m.p_faces)) < first(runaway_m.p_faces)
        from_hot_on_runaway = MollerEnergyKernel(runaway_m, hot_m, cutoff)
        @test all(==(0), from_hot_on_runaway.rates)
        @test all(==(0), from_hot_on_runaway.total_rates)
    end

    hot_from_hot = KnockOnOperator(hot, hot, cutoff)
    hot_from_runaway = KnockOnOperator(hot, runaway, cutoff)
    runaway_from_runaway = KnockOnOperator(runaway, runaway, cutoff)

    @testset "production on the two grids" begin
        f = [1e10 * exp(-p / 10) * (1.5 + xi)
             for p in runaway_m.p_centers, xi in runaway_m.xi_centers]
        for ir in 1:2
            on_hot = apply_at_radius!(
                zeros(size(hot)[1:2]), hot_from_runaway, f, density[ir], ir)
            on_runaway = apply_at_radius!(
                zeros(size(runaway)[1:2]), runaway_from_runaway, f, density[ir], ir)
            @test all(>=(0), on_hot) && all(>=(0), on_runaway)
            content = source_per_momentum_cell(runaway, f, ir) ./ rad.spatial_jacobian[ir]
            expected_hot = density[ir] *
                           sum(content .* energy_kernel(hot_from_runaway, ir).total_rates)
            expected_runaway = density[ir] * sum(content .*
                                   energy_kernel(runaway_from_runaway, ir).total_rates)
            @test source_density(hot, on_hot, ir) ≈ expected_hot rtol = 1e-12
            @test source_density(runaway, on_runaway, ir) ≈ expected_runaway rtol = 1e-12
            all_knockons = density[ir] * sum(content[k] *
                               (moller_total_rate(runaway_m.p_centers[k], cutoff) -
                                energy_kernel(runaway_from_runaway, ir).rates_above_grid[k])
            for k in eachindex(content))
            @test source_density(hot, on_hot, ir) +
                  source_density(runaway, on_runaway, ir) ≈ all_knockons rtol = 1e-12
        end
    end

    @testset "one grid cut in two" begin
        # with one pitch grid on both grids the three pairings are the blocks of the operator
        # of the whole grid
        pitch = uniform_xi_faces(10)
        whole = cylindrical_grid(MomentumGrid(vcat(hot_p_faces, runaway_p_faces[2:end]), pitch), rad)
        above = runaway_grid(hot, runaway_p_faces[2:end], pitch)
        cut = momentum_cell_count(hot_m)
        whole_operator = KnockOnOperator(whole, whole, cutoff)
        pairings = (KnockOnOperator(hot, hot, cutoff), KnockOnOperator(hot, above, cutoff),
            KnockOnOperator(above, above, cutoff))
        rates = energy_kernel(whole_operator).rates
        @test energy_kernel(pairings[1]).rates ≈ rates[1:cut, 1:cut] rtol = 1e-14
        @test energy_kernel(pairings[2]).rates ≈ rates[1:cut, (cut + 1):end] rtol = 1e-14
        @test energy_kernel(pairings[3]).rates ≈ rates[(cut + 1):end, (cut + 1):end] rtol = 1e-14
        @test all(==(0), rates[(cut + 1):end, 1:cut])
        f = [1e10 * (exp(-p / 0.3) + 1e-3 * exp(-p / 10)) * (1.5 + xi)
             for p in momentum_grid(whole).p_centers, xi in momentum_grid(whole).xi_centers]
        f_hot, f_runaway = f[1:cut, :], f[(cut + 1):end, :]
        expected = apply_at_radius!(zeros(size(f)), whole_operator, f, density[1], 1)
        on_hot = apply_at_radius!(zeros(size(f_hot)), pairings[1], f_hot, density[1], 1)
        apply_at_radius!(on_hot, pairings[2], f_runaway, density[1], 1)
        on_runaway = apply_at_radius!(
            zeros(size(f_runaway)), pairings[3], f_runaway, density[1], 1)
        # the production of every momentum cell is the same to round-off
        @test source_per_momentum_cell(hot, on_hot, 1) ≈
              source_per_momentum_cell(whole, expected, 1)[1:cut] rtol = 1e-12
        @test source_per_momentum_cell(above, on_runaway, 1) ≈
              source_per_momentum_cell(whole, expected, 1)[(cut + 1):end] rtol = 1e-12
        # its distribution over the pitch differs by the interpolation in the ξ* tables,
        # which cover the pairs of each operator
        @test on_hot ≈ expected[1:cut, :] rtol = 2e-2
        @test on_runaway ≈ expected[(cut + 1):end, :] rtol = 2e-2
    end

    @testset "terms of the equation system ($treatment)" for treatment in (:implicit, :explicit)
        layout = UnknownLayout(rad,
            [KineticUnknown(:hot_tail_distribution, hot; scale = 1e10),
                KineticUnknown(:runaway_distribution, runaway; scale = 1e10)])
        decay(name) = LinearTerm(name; coefficient = -3.0)
        system = EquationSystem(layout,
            [
                Equation(
                    :hot_tail_distribution, TimeDerivative(), decay(:hot_tail_distribution),
                    KnockOnTerm(:hot_tail_distribution, hot_from_hot; target_density = density, treatment),
                    KnockOnTerm(:runaway_distribution, hot_from_runaway; target_density = density, treatment)),
                Equation(
                    :runaway_distribution, TimeDerivative(), decay(:runaway_distribution),
                    KnockOnTerm(:runaway_distribution, runaway_from_runaway;
                        target_density = density, treatment))])
        hot_id, runaway_id = unknown_id(layout, :hot_tail_distribution),
        unknown_id(layout, :runaway_distribution)
        u = 1e10 .* rand(rng, length(layout))
        du = zeros(length(layout))
        rhs!(du, u, system, 0.0)
        for ir in 1:2
            f_hot, f_runaway = kinetic_view(u, layout, hot_id, ir),
            kinetic_view(u, layout, runaway_id, ir)
            on_hot = -3.0 .* f_hot
            apply_at_radius!(on_hot, hot_from_hot, f_hot, density[ir], ir)
            apply_at_radius!(on_hot, hot_from_runaway, f_runaway, density[ir], ir)
            on_runaway = -3.0 .* f_runaway
            apply_at_radius!(on_runaway, runaway_from_runaway, f_runaway, density[ir], ir)
            @test kinetic_view(du, layout, hot_id, ir) ≈ on_hot rtol = 1e-13
            @test kinetic_view(du, layout, runaway_id, ir) ≈ on_runaway rtol = 1e-13
        end
        if treatment === :implicit
            # the blocks (hot-tail, runaway) and (runaway, runaway) of the Jacobian are
            # matrix-free, and the terms are linear: J v is the right-hand side at v
            J = jacobian_prototype(system)
            jacobian!(J, system, u, 0.0)
            operator = jacobian_operator(J, system, u, 0.0)
            v = 1e10 .* rand(rng, length(layout))
            rhs!(du, v, system, 0.0)
            @test operator * v ≈ du rtol = 1e-12
        end
        # a pairing with the grids exchanged is refused
        @test_throws DimensionMismatch EquationSystem(layout,
            [
                Equation(
                    :hot_tail_distribution, TimeDerivative(), decay(:hot_tail_distribution),
                    KnockOnTerm(:runaway_distribution, runaway_from_runaway;
                        target_density = density, treatment)),
                Equation(:runaway_distribution, TimeDerivative(), decay(:runaway_distribution))])
    end

    @testset "one cutoff for the three pairings" begin
        plasma_at(t) = PlasmaState(
            cold_electron_temperature = fill(2000.0 * exp(-t / 1e-3), 2),
            ions = [IonSpecies("D", 1, 1, [1e20, 3e19])],
            electric_field = [3.0, 1.0], magnetic_field = [5.0, 5.0])
        operators = (
            KnockOnOperator(hot, hot, cutoff), KnockOnOperator(hot, runaway, cutoff),
            KnockOnOperator(runaway, runaway, cutoff))
        update = KnockOnCutoffUpdate(operators[1], plasma_at; linked = [
            operators[2], operators[3]])
        layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, hot)])
        u = zeros(length(layout))
        seen = Set{Vector{Float64}}()
        for t in (0.0, 1e-3, 3e-3, 6e-3)
            Runes2P.apply_update!(update, u, layout, t)
            @test operators[2].p_cutoff == operators[1].p_cutoff
            @test operators[3].p_cutoff == operators[1].p_cutoff
            @test all(in(hot_p_faces), operators[1].p_cutoff)
            push!(seen, copy(operators[1].p_cutoff))
            for ir in 1:2
                @test energy_kernel(operators[2], ir).p_cutoff == operators[1].p_cutoff[ir]
                @test energy_kernel(operators[3], ir).p_cutoff == operators[1].p_cutoff[ir]
            end
        end
        @test length(seen) > 1   # the cutoff followed the temperature
        # a restart restores the cutoff on all of them
        state = Runes2P.update_state(update)
        fresh = (KnockOnOperator(hot, hot, cutoff), KnockOnOperator(hot, runaway, cutoff),
            KnockOnOperator(runaway, runaway, cutoff))
        restored = KnockOnCutoffUpdate(fresh[1], plasma_at; linked = collect(fresh[2:3]))
        Runes2P.restore_update!(restored, state)
        @test all(op -> op.p_cutoff == operators[1].p_cutoff, fresh)
        record = describe(restored, DescriptionContext(; times = [0.0, 1e-3]))
        @test length(record["linked_operators"]) == 2
        @test record["linked_operators"]["operator_2"]["p_cutoff"] == operators[1].p_cutoff
        @test !haskey(
            describe(KnockOnCutoffUpdate(fresh[1], plasma_at), DescriptionContext(;
                times = [0.0])),
            "linked_operators")
        fields = settings_fields(KnockOnCutoffUpdate)
        @test Set(union(fields.described, fields.derived)) ==
              Set(fieldnames(KnockOnCutoffUpdate))
        @test_throws ArgumentError KnockOnCutoffUpdate(fresh[1], plasma_at; linked = [fresh[1]])
        other_radii = cylindrical_grid(hot_m, RadialGrid(uniform_r_faces(3, 0.5)))
        @test_throws DimensionMismatch KnockOnCutoffUpdate(fresh[1], plasma_at;
            linked = [KnockOnOperator(other_radii, other_radii, cutoff)])
    end
end
