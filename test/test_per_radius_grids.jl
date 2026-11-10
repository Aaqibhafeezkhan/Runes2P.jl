# Kinetic grids with a momentum grid per radius: shared detection, accessors, operators
# that need the same cell counts refusing others, the knock-on operator on different
# momentum grids at different radii, and the buffers that the friction term and the moments
# over a region that follows the temperature keep, on grids of two shapes.
using Random: Xoshiro
using SparseArrays: nonzeros, sparse

# an assertion on allocations holds for optimised code alone: it is skipped, and reported
# as broken, in a development run at -O0 or -O1
const OPTIMIZED = Base.JLOptions().opt_level >= 2

@testset "momentum grid per radius" begin
    rad = RadialGrid(uniform_r_faces(3, 0.6))
    m1 = MomentumGrid(biuniform_p_faces(24, 8, 0.0, 0.6, 4.0), uniform_xi_faces(10))
    m2 = MomentumGrid(uniform_p_faces(18, 0.0, 3.5), -cos.(range(0, π; length = 13)))
    m3 = MomentumGrid(uniform_p_faces(30, 0.0, 5.0), uniform_xi_faces(8))

    @testset "shared grids" begin
        single = cylindrical_grid(m1, rad)
        same_object = cylindrical_grid([m1, m1, m1], rad)
        copied = MomentumGrid(copy(m1.p_faces), copy(m1.xi_faces))
        equal_faces = cylindrical_grid([m1, copied, m1], rad)
        for g in (single, same_object, equal_faces)
            @test has_shared_momentum_grid(g)
            @test size(g) == (24, 10, 3)
            @test cell_count(g) == 24 * 10 * 3
            @test momentum_grid(g) === momentum_grid(g, 2)
            for ir in 1:3
                @test phase_space_jacobian(g, ir) == phase_space_jacobian(single, ir)
                @test phase_space_jacobian_p_faces(g, ir) ==
                      phase_space_jacobian_p_faces(single, ir)
                @test phase_space_jacobian_xi_faces(g, ir) ==
                      phase_space_jacobian_xi_faces(single, ir)
            end
            @test phase_space_jacobian_r_faces(g) == phase_space_jacobian_r_faces(single)
        end
        @test_throws DimensionMismatch cylindrical_grid([m1, m1], rad)
    end

    g = cylindrical_grid([m1, m2, m3], rad)

    @testset "accessors on different grids" begin
        @test !has_shared_momentum_grid(g)
        @test [momentum_cell_count(g, ir) for ir in 1:3] == [24, 18, 30]
        @test [pitch_cell_count(g, ir) for ir in 1:3] == [10, 12, 8]
        @test cell_count(g) == 24 * 10 + 18 * 12 + 30 * 8
        @test radial_cell_count(g) == 3
        for (ir, m) in enumerate((m1, m2, m3))
            @test momentum_grid(g, ir) === m
            expected = [2π * p^2 * rad.spatial_jacobian[ir]
                        for p in m.p_centers, _ in m.xi_centers]
            @test phase_space_jacobian(g, ir) ≈ expected rtol = 1e-15
            np, nxi = momentum_cell_count(m), pitch_cell_count(m)
            @test size(phase_space_jacobian_p_faces(g, ir)) == (np + 1, nxi)
            @test size(phase_space_jacobian_xi_faces(g, ir)) == (np, nxi + 1)
            @test length(phase_space_jacobian_over_p2_at_zero(g, ir)) == pitch_cell_count(m)
        end
        @test_throws ArgumentError size(g)
        @test_throws ArgumentError momentum_grid(g)
        @test_throws ArgumentError momentum_cell_count(g)
        @test_throws ArgumentError phase_space_jacobian_r_faces(g)
    end

    @testset "operators that need the same cell counts refuse others" begin
        # the pitch faces may differ between radii (test_per_radius_pitch_grids.jl), the cell
        # counts and the p faces may not
        @test_throws ArgumentError AdvectionDiffusion(g)
        plasma = PlasmaState(cold_electron_temperature = fill(100.0, 3),
            ions = [IonSpecies("D", 1, 1, fill(1e20, 3))],
            electric_field = fill(1.0, 3), magnetic_field = fill(5.0, 3))
        @test_throws ArgumentError fokker_planck_operator(g, plasma)
        # the message names the reason, not the accessor that failed
        @test_throws "the same cell counts and the same p faces" AdvectionDiffusion(g)
        @test_throws "the same cell counts and the same p faces" fokker_planck_operator(g, plasma)
    end

    @testset "knock-on operator on different grids" begin
        # each radius gives the result of a shared-grid operator built from that radius's
        # grids; radii 1 and 3 have the same knock-on and primary grids
        gK = cylindrical_grid([m1, m2, m1], rad)
        gP = cylindrical_grid([m3, m2, m3], rad)
        n_tot = [1e20, 2e20, 3e20]
        cutoffs = [0.1, 0.15, 0.1]
        for method in (:loop, :blas)
            op = KnockOnOperator(gK, gP, cutoffs; method)
            for ir in 1:3
                mK, mP = momentum_grid(gK, ir), momentum_grid(gP, ir)
                f = [exp(-p / 0.8) * (1.2 + ξ) for p in mP.p_centers, ξ in mP.xi_centers]
                df = zeros(momentum_cell_count(mK), pitch_cell_count(mK))
                apply_at_radius!(df, op, f, n_tot[ir], ir; scale = 0.5)
                # reference: shared grids with the momentum grids of radius ir
                reference = KnockOnOperator(
                    cylindrical_grid(mK, rad), cylindrical_grid(mP, rad),
                    cutoffs[ir]; method)
                f3 = zeros(momentum_cell_count(mP), pitch_cell_count(mP), 3)
                f3[:, :, ir] .= f
                df3 = zeros(momentum_cell_count(mK), pitch_cell_count(mK), 3)
                apply!(df3, reference, f3, n_tot; scale = 0.5)
                @test df == df3[:, :, ir]
            end
            # radii 1 and 3 have the same grids and cutoff and share their kernels
            @test energy_kernel(op, 1) === energy_kernel(op, 3)
            @test energy_kernel(op, 1) !== energy_kernel(op, 2)
            @test length(op.cache) == 2
            @test_throws ArgumentError apply!(zeros(24, 10, 3), op, zeros(30, 8, 3), n_tot)
            @test_throws DimensionMismatch apply_at_radius!(
                zeros(3, 3), op, zeros(30, 8), 1e20, 1)
            df1, f1 = zeros(24, 10), ones(30, 8)
            apply_at_radius!(df1, op, f1, 1e20, 1)
            @test (@allocated apply_at_radius!(df1, op, f1, 1e20, 1)) == 0 skip = !OPTIMIZED
        end
    end
end

# the column `column` of the Jacobian of `residual!(out, state)` at `state` by central
# differences
function central_difference_column(residual!, state, column)
    h = 1e-6 * max(abs(state[column]), 1.0)
    plus = copy(state)
    minus = copy(state)
    plus[column] += h
    minus[column] -= h
    upper = residual!(zeros(length(state)), plus)
    lower = residual!(zeros(length(state)), minus)
    return (upper .- lower) ./ (2h)
end

# the largest difference between the entries of J at `rows` and `columns` and the central
# differences of `residual!` at `state`, relative to the larger of the two at each entry
function worst_central_difference(J, residual!, state, rows, columns)
    worst = 0.0
    for column in columns
        difference = central_difference_column(residual!, state, column)
        for r in rows
            scale = max(abs(difference[r]), abs(J[r, column]))
            if scale > 0
                worst = max(worst, abs(J[r, column] - difference[r]) / scale)
            end
        end
    end
    return worst
end

@testset "kept buffers on grids of two shapes" begin
    # two radii with different momentum and pitch cell counts: the friction term keeps a
    # scratch per shape of the cells, and the moments over a region that follows the
    # temperature keep positions and weights per radius; the kept terms against terms built
    # afresh, bit for bit, and their Jacobians against central differences
    radial = RadialGrid([0.0, 0.25, 0.5])
    nr = 2
    m1 = MomentumGrid(collect(range(0, 0.8; length = 21)), collect(range(-1, 1; length = 5)))
    m2 = MomentumGrid(collect(range(0, 0.8; length = 15)), collect(range(-1, 1; length = 7)))
    ragged = cylindrical_grid([m1, m2], radial)
    temperature = [300.0, 800.0]
    density = [3e19, 8e19]
    argon = IonSpecies("Ar", 18, zeros(19, nr))
    argon.charge_state_densities[2, :] .= [1e19, 2e19]
    deuterium = IonSpecies("D", 1, [zeros(1, nr); fill(5e19, 1, nr)])
    prescribed = PlasmaState(; cold_electron_temperature = temperature,
        cold_electron_density = density, ions = [deuterium, argon],
        electric_field = zeros(nr), magnetic_field = fill(2.5, nr))
    composition = IonComposition(["D" => 1, "Ar" => 18])
    unknowns() = [RadialUnknown(:cold_electron_temperature; scale = 10.0),
        RadialUnknown(:cold_electron_density; scale = 1e19),
        ion_densities_unknown(composition), RadialUnknown(:heating; scale = 1e6),
        KineticUnknown(:f, ragged; scale = 1e17)]
    layout = UnknownLayout(radial, unknowns())
    # a second layout, equal in value
    same_layout = UnknownLayout(radial, unknowns())
    n = length(layout)
    T_id = unknown_id(layout, :cold_electron_temperature)
    n_id = unknown_id(layout, :cold_electron_density)
    ion_id = unknown_id(layout, :ion_densities)
    f_id = unknown_id(layout, :f)
    row = unknown_id(layout, :heating)
    u0 = zeros(n)
    unknown_values(u0, layout, T_id) .= temperature
    unknown_values(u0, layout, n_id) .= density
    for ir in 1:nr
        radial_view(u0, layout, ion_id, ir) .= vcat(
            deuterium.charge_state_densities[:, ir], argon.charge_state_densities[:, ir])
    end
    rng = Xoshiro(7)
    for ir in 1:nr
        m = momentum_grid(ragged, ir)
        kinetic_view(u0, layout, f_id, ir) .= [1e17 * exp(-3p) * (1 + 0.2 * rand(rng))
                                               for p in m.p_centers, _ in m.xi_centers]
    end
    u1 = copy(u0)
    unknown_values(u1, layout, T_id) .*= 1.3
    unknown_values(u1, layout, n_id) .*= 0.9
    rows = [first(unknown_range(layout, row, ir)) for ir in 1:nr]
    # a matrix with the pattern of `entries` and zero values
    pattern(entries) = sparse(first.(entries), last.(entries), 0.0, n, n)

    @testset "the friction term" begin
        plasma = PlasmaFromUnknowns(prescribed; ions = :ion_densities)
        new_friction() = FrictionEnergyTransferTerm(:f, ragged, plasma; coefficient = -2.0)
        term = new_friction()
        Runes2P.validate(term, layout, row)
        @test length(term.products) == 2
        @test term.shapes == [1, 2]
        entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, row)
        J = pattern(entries)
        for state in (u0, u1, u0, u1)
            fresh = new_friction()
            kept_residual = rhs!(zeros(n), term, state, layout, row, 0.0)
            fresh_residual = rhs!(zeros(n), fresh, state, layout, row, 0.0)
            @test isequal(kept_residual, fresh_residual)
            fill!(nonzeros(J), 0.0)
            Runes2P.add_jacobian!(J, term, state, layout, row, 0.0)
            J_fresh = pattern(entries)
            Runes2P.add_jacobian!(J_fresh, fresh, state, layout, row, 0.0)
            @test isequal(nonzeros(J), nonzeros(J_fresh))
        end
        # the columns of the plasma and of f at both radii, at u1
        fill!(nonzeros(J), 0.0)
        Runes2P.add_jacobian!(J, term, u1, layout, row, 0.0)
        columns = vcat(collect(unknown_range(layout, T_id, 1)),
            collect(unknown_range(layout, T_id, 2)), collect(unknown_range(layout, n_id, 2)),
            collect(unknown_range(layout, ion_id, 1))[[2, 4]],
            collect(unknown_range(layout, ion_id, 2))[[2, 4]],
            collect(unknown_range(layout, f_id, 1))[[1, 33, 80]],
            collect(unknown_range(layout, f_id, 2))[[1, 50, 84]])
        function friction_residual!(out, state)
            rhs!(out, new_friction(), state, layout, row, 0.0)
            return out
        end
        worst = worst_central_difference(J, friction_residual!, u1, rows, columns)
        @test worst < 1e-5
    end

    @testset "the moments over a region that follows the temperature ($name)" for (name, new_moment) in (
        ("DensityMomentTerm", region -> DensityMomentTerm(:f, ragged; region)),
        ("CurrentMomentTerm", region -> CurrentMomentTerm(:f, ragged; region)))
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 5))
        term = new_moment(region)
        entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, row)
        J = pattern(entries)
        # a second matrix of the pattern with other arrays, so that the kept positions move
        # between the two, and the layout equal in value
        other_J = pattern(entries)
        for (state, matrix, on_layout) in ((u0, J, layout), (u1, J, layout),
            (u1, other_J, layout), (u0, J, layout), (u1, J, same_layout), (u0, J, layout))
            fill!(nonzeros(matrix), 0.0)
            Runes2P.add_jacobian!(matrix, term, state, on_layout, row, 0.0)
            reference = pattern(entries)
            Runes2P.add_jacobian!(reference, new_moment(region), state, on_layout, row, 0.0)
            @test isequal(nonzeros(matrix), nonzeros(reference))
        end
        fill!(nonzeros(J), 0.0)
        Runes2P.add_jacobian!(J, term, u1, layout, row, 0.0)
        columns = vcat(collect(unknown_range(layout, T_id, 1)),
            collect(unknown_range(layout, T_id, 2)), collect(unknown_range(layout, f_id, 1)),
            collect(unknown_range(layout, f_id, 2)))
        function moment_residual!(out, state)
            rhs!(out, new_moment(region), state, layout, row, 0.0)
            return out
        end
        worst = worst_central_difference(J, moment_residual!, u1, rows, columns)
        @test worst < 1e-4
    end
end
