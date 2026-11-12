# The kinetic energy moment of a distribution: the energy of a resolved
# relativistic Maxwellian against the mean kinetic energy by quadrature, the split into the
# regions above and below a threshold that follows the temperature, and the Jacobian with
# the column of the temperature, also on a kept matrix, the entries of the Jacobian and the
# residual against the products written out in their order, and the allocation of the
# Jacobian.
using QuadGK: quadgk
using SparseArrays: sparse

@testset "energy moment" begin
    temperatures, densities = [100.0, 400.0], [1e19, 4e18]
    theta_max = maximum(temperatures) / Constants.ELECTRON_REST_ENERGY
    p_max = 12 * sqrt(2 * theta_max)
    grid = cylindrical_grid(
        MomentumGrid(uniform_p_faces(400, 0.0, p_max), uniform_xi_faces(4)),
        RadialGrid(uniform_r_faces(2, 0.5)))
    np, nxi, nr = size(grid)
    f = zeros(np, nxi, nr)
    for ir in 1:nr, (i, p) in enumerate(momentum_grid(grid).p_centers)

        f[i, :, ir] .= exp(-p^2 / (1 + sqrt(1 + p^2)) /
                           (temperatures[ir] / Constants.ELECTRON_REST_ENERGY))
    end
    f .*= reshape(densities ./ density(grid, f), 1, 1, :)
    layout = UnknownLayout(grid.radial_grid,
        [KineticUnknown(:f, grid; scale = 1e19),
            RadialUnknown(:cold_electron_temperature; scale = 100.0),
            RadialUnknown(:energy; scale = 1e3)])
    u = zeros(length(layout))
    kinetic_array(u, layout, 1) .= f
    unknown_values(u, layout, 2) .= temperatures
    moment(term) = (out = zeros(length(u)); Runes2P.rhs!(out, term, u, layout, 3, 0.0);
        unknown_values(out, layout, 3))

    @testset "a relativistic Maxwellian" begin
        e = Constants.ELEMENTARY_CHARGE
        for ir in 1:nr
            theta = temperatures[ir] / Constants.ELECTRON_REST_ENERGY
            weight(p) = p^2 * exp(-p^2 / (1 + sqrt(1 + p^2)) / theta)
            mean = quadgk(p -> weight(p) * p^2 / (1 + sqrt(1 + p^2)), 0, p_max)[1] /
                   quadgk(weight, 0, p_max)[1]
            expected = densities[ir] * mean * Constants.ELECTRON_REST_ENERGY * e
            @test moment(EnergyMomentTerm(:f, grid))[ir] ≈ expected rtol = 1e-4
            # 3/2 n T at a temperature well below mₑc²
            @test expected ≈ 1.5 * densities[ir] * temperatures[ir] * e rtol = 2e-3
        end
    end

    @testset "regions" begin
        above = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 2))
        below = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 2); side = :below)
        whole = moment(EnergyMomentTerm(:f, grid))
        hot = moment(EnergyMomentTerm(:f, grid; region = above))
        @test all(0 .< hot .< whole)
        @test hot .+ moment(EnergyMomentTerm(:f, grid; region = below)) ≈ whole rtol = 1e-13
        @test moment(EnergyMomentTerm(:f, grid; coefficient = -2.0)) ≈ -2 .* whole rtol = 1e-14
        @test Runes2P._temperature_source(EnergyMomentTerm(:f, grid; region = above)) ==
              (:unknown, :cold_electron_temperature)
        @test Runes2P._temperature_source(EnergyMomentTerm(:f, grid)) === nothing
        @test !is_linear(EnergyMomentTerm(:f, grid; region = above)) &&
              is_linear(EnergyMomentTerm(:f, grid))
        record = describe(EnergyMomentTerm(:f, grid; region = above), Runes2P.DescriptionContext())
        @test record["kinetic"] == "f"
    end

    @testset "Jacobian" begin
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 2))
        system = EquationSystem(layout,
            [Equation(:f, TimeDerivative(), LinearTerm(:f; coefficient = -1.0)),
                Equation(:cold_electron_temperature, Algebraic(),
                    PrescribedValueTerm(temperatures),
                    LinearTerm(:cold_electron_temperature; coefficient = -1)),
                Equation(:energy, Algebraic(), EnergyMomentTerm(:f, grid; region),
                    LinearTerm(:energy; coefficient = -1))];
            initial_state = u)
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        columns = [first(Runes2P.unknown_range(layout, 2, ir)) for ir in 1:nr]
        append!(columns, [Runes2P._kinetic_index(layout, 1, ir, 30, 2) for ir in 1:nr])
        for column in columns
            h = 1e-6 * max(abs(u[column]), 1.0)
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            F_up, F_down = zeros(length(u)), zeros(length(u))
            rhs!(F_up, up, system, 0.0)
            rhs!(F_down, down, system, 0.0)
            @test Vector(J[:, column])≈(F_up .- F_down) ./ (2h) rtol=1e-6 atol=1e-12*maximum(abs, F_up)
        end
        # The term keeps the positions and weights of its row in J: its block on a kept
        # matrix is the same bit for bit as that of a new term, at the state and at a state
        # whose threshold lies in other cells, and the same again at the state after the
        # other state and after a new matrix of the pattern.
        term = EnergyMomentTerm(:f, grid; coefficient = 0.3, region)
        function block(J, term, u)
            fill!(J.nzval, 0.0)
            Runes2P.add_jacobian!(J, term, u, layout, 3, 0.0)
            return copy(J.nzval)
        end
        new_block(u) = block(jacobian_prototype(system),
            EnergyMomentTerm(:f, grid; coefficient = 0.3, region), u)
        other = copy(u)
        unknown_values(other, layout, 2) .*= 1.5
        kinetic_array(other, layout, 1) .*= 1.1
        kept = jacobian_prototype(system)
        first_block = block(kept, term, u)
        @test isequal(new_block(u), first_block)
        other_block = block(kept, term, other)
        @test !isequal(other_block, first_block)
        @test isequal(new_block(other), other_block)
        @test isequal(block(kept, term, u), first_block)
        @test isequal(block(jacobian_prototype(system), term, other), other_block)
        @test isequal(block(kept, term, u), first_block)
        # The bytes of the Jacobian with the positions and weights the term keeps, after a
        # first call: none without a region, and with a region that follows the temperature
        # the limits of the region at the radii, below a margin of 1e3 bytes (0 and 1.6e2 at
        # -O2).
        whole = EnergyMomentTerm(:f, grid; coefficient = 0.3)
        @test call_allocation(Runes2P.add_jacobian!, kept, whole, u, layout, 3, 0.0) == 0 skip = !OPTIMIZED
        @test call_allocation(Runes2P.add_jacobian!, kept, term, u, layout, 3, 0.0) < 1e3 skip = !OPTIMIZED
    end

    @testset "order of the products" begin
        # The entry of the cell (i, j) of the radius ir is (c * envelope) * w with the
        # weight w = E * V * Δp * Δξ / V′ evaluated from the left: E = mₑc²(γ − 1) at the
        # centre of the cell, V the phase-space Jacobian, Δp and Δξ the widths of the cell
        # and V′ the spatial Jacobian. The residual is c times the sum of
        # (envelope * w) * f, i inside j. Both are compared with ==, and the test sees a
        # change of the order: the products from the left with c * envelope * E first, and
        # c * (envelope * w), differ in 1480 and 0 of the 3200 entries without a region, and
        # in 1312 and 12 with a region whose threshold follows the temperature, where only
        # the cells with an envelope between 0 and 1 can differ in the second. The residual
        # with c inside the sum differs at 2 and 1 of the 2 radii. The column of the
        # temperature is checked against central differences above.
        c = 0.3
        rest_energy = Constants.ELECTRON_REST_ENERGY * Constants.ELEMENTARY_CHARGE
        above = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor = 2))
        n = length(u)
        for region in (nothing, above)
            term = EnergyMomentTerm(:f, grid; coefficient = c, region)
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, 3)
            J = sparse(first.(entries), last.(entries), zeros(length(entries)), n, n)
            Runes2P.add_jacobian!(J, term, u, layout, 3, 0.0)
            out = zeros(n)
            Runes2P.rhs!(out, term, u, layout, 3, 0.0)
            # the entries of J and their values in the order of the term and in two others
            actual, expected = Float64[], Float64[]
            coefficient_first, envelope_last = Float64[], Float64[]
            # the residual per radius, its value in the order of the term and with c inside
            residual, expected_residual = Float64[], Float64[]
            residual_reordered = Float64[]
            for ir in 1:nr
                m = momentum_grid(grid, ir)
                V = phase_space_jacobian(grid, ir)
                spatial_jacobian = grid.radial_grid.spatial_jacobian[ir]
                envelope = if region === nothing
                    ones(np)
                else
                    p0 = 2 * sqrt(2 * temperatures[ir] / Constants.ELECTRON_REST_ENERGY)
                    momentum_envelope(region, m, p0)
                end
                region === nothing || @test any(e -> 0 < e < 1, envelope)
                f = kinetic_view(u, layout, 1, ir)
                r = first(unknown_range(layout, 3, ir))
                W, W_reordered = 0.0, 0.0
                for j in 1:nxi, i in 1:np

                    p = m.p_centers[i]
                    E = rest_energy * p^2 / (1 + sqrt(1 + p^2))
                    w = E * V[i, j] * m.p_widths[i] * m.xi_widths[j] / spatial_jacobian
                    push!(actual, J[r, Runes2P._kinetic_index(layout, 1, ir, i, j)])
                    push!(expected, (c * envelope[i]) * w)
                    push!(coefficient_first,
                        c * envelope[i] * E * V[i, j] * m.p_widths[i] * m.xi_widths[j] /
                        spatial_jacobian)
                    push!(envelope_last, c * (envelope[i] * w))
                    W += (envelope[i] * w) * f[i, j]
                    W_reordered += (c * envelope[i] * w) * f[i, j]
                end
                push!(residual, out[r])
                push!(expected_residual, c * W)
                push!(residual_reordered, W_reordered)
            end
            @test actual == expected
            @test residual == expected_residual
            @test any(expected .!= coefficient_first)
            region === nothing || @test any(expected .!= envelope_last)
            @test any(expected_residual .!= residual_reordered)
        end
    end
end
