# The initial state of a run: a Maxwellian from profiles of the density and the temperature,
# whose density on the grid is the profile, whose energy is the mean kinetic energy by
# quadrature, and whose values at the cell centres follow DREAM's shape, on a grid shared by
# the radii and on grids per radius. These helpers build a state without a system;
# test_initial_values.jl tests the initial values that the rows of a system give.
using QuadGK: quadgk

@testset "initial state" begin
    @testset "a Maxwellian from profiles" begin
        temperatures, densities = [100.0, 400.0], [1e19, 4e18]
        theta_max = maximum(temperatures) / Constants.ELECTRON_REST_ENERGY
        p_max = 12 * sqrt(2 * theta_max)
        rad = RadialGrid(uniform_r_faces(2, 0.5))
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(400, 0.0, p_max), uniform_xi_faces(4)), rad)
        layout = UnknownLayout(rad,
            [RadialUnknown(:cold_electron_temperature; scale = 100.0),
                KineticUnknown(:f, grid; scale = 1e19), RadialUnknown(:energy; scale = 1e3)])
        u = fill(-1.0, length(layout))
        @test fill_maxwellian!(u, layout, :f, densities, temperatures) === u
        # the other unknowns keep their values
        @test all(==(-1.0), unknown_values(u, layout, 1))
        @test all(==(-1.0), unknown_values(u, layout, 3))
        f = kinetic_array(u, layout, 2)
        @test density(grid, f) ≈ densities rtol = 1e-14
        # DREAM's shape at the cell centres: exp(-(γ − 1)/Θ), rescaled to the density on the grid
        for ir in 1:2
            theta = temperatures[ir] / Constants.ELECTRON_REST_ENERGY
            p = momentum_grid(grid).p_centers
            shape = exp.(-(sqrt.(1 .+ p .^ 2) .- 1) ./ theta)
            @test f[:, 1, ir] ./ f[1, 1, ir] ≈ shape ./ shape[1] rtol = 1e-12
            @test all(j -> f[:, j, ir] == f[:, 1, ir], 2:4)
        end
        # the energy: the mean kinetic energy by quadrature, 3/2 n T well below mₑc²
        out = zeros(length(u))
        Runes2P.rhs!(out, EnergyMomentTerm(:f, grid), u, layout, 3, 0.0)
        e = Constants.ELEMENTARY_CHARGE
        for ir in 1:2
            theta = temperatures[ir] / Constants.ELECTRON_REST_ENERGY
            weight(p) = p^2 * exp(-p^2 / (1 + sqrt(1 + p^2)) / theta)
            mean = quadgk(p -> weight(p) * p^2 / (1 + sqrt(1 + p^2)), 0, p_max)[1] /
                   quadgk(weight, 0, p_max)[1]
            expected = densities[ir] * mean * Constants.ELECTRON_REST_ENERGY * e
            @test unknown_values(out, layout, 3)[ir] ≈ expected rtol = 1e-4
            @test expected ≈ 1.5 * densities[ir] * temperatures[ir] * e rtol = 2e-3
        end
        # the profiles as a number and as a callable of r on the cell centres
        fill_maxwellian!(u, layout, :f, 2e19, r -> 100.0 + 600.0 * r)
        @test density(grid, kinetic_array(u, layout, 2)) ≈ [2e19, 2e19] rtol = 1e-14
        expected = similar(u)
        fill_maxwellian!(
            expected, layout, :f, [2e19, 2e19], 100.0 .+ 600.0 .* rad.r_centers)
        @test kinetic_array(u, layout, 2) == kinetic_array(expected, layout, 2)
        # the non-relativistic shape of the particle source
        fill_maxwellian!(u, layout, :f, densities, temperatures; shape = :maxwellian)
        f = kinetic_array(u, layout, 2)
        @test density(grid, f) ≈ densities rtol = 1e-14
        theta = temperatures[1] / Constants.ELECTRON_REST_ENERGY
        p = momentum_grid(grid).p_centers
        @test f[:, 1, 1] ./ f[1, 1, 1] ≈ exp.(-(p .^ 2 .- p[1]^2) ./ (2theta)) rtol = 1e-12
        # zero density at a radius
        fill_maxwellian!(u, layout, :f, [1e19, 0.0], temperatures)
        @test all(iszero, kinetic_array(u, layout, 2)[:, :, 2])
    end

    @testset "grids per radius" begin
        rad = RadialGrid(uniform_r_faces(2, 0.5))
        grid = cylindrical_grid(
            [MomentumGrid(uniform_p_faces(120, 0.0, 0.2), uniform_xi_faces(4)),
                MomentumGrid(uniform_p_faces(80, 0.0, 0.3), uniform_xi_faces(6))], rad)
        layout = UnknownLayout(rad, [KineticUnknown(:f, grid; scale = 1e19)])
        u = zeros(length(layout))
        fill_maxwellian!(u, layout, :f, [1e19, 3e19], [200.0, 500.0])
        for (ir, n) in enumerate([1e19, 3e19])
            m = momentum_grid(grid, ir)
            V = phase_space_jacobian(grid, ir)
            f = kinetic_view(u, layout, 1, ir)
            @test size(f) == (momentum_cell_count(m), pitch_cell_count(m))
            moment = sum(V[i, j] * m.p_widths[i] * m.xi_widths[j] * f[i, j]
            for i in axes(f, 1), j in axes(f, 2)) / rad.spatial_jacobian[ir]
            @test moment ≈ n rtol = 1e-14
        end
    end

    @testset "refused input" begin
        rad = RadialGrid(uniform_r_faces(2, 0.5))
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(40, 0.0, 0.2), uniform_xi_faces(2)), rad)
        layout = UnknownLayout(rad,
            [KineticUnknown(:f, grid; scale = 1e19), RadialUnknown(:n; scale = 1e19)])
        u = zeros(length(layout))
        @test_throws "not a kinetic unknown" fill_maxwellian!(u, layout, :n, 1e19, 100.0)
        @test_throws "shape must be" fill_maxwellian!(
            u, layout, :f, 1e19, 100.0; shape = :lowest_cell)
        # a refused value at the second radius leaves the first one as it was
        @test_throws "finite and not negative" fill_maxwellian!(
            u, layout, :f, [1e19, -1.0], 100.0)
        @test all(iszero, u)
        @test_throws "finite and not negative" fill_maxwellian!(u, layout, :f, Inf, 100.0)
        @test_throws "finite and positive" fill_maxwellian!(
            u, layout, :f, 1e19, [100.0, 0.0])
        @test_throws "finite and positive" fill_maxwellian!(u, layout, :f, 1e19, Inf)
        @test all(iszero, u)
        # a temperature far below the first momentum cell leaves no density on the grid
        @test_throws DomainError fill_maxwellian!(u, layout, :f, 1e19, 1e-6)
        @test_throws DimensionMismatch fill_maxwellian!(u, layout, :f, [1e19], 100.0)
    end
end
