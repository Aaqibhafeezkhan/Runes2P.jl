# The initial state of a run: a Maxwellian from profiles of the density and the temperature,
# whose density on the grid is the profile, whose energy is the mean kinetic energy by
# quadrature, and whose values at the cell centres follow DREAM's shape, on a grid shared by
# the radii and on grids per radius; runaways from their density in DREAM's four shapes, whose
# density on the grid is the profile; and ions in coronal equilibrium at the free-electron
# density they give, the fixed point of DREAM's init_equil. These helpers build a state without
# a system; test_initial_values.jl tests the initial values that the rows of a system give.
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

    @testset "runaways from their density" begin
        rad = RadialGrid(uniform_r_faces(2, 0.5))
        grid = cylindrical_grid(
            MomentumGrid(uniform_p_faces(30, 0.0, 20.0), uniform_xi_faces(6)), rad)
        layout = UnknownLayout(rad,
            [RadialUnknown(:runaway_density; scale = 1e16),
                KineticUnknown(:f_re, grid; scale = 1e16)])
        densities = [1e16, 3e15]
        id = unknown_id(layout, :f_re)
        for shape in RUNAWAY_SHAPES
            u = fill(-1.0, length(layout))
            @test fill_runaway_distribution!(u, layout, :f_re, densities; shape,
                electric_field = [1.5, -0.5]) === u
            # the other unknowns keep their values, and the moment of f is the density
            @test all(==(-1.0), unknown_values(u, layout, 1))
            @test density(grid, kinetic_array(u, layout, id)) ≈ densities rtol = 1e-14
            for ir in 1:2
                f = kinetic_view(u, layout, id, ir)
                # every electron in the first momentum cell
                @test all(iszero, f[2:end, :])
                filled = findall(!iszero, f[1, :])
                expected = if shape === :isotropic
                    1:6
                elseif shape === :xi_negative || (shape === :forward && ir == 2)
                    [1]
                else
                    [6]
                end
                @test filled == collect(expected)
                if shape === :isotropic
                    # uniform in ξ: the same density per pitch width in every pitch cell
                    V = phase_space_jacobian(grid, ir)
                    per_width = [f[1, j] * V[1, j] for j in 1:6]
                    @test all(x -> x ≈ per_width[1], per_width)
                end
            end
        end
        u = fill(-1.0, length(layout))
        @test_throws "needs the electric field" fill_runaway_distribution!(
            u, layout, :f_re, 1e16)
        @test_throws "shape must be" fill_runaway_distribution!(
            u, layout, :f_re, 1e16; shape = :avalanche)
        @test_throws "not a kinetic unknown" fill_runaway_distribution!(
            u, layout, :runaway_density, 1e16; shape = :isotropic)
        @test_throws "finite and not negative" fill_runaway_distribution!(
            u, layout, :f_re, [1e16, -1.0]; shape = :isotropic)
        @test_throws "field must be finite" fill_runaway_distribution!(
            u, layout, :f_re, 1e16; electric_field = [1.0, NaN])
        @test_throws DimensionMismatch fill_runaway_distribution!(
            u, layout, :f_re, [1e16]; shape = :isotropic)
        # the state is left as it was
        @test all(==(-1.0), u)
    end

    @testset "ions in coronal equilibrium" begin
        if !isdir(adas_cache_directory())
            @info "no ADF11 cache at $(adas_cache_directory()): the coronal start is skipped"
            @test_skip isdir(adas_cache_directory())
        else
            composition = IonComposition(["D" => 1, "Ne" => 10])
            rad = RadialGrid(uniform_r_faces(3, 0.42))
            layout = UnknownLayout(rad,
                [
                    ion_densities_unknown(composition), RadialUnknown(:temperature; scale = 100.0)])
            neon = species_components(composition, "Ne")
            deuterium = species_components(composition, "D")
            u = zeros(length(layout))
            totals = [1e19, 2e19, 5e18]
            for ir in 1:3
                states = radial_view(u, layout, unknown_id(layout, :ion_densities), ir)
                states[deuterium] .= [0.0, 5e19]   # fully ionized, kept
                states[first(neon)] = totals[ir]   # neutral, put in equilibrium
            end
            temperatures = [800.0, 40.0, 3.0]
            @test fill_coronal_equilibrium!(u, layout, composition, temperatures;
                species = ["Ne"]) === u
            term = IonRateTerm(composition)
            weights = free_electron_weights(composition)
            for ir in 1:3
                states = radial_view(u, layout, unknown_id(layout, :ion_densities), ir)
                @test states[deuterium] == [0.0, 5e19]
                @test sum(states[neon]) ≈ totals[ir] rtol = 1e-14
                # the fixed point: the charge states of coronal_equilibrium at the free-electron
                # density of the state itself
                free = sum(weights .* states)
                @test states[neon] ≈ coronal_equilibrium(term, 2, free, temperatures[ir];
                    species_density = totals[ir]) rtol = 1e-12
            end
            # the equilibrium by another path: the states lie in the null space of the rates
            for ir in 1:3
                states = radial_view(u, layout, unknown_id(layout, :ion_densities), ir)
                A = ion_rate_matrix(term, 2, sum(weights .* states), temperatures[ir])
                n = states[neon]
                @test maximum(abs, A * n) <= 1e-12 * maximum(abs.(A) * n)
            end
            # hot neon is ionized well beyond the cold
            charge(ir) = sum((0:10) .*
                             radial_view(u, layout, unknown_id(layout, :ion_densities), ir)[neon]) /
                         totals[ir]
            @test charge(1) > charge(2) > charge(3)
            # the method of a term gives the same, and a single name is taken as one species
            v = zeros(length(layout))
            for ir in 1:3
                states = radial_view(v, layout, unknown_id(layout, :ion_densities), ir)
                states[deuterium] .= [0.0, 5e19]
                states[first(neon)] = totals[ir]
            end
            @test fill_coronal_equilibrium!(
                v, layout, term, temperatures; species = "Ne") == u
            # both species in equilibrium together, coupled through the free-electron density
            fill_coronal_equilibrium!(v, layout, term, temperatures)
            for ir in 1:3
                states = radial_view(v, layout, unknown_id(layout, :ion_densities), ir)
                free = sum(weights .* states)
                @test sum(states[deuterium]) ≈ 5e19 rtol = 1e-14
                @test states[deuterium] ≈
                      coronal_equilibrium(term, 1, free, temperatures[ir];
                    species_density = 5e19) rtol = 1e-12
                @test states[neon] ≈ coronal_equilibrium(term, 2, free, temperatures[ir];
                    species_density = totals[ir]) rtol = 1e-12
            end
            w = copy(u)
            @test_throws "not in the composition" fill_coronal_equilibrium!(
                w, layout, composition, 100.0; species = ["Ar"])
            @test_throws "named twice" fill_coronal_equilibrium!(
                w, layout, term, 100.0; species = ["Ne", "Ne"])
            @test_throws DimensionMismatch fill_coronal_equilibrium!(
                w, layout, term, [100.0, 10.0])
            @test w == u
            @test_throws "finite and positive" fill_coronal_equilibrium!(
                w, layout, composition, [100.0, 0.0, 10.0])
            @test MAXIMUM_EQUILIBRIUM_ITERATIONS == 50
        end
    end
end
