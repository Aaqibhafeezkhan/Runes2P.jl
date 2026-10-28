# The kinetic energy moment of a distribution: the energy of a resolved
# relativistic Maxwellian against the mean kinetic energy by quadrature, the split into the
# regions above and below a threshold that follows the temperature, and the Jacobian with
# the column of the temperature.
using QuadGK: quadgk

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
    end
end
