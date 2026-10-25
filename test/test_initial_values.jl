# Initial values from the equations: the energy of the cold electrons from their temperature
# and density at the start of a run, rows in any order, and the refusals of rows that cannot
# give a value.

@testset "initial values from the equations" begin
    radial = RadialGrid(uniform_r_faces(2, 0.5))
    temperature, density = [100.0, 60.0], [2e19, 1e19]
    thermal = THERMAL_ENERGY_FACTOR .* density .* temperature
    minus(name) = LinearTerm(name; coefficient = -1)
    layout = UnknownLayout(radial,
        [RadialUnknown(:cold_electron_temperature; scale = 100.0),
            RadialUnknown(:cold_electron_density; scale = 1e20),
            RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e22)])
    # a heating of zero: the temperature holds, and the energy with it
    equations = [
        Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy),
            PrescribedValueTerm(zeros(2))),
        Equation(:cold_electron_density, Algebraic(),
            PrescribedValueTerm(density), minus(:cold_electron_density)),
        thermal_energy_equation()]
    function start(energy)
        u = zeros(length(layout))
        radial_values(u, layout, 1) .= temperature
        radial_values(u, layout, 2) .= density
        radial_values(u, layout, 3) .= energy
        return u
    end
    system = EquationSystem(layout, equations; initial_state = start(thermal))
    W = unknown_id(layout, :cold_thermal_energy)

    @testset "the energy of the cold electrons" begin
        @test only(filter(eq -> eq.unknown === :cold_thermal_energy, system.equations)).initial_value ===
              :from_equation
        # a state with no energy and one with a stale energy both start from 3/2 e n T
        for energy in (zeros(2), 2 .* thermal)
            u = start(energy)
            @test fill_initial_values!(u, system, 0.0) == [:cold_thermal_energy]
            @test radial_values(u, layout, W) ≈ thermal rtol = 1e-14
            @test radial_values(u, layout, 1) == temperature
            solution = integrate(system, start(energy), [0.0, 1e-6, 2e-6]; stepper = FixedSteps())
            for k in 1:3
                @test radial_values(solution.states[:, k], layout, W) ≈ thermal rtol = 1e-12
            end
            @test solution.settings["segment"]["initial_values_from_equations"] ==
                  ["cold_thermal_energy"]
        end
        @test describe(system)["equations"]["cold_thermal_energy"]["initial_value"] ==
              "from_equation"
        @test !haskey(describe(system)["equations"]["cold_electron_density"], "initial_value")
        # a resume fills only the unknowns the stored layout lacks
        u = start(zeros(2))
        @test isempty(fill_initial_values!(u, system, 0.0; unknowns = Symbol[]))
        @test radial_values(u, layout, W) == zeros(2)
    end

    @testset "rows in any order" begin
        # the density from a second unknown, n = 2 m, filled after the energy that reads it
        ordered = UnknownLayout(radial,
            [RadialUnknown(:cold_electron_temperature; scale = 100.0),
                RadialUnknown(:half_density; scale = 1e20),
                RadialUnknown(:cold_electron_density; scale = 1e20),
                RadialUnknown(:cold_thermal_energy; scale = THERMAL_ENERGY_FACTOR * 1e22)])
        rows = [thermal_energy_equation(),
            Equation(:cold_electron_density, Algebraic(),
                LinearTerm(:half_density; coefficient = 2.0),
                minus(:cold_electron_density); initial_value = :from_equation),
            Equation(:half_density, Algebraic(), PrescribedValueTerm(density ./ 2), minus(:half_density)),
            Equation(:cold_electron_temperature, TimeDerivative(:cold_thermal_energy),
                PrescribedValueTerm(zeros(2)))]
        u = zeros(length(ordered))
        radial_values(u, ordered, 1) .= temperature
        radial_values(u, ordered, 2) .= density ./ 2
        radial_values(u, ordered, 3) .= 1.0          # a placeholder the fill replaces
        radial_values(u, ordered, 4) .= thermal
        both = EquationSystem(ordered, rows; initial_state = u)
        radial_values(u, ordered, 4) .= 0.0
        @test Set(fill_initial_values!(u, both, 0.0)) ==
              Set([:cold_thermal_energy, :cold_electron_density])
        @test radial_values(u, ordered, 3) ≈ density rtol = 1e-14
        @test radial_values(u, ordered, 4) ≈ thermal rtol = 1e-14
    end

    @testset "refusals" begin
        # only an algebraic row gives an initial value
        @test_throws ArgumentError Equation(
            :cold_electron_temperature, TimeDerivative(), PrescribedValueTerm(zeros(2));
            initial_value = :from_equation)
        @test_throws ArgumentError Equation(:x, Algebraic(), minus(:x); initial_value = :other)
        pair = UnknownLayout(radial, [
            RadialUnknown(:a; scale = 1.0), RadialUnknown(:b; scale = 1.0)])
        u = ones(length(pair))
        # a = b + 1 and b = a have no solution: the passes never settle
        cycle = EquationSystem(pair,
            [
                Equation(:a, Algebraic(), LinearTerm(:b),
                    PrescribedValueTerm(ones(2)), minus(:a);
                    initial_value = :from_equation),
                Equation(:b, Algebraic(), LinearTerm(:a), minus(:b);
                    initial_value = :from_equation)])
        @test_throws ArgumentError fill_initial_values!(copy(u), cycle, 0.0)
        # a row quadratic in its unknown, a² = 4 b
        quadratic = EquationSystem(pair,
            [
                Equation(:a, Algebraic(), ProductTerm(:a, :a),
                    LinearTerm(:b; coefficient = -4.0);
                    initial_value = :from_equation),
                Equation(:b, Algebraic(), PrescribedValueTerm(ones(2)), minus(:b))];
            initial_state = u)
        @test_throws ArgumentError fill_initial_values!(copy(u), quadratic, 0.0)
    end
end
