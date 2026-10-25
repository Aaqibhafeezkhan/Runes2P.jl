# Radial profiles on points of their own: numbers, values or callables of r
# evaluated on the cell centres or the faces of the run's grid, interpolators of
# Interpolations.jl through the extension Runes2PInterpolationsExt, and their records.
# Synthetic profiles only.

using Interpolations

# a callable of r that counts its evaluations
struct CountingProfile{F}
    profile::F
    calls::Base.RefValue{Int}
end
CountingProfile(profile) = CountingProfile(profile, Ref(0))
(c::CountingProfile)(r) = (c.calls[] += 1; c.profile(r))

@testset "radial profiles" begin
    minor_radius = 0.5
    grid = RadialGrid(uniform_r_faces(11, minor_radius))
    centers, faces = grid.r_centers, grid.r_faces

    @testset "evaluation" begin
        @test radial_values(2.0, centers) == fill(2.0, 11)
        values = collect(1.0:11.0)
        @test radial_values(values, centers) === values     # a vector is not copied
        @test_throws DimensionMismatch radial_values(values, faces)
        @test radial_values(r -> 2r, faces) == 2 .* faces
        # a linear profile on five points is reproduced on the centres and the faces
        linear(r) = 2 + 3r
        points = collect(range(0, minor_radius; length = 5))
        profile = linear_interpolation(points, linear.(points))
        @test radial_values(profile, centers) ≈ linear.(centers) rtol = 1e-14
        @test radial_values(profile, faces) ≈ linear.(faces) rtol = 1e-14
        # an interpolator is an array of its knot values: with as many knots as points it is
        # still evaluated, not taken for the values
        eleven = collect(range(0, minor_radius; length = 11))
        quadratic = linear_interpolation(eleven, eleven .^ 2)
        @test maximum(abs, radial_values(quadratic, centers) .- centers .^ 2) <= 0.05^2 / 4  # linear interpolation of r²
        @test radial_values(quadratic, centers) != eleven .^ 2
        # beyond the last point: an error by default, the last value with Flat()
        short = collect(range(0, 0.4; length = 5))
        @test_throws BoundsError radial_values(linear_interpolation(short, linear.(short)), faces)
        flat = linear_interpolation(short, linear.(short); extrapolation_bc = Flat())
        @test radial_values(flat, faces)[end] == linear(0.4)
        # a monotone interpolation keeps a steep density edge positive and monotone
        edge = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        density = [1e20, 1e20, 9e19, 1e18, 1e16, 1e16]
        monotone = interpolate(edge, density, FritschCarlsonMonotonicInterpolation())
        fine = radial_values(monotone, collect(range(0, 0.5; length = 1001)))
        @test all(>=(1e16 * (1 - 1e-12)), fine) && all(<=(1e20 * (1 + 1e-12)), fine)
        @test issorted(fine; rev = true)
    end

    @testset "plasma state on a radial grid" begin
        temperature(r) = 5 + 100 * (1 - (r / minor_radius)^2)
        edge = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        argon = interpolate(edge, [1e18, 1e18, 9e17, 1e16, 1e14, 1e14], FritschCarlsonMonotonicInterpolation())
        plasma = PlasmaState(grid; cold_electron_temperature = temperature,
            ions = [
                IonSpecies("D", 1, 1, r -> 1e20, grid), IonSpecies(
                    "Ar", 18, 1, argon, grid)],
            electric_field = 0.1, magnetic_field = fill(5.0, 11))
        reference = PlasmaState(; cold_electron_temperature = temperature.(centers),
            ions = [
                IonSpecies("D", 1, 1, fill(1e20, 11)), IonSpecies("Ar", 18, 1, argon.(centers))],
            electric_field = fill(0.1, 11), magnetic_field = fill(5.0, 11))
        for field in (:cold_electron_temperature, :cold_electron_density, :electric_field, :magnetic_field)
            @test getfield(plasma, field) == getfield(reference, field)
        end
        @test plasma.ions[2].charge_state_densities ==
              reference.ions[2].charge_state_densities
        @test all(>(0), plasma.ions[2].charge_state_densities[2, :])
        # one profile per charge state, and the kinetic grid in place of the radial one
        kinetic = cylindrical_grid(MomentumGrid(uniform_p_faces(4, 0.0, 1.0), uniform_xi_faces(2)), grid)
        states = IonSpecies("Ar", 18, [r -> (j == 2 ? 1e17 : 0.0) for j in 1:19], kinetic)
        @test states.charge_state_densities ==
              IonSpecies("Ar", 18, 1, fill(1e17, 11)).charge_state_densities
        @test_throws ArgumentError IonSpecies("Ar", 18, [r -> 0.0], grid)
        @test PlasmaState(kinetic; cold_electron_temperature = 10.0,
            ions = [IonSpecies("D", 1, 1, 1e20, kinetic)],
            electric_field = 0.0, magnetic_field = 5.0).cold_electron_density ==
              fill(1e20, 11)
    end

    @testset "records of interpolators" begin
        points = [0.0, 0.1, 0.25, 0.5]
        record = describe(linear_interpolation(points, [4.0, 3.0, 2.0, 1.0]; extrapolation_bc = Flat()))
        @test record["type"] == "Interpolations.GriddedInterpolation"
        @test occursin("Linear", record["method"]) && record["extrapolation"] == "Flat()"
        @test record["knots"] == points && record["values"] == [4.0, 3.0, 2.0, 1.0]
        @test describe(linear_interpolation(points, [4.0, 3.0, 2.0, 1.0]))["extrapolation"] ==
              "Throw()"
        monotone = describe(interpolate(points, [4.0, 3.0, 2.0, 1.0], FritschCarlsonMonotonicInterpolation()))
        @test monotone["method"] == "FritschCarlsonMonotonicInterpolation()" &&
              monotone["knots"] == points
        filled = describe(extrapolate(interpolate((points,), [4.0, 3.0, 2.0, 1.0], Gridded(Linear())), 0.0))
        @test filled["extrapolation"] == "fill" && filled["fill_value"] == 0.0
        in_time = describe(linear_interpolation(
            ([0.0, 1.0], points), [1.0 2.0 3.0 4.0;
                                   5.0 6.0 7.0 8.0]))
        @test in_time["knots_1"] == [0.0, 1.0] && in_time["knots_2"] == points &&
              size(in_time["values"]) == (2, 4)
    end

    @testset "Rechester-Rosenbluth level on five points" begin
        # δB/B and the safety factor given on five points, evaluated on the faces of 11 and
        # 20 radii: the run equals the run with the values on the faces
        momentum = MomentumGrid(biuniform_p_faces(6, 3, 0.0, 0.5, 3.0), uniform_xi_faces(4))
        points = collect(range(0, minor_radius; length = 5))
        level = linear_interpolation(points, 1e-3 .* [1.0, 0.8, 0.5, 0.3, 0.2])
        safety_factor(r) = 1 + 2 * (r / minor_radius)^2
        times = collect(range(0, 1e-3; length = 6))
        for radii in (11, 20)
            kinetic = cylindrical_grid(momentum, RadialGrid(uniform_r_faces(radii, minor_radius)))
            on_faces = kinetic.radial_grid.r_faces
            function run(level_at, q)
                radial = AdvectionDiffusion(kinetic; scheme = :central, r_boundary = :absorbing)
                layout = UnknownLayout(kinetic.radial_grid, [KineticUnknown(:f, kinetic)])
                system = EquationSystem(layout, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, radial))])
                # names of their own: a closure assigns the variables of the loop body
                transport = RechesterRosenbluthUpdate(radial, level_at; major_radius = 6.2, safety_factor = q)
                f0 = [1 + 0.2 * xi + 0.1 * ir
                      for _ in momentum.p_centers, xi in momentum.xi_centers, ir in 1:radii]
                return integrate(system, vec(f0), times; updates = (transport,)), transport
            end
            profiled, update = run(t -> level, safety_factor)
            valued, _ = run(t -> level.(on_faces), safety_factor.(on_faces))
            @test profiled.states == valued.states
            @test update.delta_b_over_b == level.(on_faces)
            record = describe(update, DescriptionContext(; times))
            @test record["delta_b_over_b"]["profile"]["knots"] == points
            @test record["delta_b_over_b"]["profile"]["values"] ≈
                  1e-3 .* [1.0, 0.8, 0.5, 0.3, 0.2] rtol = 1e-15
            @test record["delta_b_over_b"]["values"][:, 1] == level.(on_faces)
            @test record["safety_factor_profile"]["type"] == "callable"
            @test record["safety_factor"] == safety_factor.(on_faces)
        end
        # a profile that changes in time is recorded per sampled time
        kinetic = cylindrical_grid(momentum, grid)
        radial = AdvectionDiffusion(kinetic; scheme = :central, r_boundary = :absorbing)
        changing = RechesterRosenbluthUpdate(radial, t -> (t < 5e-4 ? level : r -> 0.0); major_radius = 6.2)
        record = describe(changing, DescriptionContext(; times))
        @test record["delta_b_over_b"]["profiles"]["sample_1"]["knots"] == points
        @test record["delta_b_over_b"]["profiles"]["sample_6"]["type"] == "callable"
        @test !haskey(
            describe(RechesterRosenbluthUpdate(radial, t -> 1e-3; major_radius = 6.2),
                DescriptionContext(; times)),
            "safety_factor_profile")
        # a history of profiles records each one
        history = describe(PiecewiseConstant([level, r -> 0.0], [5e-4]))
        @test history["profiles"]["interval_1"]["knots"] == points
        @test history["profiles"]["interval_2"]["type"] == "callable" &&
              history["switch_times"] == [5e-4]
        # an interpolator in r in place of the function of time is refused
        @test_throws "function of time" RechesterRosenbluthUpdate(radial, level; major_radius = 6.2)
    end

    @testset "prescribed value as a profile" begin
        # a prescribed field E(r) on five points, held by an algebraic equation: evaluated
        # once per time, not in every Newton iteration
        points = collect(range(0, minor_radius; length = 5))
        field = CountingProfile(linear_interpolation(points, [0.1, 0.2, 0.4, 0.3, 0.1]))
        term = PrescribedValueTerm(t -> field)
        layout = UnknownLayout(grid, [RadialUnknown(:electric_field)])
        system = EquationSystem(layout,
            [Equation(:electric_field, Algebraic(), term,
                LinearTerm(:electric_field; coefficient = -1))])
        times = collect(range(0, 1e-3; length = 5))
        solution = integrate(system, zeros(11), times)
        @test solution.states[:, end] ≈ field.profile.(centers) rtol = 1e-12
        # once per time for the steps and once per time for the record of the run, which
        # samples the function at the time points (observed: exactly these 2 × 11 × 5)
        @test field.calls[] <= 2 * 11 * length(times)
        record = describe(term, DescriptionContext(; times))
        @test record["values"]["profile"]["type"] == "callable"
        @test record["values"]["values"][:, 1] ≈ field.profile.(centers) rtol = 1e-15
        # an interpolator itself is refused: in r or in t, it is ambiguous
        @test_throws "function of time" PrescribedValueTerm(linear_interpolation(points, points))
    end
end
