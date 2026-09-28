# Moments over a region in momentum: the envelope of DREAM's hot region
# against a transcription of DREAM's function, the accuracy of the moments above a limit
# against the analytic integral, the limit that follows the temperature, and the moment
# terms with a region in a system.
using Random: Xoshiro

# DREAM's envelope of P_THRESHOLD_MODE_MIN_THERMAL for the cell i (1-based), transcribed
# from fvm/Equation/MomentQuantity.cpp (ThresholdEnvelope and
# EvaluateMinThermalInterpolationEnvelope) with its own variable names
function dream_min_thermal_envelope(p_f, i, p0)
    p = (p_f[1:end-1] .+ p_f[2:end]) ./ 2
    np = length(p)
    p_u, p_l, dp = p_f[i+1], p_f[i], p_f[i+1] - p_f[i]
    p_ll = i > 1 ? p_f[i-1] : 0.0
    envelope = p0 <= p_l ? 1.0 : 0.0
    if p0 > p_l && p0 < p_u
        p32 = i < np ? p[i+1] : p_u
        q = 0.5 * (p0 + p_u)
        envelope += (p_u - p0) / dp * (p32 - q) / (p32 - p[i])
    elseif p0 > p_ll && p0 < p_l
        p12 = p[i-1]
        q = 0.5 * (p0 + p_l)
        envelope += (p_l - p0) / dp * (q - p12) / (p[i] - p12)
    end
    return envelope
end

# ∫ p² exp(-p/a) dp from p0 to p1
function exponential_moment(a, p0, p1)
    primitive(p) = -a * exp(-p / a) * (p^2 + 2a * p + 2a^2)
    return primitive(p1) - primitive(p0)
end

@testset "moments over a region in momentum" begin
    rng = Xoshiro(521)
    p_faces = biuniform_p_faces(24, 10, 0.0, 0.4, 3.0)
    m = MomentumGrid(p_faces, uniform_xi_faces(6))
    np = momentum_cell_count(m)

    @testset "envelope of the hot region" begin
        region = MomentumRegion(0.0)
        @test region.side === :above && region.envelope === :interpolated
        limits = vcat(0.0, p_faces[3], p_faces[3] + 1e-9, p_faces[4] - 1e-9, m.p_centers[5], 0.013, 0.39, 0.4, 0.41,
                      1.7, p_faces[end-1], p_faces[end-1] + 0.01, p_faces[end] - 1e-3, p_faces[end], 3.5,
                      0.4 .* rand(rng, 20), 0.4 .+ 2.6 .* rand(rng, 20))
        for p0 in limits
            values = momentum_envelope(region, m, p0)
            for i in 1:np
                @test values[i] ≈ dream_min_thermal_envelope(p_faces, i, p0) atol = 1e-14
            end
            # 1 above the cells at the limit, 0 below them
            for i in 1:np
                p_faces[i] >= p0 && (i == 1 || p_faces[i-1] >= p0) && @test values[i] == 1
                i < np && p_faces[i+2] <= p0 && @test values[i] == 0
            end
            @test momentum_envelope(MomentumRegion(p0; side=:below), m, p0) ≈ 1 .- values atol = 1e-14
        end
        @test all(==(1), momentum_envelope(region, m, 0.0))
        @test all(==(0), momentum_envelope(region, m, 3.5))
        # a limit below a grid that starts above zero counts every cell, without a second part
        upper = MomentumGrid(uniform_p_faces(8, 3.0, 11.0), uniform_xi_faces(6))
        @test all(==(1), momentum_envelope(region, upper, 1.5))
        @test all(==(1), momentum_envelope(region, upper, 3.0))

        # the fraction of the cell (DREAM's MIN_MC and MAX_MC)
        fraction = MomentumRegion(0.0; envelope=:cell_fraction)
        k = 13
        p0 = p_faces[k] + 0.3 * m.p_widths[k]
        values = momentum_envelope(fraction, m, p0)
        @test values[1:k-1] == zeros(k - 1) && values[k+1:end] == ones(np - k)
        @test values[k] ≈ 0.7 rtol = 1e-12
        @test momentum_envelope(MomentumRegion(0.0; envelope=:cell_fraction, side=:below), m, p0)[k] ≈ 0.3 rtol = 1e-12
        @test momentum_envelope(fraction, m, p_faces[k])[k-1:k] == [0.0, 1.0]

        # the smooth step (DREAM's MIN_THERMAL_SMOOTH): the width of the first cell above the limit
        smooth = MomentumRegion(0.0; envelope=:smooth)
        @test smooth.smooth_width == 2
        values = momentum_envelope(smooth, m, p0)
        step = m.p_widths[findfirst(>(p0), m.p_centers)]
        @test values ≈ [(1 + tanh((p - p0) / (2 * step))) / 2 for p in m.p_centers] rtol = 1e-14
        wide = momentum_envelope(MomentumRegion(0.0; envelope=:smooth, smooth_width=4), m, p0)
        @test wide ≈ [(1 + tanh((p - p0) / (4 * step))) / 2 for p in m.p_centers] rtol = 1e-14
        above_grid = momentum_envelope(smooth, m, 3.2)
        @test above_grid ≈ [(1 + tanh((p - 3.2) / (2 * (3.2 - m.p_centers[end])))) / 2 for p in m.p_centers] rtol = 1e-14

        @test_throws ArgumentError MomentumRegion(0.5; side=:between)
        @test_throws ArgumentError MomentumRegion(0.5; envelope=:sharp)
        @test_throws ArgumentError MomentumRegion(0.5; smooth_width=0)
        @test_throws ArgumentError MomentumRegion(-0.5)
        @test_throws ArgumentError momentum_envelope(region, m, -0.1)
        @test_throws DimensionMismatch momentum_envelope!(zeros(np - 1), region, m, 0.5)
    end

    @testset "limit of the region" begin
        rad = RadialGrid(uniform_r_faces(3, 0.6))
        grid = cylindrical_grid(m, rad)
        temperature = [10.0, 100.0, 1000.0]
        thermal = ThermalMomentum(temperature)
        @test thermal.factor == 7
        expected = 7 .* sqrt.(2 .* temperature ./ Constants.ELECTRON_REST_ENERGY)
        @test thermal(0.0) ≈ expected rtol = 1e-15
        @test ThermalMomentum(temperature; factor=5)(0.0) ≈ expected .* 5 ./ 7 rtol = 1e-15
        cooling(t) = temperature .* exp(-t / 1e-3)
        @test ThermalMomentum(cooling)(1e-3) ≈ expected .* exp(-0.5) rtol = 1e-14
        plasma_at(t) = PlasmaState(cold_electron_temperature=cooling(t), ions=[IonSpecies("D", 1, 1, fill(1e19, 3))],
                                   electric_field=zeros(3), magnetic_field=fill(5.0, 3))
        @test ThermalMomentum(plasma_at)(1e-3) ≈ expected .* exp(-0.5) rtol = 1e-14
        @test_throws ArgumentError ThermalMomentum(temperature; factor=0)

        @test !is_time_dependent(MomentumRegion(0.5)) && !is_time_dependent(MomentumRegion([0.1, 0.2, 0.3]))
        @test is_time_dependent(MomentumRegion(thermal)) && is_time_dependent(MomentumRegion(t -> 0.5 + t))
        @test momentum_limit(MomentumRegion(0.5), 2) == 0.5
        @test momentum_limit(MomentumRegion([0.1, 0.2, 0.3]), 2) == 0.2
        @test momentum_limit(MomentumRegion(ThermalMomentum(cooling)), 3, 1e-3) ≈ expected[3] * exp(-0.5) rtol = 1e-14
        @test momentum_limit(MomentumRegion(t -> 0.5 + t), 1, 0.25) == 0.75
        @test_throws DimensionMismatch momentum_limit(MomentumRegion([0.1, 0.2]), 1; radii=3)
        region = MomentumRegion(ThermalMomentum(cooling))
        for ir in 1:3
            @test momentum_envelope(region, grid, ir, 1e-3) ≈
                  momentum_envelope(region, m, expected[ir] * exp(-0.5)) rtol = 1e-12
        end
    end

    @testset "accuracy of the moments above a limit" begin
        rad = RadialGrid(uniform_r_faces(1, 1.0))
        a, p_max, p0 = 0.5, 6.0, 1.234
        exact_density = 4π * exponential_moment(a, p0, p_max)
        errors = Dict{Symbol,Vector{Float64}}(:interpolated => Float64[], :cell_fraction => Float64[])
        for cells in (30, 60, 120, 240), envelope in (:interpolated, :cell_fraction)
            grid = cylindrical_grid(MomentumGrid(uniform_p_faces(cells, 0.0, p_max), uniform_xi_faces(4)), rad)
            f = [exp(-p / a) for p in momentum_grid(grid, 1).p_centers, _ in 1:4, _ in 1:1]
            n = density(grid, f; region=MomentumRegion(p0; envelope))[1]
            push!(errors[envelope], abs(n / exact_density - 1))
            # the two sides add up to the moment of all momenta
            below = density(grid, f; region=MomentumRegion(p0; envelope, side=:below))[1]
            @test n + below ≈ density(grid, f)[1] rtol = 1e-13
        end
        # second order: the error of the midpoint rule with that of the cell at the limit
        for k in 1:3
            @test 3.3 < errors[:interpolated][k] / errors[:interpolated][k+1] < 5
        end
        @test errors[:interpolated][end] < 2e-5
        @test errors[:cell_fraction][end] < 1e-4

        # the limit moves through a cell: the error of the interpolated envelope stays that
        # of the grid, the moment is continuous across the faces
        grid = cylindrical_grid(MomentumGrid(uniform_p_faces(60, 0.0, p_max), uniform_xi_faces(4)), rad)
        mg = momentum_grid(grid, 1)
        f = [exp(-p / a) for p in mg.p_centers, _ in 1:4, _ in 1:1]
        moment(limit, envelope) = density(grid, f; region=MomentumRegion(limit; envelope))[1]
        worst = Dict(:interpolated => 0.0, :cell_fraction => 0.0)
        for limit in range(mg.p_faces[12], mg.p_faces[14]; length=41), envelope in (:interpolated, :cell_fraction)
            error = abs(moment(limit, envelope) / (4π * exponential_moment(a, limit, p_max)) - 1)
            worst[envelope] = max(worst[envelope], error)
        end
        @test worst[:interpolated] < 1.5e-3
        @test worst[:interpolated] < worst[:cell_fraction]
        for envelope in (:interpolated, :cell_fraction), face in (5, 13, 40)
            below, above = moment(mg.p_faces[face] - 1e-10, envelope), moment(mg.p_faces[face] + 1e-10, envelope)
            @test below ≈ above rtol = 1e-8
            @test below ≈ moment(mg.p_faces[face], envelope) rtol = 1e-8
        end

        # the current of a distribution f = g(p) (1 + ξ): e c (4π/3) ∫ p² (p/γ) g dp, above the
        # limit against a fine trapezoidal rule
        g(p) = exp(-p / a)
        fine = range(p0, p_max; length=200_001)
        integrand = [p^3 / sqrt(1 + p^2) * g(p) for p in fine]
        integral = step(fine) * (sum(integrand) - (integrand[1] + integrand[end]) / 2)
        exact_current = Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT * 4π / 3 * integral
        previous = Inf
        for cells in (60, 120, 240)
            grid = cylindrical_grid(MomentumGrid(uniform_p_faces(cells, 0.0, p_max), uniform_xi_faces(8)), rad)
            mg = momentum_grid(grid, 1)
            f = [g(p) * (1 + xi) for p in mg.p_centers, xi in mg.xi_centers, _ in 1:1]
            j = current_density(grid, f; region=MomentumRegion(p0))[1]
            error = abs(j / exact_current - 1)
            @test error < previous / 3.5
            previous = error
            whole = current_density(grid, f)[1]
            @test j + current_density(grid, f; region=MomentumRegion(p0; side=:below))[1] ≈ whole rtol = 1e-13
            cell_rule = current_density(grid, f; pitch_rule=:cell_center, region=MomentumRegion(p0))[1]
            @test cell_rule != j
            @test cell_rule ≈ j rtol = 2e-2
        end
        @test previous < 2e-5
    end

    @testset "moment terms with a region" begin
        rad = RadialGrid(uniform_r_faces(3, 0.6))
        # another momentum grid at each radius
        grids = [MomentumGrid(biuniform_p_faces(16 + 2ir, 6, 0.0, 0.4, 3.0), uniform_xi_faces(4 + ir)) for ir in 1:3]
        grid = cylindrical_grid(grids, rad)
        cooling(t) = [10.0, 100.0, 1000.0] .* exp(-t / 1e-3)
        region = MomentumRegion(ThermalMomentum(cooling))
        fixed = MomentumRegion([0.05, 0.2, 0.7]; envelope=:cell_fraction)
        layout = UnknownLayout(rad, [RadialUnknown(:hot_electron_density; scale=1e15),
                                     RadialUnknown(:hot_electron_current_density; scale=1e4),
                                     RadialUnknown(:ohmic_current_density; scale=1e4),
                                     KineticUnknown(:hot_tail_distribution, grid; scale=1e16)])
        hot_current = CurrentMomentTerm(:hot_tail_distribution, grid; pitch_rule=:cell_center, region)
        equations = [
            Equation(:hot_electron_density, Algebraic(), DensityMomentTerm(:hot_tail_distribution, grid; region),
                     LinearTerm(:hot_electron_density; coefficient=-1)),
            Equation(:hot_electron_current_density, Algebraic(), hot_current,
                     LinearTerm(:hot_electron_current_density; coefficient=-1)),
            # DREAM's ohmic current of the fully kinetic mode: all momenta minus the hot current
            Equation(:ohmic_current_density, Algebraic(),
                     CurrentMomentTerm(:hot_tail_distribution, grid; pitch_rule=:cell_center),
                     LinearTerm(:hot_electron_current_density; coefficient=-1),
                     LinearTerm(:ohmic_current_density; coefficient=-1)),
            Equation(:hot_tail_distribution, TimeDerivative(),
                     LinearTerm(:hot_tail_distribution; coefficient=[-1.0, -2.0, -3.0]))]
        system = EquationSystem(layout, equations)
        f_id = unknown_id(layout, :hot_tail_distribution)
        u = zeros(length(layout))
        f = [1e16 .* rand(rng, momentum_cell_count(grid, ir), pitch_cell_count(grid, ir)) for ir in 1:3]
        for ir in 1:3
            kinetic_view(u, layout, f_id, ir) .= f[ir]
        end
        du = zeros(length(layout))
        for t in (0.0, 1e-3, 0.0)
            rhs!(du, u, system, t)
            for ir in 1:3
                mg = grids[ir]
                envelope = momentum_envelope(region, grid, ir, t)
                @test envelope ≈ momentum_envelope(region, mg, 7 * sqrt(2 * cooling(t)[ir] / Constants.ELECTRON_REST_ENERGY)) rtol = 1e-12
                volume = 2π .* mg.p_centers .^ 2 .* mg.p_widths
                n_hot = sum(envelope[i] * volume[i] * mg.xi_widths[j] * f[ir][i, j]
                            for i in 1:momentum_cell_count(mg), j in 1:pitch_cell_count(mg))
                speed = mg.p_centers ./ sqrt.(1 .+ mg.p_centers .^ 2)
                charge = Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT
                j_all = charge * sum(volume[i] * speed[i] * mg.xi_centers[j] * mg.xi_widths[j] * f[ir][i, j]
                                     for i in 1:momentum_cell_count(mg), j in 1:pitch_cell_count(mg))
                j_hot = charge * sum(envelope[i] * volume[i] * speed[i] * mg.xi_centers[j] * mg.xi_widths[j] * f[ir][i, j]
                                     for i in 1:momentum_cell_count(mg), j in 1:pitch_cell_count(mg))
                @test only(unknown_view(du, layout, :hot_electron_density, ir)) ≈ n_hot rtol = 1e-13
                @test only(unknown_view(du, layout, :hot_electron_current_density, ir)) ≈ j_hot rtol = 1e-12 atol = 1e-12 * abs(j_all)
                @test only(unknown_view(du, layout, :ohmic_current_density, ir)) ≈ j_all rtol = 1e-12
                @test unknown_view(du, layout, :hot_tail_distribution, ir) == -ir .* f[ir]
            end
        end
        # the limit falls with the temperature: more electrons count as hot
        early, late = zeros(length(layout)), zeros(length(layout))
        rhs!(early, u, system, 0.0)
        rhs!(late, u, system, 2e-3)
        for ir in 1:3
            @test only(unknown_view(late, layout, :hot_electron_density, ir)) >
                  only(unknown_view(early, layout, :hot_electron_density, ir))
        end
        @test (@allocated rhs!(du, u, system, 2e-3)) == 0

        # the stored Jacobian of the linear system gives the right-hand side
        for t in (0.0, 1e-3)
            J = jacobian_prototype(system)
            jacobian!(J, system, u, t)
            rhs!(du, u, system, t)
            @test J * u ≈ du rtol = 1e-12
            v = unknown_scales(layout) .* randn(rng, length(layout))
            rhs!(du, v, system, t)
            @test J * v ≈ du rtol = 1e-12
        end
        @test jacobian_colors(system) <= 3
        @test is_linear(hot_current) && jacobian_assembly(hot_current) === :direct

        # a region of another grid, and a limit per radius of another length
        other = cylindrical_grid(grids[1], rad)
        @test_throws DimensionMismatch EquationSystem(layout, [
            Equation(:hot_electron_density, Algebraic(),
                     DensityMomentTerm(:hot_tail_distribution, other; region=fixed),
                     LinearTerm(:hot_electron_density; coefficient=-1)), equations[2:4]...])
        short = DensityMomentTerm(:hot_tail_distribution, grid; region=MomentumRegion([0.1, 0.2]))
        @test_throws DimensionMismatch rhs!(du, short, u, layout, 1, 0.0)
        # without a region the term is the one of all momenta
        plain = DensityMomentTerm(:hot_tail_distribution; coefficient=2)
        @test plain.envelopes === nothing
        fill!(du, 0)
        rhs!(du, plain, u, layout, 1, 0.0)
        @test only(unknown_view(du, layout, :hot_electron_density, 2)) ≈
              2 * sum(2π .* grids[2].p_centers .^ 2 .* grids[2].p_widths .* (f[2] * grids[2].xi_widths)) rtol = 1e-13

        @testset "records" begin
            times = collect(range(0, 2e-3; length=5))
            record = describe(system, DescriptionContext(; times))
            density_record = record["equations"]["hot_electron_density"]["terms"]["term_1"]
            @test density_record["region"]["side"] == "above"
            @test density_record["region"]["envelope"] == "interpolated"
            @test density_record["region"]["limit"]["type"] == "ThermalMomentum"
            @test density_record["region"]["limit"]["factor"] == 7
            @test density_record["region"]["limit"]["temperature"]["values"][:, 3] ≈ cooling(times[3])
            ohmic_record = record["equations"]["ohmic_current_density"]["terms"]["term_1"]
            @test !haskey(ohmic_record, "region")
            decay = record["equations"]["hot_tail_distribution"]["terms"]["term_1"]
            @test decay["coefficient"] == [-1.0, -2.0, -3.0]
            fixed_record = describe(fixed)
            @test fixed_record["limit"] == [0.05, 0.2, 0.7] && fixed_record["envelope"] == "cell_fraction"
            @test !haskey(fixed_record, "smooth_width")
            @test describe(MomentumRegion(0.3; envelope=:smooth))["smooth_width"] == 2
            for T in (DensityMomentTerm, CurrentMomentTerm)
                fields = settings_fields(T)
                @test Set(union(fields.described, fields.derived)) == Set(fieldnames(T))
            end
        end
    end

    @testset "coefficient per radius of a linear term" begin
        rad = RadialGrid(uniform_r_faces(3, 0.6))
        layout = UnknownLayout(rad, [RadialUnknown(:electric_field), RadialUnknown(:ohmic_current_density),
                                     ScalarUnknown(:plasma_current)])
        conductivity = [1e6, 2e6, 3e6]
        term = LinearTerm(:electric_field; coefficient=conductivity)
        @test term.coefficient === conductivity
        @test LinearTerm(:electric_field; coefficient=1:3).coefficient == [1.0, 2.0, 3.0]
        u = collect(1.0:length(layout))
        du = zeros(length(layout))
        row = unknown_id(layout, :ohmic_current_density)
        Runes2P.validate(term, layout, row)
        rhs!(du, term, u, layout, row, 0.0)
        for ir in 1:3
            @test only(unknown_view(du, layout, :ohmic_current_density, ir)) ==
                  conductivity[ir] * only(unknown_view(u, layout, :electric_field, ir))
        end
        @test_throws DimensionMismatch Runes2P.validate(LinearTerm(:electric_field; coefficient=[1.0, 2.0]), layout, row)
        @test_throws DimensionMismatch Runes2P.validate(LinearTerm(:plasma_current; coefficient=[1.0, 2.0, 3.0]), layout,
                                                     unknown_id(layout, :plasma_current))
    end
end
