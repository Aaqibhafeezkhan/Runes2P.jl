# Runaway grid above the hot-tail grid and the flux handover between them: the grid, the
# interface term against a single grid cut in two, conservation across different pitch
# grids, DREAM's pitch interpolation, the Jacobian.
using Random: Xoshiro
using SparseArrays: findnz

# Σ 𝒱' Δp Δξ Δr of a kinetic unknown in a state or a rate
function kinetic_content(layout, name, values)
    id = unknown_id(layout, name)
    grid = kinetic_grid(layout, id)
    total = zero(eltype(values))
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        V = phase_space_jacobian(grid, ir)
        f = kinetic_view(values, layout, id, ir)
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)

            total += V[i, j] * m.p_widths[i] * m.xi_widths[j] *
                     grid.radial_grid.r_widths[ir] * f[i, j]
        end
    end
    return total
end

function random_coefficients!(op, rng; cross_terms = false)
    for A in (op.advection_p, op.advection_xi, op.advection_r)
        A .= randn(rng, size(A))
    end
    for D in (op.diffusion_p_p, op.diffusion_xi_xi, op.diffusion_r_r)
        D .= rand(rng, size(D))
    end
    if cross_terms
        op.diffusion_p_xi .= randn(rng, size(op.diffusion_p_xi))
        op.diffusion_xi_p .= randn(rng, size(op.diffusion_xi_p))
    end
    return op
end

# hot-tail and runaway unknowns with their Fokker-Planck terms and the interface term
function two_grid_system(hot_op, runaway_op; interface_options...)
    rad = hot_op.grid.radial_grid
    layout = UnknownLayout(rad,
        [KineticUnknown(:hot_tail_distribution, hot_op.grid),
            KineticUnknown(:runaway_distribution, runaway_op.grid)])
    hot_term = FokkerPlanckTerm(:hot_tail_distribution, hot_op)
    interface = MomentumInterfaceTerm(
        :hot_tail_distribution, :runaway_distribution, hot_term, runaway_op;
        interface_options...)
    system = EquationSystem(layout,
        [
            Equation(:hot_tail_distribution, TimeDerivative(), hot_term, interface),
            Equation(:runaway_distribution, TimeDerivative(),
                FokkerPlanckTerm(:runaway_distribution, runaway_op), interface)])
    return (; layout, system, interface)
end

@testset "momentum interface" begin
    rng = Xoshiro(54)
    rad = RadialGrid(uniform_r_faces(2, 0.5))
    hot_pitch = uniform_xi_faces(9)
    hot_momentum = MomentumGrid(biuniform_p_faces(9, 4, 0.0, 0.4, 2.5), hot_pitch)
    hot = cylindrical_grid(hot_momentum, rad)

    @testset "runaway grid" begin
        runaway = runaway_grid(hot, uniform_p_faces(6, 2.5, 20.0)[2:end])
        @test has_shared_momentum_grid(runaway)
        m = momentum_grid(runaway)
        @test m.p_faces[1] === hot_momentum.p_faces[end]
        @test m.p_faces == uniform_p_faces(6, 2.5, 20.0)
        # the default pitch grid of the runaway grid: 12 cells to 0.9 rad, 8 to π
        @test m.xi_faces == biuniform_theta_xi_faces(20, 12, 0.9)
        @test count(>=(cos(0.9) - 1e-14), m.xi_faces) == 13
        @test runaway.radial_grid === hot.radial_grid
        with_pitch = runaway_grid(hot, [10.0, 20.0], biuniform_theta_xi_faces(15, 10, 0.5))
        @test size(with_pitch) == (2, 15, 2)
        @test_throws ArgumentError runaway_grid(hot, [2.5, 5.0])
        @test_throws ArgumentError runaway_grid(hot, Float64[])

        # the Fokker-Planck operator on a grid with p_min > 0
        plasma = PlasmaState(cold_electron_temperature = [100.0, 50.0],
            ions = [IonSpecies("D", 1, 1, [1e20, 5e19])],
            electric_field = [1.0, 2.0], magnetic_field = [5.0, 5.0])
        op = fokker_planck_operator(
            runaway, plasma; synchrotron = true, p_lower_boundary = :interface,
            p_upper_boundary = :f_zero)
        @test op.p_lower_boundary === :interface
        @test all(==(0), op.advection_p_times_p2_at_zero)
        f = rand(rng, size(runaway)...)
        df = apply!(zeros(size(runaway)), op, f)
        @test all(isfinite, df)
        # no flux through the interface face from the operator itself
        @test all(momentum_flux(op, f, 1, ir) == 0 for ir in 1:2)
        @test check_stencil_pattern(op)
        @test_throws ArgumentError AdvectionDiffusion(hot; p_lower_boundary = :interface)
    end

    runaway = runaway_grid(hot, uniform_p_faces(5, 2.5, 12.0)[2:end])

    @testset "arguments" begin
        hot_op = AdvectionDiffusion(hot; p_upper_boundary = :interface)
        runaway_op = AdvectionDiffusion(runaway; p_lower_boundary = :interface)
        term = MomentumInterfaceTerm(:hot_tail_distribution, :runaway_distribution, hot_op, runaway_op)
        @test term.pitch_mapping === :overlap && term.scheme === :upwind
        @test dependencies(term) == [:hot_tail_distribution, :runaway_distribution]
        @test is_linear(term) && coupling(term) === :sparse && treatment(term) === :implicit
        # the scheme follows the lower operator; a four-point scheme gives upwind
        for (scheme, expected) in ((:central, :central), (
            :exponential_fitting, :exponential_fitting),
            (:tcdf, :upwind))
            op = AdvectionDiffusion(hot; scheme, p_upper_boundary = :interface)
            @test MomentumInterfaceTerm(:a, :b, op, runaway_op).scheme === expected
        end
        @test MomentumInterfaceTerm(:a, :b, hot_op, runaway_op; scheme = :central).scheme ===
              :central
        @test_throws ArgumentError MomentumInterfaceTerm(
            :a, :b, hot_op, runaway_op; scheme = :tcdf)
        @test_throws ArgumentError MomentumInterfaceTerm(
            :a, :b, hot_op, runaway_op; pitch_mapping = :nearest)
        @test_throws ArgumentError MomentumInterfaceTerm(:a, :a, hot_op, runaway_op)
        @test_throws "p_upper_boundary = :interface" MomentumInterfaceTerm(
            :a, :b, AdvectionDiffusion(hot),
            runaway_op)
        @test_throws "p_lower_boundary = :interface" MomentumInterfaceTerm(:a, :b, hot_op,
            AdvectionDiffusion(runaway))
        # the two grids must meet at one face
        gap = cylindrical_grid(MomentumGrid(uniform_p_faces(5, 2.6, 12.0), hot_pitch), rad)
        @test_throws "the same face" MomentumInterfaceTerm(:a, :b, hot_op,
            AdvectionDiffusion(gap; p_lower_boundary = :interface))
        other_radii = cylindrical_grid(momentum_grid(runaway), RadialGrid(uniform_r_faces(2, 0.7)))
        @test_throws ArgumentError MomentumInterfaceTerm(:a, :b, hot_op,
            AdvectionDiffusion(other_radii; p_lower_boundary = :interface))
        # the term belongs to the equations of its two unknowns
        layout = UnknownLayout(rad,
            [KineticUnknown(:hot_tail_distribution, hot),
                KineticUnknown(:runaway_distribution, runaway),
                KineticUnknown(:other, hot)])
        @test_throws ArgumentError EquationSystem(layout,
            [
                Equation(:hot_tail_distribution, TimeDerivative(), term),
                Equation(:runaway_distribution, TimeDerivative(), term),
                Equation(:other, TimeDerivative(), term)])
    end

    @testset "one grid cut in two ($scheme, $mapping)" for (scheme, mapping) in (
        (:upwind, :overlap), (:central, :overlap),
        (:exponential_fitting, :overlap), (:upwind, :dream))
        # the hot-tail and the runaway grid with the same pitch grid are the two parts of one
        # grid, and the interface flux is the flux through an interior face of that grid
        p_faces = vcat(biuniform_p_faces(9, 4, 0.0, 0.4, 2.5), uniform_p_faces(5, 2.5, 12.0)[2:end])
        cut = 9
        whole = cylindrical_grid(MomentumGrid(p_faces, hot_pitch), rad)
        above = runaway_grid(hot, p_faces[(cut + 2):end], hot_pitch)
        whole_op = random_coefficients!(AdvectionDiffusion(whole; scheme, p_upper_boundary = :f_zero), rng)
        whole_op.advection_p_times_p2_at_zero .= randn(rng, size(whole_op.advection_p_times_p2_at_zero))
        hot_op = AdvectionDiffusion(hot; scheme, p_upper_boundary = :interface)
        runaway_op = AdvectionDiffusion(
            above; scheme, p_lower_boundary = :interface, p_upper_boundary = :f_zero)
        for name in (:advection_p, :diffusion_p_p)
            getfield(hot_op, name) .= getfield(whole_op, name)[1:(cut + 1), :, :]
            getfield(runaway_op, name) .= getfield(whole_op, name)[(cut + 1):end, :, :]
        end
        for name in (:advection_xi, :diffusion_xi_xi, :advection_r, :diffusion_r_r)
            getfield(hot_op, name) .= getfield(whole_op, name)[1:cut, :, :]
            getfield(runaway_op, name) .= getfield(whole_op, name)[(cut + 1):end, :, :]
        end
        hot_op.advection_p_times_p2_at_zero .= whole_op.advection_p_times_p2_at_zero
        case = two_grid_system(hot_op, runaway_op; pitch_mapping = mapping)
        f = rand(rng, size(whole)...)
        expected = apply!(zeros(size(whole)), whole_op, f)
        u = zeros(length(case.layout))
        hot_id = unknown_id(case.layout, :hot_tail_distribution)
        runaway_id = unknown_id(case.layout, :runaway_distribution)
        for ir in 1:2
            kinetic_view(u, case.layout, hot_id, ir) .= f[1:cut, :, ir]
            kinetic_view(u, case.layout, runaway_id, ir) .= f[(cut + 1):end, :, ir]
        end
        du = zeros(length(case.layout))
        rhs!(du, u, case.system, 0.0)
        scale = maximum(abs, expected)
        for ir in 1:2
            @test kinetic_view(du, case.layout, hot_id, ir) ≈ expected[1:cut, :, ir] atol = 1e-13 *
                                                                                            scale
            @test kinetic_view(du, case.layout, runaway_id, ir) ≈
                  expected[(cut + 1):end, :, ir] atol = 1e-13 * scale
            # the flux through the interface is the flux through that face of the whole grid
            @test interface_flux(case.interface, u, case.layout, ir) ≈
                  momentum_flux(whole_op, f, cut + 1, ir) rtol = 1e-13
        end
    end

    @testset "conservation across different pitch grids ($scheme, $mapping)" for scheme in (:upwind, :central, :exponential_fitting),
        mapping in (:overlap, :dream)

        hot_op = random_coefficients!(
            AdvectionDiffusion(hot; scheme, p_upper_boundary = :interface), rng;
            cross_terms = true)
        runaway_op = random_coefficients!(
            AdvectionDiffusion(runaway; scheme, p_lower_boundary = :interface), rng;
            cross_terms = true)
        runaway_op.advection_p[end, :, :] .= 0   # close the outer p boundary
        case = two_grid_system(hot_op, runaway_op; pitch_mapping = mapping)
        layout = case.layout
        u = rand(rng, length(layout))
        du = zeros(length(layout))
        rhs!(du, u, case.system, 0.0)
        hot_change = kinetic_content(layout, :hot_tail_distribution, du)
        runaway_change = kinetic_content(layout, :runaway_distribution, du)
        scale = kinetic_content(layout, :hot_tail_distribution, abs.(du)) +
                kinetic_content(layout, :runaway_distribution, abs.(du))
        @test abs(hot_change + runaway_change) < 1e-13 * scale

        # the interface term alone moves what it takes from the lower grid to the upper one
        only_interface = EquationSystem(layout,
            [
                Equation(:hot_tail_distribution, TimeDerivative(), case.interface),
                Equation(:runaway_distribution, TimeDerivative(), case.interface)])
        rhs!(du, u, only_interface, 0.0)
        hot_change = kinetic_content(layout, :hot_tail_distribution, du)
        runaway_change = kinetic_content(layout, :runaway_distribution, du)
        @test abs(hot_change) > 0
        @test runaway_change ≈ -hot_change rtol = 1e-13
        through_face = sum(interface_flux(case.interface, u, layout, ir) *
                           rad.spatial_jacobian[ir] * rad.r_widths[ir]
        for ir in 1:2)
        @test runaway_change ≈ through_face rtol = 1e-13
        # only the cells at the face change
        hot_id, runaway_id = unknown_id(layout, :hot_tail_distribution),
        unknown_id(layout, :runaway_distribution)
        for ir in 1:2
            @test all(==(0), kinetic_view(du, layout, hot_id, ir)[1:(end - 1), :])
            @test all(==(0), kinetic_view(du, layout, runaway_id, ir)[2:end, :])
        end

        # the stored Jacobian is the term
        system = case.system
        rhs!(du, u, system, 0.0)
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        @test J * u ≈ du rtol = 1e-12
        other = rand(rng, length(layout))
        jacobian!(J, system, other, 0.0)
        @test J * u ≈ du rtol = 1e-12
        @test (@allocated rhs!(du, u, system, 0.0)) == 0
    end

    @testset "four-point and limiter schemes below the interface ($scheme, $mapping)" for scheme in (:quick, :tcdf),
        mapping in (:overlap, :dream)
        # the interface flux takes upwind where the lower operator has a scheme the interface
        # does not, and the two grids keep the particles
        lower = random_coefficients!(AdvectionDiffusion(hot; scheme, p_upper_boundary = :interface), rng)
        upper = random_coefficients!(AdvectionDiffusion(runaway; scheme, p_lower_boundary = :interface), rng)
        upper.advection_p[end, :, :] .= 0   # close the outer p boundary
        pair = two_grid_system(lower, upper; pitch_mapping = mapping)
        @test pair.interface.scheme === :upwind
        state = rand(rng, length(pair.layout))
        change = zeros(length(pair.layout))
        rhs!(change, state, pair.system, 0.0)
        total = kinetic_content(pair.layout, :hot_tail_distribution, change) +
                kinetic_content(pair.layout, :runaway_distribution, change)
        scale = kinetic_content(pair.layout, :hot_tail_distribution, abs.(change)) +
                kinetic_content(pair.layout, :runaway_distribution, abs.(change))
        @test abs(total) < 1e-13 * scale
        # the interface alone moves the flux through the face
        moved = EquationSystem(pair.layout,
            [Equation(:hot_tail_distribution, TimeDerivative(), pair.interface),
                Equation(:runaway_distribution, TimeDerivative(), pair.interface)])
        rhs!(change, state, moved, 0.0)
        through_face = sum(interface_flux(pair.interface, state, pair.layout, ir) *
                           rad.spatial_jacobian[ir] *
                           rad.r_widths[ir] for ir in 1:2)
        @test kinetic_content(pair.layout, :runaway_distribution, change) ≈ through_face rtol = 1e-13
    end

    @testset "the bare operator of a prescribed term" begin
        # an interface that reads the operator of a prescribed term in place of the term would
        # take the coefficients of its last rebuild: the system refuses it, and the update of
        # the radial transport finds a prescribed term behind an interface
        plasma = PlasmaState(cold_electron_temperature = fill(100.0, 2),
            ions = [IonSpecies("D", 1, 1, fill(1e19, 2))],
            electric_field = zeros(2), magnetic_field = fill(5.0, 2))
        settings = CollisionSettings(mode = :superthermal, screening = :complete)
        prescribed = PrescribedFokkerPlanckTerm(:hot_tail_distribution,
            AdvectionDiffusion(hot; p_upper_boundary = :interface), t -> plasma;
            settings)
        upper = fokker_planck_operator(runaway, plasma, settings; p_lower_boundary = :interface)
        layout = UnknownLayout(rad,
            [KineticUnknown(:hot_tail_distribution, hot),
                KineticUnknown(:runaway_distribution, runaway)])
        system_with(lower_term) = EquationSystem(layout,
            [
                Equation(:hot_tail_distribution, TimeDerivative(), prescribed,
                    MomentumInterfaceTerm(:hot_tail_distribution, :runaway_distribution, lower_term, upper)),
                Equation(:runaway_distribution, TimeDerivative(),
                    FokkerPlanckTerm(:runaway_distribution, upper))])
        @test_throws "bare operator" system_with(prescribed.operator)
        @test system_with(prescribed) isa EquationSystem
        # the interface reaches the prescribed term, which is not a term of the equations here
        behind = EquationSystem(layout,
            [
                Equation(:hot_tail_distribution, TimeDerivative(),
                    MomentumInterfaceTerm(:hot_tail_distribution, :runaway_distribution, prescribed, upper)),
                Equation(:runaway_distribution, TimeDerivative(),
                    FokkerPlanckTerm(:runaway_distribution, upper))])
        update = RechesterRosenbluthUpdate(prescribed.operator, t -> 1e-3; major_radius = 1.0)
        @test_throws "rebuild clears it" Runes2P.check_update(update, behind)
    end

    @testset "diffusion across the interface" begin
        # a constant is steady. With the overlap the couplings between cells are
        # non-negative. DREAM's mapping interpolates the runaway values in the pitch and gives
        # the flux to the runaway cells by their overlap, so a runaway cell is coupled to its
        # neighbours in the pitch with negative weights where the pitch grids differ.
        for mapping in (:overlap, :dream), scheme in (:central, :exponential_fitting)

            hot_op = AdvectionDiffusion(hot; scheme, p_upper_boundary = :interface)
            runaway_op = AdvectionDiffusion(runaway; scheme, p_lower_boundary = :interface)
            hot_op.diffusion_p_p .= rand(rng, size(hot_op.diffusion_p_p))
            runaway_op.diffusion_p_p[2:(end - 1), :, :] .= 0.3
            case = two_grid_system(hot_op, runaway_op; pitch_mapping = mapping)
            du = zeros(length(case.layout))
            rhs!(du, fill(1.5, length(case.layout)), case.system, 0.0)
            @test maximum(abs, du) < 1e-12
            J = jacobian_prototype(case.system)
            jacobian!(J, case.system, zeros(length(case.layout)), 0.0)
            rows, columns, values = findnz(J)
            negative = [(rows[k], columns[k])
                        for k in eachindex(values)
                        if rows[k] != columns[k] && values[k] < 0]
            if mapping === :overlap
                @test isempty(negative)
            else
                runaway_id = unknown_id(case.layout, :runaway_distribution)
                first_cells = reduce(vcat,
                    [[Runes2P._kinetic_index(case.layout, runaway_id, ir, 1, J)
                      for J in 1:20]
                     for ir in 1:2])
                @test !isempty(negative)
                @test all(pair -> pair[1] in first_cells && pair[2] in first_cells, negative)
            end
        end
        # with one pitch grid on both sides DREAM's mapping has no negative couplings
        same_pitch = runaway_grid(hot, uniform_p_faces(5, 2.5, 12.0)[2:end], hot_pitch)
        hot_op = AdvectionDiffusion(hot; scheme = :central, p_upper_boundary = :interface)
        hot_op.diffusion_p_p .= rand(rng, size(hot_op.diffusion_p_p))
        case = two_grid_system(
            hot_op, AdvectionDiffusion(same_pitch; p_lower_boundary = :interface);
            pitch_mapping = :dream)
        J = jacobian_prototype(case.system)
        jacobian!(J, case.system, zeros(length(case.layout)), 0.0)
        rows, columns, values = findnz(J)
        @test all(values[k] >= 0 for k in eachindex(values) if rows[k] != columns[k])
        # exponential fitting keeps the sign with advection of either direction
        hot_op = random_coefficients!(
            AdvectionDiffusion(hot; scheme = :exponential_fitting,
                p_upper_boundary = :interface),
            rng)
        runaway_op = AdvectionDiffusion(runaway; scheme = :exponential_fitting, p_lower_boundary = :interface)
        hot_op.advection_xi .= 0
        hot_op.advection_r .= 0
        case = two_grid_system(hot_op, runaway_op)
        J = jacobian_prototype(case.system)
        jacobian!(J, case.system, zeros(length(case.layout)), 0.0)
        rows, columns, values = findnz(J)
        @test all(values[k] >= 0 for k in eachindex(values) if rows[k] != columns[k])
    end

    @testset "pitch interpolation of DREAM" begin
        upper = [-0.8, -0.2, 0.3, 0.9]
        lower = [-0.9, -0.5, -0.2, 0.6, 0.95]
        cells, weights = Runes2P._dream_pitch_interpolation(lower, upper)
        # below the first upper centre: the first cell. Between two centres: linear. On a
        # centre: that cell. Above the last upper centre: DREAM takes the cell before the last
        @test cells == [1, 2, 2, 4, 4]
        @test weights ≈ [1.0, 0.5, 1.0, 0.5, 0.0]
        hot_op = AdvectionDiffusion(hot; p_upper_boundary = :interface)
        runaway_op = AdvectionDiffusion(runaway; p_lower_boundary = :interface)
        term = MomentumInterfaceTerm(:a, :b, hot_op, runaway_op; pitch_mapping = :dream)
        f_upper = reshape(collect(1.0:20.0), 1, 20)
        hot_centers, runaway_centers = hot_momentum.xi_centers,
        momentum_grid(runaway).xi_centers
        for j in 1:9
            value = Runes2P._dream_upper_value(term, f_upper, j, 1)
            if hot_centers[j] <= runaway_centers[1]
                @test value == 1.0
            elseif hot_centers[j] > runaway_centers[end]
                @test value == 19.0
            else
                k = findfirst(>=(hot_centers[j]), runaway_centers)
                expected = k - 1 +
                           (hot_centers[j] - runaway_centers[k - 1]) /
                           (runaway_centers[k] - runaway_centers[k - 1])
                @test value ≈ expected rtol = 1e-14
            end
        end
    end

    @testset "advection from the hot-tail to the runaway grid" begin
        # with a positive advection alone both mappings carry the upwind flux of the hot-tail
        # cells, Σ_j 𝒱' Δξ_j Aᵖ_j f_j, and every runaway cell gets its share by the overlap
        hot_op = AdvectionDiffusion(hot; p_upper_boundary = :interface)
        runaway_op = AdvectionDiffusion(runaway; p_lower_boundary = :interface)
        hot_op.advection_p[end, :, :] .= 1 .+ rand(rng, 9, 2)
        cases = map((:overlap, :dream)) do mapping
            case = two_grid_system(hot_op, runaway_op; pitch_mapping = mapping)
            u = rand(rng, length(case.layout))
            hot_id = unknown_id(case.layout, :hot_tail_distribution)
            for ir in 1:2
                f = kinetic_view(u, case.layout, hot_id, ir)
                V = phase_space_jacobian_p_faces(hot, ir)
                expected = sum(V[end, j] * hot_momentum.xi_widths[j] *
                               hot_op.advection_p[end, j, ir] * f[end, j]
                for j in 1:9) / rad.spatial_jacobian[ir]
                @test interface_flux(case.interface, u, case.layout, ir) ≈ expected rtol = 1e-13
            end
            case
        end
        # the same state gives the same rates with both mappings
        case_overlap, case_dream = cases
        u = rand(rng, length(case_overlap.layout))
        du_overlap, du_dream = zeros(length(u)), zeros(length(u))
        rhs!(du_overlap, u, case_overlap.system, 0.0)
        rhs!(du_dream, u, case_dream.system, 0.0)
        @test du_overlap ≈ du_dream rtol = 1e-13
    end

    @testset "settings record" begin
        hot_op = AdvectionDiffusion(hot; p_upper_boundary = :interface)
        runaway_op = AdvectionDiffusion(runaway; p_lower_boundary = :interface)
        case = two_grid_system(hot_op, runaway_op; pitch_mapping = :dream)
        fields = settings_fields(MomentumInterfaceTerm)
        @test isempty(intersect(fields.described, fields.derived))
        @test Set(union(fields.described, fields.derived)) ==
              Set(fieldnames(MomentumInterfaceTerm))
        record = describe(case.interface)
        @test record["type"] == "MomentumInterfaceTerm"
        @test record["lower"] == "hot_tail_distribution" &&
              record["upper"] == "runaway_distribution"
        @test record["pitch_mapping"] == "dream" && record["scheme"] == "upwind"
        @test !haskey(record, "described")
        @test record["lower_grid"] != record["upper_grid"]
    end
end
