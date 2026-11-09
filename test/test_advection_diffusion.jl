using ForwardDiff
using Random

@testset "advection-diffusion" begin
    rng = Xoshiro(1)
    mom = MomentumGrid(biuniform_p_faces(20, 8, 0.0, 0.5, 3.0), uniform_xi_faces(10))
    rad = RadialGrid(uniform_r_faces(6, 1.0))
    g = cylindrical_grid(mom, rad)

    @testset "exact conservation with random coefficients ($scheme)" for scheme in (:upwind, :central)
        op = AdvectionDiffusion(g; scheme)
        for A in (op.advection_p, op.advection_xi, op.advection_r, op.diffusion_p_xi, op.diffusion_xi_p)
            A .= randn(rng, size(A))
        end
        for D in (op.diffusion_p_p, op.diffusion_xi_xi, op.diffusion_r_r)
            D .= rand(rng, size(D))
        end
        op.advection_p[end, :, :] .= 0  # close the outer p boundary
        f = rand(rng, size(g)...)
        df = zeros(size(g))
        apply!(df, op, f)
        jacobian = stack(phase_space_jacobian(g, ir) for ir in 1:radial_cell_count(g))
        scale = sum(abs, jacobian .* df) * maximum(mom.p_widths) * maximum(rad.r_widths)
        @test abs(_conservation_residual(g, df)) < 1e-13 * scale
    end

    @testset "isotropic f is steady under pitch-angle scattering" begin
        op = AdvectionDiffusion(g)
        for k in 1:(pitch_cell_count(g) + 1), i in 1:momentum_cell_count(g)

            op.diffusion_xi_xi[i, k, :] .= (1 - mom.xi_faces[k]^2) / 2
        end
        f = [exp(-mom.p_centers[i])
             for i in 1:momentum_cell_count(g), _ in 1:pitch_cell_count(g),
        _ in 1:radial_cell_count(g)]
        df = zeros(size(g))
        apply!(df, op, f)
        @test maximum(abs, df) < 1e-14
    end

    @testset "Maxwellian steady state converges at second order (central)" begin
        θ = 0.1
        errs = Float64[]
        for n in (40, 80, 160)
            gm = cylindrical_grid(
                MomentumGrid(uniform_p_faces(n, 0.0, 2.0), uniform_xi_faces(2)),
                RadialGrid(uniform_r_faces(1, 1.0)))
            op = AdvectionDiffusion(gm; scheme = :central)
            op.diffusion_p_p .= 1
            for k in 1:(n + 1)
                op.advection_p[k, :, :] .= -momentum_grid(gm).p_faces[k] / θ  # Γ = 0 for f ∝ exp(-p²/2θ)
            end
            op.advection_p[end, :, :] .= 0
            op.diffusion_p_p[end, :, :] .= 0
            f = [exp(-momentum_grid(gm).p_centers[i]^2 / (2θ))
                 for i in 1:n, _ in 1:2, _ in 1:1]
            df = zeros(size(gm))
            apply!(df, op, f)
            push!(errs, maximum(abs, df[1:(n ÷ 2), :, :]))
        end
        rates = log2.(errs[1:(end - 1)] ./ errs[2:end])
        @test all(rates .> 1.8)
    end

    @testset "absorbing radial wall removes particles" begin
        op = AdvectionDiffusion(g; r_boundary = :absorbing)
        op.diffusion_r_r .= 0.3
        f = ones(size(g))
        df = zeros(size(g))
        apply!(df, op, f)
        # only the outermost cell changes; loss equals the diffusive wall flux
        @test maximum(abs, df[:, :, 1:(end - 1)]) < 1e-14
        loss = _conservation_residual(g, df)
        wall = sum(phase_space_jacobian_r_faces(g)[i, j, end] * mom.p_widths[i] *
                   mom.xi_widths[j] * 0.3 /
                   (rad.r_faces[end] - rad.r_centers[end])
        for i in 1:momentum_cell_count(g), j in 1:pitch_cell_count(g))
        @test loss ≈ -wall
    end

    @testset "no flux through p = 0 (p_zero_boundary = :zero_flux, $scheme)" for scheme in (:upwind, :quick)
        gz = cylindrical_grid(
            MomentumGrid(uniform_p_faces(6, 0.0, 1.5), uniform_xi_faces(3)),
            RadialGrid(uniform_r_faces(1, 1.0)))
        mz = momentum_grid(gz)
        np = momentum_cell_count(gz)
        function random_operator(boundary)
            op = AdvectionDiffusion(gz; scheme, p_zero_boundary = boundary)
            rz = Xoshiro(7)
            op.advection_p .= randn(rz, size(op.advection_p))
            op.advection_xi .= randn(rz, size(op.advection_xi))
            op.diffusion_p_p .= rand(rz, size(op.diffusion_p_p))
            op.diffusion_xi_xi .= rand(rz, size(op.diffusion_xi_xi))
            op.advection_p[end, :, :] .= 1.0         # outflow through p_max
            op.advection_p_times_p2_at_zero .= -0.7  # the friction's limit at p = 0
            return op
        end
        sink, closed = random_operator(:outflow), random_operator(:zero_flux)
        f = rand(Xoshiro(8), size(gz)...)
        df_sink, df_closed = zeros(size(gz)), zeros(size(gz))
        apply!(df_sink, sink, f)
        apply!(df_closed, closed, f)

        # no particles leave through p = 0: the density changes by the flux through p_max alone
        @test momentum_flux(closed, f, 1) == 0
        outflow = momentum_flux(closed, f, np + 1)
        @test outflow > 0
        @test density(gz, df_closed)[1] ≈ -outflow rtol = 1e-13
        # the default keeps the flux (𝒱'/p²)|₀ (p²Aᵖ)|₀ f₁ through p = 0 (f₁ for upwind, the
        # scheme's own face value for QUICK)
        @test momentum_flux(sink, f, 1) < 0
        @test density(gz, df_sink)[1] ≈
              momentum_flux(sink, f, 1) - momentum_flux(sink, f, np + 1) rtol = 1e-13
        if scheme === :upwind
            # the two operators differ in the first momentum cell only
            @test df_sink[2:end, :, :] == df_closed[2:end, :, :]
        else
            # QUICK: the ghost cells below p = 0 hold zero under :zero_flux (DREAM's Dirichlet
            # condition) and are mirrored by default, which changes the face between cells 1 and 2
            @test df_sink[4:end, :, :] == df_closed[4:end, :, :]
            @test df_sink[2, :, :] != df_closed[2, :, :]
        end

        # the Jacobian follows: the stored matrix against central differences on the column
        # of f₁, and the column of the default differs by the flux through p = 0 alone
        M_closed, M_sink = operator_matrix(closed), operator_matrix(sink)
        @test M_closed * vec(f) ≈ vec(df_closed) rtol = 1e-13
        h = 1e-6
        for j in 1:3
            column = LinearIndices(size(gz))[1, j, 1]
            up, down = copy(f), copy(f)
            up[1, j, 1] += h
            down[1, j, 1] -= h
            df_up, df_down = zeros(size(gz)), zeros(size(gz))
            apply!(df_up, closed, up)
            apply!(df_down, closed, down)
            central = vec(df_up .- df_down) ./ (2h)
            @test Vector(M_closed[:, column])≈central rtol=1e-8 atol=1e-8*maximum(abs, central)
            difference = Vector(M_sink[:, column] .- M_closed[:, column])
            if scheme === :upwind
                # the sink of cell (1, j): -(𝒱'/p²)|₀ (p²Aᵖ)|₀ / (𝒱'₁ Δp₁)
                sink_rate = 0.7 * phase_space_jacobian_over_p2_at_zero(gz, 1)[j] /
                            (phase_space_jacobian(gz, 1)[1, j] * mz.p_widths[1])
                @test difference[column] ≈ -sink_rate rtol = 1e-13
                difference[column] = 0
                @test all(iszero, difference)
            end
        end
    end

    @testset "p_zero_boundary needs p_min = 0" begin
        lifted = cylindrical_grid(
            MomentumGrid(uniform_p_faces(4, 0.5, 1.5), uniform_xi_faces(2)),
            RadialGrid(uniform_r_faces(1, 1.0)))
        @test_throws ArgumentError AdvectionDiffusion(lifted; p_zero_boundary = :zero_flux)
        @test_throws ArgumentError AdvectionDiffusion(g; p_zero_boundary = :mirrored)
        @test AdvectionDiffusion(g).p_zero_boundary === :outflow
    end

    @testset "p_lower_boundary needs p_min > 0" begin
        for boundary in (:prescribed, :interface)
            @test_throws "p_zero_boundary" AdvectionDiffusion(g; p_lower_boundary = boundary)
        end
        @test AdvectionDiffusion(g; p_lower_boundary = :zero_flux).p_lower_boundary ===
              :zero_flux
    end
end

# apply! with every face flux evaluated from both of its cells, the reference of the loop
# of the operator, which evaluates each p face of a pitch row once
function _apply_from_both_sides!(df, op, f; scale = true, limiter_state = nothing,
        electric_field = nothing)
    mode = Runes2P._radial_flux_mode(op)
    rg = op.grid.radial_grid
    for ir in 1:radial_cell_count(op.grid)
        m = op.momentum_grids[ir]
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)

            V = op.jacobian_centers[i, j, ir]
            V == 0 && continue
            divergence = (Runes2P._p_flux(
                             op, f, i + 1, j, ir, limiter_state, electric_field) -
                          Runes2P._p_flux(op, f, i, j, ir, limiter_state, electric_field)) /
                         m.p_widths[i] +
                         (Runes2P._xi_flux(
                             op, f, i, j + 1, ir, limiter_state, electric_field) -
                          Runes2P._xi_flux(
                             op, f, i, j, ir, limiter_state, electric_field)) /
                         m.xi_widths[j] +
                         (Runes2P._r_flux(op, f, i, j, ir, ir + 1, limiter_state, mode) -
                          Runes2P._r_flux(op, f, i, j, ir, ir, limiter_state, mode)) /
                         rg.r_widths[ir]
            df[i, j, ir] -= scale * divergence / V
        end
    end
    return df
end

const BOTH_SIDES_COEFFICIENTS = (
    :advection_p, :diffusion_p_p, :diffusion_p_xi, :advection_xi, :diffusion_xi_xi,
    :diffusion_xi_p, :advection_r, :diffusion_r_r, :unit_field_advection_p,
    :unit_field_advection_xi, :p_flux_at_lower_boundary, :advection_p_times_p2_at_zero)

# apply! from `df` against the reference, bit for bit; `label` goes into `failures` when
# they differ
function _check_both_sides!(failures, label, df, op, f; keywords...)
    carried = apply!(copy(df), op, f; keywords...)
    reference = _apply_from_both_sides!(copy(df), op, f; keywords...)
    if !isequal(carried, reference)
        push!(failures, label)
    end
    return failures
end

@testset "each p face once: the fluxes evaluated from both cells, bit for bit" begin
    rng = Xoshiro(7)
    rad = RadialGrid(uniform_r_faces(3, 1.0))
    from_zero = MomentumGrid(uniform_p_faces(6, 0.0, 2.0), uniform_xi_faces(4))
    shifted = vcat(-1.0, uniform_xi_faces(4)[2:(end - 1)] .+ 0.1, 1.0)
    configurations = [
        (cylindrical_grid(from_zero, rad), (; p_zero_boundary = :outflow)),
        (cylindrical_grid(from_zero, rad),
            (; p_zero_boundary = :zero_flux, r_scheme = :quick, r_boundary = :absorbing)),
        (
            cylindrical_grid(MomentumGrid(uniform_p_faces(6, 0.5, 2.0), uniform_xi_faces(4)), rad),
            (; p_lower_boundary = :prescribed, r_boundary = :absorbing_dream)),
        (
            cylindrical_grid(
                [MomentumGrid(uniform_p_faces(6, 0.0, 2.0), faces)
                 for faces in (uniform_xi_faces(4), shifted, uniform_xi_faces(4))], rad),
            (;))]
    schemes = (
        :upwind, :central, :exponential_fitting, :quick, :upwind_second_order, :tcdf,
        :muscl, :smart, :ospre, :van_albada)
    upper_boundaries = (:outflow, :f_zero, :constant_flux, :extrapolated_flux,
        :extrapolated_flux_consistent, :interface)
    K = 4
    P = 25
    failures = String[]
    for (grid, options) in configurations, scheme in schemes, upper in upper_boundaries
        scheme === :exponential_fitting && upper === :extrapolated_flux && continue
        op = AdvectionDiffusion(grid; scheme, p_upper_boundary = upper, options...)
        for name in BOTH_SIDES_COEFFICIENTS
            coefficients = getfield(op, name)
            coefficients .= randn(rng, size(coefficients))
        end
        for D in (op.diffusion_p_p, op.diffusion_xi_xi, op.diffusion_r_r)
            D .= rand(rng, size(D))
        end
        # cells and a p face with no phase-space volume, at both ends of a pitch row and
        # inside one: the flux carried past such a cell is read by the next
        op.jacobian_centers[1, 2, 1] = 0
        op.jacobian_centers[6, 3, 2] = 0
        op.jacobian_centers[3, 1, 3] = 0
        op.jacobian_p_faces[4, 2, 2] = 0
        f = 0.1 .+ rand(rng, size(grid)...)
        state = 0.1 .+ rand(rng, size(grid)...)
        field = randn(rng, radial_cell_count(grid))
        df = randn(rng, size(grid)...)
        label = "$scheme, p_max $upper, $options"
        _check_both_sides!(failures, "Float64, $label", df, op, f)
        _check_both_sides!(failures, "Float64 scaled, frozen, field, $label", df, op, f;
            scale = -0.7, limiter_state = state, electric_field = field)
        # a dual state and a dual field (the colored Jacobian, the field response)
        dual_f = [ForwardDiff.Dual{Nothing}(x, ntuple(_ -> randn(rng), K)...) for x in f]
        dual_field = [ForwardDiff.Dual{Nothing}(x, ntuple(_ -> randn(rng), K)...)
                      for x in field]
        dual_df = zeros(eltype(dual_f), size(grid))
        _check_both_sides!(failures, "dual f, $label", dual_df, op, dual_f)
        _check_both_sides!(failures, "dual f scaled, frozen, dual field, $label", dual_df,
            op, dual_f; scale = 2.5, limiter_state = state, electric_field = dual_field)
        # dual coefficients on a Float64 state (the plasma columns)
        C = ForwardDiff.Dual{Nothing, Float64, P}
        dual_op = Runes2P.similar_coefficients(op, C)
        for name in BOTH_SIDES_COEFFICIENTS
            dual_coefficients = getfield(dual_op, name)
            for (k, x) in pairs(getfield(op, name))
                dual_coefficients[k] = C(x, ForwardDiff.Partials(ntuple(_ -> randn(rng), P)))
            end
        end
        coefficient_df = zeros(C, size(grid))
        _check_both_sides!(
            failures, "dual coefficients, $label", coefficient_df, dual_op, f)
        _check_both_sides!(failures, "dual coefficients frozen, field, $label",
            coefficient_df, dual_op, f; limiter_state = state, electric_field = field)
    end
    @test isempty(failures)
end

@testset "each p face once: allocation free for Float64 and dual states" begin
    rng = Xoshiro(8)
    grid = cylindrical_grid(
        MomentumGrid(uniform_p_faces(6, 0.0, 2.0), uniform_xi_faces(4)),
        RadialGrid(uniform_r_faces(3, 1.0)))
    op = AdvectionDiffusion(grid; scheme = :tcdf, p_upper_boundary = :extrapolated_flux)
    f = 0.1 .+ rand(rng, size(grid)...)
    df = zeros(size(grid))
    apply!(df, op, f; limiter_state = f)
    @test (@allocated apply!(df, op, f; limiter_state = f)) == 0
    dual_f = [ForwardDiff.Dual{Nothing}(x, ntuple(_ -> randn(rng), 4)...) for x in f]
    dual_df = zeros(eltype(dual_f), size(grid))
    apply!(dual_df, op, dual_f)
    @test (@allocated apply!(dual_df, op, dual_f)) == 0
end
