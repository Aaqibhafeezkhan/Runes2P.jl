using Random

function _conservation_residual(g, df)
    s = 0.0
    for ir in 1:radial_cell_count(g), j in 1:pitch_cell_count(g), i in 1:momentum_cell_count(g)
        s += phase_space_jacobian(g, ir)[i, j] * momentum_grid(g).p_widths[i] *
             momentum_grid(g).xi_widths[j] * g.radial_grid.r_widths[ir] * df[i, j, ir]
    end
    return s
end

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
        for k in 1:pitch_cell_count(g)+1, i in 1:momentum_cell_count(g)
            op.diffusion_xi_xi[i, k, :] .= (1 - mom.xi_faces[k]^2) / 2
        end
        f = [exp(-mom.p_centers[i]) for i in 1:momentum_cell_count(g), _ in 1:pitch_cell_count(g),
                                        _ in 1:radial_cell_count(g)]
        df = zeros(size(g))
        apply!(df, op, f)
        @test maximum(abs, df) < 1e-14
    end

    @testset "Maxwellian steady state converges at second order (central)" begin
        θ = 0.1
        errs = Float64[]
        for n in (40, 80, 160)
            gm = cylindrical_grid(MomentumGrid(uniform_p_faces(n, 0.0, 2.0), uniform_xi_faces(2)),
                                  RadialGrid(uniform_r_faces(1, 1.0)))
            op = AdvectionDiffusion(gm; scheme=:central)
            op.diffusion_p_p .= 1
            for k in 1:n+1
                op.advection_p[k, :, :] .= -momentum_grid(gm).p_faces[k] / θ  # Γ = 0 for f ∝ exp(-p²/2θ)
            end
            op.advection_p[end, :, :] .= 0
            op.diffusion_p_p[end, :, :] .= 0
            f = [exp(-momentum_grid(gm).p_centers[i]^2 / (2θ)) for i in 1:n, _ in 1:2, _ in 1:1]
            df = zeros(size(gm))
            apply!(df, op, f)
            push!(errs, maximum(abs, df[1:n÷2, :, :]))
        end
        rates = log2.(errs[1:end-1] ./ errs[2:end])
        @test all(rates .> 1.8)
    end

    @testset "absorbing radial wall removes particles" begin
        op = AdvectionDiffusion(g; r_boundary=:absorbing)
        op.diffusion_r_r .= 0.3
        f = ones(size(g))
        df = zeros(size(g))
        apply!(df, op, f)
        # only the outermost cell changes; loss equals the diffusive wall flux
        @test maximum(abs, df[:, :, 1:end-1]) < 1e-14
        loss = _conservation_residual(g, df)
        wall = sum(phase_space_jacobian_r_faces(g)[i, j, end] * mom.p_widths[i] * mom.xi_widths[j] * 0.3 /
                   (rad.r_faces[end] - rad.r_centers[end])
                   for i in 1:momentum_cell_count(g), j in 1:pitch_cell_count(g))
        @test loss ≈ -wall
    end
end
