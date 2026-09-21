using Random

function _conservation_residual(g, df)
    s = 0.0
    for ir in 1:nr(g), j in 1:nxi(g), i in 1:np(g)
        s += g.Vp[i, j, ir] * g.mom.dp[i] * g.mom.dxi[j] * g.rad.dr[ir] * df[i, j, ir]
    end
    return s
end

@testset "advection-diffusion" begin
    rng = Xoshiro(1)
    mom = PXiGrid(biuniform_p_faces(20, 8, 0.0, 0.5, 3.0), uniform_xi_faces(10))
    rad = RadialGrid(uniform_r_faces(6, 1.0))
    g = cylindrical_grid(mom, rad)

    @testset "exact conservation with random coefficients ($scheme)" for scheme in (:upwind, :central)
        op = AdvectionDiffusion(g; scheme)
        for A in (op.Ap, op.Axi, op.Ar, op.Dpxi, op.Dxip)
            A .= randn(rng, size(A))
        end
        for D in (op.Dpp, op.Dxixi, op.Drr)
            D .= rand(rng, size(D))
        end
        op.Ap[end, :, :] .= 0  # close the outer p boundary
        f = rand(rng, size(g)...)
        df = zeros(size(g))
        apply!(df, op, f)
        scale = sum(abs, g.Vp .* df) * maximum(mom.dp) * maximum(rad.dr)
        @test abs(_conservation_residual(g, df)) < 1e-13 * scale
    end

    @testset "isotropic f is steady under pitch-angle scattering" begin
        op = AdvectionDiffusion(g)
        for k in 1:nxi(g)+1, i in 1:np(g)
            op.Dxixi[i, k, :] .= (1 - mom.xi_f[k]^2) / 2
        end
        f = [exp(-mom.p[i]) for i in 1:np(g), _ in 1:nxi(g), _ in 1:nr(g)]
        df = zeros(size(g))
        apply!(df, op, f)
        @test maximum(abs, df) < 1e-14
    end

    @testset "Maxwellian steady state converges at second order (central)" begin
        θ = 0.1
        errs = Float64[]
        for n in (40, 80, 160)
            gm = cylindrical_grid(PXiGrid(uniform_p_faces(n, 0.0, 2.0), uniform_xi_faces(2)),
                                  RadialGrid(uniform_r_faces(1, 1.0)))
            op = AdvectionDiffusion(gm; scheme=:central)
            op.Dpp .= 1
            for k in 1:n+1
                op.Ap[k, :, :] .= -gm.mom.p_f[k] / θ  # Γ = 0 for f ∝ exp(-p²/2θ)
            end
            op.Ap[end, :, :] .= 0
            op.Dpp[end, :, :] .= 0
            f = [exp(-gm.mom.p[i]^2 / (2θ)) for i in 1:n, _ in 1:2, _ in 1:1]
            df = zeros(size(gm))
            apply!(df, op, f)
            push!(errs, maximum(abs, df[1:n÷2, :, :]))
        end
        rates = log2.(errs[1:end-1] ./ errs[2:end])
        @test all(rates .> 1.8)
    end

    @testset "absorbing radial wall removes particles" begin
        op = AdvectionDiffusion(g; rbc=:absorbing)
        op.Drr .= 0.3
        f = ones(size(g))
        df = zeros(size(g))
        apply!(df, op, f)
        # only the outermost cell changes; loss equals the diffusive wall flux
        @test maximum(abs, df[:, :, 1:end-1]) < 1e-14
        loss = _conservation_residual(g, df)
        wall = sum(g.Vp_fr[i, j, end] * mom.dp[i] * mom.dxi[j] * 0.3 / (rad.r_f[end] - rad.r[end])
                   for i in 1:np(g), j in 1:nxi(g))
        @test loss ≈ -wall
    end
end
