# Order of accuracy of the advection-diffusion operator by manufactured solutions,
# independent of DREAM: one backward-Euler step of unit length,
# (I - L) f = f_exact - L_exact f_exact, on uniform, biuniform and tanh momentum grids,
# with every coefficient of the p and ξ fluxes set, including the p-ξ cross terms. The
# biuniform grid has the shape of DREAM's avalanche/generateCylindrical.py (a fifth of the
# cells below p = 0.7, p_max = 30, a spacing jump of 10.5). The unit sink makes the
# problem well posed: the steady operator with zero flux at p = 0 is nearly singular.
# Nonuniform pitch grids (the example's geometric faces, DREAM's uniform-θ and biuniform
# types) are tested in the pitch direction alone as well.

using LinearAlgebra: I

module ManufacturedSolution
# bulk resolved by the fine region below p = 0.7, tail over the coarse region; the tail is
# 1e-15 at p = 30, so the f = 0 ghost at p_max is consistent with it
g(p) = exp(-p^2 / 0.08) + 1e-2 * exp(-p)
dg(p) = -2p / 0.08 * exp(-p^2 / 0.08) - 1e-2 * exp(-p)
h(xi) = 1 + 0.3xi + 0.2 * (3xi^2 - 1) / 2
dh(xi) = 0.3 + 0.6xi
f(p, xi) = g(p) * h(xi)
f_p(p, xi) = dg(p) * h(xi)
f_xi(p, xi) = g(p) * dh(xi)
# coefficients; the ξ fluxes vanish at ξ = ±1, as the operator's boundary condition requires
advection_p(p, xi) = -(1 + 0.1p) * (1 + 0.2xi)
diffusion_p_p(p, xi) = 0.5 + 0.1p
cross(p, xi) = 0.3 * (1 - xi^2)
advection_xi(p, xi) = 0.5 * (1 - xi^2)
diffusion_xi_xi(p, xi) = 0.4 * (1 - xi^2)
flux_p(p, xi) = advection_p(p, xi) * f(p, xi) - diffusion_p_p(p, xi) * f_p(p, xi) - cross(p, xi) * f_xi(p, xi)
flux_xi(p, xi) = advection_xi(p, xi) * f(p, xi) - diffusion_xi_xi(p, xi) * f_xi(p, xi) - cross(p, xi) * f_p(p, xi)
end

# the same coefficients with a beam-like pitch profile, peaked at ξ = 1 where runaways
# gather and where the pitch grids of DREAM's examples are finest; a quadratic profile
# would be treated too well by the centred differences in ξ
module BeamSolution
using ..ManufacturedSolution: g, dg, advection_p, diffusion_p_p, cross, advection_xi, diffusion_xi_xi
h(xi) = 1 + 0.3xi + 0.2 * exp(10 * (xi - 1))
dh(xi) = 0.3 + 2.0 * exp(10 * (xi - 1))
f(p, xi) = g(p) * h(xi)
f_p(p, xi) = dg(p) * h(xi)
f_xi(p, xi) = g(p) * dh(xi)
flux_p(p, xi) = advection_p(p, xi) * f(p, xi) - diffusion_p_p(p, xi) * f_p(p, xi) - cross(p, xi) * f_xi(p, xi)
flux_xi(p, xi) = advection_xi(p, xi) * f(p, xi) - diffusion_xi_xi(p, xi) * f_xi(p, xi) - cross(p, xi) * f_p(p, xi)
end

# pitch faces: those of avalanche/generateCylindrical.py (geometric growth with exponent
# 1.5, finest at ξ = 1), and DREAM's pitch grid types uniform in θ and biuniform in ξ
# (half of the cells above ξ = 0.6, a split without an example)
geometric_xi_faces(cell_count) = reverse(1 .- 2 .* range(0, 1; length=cell_count + 1) .^ 1.5)
uniform_theta_xi_faces(cell_count) = cos.(range(pi, 0; length=cell_count + 1))
biuniform_xi_faces(cell_count) = vcat(collect(range(-1.0, 0.6; length=cell_count - cell_count ÷ 2 + 1)),
                                      collect(range(0.6, 1.0; length=cell_count ÷ 2 + 1))[2:end])

# errors of the step on a grid, and the local truncation error L f_exact - L_exact f_exact
# next to the separation p = 0.7 and in the bulk below it
function manufactured_errors(p_faces, pitch_cells; scheme=:central, xi_faces=uniform_xi_faces(pitch_cells))
    step = manufactured_step(p_faces, xi_faces; scheme)
    error = step.solution .- step.f_exact
    separation = findfirst(>=(0.7 - 1e-12), p_faces)
    return (; max=maximum(abs, error),
            l1=sum(abs.(error) .* step.weights) / sum(abs.(step.f_exact) .* step.weights),
            truncation_at_separation=maximum(abs, step.truncation[separation-2:separation+1, :]),
            truncation_in_bulk=maximum(abs, step.truncation[separation÷2-1:separation÷2+1, :]))
end

# the manufactured step on a grid: discrete solution, exact solution at the cell centres,
# local truncation error and cell volumes
function manufactured_step(p_faces, xi_faces; scheme=:central, ms=ManufacturedSolution)
    mom = MomentumGrid(p_faces, xi_faces)
    pitch_cells = pitch_cell_count(mom)
    grid = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
    op = AdvectionDiffusion(grid; scheme, p_upper_boundary=:f_zero)
    pc, pf, xc, xf = mom.p_centers, mom.p_faces, mom.xi_centers, mom.xi_faces
    np = length(pc)
    for j in 1:pitch_cells, k in 1:np+1
        op.advection_p[k, j, 1] = ms.advection_p(pf[k], xc[j])
        op.diffusion_p_p[k, j, 1] = ms.diffusion_p_p(pf[k], xc[j])
        op.diffusion_p_xi[k, j, 1] = ms.cross(pf[k], xc[j])
    end
    for k in 1:pitch_cells+1, i in 1:np
        op.advection_xi[i, k, 1] = ms.advection_xi(pc[i], xf[k])
        op.diffusion_xi_xi[i, k, 1] = ms.diffusion_xi_xi(pc[i], xf[k])
        op.diffusion_xi_p[i, k, 1] = ms.cross(pc[i], xf[k])
    end
    L = operator_matrix(op)
    V = phase_space_jacobian(grid, 1)
    Vp = phase_space_jacobian_p_faces(grid, 1)
    Vxi = phase_space_jacobian_xi_faces(grid, 1)
    # exact operator from the exact fluxes (midpoint rule on the faces, second order)
    exact = [-((Vp[i+1, j] * ms.flux_p(pf[i+1], xc[j]) - Vp[i, j] * ms.flux_p(pf[i], xc[j])) / mom.p_widths[i] +
               (Vxi[i, j+1] * ms.flux_xi(pc[i], xf[j+1]) - Vxi[i, j] * ms.flux_xi(pc[i], xf[j])) / mom.xi_widths[j]) /
             V[i, j] for i in 1:np, j in 1:pitch_cells]
    f_exact = [ms.f(p, xi) for p in pc, xi in xc]
    solution = reshape((I - L) \ vec(f_exact .- exact), np, pitch_cells)
    truncation = reshape(L * vec(f_exact), np, pitch_cells) .- exact
    weights = V .* mom.p_widths .* mom.xi_widths'
    return (; solution, f_exact, truncation, weights, momentum=mom)
end

# Error of the pitch discretization alone: the step on coarse pitch faces against the step
# on nested fine pitch faces averaged onto the coarse cells, with the same momentum grid, so
# the momentum error cancels. The volume element 𝒱' ∝ p² does not depend on ξ.
function pitch_errors(p_faces, xi_faces_of, pitch_cells, reference_cells)
    fine = manufactured_step(p_faces, xi_faces_of(reference_cells); ms=BeamSolution)
    fine_faces = fine.momentum.xi_faces
    map(pitch_cells) do n
        coarse = manufactured_step(p_faces, xi_faces_of(n); ms=BeamSolution)
        faces = coarse.momentum.xi_faces
        reference = similar(coarse.solution)
        for j in 1:n
            cells = findall(k -> fine_faces[k] >= faces[j] - 1e-12 && fine_faces[k+1] <= faces[j+1] + 1e-12,
                            1:length(fine_faces)-1)
            widths = diff(fine_faces)[cells]
            length(cells) * n == reference_cells || error("the pitch faces are not nested")
            reference[:, j] = fine.solution[:, cells] * widths / sum(widths)
        end
        error = coarse.solution .- reference
        (; max=maximum(abs, error), l1=sum(abs.(error) .* coarse.weights) / sum(abs.(reference) .* coarse.weights))
    end
end

@testset "manufactured solutions" begin
    @testset "tanh momentum grid" begin
        faces = tanh_p_faces(200, 40, 0.0, 0.7, 30.0; transition_cells=4)
        @test length(faces) == 201
        @test (faces[1], faces[41], faces[end]) == (0.0, 0.7, 30.0)
        @test issorted(faces; lt=<=)
        widths = diff(faces)
        # smooth: neighbouring cells differ by at most 31% (observed 1.305), against the
        # jump of 10.5 of the biuniform grid
        @test maximum(widths[2:end] ./ widths[1:end-1]) < 1.35
        biuniform = biuniform_p_faces(200, 40, 0.0, 0.7, 30.0)
        @test tanh_p_faces(200, 40, 0.0, 0.7, 30.0; transition_cells=0) == biuniform
        # a narrow transition approaches the biuniform grid (observed 4.9e-5 at w = 1e-4)
        @test maximum(abs, tanh_p_faces(200, 40, 0.0, 0.7, 30.0; transition_cells=1e-4) .- biuniform) < 1e-3
        # the bulk keeps the biuniform resolution up to the separation (observed 0.9987
        # and 1.0187 times it); the transition widens the upper cells a little (1.073)
        @test widths[1] ≈ 0.7 / 40 rtol = 0.01
        @test widths[40] < 1.03 * 0.7 / 40
        @test widths[end] ≈ 29.3 / 160 rtol = 0.1
        @test_throws ArgumentError tanh_p_faces(200, 40, 0.0, 0.7, 30.0; transition_cells=-1)
        @test_throws ArgumentError tanh_p_faces(200, 40, 0.0, 0.7, 30.0; transition_cells=40)
        @test_throws ArgumentError tanh_p_faces(200, 0, 0.0, 0.7, 30.0)
    end

    @testset "second order on uniform, biuniform and tanh grids" begin
        grids = (uniform=N -> uniform_p_faces(N, 0.0, 30.0),
                 biuniform=N -> biuniform_p_faces(N, N ÷ 5, 0.0, 0.7, 30.0),
                 tanh=N -> tanh_p_faces(N, N ÷ 5, 0.0, 0.7, 30.0; transition_cells=N / 20))
        results = Dict(name => [manufactured_errors(make(N), N ÷ 25) for N in (200, 400, 800)]
                       for (name, make) in pairs(grids))
        # orders from 400 to 800 cells, observed: uniform 2.05 and 2.05 (max, L1),
        # biuniform 1.96 and 2.05, tanh 1.77 and 2.00; the max-norm order of the tanh grid
        # is still rising (1.52, 1.77, 1.87 up to 1600 cells)
        order(r, norm) = log2(getfield(r[2], norm) / getfield(r[3], norm))
        for name in (:uniform, :biuniform)
            @test order(results[name], :max) > 1.9
            @test order(results[name], :l1) > 1.9
        end
        @test order(results[:tanh], :l1) > 1.9
        @test order(results[:tanh], :max) > 1.7
        # The biuniform grid is first order locally: at its jump the face is not midway
        # between the cell centres, and the truncation error there does not fall with the
        # cell count (observed 0.8, 1.0, 1.1), while it falls by 4 per refinement in the
        # bulk. The global error is still second order (supraconvergence).
        biuniform = results[:biuniform]
        @test biuniform[3].truncation_at_separation > 0.5
        @test biuniform[3].truncation_at_separation >= biuniform[1].truncation_at_separation
        @test biuniform[2].truncation_in_bulk / biuniform[3].truncation_in_bulk > 3.5
        # the tanh grid with a transition that widens with the cell count is second order
        # locally too (observed a factor 4.1)
        tanh = results[:tanh]
        @test tanh[2].truncation_at_separation / tanh[3].truncation_at_separation > 3.5
        # At equal cell count the tanh grid is 3.5 times more accurate here (observed L1
        # 3.0e-4 against 1.04e-3 at 800 cells): its transition cells resolve the part of
        # the bulk just above p = 0.7, where the biuniform grid is already coarse. The gain
        # depends on how much the solution still varies there.
        @test tanh[3].l1 < 0.5 * biuniform[3].l1
    end

    @testset "nonuniform pitch grids" begin
        # the example's grid shape in both directions: biuniform p split at 0.7, geometric
        # pitch faces; joint refinement as above
        r = [manufactured_errors(biuniform_p_faces(N, N ÷ 5, 0.0, 0.7, 30.0), N ÷ 25;
                                 xi_faces=geometric_xi_faces(N ÷ 25)) for N in (200, 400, 800)]
        @test log2(r[2].max / r[3].max) > 1.9
        @test log2(r[2].l1 / r[3].l1) > 1.9
        # The pitch direction alone, where the momentum error would hide it otherwise: the
        # cross terms take plain means of neighbouring derivatives at faces that are not
        # midway between cell centres in every pitch cell of these grids, and the global
        # error is still second order
        orders = Dict{Symbol,NamedTuple}()
        for (name, faces) in pairs((uniform=uniform_xi_faces, geometric=geometric_xi_faces,
                                    uniform_theta=uniform_theta_xi_faces, biuniform=biuniform_xi_faces))
            e = pitch_errors(tanh_p_faces(200, 40, 0.0, 0.7, 30.0), faces, (16, 32, 64), 256)
            orders[name] = (max=log2(e[2].max / e[3].max), l1=log2(e[2].l1 / e[3].l1))
        end
        # observed orders from 32 to 64 pitch cells: L1 2.06-2.15; max 1.90 (uniform),
        # 1.40 (geometric), 1.96 (uniform θ), 2.13 (biuniform). The max norm of the
        # geometric grid is still pre-asymptotic at the beam, where its spacing law s^1.5 is
        # not smooth (1.82 from 64 to 128 cells against a 512-cell reference)
        for (name, o) in orders
            @test o.l1 > 1.9
            @test o.max > 1.2
        end
    end

    @testset "upwind is first order" begin
        for make in (N -> biuniform_p_faces(N, N ÷ 5, 0.0, 0.7, 30.0),
                     N -> tanh_p_faces(N, N ÷ 5, 0.0, 0.7, 30.0))
            r = [manufactured_errors(make(N), N ÷ 25; scheme=:upwind) for N in (400, 800)]
            @test 0.8 < log2(r[1].l1 / r[2].l1) < 1.3
        end
    end
end
