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

# Error of the pitch discretization alone: the step on coarse pitch faces against the step
# on nested fine pitch faces averaged onto the coarse cells, with the same momentum grid, so
# the momentum error cancels. The volume element 𝒱' ∝ p² does not depend on ξ.
function pitch_errors(p_faces, xi_faces_of, pitch_cells, reference_cells)
    fine = manufactured_step(p_faces, xi_faces_of(reference_cells); ms = BeamSolution)
    fine_faces = fine.momentum.xi_faces
    map(pitch_cells) do n
        coarse = manufactured_step(p_faces, xi_faces_of(n); ms = BeamSolution)
        faces = coarse.momentum.xi_faces
        reference = similar(coarse.solution)
        for j in 1:n
            cells = findall(
                k -> fine_faces[k] >= faces[j] - 1e-12 &&
                     fine_faces[k + 1] <= faces[j + 1] + 1e-12,
                1:(length(fine_faces) - 1))
            widths = diff(fine_faces)[cells]
            length(cells) * n == reference_cells || error("the pitch faces are not nested")
            reference[:, j] = fine.solution[:, cells] * widths / sum(widths)
        end
        error = coarse.solution .- reference
        (; max = maximum(abs, error),
            l1 = sum(abs.(error) .* coarse.weights) /
                 sum(abs.(reference) .* coarse.weights))
    end
end

@testset "manufactured solutions" begin
    @testset "tanh momentum grid" begin
        faces = tanh_p_faces(200, 40, 0.0, 0.7, 30.0; transition_cells = 4)
        @test length(faces) == 201
        @test (faces[1], faces[41], faces[end]) == (0.0, 0.7, 30.0)
        @test issorted(faces; lt = <=)
        widths = diff(faces)
        # smooth: neighbouring cells differ by at most 31% (observed 1.305), against the
        # jump of 10.5 of the biuniform grid
        @test maximum(widths[2:end] ./ widths[1:(end - 1)]) < 1.35
        biuniform = biuniform_p_faces(200, 40, 0.0, 0.7, 30.0)
        @test tanh_p_faces(200, 40, 0.0, 0.7, 30.0; transition_cells = 0) == biuniform
        # a narrow transition approaches the biuniform grid (observed 4.9e-5 at w = 1e-4)
        @test maximum(abs, tanh_p_faces(200, 40, 0.0, 0.7, 30.0; transition_cells = 1e-4) .-
                           biuniform) < 1e-3
        # the bulk keeps the biuniform resolution up to the separation (observed 0.9987
        # and 1.0187 times it); the transition widens the upper cells a little (1.073)
        @test widths[1] ≈ 0.7 / 40 rtol = 0.01
        @test widths[40] < 1.03 * 0.7 / 40
        @test widths[end] ≈ 29.3 / 160 rtol = 0.1
        @test_throws ArgumentError tanh_p_faces(
            200, 40, 0.0, 0.7, 30.0; transition_cells = -1)
        @test_throws ArgumentError tanh_p_faces(
            200, 40, 0.0, 0.7, 30.0; transition_cells = 40)
        @test_throws ArgumentError tanh_p_faces(200, 0, 0.0, 0.7, 30.0)
    end

    @testset "second order on uniform, biuniform and tanh grids" begin
        grids = (uniform = N -> uniform_p_faces(N, 0.0, 30.0),
            biuniform = N -> biuniform_p_faces(N, N ÷ 5, 0.0, 0.7, 30.0),
            tanh = N -> tanh_p_faces(N, N ÷ 5, 0.0, 0.7, 30.0; transition_cells = N / 20))
        results = Dict(name =>
                           [manufactured_errors(make(N), N ÷ 25) for N in (200, 400, 800)]
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
                 xi_faces = geometric_xi_faces(N ÷ 25)) for N in (200, 400, 800)]
        @test log2(r[2].max / r[3].max) > 1.9
        @test log2(r[2].l1 / r[3].l1) > 1.9
        # The pitch direction alone, where the momentum error would hide it otherwise: the
        # cross terms take plain means of neighbouring derivatives at faces that are not
        # midway between cell centres in every pitch cell of these grids, and the global
        # error is still second order
        orders = Dict{Symbol, NamedTuple}()
        for (name, faces) in pairs((
            uniform = uniform_xi_faces, geometric = geometric_xi_faces,
            uniform_theta = uniform_theta_xi_faces, biuniform = biuniform_xi_faces))
            e = pitch_errors(tanh_p_faces(200, 40, 0.0, 0.7, 30.0), faces, (16, 32, 64), 256)
            orders[name] = (max = log2(e[2].max / e[3].max), l1 = log2(e[2].l1 / e[3].l1))
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
            r = [manufactured_errors(make(N), N ÷ 25; scheme = :upwind) for N in (400, 800)]
            @test 0.8 < log2(r[1].l1 / r[2].l1) < 1.3
        end
    end
end
