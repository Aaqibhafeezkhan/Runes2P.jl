# Overlap of two pitch grids and the conservative remap between them:
# structure of the overlap cells, conservation to round-off, bounds, order of the remap.
using Random: MersenneTwister

@testset "pitch overlap" begin
    content(faces, f) = sum(diff(faces) .* f)

    @testset "identical grids" begin
        faces = biuniform_theta_xi_faces(20, 12, 0.9)
        overlap = PitchOverlap(faces, copy(faces))
        @test has_identical_grids(overlap)
        @test overlap_cell_count(overlap) == 20
        @test overlap.left_cells == overlap.right_cells == 1:20
        @test overlap.left_ranges == overlap.right_ranges == [j:j for j in 1:20]
        @test all(==(1.0), overlap.left_fractions)
        @test all(==(1.0), overlap.right_fractions)
        @test overlap.xi_faces == faces
        f = rand(MersenneTwister(1), 20)
        @test remap_to_right!(zeros(20), overlap, f) == f
        @test remap_to_left!(zeros(20), overlap, f) == f
    end

    @testset "nested grids" begin
        coarse, fine = uniform_xi_faces(4), uniform_xi_faces(12)
        overlap = PitchOverlap(coarse, fine)
        @test !has_identical_grids(overlap)
        @test overlap_cell_count(overlap) == 12
        @test overlap.left_ranges == [1:3, 4:6, 7:9, 10:12]
        @test overlap.right_ranges == [j:j for j in 1:12]
        @test overlap.left_fractions ≈ fill(1 / 3, 12) rtol = 1e-14
        @test overlap.right_fractions ≈ ones(12) rtol = 1e-14
        f_coarse = [1.0, -2.0, 3.0, 0.5]
        f_fine = remap_to_right!(zeros(12), overlap, f_coarse)
        @test f_fine ≈ repeat(f_coarse; inner=3) rtol = 1e-14
        # the mean over the fine cells of each coarse cell
        values = collect(1.0:12.0)
        @test remap_to_left!(zeros(4), overlap, values) ≈ [2.0, 5.0, 8.0, 11.0] rtol = 1e-14
    end

    @testset "grids without common interior faces" begin
        left = uniform_xi_faces(10)
        right = biuniform_theta_xi_faces(20, 12, 0.9)   # the runaway pitch grid
        overlap = PitchOverlap(left, right)
        # cos(π/2) of the right grid and the face 0 of the left grid are one face
        @test count(x -> abs(x) < 1e-10, overlap.xi_faces) == 1
        count_expected = length(unique(x -> round(x; digits=10), vcat(left, right))) - 1
        @test overlap_cell_count(overlap) == count_expected
        @test sum(overlap.xi_widths) ≈ 2 rtol = 1e-15
        @test all(>(0), overlap.xi_widths)
        @test issorted(overlap.left_cells) && issorted(overlap.right_cells)
        for (ranges, fractions, cells) in ((overlap.left_ranges, overlap.left_fractions, overlap.left_cells),
                                           (overlap.right_ranges, overlap.right_fractions, overlap.right_cells))
            @test vcat(collect.(ranges)...) == 1:overlap_cell_count(overlap)
            for (j, range) in enumerate(ranges)
                @test !isempty(range)
                @test all(==(j), cells[range])
                @test sum(fractions[range]) ≈ 1 rtol = 1e-12
            end
        end
        # every overlap cell lies inside its two cells
        for s in 1:overlap_cell_count(overlap)
            jl, jr = overlap.left_cells[s], overlap.right_cells[s]
            @test left[jl] - 1e-12 <= overlap.xi_faces[s] && overlap.xi_faces[s+1] <= left[jl+1] + 1e-12
            @test right[jr] - 1e-12 <= overlap.xi_faces[s] && overlap.xi_faces[s+1] <= right[jr+1] + 1e-12
        end

        rng = MersenneTwister(5)
        for _ in 1:20
            f_left = randn(rng, 10)
            f_right = remap_to_right!(zeros(20), overlap, f_left)
            @test content(right, f_right) ≈ content(left, f_left) rtol = 1e-13 atol = 1e-14
            @test minimum(f_left) - 1e-14 <= minimum(f_right) && maximum(f_right) <= maximum(f_left) + 1e-14
            g_right = randn(rng, 20)
            g_left = remap_to_left!(zeros(10), overlap, g_right)
            @test content(left, g_left) ≈ content(right, g_right) rtol = 1e-13 atol = 1e-14
            @test minimum(g_right) - 1e-14 <= minimum(g_left) && maximum(g_left) <= maximum(g_right) + 1e-14
        end
        # constants and signs are kept
        @test remap_to_right!(zeros(20), overlap, fill(3.5, 10)) ≈ fill(3.5, 20) rtol = 1e-12
        @test remap_to_left!(zeros(10), overlap, fill(3.5, 20)) ≈ fill(3.5, 10) rtol = 1e-12
        @test all(>=(0), remap_to_right!(zeros(20), overlap, rand(rng, 10)))

        # matrices: each momentum cell on its own, overwriting the target
        f = rand(rng, 7, 10)
        target = fill(NaN, 7, 20)
        remap_to_right!(target, overlap, f)
        for i in 1:7
            @test target[i, :] == remap_to_right!(zeros(20), overlap, f[i, :])
        end
        @test (@allocated remap_to_right!(target, overlap, f)) == 0
        back = zeros(7, 10)
        @test (@allocated remap_to_left!(back, overlap, target)) == 0
        @test_throws DimensionMismatch remap_to_right!(zeros(7, 19), overlap, f)
        @test_throws DimensionMismatch remap_to_right!(zeros(6, 20), overlap, f)
        @test_throws DimensionMismatch remap_to_left!(zeros(20), overlap, zeros(20))
    end

    @testset "momentum grids and arguments" begin
        left = MomentumGrid(uniform_p_faces(5, 0.0, 1.0), uniform_xi_faces(6))
        right = MomentumGrid(uniform_p_faces(8, 1.0, 3.0), uniform_xi_faces(9))
        overlap = PitchOverlap(left, right)
        @test overlap_cell_count(overlap) == 12   # faces at ±1 and ±1/3 are common
        @test_throws ArgumentError PitchOverlap([-1.0, 0.0, 1.0], [-1.0, 0.0, 0.5])
        @test_throws ArgumentError PitchOverlap([-1.0, 0.0, 1.0], [-0.5, 0.0, 1.0])
        @test_throws ArgumentError PitchOverlap([-1.0, 1.0], [-1.0, 0.5, 0.2, 1.0])
        @test_throws ArgumentError PitchOverlap([-1.0], [-1.0, 1.0])
        # a half range, as the pitch grid of one direction
        half = PitchOverlap([0.0, 0.5, 1.0], [0.0, 0.25, 0.75, 1.0])
        @test overlap_cell_count(half) == 4
        @test half.left_ranges == [1:2, 3:4]
        @test half.right_ranges == [1:1, 2:3, 4:4]
    end

    @testset "measure that varies with ξ" begin
        # ∫ (1 + ξ²) dξ as the measure: the remap conserves the content in that measure
        measure(a, b) = (b - a) + (b^3 - a^3) / 3
        left, right = uniform_xi_faces(7), biuniform_theta_xi_faces(15, 10, 0.5)
        overlap = PitchOverlap(left, right; measure)
        weighted(faces, f) = sum(measure(faces[j], faces[j+1]) * f[j] for j in eachindex(f))
        rng = MersenneTwister(7)
        f_left = rand(rng, 7)
        f_right = remap_to_right!(zeros(15), overlap, f_left)
        @test weighted(right, f_right) ≈ weighted(left, f_left) rtol = 1e-13
        @test remap_to_left!(zeros(7), overlap, fill(2.0, 15)) ≈ fill(2.0, 7) rtol = 1e-12
        @test_throws ArgumentError PitchOverlap(left, right; measure=(a, b) -> a - b)
    end

    @testset "order of the remap" begin
        # remap of the cell averages of a smooth function to a shifted grid: first order
        shape(x) = exp(2x)
        average(faces, j) = (shape(faces[j+1]) - shape(faces[j])) / (2 * (faces[j+1] - faces[j]))
        errors = map((16, 32, 64, 128)) do cells
            left = uniform_xi_faces(cells)
            # the interior faces moved by 0.4 cell widths
            right = vcat(-1.0, left[2:end-1] .+ 0.4 * 2 / cells, 1.0)
            overlap = PitchOverlap(left, right)
            f_left = [average(left, j) for j in 1:cells]
            f_right = remap_to_right!(zeros(cells), overlap, f_left)
            exact = [average(right, j) for j in 1:cells]
            maximum(abs.(f_right .- exact) ./ exact)
        end
        orders = log2.(errors[1:end-1] ./ errors[2:end])
        @test all(order -> 0.9 < order < 1.1, orders)
        @test errors[end] < 0.01
    end
end
