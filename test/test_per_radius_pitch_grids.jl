# Operators on pitch grids that differ between radii: the pitch cells of
# the radial faces, conservation of the radial fluxes between different pitch grids, the
# sparsity pattern, and the size of the error the different grids add.
using Random: Xoshiro
using SparseArrays: findnz

# interior faces of a uniform pitch grid moved by `shift` cell widths
shifted_xi_faces(cell_count, shift) =
    vcat(-1.0, uniform_xi_faces(cell_count)[2:end-1] .+ shift * 2 / cell_count, 1.0)

# Σ 𝒱' Δp Δξ Δr df with the pitch widths of each radius
function particle_change(grid, df)
    total = 0.0
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        V = phase_space_jacobian(grid, ir)
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            total += V[i, j] * m.p_widths[i] * m.xi_widths[j] * grid.radial_grid.r_widths[ir] * df[i, j, ir]
        end
    end
    return total
end

function absolute_particle_change(grid, df)
    total = 0.0
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        V = phase_space_jacobian(grid, ir)
        for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            total += abs(V[i, j] * m.p_widths[i] * m.xi_widths[j] * grid.radial_grid.r_widths[ir] * df[i, j, ir])
        end
    end
    return total
end

@testset "pitch grids per radius" begin
    rng = Xoshiro(50)
    rad = RadialGrid(uniform_r_faces(4, 0.8))
    p_faces = biuniform_p_faces(8, 3, 0.0, 0.5, 3.0)
    xi_faces = [uniform_xi_faces(10), biuniform_theta_xi_faces(10, 6, 0.9), shifted_xi_faces(10, 0.3),
                shifted_xi_faces(10, 0.3)]
    grids = [MomentumGrid(p_faces, faces) for faces in xi_faces]
    grid = cylindrical_grid(grids, rad)
    np, nxi, nr = 8, 10, 4

    @testset "grid" begin
        @test !has_shared_momentum_grid(grid)
        @test has_same_cell_counts(grid)
        @test has_same_p_faces(grid)
        @test size(grid) == (np, nxi, nr)
        @test momentum_cell_count(grid) == np && pitch_cell_count(grid) == nxi
        @test_throws ArgumentError momentum_grid(grid)
        # the first and the last face carry the pitch grid of their radius
        @test has_identical_grids(radial_face_overlap(grid, 1))
        @test radial_face_overlap(grid, 1).xi_faces == xi_faces[1]
        @test has_identical_grids(radial_face_overlap(grid, 5))
        @test radial_face_overlap(grid, 5).xi_faces == xi_faces[4]
        # radii 3 and 4 have the same pitch faces
        @test has_identical_grids(radial_face_overlap(grid, 4))
        @test radial_face_pitch_cell_count(grid, 4) == nxi
        for face in 2:3
            overlap = radial_face_overlap(grid, face)
            @test !has_identical_grids(overlap)
            @test overlap_cell_count(overlap) == radial_face_pitch_cell_count(grid, face) > nxi
            @test length(overlap.left_ranges) == length(overlap.right_ranges) == nxi
        end
        face_cells = radial_face_pitch_cell_count(grid)
        @test face_cells == maximum(radial_face_pitch_cell_count(grid, face) for face in 1:5)
        V = phase_space_jacobian_r_faces(grid)
        @test size(V) == (np, face_cells, nr + 1)
        p = grids[1].p_centers
        for face in 1:5, s in 1:face_cells, i in 1:np
            expected = s <= radial_face_pitch_cell_count(grid, face) ?
                       2pi * p[i]^2 * rad.spatial_jacobian_faces[face] : 0.0
            @test V[i, s, face] == expected
        end
        # neighbours across the radial faces
        @test radial_neighbour_pitch_cells(grid, 1, 3, -1) == 1:0
        @test radial_neighbour_pitch_cells(grid, 4, 3, 1) == 1:0
        @test radial_neighbour_pitch_cells(grid, 3, 7, 1) == 7:7
        @test radial_neighbour_pitch_cells(grid, 4, 7, -1) == 7:7
        for ir in 1:3, j in 1:nxi
            lower, upper = xi_faces[ir][j], xi_faces[ir][j+1]
            overlapping = [k for k in 1:nxi
                           if min(upper, xi_faces[ir+1][k+1]) - max(lower, xi_faces[ir+1][k]) > 1e-12]
            @test radial_neighbour_pitch_cells(grid, ir, j, 1) == first(overlapping):last(overlapping)
        end
        @test_throws ArgumentError radial_neighbour_pitch_cells(grid, 1, 3, 0)

        # different p faces: no radial faces, and the operator refuses the grid
        other_p = cylindrical_grid([MomentumGrid(p_faces, xi_faces[1]),
                                    MomentumGrid(uniform_p_faces(8, 0.0, 3.0), xi_faces[1]),
                                    MomentumGrid(p_faces, xi_faces[1]), MomentumGrid(p_faces, xi_faces[1])], rad)
        @test has_same_cell_counts(other_p) && !has_same_p_faces(other_p)
        @test_throws ArgumentError phase_space_jacobian_r_faces(other_p)
        @test_throws "the same p faces" AdvectionDiffusion(other_p)
        # pitch grids on different intervals
        half = cylindrical_grid([MomentumGrid(p_faces, xi_faces[1]),
                                 MomentumGrid(p_faces, collect(range(0.0, 1.0; length=11))),
                                 MomentumGrid(p_faces, xi_faces[1]), MomentumGrid(p_faces, xi_faces[1])], rad)
        @test_throws ArgumentError radial_face_overlap(half, 2)
        @test_throws "the same interval" AdvectionDiffusion(half)
    end

    @testset "equal pitch faces give the shared-grid operator" begin
        shared = cylindrical_grid(grids[2], rad)
        copies = cylindrical_grid([MomentumGrid(copy(p_faces), copy(xi_faces[2])) for _ in 1:nr], rad)
        @test has_shared_momentum_grid(copies)
        f = rand(rng, np, nxi, nr)
        results = map((shared, copies)) do g
            op = AdvectionDiffusion(g; scheme=:central, r_boundary=:absorbing)
            local_rng = Xoshiro(3)
            for A in (op.advection_p, op.advection_xi, op.advection_r, op.diffusion_p_xi, op.diffusion_xi_p)
                A .= randn(local_rng, size(A))
            end
            for D in (op.diffusion_p_p, op.diffusion_xi_xi, op.diffusion_r_r)
                D .= rand(local_rng, size(D))
            end
            apply!(zeros(np, nxi, nr), op, f)
        end
        @test results[1] == results[2]
    end

    @testset "conservation with random coefficients ($scheme)" for scheme in (:upwind, :central,
                                                                                :exponential_fitting)
        op = AdvectionDiffusion(grid; scheme)
        @test size(op.advection_r) == size(op.diffusion_r_r) == (np, radial_face_pitch_cell_count(grid), nr + 1)
        for A in (op.advection_p, op.advection_xi, op.advection_r, op.diffusion_p_xi, op.diffusion_xi_p)
            A .= randn(rng, size(A))
        end
        for D in (op.diffusion_p_p, op.diffusion_xi_xi, op.diffusion_r_r)
            D .= rand(rng, size(D))
        end
        op.advection_p[end, :, :] .= 0  # close the outer p boundary
        f = rand(rng, np, nxi, nr)
        df = zeros(np, nxi, nr)
        apply!(df, op, f)
        @test abs(particle_change(grid, df)) < 1e-13 * absolute_particle_change(grid, df)
        @test (@allocated apply!(df, op, f)) == 0

        # the radial fluxes alone: every momentum cell keeps its particles
        radial = AdvectionDiffusion(grid; scheme)
        radial.advection_r .= randn(rng, size(radial.advection_r))
        radial.diffusion_r_r .= rand(rng, size(radial.diffusion_r_r))
        fill!(df, 0)
        apply!(df, radial, f)
        for i in 1:np
            only_i = zeros(np, nxi, nr)
            only_i[i, :, :] .= df[i, :, :]
            @test abs(particle_change(grid, only_i)) < 1e-13 * absolute_particle_change(grid, only_i)
        end
        @test maximum(abs, df) > 0
    end

    @testset "radial diffusion keeps a constant and the sign" begin
        op = AdvectionDiffusion(grid; scheme=:central)
        op.diffusion_r_r .= rand(rng, size(op.diffusion_r_r))
        df = zeros(np, nxi, nr)
        apply!(df, op, fill(2.5, np, nxi, nr))
        scale = maximum(op.diffusion_r_r) * 2.5 / minimum(rad.r_widths)^2
        @test maximum(abs, df) < 1e-12 * scale
        # diffusion between different pitch grids couples cells with non-negative weights
        M = operator_matrix(op)
        rows, columns, values = findnz(M)
        @test all(values[k] >= 0 for k in eachindex(values) if rows[k] != columns[k])
        @test all(values[k] <= 0 for k in eachindex(values) if rows[k] == columns[k])
    end

    @testset "absorbing wall" begin
        op = AdvectionDiffusion(grid; r_boundary=:absorbing)
        op.diffusion_r_r .= 0.3
        df = zeros(np, nxi, nr)
        apply!(df, op, ones(np, nxi, nr))
        @test maximum(abs, df[:, :, 1:end-1]) < 1e-13
        m = grids[4]
        V = phase_space_jacobian_r_faces(grid)
        wall = sum(V[i, j, end] * m.p_widths[i] * m.xi_widths[j] * 0.3 / (rad.r_faces[end] - rad.r_centers[end])
                   for i in 1:np, j in 1:nxi)
        @test particle_change(grid, df) ≈ -wall rtol = 1e-13
    end

    @testset "sparsity pattern ($scheme)" for scheme in (:central, :exponential_fitting, :quick)
        op = AdvectionDiffusion(grid; scheme)
        for A in (op.advection_p, op.advection_xi, op.advection_r, op.diffusion_p_xi, op.diffusion_xi_p)
            A .= randn(rng, size(A))
        end
        for D in (op.diffusion_p_p, op.diffusion_xi_xi, op.diffusion_r_r)
            D .= rand(rng, size(D))
        end
        @test check_stencil_pattern(op)
        # the pattern holds the overlapping cells of the neighbouring radii
        pattern = stencil_pattern(op)
        index = LinearIndices((np, nxi, nr))
        for ir in 1:nr, j in 1:nxi, dr in (-1, 1)
            1 <= ir + dr <= nr || continue
            neighbours = radial_neighbour_pitch_cells(grid, ir, j, dr)
            for k in 1:nxi
                @test pattern[index[2, j, ir], index[2, k, ir+dr]] == (k in neighbours)
            end
        end
        # the matrix assembled with the pattern is the operator
        M = operator_matrix(op)
        f = rand(rng, np, nxi, nr)
        df = zeros(np, nxi, nr)
        apply!(df, op, f)
        @test M * vec(f) ≈ vec(df) rtol = 1e-12

        # the same through the equation system
        layout = UnknownLayout(rad, [RadialUnknown(:electric_field),
                                     KineticUnknown(:hot_tail_distribution, grid)])
        system = EquationSystem(layout, [
            Equation(:electric_field, Algebraic(), LinearTerm(:electric_field; coefficient=-1)),
            Equation(:hot_tail_distribution, TimeDerivative(), FokkerPlanckTerm(:hot_tail_distribution, op))])
        id = unknown_id(layout, :hot_tail_distribution)
        u = zeros(length(layout))
        for ir in 1:nr
            kinetic_view(u, layout, id, ir) .= f[:, :, ir]
        end
        du = zeros(length(layout))
        rhs!(du, u, system, 0.0)
        for ir in 1:nr
            @test kinetic_view(du, layout, id, ir) == df[:, :, ir]
        end
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        @test J * u ≈ du rtol = 1e-12
        @test kinetic_array(u, layout, id) == f
    end

    @testset "error of the radial flux between different pitch grids" begin
        # f depends on the pitch alone, so the exact radial flux vanishes. With the pitch grid
        # of the middle radius moved by θ cell widths h, the values of both sides taken as
        # constant over their cells leave (D/Δr²) θ(1 - θ) h² f''(ξ) in the middle radius:
        # nothing for f linear in ξ on these grids, 2 D θ(1 - θ) h²/Δr² for f = ξ².
        radii = RadialGrid(uniform_r_faces(3, 0.6))
        cells, shift, diffusion = 16, 0.3, 0.7
        h, dr = 2 / cells, 0.2
        uniform = MomentumGrid(uniform_p_faces(2, 0.0, 1.0), uniform_xi_faces(cells))
        moved = MomentumGrid(uniform_p_faces(2, 0.0, 1.0), shifted_xi_faces(cells, shift))
        g3 = cylindrical_grid([uniform, moved, uniform], radii)
        op = AdvectionDiffusion(g3; scheme=:central)
        op.diffusion_r_r .= diffusion
        values(shape) = [shape(momentum_grid(g3, ir).xi_centers[j]) for _ in 1:2, j in 1:cells, ir in 1:3]
        interior = 2:cells-1   # the end cells of the moved grid have another width
        df = apply!(zeros(2, cells, 3), op, values(identity))
        @test maximum(abs, df[:, interior, 2]) < 1e-13 * diffusion / dr^2
        df = apply!(zeros(2, cells, 3), op, values(x -> x^2))
        expected = 2 * diffusion * shift * (1 - shift) * h^2 / dr^2
        @test all(x -> isapprox(x, expected; rtol=1e-11), df[:, interior, 2])
        @test abs(particle_change(g3, df)) < 1e-13 * absolute_particle_change(g3, df)
        # the error falls with the square of the pitch cell width
        finer = cylindrical_grid([MomentumGrid(uniform_p_faces(2, 0.0, 1.0), uniform_xi_faces(2cells)),
                                  MomentumGrid(uniform_p_faces(2, 0.0, 1.0), shifted_xi_faces(2cells, shift)),
                                  MomentumGrid(uniform_p_faces(2, 0.0, 1.0), uniform_xi_faces(2cells))], radii)
        op_finer = AdvectionDiffusion(finer; scheme=:central)
        op_finer.diffusion_r_r .= diffusion
        f_finer = [momentum_grid(finer, ir).xi_centers[j]^2 for _ in 1:2, j in 1:2cells, ir in 1:3]
        df_finer = apply!(zeros(2, 2cells, 3), op_finer, f_finer)
        @test all(x -> isapprox(x, expected / 4; rtol=1e-11), df_finer[:, 2:2cells-1, 2])
    end
end
