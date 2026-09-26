# Equation system: right-hand side from terms, the stored Jacobian from colored
# AD over the sparse terms alone, the matrix-free Jacobian operator, mass matrix, checks on
# construction, residual with explicit terms, scales and tolerances.
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal, mul!
using Random: Xoshiro
using SparseArrays: nnz

# test terms: a nonlinear decay -c x² of a radial unknown, and a scalar that sums a radial
# unknown over the radii with the volume weights V' Δr
struct QuadraticDecayTerm <: EquationTerm
    unknown::Symbol
    rate::Float64
end
Runes2P.dependencies(term::QuadraticDecayTerm) = [term.unknown]
function Runes2P.rhs!(out, term::QuadraticDecayTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout),
        (a, b) in zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir))
        out[a] -= term.rate * u[b]^2
    end
    return out
end
function Runes2P.sparsity!(entries, term::QuadraticDecayTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        append!(entries, zip(unknown_range(layout, row, ir), unknown_range(layout, id, ir)))
    end
    return entries
end

struct RadialSumTerm <: EquationTerm
    unknown::Symbol
end
Runes2P.dependencies(term::RadialSumTerm) = [term.unknown]
function Runes2P.rhs!(out, term::RadialSumTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    rad = layout.radial_grid
    total = zero(eltype(out))
    for ir in 1:radial_cell_count(layout)
        total += rad.spatial_jacobian[ir] * rad.r_widths[ir] * radial_view(u, layout, id, ir)[1]
    end
    out[first(unknown_range(layout, row))] += total
    return out
end
function Runes2P.sparsity!(entries, term::RadialSumTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    r = first(unknown_range(layout, row))
    for ir in 1:radial_cell_count(layout)
        push!(entries, (r, first(unknown_range(layout, id, ir))))
    end
    return entries
end

# dense ∂rhs/∂u by central differences: exact up to round-off for terms at most quadratic
function central_difference_jacobian(system, u, t)
    n = length(u)
    J = zeros(n, n)
    scales = unknown_scales(system.layout)
    up, down, rp, rm = copy(u), copy(u), zeros(n), zeros(n)
    for k in 1:n
        h = 1e-3 * max(abs(u[k]), scales[k])
        up[k] += h
        down[k] -= h
        rhs!(rp, up, system, t)
        rhs!(rm, down, system, t)
        J[:, k] .= (rp .- rm) ./ (2h)
        up[k] = down[k] = u[k]
    end
    return J
end

@testset "equation system" begin
    rad = RadialGrid(uniform_r_faces(2, 0.5))
    mom = MomentumGrid(biuniform_p_faces(10, 4, 0.0, 0.5, 3.0), uniform_xi_faces(6))
    g = cylindrical_grid(mom, rad)
    plasma = PlasmaState(cold_electron_temperature=[100.0, 300.0],
                         ions=[IonSpecies("D", 1, 1, [1e20, 5e19])],
                         electric_field=[5.0, 2.0], magnetic_field=[5.0, 5.0])
    # a scheme linear in f: the finite-difference Jacobian reference below is exact only
    # where the operator is smooth, which a limiter is not at random states
    fp = fokker_planck_operator(g, plasma; synchrotron=true, scheme=:exponential_fitting)
    knockon = KnockOnOperator(g, g, 0.2)
    layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, g; scale=1e16),
                                 RadialUnknown(:total_electron_density; scale=1e20),
                                 RadialUnknown(:hot_electron_density; scale=1e16),
                                 RadialUnknown(:temperature; scale=100.0),
                                 ScalarUnknown(:hot_electron_content; scale=1e15)])
    equations(knockon_treatment) = [
        Equation(:hot_tail_distribution, TimeDerivative(), FokkerPlanckTerm(:hot_tail_distribution, fp),
                 KnockOnTerm(:hot_tail_distribution, knockon; target_density=:total_electron_density,
                             treatment=knockon_treatment)),
        Equation(:total_electron_density, TimeDerivative(),
                 LinearTerm(:total_electron_density; coefficient=-0.1)),
        Equation(:hot_electron_density, Algebraic(), DensityMomentTerm(:hot_tail_distribution),
                 LinearTerm(:hot_electron_density; coefficient=-1)),
        Equation(:temperature, TimeDerivative(), QuadraticDecayTerm(:temperature, 1e-3)),
        Equation(:hot_electron_content, Algebraic(), RadialSumTerm(:hot_electron_density),
                 LinearTerm(:hot_electron_content; coefficient=-1))]
    system = EquationSystem(layout, equations(:implicit); time_scale=1e-3)

    # a state with a smooth distribution and distinct values elsewhere
    f_id, ntot_id, nhot_id, T_id, content_id = (unknown_id(layout, name) for name in
        (:hot_tail_distribution, :total_electron_density, :hot_electron_density, :temperature,
         :hot_electron_content))
    f3 = [1e16 * exp(-p / 0.4) * (1.2 + ξ) * (1 + 0.3ir)
          for p in mom.p_centers, ξ in mom.xi_centers, ir in 1:2]
    u = zeros(length(layout))
    for ir in 1:2
        kinetic_view(u, layout, f_id, ir) .= f3[:, :, ir]
        radial_view(u, layout, ntot_id, ir) .= 1e20 * (1 + 0.5ir)
        radial_view(u, layout, nhot_id, ir) .= 2e16 * ir
        radial_view(u, layout, T_id, ir) .= 100.0 * ir
    end
    scalar_view(u, layout, content_id)[] = 3e15

    @testset "right-hand side equals the sum of the operators" begin
        du = zeros(length(layout))
        rhs!(du, u, system, 0.0)
        expected = zeros(size(g))
        apply!(expected, fp, f3)
        apply!(expected, knockon, f3, [1.5e20, 2e20])
        for ir in 1:2
            @test kinetic_view(du, layout, f_id, ir) ≈ expected[:, :, ir] rtol = 1e-13
            @test radial_view(du, layout, ntot_id, ir)[1] ≈ -0.1 * 1e20 * (1 + 0.5ir)
            @test radial_view(du, layout, nhot_id, ir)[1] ≈ density(g, f3)[ir] - 2e16 * ir rtol = 1e-13
            @test radial_view(du, layout, T_id, ir)[1] ≈ -1e-3 * (100.0 * ir)^2
        end
        content = sum(rad.spatial_jacobian[ir] * rad.r_widths[ir] * 2e16 * ir for ir in 1:2)
        @test scalar_view(du, layout, content_id)[] ≈ content - 3e15
        # kinetic_array reads and writes the state in place
        a = kinetic_array(u, layout, f_id)
        @test a == f3 && size(a) == size(g)
    end

    @testset "mass matrix and scales" begin
        M = mass_matrix(system)
        @test M isa Diagonal
        @test all(M.diag[unknown_range(layout, f_id, 2)] .== 1)
        @test M.diag[first(unknown_range(layout, nhot_id, 1))] == 0
        @test M.diag[first(unknown_range(layout, content_id))] == 0
        scales = residual_scales(system)
        @test scales[first(unknown_range(layout, f_id, 1))] == 1e16 / 1e-3
        @test scales[first(unknown_range(layout, nhot_id, 2))] == 1e16
        @test scales[first(unknown_range(layout, content_id))] == 1e15
    end

    J = jacobian_prototype(system)
    jacobian!(J, system, u, 0.0)

    @testset "stored Jacobian: the sparse terms alone" begin
        # the colored AD Jacobian equals the dense AD Jacobian of the sparse terms; a
        # matrix-free term leaking into the colored pass would change it
        function sparse_part(x)
            out = zeros(eltype(x), length(x))
            for bound in system.sparse_terms
                Runes2P.rhs!(out, bound.term, x, layout, bound.row, 0.0)
            end
            return out
        end
        dense = ForwardDiff.jacobian(sparse_part, u)
        @test Matrix(J) ≈ dense rtol = 1e-13
        @test all(jacobian_sparsity(system)[findall(!iszero, dense)])
        # the same stored Jacobian without the knock-on term
        without = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(),
                                                   FokkerPlanckTerm(:hot_tail_distribution, fp));
                                          equations(:implicit)[2:end]])
        J2 = jacobian_prototype(without)
        jacobian!(J2, without, u, 0.0)
        @test J2 == J
        # jacobian! allocates nothing
        jacobian!(J, system, u, 0.0)
        @test (@allocated jacobian!(J, system, u, 0.0)) == 0
    end

    @testset "Jacobian operator: all implicit terms" begin
        L = jacobian_operator(J, system, u, 0.0)
        reference = central_difference_jacobian(system, u, 0.0)
        dense = Matrix(L)
        @test maximum(abs, dense - reference) < 1e-8 * maximum(abs, reference)
        # the knock-on block lies outside the stored pattern (dense per radius)
        @test count(!iszero, dense) > nnz(J)
        v = randn(Xoshiro(2), length(u)) .* unknown_scales(layout)
        @test L * v ≈ reference * v rtol = 1e-8
        # 5-argument product
        w = ones(length(u))
        @test mul!(copy(w), L, v, 2.0, 0.5) ≈ 2 .* (reference * v) .+ 0.5 .* w rtol = 1e-8
    end

    @testset "residual, explicit terms at the previous time level" begin
        u_old = u .* (1 .+ 0.01 .* sin.(1:length(u)))
        dt = 1e-4
        F = zeros(length(u))
        residual!(F, u, u_old, system, 0.0, dt)
        du = zeros(length(u))
        rhs!(du, u, system, 0.0)
        @test F ≈ mass_matrix(system) * (u .- u_old) ./ dt .- du rtol = 1e-12
        explicit = EquationSystem(layout, equations(:explicit))
        Fe = zeros(length(u))
        residual!(Fe, u, u_old, explicit, 0.0, dt)
        # fokker-planck at u, knock-on at u_old
        knock_old = zeros(size(g))
        f_old = stack(kinetic_view(u_old, layout, f_id, ir) for ir in 1:2)
        apply!(knock_old, knockon, f_old, [radial_view(u_old, layout, ntot_id, ir)[1] for ir in 1:2])
        fp_new = zeros(size(g))
        apply!(fp_new, fp, f3)
        for ir in 1:2
            expected = (kinetic_view(u, layout, f_id, ir) .- kinetic_view(u_old, layout, f_id, ir)) ./ dt .-
                       fp_new[:, :, ir] .- knock_old[:, :, ir]
            @test kinetic_view(Fe, layout, f_id, ir) ≈ expected rtol = 1e-12
        end
        # the explicit term is absent from both Jacobians
        Je = jacobian_prototype(explicit)
        jacobian!(Je, explicit, u, 0.0)
        @test Matrix(jacobian_operator(Je, explicit, u, 0.0)) == Matrix(Je)
        residual!(F, u, u_old, system, 0.0, dt)
        @test (@allocated residual!(F, u, u_old, system, 0.0, dt)) == 0
        rhs!(du, u, system, 0.0)
        @test (@allocated rhs!(du, u, system, 0.0)) == 0
    end

    @testset "checks on construction" begin
        eqs = equations(:implicit)
        # an algebraic unknown missing from its own equation cannot be determined
        undetermined = Equation(:hot_electron_density, Algebraic(), DensityMomentTerm(:hot_tail_distribution))
        @test_throws ArgumentError EquationSystem(layout, [eqs[1:2]; undetermined; eqs[4:5]])
        # an explicit term is evaluated at the previous time level and determines nothing
        kinetic_only = UnknownLayout(layout.radial_grid, [KineticUnknown(:hot_tail_distribution, g)])
        explicit_source = MaxwellianSourceTerm(:hot_tail_distribution, g, 500.0; target_density=1e20, rate=1.0)
        @test_throws "no implicit term" EquationSystem(kinetic_only,
            [Equation(:hot_tail_distribution, Algebraic(), explicit_source)])
        @test EquationSystem(kinetic_only, [Equation(:hot_tail_distribution, Algebraic(), explicit_source,
                                                     LinearTerm(:hot_tail_distribution; coefficient=-1.0))]) isa
              EquationSystem
        @test_throws ArgumentError EquationSystem(layout, eqs[1:4])
        @test_throws ArgumentError EquationSystem(layout, [eqs; eqs[2]])
        undeclared = Equation(:temperature, TimeDerivative(), LinearTerm(:pressure))
        @test_throws ArgumentError EquationSystem(layout, [eqs[1:3]; undeclared; eqs[5]])
        # terms check themselves against the layout
        wrong_kind = Equation(:hot_electron_density, Algebraic(), LinearTerm(:hot_electron_density),
                              DensityMomentTerm(:temperature))
        @test_throws ArgumentError EquationSystem(layout, [eqs[1:2]; wrong_kind; eqs[4:5]])
        wrong_size = Equation(:temperature, TimeDerivative(), LinearTerm(:hot_electron_content))
        @test_throws DimensionMismatch EquationSystem(layout, [eqs[1:3]; wrong_size; eqs[5]])
        # non-local operators still refuse assembly with the stencil pattern
        @test_throws ArgumentError operator_matrix(knockon)
    end

    @testset "scales and tolerances" begin
        @test tolerance(system, :hot_tail_distribution) == Tolerance(1e-10 * 1e16, 1e-10)
        custom = EquationSystem(layout, equations(:implicit);
                                newton_tolerances=Dict(:temperature => Tolerance(1e-3, 1e-8)))
        @test tolerance(custom, :temperature) == Tolerance(1e-3, 1e-8)
        @test tolerance(custom, :temperature; kind=:time_step) == Tolerance(1e-10 * 100, 1e-10)
        @test_throws ArgumentError tolerance(custom, :temperature; kind=:other)
    end

    @testset "system on different momentum grids at different radii" begin
        # knock-on and density moment work per radius; the result equals the per-radius
        # operator applied to the same values
        m2 = MomentumGrid(uniform_p_faces(7, 0.0, 3.0), -cos.(range(0, π; length=9)))
        ragged = cylindrical_grid([mom, m2], rad)
        op = KnockOnOperator(ragged, ragged, 0.2)
        layout2 = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, ragged; scale=1e16),
                                      RadialUnknown(:hot_electron_density; scale=1e16)])
        system2 = EquationSystem(layout2, [
            Equation(:hot_tail_distribution, TimeDerivative(),
                     KnockOnTerm(:hot_tail_distribution, op; target_density=[1e20, 2e20]),
                     LinearTerm(:hot_tail_distribution; coefficient=-10.0)),
            Equation(:hot_electron_density, Algebraic(), DensityMomentTerm(:hot_tail_distribution),
                     LinearTerm(:hot_electron_density; coefficient=-1))])
        id = unknown_id(layout2, :hot_tail_distribution)
        u2 = zeros(length(layout2))
        for ir in 1:2
            m = momentum_grid(ragged, ir)
            kinetic_view(u2, layout2, id, ir) .= [1e16 * exp(-p / 0.5) * (1 + ξ / 2) for p in m.p_centers,
                                                                                          ξ in m.xi_centers]
        end
        du2 = zeros(length(layout2))
        rhs!(du2, u2, system2, 0.0)
        for ir in 1:2
            f = kinetic_view(u2, layout2, id, ir)
            expected = -10.0 .* f
            apply_at_radius!(expected, op, f, [1e20, 2e20][ir], ir)
            @test kinetic_view(du2, layout2, id, ir) ≈ expected rtol = 1e-13
        end
        J2 = jacobian_prototype(system2)
        jacobian!(J2, system2, u2, 0.0)
        reference = central_difference_jacobian(system2, u2, 0.0)
        @test maximum(abs, Matrix(jacobian_operator(J2, system2, u2, 0.0)) - reference) <
              1e-8 * maximum(abs, reference)
        # the Fokker-Planck term needs a shared grid
        @test_throws ArgumentError AdvectionDiffusion(ragged)
    end
end

@testset "system on a single radius" begin
    # the radius offsets of a kinetic unknown are indexed as a range, which needs a nonzero
    # step even when there is only one radius
    rad = RadialGrid(uniform_r_faces(1, 0.5))
    mom = MomentumGrid(uniform_p_faces(6, 0.0, 2.0), uniform_xi_faces(4))
    g = cylindrical_grid(mom, rad)
    plasma = PlasmaState(cold_electron_temperature=[200.0], ions=[IonSpecies("D", 1, 1, [1e20])],
                         electric_field=[4.0], magnetic_field=[5.0])
    op = fokker_planck_operator(g, plasma)
    layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, g; scale=1e18)])
    system = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(),
                                              FokkerPlanckTerm(:hot_tail_distribution, op))])
    u = vec([exp(-p) * (1 + 0.3ξ) for p in mom.p_centers, ξ in mom.xi_centers])
    du = zeros(length(layout))
    rhs!(du, u, system, 0.0)
    expected = zeros(size(g))
    apply!(expected, op, reshape(copy(u), size(g)))
    @test du ≈ vec(expected) rtol = 1e-14
end

@testset "Maxwellian source term" begin
    # adds exactly rate × (target - bulk density) particles per second, distributed as a
    # Maxwellian, and nothing else
    rad = RadialGrid(uniform_r_faces(2, 0.5))
    mom = MomentumGrid(biuniform_p_faces(30, 20, 0.0, 0.3, 2.0), uniform_xi_faces(6))
    g = cylindrical_grid(mom, rad)
    layout = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, g; scale=1e20)])
    temperature, target, rate, bound = 500.0, 1e20, 250.0, 0.1
    term = MaxwellianSourceTerm(:hot_tail_distribution, g, temperature; target_density=target,
                                rate, bulk_momentum=bound)
    system = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(), term)])
    @test Runes2P.treatment(term) === :explicit
    # an empty distribution: the source adds rate × target particles per second
    du = zeros(length(layout))
    rhs!(du, zeros(length(layout)), system, 0.0)
    @test density(g, reshape(du, size(g))) ≈ fill(rate * target, 2) rtol = 1e-12
    # the shape is the Maxwellian, and the source vanishes when the bulk is at its target
    theta = temperature / Constants.ELECTRON_REST_ENERGY
    shape = [exp(-p^2 / (2theta)) for p in mom.p_centers, _ in 1:pitch_cell_count(mom)]
    added = kinetic_view(du, layout, 1, 1)
    @test added ./ added[1, 1] ≈ shape ./ shape[1, 1] rtol = 1e-12
    u = zeros(length(layout))
    f = kinetic_view(u, layout, 1, 1)
    f .= shape
    bulk = sum(phase_space_jacobian(g, 1)[i, j] * mom.p_widths[i] * mom.xi_widths[j] * f[i, j]
               for i in 1:momentum_cell_count(mom), j in 1:pitch_cell_count(mom)
               if mom.p_centers[i] <= bound) / rad.spatial_jacobian[1]
    at_target = MaxwellianSourceTerm(:hot_tail_distribution, g, temperature; target_density=bulk,
                                     rate, bulk_momentum=bound)
    system_at_target = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(), at_target)])
    fill!(du, 0)
    rhs!(du, u, system_at_target, 0.0)
    @test maximum(abs, kinetic_view(du, layout, 1, 1)) < 1e-6 * maximum(shape) * rate
    # only the cells below the bound are counted: particles above it do not cancel the source
    f[end, :] .= 1e3 * maximum(shape)
    fill!(du, 0)
    rhs!(du, u, system_at_target, 0.0)
    @test maximum(abs, kinetic_view(du, layout, 1, 1)) < 1e-6 * maximum(shape) * rate
    @test_throws ArgumentError MaxwellianSourceTerm(:hot_tail_distribution, g, -1.0;
                                                    target_density=target, rate)
    @test_throws ArgumentError MaxwellianSourceTerm(:hot_tail_distribution, g, temperature;
                                                    target_density=target, rate, bulk_momentum=1e-9)
    # the unknown the term reads and the row it adds to both live on the term's grid
    other = cylindrical_grid(MomentumGrid(uniform_p_faces(12, 0.0, 2.0), uniform_xi_faces(4)), rad)
    two = UnknownLayout(rad, [KineticUnknown(:hot_tail_distribution, g; scale=1e20),
                              KineticUnknown(:runaway_distribution, other; scale=1e20)])
    source(row, source_term) = Equation(row, TimeDerivative(), source_term)
    decay(row) = Equation(row, TimeDerivative(), LinearTerm(row; coefficient=-1.0))
    @test EquationSystem(two, [source(:hot_tail_distribution, term), decay(:runaway_distribution)]) isa
          EquationSystem
    @test_throws DimensionMismatch EquationSystem(two, [decay(:hot_tail_distribution),
                                                        source(:runaway_distribution, term)])
    reads_other = MaxwellianSourceTerm(:runaway_distribution, g, temperature; target_density=target, rate,
                                       bulk_momentum=bound)
    @test_throws DimensionMismatch EquationSystem(two, [source(:hot_tail_distribution, reads_other),
                                                        decay(:runaway_distribution)])
    # a grid with the same faces built separately is accepted
    same_faces = cylindrical_grid(MomentumGrid(copy(mom.p_faces), copy(mom.xi_faces)), rad)
    on_copy = MaxwellianSourceTerm(:hot_tail_distribution, same_faces, temperature; target_density=target, rate,
                                   bulk_momentum=bound)
    @test EquationSystem(layout, [source(:hot_tail_distribution, on_copy)]) isa EquationSystem
end
