# The radial diffusion of the ion charge states: conservation of
# each state with a zero flux at the edge, the slowest decay with an absorbing edge against
# the Bessel rate of the cylinder, the neutrals untouched by the charged diffusion, and the
# transport history of a disruption run, which switches the diffusivity at a forced time
# point.
using LinearAlgebra: eigvals

const TRANSPORT_COMPOSITION = IonComposition(["D" => 1, "Ar" => 18])

function _transport_case(term; nr=12, a=1.0)
    radial = RadialGrid(uniform_r_faces(nr, a))
    layout = UnknownLayout(radial, [ion_densities_unknown(TRANSPORT_COMPOSITION)])
    system = EquationSystem(layout, [Equation(:ion_densities, TimeDerivative(), term)])
    u = zeros(length(layout))
    for ir in 1:nr
        r = radial.r_centers[ir]
        radial_view(u, layout, 1, ir) .= [(1 + c) * 1e17 * (1 + cos(π * r / a)) for c in 1:21]
    end
    return layout, system, u
end

# the content Σ V' Δr n of component c
_content(layout, u, c) = sum(layout.radial_grid.spatial_jacobian[ir] * layout.radial_grid.r_widths[ir] *
                             radial_view(u, layout, 1, ir)[c] for ir in 1:radial_cell_count(layout))

@testset "ion transport" begin
    @testset "zero flux: each state conserved, neutrals untouched" begin
        term = IonDiffusionTerm(TRANSPORT_COMPOSITION; diffusion=1.0)
        layout, system, u = _transport_case(term)
        solution = integrate(system, u, collect(range(0.0, 0.05; length=11)))
        final = solution.states[:, end]
        for c in 1:21
            @test _content(layout, final, c) ≈ _content(layout, u, c) rtol = 1e-12
        end
        # the neutrals of both species keep their profiles, the charged states flatten
        for c in (1, 3)
            @test [radial_view(final, layout, 1, ir)[c] for ir in 1:12] == [radial_view(u, layout, 1, ir)[c] for ir in 1:12]
        end
        spread(x, c) = radial_view(x, layout, 1, 1)[c] - radial_view(x, layout, 1, 12)[c]
        @test spread(final, 2) < 0.5 * spread(u, 2)
        # all states with states = :all
        every = IonDiffusionTerm(TRANSPORT_COMPOSITION; diffusion=1.0, states=:all)
        layout, system, u = _transport_case(every)
        final = integrate(system, u, [0.0, 0.05]).states[:, end]
        @test spread(final, 1) < spread(u, 1)
        @test_throws ArgumentError IonDiffusionTerm(TRANSPORT_COMPOSITION)
        @test_throws ArgumentError IonDiffusionTerm(TRANSPORT_COMPOSITION; diffusion=1.0, states=:neutral)
        @test_throws ArgumentError IonDiffusionTerm(TRANSPORT_COMPOSITION; diffusion=1.0, edge=:open)
    end

    @testset "absorbing edge: the slowest decay of the cylinder" begin
        D, a, nr = 2.0, 0.5, 40
        term = IonDiffusionTerm(TRANSPORT_COMPOSITION; diffusion=D, edge=:absorbing)
        layout, system, u = _transport_case(term; nr, a)
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 0.0)
        # the block of one charged state of argon
        rows = [unknown_range(layout, 1, ir)[5] for ir in 1:nr]
        slowest = minimum(abs, eigvals(Matrix(J[rows, rows])))
        exact = D * 2.404825557695773^2 / a^2
        @test slowest ≈ exact rtol = 1e-2
        @test all(<(0), real(eigvals(Matrix(J[rows, rows]))))
        # the Jacobian of the term against a central difference of its residual
        rhs_all(x) = (out = zeros(length(x)); rhs!(out, x, system, 0.0); out)
        j = rows[7]
        up, down = copy(u), copy(u)
        up[j] += 1e15
        down[j] -= 1e15
        @test Vector(J[:, j]) ≈ (rhs_all(up) - rhs_all(down)) / 2e15 rtol = 1e-8 atol = 1e-8 * maximum(abs, J[:, j])
    end

    @testset "transport history: 1000 m²/s, then a hundredth" begin
        history = PiecewiseConstant([1000.0, 10.0], [1e-3])
        term = IonDiffusionTerm(TRANSPORT_COMPOSITION; diffusion_at=history)
        @test forced_time_points(term) == [1e-3]
        layout, system, u = _transport_case(term)
        # the switch must be a time point of the run
        @test_throws ArgumentError integrate(system, u, [0.0, 2e-3])
        solution = integrate(system, u, collect(range(0.0, 2e-3; length=5)))
        @test run_status(solution) === :finished
        # the diffusivity of each interval: a backward-Euler step of 5e-4 s divides the slowest
        # mode of the zero-flux cylinder, rate D j₁₁²/a², by 1 + 7.3 before the switch and by
        # 1 + 0.073 after it
        spread(x) = radial_view(x, layout, 1, 1)[2] - radial_view(x, layout, 1, 12)[2]
        states = solution.states
        early = spread(states[:, 2]) / spread(states[:, 1])
        late = spread(states[:, 5]) / spread(states[:, 4])
        @test early < 0.5
        @test 0.9 < late < 0.97
        @test Runes2P.describe(term, Runes2P.DescriptionContext())["diffusion"] == "PiecewiseConstant"
    end
end
