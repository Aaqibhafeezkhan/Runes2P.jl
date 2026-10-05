# The rate equations of the ion charge states: the conservation of each
# species, the Jacobian against a central difference of the residual, the coronal
# equilibrium three ways (the recurrence in log space, the null vector of the rate matrix by
# a dense solve, and the rows of the term run to their steady state), the particle budget of
# a deposition at a fixed temperature, and a trace impurity converged on its own scale.
# Needs the ADF11 cache (scripts/fetch_adas.py).
using LinearAlgebra: nullspace, eigvals

const RATE_COMPOSITION = IonComposition(["D" => 1, "Ar" => 18])

# ion densities and a prescribed temperature and electron density, or the electron density
# as the closure of the ions
function _rate_system(; nr=2, temperature=[20.0, 5.0], electron_density=nothing)
    radial = RadialGrid(uniform_r_faces(nr, 0.5))
    layout = UnknownLayout(radial, [ion_densities_unknown(RATE_COMPOSITION),
                                    RadialUnknown(:cold_electron_density; scale=1e19),
                                    RadialUnknown(:cold_electron_temperature; scale=10.0)])
    hold(name, value) = Equation(name, Algebraic(), PrescribedValueTerm(value), LinearTerm(name; coefficient=-1))
    density = electron_density === nothing ? population_equations(RATE_COMPOSITION) :
              [hold(:cold_electron_density, electron_density)]
    equations = [Equation(:ion_densities, TimeDerivative(), IonRateTerm(RATE_COMPOSITION)), density...,
                 hold(:cold_electron_temperature, temperature)]
    return layout, equations
end

function _rate_state(layout; deuterium=2.6e19, argon=2.6e19, temperature=[20.0, 5.0], electron_density=nothing)
    nr = radial_cell_count(layout)
    u = zeros(length(layout))
    set_ion_densities!(u, layout, [IonSpecies("D", 1, 1, fill(deuterium, nr)), IonSpecies("Ar", 18, zeros(19, nr))])
    deposit_neutrals!(u, layout, RATE_COMPOSITION, "D" => deuterium, "Ar" => argon)
    radial_values(u, layout, 2) .= electron_density === nothing ? deuterium : electron_density
    radial_values(u, layout, 3) .= temperature
    return u
end

@testset "ion rates" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the ion rate tests are skipped"
    else
        term = IonRateTerm(RATE_COMPOSITION)
        @test_throws ArgumentError IonRateTerm(IonComposition(["Xe" => 54]))
        @test_throws ArgumentError IonRateTerm(IonComposition(["Ar" => 17]))
        @test_throws ArgumentError IonRateTerm(RATE_COMPOSITION; extrapolation=:none)

        @testset "rate matrix and conservation" begin
            for (n_e, T) in ((1e19, 2.0), (1e20, 30.0), (5e19, 500.0))
                A = ion_rate_matrix(term, 2, n_e, T)
                @test size(A) == (19, 19)
                @test maximum(abs, sum(A; dims=1)) <= 1e-12 * maximum(abs, A)
                # ionization moves up, recombination down: a tridiagonal matrix
                @test all(iszero(A[i, j]) for i in 1:19, j in 1:19 if abs(i - j) > 1)
                @test all(A[j+1, j] > 0 for j in 1:18) && all(A[j, j+1] > 0 for j in 1:18)
            end
            layout, equations = _rate_system()
            system = EquationSystem(layout, equations)
            u = _rate_state(layout)
            u[unknown_range(layout, 1, 1)] .+= 1e17 .* (1:21)
            du = Runes2P.rhs!(zeros(length(u)), u, system, 0.0)
            for ir in 1:2, s in 1:2
                rates = view(du, unknown_range(layout, 1, ir)[species_components(RATE_COMPOSITION, s)])
                @test abs(sum(rates)) <= 1e-12 * maximum(abs, rates)
            end
        end

        @testset "Jacobian against a central difference" begin
            layout, equations = _rate_system()
            system = EquationSystem(layout, equations)
            u = _rate_state(layout)
            u[unknown_range(layout, 1, 2)] .+= 1e16 .* (1:21)
            J = Runes2P.jacobian_prototype(system)
            jacobian!(J, system, u, 0.0)
            rhs(x) = Runes2P.rhs!(zeros(length(x)), x, system, 0.0)
            scales = unknown_scales(layout)
            ions = Set(i for ir in 1:2 for i in unknown_range(layout, 1, ir))
            size_of_rhs = maximum(abs, rhs(u))
            for j in eachindex(u)
                # the rows are linear in the ion densities: a large step is exact there, and a
                # small one for the temperature and the density, where the coefficients bend
                h = j in ions ? 1e-3 * scales[j] : 1e-6 * abs(u[j])
                up, down = copy(u), copy(u)
                up[j] += h
                down[j] -= h
                column = (rhs(up) - rhs(down)) / 2h
                # truncation of the central difference, and the round-off of the residual
                @test maximum(abs, J[:, j] - column) <= 1e-6 * maximum(abs, column) + 1e-14 * size_of_rhs / h
            end
        end

        @testset "coronal equilibrium three ways" begin
            for (n_e, T) in ((1e20, 3.0), (1e20, 30.0), (1e19, 300.0))
                coronal = coronal_equilibrium(term, 2, n_e, T; density=1e18)
                @test sum(coronal) ≈ 1e18 rtol = 1e-12
                # the null vector of the rate matrix
                null = vec(nullspace(ion_rate_matrix(term, 2, n_e, T) ./ (n_e * 1e-14)))
                null .*= 1e18 / sum(null)
                @test maximum(abs, null - coronal) <= 1e-9 * 1e18
            end
            # the rows of the term, held at n_e and T until they stop changing: ten backward
            # Euler steps of 30 times the slowest relaxation time of either radius, each of
            # which leaves 1/31 of the distance to the steady state
            n_e, temperatures = 1e20, [3.0, 30.0]
            # the second smallest magnitude: the smallest is the zero of the conservation
            slowest = minimum(T -> sort(abs.(eigvals(ion_rate_matrix(term, 2, n_e, T))))[2], temperatures)
            layout, equations = _rate_system(; temperature=temperatures, electron_density=n_e)
            system = EquationSystem(layout, equations)
            u = _rate_state(layout; argon=1e18, temperature=temperatures, electron_density=n_e)
            solution = integrate(system, u, collect(range(0.0, 300 / slowest; length=11)))
            final = solution.states[:, end]
            for (ir, T) in enumerate(temperatures)
                argon = final[unknown_range(layout, 1, ir)[species_components(RATE_COMPOSITION, "Ar")]]
                @test maximum(abs, argon - coronal_equilibrium(term, 2, n_e, T; density=1e18)) <= 1e-9 * 1e18
            end
        end

        @testset "particle budget of a deposition" begin
            layout, equations = _rate_system()
            u = _rate_state(layout)
            system = EquationSystem(layout, equations; time_scale=1e-7, initial_state=u)
            times = collect(range(0.0, 20e-6; length=101))
            solution = integrate(system, u, times; stepper=FixedSteps())
            @test run_status(solution) === :finished
            for k in eachindex(times), ir in 1:2
                state = solution.states[:, k]
                ions = ion_densities(state, layout, RATE_COMPOSITION)
                @test sum(ions[1].charge_state_densities[:, ir]) ≈ 2 * 2.6e19 rtol = 1e-12
                @test sum(ions[2].charge_state_densities[:, ir]) ≈ 2.6e19 rtol = 1e-12
                @test radial_values(state, layout, 2)[ir] ≈ free_electron_density(ions, ir) rtol = 1e-12
                @test all(>=(-1e-10 * 2.6e19), ions[2].charge_state_densities)
            end
            # the electrons of the ionization: the hotter radius ionizes further
            final = ion_densities(solution.states[:, end], layout, RATE_COMPOSITION)
            @test free_electron_density(final, 1) > free_electron_density(final, 2) > 2.6e19
        end

        @testset "step controller on the ionization" begin
            # trace argon: its states set the step by the argon density, not the scale of the
            # unknown (the absolute part of a grouped unknown is 0)
            layout, equations = _rate_system()
            u = _rate_state(layout; argon=2.6e15)
            system = EquationSystem(layout, equations; time_scale=1e-7, initial_state=u)
            controlled = integrate(system, u, [0.0, 20e-6]; stepper=StepController(; relative_tolerance=1e-4))
            @test run_status(controlled) === :finished
            reference = integrate(system, u, collect(range(0.0, 20e-6; length=2001)); stepper=FixedSteps())
            argon = unknown_range(layout, 1, 1)[species_components(RATE_COMPOSITION, "Ar")]
            # backward Euler at the controlled steps against 2000 steps: the argon states agree
            # to the order of the tolerance on the argon density, not on 1e19
            @test maximum(abs, controlled.states[argon, end] - reference.states[argon, end]) <= 1e-2 * 2.6e15
            for ir in 1:2
                ions = ion_densities(controlled.states[:, end], layout, RATE_COMPOSITION)
                @test sum(ions[2].charge_state_densities[:, ir]) ≈ 2.6e15 rtol = 1e-10
            end
        end

        @testset "trace argon converged on its own scale" begin
            # at a prescribed electron density the argon is linear in its density: argon at
            # 1e-4 of the deuterium density is 1e-4 of the argon at the deuterium density, to
            # the tolerance relative to the argon density
            n_e = 5e19
            layout, equations = _rate_system(; electron_density=n_e)
            times = collect(range(0.0, 20e-6; length=21))
            run(argon) = integrate(EquationSystem(layout, equations), _rate_state(layout; argon, electron_density=n_e),
                                   times; stepper=FixedSteps())
            trace, full = run(2.6e15), run(2.6e19)
            argon = unknown_range(layout, 1, 1)[species_components(RATE_COMPOSITION, "Ar")]
            for k in eachindex(times)
                @test maximum(abs, 1e4 .* trace.states[argon, k] - full.states[argon, k]) <= 1e-8 * 2.6e19
            end
        end
    end
end
