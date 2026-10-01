# Exponential fitting (Scharfetter-Gummel / Chang-Cooper) of the advection-diffusion flux:
# positivity for every Péclet number, the collisional Maxwellian steady
# to second order, second order at small Péclet number. Full collision mode with D and
# neutral Ar, the plasma of DREAM's thermalquench/iter.py (Mathias Hoppe, 2021: n_D 1e20,
# n_Ar 3e19, B 5.3 T), on the momentum grids of iter.py (200 cells to p = 2, 5 pitch cells)
# and runaway/basic.py (Mathias Hoppe, 300 cells to p = 1, 20 pitch cells).

using LinearAlgebra: I, dot
using SparseArrays: findnz

negative_couplings(L) = (IJV = findnz(L); count(k -> IJV[1][k] != IJV[2][k] && IJV[3][k] < 0, eachindex(IJV[3])))

@testset "exponential fitting" begin
    iter_grid = MomentumGrid(uniform_p_faces(200, 0.0, 2.0), uniform_xi_faces(5))
    basic_grid = MomentumGrid(uniform_p_faces(300, 0.0, 1.0), uniform_xi_faces(20))

    @testset "flux limits" begin
        # the Bernoulli function to round-off on both sides of the switch to its series
        for x in (-20.0, -0.3, -1.0001e-3, -0.9999e-3, 1e-7, 0.9999e-3, 1.0001e-3, 0.3, 20.0)
            @test Runes2P._bernoulli(x) ≈ Float64(big(x) / expm1(big(x))) rtol = 1e-14
        end
        @test Runes2P._bernoulli(0.0) == 1
        @test Runes2P._bernoulli(800.0) == 0          # no overflow
        # finite derivatives under ForwardDiff at Péclet numbers where eˣ overflows
        for w in (-800.0, 800.0)
            derivative = Runes2P.ForwardDiff.derivative(A -> Runes2P._face_flux(:exponential_fitting, A, 1.0, 2.0, 1.0, 0.1, 0.5),
                                                     w * 10)
            @test isfinite(derivative)
        end
        flux(A, D) = Runes2P._face_flux(:exponential_fitting, A, D, 2.0, 1.0, 0.1, 0.5)
        # small Péclet number: the central flux with midpoint weights
        @test flux(1e-6, 1.0) ≈ 1e-6 * 1.5 - 1.0 * (1.0 - 2.0) / 0.1 rtol = 1e-12
        # large Péclet number and D = 0: upwind
        @test flux(1e4, 1e-3) ≈ 1e4 * 2.0 rtol = 1e-6
        @test flux(-1e4, 1e-3) ≈ -1e4 * 1.0 rtol = 1e-6
        @test flux(3.0, 0.0) == 3.0 * 2.0
        # exact for the steady exponential profile: zero flux when f_R/f_L = exp(w)
        w = 2.7
        @test Runes2P._face_flux(:exponential_fitting, w * 0.5 / 0.1, 0.5, 1.0, exp(w), 0.1, 0.5) ≈ 0 atol = 1e-12
        # AD through the flux (the stored Jacobian comes from ForwardDiff)
        g, op = iter_plasma_operator(MomentumGrid(uniform_p_faces(20, 0.0, 2.0), uniform_xi_faces(5)), 100.0, 1.0;
                                     scheme=:exponential_fitting)
        @test check_stencil_pattern(op)
    end

    @testset "no negative couplings in full collision mode" begin
        # The central scheme loses positivity where the cell Péclet number exceeds 2;
        # exponential fitting keeps every off-diagonal coupling non-negative, including
        # the rows at p = 0 and p_max (observed for central: 407 to 5920 negative couplings
        # between 1 keV and 10 eV, none on the basic.py grid at 1 keV without field)
        for mom in (iter_grid, basic_grid), temperature in (1e3, 100.0, 10.0), field in (0.0, 1.0)
            _, central = iter_plasma_operator(mom, temperature, field; scheme=:central)
            _, fitted = iter_plasma_operator(mom, temperature, field; scheme=:exponential_fitting)
            @test negative_couplings(operator_matrix(fitted)) == 0
            temperature <= 100.0 && @test negative_couplings(operator_matrix(central)) > 0
        end
        # I/Δt - L is then an M-matrix: its inverse is non-negative for every Δt, so a
        # backward-Euler step keeps f ≥ 0 (observed minimum -2.6e-21 against a maximum 0.09)
        small = MomentumGrid(uniform_p_faces(40, 0.0, 2.0), uniform_xi_faces(5))
        for dt in (1e-6, 1e-3)
            _, fitted = iter_plasma_operator(small, 100.0, 1.0; scheme=:exponential_fitting)
            inverse = inv(I / dt - Matrix(operator_matrix(fitted)))
            @test minimum(inverse) >= -1e-15 * maximum(inverse)
        end
        _, central = iter_plasma_operator(small, 100.0, 1.0; scheme=:central)
        inverse = inv(I / 1e-3 - Matrix(operator_matrix(central)))
        @test minimum(inverse) < -0.1 * maximum(inverse)
    end

    @testset "backward Euler keeps f non-negative" begin
        # a 1 keV Maxwellian relaxing on a 100 eV plasma with E = 1 V/m, 40 steps of 50 μs,
        # on the grid of runaway/basic.py (observed with central: 26 112 negative values
        # over the saved states, 194 in the final one)
        function relax(scheme)
            g, op = iter_plasma_operator(basic_grid, 100.0, 1.0; scheme)
            layout = UnknownLayout(g.radial_grid, [KineticUnknown(:hot_tail_distribution, g; scale=1e20)])
            system = EquationSystem(layout, [Equation(:hot_tail_distribution, TimeDerivative(),
                                                      FokkerPlanckTerm(:hot_tail_distribution, op))])
            theta = 1e3 / Constants.ELECTRON_REST_ENERGY
            f0 = [exp(-(sqrt(1 + p^2) - 1) / theta) for p in basic_grid.p_centers, _ in 1:20, _ in 1:1]
            f0 .*= 1e20 / Runes2P.density(g, f0)[1]
            solution = integrate(system, vec(f0), collect(range(0, 2e-3; length=41)))
            return count(<(0), solution.states)
        end
        @test relax(:exponential_fitting) == 0
        @test relax(:central) > 0
    end

    @testset "Maxwell-Jüttner steady state" begin
        # Full collisions with E = 0: the discrete steady state (the null vector of L at
        # the Maxwellian's density) against Maxwell-Jüttner. Not exact, since A/D varies
        # over a cell, but second order and far below the central scheme's error
        # (observed 1.9e-5 and 4.6e-6 against 4.1e-2 and 1.0e-2 at 100 eV)
        function steady_state_error(cells, scheme)
            mom = MomentumGrid(uniform_p_faces(cells, 0.0, 1.0), uniform_xi_faces(4))
            g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 2.0)))
            plasma = PlasmaState(cold_electron_temperature=[100.0], cold_electron_density=[1e20],
                                 ions=[IonSpecies("D", 1, 1, [1e20])], electric_field=[0.0], magnetic_field=[5.3])
            op = fokker_planck_operator(g, plasma, CollisionSettings(mode=:full, screening=:complete,
                                                                     coulomb_logarithm=:thermal);
                                        scheme, p_upper_boundary=:f_zero)
            L = operator_matrix(op)
            theta = 100.0 / Constants.ELECTRON_REST_ENERGY
            maxwellian = vec([exp(-(sqrt(1 + p^2) - 1) / theta) for p in mom.p_centers, _ in 1:4])
            weights = vec(phase_space_jacobian(g, 1) .* mom.p_widths .* mom.xi_widths')
            A = copy(L)
            A[1, :] .= weights
            b = zeros(size(L, 1))
            b[1] = dot(weights, maxwellian)
            return maximum(abs, A \ b .- maxwellian) / maximum(maxwellian)
        end
        fitted = [steady_state_error(n, :exponential_fitting) for n in (200, 400)]
        central = [steady_state_error(n, :central) for n in (200, 400)]
        @test fitted[1] < 1e-4
        @test fitted[1] / fitted[2] > 3.5
        @test all(central .> 100 .* fitted)
    end

    @testset "manufactured solutions" begin
        # second order at small Péclet number, as the central scheme, in the 2D
        # manufactured step of test_manufactured_solutions.jl (observed orders 1.9-2.0)
        r = [manufactured_errors(biuniform_p_faces(N, N ÷ 5, 0.0, 0.7, 30.0), N ÷ 25; scheme=:exponential_fitting)
             for N in (200, 400, 800)]
        @test log2(r[2].max / r[3].max) > 1.85
        @test log2(r[2].l1 / r[3].l1) > 1.85
    end

    @testset "default of fokker_planck_operator" begin
        # exponential fitting is the default again, after van Albada; van Albada and DREAM's
        # centred interpolation are chosen explicitly
        mom = MomentumGrid(uniform_p_faces(10, 0.0, 1.0), uniform_xi_faces(4))
        g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
        plasma = PlasmaState(cold_electron_temperature=[100.0], cold_electron_density=[1e20],
                             ions=[IonSpecies("D", 1, 1, [1e20])], electric_field=[1.0], magnetic_field=[5.0])
        op = fokker_planck_operator(g, plasma)
        @test op.p_scheme === :exponential_fitting && op.xi_scheme === :exponential_fitting
        @test fokker_planck_operator(g, plasma; scheme=:van_albada).p_scheme === :van_albada
        @test fokker_planck_operator(g, plasma; scheme=:central).p_scheme === :central
    end
end
