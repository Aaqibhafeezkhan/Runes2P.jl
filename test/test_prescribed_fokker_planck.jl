# Fokker-Planck coefficients that follow a prescribed, time-dependent plasma state, such as
# the temperature T(t) of a cooling plasma. Independent of DREAM: the references are a fresh
# operator built at the same state and the static FokkerPlanckTerm.
using ForwardDiff: ForwardDiff
using Random: Xoshiro

function _test_plasma(temperature; density = 1e20, field = 1.0, nr = 1)
    PlasmaState(cold_electron_temperature = fill(temperature, nr),
        cold_electron_density = fill(density, nr),
        ions = [IonSpecies("D", 1, 1, fill(density, nr))], electric_field = fill(field, nr),
        magnetic_field = fill(5.0, nr))
end

@testset "PrescribedFokkerPlanckTerm and set_fokker_planck_coefficients!" begin
    mom = MomentumGrid(uniform_p_faces(20, 0.0, 1.0), uniform_xi_faces(6))
    g = cylindrical_grid(mom, RadialGrid(uniform_r_faces(1, 1.0)))
    settings = CollisionSettings(mode = :full)

    @testset "set_fokker_planck_coefficients! matches a fresh operator" begin
        for T in (500.0, 100.0, 20.0)
            plasma = _test_plasma(T)
            fresh = fokker_planck_operator(g, plasma, settings)
            op = AdvectionDiffusion(g; scheme = :central, p_upper_boundary = :f_zero)
            set_fokker_planck_coefficients!(op, plasma, settings)
            for field in (:advection_p, :diffusion_p_p, :advection_xi, :diffusion_xi_xi,
                :advection_p_times_p2_at_zero)
                @test getfield(op, field) == getfield(fresh, field)
            end
        end
    end

    @testset "rebuilding twice at the same state does not accumulate" begin
        plasma = _test_plasma(200.0)
        op = AdvectionDiffusion(g; scheme = :central, p_upper_boundary = :f_zero)
        set_fokker_planck_coefficients!(op, plasma, settings)
        first_advection = copy(op.advection_p)
        set_fokker_planck_coefficients!(op, plasma, settings)
        @test op.advection_p == first_advection
    end

    @testset "rebuild is cached on t: a constant T(t) reproduces FokkerPlanckTerm bit for bit" begin
        op_static = fokker_planck_operator(g, _test_plasma(150.0), settings; scheme = :central)
        op_prescribed = AdvectionDiffusion(g; scheme = :central, p_upper_boundary = :f_zero)
        calls = Ref(0)
        plasma_at(t) = (calls[] += 1; _test_plasma(150.0))
        layout1 = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = 1.0)])
        layout2 = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = 1.0)])
        static_system = EquationSystem(
            layout1, [Equation(:f, TimeDerivative(), FokkerPlanckTerm(:f, op_static))])
        prescribed_term = PrescribedFokkerPlanckTerm(:f, op_prescribed, plasma_at; settings)
        prescribed_system = EquationSystem(layout2, [Equation(:f, TimeDerivative(), prescribed_term)])

        u0 = rand(Xoshiro(1), length(layout1))
        times = [0.0, 1e-3, 2e-3, 3e-3]
        static_solution = integrate(static_system, u0, times)
        prescribed_solution = integrate(prescribed_system, u0, times)
        @test static_solution.states ≈ prescribed_solution.states rtol = 1e-13

        # one rebuild per distinct t (plus the solver's own initial residual check at
        # t = 0), not one per colour or per Newton iteration; without the cache this
        # would run into the dozens for 3 steps (colours x Newton iterations). The record
        # of the run samples the plasma once more at every time point.
        @test calls[] <= 2 * length(times)
    end

    @testset "a plasma state defined only on the run's time interval" begin
        # the construction evaluates the term once, at initial_time
        plasma_from_one(t) = t >= 1 ? _test_plasma(300.0 - 1e3 * (t - 1)) :
                             throw(ArgumentError("no plasma state before t = 1"))
        op = AdvectionDiffusion(g; scheme = :central, p_upper_boundary = :f_zero)
        term = PrescribedFokkerPlanckTerm(:f, op, plasma_from_one; settings)
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = 1.0)])
        @test_throws ArgumentError EquationSystem(layout, [Equation(:f, TimeDerivative(), term)])
        system = EquationSystem(layout, [Equation(:f, TimeDerivative(), term)]; initial_time = 1.0)
        solution = integrate(system, rand(Xoshiro(3), length(layout)), [1.0, 1.001, 1.002])
        @test all(solution.statistics.accepted)
    end

    @testset "EquationSystem Jacobian" begin
        plasma_at(t) = _test_plasma(300.0 - 1e5 * t)  # an arbitrary time dependence
        op = AdvectionDiffusion(g; scheme = :central, p_upper_boundary = :f_zero)
        term = PrescribedFokkerPlanckTerm(:f, op, plasma_at; settings)
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = 1.0)])
        system = EquationSystem(layout, [Equation(:f, TimeDerivative(), term)])
        u = rand(Xoshiro(2), length(layout))
        J = jacobian_prototype(system)
        jacobian!(J, system, u, 1e-3)
        function sparse_part(x)
            out = zeros(eltype(x), length(x))
            for bound in system.sparse_terms
                Runes2P.rhs!(out, bound.term, x, layout, bound.row, 1e-3)
            end
            return out
        end
        dense = ForwardDiff.jacobian(sparse_part, u)
        @test Matrix(J) ≈ dense rtol = 1e-10
        @test all(jacobian_sparsity(system)[findall(!iszero, dense)])
    end

    @testset "validate delegates to FokkerPlanckTerm" begin
        op = AdvectionDiffusion(g; scheme = :central, p_upper_boundary = :f_zero)
        term = PrescribedFokkerPlanckTerm(:f, op, t -> _test_plasma(100.0); settings)
        wrong_layout = UnknownLayout(g.radial_grid, [ScalarUnknown(:f)])
        @test_throws ArgumentError Runes2P.validate(term, wrong_layout, 1)
    end
end
