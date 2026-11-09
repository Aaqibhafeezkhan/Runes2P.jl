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

@testset "prescribed_fokker_planck_term and set_fokker_planck_coefficients!" begin
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
        prescribed_term = prescribed_fokker_planck_term(:f, op_prescribed, plasma_at; settings)
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
        term = prescribed_fokker_planck_term(:f, op, plasma_from_one; settings)
        layout = UnknownLayout(g.radial_grid, [KineticUnknown(:f, g; scale = 1.0)])
        @test_throws ArgumentError EquationSystem(layout, [Equation(:f, TimeDerivative(), term)])
        system = EquationSystem(layout, [Equation(:f, TimeDerivative(), term)]; initial_time = 1.0)
        solution = integrate(system, rand(Xoshiro(3), length(layout)), [1.0, 1.001, 1.002])
        @test all(solution.statistics.accepted)
    end

    @testset "EquationSystem Jacobian" begin
        plasma_at(t) = _test_plasma(300.0 - 1e5 * t)  # an arbitrary time dependence
        op = AdvectionDiffusion(g; scheme = :central, p_upper_boundary = :f_zero)
        term = prescribed_fokker_planck_term(:f, op, plasma_at; settings)
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
        term = prescribed_fokker_planck_term(:f, op, t -> _test_plasma(100.0); settings)
        wrong_layout = UnknownLayout(g.radial_grid, [ScalarUnknown(:f)])
        @test_throws ArgumentError Runes2P.validate(term, wrong_layout, 1)
    end

    # The rebuild writes the friction and the energy diffusion in one sweep that evaluates ν_s
    # once per face. The reference writes every coefficient from the public frequency
    # functions face by face, in the order of the terms; all arrays agree bit for bit. The
    # plasma holds partially ionized argon and the bremsstrahlung stopping, so that the
    # collisional part of ν_s differs from the whole. The values and the partials of the dual
    # coefficients are compared on their own.
    @testset "one sweep of ν_s and ν_∥ against the frequencies face by face" begin
        nr = 2
        argon = zeros(19, nr)
        argon[1, :] = [1e18, 2e17]
        argon[2, :] = [3e18, 1e18]
        argon[4, :] = [5e17, 4e18]
        deuterium = [1e19 2e18; 9e19 1e20]
        plasma = PlasmaState(cold_electron_temperature = [50.0, 5.0],
            cold_electron_density = [1e20, 1.2e20],
            ions = [IonSpecies("D", 1, deuterium), IonSpecies("Ar", 18, argon)],
            electric_field = [2.0, 30.0], magnetic_field = [5.0, 5.0])
        xi = uniform_xi_faces(4)
        radii = RadialGrid(uniform_r_faces(nr, 1.0))
        grids = (cylindrical_grid(MomentumGrid(uniform_p_faces(12, 0.0, 2.0), xi), radii),
            cylindrical_grid(MomentumGrid(uniform_p_faces(8, 2.0, 20.0), xi), radii))
        arrays = (:advection_p, :diffusion_p_p, :diffusion_p_xi, :advection_xi,
            :diffusion_xi_xi, :diffusion_xi_p, :advection_r, :diffusion_r_r,
            :advection_p_times_p2_at_zero)
        function reference!(op, plasma, settings)
            zero_coefficients!(op)
            g = op.grid
            np, nxi, nr = size(g)
            faces = [momentum_grid(g, ir).p_faces for ir in 1:nr]
            model = CollisionModel(plasma, settings; faces)
            add_electric_field!(op, plasma)
            for ir in 1:nr
                m = momentum_grid(g, ir)
                for k in 1:(np + 1), j in 1:nxi

                    p = m.p_faces[k]
                    op.advection_p[k, j, ir] += -p * slowing_down_frequency(model, ir, p)
                end
                if m.p_faces[1] == 0
                    op.advection_p_times_p2_at_zero[:, ir] .-= p3_slowing_down_frequency_at_zero(
                        model, ir)
                end
                for i in 1:np, k in 1:(nxi + 1)

                    nu = deflection_frequency(model, ir, m.p_centers[i])
                    op.diffusion_xi_xi[i, k, ir] += nu * (1 - m.xi_faces[k]^2) / 2
                end
                if settings.mode === :full
                    for k in 1:(np + 1), j in 1:nxi

                        op.diffusion_p_p[k, j, ir] += parallel_diffusion_frequency(
                            model, ir, m.p_faces[k])
                    end
                end
            end
            add_synchrotron!(op, plasma)
            return op
        end
        value_parts(A) = ForwardDiff.value.(A)
        partial_parts(A) = [ForwardDiff.partials(x).values for x in A]
        dual = Runes2P.dual_plasma(plasma, true, true)
        D = eltype(dual.cold_electron_temperature)
        for mode in (:full, :superthermal), bremsstrahlung_diffusion in (:dream, :none),
            screened_diffusion in (:maxwellian, :zero), g in grids
            settings = CollisionSettings(;
                mode, screening = :partial, bremsstrahlung = :stopping_power,
                bremsstrahlung_diffusion, screened_diffusion)
            built = set_fokker_planck_coefficients!(AdvectionDiffusion(g), plasma, settings)
            expected = reference!(AdvectionDiffusion(g), plasma, settings)
            for name in arrays
                @test isequal(getfield(built, name), getfield(expected, name))
            end
            dual_built = set_fokker_planck_coefficients!(
                Runes2P.similar_coefficients(AdvectionDiffusion(g), D), dual, settings)
            dual_expected = reference!(
                Runes2P.similar_coefficients(AdvectionDiffusion(g), D), dual, settings)
            for name in arrays
                @test isequal(value_parts(getfield(dual_built, name)),
                    value_parts(getfield(dual_expected, name)))
                @test isequal(partial_parts(getfield(dual_built, name)),
                    partial_parts(getfield(dual_expected, name)))
            end
        end

        # the settings reach the branches they name: the energy diffusion takes the
        # radiative part of ν_s with :dream only, and the flux through p = 0 is written (it
        # vanishes in :full mode, whose thermal factor is zero at p = 0)
        g = grids[1]
        diffusion(bremsstrahlung_diffusion) = set_fokker_planck_coefficients!(
            AdvectionDiffusion(g), plasma,
            CollisionSettings(; bremsstrahlung_diffusion)).diffusion_p_p
        @test diffusion(:dream) != diffusion(:none)
        @test !iszero(set_fokker_planck_coefficients!(AdvectionDiffusion(g), plasma,
            CollisionSettings(mode = :superthermal)).advection_p_times_p2_at_zero)
        @test any(x -> any(!iszero, x),
            partial_parts(set_fokker_planck_coefficients!(
                Runes2P.similar_coefficients(AdvectionDiffusion(g), D), dual,
                CollisionSettings()).diffusion_p_p))
    end
end
