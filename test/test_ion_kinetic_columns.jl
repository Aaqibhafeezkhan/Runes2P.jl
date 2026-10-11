# The kinetic terms with the ion charge states from the unknowns: the
# columns of the charge states in the Fokker-Planck term and in the energy loss of the
# electrons against central differences, the lag of the ion group (no columns, the ions of
# the start of the step), the target of the e-e collisions of a two-grid system with the
# population closure, and the Newton iterations of a stiff ionization step with
# the ion group implicit and lagged.

using SparseArrays: sparse, nnz

const KINETIC_COMPOSITION = IonComposition(["D" => 1, "Ar" => 18])

function _kinetic_ion_case(; lagged=(), temperatures=[5.0, 3.0],
                           settings=CollisionSettings(; mode=:full, screening=:partial))
    hot = cylindrical_grid(MomentumGrid(uniform_p_faces(24, 0.0, 0.05), uniform_xi_faces(3)),
                           RadialGrid(uniform_r_faces(2, 0.5)))
    argon = zeros(19, 2)
    argon[1, :] .= [2e17, 4e17]
    argon[2, :] .= [5e17, 3e17]
    argon[3, :] .= [1e17, 0.0]
    ions = [IonSpecies("D", 1, [1e18 2e18; 5e19 4e19]), IonSpecies("Ar", 18, argon)]
    prescribed = PlasmaState(; cold_electron_temperature=temperatures, ions, electric_field=zeros(2),
                             magnetic_field=fill(5.0, 2))
    layout = UnknownLayout(hot.radial_grid, [ion_densities_unknown(KINETIC_COMPOSITION),
                                             RadialUnknown(:cold_electron_density; scale=1e19),
                                             RadialUnknown(:cold_electron_temperature; scale=10.0),
                                             KineticUnknown(:f, hot; scale=1e19)])
    source = PlasmaFromUnknowns(prescribed; ions=:ion_densities)
    fp = FokkerPlanckTerm(:f, AdvectionDiffusion(hot; p_upper_boundary=:f_zero); plasma=source,
                          settings, lagged)
    u = zeros(length(layout))
    set_ion_densities!(u, layout, ions)
    radial_values(u, layout, 2) .= [free_electron_density(ions, ir) for ir in 1:2]
    radial_values(u, layout, 3) .= temperatures
    theta = temperatures ./ Constants.ELECTRON_REST_ENERGY
    for ir in 1:2
        kinetic_view(u, layout, 4, ir) .= [1e19 * exp(-p^2 / (2theta[ir])) * (1 + 0.1xi)
                                           for p in momentum_grid(hot, ir).p_centers, xi in momentum_grid(hot, ir).xi_centers]
    end
    return layout, fp, u
end

_hold(name, value) = Equation(name, Algebraic(), PrescribedValueTerm(value), LinearTerm(name; coefficient=-1))

@testset "ion kinetic columns" begin
    @testset "the composition is read at the initial time" begin
        # a prescribed part defined only from t = 1 on: the check of the ion unknown reads
        # it at the initial time of the system, not at t = 0
        layout, fp, u = _kinetic_ion_case()
        prescribed = Runes2P.plasma_values(fp.plasma.prescribed, nothing, nothing, 0.0)
        from_one(t) = t >= 1 ? prescribed : throw(ArgumentError("no plasma state before t = 1"))
        late = FokkerPlanckTerm(:f, fp.operator; plasma=PlasmaFromUnknowns(from_one; ions=:ion_densities),
                                settings=fp.settings)
        row = unknown_id(layout, :f)
        @test Runes2P.validate(late, layout, row, 1.0) === nothing
        @test_throws ArgumentError Runes2P.validate(late, layout, row, 0.0)
    end

    @testset "Fokker-Planck columns of the charge states" begin
        layout, fp, u = _kinetic_ion_case()
        equations = [Equation(:ion_densities, TimeDerivative()), population_equations(KINETIC_COMPOSITION)...,
                     _hold(:cold_electron_temperature, [5.0, 3.0]), Equation(:f, TimeDerivative(), fp)]
        system = EquationSystem(layout, equations; initial_state=u)
        @test Runes2P.has_plasma_columns(fp)
        J = jacobian_prototype(system)
        du = zeros(length(u))
        rhs!(du, u, system, 0.0)
        jacobian!(J, system, u, 0.0)
        f_rows = [i for ir in 1:2 for i in unknown_range(layout, 4, ir)]
        size_of_rhs = maximum(abs, du[f_rows])
        rhs_f(x) = (out = zeros(length(x)); rhs!(out, x, system, 0.0); out[f_rows])
        for ir in 1:2, (k, column) in enumerate(plasma_columns(fp.plasma, layout, ir))
            h = k == 1 ? 1e-6 * u[column] : 1e-6 * 1e19
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            difference = (rhs_f(up) - rhs_f(down)) / 2h
            @test maximum(abs, Vector(J[f_rows, column]) - difference) <=
                  1e-5 * maximum(abs, difference) + 1e-12 * size_of_rhs / h
        end
        # the pattern holds the columns of the charge states
        @test nnz(J[f_rows, [i for ir in 1:2 for i in unknown_range(layout, 1, ir)]]) > 0
        # a charge state at zero density: the column is the derivative there, which the
        # screened collisions of argon make nonzero
        empty = unknown_range(layout, 1, 2)[2 + 3]
        @test u[empty] == 0 && any(!iszero, J[f_rows, empty])
    end

    @testset "Fokker-Planck columns with the friction tied to the ionization (option (d) of #177)" begin
        # the columns of the charge states and of the temperature, through the envelope of the
        # threshold at 5 thermal momenta (about 125 and 75 eV here, inside the grid) and the
        # floor of the excitation part, against central differences
        settings = CollisionSettings(; mode=:full, screening=:partial, bound_friction=:ionization_split,
                                     ionization_threshold=5.0)
        layout, fp, u = _kinetic_ion_case(; settings)
        grid = kinetic_grid(layout, 4)
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor=5))
        equations = [Equation(:ion_densities, TimeDerivative(), KineticIonizationTerm(KINETIC_COMPOSITION, :f, grid; region)),
                     population_equations(KINETIC_COMPOSITION)..., _hold(:cold_electron_temperature, [5.0, 3.0]),
                     Equation(:f, TimeDerivative(), fp)]
        system = EquationSystem(layout, equations; initial_state=u)
        J = jacobian_prototype(system)
        du = zeros(length(u))
        rhs!(du, u, system, 0.0)
        jacobian!(J, system, u, 0.0)
        f_rows = [i for ir in 1:2 for i in unknown_range(layout, 4, ir)]
        size_of_rhs = maximum(abs, du[f_rows])
        rhs_f(x) = (out = zeros(length(x)); rhs!(out, x, system, 0.0); out[f_rows])
        for ir in 1:2, (k, column) in enumerate(plasma_columns(fp.plasma, layout, ir))
            h = k == 1 ? 1e-6 * u[column] : 1e-6 * 1e19
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            difference = (rhs_f(up) - rhs_f(down)) / 2h
            @test maximum(abs, Vector(J[f_rows, column]) - difference) <=
                  1e-5 * maximum(abs, difference) + 1e-12 * size_of_rhs / h
        end
        # the split changes the operator above the threshold alone, where the Maxwellian at 5 eV
        # is about e⁻²⁵ of its bulk: the cells whose faces lie below the step of the faces are as
        # with the Bethe friction, the others change
        _, bethe, _ = _kinetic_ion_case()
        bethe_system = EquationSystem(layout, [equations[1:end-1]; Equation(:f, TimeDerivative(), bethe)]; initial_state=u)
        du_bethe = zeros(length(u))
        rhs!(du_bethe, u, bethe_system, 0.0)
        for ir in 1:2
            m = momentum_grid(grid, ir)
            p0 = 5 * sqrt(2 * [5.0, 3.0][ir] / Constants.ELECTRON_REST_ENERGY)
            below = [i for i in 1:length(m.p_centers) if m.p_faces[i+1] < 0.9p0]
            @test !isempty(below) && length(below) < length(m.p_centers)
            cells(i) = [Runes2P._kinetic_index(layout, 4, ir, i, j) for j in 1:3]
            @test all(du[cells(i)] == du_bethe[cells(i)] for i in below)
            @test any(du[cells(i)] != du_bethe[cells(i)] for i in setdiff(1:length(m.p_centers), below))
        end
    end

    @testset "the default lags the ion densities" begin
        layout, fp, _ = _kinetic_ion_case()
        default = FokkerPlanckTerm(:f, fp.operator; plasma=fp.plasma, settings=fp.settings)
        @test default.lagged == (:ion_densities,) && Runes2P._lags_ions(default)
        @test fp.lagged == ()      # the case asks for the implicit charge states
        # a plasma without ion unknowns, and lagged coefficients, lag no group by name
        prescribed = PrescribedFokkerPlanckTerm(:f, fp.operator, Runes2P.plasma_values(fp.plasma.prescribed, nothing, nothing, 0.0);
                                                settings=fp.settings)
        @test prescribed.lagged == ()
        @test FokkerPlanckTerm(:f, fp.operator; plasma=fp.plasma, settings=fp.settings, coefficients=:lagged).lagged == ()
        @test_throws ArgumentError FokkerPlanckTerm(:f, fp.operator; plasma=fp.plasma, settings=fp.settings, lagged=:other)
    end

    @testset "lagged ion group" begin
        layout, fp, u = _kinetic_ion_case(; lagged=(:ion_densities,))
        equations = [Equation(:ion_densities, TimeDerivative()), population_equations(KINETIC_COMPOSITION)...,
                     _hold(:cold_electron_temperature, [5.0, 3.0]), Equation(:f, TimeDerivative(), fp)]
        system = EquationSystem(layout, equations; initial_state=u)
        @test Runes2P.has_plasma_columns(fp)
        # BDF2 warns of the first order of a term with lagged ions and of its energy loss
        @test Runes2P._first_order_term(fp)
        @test Runes2P._first_order_term(KineticEnergyLossTerm(fp; part=:sinks))
        J = jacobian_prototype(system)
        rhs!(zeros(length(u)), u, system, 0.0)
        jacobian!(J, system, u, 0.0)
        f_rows = [i for ir in 1:2 for i in unknown_range(layout, 4, ir)]
        ion_columns = [i for ir in 1:2 for i in unknown_range(layout, 1, ir)]
        # no entries of the charge states in the pattern of f
        @test nnz(J[f_rows, ion_columns]) == 0
        # the ions of the start of the step: a change of the ions of the iterate leaves the
        # term as it is, a change of the temperature does not
        Runes2P.prepare_step!(fp, u, layout, 4, 0.0)
        term_rhs(x) = Runes2P.rhs!(zeros(length(x)), fp, x, layout, 4, 0.0)
        before = term_rhs(u)
        changed = copy(u)
        changed[unknown_range(layout, 1, 1)[3]] *= 2
        @test term_rhs(changed) == before
        warmer = copy(u)
        radial_values(warmer, layout, 3)[1] *= 1.1
        @test term_rhs(warmer) != before
        # at the next step the ions of its start
        Runes2P.prepare_step!(fp, changed, layout, 4, 0.0)
        @test term_rhs(changed) != before
        @test Runes2P.describe(fp, Runes2P.DescriptionContext())["lagged"] == ["ion_densities"]
        # the energy loss of the term takes the same ions of the start of the step: no change
        # with the ions of the iterate, and zero columns of the charge states
        loss = KineticEnergyLossTerm(fp; part=:sinks)
        Runes2P.prepare_step!(fp, u, layout, 4, 0.0)
        loss_rhs(x) = Runes2P.rhs!(zeros(length(x)), loss, x, layout, 3, 0.0)
        @test loss_rhs(changed) == loss_rhs(u)
        @test loss_rhs(warmer) != loss_rhs(u)
        loss_entries = Runes2P.sparsity!(Tuple{Int,Int}[], loss, layout, 3)
        n = length(u)
        JL = sparse(first.(loss_entries), last.(loss_entries), zeros(length(loss_entries)), n, n)
        Runes2P.add_jacobian!(JL, loss, u, layout, 3, 0.0)
        @test all(iszero, JL[:, ion_columns])
        @test any(!iszero, JL[:, [first(unknown_range(layout, 3, ir)) for ir in 1:2]])
    end

    @testset "energy loss columns of the charge states" begin
        layout, fp, u = _kinetic_ion_case()
        loss = KineticEnergyLossTerm(fp; part=:free_electrons)
        row = 3
        n = length(u)
        Runes2P.validate(loss, layout, row)
        entries = Runes2P.sparsity!(Tuple{Int,Int}[], loss, layout, row)
        J = sparse(first.(entries), last.(entries), zeros(length(entries)), n, n)
        Runes2P.add_jacobian!(J, loss, u, layout, row, 0.0)
        term_rhs(x) = Runes2P.rhs!(zeros(n), loss, x, layout, row, 0.0)
        size_of_rhs = maximum(abs, term_rhs(u))
        for ir in 1:2, column in unknown_range(layout, 1, ir)
            h = 1e-6 * 1e19
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            difference = (term_rhs(up) - term_rhs(down)) / 2h
            @test maximum(abs, Vector(J[:, column]) - difference) <= 1e-5 * maximum(abs, difference) + 1e-12 * size_of_rhs / h
        end
    end

    @testset "the target of the e-e collisions of two grids (issue #154)" begin
        radial = RadialGrid(uniform_r_faces(2, 0.5))
        hot = cylindrical_grid(MomentumGrid(uniform_p_faces(20, 0.0, 0.05), uniform_xi_faces(3)), radial)
        runaway = cylindrical_grid(MomentumGrid(uniform_p_faces(10, 0.05, 0.3), uniform_xi_faces(3)), radial)
        ions = [IonSpecies("D", 1, 1, [5e19, 4e19]), IonSpecies("Ar", 18, 1, [1e18, 2e18])]
        prescribed = PlasmaState(; cold_electron_temperature=[5.0, 3.0], ions, electric_field=zeros(2),
                                 magnetic_field=fill(5.0, 2))
        layout = UnknownLayout(radial, [ion_densities_unknown(KINETIC_COMPOSITION),
                                        RadialUnknown(:cold_electron_density; scale=1e19),
                                        RadialUnknown(:cold_electron_temperature; scale=10.0),
                                        RadialUnknown(:hot_electron_density; scale=1e19),
                                        RadialUnknown(:runaway_density; scale=1e15),
                                        KineticUnknown(:f_hot, hot; scale=1e25), KineticUnknown(:f_re, runaway; scale=1e15)])
        source = PlasmaFromUnknowns(prescribed; ions=:ion_densities)
        fp = FokkerPlanckTerm(:f_hot, AdvectionDiffusion(hot; p_upper_boundary=:f_zero); plasma=source,
                              settings=CollisionSettings(; mode=:full))
        minus(name) = LinearTerm(name; coefficient=-1)
        equations = [Equation(:ion_densities, TimeDerivative()),
                     population_equations(KINETIC_COMPOSITION; hot=:hot_electron_density, runaway=:runaway_density)...,
                     _hold(:cold_electron_temperature, [5.0, 3.0]),
                     Equation(:hot_electron_density, Algebraic(), DensityMomentTerm(:f_hot), minus(:hot_electron_density)),
                     Equation(:runaway_density, Algebraic(), DensityMomentTerm(:f_re), minus(:runaway_density)),
                     Equation(:f_hot, TimeDerivative(), fp), Equation(:f_re, TimeDerivative())]
        u = zeros(length(layout))
        set_ion_densities!(u, layout, ions)
        radial_values(u, layout, 3) .= [5.0, 3.0]
        for ir in 1:2
            # a Maxwellian of θ = 1e-5 (5 eV) with a density near 1e19
            kinetic_view(u, layout, 6, ir) .= [1e19 / (2π * 1e-5)^1.5 * exp(-p^2 / 2e-5)
                                               for p in momentum_grid(hot, ir).p_centers, _ in 1:3]
            kinetic_view(u, layout, 7, ir) .= 1e15
        end
        system = EquationSystem(layout, equations; initial_state=u)
        solution = integrate(system, u, [0.0, 1e-9])
        state = solution.states[:, end]
        # the densities of the grids from their moments, the free electrons from the ions
        moment(name, row) = radial_values(Runes2P.rhs!(zeros(length(state)), DensityMomentTerm(name), state, layout, row, 0.0),
                                          layout, row)
        n_hot, n_re = moment(:f_hot, 4), moment(:f_re, 5)
        n_free = [free_electron_density(ion_densities(state, layout, KINETIC_COMPOSITION), ir) for ir in 1:2]
        @test all(n_hot .> 0.1 .* n_free) && all(n_re .> 0)
        target = plasma_values(fp.plasma, state, layout, 1e-9).cold_electron_density
        @test target ≈ n_free .- n_hot .- n_re rtol = 1e-9
    end

    @testset "Newton iterations of an ionization step, implicit and lagged" begin
        if !isdir(adas_cache_directory())
            @info "no ADF11 cache: the ionization step is skipped"
        else
            counts, seconds = Dict{Tuple,Int}(), Dict{Tuple,Float64}()
            for lagged in ((), (:ion_densities,))
                layout, fp, u = _kinetic_ion_case(; lagged, temperatures=[20.0, 10.0])
                deposit_neutrals!(u, layout, KINETIC_COMPOSITION, "Ar" => 2e19)
                radial_values(u, layout, 2) .= [free_electron_density(ion_densities(u, layout, KINETIC_COMPOSITION), ir)
                                                for ir in 1:2]
                equations = [Equation(:ion_densities, TimeDerivative(), IonRateTerm(KINETIC_COMPOSITION)),
                             population_equations(KINETIC_COMPOSITION)...,
                             _hold(:cold_electron_temperature, [20.0, 10.0]), Equation(:f, TimeDerivative(), fp)]
                system = EquationSystem(layout, equations; time_scale=1e-7, initial_state=u)
                # the cost of the columns: the assembly of the stored Jacobian, the second
                # call after the compilation, both in this test run
                J = jacobian_prototype(system)
                rhs!(zeros(length(u)), u, system, 0.0)
                jacobian!(J, system, u, 0.0)
                seconds[lagged] = @elapsed jacobian!(J, system, u, 0.0)
                solution = integrate(system, u, [0.0, 2e-7]; stepper=FixedSteps())
                @test run_status(solution) === :finished
                counts[lagged] = sum(solution.statistics.newton_iterations)
            end
            @info "ionization step: Newton iterations implicit $(counts[()]), lagged ions $(counts[(:ion_densities,)]); " *
                  "Jacobian $(round(1e3 * seconds[()]; digits=2)) ms against $(round(1e3 * seconds[(:ion_densities,)]; digits=2)) ms"
            # two consistent systems, the lagged one less nonlinear: both converge quickly
            @test counts[()] <= 6 && counts[(:ion_densities,)] <= 6
            # the step on the Newton-Krylov solver with its default preconditioner, the solver
            # of the disruption case, with the block of the 21 charge states in its fluid part
            layout, fp, u = _kinetic_ion_case(; temperatures=[20.0, 10.0])
            deposit_neutrals!(u, layout, KINETIC_COMPOSITION, "Ar" => 2e19)
            radial_values(u, layout, 2) .= [free_electron_density(ion_densities(u, layout, KINETIC_COMPOSITION), ir)
                                            for ir in 1:2]
            equations = [Equation(:ion_densities, TimeDerivative(), IonRateTerm(KINETIC_COMPOSITION)),
                         population_equations(KINETIC_COMPOSITION)...,
                         _hold(:cold_electron_temperature, [20.0, 10.0]), Equation(:f, TimeDerivative(), fp)]
            system = EquationSystem(layout, equations; time_scale=1e-7, initial_state=u)
            krylov = integrate(system, u, [0.0, 2e-7]; stepper=FixedSteps(), solver=NewtonSolver(linear_solver=GMRES()))
            @test run_status(krylov) === :finished
            @test sum(krylov.statistics.krylov_iterations) > 0
        end
    end
end
