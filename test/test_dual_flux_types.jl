# The fluxes of the Fokker-Planck operator with dual coefficients: the plasma
# columns apply an operator whose coefficients are dual numbers, one partial
# per plasma unknown, to a Float64 f. With the element type of f alone the fluxes returned
# Union{Float64, Dual}, and apply! returned wrong partials under exponential fitting with 23
# partials (T, n and the 21 charge states of D and Ar): the charge-state columns of the
# Fokker-Planck term and of its energy loss were off by factors of 1e4 to 1e8, and Newton
# converged linearly. The wrong partials appeared at the default optimization level and not
# at -O0, so the comparisons of the partials catch them only in a run at the default level,
# the level of the full suite; the inferred types of the fluxes are checked at every level.

using ForwardDiff

const DUAL_FLUX_COMPOSITION = IonComposition(["D" => 1, "Ar" => 18])

# two radii of D and Ar with neutral and ionized states, T of 30 and 20 eV, f a shifted
# Maxwellian, the Fokker-Planck term with the implicit charge states
function _dual_flux_case(scheme)
    grid = cylindrical_grid(MomentumGrid(uniform_p_faces(24, 0.0, 0.05), uniform_xi_faces(3)),
                            RadialGrid(uniform_r_faces(2, 0.5)))
    argon = zeros(19, 2)
    argon[1, :] .= [2e17, 4e17]
    argon[2, :] .= [5e17, 3e17]
    argon[3, :] .= [1e17, 2e16]
    ions = [IonSpecies("D", 1, [1e18 2e18; 5e19 4e19]), IonSpecies("Ar", 18, argon)]
    temperatures = [30.0, 20.0]
    prescribed = PlasmaState(; cold_electron_temperature=temperatures, ions, electric_field=zeros(2),
                             magnetic_field=fill(5.0, 2))
    layout = UnknownLayout(grid.radial_grid, [ion_densities_unknown(DUAL_FLUX_COMPOSITION),
                                              RadialUnknown(:cold_electron_density; scale=1e19),
                                              RadialUnknown(:cold_electron_temperature; scale=10.0),
                                              KineticUnknown(:f, grid; scale=1e19)])
    source = PlasmaFromUnknowns(prescribed; ions=:ion_densities)
    fp = FokkerPlanckTerm(:f, AdvectionDiffusion(grid; scheme, p_upper_boundary=:f_zero); plasma=source,
                          settings=CollisionSettings(; mode=:full, screening=:partial), lagged=())
    u = zeros(length(layout))
    set_ion_densities!(u, layout, ions)
    radial_values(u, layout, 2) .= [free_electron_density(ions, ir) for ir in 1:2]
    radial_values(u, layout, 3) .= temperatures
    theta = temperatures ./ Constants.ELECTRON_REST_ENERGY
    for ir in 1:2
        m = momentum_grid(grid, ir)
        kinetic_view(u, layout, 4, ir) .= [1e19 * exp(-p^2 / (2theta[ir])) * (1 + 0.1xi) for p in m.p_centers, xi in m.xi_centers]
    end
    return layout, fp, u
end

@testset "fluxes with dual coefficients" begin
    # the limiters TCDF (the scheme of the DREAM disruption case) and van Albada among the schemes,
    # with the limiter frozen at a state as well
    @testset "one element type per flux ($scheme)" for scheme in (:exponential_fitting, :upwind, :central, :quick,
                                                                 :tcdf, :van_albada)
        layout, fp, u = _dual_flux_case(scheme)
        D = ForwardDiff.Dual{Runes2P.PlasmaColumnTag,Float64,23}
        op = Runes2P.similar_coefficients(fp.operator, D)
        f = kinetic_array(u, layout, 4)
        mode = Runes2P._radial_flux_mode(op)
        for k in (1, 2, 25), j in 1:3
            @test (@inferred Runes2P._p_flux(op, f, k, j, 1)) isa D
            @test (@inferred Runes2P._p_flux(op, f, k, j, 1, f, nothing)) isa D
        end
        for k in 1:4
            @test (@inferred Runes2P._xi_flux(op, f, 2, k, 1)) isa D
        end
        for k in 1:3
            @test (@inferred Runes2P._r_flux(op, f, 2, 2, 1, k, nothing, mode)) isa D
        end
        @test (@inferred Runes2P._r_edge_flux(op, f, 2, 2, 3)) isa D
        @test (@inferred Runes2P.momentum_flux(op, f, 2, 1)) isa D
        # a dual electric field on Float64 coefficients: the response of the conductivity and
        # the boundary flux of the population closure (no caller combines it with dual
        # coefficients, whose plasma columns take the field as numbers)
        F = ForwardDiff.Dual{Nothing,Float64,1}
        field = [F(0.1, ForwardDiff.Partials((1.0,))), F(0.2, ForwardDiff.Partials((0.0,)))]
        for k in (1, 2, 25), j in 1:3
            @test (@inferred Runes2P._p_flux(fp.operator, f, k, j, 1, nothing, field)) isa F
        end
        for k in 1:4
            @test (@inferred Runes2P._xi_flux(fp.operator, f, 2, k, 1, nothing, field)) isa F
        end
        @test (@inferred Runes2P.momentum_flux(fp.operator, f, 2, 1; electric_field=field)) isa F
        perturbed = Runes2P.PerturbedField([0.1, 0.2], 1)
        @test (@inferred Runes2P._p_flux(fp.operator, f, 2, 1, 1, nothing, perturbed)) isa eltype(perturbed)
    end

    @testset "apply! on a Float64 f and on a dual f ($scheme)" for scheme in (:exponential_fitting, :tcdf, :van_albada)
        layout, fp, u = _dual_flux_case(scheme)
        plasma = Runes2P.plasma_values(fp.plasma, u, layout, 0.0)
        dual = Runes2P.dual_plasma(plasma, true, true)
        D = eltype(dual.cold_electron_temperature)
        op = Runes2P.similar_coefficients(fp.operator, D)
        set_fokker_planck_coefficients!(op, dual, fp.settings; synchrotron=fp.synchrotron)
        f = kinetic_array(u, layout, 4)
        from_float = Runes2P.apply!(zeros(D, size(f)), op, f)
        from_dual = Runes2P.apply!(zeros(D, size(f)), op, D.(f))
        @test ForwardDiff.value.(from_float) == ForwardDiff.value.(from_dual)
        @test ForwardDiff.partials.(from_float) == ForwardDiff.partials.(from_dual)
    end

    # the coefficient helpers return one number type with a dual plasma: the friction on bound
    # electrons under both forms of bound_friction
    @testset "one number type of the bound friction" begin
        layout, fp, u = _dual_flux_case(:exponential_fitting)
        plasma = Runes2P.plasma_values(fp.plasma, u, layout, 0.0)
        dual = Runes2P.dual_plasma(plasma, true, true)
        D = eltype(dual.cold_electron_temperature)
        for bound_friction in (:bethe, :ionization_split)
            settings = CollisionSettings(; mode=:full, screening=:partial, bound_friction)
            model = Runes2P.CollisionModel(dual, settings)
            argon = findfirst(s -> s.nuclear_charge == 18 && s.charge == 1, model.charge_states)
            @test (@inferred Runes2P._bound_slowing_down(model, argon, model.charge_states[argon], 1, 0.3, 1.0)) isa D
            @test (@inferred slowing_down_frequency(model, 1, 0.3)) isa D
        end
    end

    # the columns of the charge states in the rows of f and of T against central differences
    # under exponential fitting; the step is 1e-2 of the density of the state, and the
    # tolerance adds to a part of the column the round-off of the difference quotient of each
    # row, 1e-14 |rhs|/h
    @testset "charge-state columns under exponential fitting ($part)" for part in (:fokker_planck, :sinks)
        layout, fp, u = _dual_flux_case(:exponential_fitting)
        hold(name, value) = Equation(name, Algebraic(), PrescribedValueTerm(value), LinearTerm(name; coefficient=-1))
        T_equation = part === :sinks ?
                     Equation(:cold_electron_temperature, TimeDerivative(), KineticEnergyLossTerm(fp; part=:sinks)) :
                     hold(:cold_electron_temperature, [30.0, 20.0])
        system = EquationSystem(layout, [Equation(:ion_densities, TimeDerivative()),
                                         population_equations(DUAL_FLUX_COMPOSITION)..., T_equation,
                                         Equation(:f, TimeDerivative(), fp)]; initial_state=u)
        J = jacobian_prototype(system)
        rhs!(zeros(length(u)), u, system, 0.0)
        jacobian!(J, system, u, 0.0)
        row = part === :sinks ? 3 : 4
        # the rows of f without the first momentum cell, where the fluxes in ξ cancel to many
        # digits and the difference quotients carry a round-off of a few 1e-3 of the column at
        # these steps (before the fluxes were type-stable the error was 1e7 there and in the
        # second cell)
        rows = part === :sinks ? [first(unknown_range(layout, row, ir)) for ir in 1:2] :
               [Runes2P._kinetic_index(layout, row, ir, i, j) for ir in 1:2 for j in 1:3 for i in 2:24]
        evaluate(x) = (out = zeros(length(x)); rhs!(out, x, system, 0.0); out[rows])
        size_of_rows = abs.(evaluate(u))
        # the energy loss is the difference of two applications of the collisions (all of
        # them and the electron-electron part), whose difference quotients scatter by a few
        # 1e-2 of the loss between steps of 1e-2 and 1e-4 of the density; the rows of f near the
        # first cells carry a round-off of about 1e-3 of the column at these steps
        relative = part === :sinks ? 1e-2 : 3e-3
        for ir in 1:2, column in unknown_range(layout, 1, ir)
            u[column] > 0 || continue
            h = 1e-2 * u[column]
            up, down = copy(u), copy(u)
            up[column] += h
            down[column] -= h
            difference = (evaluate(up) - evaluate(down)) / 2h
            error = abs.(Vector(J[rows, column]) - difference)
            @test all(error .<= relative * maximum(abs, difference) .+ 1e-14 .* size_of_rows ./ h)
        end
    end
end
