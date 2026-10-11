# The renamed keywords: a
# deprecated keyword gives what its new name gives, and both together are refused. The
# deprecation warning itself is not asserted, since its level depends on how the suite runs.

@testset "renamed keywords" begin
    composition = IonComposition(["D" => 1, "Ar" => 18])
    grid = cylindrical_grid(MomentumGrid(uniform_p_faces(8, 0.0, 0.1), uniform_xi_faces(2)),
                            RadialGrid(uniform_r_faces(2, 0.5)))
    rows(equations) = [(e.unknown, [describe(t) for t in e.terms]) for e in equations]

    @testset "ion helpers: name is ions" begin
        @test ion_densities_unknown(composition; name=:n_i).name === :n_i
        @test_throws ArgumentError ion_densities_unknown(composition; name=:n_i, ions=:other)
        layout = UnknownLayout(grid.radial_grid, [ion_densities_unknown(composition; ions=:n_i)])
        species = [IonSpecies("D", 1, [1e19 2e19; 0.0 1e18]), IonSpecies("Ar", 18, fill(1e16, 19, 2))]
        new, old = zeros(length(layout)), zeros(length(layout))
        set_ion_densities!(new, layout, species; ions=:n_i)
        set_ion_densities!(old, layout, species; name=:n_i)
        @test new == old && any(!iszero, new)
        deposit_neutrals!(new, layout, composition, "Ar" => 1e17; ions=:n_i)
        deposit_neutrals!(old, layout, composition, "Ar" => 1e17; name=:n_i)
        @test new == old
        @test [s.charge_state_densities for s in ion_densities(old, layout, composition; name=:n_i)] ==
              [s.charge_state_densities for s in ion_densities(new, layout, composition; ions=:n_i)]
    end

    @testset "population_equations: cold, hot and runaway are density, hot_density and runaway_density" begin
        new = population_equations(composition; density=:n_c, hot_density=:n_h, runaway_density=:n_r)
        old = population_equations(composition; cold=:n_c, hot=:n_h, runaway=:n_r)
        @test rows(new) == rows(old) && only(new).unknown === :n_c
        @test_throws ArgumentError population_equations(composition; cold=:n_c, density=:n_d)
    end

    @testset "total_energy_equation: cold is energy" begin
        region = MomentumRegion(ThermalMomentum(:cold_electron_temperature; factor=3))
        new = total_energy_equation(:f, grid; region, energy=:w_c)
        old = total_energy_equation(:f, grid; region, cold=:w_c)
        @test rows([new]) == rows([old]) && :w_c in dependencies(first(new.terms))
    end

    @testset "FokkerPlanckTerm: one keyword for the lag" begin
        base = PlasmaState(; cold_electron_temperature=[5.0, 3.0], ions=[IonSpecies("D", 1, 1, [1e19, 2e19])],
                           electric_field=zeros(2), magnetic_field=zeros(2))
        plasma = PlasmaFromUnknowns(base; ions=:ion_densities)
        term(; kwargs...) = FokkerPlanckTerm(:f, AdvectionDiffusion(grid; scheme=:exponential_fitting); plasma,
                                             kwargs...)
        context = Runes2P.DescriptionContext()
        @test describe(term(coefficients=:lagged), context) == describe(term(lagged=:all), context)
        @test describe(term(lagged=:all), context)["lagged"] == "all"
        @test describe(term(coefficients=:implicit), context) == describe(term(), context)
        @test term().lagged == (:ion_densities,) && !Runes2P._is_lagged(term())
        @test Runes2P._is_lagged(term(lagged=:all)) && term(lagged=:all).lagged == ()
        @test !haskey(describe(term(lagged=()), context), "lagged")
        @test_throws ArgumentError term(coefficients=:lagged, lagged=())
        @test_throws ArgumentError term(coefficients=:other)
    end

    @testset "plasma_columns: ions is with_ions" begin
        base = PlasmaState(; cold_electron_temperature=[5.0, 3.0], ions=[IonSpecies("D", 1, 1, [1e19, 2e19])],
                           electric_field=zeros(2), magnetic_field=zeros(2))
        source = PlasmaFromUnknowns(base; ions=:ion_densities)
        layout = UnknownLayout(grid.radial_grid, [ion_densities_unknown(IonComposition(["D" => 1])),
                                                  RadialUnknown(:cold_electron_density),
                                                  RadialUnknown(:cold_electron_temperature)])
        for flag in (true, false)
            @test plasma_columns(source, layout, 2; ions=flag) == plasma_columns(source, layout, 2; with_ions=flag)
        end
        @test length(plasma_columns(source, layout, 2)) == 4
    end
end
