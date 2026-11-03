# FrictionEnergyTransferTerm: the power that the electrons of f give to the cold electrons by
# friction. The power of one fast electron against an independent reference, the relativistic
# friction on free electrons F = e E_c / β² with the critical field of Connor and Hastie,
# E_c = n e³ lnΛ / (4π ε₀² mₑ c²) (J. W. Connor and R. J. Hastie, Nucl. Fusion 15, 415 (1975)),
# so that P = v F = c e E_c / β; its Jacobian against central differences of its residual, with
# the temperature, the cold density and the ion charge states as unknowns. The Coulomb logarithm
# is the package's (`coulomb_logarithm`), an input that the reference shares.
using SparseArrays

@testset "FrictionEnergyTransferTerm" begin
    radial = RadialGrid([0.0, 0.25, 0.5])
    grid = cylindrical_grid(
        MomentumGrid(collect(range(0, 0.8; length = 21)),
            collect(range(-1, 1; length = 5))),
        radial)
    nr = radial_cell_count(radial)
    temperature, density = [12.0, 30.0], [3e19, 8e19]
    argon = IonSpecies("Ar", 18, zeros(19, nr))
    argon.charge_state_densities[2, :] .= [1e19, 2e19]
    deuterium = IonSpecies("D", 1, [zeros(1, nr); fill(5e19, 1, nr)])
    prescribed = PlasmaState(; cold_electron_temperature = temperature,
        cold_electron_density = density, ions = [deuterium, argon],
        electric_field = zeros(nr), magnetic_field = fill(2.5, nr))
    composition = IonComposition(["D" => 1, "Ar" => 18])
    layout = UnknownLayout(radial,
        [RadialUnknown(:cold_electron_temperature; scale = 10.0),
            RadialUnknown(:cold_electron_density; scale = 1e19),
            ion_densities_unknown(composition), RadialUnknown(:heating; scale = 1e6),
            KineticUnknown(:f, grid; scale = 1e17)])
    u = zeros(length(layout))
    unknown_values(u, layout, unknown_id(layout, :cold_electron_temperature)) .= temperature
    unknown_values(u, layout, unknown_id(layout, :cold_electron_density)) .= density
    for ir in 1:nr
        radial_view(u, layout, unknown_id(layout, :ion_densities), ir) .= vcat(
            deuterium.charge_state_densities[:, ir], argon.charge_state_densities[:, ir])
    end
    row = unknown_id(layout, :heating)
    f_id = unknown_id(layout, :f)

    @testset "one fast electron against the friction of Connor and Hastie" begin
        term = FrictionEnergyTransferTerm(:f, grid, prescribed)
        Runes2P.validate(term, layout, row)
        m = momentum_grid(grid, 1)
        for i in (3, 10, 20), ir in 1:nr

            state = copy(u)
            f = kinetic_view(state, layout, f_id, ir)
            f[i, 2] = 1.0
            # the density of the electrons of that cell, from the density moment of the grid
            cell_density = Runes2P.density(grid, kinetic_array(state, layout, f_id))[ir]
            p = m.p_centers[i]
            beta = p / sqrt(1 + p^2)
            logarithm = coulomb_logarithm(:electron_electron, :energy_dependent, p,
                free_electron_density(prescribed, ir), temperature[ir])
            e, c = Constants.ELEMENTARY_CHARGE, Constants.SPEED_OF_LIGHT
            critical_field = density[ir] * e^3 * logarithm /
                             (4π * Constants.VACUUM_PERMITTIVITY^2 *
                              Constants.ELECTRON_MASS *
                              c^2)
            expected = cell_density * c * e * critical_field / beta
            out = Runes2P.rhs!(zeros(length(u)), term, state, layout, row, 0.0)
            # 3.0e-9 apart: the classical electron radius of the package (CODATA 2014) against
            # e²/(4π ε₀ mₑ c²) of its other constants
            @test unknown_values(out, layout, row)[ir] ≈ expected rtol = 1e-8
            # the other radius takes nothing
            @test unknown_values(out, layout, row)[3 - ir] == 0
        end
    end

    @testset "the Jacobian against central differences" begin
        variants = (PlasmaFromUnknowns(prescribed; density = nothing),
            PlasmaFromUnknowns(prescribed), PlasmaFromUnknowns(prescribed; ions = :ion_densities))
        for plasma in variants
            ions = Runes2P.ion_dependency(plasma)
            term = FrictionEnergyTransferTerm(:f, grid, plasma; coefficient = -2.0)
            Runes2P.validate(term, layout, row)
            @test !Runes2P.is_linear(term)
            state = copy(u)
            for ir in 1:nr
                f = kinetic_view(state, layout, f_id, ir)
                f .= [1e17 * exp(-10p) for p in momentum_grid(grid, ir).p_centers, _ in 1:4]
            end
            entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, row)
            J = sparse(first.(entries), last.(entries), 0.0, length(u), length(u))
            Runes2P.add_jacobian!(J, term, state, layout, row, 0.0)
            columns = vcat(
                collect.(Runes2P._unknown_ranges(layout,
                    unknown_id(layout, :cold_electron_temperature)))...,
                collect.(Runes2P._unknown_ranges(layout,
                    unknown_id(layout, :cold_electron_density)))...,
                collect(unknown_range(layout, f_id, 1))[[2, 33]],
                collect(unknown_range(layout, f_id, 2))[[1, 7, 40]])
            if ions !== nothing
                for ir in 1:nr
                    append!(columns,
                        collect(unknown_range(layout,
                            unknown_id(layout, :ion_densities), ir))[[2, 4]])
                end
            end
            rows = [first(unknown_range(layout, row, ir)) for ir in 1:nr]
            for column in columns
                h = 1e-6 * max(abs(state[column]), 1.0)
                plus, minus = copy(state), copy(state)
                plus[column] += h
                minus[column] -= h
                upper = Runes2P.rhs!(zeros(length(u)), term, plus, layout, row, 0.0)
                lower = Runes2P.rhs!(zeros(length(u)), term, minus, layout, row, 0.0)
                difference = (upper .- lower) ./ (2h)
                floor = 1e-12 * maximum(abs, difference)
                for r in rows
                    @test J[r, column]≈difference[r] rtol=1e-6 atol=floor
                end
            end
        end
    end

    @testset "checks" begin
        term = FrictionEnergyTransferTerm(:f, grid, prescribed)
        @test_throws ArgumentError Runes2P.validate(
            FrictionEnergyTransferTerm(:heating, grid, prescribed), layout, row)
        @test_throws ArgumentError Runes2P.validate(term, layout, f_id)
        @test Runes2P.is_linear(term)
        # another grid of the same size
        other = cylindrical_grid(
            MomentumGrid(collect(range(0, 1.0; length = 21)),
                collect(range(-1, 1; length = 5))),
            radial)
        @test_throws ArgumentError Runes2P.validate(
            FrictionEnergyTransferTerm(:f, other, prescribed), layout, row)
        @test FRICTION_TRANSFER_SETTINGS.mode === :superthermal
        @test FRICTION_TRANSFER_SETTINGS.screening === :complete
        @test FRICTION_TRANSFER_SETTINGS.bremsstrahlung === :neglect
        @test FRICTION_TRANSFER_SETTINGS.coulomb_logarithm === :energy_dependent
    end
end
