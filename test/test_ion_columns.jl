# The terms of the fluid rows that read the ion charge states from the unknowns: the radiated
# power, the conductivity, the energy exchange with the ions and the heating by fluid
# runaways. Each gives the residual of the same ions prescribed, its direct
# Jacobian agrees with the forward-mode Jacobian of its residual and with a central
# difference, and the binding energy of the radiated power is the ionization energy times
# the flows between the charge states of the rate equations. Needs the ADF11 cache.
using ForwardDiff
using SparseArrays: sparse, nnz

const COLUMN_COMPOSITION = IonComposition(["D" => 1, "Ar" => 18])

function _column_case(; nr = 2)
    radial = RadialGrid(uniform_r_faces(nr, 0.5))
    layout = UnknownLayout(radial,
        [ion_densities_unknown(COLUMN_COMPOSITION),
            RadialUnknown(:cold_electron_density; scale = 1e19),
            RadialUnknown(:cold_electron_temperature; scale = 10.0),
            RadialUnknown(:electric_field), RadialUnknown(:ohmic_current_density; scale = 1e6),
            RadialUnknown(:ion_thermal_energy; components = 2, scale = 1e3),
            RadialUnknown(:runaway_density; scale = 1e15)])
    argon = zeros(19, nr)
    argon[1, :] .= 1e17
    argon[2, :] .= [3e17, 1e17]
    argon[4, :] .= [2e17, 5e16]
    ions = [IonSpecies("D", 1, [2e18 2e18; 5e19 4e19]), IonSpecies("Ar", 18, argon)]
    prescribed = PlasmaState(;
        cold_electron_temperature = [20.0, 8.0], ions, electric_field = zeros(nr),
        magnetic_field = zeros(nr))
    u = zeros(length(layout))
    set_ion_densities!(u, layout, ions)
    radial_values(u, layout, 2) .= [free_electron_density(ions, ir) for ir in 1:nr]
    radial_values(u, layout, 3) .= [20.0, 8.0]
    radial_values(u, layout, 4) .= [1.5, 2.0]
    radial_values(u, layout, 5) .= 1e6
    for ir in 1:nr
        radial_view(u, layout, 6, ir) .= [2e3, 5e2]
    end
    radial_values(u, layout, 7) .= 1e16
    return layout, prescribed, u
end

# the column of the unknown entry j by a central difference, with a large step on the ion
# densities, where the terms bend only through the logarithms
function _central_column(rhs, u, j, h)
    up, down = copy(u), copy(u)
    up[j] += h
    down[j] -= h
    return (rhs(up) - rhs(down)) / 2h
end

@testset "ion columns" begin
    if !isdir(adas_cache_directory())
        @info "no ADF11 cache at $(adas_cache_directory()): the ion column tests are skipped"
        @test_skip isdir(adas_cache_directory())
    else
        layout, prescribed, u = _column_case()
        n = length(layout)
        row(name) = unknown_id(layout, name)
        from_unknowns = PlasmaFromUnknowns(prescribed; ions = :ion_densities)
        cases = [
            ("radiated power :recombination",
                RadiatedPowerTerm(prescribed; ions = :ion_densities),
                RadiatedPowerTerm(prescribed), row(:cold_electron_temperature)),
            ("radiated power :dream",
                RadiatedPowerTerm(prescribed; radiation = :dream, ions = :ion_densities),
                RadiatedPowerTerm(prescribed; radiation = :dream), row(:cold_electron_temperature)),
            ("conductivity",
                ConductivityTerm(:electric_field, ConductivityModel(from_unknowns)),
                ConductivityTerm(:electric_field,
                    ConductivityModel(PlasmaFromUnknowns(prescribed; density = nothing))),
                row(:ohmic_current_density)),
            ("conductivity, DREAM's fit",
                ConductivityTerm(:electric_field, ConductivityModel(from_unknowns; kinetic = :dream_fit)),
                ConductivityTerm(:electric_field,
                    ConductivityModel(PlasmaFromUnknowns(prescribed; density = nothing);
                        kinetic = :dream_fit)),
                row(:ohmic_current_density)),
            ("exchange, ions",
                IonEnergyExchangeTerm(prescribed; side = :ions, ions = :ion_densities),
                IonEnergyExchangeTerm(prescribed; side = :ions), row(:ion_thermal_energy)),
            ("exchange, electrons",
                IonEnergyExchangeTerm(prescribed; side = :electrons, ions = :ion_densities),
                IonEnergyExchangeTerm(prescribed; side = :electrons), row(:cold_electron_temperature)),
            ("runaway heating",
                RunawayFluidEnergyTransferTerm(prescribed; ions = :ion_densities),
                RunawayFluidEnergyTransferTerm(prescribed), row(:cold_electron_temperature))]
        for (name, term, reference, r) in cases
            @testset "$name" begin
                Runes2P.validate(term, layout, r, 0.0)
                @test Runes2P.jacobian_assembly(term) === :direct
                @test :ion_densities in Runes2P.dependencies(term)
                rhs(x) = Runes2P.rhs!(zeros(eltype(x), n), term, x, layout, r, 0.0)
                # the same residual as the same ions prescribed
                @test rhs(u) ≈ Runes2P.rhs!(zeros(n), reference, u, layout, r, 0.0) rtol = 1e-13
                # the direct block against the forward-mode Jacobian of the residual, on the
                # pattern of the term
                entries = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, r)
                pattern = sparse(
                    first.(entries), last.(entries), zeros(length(entries)), n, n)
                J = copy(pattern)
                add_local_jacobian!(J, term, u, layout, r, 0.0)
                reference_jacobian = ForwardDiff.jacobian(rhs, u)
                @test Matrix(J)≈reference_jacobian rtol=1e-12 atol=1e-12*maximum(abs, reference_jacobian)
                # every nonzero of the forward-mode Jacobian is in the pattern
                listed = Set(entries)
                @test all(iszero(reference_jacobian[i, j]) || (i, j) in listed
                for i in 1:n, j in 1:n)
                # the columns of a radius against a central difference
                size_of_rhs = maximum(abs, rhs(u))
                for j in unknown_range(layout, 1, 2)
                    column = _central_column(rhs, u, j, 1e13)
                    @test maximum(abs, Matrix(J)[:, j] - column) <=
                          1e-5 * maximum(abs, column) + 1e-13 * size_of_rhs / 1e13
                end
                # the derivative of a charge state that is empty: the column holds the
                # derivative at zero density
                empty = unknown_range(layout, 1, 1)[3 + 10]
                @test u[empty] == 0
                @test Matrix(J)[:, empty] ≈ reference_jacobian[:, empty] rtol = 1e-12
            end
        end

        @testset "the local Jacobian does not grow with the kinetic unknowns" begin
            # the state of these cases with a kinetic unknown f of 200 × 10 cells per radius
            # appended after the radial unknowns: the block of each term is the same, and the
            # evaluation reads a state of the radial dependencies alone
            grid = cylindrical_grid(
                MomentumGrid(uniform_p_faces(201, 0.0, 1.0), uniform_xi_faces(10)),
                layout.radial_grid)
            larger = UnknownLayout(layout.radial_grid,
                [layout.unknowns; KineticUnknown(:f, grid; scale = 1e19)])
            positions = zeros(Int, n)
            for id in eachindex(layout.unknowns), ir in 1:radial_cell_count(layout)

                positions[unknown_range(layout, id, ir)] .= unknown_range(larger, id, ir)
            end
            v = zeros(length(larger))
            v[positions] .= u
            v[unknown_range(larger, length(larger.unknowns), 1)] .= 1e17
            for (name, term, _, r) in cases
                small = Runes2P.sparsity!(Tuple{Int, Int}[], term, layout, r)
                J = sparse(first.(small), last.(small), zeros(length(small)), n, n)
                add_local_jacobian!(J, term, u, layout, r, 0.0)
                large = Runes2P.sparsity!(Tuple{Int, Int}[], term, larger, r)
                K = sparse(first.(large), last.(large), zeros(length(large)),
                    length(larger), length(larger))
                add_local_jacobian!(K, term, v, larger, r, 0.0)
                @test Matrix(K[positions, positions]) == Matrix(J)
                @test nnz(K) == nnz(J)
                bytes_small = @allocated add_local_jacobian!(J, term, u, layout, r, 0.0)
                bytes_large = @allocated add_local_jacobian!(K, term, v, larger, r, 0.0)
                # a dual copy of the larger state would take 2 × 8 × (K + 1) bytes per cell,
                # above 1.5 MB here
                @test bytes_large <= bytes_small + 50_000
            end
            # the local layout holds the radial unknowns alone, without their lower limits
            reduced = local_layout(larger,
                [
                    unknown_id(larger, :ion_densities), unknown_id(larger, :cold_electron_temperature)])
            @test length(reduced) == radial_cell_count(layout) * (layout.sizes[1, 1] + 1)
            @test all(x -> x.lower_limit === nothing, reduced.unknowns)
            @test_throws ArgumentError local_layout(larger, [unknown_id(larger, :f)])
        end

        @testset "binding energy against the flows of the rate equations" begin
            radiation = RadiatedPowerTerm(prescribed; ions = :ion_densities)
            rates = IonRateTerm(COLUMN_COMPOSITION)
            binding = radiated_power(radiation, u, layout, 0.0).binding
            du = Runes2P.rhs!(zeros(n), rates, u, layout, 1, 0.0)
            e = Constants.ELEMENTARY_CHARGE
            for ir in 1:2
                expected = 0.0
                for s in 1:2
                    name, Z = COLUMN_COMPOSITION.names[s],
                    COLUMN_COMPOSITION.nuclear_charges[s]
                    change = du[unknown_range(layout, 1, ir)[species_components(COLUMN_COMPOSITION, s)]]
                    # the flow from state j to j + 1 is minus the change of the states up to j
                    for j in 0:(Z - 1)
                        expected += e * ionization_energy(name, j) * -sum(change[1:(j + 1)])
                    end
                end
                @test binding[ir] ≈ expected rtol = 1e-10
            end
        end

        @testset "checks" begin
            wrong = UnknownLayout(layout.radial_grid,
                [RadialUnknown(:ion_densities; components = 21),
                    layout.unknowns[2:end]...])
            @test_throws ArgumentError Runes2P.validate(
                RadiatedPowerTerm(prescribed; ions = :ion_densities), wrong, 3, 0.0)
            @test Runes2P.describe(RadiatedPowerTerm(prescribed; ions = :ion_densities),
                Runes2P.DescriptionContext())["ions"] ==
                  "ion_densities"
            @test !haskey(Runes2P.describe(RadiatedPowerTerm(prescribed), Runes2P.DescriptionContext()), "ions")
        end
    end
end
