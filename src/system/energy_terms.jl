# The energy of the cold electrons: the unknown W_cold = 3/2 e n_cold T_cold
# of DREAM, whose time derivative is held by the row of T_cold, and the product term of its
# algebraic row. The heating and loss terms of the energy row are separate terms.

"""
    ProductTerm(first, second; coefficient=1)

`coefficient` times the product of the radial unknowns `first` and `second`, each with one
component, at every radius, added to the row of a radial unknown with one component
(DREAM's `DiagonalQuadraticTerm`). Not linear in the state.
"""
struct ProductTerm <: EquationTerm
    first::Symbol
    second::Symbol
    coefficient::Float64
end
ProductTerm(first::Symbol, second::Symbol; coefficient::Real=1.0) = ProductTerm(first, second, Float64(coefficient))

dependencies(term::ProductTerm) = unique([term.first, term.second])

function validate(term::ProductTerm, layout, row)
    for id in (unknown_id(layout, term.first), unknown_id(layout, term.second), row)
        layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
            throw(ArgumentError("ProductTerm takes and adds to radial unknowns with one component, not " *
                                "$(layout.unknowns[id].name)"))
    end
    return nothing
end

function rhs!(out, term::ProductTerm, u, layout, row, t)
    a = radial_values(u, layout, unknown_id(layout, term.first))
    b = radial_values(u, layout, unknown_id(layout, term.second))
    @inbounds for ir in 1:radial_cell_count(layout)
        out[first(unknown_range(layout, row, ir))] += term.coefficient * a[ir] * b[ir]
    end
    return out
end

function sparsity!(entries, term::ProductTerm, layout, row)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for name in dependencies(term)
            push!(entries, (r, first(unknown_range(layout, unknown_id(layout, name), ir))))
        end
    end
    return entries
end

"""
    THERMAL_ENERGY_FACTOR

`3/2 e` [J/eV]: the thermal energy density `W = 3/2 e n T` [J/m³] of electrons with the
density `n` [m⁻³] and the temperature `T` [eV] (DREAM's `ElectronHeatTerm`).
"""
const THERMAL_ENERGY_FACTOR = 1.5 * Constants.ELEMENTARY_CHARGE

"""
    thermal_energy_equation(; energy=:cold_thermal_energy, temperature=:cold_electron_temperature,
                            density=:cold_electron_density)

The algebraic equation of the energy of the cold electrons, `0 = 3/2 e n T − W` with `W`
in J/m³ and `T` in eV (DREAM's `W_cold`). The row of the temperature holds the time
derivative of the energy, `Equation(temperature, TimeDerivative(energy), ...)`, with
the heating and loss terms in W/m³, as in DREAM, where the energy is the unknown of the time
derivative and the temperature that of its own row.

The energy takes its initial value from the row (`initial_value = :from_equation`):
[`integrate`](@ref) sets it from the temperature and the density of the initial
state, and the value of the energy in that state is not read.
"""
thermal_energy_equation(; energy::Symbol=:cold_thermal_energy, temperature::Symbol=:cold_electron_temperature,
                        density::Symbol=:cold_electron_density) =
    Equation(energy, Algebraic(), ProductTerm(density, temperature; coefficient=THERMAL_ENERGY_FACTOR),
             LinearTerm(energy; coefficient=-1); initial_value=:from_equation)

"""
    set_thermal_energy!(u, layout; energy=:cold_thermal_energy,
                        temperature=:cold_electron_temperature, density=:cold_electron_density)

Set the energy of the cold electrons in the state `u` from its temperature and density,
`W = 3/2 e n T` at every radius. DREAM's order of the initial values: the temperature from
the input, the density from its row, and then the energy (`T_cold.cpp`, `n_cold.cpp`). The
density must already hold its value.
"""
function set_thermal_energy!(u::AbstractVector, layout::UnknownLayout; energy::Symbol=:cold_thermal_energy,
                             temperature::Symbol=:cold_electron_temperature, density::Symbol=:cold_electron_density)
    W = radial_values(u, layout, unknown_id(layout, energy))
    T = radial_values(u, layout, unknown_id(layout, temperature))
    n = radial_values(u, layout, unknown_id(layout, density))
    for ir in eachindex(W)
        W[ir] = THERMAL_ENERGY_FACTOR * n[ir] * T[ir]
    end
    return u
end
