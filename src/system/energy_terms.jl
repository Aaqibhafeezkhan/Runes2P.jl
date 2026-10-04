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

"""
    EnergyMomentTerm(kinetic, grid; coefficient=1, region=nothing)

`coefficient` times the kinetic energy density `W(r) = (1/V') Σ 𝒱' Δp Δξ mₑc²(γ − 1) f`
[J/m³ for f in m⁻³] of the kinetic unknown `kinetic` on `grid`, added to a radial row
unknown with one component, with `γ` at the momentum cell centres (DREAM's
`KineticEnergyFromDistributionFunction`). With `region`, a [`MomentumRegion`](@ref), the
moment covers that part of the grid, as [`DensityMomentTerm`](@ref): the energy of the hot
electrons above the threshold, whose limit may follow the temperature unknown. Linear in
the distribution; the column of the temperature is that of the envelope.
"""
struct EnergyMomentTerm{E} <: EquationTerm
    kinetic::Symbol
    coefficient::Float64
    envelopes::E    # nothing, or the envelopes of the region per radius
end
EnergyMomentTerm(kinetic::Symbol, grid::KineticGrid; coefficient::Real=1.0,
                 region::Union{Nothing,MomentumRegion}=nothing) =
    EnergyMomentTerm(kinetic, Float64(coefficient), RegionEnvelopes(region, grid))

dependencies(term::EnergyMomentTerm) = [term.kinetic; collect(Symbol, _region_unknowns(term.envelopes))]
is_linear(term::EnergyMomentTerm) = isempty(_region_unknowns(term.envelopes))

function validate(term::EnergyMomentTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic || throw(ArgumentError("EnergyMomentTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("EnergyMomentTerm adds to a radial unknown with one component"))
    _check_envelopes(term.envelopes, kinetic_grid(layout, id))
    return nothing
end

# mₑc²(γ − 1) [J] at the momentum cell centres, in the form without cancellation at small p
_kinetic_energies(m::MomentumGrid) =
    [Constants.ELECTRON_REST_ENERGY * Constants.ELEMENTARY_CHARGE * p^2 / (1 + sqrt(1 + p^2)) for p in m.p_centers]

# the weight of cell (i, j) at radius ir without the envelope
function _energy_weight(grid, ir)
    m = momentum_grid(grid, ir)
    jacobian = phase_space_jacobian(grid, ir)
    energies = _kinetic_energies(m)
    return (i, j) -> energies[i] * jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] / grid.radial_grid.spatial_jacobian[ir]
end

function rhs!(out, term::EnergyMomentTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        weight = _energy_weight(grid, ir)
        envelope = _moment_envelope(term.envelopes, grid, ir, u, layout, t)
        f = kinetic_view(u, layout, id, ir)
        W = zero(eltype(out))
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            W += _envelope_value(envelope, i) * weight(i, j) * f[i, j]
        end
        out[first(unknown_range(layout, row, ir))] += term.coefficient * W
    end
    return out
end

jacobian_assembly(::EnergyMomentTerm) = :direct

function add_jacobian!(J, term::EnergyMomentTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    weights = [_energy_weight(grid, ir) for ir in 1:radial_cell_count(layout)]
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        envelope = _envelope_at(term.envelopes, grid, ir, u, layout, t)
        r = first(unknown_range(layout, row, ir))
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
            _add_entry!(J, r, _kinetic_index(layout, id, ir, i, j),
                        term.coefficient * _envelope_value(envelope, i) * weights[ir](i, j))
        end
    end
    _add_threshold_column!(J, term.envelopes, grid, u, layout, row, id) do ir, i, j
        term.coefficient * weights[ir](i, j)
    end
    return J
end

sparsity!(entries, term::EnergyMomentTerm, layout, row) =
    _moment_sparsity!(entries, term.kinetic, layout, row, term.envelopes)

"""
    total_energy_equation(distribution, grid; region, total=:total_electron_energy,
                          cold=:cold_thermal_energy)

The algebraic equation of the total energy of the electrons in the fully kinetic mode:

```math
W_\\mathrm{tot} = W_\\mathrm{cold} + \\sum_{p > p_0} m_e c^2 (γ − 1) f\\, ΔV,
```

the energy of the cold electrons and that of the electrons of `distribution` on `grid` in
the hot region `region` ([`EnergyMomentTerm`](@ref)), the region of the hot density. The row
of the temperature then holds the time derivative of `total` (`TimeDerivative(total)`) with
the channels that leave or enter the electrons: the work of the field on all of them
(`E j_ohm + E j_hot` in the fully kinetic mode, where `j_ohm = j(f) − j_hot + E(σ − σ_num)`
already leaves out the hot current, so the two products count each electron once), the
radiated power of the ions, the heat transport and the exchange with the ions. The
collisional transfer from the hot electrons to the cold ones, the electrons that cross the
threshold, and the threshold that moves with the temperature are then exact for any time
stepper, since the time derivative acts on the energy of the hot region itself. DREAM's
fully kinetic mode has none of them (`T_cold.cpp:313-321`).

`total` takes its initial value from the row, after the energy of the cold electrons.
"""
total_energy_equation(distribution::Symbol, grid::KineticGrid; region::MomentumRegion,
                      total::Symbol=:total_electron_energy, cold::Symbol=:cold_thermal_energy) =
    Equation(total, Algebraic(), LinearTerm(cold), EnergyMomentTerm(distribution, grid; region),
             LinearTerm(total; coefficient=-1); initial_value=:from_equation)

"""
    RunawayFluidEnergyTransferTerm(plasma; runaway_density=:runaway_density,
                                   temperature=:cold_electron_temperature,
                                   density=:cold_electron_density,
                                   coulomb_logarithm=:energy_dependent)

DREAM's heating of the cold electrons by the friction of fluid runaways when there is no
runaway grid (`CollisionalEnergyTransferREFluidTerm`), in W/m³ for the row of the
temperature:

```math
e c E_c n_\\mathrm{re} = 4π r_0^2 m_e c^3 \\ln Λ_{ee}(p = 20)\\, n_\\mathrm{cold} n_\\mathrm{re},
```

the friction at the speed of light on runaways of momentum `20 mₑc`. The Coulomb logarithm
of electrons takes the temperature unknown and the free-electron density of the prescribed
ions in `plasma` (a [`PlasmaState`](@ref) or a function of time); `coulomb_logarithm` is the
model of [`coulomb_logarithm`](@ref). The DREAM-compatible mode of the energy
equation.
"""
struct RunawayFluidEnergyTransferTerm{S<:PrescribedPlasma} <: EquationTerm
    plasma::S
    runaway_density::Symbol
    temperature::Symbol
    density::Symbol
    coulomb_logarithm::Symbol
end
function RunawayFluidEnergyTransferTerm(plasma; runaway_density::Symbol=:runaway_density,
                                        temperature::Symbol=:cold_electron_temperature,
                                        density::Symbol=:cold_electron_density,
                                        coulomb_logarithm::Symbol=:energy_dependent)
    coulomb_logarithm in (:energy_dependent, :thermal, :relativistic) ||
        throw(ArgumentError("coulomb_logarithm must be :energy_dependent, :thermal or :relativistic"))
    source = plasma isa PrescribedPlasma ? plasma : PrescribedPlasma(plasma)
    return RunawayFluidEnergyTransferTerm(source, runaway_density, temperature, density, coulomb_logarithm)
end

dependencies(term::RunawayFluidEnergyTransferTerm) = [term.runaway_density, term.temperature, term.density]

function validate(term::RunawayFluidEnergyTransferTerm, layout, row)
    for id in (unknown_id(layout, term.runaway_density), unknown_id(layout, term.temperature),
               unknown_id(layout, term.density), row)
        layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
            throw(ArgumentError("RunawayFluidEnergyTransferTerm takes and adds to radial unknowns with one " *
                                "component, not $(layout.unknowns[id].name)"))
    end
    return nothing
end

const RUNAWAY_FLUID_ENERGY_PREFACTOR = 4π * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.ELECTRON_MASS *
                                       Constants.SPEED_OF_LIGHT^3

function rhs!(out, term::RunawayFluidEnergyTransferTerm, u, layout, row, t)
    plasma = plasma_values(term.plasma, u, layout, t)
    n_re = radial_values(u, layout, unknown_id(layout, term.runaway_density))
    T = radial_values(u, layout, unknown_id(layout, term.temperature))
    n = radial_values(u, layout, unknown_id(layout, term.density))
    for ir in 1:radial_cell_count(layout)
        lnL = coulomb_logarithm(:electron_electron, term.coulomb_logarithm, 20.0, T[ir], free_electron_density(plasma, ir))
        out[first(unknown_range(layout, row, ir))] += RUNAWAY_FLUID_ENERGY_PREFACTOR * lnL * n[ir] * n_re[ir]
    end
    return out
end

function sparsity!(entries, term::RunawayFluidEnergyTransferTerm, layout, row)
    for ir in 1:radial_cell_count(layout), name in dependencies(term)
        push!(entries, (first(unknown_range(layout, row, ir)), first(unknown_range(layout, unknown_id(layout, name), ir))))
    end
    return entries
end
