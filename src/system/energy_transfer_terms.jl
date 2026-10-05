# The energy that the distributions exchange with the total energy of the electrons: the
# collisional transfer from the runaway grid to the cold electrons, the sinks of the hot
# electrons (bound electrons, bremsstrahlung, synchrotron), and the energy that electrons
# carry out of the hot-tail grid through the interface. Included after
# operators/field_particle.jl, whose electron-electron operator it uses.

using SparseArrays: findnz, sparse

"""
    KineticEnergyLossTerm(source::FokkerPlanckTerm; part=:free_electrons, region=nothing,
                          coefficient=part === :free_electrons ? 1 : -1)

`coefficient` times the power [W/m³] that the electrons of the kinetic unknown of `source`
lose through a part of their collisions, for the row of the temperature of the cold
electrons:

```math
L = −\\frac{1}{V'} \\sum_{ij} w_i\\, m_e c^2 (γ_i − 1)\\, [C_a f − C_b f]_{ij}\\, 𝒱'_{ij} Δp_i Δξ_j,
```

with the envelope `w` of `region` (1 without one, [`MomentumRegion`](@ref)) and the
operators of the plasma, collision settings, scheme and boundaries of `source`:

- `part = :free_electrons`: `C_a` the electron-electron part ([`electron_electron_operator`](@ref)),
  `C_b = 0`: the energy that the electrons give to the free cold electrons, the friction
  and, in the fully kinetic mode, the energy diffusion. For a superthermal electron it is
  DREAM's `∫ v F_s f` of `CollisionalEnergyTransferKineticTerm`. With `coefficient = 1` it
  is the channel of the runaway grid into the total energy of the electrons on the
  hot-tail grid ([`total_energy_equation`](@ref)).
- `part = :sinks`: `C_a` the collisions of `source` without the field, `C_b` their
  electron-electron part: the energy that leaves the electrons by the friction on bound
  electrons (excitation and ionization), bremsstrahlung and synchrotron radiation. With
  `coefficient = −1` and the hot region of the total energy as `region` it is the loss of
  that total, so the cold electrons receive exactly the free-electron part of what the hot
  electrons lose. Both operators are applied, since the weights of exponential fitting
  depend on all coefficients of an operator. The friction on bound electrons holds the
  energy of the kinetic ionization ([`KineticIonizationTerm`](@ref)): the ionization
  potentials, which the ions store and release at recombination, and the energy of the
  secondaries, which [`SecondaryHeatingTerm`](@ref) gives back to the cold electrons
  ([`ionization_energy_flows`](@ref)).

The term takes the plasma of `source` in the same treatment: implicit, with the columns of
the temperature and density unknowns, or lagged at the state of the start of the step. A
region whose limit follows the temperature adds the column of its envelope.
"""
struct KineticEnergyLossTerm{F<:FokkerPlanckTerm,E} <: EquationTerm
    kinetic::Symbol
    source::F
    part::Symbol
    coefficient::Float64
    envelopes::E                        # nothing, or the envelopes of the region per radius
    lagged_plasma::Base.RefValue{Any}   # the plasma of the start of the step, for a lagged source
end
function KineticEnergyLossTerm(source::FokkerPlanckTerm; part::Symbol=:free_electrons,
                               region::Union{Nothing,MomentumRegion}=nothing,
                               coefficient::Union{Nothing,Real}=nothing)
    part in (:free_electrons, :sinks) || throw(ArgumentError("part must be :free_electrons or :sinks, not $part"))
    value = coefficient === nothing ? (part === :free_electrons ? 1.0 : -1.0) : Float64(coefficient)
    return KineticEnergyLossTerm(source.unknown, source, part, value, RegionEnvelopes(region, source.operator.grid),
                                 Ref{Any}(nothing))
end

"""
    FreeElectronEnergyTransferTerm(source::FokkerPlanckTerm)

The power that the electrons of the unknown of `source` give to the free cold electrons by
collisions: [`KineticEnergyLossTerm`](@ref) with `part = :free_electrons`, the transfer from
the runaway grid to the cold electrons.
"""
FreeElectronEnergyTransferTerm(source::FokkerPlanckTerm) = KineticEnergyLossTerm(source; part=:free_electrons)

lags_in_ode_form(term::KineticEnergyLossTerm) = lags_in_ode_form(term.source)
_follows_unknowns(term::KineticEnergyLossTerm) = is_state_dependent(term.source.plasma) && !_is_lagged(term.source)
_loss_plasma_unknowns(term::KineticEnergyLossTerm) =
    _follows_unknowns(term) ? collect(Symbol, source_dependencies(term.source.plasma)) : Symbol[]
dependencies(term::KineticEnergyLossTerm) =
    unique([term.kinetic; _loss_plasma_unknowns(term); collect(Symbol, _region_unknowns(term.envelopes))])
is_linear(term::KineticEnergyLossTerm) = length(dependencies(term)) == 1
jacobian_assembly(::KineticEnergyLossTerm) = :direct
_coefficient_sources(term::KineticEnergyLossTerm) = (term.source,)

function validate(term::KineticEnergyLossTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic || throw(ArgumentError("KineticEnergyLossTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("KineticEnergyLossTerm adds to a radial unknown with one component"))
    term.source.settings.mode in (:full, :superthermal) ||
        throw(ArgumentError("the energy loss takes collision mode :full or :superthermal"))
    is_linear(term.source.operator) ||
        throw(ArgumentError("the energy loss takes the collisions of a Fokker-Planck term with a linear scheme: a " *
                            "flux limiter makes it depend on f nonlinearly"))
    term.source.plasma === nothing && term.source.operator.coefficient_source[] === nothing &&
        throw(ArgumentError("the Fokker-Planck term of the energy loss has no plasma"))
    _check_envelopes(term.envelopes, kinetic_grid(layout, id))
    return nothing
end

function prepare_step!(term::KineticEnergyLossTerm, u_old, layout, row, t)
    _is_lagged(term.source) && (term.lagged_plasma[] = plasma_values(term.source.plasma, u_old, layout, t))
    return nothing
end

# the plasma of the collisions: of the state, of the start of the step, or that of the
# operator when the source has none
function _transfer_plasma(term::KineticEnergyLossTerm, u, layout, t)
    plasma = term.source.plasma
    plasma === nothing && return term.source.operator.coefficient_source[].plasma
    _is_lagged(term.source) && term.lagged_plasma[] !== nothing && return term.lagged_plasma[]::PlasmaState
    return _fokker_planck_plasma(term.source, u, layout, t)
end

# the operators C_a and C_b of the part, with the scheme and the boundaries of the source
function _loss_operators(term::KineticEnergyLossTerm, grid, plasma)
    op = term.source.operator
    free = electron_electron_operator(grid, plasma, term.source.settings; scheme=op.p_scheme,
                                      p_lower_boundary=op.p_lower_boundary, p_upper_boundary=op.p_upper_boundary)
    term.part === :free_electrons && return (free, nothing)
    collisions = AdvectionDiffusion(grid; p_scheme=op.p_scheme, xi_scheme=op.xi_scheme,
                                    p_lower_boundary=op.p_lower_boundary, p_upper_boundary=op.p_upper_boundary)
    C = eltype(plasma.cold_electron_temperature)
    C === Float64 || (collisions = similar_coefficients(collisions, C))
    set_fokker_planck_coefficients!(collisions, plasma, term.source.settings; synchrotron=term.source.synchrotron,
                                    electric_field=false)
    return (collisions, free)
end

# C_a f − C_b f in the number type of the operators
function _loss_change(ops, f)
    a, b = ops
    df = zeros(promote_type(eltype(a.advection_p), eltype(f)), size(f))
    apply!(df, a, f)
    if b !== nothing
        db = zeros(eltype(df), size(f))
        apply!(db, b, f)
        df .-= db
    end
    return df
end

# mₑc²(γ − 1) 𝒱' Δp Δξ / V' of every cell of radius ir
function _energy_weights(grid::KineticGrid, ir)
    m = momentum_grid(grid, ir)
    V = phase_space_jacobian(grid, ir)
    energies = _kinetic_energies(m)
    return [energies[i] * V[i, j] * m.p_widths[i] * m.xi_widths[j] / grid.radial_grid.spatial_jacobian[ir]
            for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)]
end

# the weights of the cells of radius ir with the envelope of the region
function _loss_weights(term::KineticEnergyLossTerm, grid, ir, u, layout, t)
    weights = _energy_weights(grid, ir)
    envelope = _envelope_at(term.envelopes, grid, ir, u, layout, t)
    for i in axes(weights, 1)
        weights[i, :] .*= _envelope_value(envelope, i)
    end
    return weights
end

function rhs!(out, term::KineticEnergyLossTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    df = _loss_change(_loss_operators(term, grid, _transfer_plasma(term, u, layout, t)), kinetic_array(u, layout, id))
    for ir in 1:radial_cell_count(layout)
        out[first(unknown_range(layout, row, ir))] -=
            term.coefficient * sum(_loss_weights(term, grid, ir, u, layout, t) .* view(df, :, :, ir))
    end
    return out
end

function add_jacobian!(J, term::KineticEnergyLossTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    plasma = _transfer_plasma(term, u, layout, t)
    ops = _loss_operators(term, grid, plasma)
    np, nxi, nr = size(grid)
    # one product per operator for all radii: the operators couple no radii, so the value
    # of each cell belongs to the row of its own radius
    weights = reduce(vcat, (vec(_loss_weights(term, grid, ir, u, layout, t)) for ir in 1:nr))
    row_values = transpose(operator_matrix(ops[1])) * weights
    ops[2] === nothing || (row_values .-= transpose(operator_matrix(ops[2])) * weights)
    # the cells of the kinetic array (p fastest, then ξ, then r) with a nonzero weight
    for (k, value) in zip(findnz(sparse(row_values))...)
        radius, within = divrem(k - 1, np * nxi)
        pitch, momentum = divrem(within, np)
        _add_entry!(J, first(unknown_range(layout, row, radius + 1)),
                    _kinetic_index(layout, id, radius + 1, momentum + 1, pitch + 1), -term.coefficient * value)
    end
    f = kinetic_array(u, layout, id)
    # the columns of the plasma unknowns: the operators of a plasma with derivative parts
    if _follows_unknowns(term)
        source = term.source.plasma
        with_ions = ion_dependency(source) !== nothing && !_lags_ions(term.source)
        dual = dual_plasma(plasma, source.density !== nothing, with_ions)
        df = _loss_change(_loss_operators(term, grid, dual), f)
        for ir in 1:nr
            partials = ForwardDiff.partials(-term.coefficient * sum(_loss_weights(term, grid, ir, u, layout, t) .*
                                                                     view(df, :, :, ir)))
            for (k, column) in enumerate(plasma_columns(source, layout, ir; ions=with_ions))
                _add_entry!(J, first(unknown_range(layout, row, ir)), column, partials[k])
            end
        end
    end
    # the column of the temperature that moves the limit of the region
    temperature = _region_unknowns(term.envelopes)
    if !isempty(temperature)
        df = _loss_change(ops, f)
        T_id = unknown_id(layout, only(temperature))
        D = ForwardDiff.Dual{PlasmaColumnTag,Float64,1}
        T = radial_values(u, layout, T_id)
        limit = term.envelopes.region.limit
        for ir in 1:nr
            m = momentum_grid(grid, ir)
            p0 = limit.factor * sqrt(2 * D(T[ir], ForwardDiff.Partials((1.0,))) / Constants.ELECTRON_REST_ENERGY)
            envelope = momentum_envelope!(zeros(D, momentum_cell_count(m)), term.envelopes.region, m, p0)
            energies = _energy_weights(grid, ir)
            value = 0.0
            for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)
                value += ForwardDiff.partials(envelope[i], 1) * energies[i, j] * df[i, j, ir]
            end
            _add_entry!(J, first(unknown_range(layout, row, ir)), first(unknown_range(layout, T_id, ir)),
                        -term.coefficient * value)
        end
    end
    return J
end

function sparsity!(entries, term::KineticEnergyLossTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    others = setdiff(dependencies(term), [term.kinetic])
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for s in unknown_range(layout, id, ir)
            push!(entries, (r, s))
        end
        for name in others
            append!(entries, ((r, c) for c in unknown_range(layout, unknown_id(layout, name), ir)))
        end
    end
    return entries
end

"""
    InterfaceEnergyFluxTerm(interface::MomentumInterfaceTerm; coefficient=-1)

`coefficient` times the kinetic energy that electrons carry through the momentum interface
from the lower grid to the upper one, per volume and second [W/m³]: `mₑc²(γ − 1)` of the
last momentum cell of the lower grid times [`interface_flux`](@ref). The interface takes
the electrons from that cell, so this is the energy that the discrete energy moment of the
lower grid loses (the energy at the interface itself differs by about Δp/(2p)). With the default `coefficient = −1` in the row of
the temperature it is the energy that leaves the total energy of the electrons on the
hot-tail grid for the runaway grid ([`total_energy_equation`](@ref)), so the electrons that
cross take their energy with them and do not heat the cold ones.
"""
struct InterfaceEnergyFluxTerm{I<:MomentumInterfaceTerm} <: EquationTerm
    interface::I
    coefficient::Float64
end
InterfaceEnergyFluxTerm(interface::MomentumInterfaceTerm; coefficient::Real=-1.0) =
    InterfaceEnergyFluxTerm(interface, Float64(coefficient))

dependencies(term::InterfaceEnergyFluxTerm) = dependencies(term.interface)
is_linear(term::InterfaceEnergyFluxTerm) = is_linear(term.interface)
_coefficient_sources(term::InterfaceEnergyFluxTerm) = _coefficient_sources(term.interface)

function validate(term::InterfaceEnergyFluxTerm, layout, row)
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("InterfaceEnergyFluxTerm adds to a radial unknown with one component"))
    return validate(term.interface, layout, unknown_id(layout, term.interface.lower))
end

function rhs!(out, term::InterfaceEnergyFluxTerm, u, layout, row, t)
    grid = term.interface.lower_grid
    for ir in 1:radial_cell_count(layout)
        p = momentum_grid(grid, ir).p_centers[end]
        energy = Constants.ELECTRON_REST_ENERGY * Constants.ELEMENTARY_CHARGE * p^2 / (1 + sqrt(1 + p^2))
        out[first(unknown_range(layout, row, ir))] += term.coefficient * energy * interface_flux(term.interface, u, layout, ir, t)
    end
    return out
end

# the row reads the last momentum cells of the lower grid, the first of the upper grid, and
# the field when the interface takes it
function sparsity!(entries, term::InterfaceEnergyFluxTerm, layout, row)
    lower, upper = unknown_id(layout, term.interface.lower), unknown_id(layout, term.interface.upper)
    field = _electric_field_of(term.interface)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        m_lower, m_upper = momentum_grid(kinetic_grid(layout, lower), ir), momentum_grid(kinetic_grid(layout, upper), ir)
        for j in 1:pitch_cell_count(m_lower)
            push!(entries, (r, _kinetic_index(layout, lower, ir, momentum_cell_count(m_lower), j)))
        end
        for j in 1:pitch_cell_count(m_upper)
            push!(entries, (r, _kinetic_index(layout, upper, ir, 1, j)))
        end
        field === nothing || push!(entries, (r, first(unknown_range(layout, unknown_id(layout, field), ir))))
    end
    return entries
end
