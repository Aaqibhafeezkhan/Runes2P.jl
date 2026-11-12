# The energy that the distributions exchange with the total energy of the electrons: the
# collisional transfer from the runaway grid to the cold electrons, the sinks of the hot
# electrons (bound electrons, bremsstrahlung, synchrotron), and the energy and the particles
# that electrons carry out of the hot-tail grid through the interface. Included after
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
  and, in the fully kinetic mode, the energy diffusion. For a superthermal electron it is the
  `∫ v F_s f` of DREAM's `CollisionalEnergyTransferKineticTerm` as the energy change of the
  operator, without the energy that the friction carries out through `p = 0`; that term
  itself, as a moment, is [`FrictionEnergyTransferTerm`](@ref). With `coefficient = 1` it
  is the channel of the runaway grid into the total energy of the electrons on the
  hot-tail grid ([`total_energy_equation`](@ref)).
- `part = :sinks`: `C_a` the collisions of `source` without the field, `C_b` their
  electron-electron part: the energy that leaves the electrons by the friction on bound
  electrons (excitation and ionization), bremsstrahlung and synchrotron radiation. With
  `coefficient = −1` and the hot region of the total energy as `region` it is the loss of
  that total, so the cold electrons receive exactly the free-electron part of what the hot
  electrons lose. Both operators are applied, since the weights of exponential fitting
  depend on all coefficients of an operator. With `bound_friction = :ionization_split` of
  [`CollisionSettings`](@ref) the friction on bound electrons holds the energy of the
  kinetic ionization ([`KineticIonizationTerm`](@ref)) up to the discretization and the
  width of the threshold (see there): the ionization
  potentials, which the ions store and release at recombination, and the energy of the
  secondaries, which [`SecondaryHeatingTerm`](@ref) gives back to the cold electrons
  ([`ionization_energy_flows`](@ref)). With DREAM's `:bethe` it holds that energy
  only on average and only where the screening window is open: from the ionization
  potential up to the mean excitation energy it takes none.

The term takes the plasma of `source` in the same treatment: implicit, with the columns of
the temperature and density unknowns, or lagged at the state of the start of the step. A
region whose limit follows the temperature adds the column of its envelope.

A lagged plasma (`lagged = :all` of `source`) is that of the last [`prepare_step!`](@ref)
of the term, which [`residual!`](@ref) and the right-hand side of the system call before
every evaluation, and which the system calls with its initial state for a term never
prepared when it builds the prototype of its Jacobian. A term evaluated on its own reads
the plasma of its last preparation, by whatever system or call made it, and a term never
prepared throws an `ArgumentError`. Lagged ion densities alone are those of the last
`prepare_step!` of `source`, or those of the state when `source` was never prepared.
"""
struct KineticEnergyLossTerm{F <: FokkerPlanckTerm, E} <: EquationTerm
    kinetic::Symbol
    source::F
    part::Symbol
    coefficient::Float64
    envelopes::E                        # nothing, or the envelopes of the region per radius
    lagged_plasma::Base.RefValue{Any}   # the plasma of the start of the step, for a lagged source
end
function KineticEnergyLossTerm(source::FokkerPlanckTerm; part::Symbol = :free_electrons,
        region::Union{Nothing, MomentumRegion} = nothing,
        coefficient::Union{Nothing, Real} = nothing)
    part in (:free_electrons, :sinks) ||
        throw(ArgumentError("part must be :free_electrons or :sinks, not $part"))
    value = if coefficient === nothing
        part === :free_electrons ? 1.0 : -1.0
    else
        Float64(coefficient)
    end
    return KineticEnergyLossTerm(
        source.unknown, source, part, value, RegionEnvelopes(region, source.operator.grid),
        Ref{Any}(nothing))
end

"""
    free_electron_energy_transfer_term(source::FokkerPlanckTerm)

The power that the electrons of the unknown of `source` give to the free cold electrons by
collisions: [`KineticEnergyLossTerm`](@ref) with `part = :free_electrons`, the transfer from
the runaway grid to the cold electrons.
"""
function free_electron_energy_transfer_term(source::FokkerPlanckTerm)
    KineticEnergyLossTerm(source; part = :free_electrons)
end

lags_in_ode_form(term::KineticEnergyLossTerm) = lags_in_ode_form(term.source)
function _follows_unknowns(term::KineticEnergyLossTerm)
    is_state_dependent(term.source.plasma) && !_is_lagged(term.source)
end
# the plasma unknowns of the iterate the loss reads: none with the whole plasma lagged, and
# without the ion densities when the source lags them
function _loss_plasma_unknowns(term::KineticEnergyLossTerm)
    _follows_unknowns(term) || return Symbol[]
    names = collect(Symbol, source_dependencies(term.source.plasma))
    if _lags_ions(term.source)
        return filter(!=(ion_dependency(term.source.plasma)), names)
    else
        return names
    end
end
function dependencies(term::KineticEnergyLossTerm)
    unique([term.kinetic; _loss_plasma_unknowns(term);
            collect(Symbol, _region_unknowns(term.envelopes))])
end
is_linear(term::KineticEnergyLossTerm) = length(dependencies(term)) == 1
jacobian_assembly(::KineticEnergyLossTerm) = :direct
_coefficient_sources(term::KineticEnergyLossTerm) = (term.source,)

function validate(term::KineticEnergyLossTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("KineticEnergyLossTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("KineticEnergyLossTerm adds to a radial unknown with one component"))
    term.source.settings.mode in (:full, :superthermal) ||
        throw(ArgumentError("the energy loss takes collision mode :full or :superthermal"))
    is_linear(term.source.operator) ||
        throw(ArgumentError("the energy loss takes the collisions of a Fokker-Planck term with a linear scheme: a " *
                            "flux limiter makes it depend on f nonlinearly"))
    term.source.plasma === nothing &&
        term.source.operator.coefficient_source[] === nothing &&
        throw(ArgumentError("the Fokker-Planck term of the energy loss has no plasma"))
    _check_envelopes(term.envelopes, kinetic_grid(layout, id))
    return nothing
end

function prepare_step!(term::KineticEnergyLossTerm, u_old, layout, row, t)
    if _is_lagged(term.source)
        term.lagged_plasma[] = plasma_values(term.source.plasma, u_old, layout, t)
    end
    return nothing
end

# A system builds the prototype of its Jacobian before any step: a lagged loss that no step
# has prepared takes the initial state there as the start of the step. A loss prepared
# before, by another system or call, keeps its plasma.
function _prepare_for_prototype!(term::KineticEnergyLossTerm, u, layout, row, t)
    if _is_lagged(term.source) && term.lagged_plasma[] === nothing
        prepare_step!(term, u, layout, row, t)
    end
    return nothing
end

# the plasma of the collisions: of the state, of the start of the step, or that of the
# operator when the source has none
function _transfer_plasma(term::KineticEnergyLossTerm, u, layout, t)
    plasma = term.source.plasma
    plasma === nothing && return term.source.operator.coefficient_source[].plasma
    if _is_lagged(term.source)
        if term.lagged_plasma[] === nothing
            throw(ArgumentError("the plasma of the start of the step of the lagged energy loss is not set: it is " *
                                "set by residual! and by the right-hand side of the system, and a term " *
                                "evaluated on its own is prepared with prepare_step! first"))
        end
        return term.lagged_plasma[]::PlasmaState
    end
    return _fokker_planck_plasma(term.source, u, layout, t)
end

# The operators C_a and C_b of the part, with the scheme and the boundaries of the source.
# The sinks take no flux through p_max (the value :interface): the energy of the electrons
# that leave there is that of MomentumBoundaryFluxTerm(...; weight = :energy) in the row of
# the total energy, and with the flux of the source's boundary the sink part of it would be
# counted twice.
function _loss_operators(term::KineticEnergyLossTerm, grid, plasma)
    op = term.source.operator
    upper = term.part === :sinks ? :interface : op.p_upper_boundary
    free = electron_electron_operator(
        grid, plasma, term.source.settings; scheme = op.p_scheme,
        p_lower_boundary = op.p_lower_boundary, p_upper_boundary = upper,
        p_zero_boundary = op.p_zero_boundary)
    term.part === :free_electrons && return (free, nothing)
    collisions = AdvectionDiffusion(grid; p_scheme = op.p_scheme, xi_scheme = op.xi_scheme,
        p_lower_boundary = op.p_lower_boundary, p_upper_boundary = upper,
        p_zero_boundary = op.p_zero_boundary)
    C = eltype(plasma.cold_electron_temperature)
    if C !== Float64
        collisions = similar_coefficients(collisions, C)
    end
    set_fokker_planck_coefficients!(
        collisions, plasma, term.source.settings; synchrotron = term.source.synchrotron,
        field_in_coefficients = false)
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
    return [energies[i] * V[i, j] * m.p_widths[i] * m.xi_widths[j] /
            grid.radial_grid.spatial_jacobian[ir]
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
    plasma = _transfer_plasma(term, u, layout, t)
    ops = _loss_operators(term, grid, plasma)
    f = kinetic_array(u, layout, id)
    df = _loss_change(ops, f)
    for ir in 1:radial_cell_count(layout)
        loss_weights = _loss_weights(term, grid, ir, u, layout, t)
        #! format: off
        out[first(unknown_range(layout, row, ir))] -=
            term.coefficient * sum(loss_weights .* view(df, :, :, ir))
        #! format: on
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
    weights = reduce(vcat, (vec(_loss_weights(term, grid, ir, u, layout, t))
    for ir in 1:nr))
    row_values = transpose(operator_matrix(ops[1])) * weights
    if ops[2] !== nothing
        row_values .-= transpose(operator_matrix(ops[2])) * weights
    end
    # the cells of the kinetic array (p fastest, then ξ, then r) with a nonzero weight
    for (k, value) in zip(findnz(sparse(row_values))...)
        radius, within = divrem(k - 1, np * nxi)
        pitch, momentum = divrem(within, np)
        #! format: off
        _add_entry!(J, first(unknown_range(layout, row, radius + 1)),
                    _kinetic_index(layout, id, radius + 1, momentum + 1, pitch + 1), -term.coefficient * value)
        #! format: on
    end
    f = kinetic_array(u, layout, id)
    # the columns of the plasma unknowns: the operators of a plasma with derivative parts
    if _follows_unknowns(term)
        source = term.source.plasma
        with_ions = ion_dependency(source) !== nothing && !_lags_ions(term.source)
        dual = dual_plasma(plasma, source.density !== nothing, with_ions)
        df = _loss_change(_loss_operators(term, grid, dual), f)
        _add_loss_plasma_columns!(J, term, df, grid, with_ions, u, layout, row, t)
    end
    # the column of the temperature that moves the limit of the region
    temperature = _region_unknowns(term.envelopes)
    if !isempty(temperature)
        df = _loss_change(ops, f)
        T_id = unknown_id(layout, only(temperature))
        D = ForwardDiff.Dual{PlasmaColumnTag, Float64, 1}
        T = unknown_values(u, layout, T_id)
        limit = term.envelopes.region.limit
        for ir in 1:nr
            m = momentum_grid(grid, ir)
            p0 = limit.factor * sqrt(2 * D(T[ir], ForwardDiff.Partials((1.0,))) /
                      Constants.ELECTRON_REST_ENERGY)
            envelope = momentum_envelope!(zeros(D, momentum_cell_count(m)), term.envelopes.region, m, p0)
            energies = _energy_weights(grid, ir)
            value = 0.0
            for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)

                value += ForwardDiff.partials(envelope[i], 1) * energies[i, j] *
                         df[i, j, ir]
            end
            _add_entry!(J, first(unknown_range(layout, row, ir)),
                first(unknown_range(layout, T_id, ir)),
                -term.coefficient * value)
        end
    end
    return J
end

# the radial rows with the change of f in dual numbers `df`; the number of partials is
# picked at run time, so the loop runs behind this barrier on the concrete type
function _add_loss_plasma_columns!(J, term::KineticEnergyLossTerm,
        df::Array{<:ForwardDiff.Dual, 3}, grid, with_ions, u, layout, row, t)
    source = term.source.plasma
    for ir in 1:size(df, 3)
        loss_weights = _loss_weights(term, grid, ir, u, layout, t)
        loss = -term.coefficient * sum(loss_weights .* view(df, :, :, ir))
        columns = plasma_columns(source, layout, ir; with_ions)::Vector{Int}
        _add_partials!(J, first(unknown_range(layout, row, ir)), columns, loss)
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
            append!(entries, ((r, c)
            for c in unknown_range(layout, unknown_id(layout, name), ir)))
        end
    end
    return entries
end

"""
    FrictionEnergyTransferTerm(kinetic, grid, plasma; settings=FRICTION_TRANSFER_SETTINGS,
                               coefficient=1)

`coefficient` times the power [W/m³] that the electrons of the kinetic unknown `kinetic` on
`grid` give to the free cold electrons by friction, for the row of the temperature of the
cold electrons, as the moment

```math
P = \\frac{1}{V'} \\sum_{ij} m_e c\\, v_i\\, p_i\\, ν_s(p_i)\\, f_{ij}\\, 𝒱'_{ij} Δp_i Δξ_j,
```

with `v` and `p` at the momentum cell centres and the slowing-down frequency `ν_s`
([`slowing_down_frequency`](@ref)) of `settings` and of `plasma`, a [`PlasmaState`](@ref), a
function of time that returns one, or a [`PlasmaSource`](@ref) such as
[`PlasmaFromUnknowns`](@ref). This is DREAM's `CollisionalEnergyTransferKineticTerm`
(`src/Equations/Fluid/CollisionalEnergyTransferKineticTerm.cpp`, DREAM v26.5), the heating of
the cold electrons by `f_hot` in a run without the full collision frequencies
(`src/Settings/Equations/T_cold.cpp:318-340`) and by `f_re` in every run (lines 341-354), for
which [`free_electron_energy_transfer_term`](@ref) is Runes2P's default. The default `settings`,
[`FRICTION_TRANSFER_SETTINGS`](@ref), are those DREAM fixes for it: the superthermal
frequencies with complete screening, so that the electrons collide with the cold electrons
alone, the energy-dependent Coulomb logarithm and no bremsstrahlung.

The moment reads the distribution at the cell centres, so it holds whatever scheme the
operator of `f` takes, and it counts the energy of an electron until it leaves the grid,
also through `p = 0` where the friction of the superthermal frequencies carries it out
under `p_zero_boundary = :outflow` (compare [`KineticEnergyLossTerm`](@ref), the energy
change of the electron-electron operator itself). The term is linear in `f`. With a plasma
of unknowns its Jacobian holds the columns of the temperature, the density and, when the
plasma reads them, the ion charge states, as DREAM's. The weights of the moment are kept
until the plasma values change. A [`PlasmaState`](@ref) given as `plasma` is read once, so a
change made to it in place afterwards is not seen.
"""
struct FrictionEnergyTransferTerm{S} <: EquationTerm
    kinetic::Symbol
    grid::KineticGrid{Float64}
    source::S
    settings::CollisionSettings
    coefficient::Float64
    plasma_key::PlasmaKey                # the plasma values of `weights`
    weights::Vector{Matrix{Float64}}     # per radius: the weights of the plasma of `plasma_key`
    shapes::Vector{Int}                  # per radius: the shape of its cells (`_friction_shapes`)
    products::Vector{Matrix{Float64}}    # per shape: the weights times f of a radius
    dual_scratch::Base.RefValue{Any}     # (weights, products) per shape in dual numbers of the plasma columns
end

"""
    FRICTION_TRANSFER_SETTINGS

The collision settings of [`FrictionEnergyTransferTerm`](@ref): `mode = :superthermal`,
`screening = :complete`, `coulomb_logarithm = :energy_dependent`, `bremsstrahlung = :neglect`,
the settings that DREAM's `CollisionalEnergyTransferKineticTerm` fixes for its slowing-down
frequency (`src/Equations/Fluid/CollisionalEnergyTransferKineticTerm.cpp:20-27`, DREAM v26.5):
the mode, the screening and the logarithm set there, and no bremsstrahlung as its comment
states, which is DREAM's default.
"""
const FRICTION_TRANSFER_SETTINGS = CollisionSettings(; mode = :superthermal,
    screening = :complete, coulomb_logarithm = :energy_dependent, bremsstrahlung = :neglect)

function FrictionEnergyTransferTerm(kinetic::Symbol, grid::KineticGrid, plasma;
        settings::CollisionSettings = FRICTION_TRANSFER_SETTINGS, coefficient::Real = 1.0)
    weights = _friction_buffers(Float64, grid)
    shapes = _friction_shapes(grid)
    products = _friction_scratch(Float64, grid, shapes)
    return FrictionEnergyTransferTerm(
        kinetic, grid, plasma_source(plasma), settings, Float64(coefficient), PlasmaKey(),
        weights, shapes, products, Ref{Any}(nothing))
end

# a matrix of the momentum and pitch cells of every radius of `grid`, of the number type T
function _friction_buffers(::Type{T}, grid::KineticGrid) where {T}
    buffers = Matrix{T}[]
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        push!(buffers, zeros(T, momentum_cell_count(m), pitch_cell_count(m)))
    end
    return buffers
end

# per radius of `grid`, the index of the shape of its cells (the momentum and pitch cell
# counts) among the shapes of the grid, numbered in the order of their first radius. A
# scratch matrix of one radius serves every radius of its shape: one of another size, or a
# view into a larger one, would sum the products in another order.
function _friction_shapes(grid::KineticGrid)
    sizes = Tuple{Int, Int}[]
    shapes = zeros(Int, radial_cell_count(grid))
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        shape = (momentum_cell_count(m), pitch_cell_count(m))
        index = findfirst(==(shape), sizes)
        if index === nothing
            push!(sizes, shape)
            index = length(sizes)
        end
        shapes[ir] = index
    end
    return shapes
end

# a matrix of the number type T for each shape of `shapes` (`_friction_shapes`)
function _friction_scratch(::Type{T}, grid::KineticGrid, shapes::Vector{Int}) where {T}
    scratch = Matrix{T}[]
    for ir in 1:radial_cell_count(grid)
        if shapes[ir] > length(scratch)
            m = momentum_grid(grid, ir)
            push!(scratch, zeros(T, momentum_cell_count(m), pitch_cell_count(m)))
        end
    end
    return scratch
end

function dependencies(term::FrictionEnergyTransferTerm)
    unique([term.kinetic; collect(Symbol, source_dependencies(term.source))])
end
is_linear(term::FrictionEnergyTransferTerm) = !is_state_dependent(term.source)
jacobian_assembly(::FrictionEnergyTransferTerm) = :direct

function validate(term::FrictionEnergyTransferTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("FrictionEnergyTransferTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("FrictionEnergyTransferTerm adds to a radial unknown with one component"))
    _same_momentum_grids(kinetic_grid(layout, id), term.grid) ||
        throw(ArgumentError("the grid of FrictionEnergyTransferTerm is not that of $(term.kinetic)"))
    ions = ion_dependency(term.source)
    if ions !== nothing
        _check_ion_unknown(layout, unknown_id(layout, ions),
            IonComposition(plasma_values(term.source.prescribed, nothing, nothing, 0.0)))
    end
    return nothing
end

# mₑc v p ν_s 𝒱' Δp Δξ / V' of every cell of radius ir, in the number type of the plasma
function _friction_weights(term::FrictionEnergyTransferTerm, model, ir)
    m = momentum_grid(term.grid, ir)
    T = eltype(model.free_electron_density)
    weights = zeros(T, momentum_cell_count(m), pitch_cell_count(m))
    return _friction_weights!(weights, term, model, ir)
end

# the weights of `_friction_weights` written into `weights`, every entry
function _friction_weights!(weights, term::FrictionEnergyTransferTerm, model, ir)
    m = momentum_grid(term.grid, ir)
    V = phase_space_jacobian(term.grid, ir)
    for i in 1:momentum_cell_count(m)
        p = m.p_centers[i]
        v = Constants.SPEED_OF_LIGHT * p / sqrt(1 + p^2)
        power = Constants.ELECTRON_MASS * Constants.SPEED_OF_LIGHT * v * p *
                slowing_down_frequency(model, ir, p)
        for j in 1:pitch_cell_count(m)
            weights[i, j] = power * V[i, j] * m.p_widths[i] * m.xi_widths[j] /
                            term.grid.radial_grid.spatial_jacobian[ir]
        end
    end
    return weights
end

# The weights of the plasma at the state u and the time t, rebuilt when the plasma values
# differ from those of the last rebuild (`changed_plasma!`); the slowing-down frequency does
# not read the electric field.
function _current_friction_weights!(term::FrictionEnergyTransferTerm, u, layout, t)
    plasma = changed_plasma!(
        term.plasma_key, term.source, u, layout, t; electric_field = false)
    plasma === nothing && return term.weights
    try
        model = CollisionModel(plasma, term.settings)
        for ir in 1:radial_cell_count(layout)
            _friction_weights!(term.weights[ir], term, model, ir)
        end
    catch
        reset_key!(term.plasma_key)
        rethrow()
    end
    return term.weights
end

function rhs!(out, term::FrictionEnergyTransferTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    # a state in dual numbers takes weights of its own evaluation, kept nowhere
    if !(u isa AbstractVector{Float64})
        model = CollisionModel(plasma_values(term.source, u, layout, t), term.settings)
        for ir in 1:radial_cell_count(layout)
            weights = _friction_weights(term, model, ir)
            power = sum(weights .* kinetic_view(u, layout, id, ir))
            out[first(unknown_range(layout, row, ir))] += term.coefficient * power
        end
        return out
    end
    weights = _current_friction_weights!(term, u, layout, t)
    for ir in 1:radial_cell_count(layout)
        products = term.products[term.shapes[ir]]
        products .= weights[ir] .* kinetic_view(u, layout, id, ir)
        power = sum(products)
        out[first(unknown_range(layout, row, ir))] += term.coefficient * power
    end
    return out
end

function add_jacobian!(J, term::FrictionEnergyTransferTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    weights = _current_friction_weights!(term, u, layout, t)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        radius_weights = weights[ir]
        for j in axes(radius_weights, 2), i in axes(radius_weights, 1)

            _add_entry!(J, r, _kinetic_index(layout, id, ir, i, j),
                term.coefficient * radius_weights[i, j])
        end
    end
    # the columns of the plasma unknowns: the weights of a plasma with derivative parts
    if is_state_dependent(term.source)
        with_ions = ion_dependency(term.source) !== nothing
        plasma = plasma_values(term.source, u, layout, t)
        dual = CollisionModel(
            dual_plasma(plasma, term.source.density !== nothing, with_ions), term.settings)
        _add_friction_plasma_columns!(J, term, dual, with_ions, u, layout, row)
    end
    return J
end

# the scratch matrices of the weights and the products in the dual numbers D, one per shape
# of the cells, made at the first use of D and kept in the term
function _dual_friction_scratch(term::FrictionEnergyTransferTerm, ::Type{D}) where {D}
    kept = term.dual_scratch[]
    if kept isa Tuple{Vector{Matrix{D}}, Vector{Matrix{D}}}
        return kept
    end
    scratch = (_friction_scratch(D, term.grid, term.shapes),
        _friction_scratch(D, term.grid, term.shapes))
    term.dual_scratch[] = scratch
    return scratch
end

# the radial rows with the collision model of a plasma in dual numbers `dual`; the number of
# partials is picked at run time, so the loop runs behind this barrier on the concrete type
function _add_friction_plasma_columns!(
        J, term::FrictionEnergyTransferTerm, dual::CollisionModel, with_ions, u, layout, row)
    id = unknown_id(layout, term.kinetic)
    D = eltype(dual.free_electron_density)
    weights, products = _dual_friction_scratch(term, D)
    for ir in 1:radial_cell_count(layout)
        shape = term.shapes[ir]
        radius_weights = _friction_weights!(weights[shape], term, dual, ir)
        radius_products = products[shape]
        radius_products .= radius_weights .* kinetic_view(u, layout, id, ir)
        power = term.coefficient * sum(radius_products)
        columns = plasma_columns(term.source, layout, ir; with_ions)::Vector{Int}
        _add_partials!(J, first(unknown_range(layout, row, ir)), columns, power)
    end
    return J
end

function sparsity!(entries, term::FrictionEnergyTransferTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    others = setdiff(dependencies(term), [term.kinetic])
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for s in unknown_range(layout, id, ir)
            push!(entries, (r, s))
        end
        for name in others
            append!(entries, ((r, c)
            for c in unknown_range(layout, unknown_id(layout, name), ir)))
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
cross take their energy with them and do not heat the cold ones. When the flux follows the
plasma unknowns of the Fokker-Planck term of the interface, the row has their columns
([`has_plasma_columns`](@ref)). The row reads the cells of both grids at the face, so the
term assembles it itself ([`jacobian_assembly`](@ref)) and the coloring of the automatic
Jacobian does not see it.
"""
struct InterfaceEnergyFluxTerm{I <: MomentumInterfaceTerm} <: EquationTerm
    interface::I
    coefficient::Float64
end
function InterfaceEnergyFluxTerm(interface::MomentumInterfaceTerm; coefficient::Real = -1.0)
    InterfaceEnergyFluxTerm(interface, Float64(coefficient))
end

"""
    InterfaceParticleFluxTerm(interface::MomentumInterfaceTerm; coefficient=1)

`coefficient` times the particle flux [`interface_flux`](@ref) through the momentum
interface from the lower grid to the upper one, per volume and second [m⁻³ s⁻¹], in a
radial row. With the default `coefficient = 1` in the row of the runaway density it is the
influx of DREAM's `n_re` from the hot-tail grid (`src/Settings/Equations/n_re.cpp:125-128`,
`PXiExternalKineticKinetic` with `TYPE_DENSITY`, DREAM v26.5). When the flux follows the
plasma unknowns of the Fokker-Planck term of the interface, the row has their columns
([`has_plasma_columns`](@ref)). The row reads the cells of both grids at the face, so the
term assembles it itself ([`jacobian_assembly`](@ref)) and the coloring of the automatic
Jacobian does not see it.
"""
struct InterfaceParticleFluxTerm{I <: MomentumInterfaceTerm} <: EquationTerm
    interface::I
    coefficient::Float64
end
function InterfaceParticleFluxTerm(interface::MomentumInterfaceTerm; coefficient::Real = 1.0)
    InterfaceParticleFluxTerm(interface, Float64(coefficient))
end

# the two terms of the flux through an interface in a radial row
const InterfaceFluxTerm = Union{InterfaceEnergyFluxTerm, InterfaceParticleFluxTerm}

dependencies(term::InterfaceFluxTerm) = dependencies(term.interface)
is_linear(term::InterfaceFluxTerm) = is_linear(term.interface)
_coefficient_sources(term::InterfaceFluxTerm) = _coefficient_sources(term.interface)
has_plasma_columns(term::InterfaceFluxTerm) = has_plasma_columns(term.interface)

# coefficient times the factor of the flux at radius ir: the kinetic energy of the last
# momentum cell of the lower grid, or 1 for the particles
_crossing_weight(term::InterfaceParticleFluxTerm, ir) = term.coefficient
function _crossing_weight(term::InterfaceEnergyFluxTerm, ir)
    p = momentum_grid(term.interface.lower_grid, ir).p_centers[end]
    energy = Constants.ELECTRON_REST_ENERGY * Constants.ELEMENTARY_CHARGE * p^2 /
             (1 + sqrt(1 + p^2))
    return term.coefficient * energy
end

function validate(term::InterfaceFluxTerm, layout, row)
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("$(nameof(typeof(term))) adds to a radial unknown with one component"))
    return validate(term.interface, layout, unknown_id(layout, term.interface.lower))
end

function rhs!(out, term::InterfaceFluxTerm, u, layout, row, t)
    for ir in 1:radial_cell_count(layout)
        flux = interface_flux(term.interface, u, layout, ir, t)
        out[first(unknown_range(layout, row, ir))] += _crossing_weight(term, ir) * flux
    end
    return out
end

jacobian_assembly(::InterfaceFluxTerm) = :direct

# the columns of the row that one evaluation of the flux in dual numbers seeds: wide enough
# for the face cells of two grids of 15 pitch cells and the field in one evaluation, with one
# compiled copy of the code for all grids. The default runaway grid has 20 pitch cells
# (runaway_grid), so next to a hot-tail grid of 15 it takes two evaluations per radius. The
# entries do not depend on it
const INTERFACE_COLUMN_CHUNK = 32

# the columns of the cells at the face and of the field, by the flux of dual numbers seeded
# on them, a chunk of columns at a time in the order of sparsity!: the lower cells, the upper
# cells, the field. The other cells of the distributions are not read, and the columns of
# the plasma unknowns come from add_plasma_columns!
function add_jacobian!(J, term::InterfaceFluxTerm, u, layout, row, t)
    interface = term.interface
    op = _operator_at(interface.lower_term, u, layout, t)
    lower = unknown_id(layout, interface.lower)
    upper = unknown_id(layout, interface.upper)
    field_name = _electric_field_of(interface)
    field_values = _field_values(field_name, u, layout)
    nr = radial_cell_count(layout)
    D = ForwardDiff.Dual{LocalColumnTag, Float64, INTERFACE_COLUMN_CHUNK}
    lower_values = zeros(D, maximum(ir -> layout.shapes[lower, ir][2], 1:nr))
    upper_values = zeros(D, maximum(ir -> layout.shapes[upper, ir][2], 1:nr))
    field = field_name === nothing ? nothing : zeros(D, nr)
    for ir in 1:nr
        r = first(unknown_range(layout, row, ir))
        f_lower = kinetic_view(u, layout, lower, ir)
        f_upper = kinetic_view(u, layout, upper, ir)
        lower_cells, lower_pitch_cells = layout.shapes[lower, ir]
        upper_cells, upper_pitch_cells = layout.shapes[upper, ir]
        face_lower = FaceCells(
            view(lower_values, 1:lower_pitch_cells), lower_cells, lower_cells)
        face_upper = FaceCells(view(upper_values, 1:upper_pitch_cells), 1, upper_cells)
        column_count = lower_pitch_cells + upper_pitch_cells
        if field !== nothing
            column_count += 1
        end
        for start in 1:INTERFACE_COLUMN_CHUNK:column_count
            # column k of the row carries partial k - start + 1, a column outside the chunk none
            for j in 1:lower_pitch_cells
                lower_values[j] = _seeded_dual(D, f_lower[lower_cells, j], j - start + 1)
            end
            for k in 1:upper_pitch_cells
                partial = lower_pitch_cells + k - start + 1
                upper_values[k] = _seeded_dual(D, f_upper[1, k], partial)
            end
            if field !== nothing
                field[ir] = _seeded_dual(D, field_values[ir], column_count - start + 1)
            end
            flux = _interface_flux_of_cells(
                interface, op, face_lower, face_upper, field, ir)
            rate = _crossing_weight(term, ir) * flux
            stop = min(start + INTERFACE_COLUMN_CHUNK - 1, column_count)
            for k in start:stop
                column = _interface_column(
                    layout, lower, upper, field_name, ir, k, lower_pitch_cells)
                _add_entry!(J, r, column, ForwardDiff.partials(rate, k - start + 1))
            end
        end
    end
    return J
end

# column k of the row at radius ir in the order of sparsity!: the last momentum cells of the
# lower grid, the first of the upper grid, the field
function _interface_column(layout, lower, upper, field_name, ir, k, lower_pitch_cells)
    if k <= lower_pitch_cells
        return _kinetic_index(layout, lower, ir, layout.shapes[lower, ir][1], k)
    end
    pitch_cell = k - lower_pitch_cells
    if pitch_cell <= layout.shapes[upper, ir][2]
        return _kinetic_index(layout, upper, ir, 1, pitch_cell)
    end
    return first(unknown_range(layout, unknown_id(layout, field_name), ir))
end

function plasma_column_sparsity!(entries, term::InterfaceFluxTerm, layout, row)
    source = term.interface.lower_term
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        columns = plasma_columns(source.plasma, layout, ir; with_ions = !_lags_ions(source))
        append!(entries, ((r, column) for column in columns))
    end
    return entries
end

# the flux with the coefficients of the Fokker-Planck term in dual numbers
function add_plasma_columns!(J, term::InterfaceFluxTerm, u, layout, row, t)
    source = term.interface.lower_term
    op, with_ions = _dual_coefficients!(source, u, layout, t)
    return _add_crossing_plasma_columns!(J, term, op, with_ions, u, layout, row)
end

# the radial rows with the operator of dual coefficients `op`; the number of partials is
# picked at run time, so the loop runs behind this barrier on the concrete type
function _add_crossing_plasma_columns!(
        J, term::InterfaceFluxTerm, op::AdvectionDiffusion, with_ions, u, layout, row)
    source = term.interface.lower_term
    for ir in 1:radial_cell_count(layout)
        flux = _interface_flux(term.interface, op, u, layout, ir)
        r = first(unknown_range(layout, row, ir))
        columns = plasma_columns(source.plasma, layout, ir; with_ions)::Vector{Int}
        _add_partials!(J, r, columns, _crossing_weight(term, ir) * flux)
    end
    return J
end

# the row reads the last momentum cells of the lower grid, the first of the upper grid, and
# the field when the interface takes it
function sparsity!(entries, term::InterfaceFluxTerm, layout, row)
    lower, upper = unknown_id(layout, term.interface.lower),
    unknown_id(layout, term.interface.upper)
    field = _electric_field_of(term.interface)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        m_lower, m_upper = momentum_grid(kinetic_grid(layout, lower), ir),
        momentum_grid(kinetic_grid(layout, upper), ir)
        for j in 1:pitch_cell_count(m_lower)
            push!(entries, (
                r, _kinetic_index(layout, lower, ir, momentum_cell_count(m_lower), j)))
        end
        for j in 1:pitch_cell_count(m_upper)
            push!(entries, (r, _kinetic_index(layout, upper, ir, 1, j)))
        end
        if field !== nothing
            push!(entries, (r, first(unknown_range(layout, unknown_id(layout, field), ir))))
        end
    end
    return entries
end
