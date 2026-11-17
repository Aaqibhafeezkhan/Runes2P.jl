# The energy that the distributions exchange with the total energy of the electrons: the
# collisional transfer from the runaway grid to the cold electrons, the sinks of the hot
# electrons (bound electrons, bremsstrahlung, synchrotron), the energy and the particles
# that electrons carry out of the hot-tail grid through the interface, the work of the
# electron-electron collisions on the electrons that cross it, and the energy that
# a source term gives a distribution. Included after operators/field_particle.jl, whose
# electron-electron operator it uses, and after system/avalanche_terms.jl, whose
# RosenbluthPutvinskiSourceTerm the energy of a source dispatches on.

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
operators of the plasma, collision settings, scheme and lower boundaries of `source`.
Neither part takes a flux through `p_max`, whatever the upper boundary of `source`: the
electrons that cross there carry their energy across the edge of the grid. That energy is
counted by [`MomentumBoundaryFluxTerm`](@ref) with `weight = :energy` at an outer boundary
and by [`InterfaceEnergyFluxTerm`](@ref) at the interface to a runaway grid. The parts:

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

With linear schemes of `source` the term takes the moment of the two operators applied to
`f`, whose face values take the side of their own advection. A flux limiter in a scheme of
`source`, in p, in ξ or in r, makes the term nonlinear in `f`. The momentum face values of
both operators then take the upwind side of the whole advection of `source`, also under a
linear momentum scheme: its collisions and its electric field, of the plasma or of the
field unknown, as the residual of `source` evaluates them. That face value depends on `f`
and on the side alone, so the field part, the free-electron part and the sinks sum to the
momentum flux of `source` face by face. Exponential fitting in p is the exception: its
face value follows the coefficients of each operator, not the side, so the parts do not
sum to the flux there. The term takes the moment in the form of the momentum faces, since
the pitch fluxes sum to zero over a momentum row:

```math
L = −\\frac{1}{V'} \\sum_j Δξ_j \\sum_k (ε_k − ε_{k−1})\\, F_{kj},
```

with `ε_k = w_k mₑc²(γ_k − 1)` [J] of momentum cell `k` (`ε_0 = 0`) and `F_{kj}` the momentum
flux of `C_a − C_b` through the lower face of cell `(k, j)` times `𝒱'`. The term reads the
sign of the whole advection, not of the field alone, and has no column of the field
unknown. Where a face changes its side, each part jumps, while their sum, the momentum
flux of `source`, stays continuous. The Jacobian holds the derivative of the limiter at a
fixed side.

The term takes the plasma of `source` in the same treatment: implicit, with the columns of
the temperature and density unknowns, or lagged at the state of the start of the step. A
region whose limit follows the temperature adds the column of its envelope. The term builds
its two operators from the plasma at every evaluation of its residual and its Jacobian.

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
    row_buffer::Base.RefValue{Any}      # the entries of the row of a radius in the columns of f (face form)
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
        Ref{Any}(nothing), Ref{Any}(nothing))
end

"""
    free_electron_energy_transfer_term(source::FokkerPlanckTerm)

The power [W/m³] that the electrons of the unknown of `source` give to the free cold
electrons by collisions: [`KineticEnergyLossTerm`](@ref) with `part = :free_electrons`, the
transfer from the runaway grid to the cold electrons. The energy that the electrons carry
through `p_max` is not in it. The work of the collisions on the electrons that cross the
interface to a hot-tail grid is [`InterfaceCollisionWorkTerm`](@ref).
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
# Whether the loss takes the form of the momentum faces (_face_energy_moment), with the parts
# on the side of the whole advection of the source: exactly when a flux limiter makes the
# operator of the source nonlinear in f. A linear source takes the moment of the applied
# operators, each on the side of its own advection.
_face_form(term::KineticEnergyLossTerm) = !is_linear(term.source.operator)
# the field unknown whose sign the face form reads, without a column of its own
function _side_field_unknowns(term::KineticEnergyLossTerm)
    field = term.source.electric_field
    return _face_form(term) && field !== nothing ? [field] : Symbol[]
end
function dependencies(term::KineticEnergyLossTerm)
    unique([term.kinetic; _loss_plasma_unknowns(term); _side_field_unknowns(term);
            collect(Symbol, _region_unknowns(term.envelopes))])
end
function is_linear(term::KineticEnergyLossTerm)
    length(dependencies(term)) == 1 && is_linear(term.source.operator)
end
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

# The operators C_a and C_b of the part, with the scheme and the lower boundaries of the
# source. Both parts take no flux through p_max (the value :interface): the electrons that
# cross there carry the energy of their cell across the edge of the grid. In the row of the
# total energy that energy is the boundary term's, MomentumBoundaryFluxTerm(...; weight =
# :energy) at an outer boundary and InterfaceEnergyFluxTerm at the interface to a runaway
# grid, which a flux in either part would count twice. The work of the collisions on the
# electrons that cross the interface is InterfaceCollisionWorkTerm's.
function _loss_operators(term::KineticEnergyLossTerm, grid, plasma)
    op = term.source.operator
    free = electron_electron_operator(
        grid, plasma, term.source.settings; scheme = op.p_scheme,
        p_lower_boundary = op.p_lower_boundary, p_upper_boundary = :interface,
        p_zero_boundary = op.p_zero_boundary)
    term.part === :free_electrons && return (free, nothing)
    collisions = AdvectionDiffusion(grid; p_scheme = op.p_scheme, xi_scheme = op.xi_scheme,
        p_lower_boundary = op.p_lower_boundary, p_upper_boundary = :interface,
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

# The advection that sets the upwind side of the faces in the face form: that of the
# operator of the source at the state, with its field, which the residual of the source takes
function _loss_side(term::KineticEnergyLossTerm, u, layout, t)
    operator = _operator_at(term.source, u, layout, t)
    return AdvectionSide(operator, _field_values(term.source.electric_field, u, layout))
end

# mₑc²(γ − 1) of momentum cell i times the envelope, or zero of its type for a cell without
# volume, which apply_radius! leaves unchanged
Base.@propagate_inbounds function _cell_energy(
        op::AdvectionDiffusion, envelope, m, i, j, ir)
    energy = _envelope_value(envelope, i) * _kinetic_energy(m.p_centers[i])
    op.jacobian_centers[i, j, ir] == 0 && return zero(energy)
    return energy
end

# F_a − F_b through p face k of the operators of the part, with the face values on the
# side of `side`
Base.@propagate_inbounds function _part_flux(ops, f, k, j, ir, side)
    a, b = ops
    flux = _p_flux(a, f, k, j, ir, nothing, nothing, side)
    b === nothing && return flux
    return flux - _p_flux(b, f, k, j, ir, nothing, nothing, side)
end

# The energy moment of C_a f − C_b f at radius ir in the form of the momentum faces: the
# moment Σ_ij ε̃_i [C f]_ij 𝒱'_ij Δp_i Δξ_j / V', with ε̃ the energy times the envelope, is
# −Σ_ij ε̃_i (F_{i+1,j} − F_{ij}) Δξ_j / V' plus the pitch divergence, which sums to zero over
# a momentum row since no flux crosses ξ = ±1; the operators carry no radial transport.
# Gathered per face: Σ_j Δξ_j Σ_k (ε̃_k − ε̃_{k−1}) F_{kj} / V', with ε̃_0 = 0 and no flux
# through p_max, where the parts take the interface. The number type follows the operators,
# f and the envelope. A weight of dual numbers is skipped only when its partials are zero as
# well as its value (ForwardDiff's iszero), so the column of the temperature keeps a face
# whose weight is zero at a limit of the region on that face.
function _face_energy_moment(ops, f, side, envelope, ir)
    a = first(ops)
    m = a.momentum_grids[ir]
    T = promote_type(_flux_eltype(a, f), typeof(_cell_energy(a, envelope, m, 1, 1, ir)))
    moment = zero(T)
    @inbounds for j in 1:pitch_cell_count(m)
        row = zero(T)
        below = zero(T)
        for k in 1:momentum_cell_count(m)
            above = _cell_energy(a, envelope, m, k, j, ir)
            weight = above - below
            below = above
            iszero(weight) && continue
            row += weight * _part_flux(ops, f, k, j, ir, side)
        end
        moment += m.xi_widths[j] * row
    end
    return moment / a.grid.radial_grid.spatial_jacobian[ir]
end

function rhs!(out, term::KineticEnergyLossTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    plasma = _transfer_plasma(term, u, layout, t)
    ops = _loss_operators(term, grid, plasma)
    f = kinetic_array(u, layout, id)
    if _face_form(term)
        side = _loss_side(term, u, layout, t)
        for ir in 1:radial_cell_count(layout)
            envelope = _envelope_at(term.envelopes, grid, ir, u, layout, t)
            moment = _face_energy_moment(ops, f, side, envelope, ir)
            out[first(unknown_range(layout, row, ir))] -= term.coefficient * moment
        end
        return out
    end
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
    f = kinetic_array(u, layout, id)
    if _face_form(term)
        return _add_face_loss_jacobian!(J, term, ops, f, grid, plasma, u, layout, row, t)
    end
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
        T = unknown_values(u, layout, T_id)
        for ir in 1:nr
            m = momentum_grid(grid, ir)
            envelope = _temperature_envelope(term, grid, T, ir)
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

# tag of the dual numbers of the row of the face form in the columns of f
struct LossRowTag end

# A momentum face k reads the momentum cells k − h to k + h − 1 of its pitch cell, mirrored
# ghost cells included, with h the stencil half width of the four-point schemes, the widest
# in p. Any 2h consecutive cells have different momentum indices modulo 2h, so 2h colors,
# that index modulo 2h, give each partial of the flux of a face to one cell.
const LOSS_STENCIL_HALF_WIDTH = stencil_half_width(first(FOUR_POINT_SCHEMES))
const LOSS_ROW_COLORS = 2 * LOSS_STENCIL_HALF_WIDTH

# The distribution f with the derivative of cell (i, j) in the partial of its color. The
# operators of the term carry no cross diffusion, whose flux would read the pitch
# neighbours: the collisions write none.
struct ColoredCells{T, A <: AbstractArray{T, 3}} <:
       AbstractArray{ForwardDiff.Dual{LossRowTag, T, LOSS_ROW_COLORS}, 3}
    values::A
end
Base.size(a::ColoredCells) = size(a.values)
@inline _cell_color(i) = mod(i, LOSS_ROW_COLORS) + 1
Base.@propagate_inbounds function Base.getindex(
        a::ColoredCells{T}, i::Int, j::Int, ir::Int) where {T}
    color = _cell_color(i)
    seeds = ntuple(c -> T(c == color), Val(LOSS_ROW_COLORS))
    return ForwardDiff.Dual{LossRowTag, T, LOSS_ROW_COLORS}(
        a.values[i, j, ir], ForwardDiff.Partials(seeds))
end

# the envelope of the region at radius ir in dual numbers of one partial, the derivative by
# the temperature unknown T that moves its limit
function _temperature_envelope(term::KineticEnergyLossTerm, grid, T, ir)
    D = ForwardDiff.Dual{PlasmaColumnTag, Float64, 1}
    m = momentum_grid(grid, ir)
    limit = term.envelopes.region.limit
    p0 = limit.factor * sqrt(2 * D(T[ir], ForwardDiff.Partials((1.0,))) /
              Constants.ELECTRON_REST_ENERGY)
    return momentum_envelope!(zeros(D, momentum_cell_count(m)), term.envelopes.region, m, p0)
end

# the buffer of the row of one radius in the columns of f, kept by the term; the operator
# takes the same cell counts at all radii
function _loss_row_buffer!(term::KineticEnergyLossTerm, np, nxi)
    buffer = term.row_buffer[]
    if !(buffer isa Matrix{Float64} && size(buffer) == (np, nxi))
        buffer = zeros(np, nxi)
        term.row_buffer[] = buffer
    end
    return buffer::Matrix{Float64}
end

# The Jacobian of the face form: the columns of f by the face fluxes of colored dual
# numbers, the plasma columns by the operators of a plasma with derivative parts, and the
# column of the temperature that moves the limit of the region. The side is read as a
# number: the derivative of the face value takes the side as fixed, as the residual of the
# source does.
function _add_face_loss_jacobian!(J, term::KineticEnergyLossTerm, ops, f, grid, plasma,
        u, layout, row, t)
    side = _loss_side(term, u, layout, t)
    _add_face_loss_columns!(J, term, ops, ColoredCells(f), side, grid, u, layout, row, t)
    if _follows_unknowns(term)
        source = term.source.plasma
        with_ions = ion_dependency(source) !== nothing && !_lags_ions(term.source)
        dual = dual_plasma(plasma, source.density !== nothing, with_ions)
        dual_ops = _loss_operators(term, grid, dual)
        _add_face_loss_plasma_columns!(
            J, term, dual_ops, f, side, grid, with_ions, u, layout, row, t)
    end
    temperature = _region_unknowns(term.envelopes)
    if !isempty(temperature)
        T_id = unknown_id(layout, only(temperature))
        T = unknown_values(u, layout, T_id)
        for ir in 1:radial_cell_count(layout)
            envelope = _temperature_envelope(term, grid, T, ir)
            moment = _face_energy_moment(ops, f, side, envelope, ir)
            _add_entry!(J, first(unknown_range(layout, row, ir)),
                first(unknown_range(layout, T_id, ir)),
                -term.coefficient * ForwardDiff.partials(moment, 1))
        end
    end
    return J
end

# The columns of f: the flux of each momentum face with a nonzero weight once in dual
# numbers of the colors of `colored`, its partials added to the cells the face reads
function _add_face_loss_columns!(J, term::KineticEnergyLossTerm, ops,
        colored::ColoredCells, side, grid, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    a = first(ops)
    np, nxi, nr = size(grid)
    buffer = _loss_row_buffer!(term, np, nxi)
    for ir in 1:nr
        m = a.momentum_grids[ir]
        envelope = _envelope_at(term.envelopes, grid, ir, u, layout, t)
        volume = a.grid.radial_grid.spatial_jacobian[ir]
        fill!(buffer, 0.0)
        @inbounds for j in 1:nxi
            below = 0.0
            for k in 1:np
                above = Float64(_cell_energy(a, envelope, m, k, j, ir))
                weight = (above - below) * m.xi_widths[j] / volume
                below = above
                weight == 0 && continue
                partials = ForwardDiff.partials(_part_flux(ops, colored, k, j, ir, side))
                cells = max(k - LOSS_STENCIL_HALF_WIDTH, 1):min(k + LOSS_STENCIL_HALF_WIDTH - 1, np)
                for i in cells
                    buffer[i, j] += weight * partials[_cell_color(i)]
                end
            end
        end
        r = first(unknown_range(layout, row, ir))
        for jj in 1:nxi, i in 1:np

            value = buffer[i, jj]
            value == 0 && continue
            entry = -term.coefficient * value
            _add_entry!(J, r, _kinetic_index(layout, id, ir, i, jj), entry)
        end
    end
    return J
end

# the plasma columns of the face form with the operators of dual coefficients `ops`; the
# number of partials is picked at run time, so the loop runs behind this barrier
function _add_face_loss_plasma_columns!(J, term::KineticEnergyLossTerm, ops, f, side, grid,
        with_ions, u, layout, row, t)
    source = term.source.plasma
    for ir in 1:radial_cell_count(layout)
        envelope = _envelope_at(term.envelopes, grid, ir, u, layout, t)
        value = -term.coefficient * _face_energy_moment(ops, f, side, envelope, ir)
        columns = plasma_columns(source, layout, ir; with_ions)::Vector{Int}
        _add_partials!(J, first(unknown_range(layout, row, ir)), columns, value)
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

# the field of the face form gives the side alone and has no column
function sparsity!(entries, term::KineticEnergyLossTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    others = setdiff(
        [_loss_plasma_unknowns(term); collect(Symbol, _region_unknowns(term.envelopes))],
        [term.kinetic])
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
cross take their energy with them and do not heat the cold ones. The energy that they gain or
lose between the two cells at the face is not in this term: the part of the
electron-electron collisions is [`InterfaceCollisionWorkTerm`](@ref). When the flux follows the
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

"""
    InterfaceCollisionWorkTerm(interface::MomentumInterfaceTerm; coefficient=-1)

`coefficient` times the work [W/m³] that the electron-electron collisions do on the
electrons that cross the momentum interface, in a radial row: the part `Γ_ee` [m⁻³ s⁻¹] of
[`interface_flux`](@ref) that the collisions with the free electrons drive, times the
energy step between the first momentum cell of the upper grid and the last of the lower
grid,

```math
P = Γ_{ee} \\left[E_k(p_{\\mathrm{up},1}) − E_k(p_{\\mathrm{low},N})\\right], \\quad E_k(p) = m_e c^2 (γ − 1).
```

`Γ_ee` is positive from the lower grid to the upper one. The step is positive, so `P` is
positive where the collisions push electrons up through the face (the energy diffusion) and
negative where they pull them down (the friction). An electron that crosses leaves the lower
grid with the energy of its last cell and enters the upper grid with the energy of its first
cell ([`InterfaceEnergyFluxTerm`](@ref)).

No operator holds the step, since neither grid takes a flux through the face in its energy
moments ([`KineticEnergyLossTerm`](@ref)). The cold electrons pay it for `Γ_ee`. The sinks
pay it for the friction on bound electrons, the energy diffusion by bound electrons under
partial screening, bremsstrahlung and synchrotron radiation. The row of a total energy
credits the cold electrons with the part of the step of the field up to the face, which the
runaway grid receives as well (see the section *The energy step at the interface* of the
docs page *Energy and current balance*).

With the default `coefficient = −1` in the row of the temperature the cold electrons pay the
work of the electron-electron collisions, as they gain the energy that these collisions take
from the electrons on either grid. Charging that work to the cold electrons is a choice of
this package (see the docs page *Choices of this package*). In every row the system refuses
a second `InterfaceCollisionWorkTerm` of one interface, which would count the work twice. In
the row of a total energy ([`total_energy_equation`](@ref)) it requires one such term next to
the [`InterfaceEnergyFluxTerm`](@ref) of the same interface, with a negative coefficient. It
refuses there one of an interface whose lower distribution, that of the grid below the face,
is not in the total energy. A row of the cold thermal energy alone ([`thermal_energy_equation`](@ref)) takes the term without these
two conditions.

Each overlap cell of the face carries the flux `Γ = A f_face − D (f_up − f_low)/Δp`. `A` and
`D` are the coefficients of the lower operator at the face, the field included, and `Δp` is
the distance of the two cell centres. The face value `f_face` is that of the scheme of the
interface, which the whole coefficients set: the upwind side of `A`, the linear
interpolation of `:central`, or, for exponential fitting, the weight `(B(−w) − 1)/w` of the
lower cell with `B(x) = x/(eˣ − 1)` and `w = A Δp/D`. With `pitch_mapping = :dream` it is
the upwind side of `A` against the upper value interpolated in the pitch.

`Γ_ee` takes the same face value with the coefficients of the electron-electron collisions
alone, `A_ee = −p ν_s,ee` [mₑc/s] and, in `mode = :full`, `D_ee = Θ γ ν_s,ee` [(mₑc)²/s].
Here `p` is the momentum of the face [mₑc], `ν_s,ee` [s⁻¹] the slowing-down frequency on the
free electrons and `Θ = T/(mₑc²)`, as in [`electron_electron_operator`](@ref). The parts of
the field, of these collisions and of the sinks therefore add up to the flux of the
interface.

The term reads the plasma and the settings of the record of the coefficients of the lower
operator at each evaluation, so a lagged plasma or lagged ions of the Fokker-Planck term are
taken as the flux takes them. It compares the values of that plasma, not the object. So it
agrees with the operator whenever the operator was rebuilt after the last change of that
plasma, as in a run. It keeps the coefficients of the face. It rebuilds them when the
Fokker-Planck term of the interface rebuilds its coefficients, or when those plasma values or
settings differ from the kept ones.

An operator not yet built from a plasma, such as that of a lagged term before its first
[`prepare_step!`](@ref), holds no collisions, and the term is zero there. The settings take
`mode = :full` or `:superthermal`.

The row reads the cells of both grids at the face. It reads the plasma columns of the
Fokker-Planck term when its coefficients follow the plasma unknowns
([`has_plasma_columns`](@ref)). It reads the column of the field only with exponential
fitting and `pitch_mapping = :overlap`, where the weight of the face follows the field. It
reads none when a Fokker-Planck term with a plasma and `mode = :superthermal` holds the
coefficients, since the energy diffusion is zero there and the face takes the upwind side.
The term assembles the row itself ([`jacobian_assembly`](@ref)).
"""
struct InterfaceCollisionWorkTerm{I <: MomentumInterfaceTerm} <: EquationTerm
    interface::I
    coefficient::Float64
    plasma_key::PlasmaKey                                 # the plasma values of the kept coefficients
    collision_settings::Base.RefValue{CollisionSettings}  # their collision settings
    operator_rebuilds::Base.RefValue{Int}                 # the rebuilds of the Fokker-Planck term at them
    rebuilds::Base.RefValue{Int}                          # the number of their rebuilds
    advection::Vector{Float64}                            # per radius: Aᵖ of the electron-electron collisions at the face
    diffusion::Vector{Float64}                            # per radius: Dᵖᵖ of the electron-electron collisions at the face
end
function InterfaceCollisionWorkTerm(
        interface::MomentumInterfaceTerm; coefficient::Real = -1.0)
    radii = radial_cell_count(interface.lower_grid)
    return InterfaceCollisionWorkTerm(interface, Float64(coefficient), PlasmaKey(),
        Ref(CollisionSettings()), Ref(0), Ref(0), zeros(radii), zeros(radii))
end

# the two terms of the flux through an interface in a radial row
const InterfaceFluxTerm = Union{InterfaceEnergyFluxTerm, InterfaceParticleFluxTerm}
# the terms of a radial row that read the cells at the face of an interface
const InterfaceRowTerm = Union{InterfaceFluxTerm, InterfaceCollisionWorkTerm}

dependencies(term::InterfaceRowTerm) = dependencies(term.interface)
is_linear(term::InterfaceRowTerm) = is_linear(term.interface)
_coefficient_sources(term::InterfaceRowTerm) = _coefficient_sources(term.interface)
has_plasma_columns(term::InterfaceRowTerm) = has_plasma_columns(term.interface)

# coefficient times the factor of the flux at radius ir: the kinetic energy of the last
# momentum cell of the lower grid, or 1 for the particles
_crossing_weight(term::InterfaceParticleFluxTerm, ir) = term.coefficient
function _crossing_weight(term::InterfaceEnergyFluxTerm, ir)
    p = momentum_grid(term.interface.lower_grid, ir).p_centers[end]
    energy = Constants.ELECTRON_REST_ENERGY * Constants.ELEMENTARY_CHARGE * p^2 /
             (1 + sqrt(1 + p^2))
    return term.coefficient * energy
end

function validate(term::InterfaceRowTerm, layout, row)
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("$(nameof(typeof(term))) adds to a radial unknown with one component"))
    _validate_collisions(term)
    return validate(term.interface, layout, unknown_id(layout, term.interface.lower))
end

_validate_collisions(::InterfaceFluxTerm) = nothing
function _validate_collisions(term::InterfaceCollisionWorkTerm)
    settings = _interface_collision_settings(term.interface)
    if settings === nothing
        throw(ArgumentError("the lower operator of the interface of InterfaceCollisionWorkTerm holds no collisions " *
                            "of a plasma: the term needs the plasma of its coefficients"))
    end
    settings.mode in (:full, :superthermal) ||
        throw(ArgumentError("InterfaceCollisionWorkTerm takes collision mode :full or :superthermal, not " *
                            ":$(settings.mode)"))
    return nothing
end

# the collision settings of the coefficients of the lower operator of an interface, nothing
# when no plasma builds them
function _interface_collision_settings(interface::MomentumInterfaceTerm)
    source = interface.lower_term
    if source isa FokkerPlanckTerm && source.plasma !== nothing
        return source.settings
    end
    built = _operator_of(source).coefficient_source[]
    built === nothing && return nothing
    return built.settings
end

function rhs!(out, term::InterfaceFluxTerm, u, layout, row, t)
    for ir in 1:radial_cell_count(layout)
        flux = interface_flux(term.interface, u, layout, ir, t)
        out[first(unknown_range(layout, row, ir))] += _crossing_weight(term, ir) * flux
    end
    return out
end

# coefficient times the energy step [J] of radius ir: the kinetic energy of the first momentum
# cell of the upper grid less that of the last cell of the lower grid
function _collision_work_weight(term::InterfaceCollisionWorkTerm, ir)
    lower = momentum_grid(term.interface.lower_grid, ir).p_centers[end]
    upper = momentum_grid(term.interface.upper_grid, ir).p_centers[1]
    step = _kinetic_energy(upper) - _kinetic_energy(lower)
    return term.coefficient * step
end

# The weight of the lower cell in the face value of exponential fitting, (B(−w) − 1)/w with
# the Péclet number w = A Δp/D and B the Bernoulli function (`_bernoulli`): the flux
# D/Δp (B(−w) f_low − B(w) f_up) is A (δ f_low + (1 − δ) f_up) − D (f_up − f_low)/Δp, since
# B(−w) = B(w) + w. Near w = 0 the Taylor series to w⁷, whose next term, w⁹/47900160, is
# below 2.1e-17 for |w| < 0.1.
function _fitting_weight(w)
    if abs(w) < 0.1
        return 1 / 2 + w / 12 - w^3 / 720 + w^5 / 30240 - w^7 / 1209600
    end
    return (_bernoulli(-w) - 1) / w
end

# the face value of the flux of the scheme between two cell centres, set by the whole
# coefficients A and D, so that Γ = A f_face − D (f_right − f_left)/distance is `_face_flux`
@inline function _interface_face_value(scheme, A, D, f_left, f_right, distance, weight_left)
    if scheme === :exponential_fitting
        if D > 0
            weight = _fitting_weight(A * distance / D)
            return weight * f_left + (1 - weight) * f_right
        end
        return A >= 0 ? f_left : f_right
    end
    return _face_value(scheme, A, f_left, f_right, weight_left)
end

# Γᵖ of the part `A_part`, `D_part` of the coefficients at the face between lower pitch cell j
# and upper pitch cell J, with the face value of the whole flux (`_interface_gamma`)
Base.@propagate_inbounds function _interface_part_gamma(
        interface::MomentumInterfaceTerm, op, A_part, D_part, f_lower, f_upper, j, J, ir,
        distance, weight_left, field)
    np = size(f_lower, 1)
    A = _advection_p(op, field, np + 1, j, ir)
    D = op.diffusion_p_p[np + 1, j, ir]
    f_left = f_lower[np, j]
    if interface.pitch_mapping === :dream
        f_right = _dream_upper_value(interface, f_upper, j, ir)
        f_face = A > 0 ? f_left : f_right
        return A_part * f_face + D_part * (f_left - f_right) / distance
    end
    f_right = f_upper[1, J]
    f_face = _interface_face_value(
        interface.scheme, A, D, f_left, f_right, distance, weight_left)
    return A_part * f_face - D_part * (f_right - f_left) / distance
end

# the flux per volume [m⁻³ s⁻¹] through the face at radius ir of the part `A_part`, `D_part` of
# the coefficients, the same in every pitch cell, as `_interface_flux_of_cells` forms the flux
function _interface_part_flux(interface::MomentumInterfaceTerm, op, A_part, D_part,
        f_lower, f_upper, field, ir)
    distance, weight_left = _interface_distance(interface, ir)
    m = momentum_grid(interface.lower_grid, ir)
    np = momentum_cell_count(m)
    overlap = interface.overlaps[ir]
    T = promote_type(eltype(f_lower), eltype(f_upper), eltype(op.advection_p),
        typeof(A_part), typeof(D_part), _field_type(field))
    total = zero(T)
    @inbounds for j in 1:pitch_cell_count(m)
        V = op.jacobian_p_faces[np + 1, j, ir]
        # the overlap cells of lower cell j with their fractions, or the cell itself (:dream)
        cell = zero(T)
        if interface.pitch_mapping === :dream
            cell += V * _interface_part_gamma(interface, op, A_part, D_part, f_lower,
                f_upper, j, 0, ir, distance, weight_left, field)
        end
        if interface.pitch_mapping === :overlap
            for s in overlap.left_ranges[j]
                gamma = _interface_part_gamma(interface, op, A_part, D_part, f_lower,
                    f_upper, j, overlap.right_cells[s], ir, distance, weight_left, field)
                cell += overlap.left_fractions[s] * (V * gamma)
            end
        end
        total += m.xi_widths[j] * cell
    end
    return total / interface.lower_grid.radial_grid.spatial_jacobian[ir]
end

_field_type(::Nothing) = Bool
_field_type(field) = eltype(field)

# Aᵖ and Dᵖᵖ of the electron-electron collisions at the face of every radius, of the plasma
# `plasma` and the settings `settings`, into `advection` and `diffusion`: the coefficients of
# `electron_electron_operator` at the last p face of the lower grid
function _electron_electron_face!(advection, diffusion, interface::MomentumInterfaceTerm,
        plasma::PlasmaState, settings::CollisionSettings)
    model = CollisionModel(plasma, settings)
    for ir in eachindex(advection)
        p = last(momentum_grid(interface.lower_grid, ir).p_faces)
        nu = _electron_electron_slowing_down_frequency(model, ir, p)
        advection[ir] = -(p * nu)
        diffusion[ir] = if settings.mode === :full
            _normalized_temperature(model, ir) * sqrt(1 + p^2) * nu
        else
            zero(nu)
        end
    end
    return advection, diffusion
end

# The record of the coefficients that `set_fokker_planck_coefficients!` writes for an operator
# of numbers. The operator holds its record in a reference of unknown type, whose fields a read
# would copy to the heap and a call with them would dispatch at run time.
const FLOAT_COLLISION_RECORD = NamedTuple{
    (:plasma, :settings, :synchrotron, :electric_field),
    Tuple{PlasmaState{Float64}, CollisionSettings, Bool, Bool}}

# the electron-electron coefficients of the face for the operator `op` of the interface, from
# the plasma and the settings its coefficients were built from: kept in the term for an
# operator of numbers, and of their own evaluation for an operator of dual coefficients
function _collision_face_coefficients!(term::InterfaceCollisionWorkTerm, op::AdvectionDiffusion)
    built = op.coefficient_source[]
    # an operator not yet built from a plasma, such as that of a lagged term before its first
    # preparation, holds no collisions, and its electrons cross without their work. The empty
    # key makes the next build from a plasma rebuild the kept coefficients.
    if built === nothing
        fill!(term.advection, 0.0)
        fill!(term.diffusion, 0.0)
        reset_key!(term.plasma_key)
        return (term.advection, term.diffusion)::Tuple{Vector{Float64}, Vector{Float64}}
    end
    if eltype(op.advection_p) === Float64
        if built isa FLOAT_COLLISION_RECORD
            # a record of a known type: its fields are read without a copy
            _keep_face_coefficients!(term, built.plasma, built.settings)
        else
            _keep_face_coefficients!(term, built.plasma, built.settings)
        end
        return (term.advection, term.diffusion)::Tuple{Vector{Float64}, Vector{Float64}}
    end
    radii = length(term.advection)
    T = eltype(op.advection_p)
    return _electron_electron_face!(zeros(T, radii), zeros(T, radii), term.interface,
        built.plasma, built.settings)
end

# the rebuilds of the coefficients of the lower operator of an interface: those of its
# Fokker-Planck term, and none of an operator that no term rebuilds
_operator_rebuild_count(source::FokkerPlanckTerm) = source.rebuilds[]
_operator_rebuild_count(::AdvectionDiffusion) = 0

# The kept coefficients of the face, rebuilt when the Fokker-Planck term of the interface
# rebuilt its coefficients since, or when the plasma values or the settings differ from those
# they were built from. The key compares the values of the plasma, not its object, so a plasma
# state changed in place and an operator rebuilt by hand from it are seen. Settings of another
# object are compared by their fields. The values are those of the plasma object of the
# record: a change in place of that object without a rebuild of the operator moves the term
# and not the operator. A run prepares both at the same time, so they agree there.
function _keep_face_coefficients!(term::InterfaceCollisionWorkTerm, plasma::PlasmaState,
        settings::CollisionSettings)
    rebuilds = _operator_rebuild_count(term.interface.lower_term)
    changed = update_key!(term.plasma_key, plasma; electric_field = false)
    kept = term.collision_settings[]
    other_settings = kept !== settings && !isequal(kept, settings)
    if changed || other_settings || rebuilds != term.operator_rebuilds[]
        try
            _electron_electron_face!(
                term.advection, term.diffusion, term.interface, plasma, settings)
        catch
            reset_key!(term.plasma_key)
            rethrow()
        end
        term.collision_settings[] = settings
        term.operator_rebuilds[] = rebuilds
        term.rebuilds[] += 1
    end
    return nothing
end

function rhs!(out, term::InterfaceCollisionWorkTerm, u, layout, row, t)
    interface = term.interface
    op = _operator_at(interface.lower_term, u, layout, t)
    return _add_collision_work!(
        out, term, op, _electric_field_of(interface), u, layout, row)
end

# the rows with the operator `op` of the interface and the field of the unknown `field_name`
# (nothing without one), whose types are known at run time only, so the loop runs behind this
# barrier on the concrete types
function _add_collision_work!(
        out, term::InterfaceCollisionWorkTerm, op::AdvectionDiffusion,
        field_name, u, layout, row)
    interface = term.interface
    field = _field_values(field_name, u, layout)
    advection, diffusion = _collision_face_coefficients!(term, op)
    lower, upper = unknown_id(layout, interface.lower), unknown_id(layout, interface.upper)
    for ir in 1:radial_cell_count(layout)
        f_lower = kinetic_view(u, layout, lower, ir)
        f_upper = kinetic_view(u, layout, upper, ir)
        flux = _interface_part_flux(interface, op, advection[ir], diffusion[ir],
            f_lower, f_upper, field, ir)
        work = _collision_work_weight(term, ir) * flux
        out[first(unknown_range(layout, row, ir))] += work
    end
    return out
end

jacobian_assembly(::InterfaceRowTerm) = :direct

# The field whose column the row holds: that of the interface for the fluxes, and for the
# work of the collisions only where the weight of the face follows the field. The upwind side
# and the linear interpolation have no derivative in the field. Exponential fitting has one
# where the energy diffusion at the face is not zero.
_row_field(term::InterfaceFluxTerm) = _electric_field_of(term.interface)
function _row_field(term::InterfaceCollisionWorkTerm)
    interface = term.interface
    if interface.pitch_mapping === :overlap && interface.scheme === :exponential_fitting &&
       _has_face_diffusion(interface.lower_term)
        return _electric_field_of(interface)
    end
    return nothing
end

# Whether the operator of the interface may hold energy diffusion at the face, decided when the
# pattern is built. A Fokker-Planck term with a plasma rebuilds its coefficients from its own
# settings alone, and only `mode = :full` writes the energy diffusion, at every radius. An
# operator without such a term may be rebuilt with other settings or take diffusion set by
# hand after the pattern is built, so its row keeps the column.
function _has_face_diffusion(source::FokkerPlanckTerm)
    source.plasma === nothing || _writes_energy_diffusion(source.settings)
end
_has_face_diffusion(::AdvectionDiffusion) = true

# the coefficients that the rate of a row reads besides the operator: none for the fluxes
_row_coefficients(::InterfaceFluxTerm, op) = nothing
function _row_coefficients(term::InterfaceCollisionWorkTerm, op)
    _collision_face_coefficients!(term, op)
end

# the value of the row at radius ir from the cells at the face
function _face_rate(term::InterfaceFluxTerm, coefficients, op, f_lower, f_upper, field, ir)
    flux = _interface_flux_of_cells(term.interface, op, f_lower, f_upper, field, ir)
    return _crossing_weight(term, ir) * flux
end
function _face_rate(
        term::InterfaceCollisionWorkTerm, coefficients, op, f_lower, f_upper, field, ir)
    advection, diffusion = coefficients
    flux = _interface_part_flux(term.interface, op, advection[ir], diffusion[ir],
        f_lower, f_upper, field, ir)
    return _collision_work_weight(term, ir) * flux
end

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
function add_jacobian!(J, term::InterfaceRowTerm, u, layout, row, t)
    interface = term.interface
    op = _operator_at(interface.lower_term, u, layout, t)
    coefficients = _row_coefficients(term, op)
    lower = unknown_id(layout, interface.lower)
    upper = unknown_id(layout, interface.upper)
    field_name = _row_field(term)
    field_values = _field_values(_electric_field_of(interface), u, layout)
    nr = radial_cell_count(layout)
    D = ForwardDiff.Dual{LocalColumnTag, Float64, INTERFACE_COLUMN_CHUNK}
    lower_values = zeros(D, maximum(ir -> layout.shapes[lower, ir][2], 1:nr))
    upper_values = zeros(D, maximum(ir -> layout.shapes[upper, ir][2], 1:nr))
    # a field without its column enters with its value
    field = field_name === nothing ? field_values : zeros(D, nr)
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
        if field_name !== nothing
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
            if field_name !== nothing
                field[ir] = _seeded_dual(D, field_values[ir], column_count - start + 1)
            end
            rate = _face_rate(term, coefficients, op, face_lower, face_upper, field, ir)
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

function plasma_column_sparsity!(entries, term::InterfaceRowTerm, layout, row)
    source = term.interface.lower_term
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        columns = plasma_columns(source.plasma, layout, ir; with_ions = !_lags_ions(source))
        append!(entries, ((r, column) for column in columns))
    end
    return entries
end

# The rows with the coefficients of the Fokker-Planck term in dual numbers. For the work of the
# collisions the face value of the whole flux and the electron-electron coefficients of the
# face are both of the dual plasma of the operator.
function add_plasma_columns!(J, term::InterfaceRowTerm, u, layout, row, t)
    source = term.interface.lower_term
    op, with_ions = _dual_coefficients!(source, u, layout, t)
    coefficients = _row_coefficients(term, op)
    return _add_row_plasma_columns!(J, term, op, coefficients, with_ions, u, layout, row)
end

# The radial rows with the operator of dual coefficients `op` and the coefficients
# `coefficients` of `_row_coefficients`. The number of partials is picked at run time, so the
# loop runs behind this barrier on the concrete type.
function _add_row_plasma_columns!(J, term::InterfaceRowTerm, op::AdvectionDiffusion,
        coefficients, with_ions, u, layout, row)
    interface = term.interface
    source = interface.lower_term
    lower, upper = unknown_id(layout, interface.lower), unknown_id(layout, interface.upper)
    field = _field_values(_electric_field_of(interface), u, layout)
    for ir in 1:radial_cell_count(layout)
        f_lower = kinetic_view(u, layout, lower, ir)
        f_upper = kinetic_view(u, layout, upper, ir)
        rate = _face_rate(term, coefficients, op, f_lower, f_upper, field, ir)
        r = first(unknown_range(layout, row, ir))
        columns = plasma_columns(source.plasma, layout, ir; with_ions)::Vector{Int}
        _add_partials!(J, r, columns, rate)
    end
    return J
end

# the row reads the last momentum cells of the lower grid, the first of the upper grid, and
# the field of `_row_field`
function sparsity!(entries, term::InterfaceRowTerm, layout, row)
    lower, upper = unknown_id(layout, term.interface.lower),
    unknown_id(layout, term.interface.upper)
    field = _row_field(term)
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

"""
    KineticEnergyChangeTerm(source; region=nothing, coefficient=1)

`coefficient` times the kinetic energy that the term `source` gives the electrons of its
distribution per volume and second [W/m³], for a radial row with one component: the energy
moment of the rate `S_ij` [m⁻³ s⁻¹] that `source` adds to the cell `(i, j)`,

```math
P = \\frac{1}{V'} \\sum_{ij} w_i\\, m_e c^2 (γ_i − 1)\\, S_{ij}\\, 𝒱'_{ij} Δp_i Δξ_j,
```

with the envelope `w` of `region` (1 without one, [`MomentumRegion`](@ref)). `source` is a
[`RosenbluthPutvinskiSourceTerm`](@ref), and another source throws an `ArgumentError`. Its
rate is bilinear in the runaway density `n_re` and the total density `n_tot` [m⁻³], so the
power is `P = coefficient n_re n_tot e^±` at each radius, with

```math
e^\\pm = \\sum_i w_i\\, m_e c^2 (γ_i − 1)\\, R_i \\quad [\\mathrm{J\\, m^3/s}],
```

the sum over the momentum cells `i` whose pitch cells hold `±ξ*`, and `R_i` the rate
[`rosenbluth_putvinski_rate`](@ref) of the cell above the cutoff of the source. The term
computes `e⁺` and `e⁻` per radius from the shapes of `source` when it is built, so it counts
exactly the cells that the source fills. It takes `e⁺` or `e⁻` by the sign of the electric
field of the source, `e⁺` without one. The Jacobian holds the columns of `n_re` and
`n_tot`, and the sign of the field enters without a derivative, as in the source.

The limit of `region` is fixed: a number or one number per radius, in units of `mₑc`. A
limit that follows the temperature or another function of time throws an `ArgumentError`.

In the row of the temperature under [`total_energy_equation`](@ref) the system requires this
term for each Rosenbluth-Putvinski source of a distribution of the total energy, with that
source itself, a positive coefficient and no region, since every knock-on is in the total
energy. It refuses the term there when its source is not in the equation of such a
distribution. The term creates the energy of the knock-ons in the system, a choice of this
package (see the docs page *Choices of this package*): their primaries, the fluid runaways
above the grid, have no energy unknown, so no population pays it. The alternative, which the
package rejects, charges that energy to the cold electrons.

Next to [`RunawayFluidEnergyTransferTerm`](@ref) in the same row, the total energy gains the
energy of the knock-ons twice, since the stopping power `e c E_c` of that heating includes
the collisions that create them (see the docs page *Energy and current balance*). The term
removes neither the thermal energy of the target electrons nor the ionization energy of a
bound target, and DREAM removes neither.
"""
struct KineticEnergyChangeTerm{S, E, C} <: EquationTerm
    kinetic::Symbol
    source::S
    coefficient::Float64
    region::E      # nothing, or the fixed region of `energies`, kept for the record and the checks
    energies::C    # the moments of the shapes of the source, of a form that the source type sets
end
function KineticEnergyChangeTerm(source; region::Union{Nothing, MomentumRegion} = nothing,
        coefficient::Real = 1.0)
    kinetic = _energy_change_kinetic(source)
    if region !== nothing && is_time_dependent(region)
        throw(ArgumentError("KineticEnergyChangeTerm takes a region with a fixed limit, a number or one " *
                            "number per radius: its energies are computed when it is built"))
    end
    energies = _energy_changes(source, region)
    return KineticEnergyChangeTerm(kinetic, source, Float64(coefficient), region, energies)
end

# the kinetic unknown of a source that the energy change takes
function _energy_change_kinetic(source)
    throw(ArgumentError("KineticEnergyChangeTerm takes a RosenbluthPutvinskiSourceTerm as its source, not a " *
                        "$(nameof(typeof(source)))"))
end
_energy_change_kinetic(source::RosenbluthPutvinskiSourceTerm) = source.kinetic

# e⁺ and e⁻ per radius [J m³/s]: the energy moments of the shapes of the source with ξ* > 0
# and ξ* < 0, over the region
function _energy_changes(source::RosenbluthPutvinskiSourceTerm, region)
    grid = source.grid
    radii = radial_cell_count(grid)
    positive, negative = zeros(radii), zeros(radii)
    for ir in 1:radii
        envelope = region === nothing ? nothing : momentum_envelope(region, grid, ir)
        weights = _energy_weights(grid, ir)
        for j in axes(weights, 2), i in axes(weights, 1)

            weight = _envelope_value(envelope, i) * weights[i, j]
            positive[ir] += weight * source.positive[ir][i, j]
            negative[ir] += weight * source.negative[ir][i, j]
        end
    end
    return (; positive, negative)
end

const RosenbluthPutvinskiEnergyTerm = KineticEnergyChangeTerm{<:RosenbluthPutvinskiSourceTerm}

dependencies(term::RosenbluthPutvinskiEnergyTerm) = dependencies(term.source)
is_linear(::RosenbluthPutvinskiEnergyTerm) = false
jacobian_assembly(::RosenbluthPutvinskiEnergyTerm) = :direct

function validate(term::RosenbluthPutvinskiEnergyTerm, layout, row)
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("KineticEnergyChangeTerm adds to a radial unknown with one component"))
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("KineticEnergyChangeTerm needs a kinetic unknown"))
    _same_momentum_grids(term.source.grid, kinetic_grid(layout, id)) ||
        throw(DimensionMismatch("the grid of the source of KineticEnergyChangeTerm does not match $(term.kinetic)"))
    for name in dependencies(term)
        k = unknown_id(layout, name)
        if !(layout.kinds[k] === :radial && layout.sizes[k, 1] == 1)
            throw(ArgumentError("KineticEnergyChangeTerm reads $name, a radial unknown with one component"))
        end
    end
    return nothing
end

# e⁺ or e⁻ of radius ir, by the rule of the source for the sign of the field
function _energy_change_at(term::RosenbluthPutvinskiEnergyTerm, u, layout, ir)
    if _rp_positive_pitch(term.source, u, layout, ir)
        return term.energies.positive[ir]
    else
        return term.energies.negative[ir]
    end
end

function rhs!(out, term::RosenbluthPutvinskiEnergyTerm, u, layout, row, t)
    n_re = unknown_values(u, layout, unknown_id(layout, term.source.runaway_density))
    n_tot = unknown_values(u, layout, unknown_id(layout, term.source.total_density))
    for ir in 1:radial_cell_count(layout)
        energy = _energy_change_at(term, u, layout, ir)
        out[first(unknown_range(layout, row, ir))] += term.coefficient * n_re[ir] *
                                                      n_tot[ir] * energy
    end
    return out
end

function add_jacobian!(J, term::RosenbluthPutvinskiEnergyTerm, u, layout, row, t)
    re = unknown_id(layout, term.source.runaway_density)
    tot = unknown_id(layout, term.source.total_density)
    n_re, n_tot = unknown_values(u, layout, re), unknown_values(u, layout, tot)
    for ir in 1:radial_cell_count(layout)
        energy = term.coefficient * _energy_change_at(term, u, layout, ir)
        r = first(unknown_range(layout, row, ir))
        _add_entry!(J, r, first(unknown_range(layout, re, ir)), n_tot[ir] * energy)
        _add_entry!(J, r, first(unknown_range(layout, tot, ir)), n_re[ir] * energy)
    end
    return J
end

# the columns of the two densities, whatever the sign of the field
function sparsity!(entries, term::RosenbluthPutvinskiEnergyTerm, layout, row)
    re = unknown_id(layout, term.source.runaway_density)
    tot = unknown_id(layout, term.source.total_density)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        push!(entries, (r, first(unknown_range(layout, re, ir))),
            (r, first(unknown_range(layout, tot, ir))))
    end
    return entries
end
