# Terms of the electron populations around the distributions: the particle
# source of the hot-tail grid with its magnitude as an unknown (DREAM's ParticleSourceTerm
# with S_particle), and the flux through the upper momentum boundary of a grid into a
# density or a current (the electrons above the runaway grid, or a fluid runaway density).

"""
    ParticleSourceTerm(source, grid, temperature; shape=:relativistic_maxwellian)

Particle source of a kinetic unknown on `grid` with the magnitude of the radial unknown
`source` [1/(m³ s)]: `S(r) g(p)`, with the shape `g` normalized on the grid to one particle
per volume, so that the term adds `S` particles per volume and second and nothing else
(DREAM's `ParticleSourceTerm`). The magnitude is an unknown whose equation is the
constraint it enforces, e.g. `0 = n(f) − n_cold − n_hot` with the kind
[`Constraint`](@ref): the source moves electrons between the cold population and the
distribution as electrons run away or the free-electron density changes.

`shape`:

- `:relativistic_maxwellian`: `exp(-(γ − 1)/Θ)` with `Θ = T/mₑc²` (DREAM's default);
- `:maxwellian`: the non-relativistic `exp(-p²/2Θ)` of [`MaxwellianSourceTerm`](@ref);
- `:lowest_cell`: the momentum cell at `p = 0` alone (DREAM's delta shape).

`temperature` is the cold-electron temperature in eV at each radius, or a function of time
that returns the temperatures or a [`PlasmaState`](@ref). The shape follows it in time and
is rebuilt when the temperature of a radius changes. It is prescribed data: the term is
linear, and its Jacobian entries are assembled directly.

`temperature` may also be the name of the radial unknown of the temperature. The term is
then not linear, and its Jacobian holds the column of the temperature, `S ∂g/∂T`, with the
derivative of the normalization on the grid, which DREAM's column leaves out
(`ParticleSourceTerm.cpp`). The momentum integral of that column vanishes, as the source
adds `S` particles at any temperature.
"""
struct ParticleSourceTerm{T,S} <: EquationTerm
    source::Symbol
    grid::KineticGrid{T}
    temperature::S
    shape_kind::Symbol
    shape::Vector{Matrix{T}}        # per radius, density 1
    temperatures::Vector{Float64}   # the temperatures of the shape, empty before the first evaluation
    time::Base.RefValue{Float64}    # the time of the last evaluation of a temperature in time
end

function ParticleSourceTerm(source::Symbol, grid::KineticGrid{T}, temperature;
                            shape::Symbol=:relativistic_maxwellian) where {T}
    shape in (:relativistic_maxwellian, :maxwellian, :lowest_cell) ||
        throw(ArgumentError("shape must be :relativistic_maxwellian, :maxwellian or :lowest_cell, not $shape"))
    temperature isa AbstractVector && length(temperature) != radial_cell_count(grid) &&
        throw(DimensionMismatch("the temperature needs one value per radius"))
    values = [zeros(T, momentum_cell_count(grid, ir), pitch_cell_count(grid, ir)) for ir in 1:radial_cell_count(grid)]
    return ParticleSourceTerm{T,typeof(temperature)}(source, grid, temperature, shape, values, Float64[], Ref(NaN))
end

_temperature_unknown(term::ParticleSourceTerm) = term.temperature isa Symbol ? (term.temperature,) : ()
dependencies(term::ParticleSourceTerm) = [term.source; collect(Symbol, _temperature_unknown(term))]
is_linear(term::ParticleSourceTerm) = isempty(_temperature_unknown(term))
jacobian_assembly(::ParticleSourceTerm) = :direct

function validate(term::ParticleSourceTerm, layout, row)
    id = unknown_id(layout, term.source)
    layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
        throw(ArgumentError("the magnitude of a ParticleSourceTerm is a radial unknown with one component"))
    layout.kinds[row] === :kinetic || throw(ArgumentError("ParticleSourceTerm adds to a kinetic unknown"))
    _same_momentum_grids(term.grid, kinetic_grid(layout, row)) ||
        throw(DimensionMismatch("the grid of the term does not match $(layout.unknowns[row].name)"))
    for name in _temperature_unknown(term)
        id = unknown_id(layout, name)
        layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
            throw(ArgumentError("the temperature of a ParticleSourceTerm is a radial unknown with one component"))
    end
    return nothing
end

# the shape of the source on one momentum grid, with density 1
function _source_shape!(values::AbstractMatrix, kind::Symbol, grid::KineticGrid, ir, temperature)
    m = momentum_grid(grid, ir)
    # a DomainError: with the temperature an unknown, a Newton iterate can reach it, and the
    # solvers take that as a failed solve
    kind === :lowest_cell || temperature > 0 ||
        throw(DomainError(temperature, "the temperature of the particle source must be positive, it is " *
                                       "$temperature at radius $ir"))
    theta = temperature / Constants.ELECTRON_REST_ENERGY
    for j in axes(values, 2), (i, p) in enumerate(m.p_centers)
        values[i, j] = kind === :lowest_cell ? (i == 1 ? 1.0 : 0.0) :
                       kind === :maxwellian ? exp(-p^2 / (2theta)) :
                       exp(-p^2 / (sqrt(1 + p^2) + 1) / theta)    # γ − 1 = p²/(γ + 1)
    end
    jacobian = phase_space_jacobian(grid, ir)
    norm = zero(eltype(values))
    for j in axes(values, 2), i in axes(values, 1)
        norm += jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] * values[i, j]
    end
    norm > 0 && isfinite(norm) ||
        throw(DomainError(temperature, "the shape of the particle source at radius $ir has no finite density on " *
                                       "the momentum grid at the temperature $temperature"))
    values ./= norm / grid.radial_grid.spatial_jacobian[ir]
    return values
end

# the shape at the state u and the time t, rebuilt when the temperatures differ from those
# of the shape
function _shape_at(term::ParticleSourceTerm, u, layout, t)
    time = Float64(_value(t))
    built = !isempty(term.temperatures)
    if term.temperature isa Symbol
        u === nothing && throw(ArgumentError("a particle source with the temperature $(term.temperature) as an " *
                                             "unknown needs the state of the system"))
        temperatures = [_value(x) for x in radial_values(u, layout, unknown_id(layout, term.temperature))]
    else
        built && (term.temperature isa AbstractVector || term.time[] == time) && return term.shape
        temperatures = _temperatures(term.temperature, time)
    end
    term.time[] = time
    length(temperatures) == radial_cell_count(term.grid) ||
        throw(DimensionMismatch("the temperature needs one value per radius"))
    update_key!(term.temperatures, temperatures) || return term.shape
    try
        for ir in 1:radial_cell_count(term.grid)
            _source_shape!(term.shape[ir], term.shape_kind, term.grid, ir, temperatures[ir])
        end
    catch
        reset_key!(term.temperatures)
        rethrow()
    end
    return term.shape
end

"""
    source_shape(term::ParticleSourceTerm, ir, t=0)

Shape of the source at the radius index `ir` and the time `t`: the particles per volume,
second and phase-space volume that a magnitude of 1/(m³ s) adds to each cell. A term whose
temperature is an unknown needs the state, `source_shape(term, ir, u, layout, t)`.
"""
source_shape(term::ParticleSourceTerm, ir::Integer, t=0.0) = _shape_at(term, nothing, nothing, t)[ir]
source_shape(term::ParticleSourceTerm, ir::Integer, u, layout, t) = _shape_at(term, u, layout, t)[ir]

# the shapes in the number type of a temperature unknown that carries derivative parts, so
# that the AD of rhs! holds the column of the temperature
function _dual_shapes(term::ParticleSourceTerm, u, layout)
    temperature = radial_values(u, layout, unknown_id(layout, term.temperature))
    return [_source_shape!(zeros(eltype(temperature), size(term.shape[ir])), term.shape_kind, term.grid, ir,
                           temperature[ir]) for ir in 1:radial_cell_count(term.grid)]
end

function rhs!(out, term::ParticleSourceTerm, u, layout, row, t)
    shapes = term.temperature isa Symbol && eltype(u) <: ForwardDiff.Dual ? _dual_shapes(term, u, layout) :
             _shape_at(term, u, layout, t)
    magnitude = radial_values(u, layout, unknown_id(layout, term.source))
    for ir in 1:radial_cell_count(layout)
        added = kinetic_view(out, layout, row, ir)
        shape = shapes[ir]
        S = magnitude[ir]
        @inbounds for k in eachindex(shape)
            added[k] += S * shape[k]
        end
    end
    return out
end

function add_jacobian!(J, term::ParticleSourceTerm, u, layout, row, t)
    shapes = _shape_at(term, u, layout, t)
    id = unknown_id(layout, term.source)
    for ir in 1:radial_cell_count(layout)
        column = first(unknown_range(layout, id, ir))
        shape = shapes[ir]
        for (k, r) in enumerate(unknown_range(layout, row, ir))
            _add_entry!(J, r, column, shape[k])
        end
    end
    term.temperature isa Symbol || return J
    # the column of the temperature: S ∂g/∂T, the normalization included
    D = ForwardDiff.Dual{PlasmaColumnTag,Float64,1}
    temperature = radial_values(u, layout, unknown_id(layout, term.temperature))
    magnitude = radial_values(u, layout, id)
    T_id = unknown_id(layout, term.temperature)
    for ir in 1:radial_cell_count(layout)
        shape = _source_shape!(zeros(D, size(term.shape[ir])), term.shape_kind, term.grid, ir,
                               D(temperature[ir], ForwardDiff.Partials((1.0,))))
        column = first(unknown_range(layout, T_id, ir))
        for (k, r) in enumerate(unknown_range(layout, row, ir))
            _add_entry!(J, r, column, magnitude[ir] * ForwardDiff.partials(shape[k], 1))
        end
    end
    return J
end

function sparsity!(entries, term::ParticleSourceTerm, layout, row)
    for name in (term.source, _temperature_unknown(term)...), ir in 1:radial_cell_count(layout)
        column = first(unknown_range(layout, unknown_id(layout, name), ir))
        append!(entries, ((r, column) for r in unknown_range(layout, row, ir)))
    end
    return entries
end

# tag of the dual numbers of the direct Jacobian of the boundary flux
struct BoundaryFluxTag end

# a 3D array with a unit derivative at one cell, for the derivative of a flux with respect
# to that cell
struct PerturbedArray{T,A<:AbstractArray{T,3}} <: AbstractArray{ForwardDiff.Dual{BoundaryFluxTag,T,1},3}
    values::A
    cell::NTuple{3,Int}
end
Base.size(a::PerturbedArray) = size(a.values)
Base.@propagate_inbounds function Base.getindex(a::PerturbedArray{T}, i::Int, j::Int, ir::Int) where {T}
    return ForwardDiff.Dual{BoundaryFluxTag,T,1}(a.values[i, j, ir],
                                                 ForwardDiff.Partials((T((i, j, ir) == a.cell),)))
end

# a field with a unit derivative at one radius
struct PerturbedField{T,A<:AbstractVector{T}} <: AbstractVector{ForwardDiff.Dual{BoundaryFluxTag,T,1}}
    values::A
    radius::Int
end
Base.size(a::PerturbedField) = size(a.values)
Base.@propagate_inbounds Base.getindex(a::PerturbedField{T}, ir::Int) where {T} =
    ForwardDiff.Dual{BoundaryFluxTag,T,1}(a.values[ir], ForwardDiff.Partials((T(ir == a.radius),)))

"""
    MomentumBoundaryFluxTerm(kinetic, source; weight=:density, coefficient=1)

`coefficient` times the flux of the kinetic unknown `kinetic` through the upper momentum
boundary of its grid, per volume, added to the row of a radial unknown with one component:
the electrons that leave the grid at `p_max`. `source` holds the coefficients of the flux:
the Fokker-Planck term of `kinetic` ([`FokkerPlanckTerm`](@ref) or
[`PrescribedFokkerPlanckTerm`](@ref)) or its operator. The flux is that of the boundary
condition of the operator at `p_max`, so the term and the operator count the same
particles. With a term that takes the electric field as an unknown, the flux holds the
field of the state.

- `weight = :density`: the particles per volume and second, `(1/V') Σ_j Δξ_j 𝒱' Γᵖ`. With
  `TimeDerivative()` the row unknown is the density of the electrons above the grid, or a
  fluid runaway density fed by the hot-tail grid.
- `weight = :current`: the flux weighted with `e c ξ`, the current that the electrons carry
  when they leave, at the speed of light. With `TimeDerivative()` the row unknown is the
  current density of the electrons above the grid.

The entries of the row are assembled directly, from the derivative of the flux of each
pitch cell with respect to the cells at the boundary.
"""
struct MomentumBoundaryFluxTerm{S} <: EquationTerm
    kinetic::Symbol
    source::S
    weight::Symbol
    coefficient::Float64
end
_coefficient_sources(term::MomentumBoundaryFluxTerm) = (term.source,)
function MomentumBoundaryFluxTerm(kinetic::Symbol, source; weight::Symbol=:density, coefficient::Real=1.0)
    source isa Union{AdvectionDiffusion,FokkerPlanckTerm} ||
        throw(ArgumentError("the source of the flux is a Fokker-Planck term or an AdvectionDiffusion operator"))
    weight in (:density, :current) || throw(ArgumentError("weight must be :density or :current, not $weight"))
    _operator_of(source).p_upper_boundary === :interface &&
        throw(ArgumentError("the flux through an interface belongs to the MomentumInterfaceTerm"))
    return MomentumBoundaryFluxTerm(kinetic, source, weight, Float64(coefficient))
end

# momentum cells below the boundary that its flux reads (the extrapolated fluxes with a
# four-point scheme reach furthest)
const BOUNDARY_FLUX_DEPTH = 5

_electric_field_of(term::MomentumBoundaryFluxTerm) = _electric_field_of(term.source)

function dependencies(term::MomentumBoundaryFluxTerm)
    field = _electric_field_of(term)
    return field === nothing ? [term.kinetic] : [term.kinetic, field]
end
is_linear(term::MomentumBoundaryFluxTerm) =
    _electric_field_of(term) === nothing && is_linear(_operator_of(term.source))
jacobian_assembly(::MomentumBoundaryFluxTerm) = :direct

function validate(term::MomentumBoundaryFluxTerm, layout, row)
    # the flux follows the plasma unknowns of the Fokker-Planck term, whose columns this
    # term does not have yet
    has_plasma_columns(term.source) &&
        throw(ArgumentError("MomentumBoundaryFluxTerm reads the coefficients of a Fokker-Planck term whose plasma follows unknowns, " *
                            "and has no Jacobian columns for them yet (issue #143): use coefficients = :lagged"))
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic || throw(ArgumentError("MomentumBoundaryFluxTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("MomentumBoundaryFluxTerm adds to a radial unknown with one component"))
    _same_momentum_grids(_operator_of(term.source).grid, kinetic_grid(layout, id)) ||
        throw(DimensionMismatch("the grid of the operator does not match $(term.kinetic)"))
    return nothing
end

# the factor of the flux of the pitch cell j in the row of the radius
Base.@propagate_inbounds function _boundary_weight(term::MomentumBoundaryFluxTerm, m::MomentumGrid, rg::RadialGrid, j, ir)
    weight = term.coefficient * m.xi_widths[j] / rg.spatial_jacobian[ir]
    term.weight === :density && return weight
    return weight * Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT * m.xi_centers[j]
end

function rhs!(out, term::MomentumBoundaryFluxTerm, u, layout, row, t)
    op = _operator_at(term.source, u, layout, t)
    f = kinetic_array(u, layout, unknown_id(layout, term.kinetic))
    name = _electric_field_of(term)
    offsets = name === nothing ? nothing : layout.radius_offsets[unknown_id(layout, name)]
    return _add_boundary_flux!(out, term, op, f, u, offsets, layout, row)
end

# behind a function barrier, with the field built from the offsets of its unknown: a field
# that is either a view of the state or nothing would be allocated on its way here
function _add_boundary_flux!(out, term::MomentumBoundaryFluxTerm, op::AdvectionDiffusion, f, u, offsets, layout, row)
    field = offsets === nothing ? nothing : LayoutRadialValues(u, offsets)
    rg = layout.radial_grid
    @inbounds for ir in 1:radial_cell_count(layout)
        m = op.momentum_grids[ir]
        face = momentum_cell_count(m) + 1
        total = zero(eltype(out))
        for j in 1:pitch_cell_count(m)
            total += _boundary_weight(term, m, rg, j, ir) * _p_flux(op, f, face, j, ir, nothing, field)
        end
        out[first(unknown_range(layout, row, ir))] += total
    end
    return out
end

function add_jacobian!(J, term::MomentumBoundaryFluxTerm, u, layout, row, t)
    op = _operator_at(term.source, u, layout, t)
    id = unknown_id(layout, term.kinetic)
    f = kinetic_array(u, layout, id)
    field_name = _electric_field_of(term)
    field = _field_values(field_name, u, layout)
    rg = layout.radial_grid
    for ir in 1:radial_cell_count(layout)
        m = op.momentum_grids[ir]
        np, nxi = momentum_cell_count(m), pitch_cell_count(m)
        r = first(unknown_range(layout, row, ir))
        for j in 1:nxi
            weight = _boundary_weight(term, m, rg, j, ir)
            for other in max(j - 1, 1):min(j + 1, nxi), i in max(np - BOUNDARY_FLUX_DEPTH + 1, 1):np
                flux = _p_flux(op, PerturbedArray(f, (i, other, ir)), np + 1, j, ir, nothing, field)
                derivative = ForwardDiff.partials(flux, 1)
                derivative == 0 || _add_entry!(J, r, _kinetic_index(layout, id, ir, i, other), weight * derivative)
            end
            if field !== nothing
                flux = _p_flux(op, f, np + 1, j, ir, nothing, PerturbedField(field, ir))
                _add_entry!(J, r, first(unknown_range(layout, unknown_id(layout, field_name), ir)),
                            weight * ForwardDiff.partials(flux, 1))
            end
        end
    end
    return J
end

function sparsity!(entries, term::MomentumBoundaryFluxTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    field = _electric_field_of(term)
    for ir in 1:radial_cell_count(layout)
        np, nxi = layout.shapes[id, ir]
        r = first(unknown_range(layout, row, ir))
        for j in 1:nxi, i in max(np - BOUNDARY_FLUX_DEPTH + 1, 1):np
            push!(entries, (r, _kinetic_index(layout, id, ir, i, j)))
        end
        field === nothing || push!(entries, (r, first(unknown_range(layout, unknown_id(layout, field), ir))))
    end
    return entries
end
