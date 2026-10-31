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
(`ParticleSourceTerm.cpp`). The momentum integral of that column
vanishes, as the source adds `S` particles at any temperature.
"""
struct ParticleSourceTerm{T, S} <: EquationTerm
    source::Symbol
    grid::KineticGrid{T}
    temperature::S
    shape_kind::Symbol
    shape::Vector{Matrix{T}}        # per radius, density 1
    temperatures::Vector{Float64}   # the temperatures of the shape, empty before the first evaluation
    time::Base.RefValue{Float64}    # the time of the last evaluation of a temperature in time
end

function ParticleSourceTerm(source::Symbol, grid::KineticGrid{T}, temperature;
        shape::Symbol = :relativistic_maxwellian) where {T}
    shape in (:relativistic_maxwellian, :maxwellian, :lowest_cell) ||
        throw(ArgumentError("shape must be :relativistic_maxwellian, :maxwellian or :lowest_cell, not $shape"))
    temperature isa AbstractVector && length(temperature) != radial_cell_count(grid) &&
        throw(DimensionMismatch("the temperature needs one value per radius"))
    values = [zeros(T, momentum_cell_count(grid, ir), pitch_cell_count(grid, ir))
              for ir in 1:radial_cell_count(grid)]
    return ParticleSourceTerm{T, typeof(temperature)}(
        source, grid, temperature, shape, values, Float64[], Ref(NaN))
end

function _temperature_unknown(term::ParticleSourceTerm)
    term.temperature isa Symbol ? (term.temperature,) : ()
end
function dependencies(term::ParticleSourceTerm)
    [term.source; collect(Symbol, _temperature_unknown(term))]
end
is_linear(term::ParticleSourceTerm) = isempty(_temperature_unknown(term))
jacobian_assembly(::ParticleSourceTerm) = :direct

function validate(term::ParticleSourceTerm, layout, row)
    id = unknown_id(layout, term.source)
    layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
        throw(ArgumentError("the magnitude of a ParticleSourceTerm is a radial unknown with one component"))
    layout.kinds[row] === :kinetic ||
        throw(ArgumentError("ParticleSourceTerm adds to a kinetic unknown"))
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
function _source_shape!(
        values::AbstractMatrix, kind::Symbol, grid::KineticGrid, ir, temperature)
    m = momentum_grid(grid, ir)
    # a DomainError: with the temperature an unknown, a Newton iterate can reach it, and the
    # solvers take that as a failed solve
    if !(kind === :lowest_cell || temperature > 0)
        throw(DomainError(temperature,
            "the temperature of the particle source must be positive, it is " *
            "$temperature at radius $ir"))
    end
    theta = temperature / Constants.ELECTRON_REST_ENERGY
    for j in axes(values, 2), (i, p) in enumerate(m.p_centers)

        values[i, j] = if kind === :lowest_cell
            i == 1 ? 1.0 : 0.0
        elseif kind === :maxwellian
            exp(-p^2 / (2theta))
        else
            exp(-p^2 / (sqrt(1 + p^2) + 1) / theta)    # γ − 1 = p²/(γ + 1)
        end
    end
    jacobian = phase_space_jacobian(grid, ir)
    norm = zero(eltype(values))
    for j in axes(values, 2), i in axes(values, 1)

        norm += jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] * values[i, j]
    end
    if !(norm > 0 && isfinite(norm))
        throw(DomainError(temperature,
            "the shape at radius $ir has no finite density on " *
            "the momentum grid at the temperature $temperature"))
    end
    values ./= norm / grid.radial_grid.spatial_jacobian[ir]
    return values
end

# the shape at the state u and the time t, rebuilt when the temperatures differ from those
# of the shape
function _shape_at(term::ParticleSourceTerm, u, layout, t)
    time = Float64(_value(t))
    built = !isempty(term.temperatures)
    if term.temperature isa Symbol
        if u === nothing
            throw(ArgumentError("a particle source with the temperature $(term.temperature) as an " *
                                "unknown needs the state of the system"))
        end
        temperatures = [_value(x)
                        for x in unknown_values(u, layout, unknown_id(layout, term.temperature))]
    else
        built && (term.temperature isa AbstractVector || term.time[] == time) &&
            return term.shape
        temperatures = _temperatures(term.temperature, time)
    end
    term.time[] = time
    if length(temperatures) != radial_cell_count(term.grid)
        throw(DimensionMismatch("the temperature needs one value per radius"))
    end
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
function source_shape(term::ParticleSourceTerm, ir::Integer, t = 0.0)
    _shape_at(term, nothing, nothing, t)[ir]
end
function source_shape(term::ParticleSourceTerm, ir::Integer, u, layout, t)
    _shape_at(term, u, layout, t)[ir]
end

# the shapes in the number type of a temperature unknown that carries derivative parts, so
# that the AD of rhs! holds the column of the temperature
function _dual_shapes(term::ParticleSourceTerm, u, layout)
    temperature = unknown_values(u, layout, unknown_id(layout, term.temperature))
    return [_source_shape!(zeros(eltype(temperature), size(term.shape[ir])),
                term.shape_kind, term.grid, ir,
                temperature[ir]) for ir in 1:radial_cell_count(term.grid)]
end

function rhs!(out, term::ParticleSourceTerm, u, layout, row, t)
    shapes = if term.temperature isa Symbol && eltype(u) <: ForwardDiff.Dual
        _dual_shapes(term, u, layout)
    else
        _shape_at(term, u, layout, t)
    end
    magnitude = unknown_values(u, layout, unknown_id(layout, term.source))
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
    D = ForwardDiff.Dual{PlasmaColumnTag, Float64, 1}
    temperature = unknown_values(u, layout, unknown_id(layout, term.temperature))
    magnitude = unknown_values(u, layout, id)
    T_id = unknown_id(layout, term.temperature)
    for ir in 1:radial_cell_count(layout)
        shape = _source_shape!(
            zeros(D, size(term.shape[ir])), term.shape_kind, term.grid, ir,
            D(temperature[ir], ForwardDiff.Partials((1.0,))))
        column = first(unknown_range(layout, T_id, ir))
        for (k, r) in enumerate(unknown_range(layout, row, ir))
            _add_entry!(J, r, column, magnitude[ir] * ForwardDiff.partials(shape[k], 1))
        end
    end
    return J
end

function sparsity!(entries, term::ParticleSourceTerm, layout, row)
    for name in (term.source, _temperature_unknown(term)...),
        ir in 1:radial_cell_count(layout)

        column = first(unknown_range(layout, unknown_id(layout, name), ir))
        append!(entries, ((r, column) for r in unknown_range(layout, row, ir)))
    end
    return entries
end

# tag of the dual numbers of the direct Jacobian of the boundary flux
struct BoundaryFluxTag end

# a 3D array with a unit derivative at one cell, for the derivative of a flux with respect
# to that cell
struct PerturbedArray{T, A <: AbstractArray{T, 3}} <:
       AbstractArray{ForwardDiff.Dual{BoundaryFluxTag, T, 1}, 3}
    values::A
    cell::NTuple{3, Int}
end
Base.size(a::PerturbedArray) = size(a.values)
Base.@propagate_inbounds function Base.getindex(a::PerturbedArray{T}, i::Int, j::Int, ir::Int) where {T}
    return ForwardDiff.Dual{BoundaryFluxTag, T, 1}(a.values[i, j, ir],
        ForwardDiff.Partials((T((i, j, ir) == a.cell),)))
end

# a field with a unit derivative at one radius
struct PerturbedField{T, A <: AbstractVector{T}} <:
       AbstractVector{ForwardDiff.Dual{BoundaryFluxTag, T, 1}}
    values::A
    radius::Int
end
Base.size(a::PerturbedField) = size(a.values)
#! format: off
Base.@propagate_inbounds Base.getindex(a::PerturbedField{T}, ir::Int) where {T} =
    ForwardDiff.Dual{BoundaryFluxTag,T,1}(a.values[ir], ForwardDiff.Partials((T(ir == a.radius),)))
#! format: on

"""
    MomentumBoundaryFluxTerm(kinetic, source; weight=:density, coefficient=1)

`coefficient` times the flux of the kinetic unknown `kinetic` through the upper momentum
boundary of its grid, per volume, added to the row of a radial unknown with one component:
the electrons that leave the grid at `p_max`. `source` holds the coefficients of the flux:
the Fokker-Planck term of `kinetic` ([`FokkerPlanckTerm`](@ref) or
[`prescribed_fokker_planck_term`](@ref)) or its operator. The flux is that of the boundary
condition of the operator at `p_max`, so the term and the operator count the same
particles. With a term that takes the electric field as an unknown, the flux holds the
field of the state.

- `weight = :density`: the particles per volume and second, `(1/V') Σ_j Δξ_j 𝒱' Γᵖ`. With
  `TimeDerivative()` the row unknown is the density of the electrons above the grid, or a
  fluid runaway density fed by the hot-tail grid.
- `weight = :current`: the flux weighted with `e c ξ`, the current that the electrons carry
  when they leave, at the speed of light. With `TimeDerivative()` the row unknown is the
  current density of the electrons above the grid.
- `weight = :energy`: the flux weighted with the kinetic energy `mₑc²(γ − 1)` of the last
  momentum cell [W/m³], the energy that the discrete energy moment of the grid loses with
  the electrons; the energy at the face itself differs by about Δp/(2p). With `coefficient = −1`
  in the row of the temperature it is the energy that leaves the total energy of the
  electrons on the hot-tail grid ([`total_energy_equation`](@ref)).

The entries of the row are assembled directly, from the derivative of the flux of each
pitch cell with respect to the cells at the boundary. With a Fokker-Planck term whose plasma
reads unknowns implicitly ([`has_plasma_columns`](@ref)) the flux follows them as well: its
columns of the temperature, the density and the charge states come from the flux of the
term's coefficients in dual numbers, through the weights of the scheme of the boundary
([`add_plasma_columns!`](@ref)).
"""
struct MomentumBoundaryFluxTerm{S} <: EquationTerm
    kinetic::Symbol
    source::S
    weight::Symbol
    coefficient::Float64
end
_coefficient_sources(term::MomentumBoundaryFluxTerm) = (term.source,)
function MomentumBoundaryFluxTerm(kinetic::Symbol, source; weight::Symbol = :density, coefficient::Real = 1.0)
    source isa Union{AdvectionDiffusion, FokkerPlanckTerm} ||
        throw(ArgumentError("the source of the flux is a Fokker-Planck term or an AdvectionDiffusion operator"))
    weight in (:density, :current, :energy) ||
        throw(ArgumentError("weight must be :density, :current or :energy, not $weight"))
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
    names = field === nothing ? [term.kinetic] : [term.kinetic, field]
    if has_plasma_columns(term)
        append!(names, _plasma_column_unknowns(term.source))
    end
    return unique!(names)
end
function is_linear(term::MomentumBoundaryFluxTerm)
    _electric_field_of(term) === nothing && is_linear(_operator_of(term.source)) &&
        !has_plasma_columns(term)
end

# the flux follows the plasma unknowns of an implicit Fokker-Planck term
function has_plasma_columns(term::MomentumBoundaryFluxTerm)
    term.source isa FokkerPlanckTerm && has_plasma_columns(term.source)
end
# the unknowns of the plasma columns of a Fokker-Planck term: without the charge states it lags
function _plasma_column_unknowns(term::FokkerPlanckTerm)
    filter(name -> !(_lags_ions(term) && name === ion_dependency(term.plasma)),
        collect(Symbol, source_dependencies(term.plasma)))
end

function plasma_column_sparsity!(entries, term::MomentumBoundaryFluxTerm, layout, row)
    source = term.source
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        append!(entries,
            ((r, column)
            for column in plasma_columns(source.plasma, layout, ir;
                with_ions = !_lags_ions(source))))
    end
    return entries
end

function add_plasma_columns!(J, term::MomentumBoundaryFluxTerm, u, layout, row, t)
    source = term.source
    op, with_ions = _dual_coefficients!(source, u, layout, t)
    f = kinetic_array(u, layout, unknown_id(layout, term.kinetic))
    name = _electric_field_of(term)
    field = name === nothing ? nothing : _field_values(name, u, layout)
    rg = layout.radial_grid
    for ir in 1:radial_cell_count(layout)
        m = op.momentum_grids[ir]
        face = momentum_cell_count(m) + 1
        total = sum(_boundary_weight(term, m, rg, j, ir) *
                    _p_flux(op, f, face, j, ir, nothing, field)
        for j in 1:pitch_cell_count(m))
        r = first(unknown_range(layout, row, ir))
        partials = ForwardDiff.partials(total)
        for (k, column) in enumerate(plasma_columns(source.plasma, layout, ir; with_ions))
            _add_entry!(J, r, column, partials[k])
        end
    end
    return J
end
jacobian_assembly(::MomentumBoundaryFluxTerm) = :direct

function validate(term::MomentumBoundaryFluxTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("MomentumBoundaryFluxTerm needs a kinetic unknown"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("MomentumBoundaryFluxTerm adds to a radial unknown with one component"))
    _same_momentum_grids(_operator_of(term.source).grid, kinetic_grid(layout, id)) ||
        throw(DimensionMismatch("the grid of the operator does not match $(term.kinetic)"))
    return nothing
end

# the factor of the flux of the pitch cell j in the row of the radius
Base.@propagate_inbounds function _boundary_weight(
        term::MomentumBoundaryFluxTerm, m::MomentumGrid, rg::RadialGrid, j, ir)
    weight = term.coefficient * m.xi_widths[j] / rg.spatial_jacobian[ir]
    term.weight === :density && return weight
    if term.weight === :energy
        # the energy the discrete moment gives the electrons of the last cell, which the flux
        # removes, so the total energy of the grid closes exactly (not the energy at the face)
        p = m.p_centers[end]
        return weight * Constants.ELECTRON_REST_ENERGY * Constants.ELEMENTARY_CHARGE * p^2 /
               (1 + sqrt(1 + p^2))
    end
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
function _add_boundary_flux!(out, term::MomentumBoundaryFluxTerm,
        op::AdvectionDiffusion, f, u, offsets, layout, row)
    field = offsets === nothing ? nothing : LayoutRadialValues(u, offsets)
    rg = layout.radial_grid
    @inbounds for ir in 1:radial_cell_count(layout)
        m = op.momentum_grids[ir]
        face = momentum_cell_count(m) + 1
        total = zero(eltype(out))
        for j in 1:pitch_cell_count(m)
            total += _boundary_weight(term, m, rg, j, ir) *
                     _p_flux(op, f, face, j, ir, nothing, field)
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
            for other in max(j - 1, 1):min(j + 1, nxi),
                i in max(np - BOUNDARY_FLUX_DEPTH + 1, 1):np

                flux = _p_flux(
                    op, PerturbedArray(f, (i, other, ir)), np + 1, j, ir, nothing, field)
                derivative = ForwardDiff.partials(flux, 1)
                #! format: off
                if derivative != 0
                    _add_entry!(J, r, _kinetic_index(layout, id, ir, i, other), weight * derivative)
                end
                #! format: on
            end
            if field !== nothing
                flux = _p_flux(op, f, np + 1, j, ir, nothing, PerturbedField(field, ir))
                field_column = first(unknown_range(layout, unknown_id(layout, field_name), ir))
                _add_entry!(J, r, field_column, weight * ForwardDiff.partials(flux, 1))
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
        if field !== nothing
            push!(entries, (r, first(unknown_range(layout, unknown_id(layout, field), ir))))
        end
    end
    return entries
end

"""
    MaxwellianSourceTerm(kinetic, grid, temperature; target_density, rate,
                         bulk_momentum=Inf)

Source with the shape of a non-relativistic Maxwellian `exp(-p²/2Θ)`, `Θ = T/mₑc²`, at
`temperature` [eV] (DREAM's `ParticleSourceTerm` uses the relativistic one), whose
magnitude drives the density of the kinetic unknown `kinetic` back to `target_density` (a
number, or one value per radius) at the rate `rate` [1/s]:

    S(p, ξ) = rate (n_target - n(f)) f_M(p) / n_M,

with `f_M` the Maxwellian on `grid` and `n_M` its density, so the source adds
`rate (n_target - n(f))` particles per second and per unit volume and nothing else.
`kinetic` and the row unknown both live on `grid`. It keeps a thermal bulk from draining
into the runaway region, which is what the published Dreicer rates assume; with
`rate = 1/Δt` a step restores the particles the previous step lost.

`n(f)` counts the cells below `bulk_momentum`, and `n_target` is the density of the bulk
there. The bound matters: with the whole grid counted, particles moving from the bulk into
the runaway tail leave the total unchanged, the source sees no deficit, and the bulk still
drains. Set it below the critical momentum and above the thermal momentum.

The term is `:explicit` (evaluated at the previous time level), so the density moment it
reads adds nothing to the Jacobian and the sparsity stays local. This is a device for
comparisons with the Dreicer rates, not the self-consistent particle source: DREAM's
`ParticleSourceTerm` fixes the magnitude from the cold-electron density through
quasi-neutrality ([`ParticleSourceTerm`](@ref)).
"""
struct MaxwellianSourceTerm{T, D} <: EquationTerm
    kinetic::Symbol
    grid::KineticGrid{T}
    shape::Vector{Matrix{T}}   # normalized Maxwellian per radius, density 1
    bulk_cells::Vector{Int}    # per radius: momentum cells counted, 1:bulk_cells
    target_density::D
    rate::Float64
    temperature::Float64       # [eV], of the shape
    bulk_momentum::Float64
end

function MaxwellianSourceTerm(kinetic::Symbol, grid::KineticGrid{T}, temperature::Real;
        target_density, rate::Real, bulk_momentum::Real = Inf) where {T}
    temperature > 0 || throw(ArgumentError("the temperature must be positive"))
    rate > 0 || throw(ArgumentError("the rate must be positive"))
    target_density isa Union{Real, AbstractVector} ||
        throw(ArgumentError("target_density must be a number or one value per radius"))
    bulk_momentum > 0 || throw(ArgumentError("bulk_momentum must be positive"))
    theta = T(temperature) / T(Constants.ELECTRON_REST_ENERGY)
    bulk_cells = Int[]
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        push!(bulk_cells, count(<=(bulk_momentum), m.p_centers))
    end
    if !all(>(0), bulk_cells)
        throw(ArgumentError("bulk_momentum is below the first momentum cell"))
    end
    shape = Matrix{T}[]
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        values = [exp(-p^2 / (2theta)) for p in m.p_centers, _ in 1:pitch_cell_count(m)]
        jacobian = phase_space_jacobian(grid, ir)
        #! format: off
        norm = sum(jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] * values[i, j]
                   for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)) /
               grid.radial_grid.spatial_jacobian[ir]
        #! format: on
        push!(shape, values ./ norm)
    end
    return MaxwellianSourceTerm{T, typeof(target_density)}(
        kinetic, grid, shape, bulk_cells, target_density,
        Float64(rate), Float64(temperature),
        Float64(bulk_momentum))
end

dependencies(term::MaxwellianSourceTerm) = [term.kinetic]
treatment(::MaxwellianSourceTerm) = :explicit

function validate(term::MaxwellianSourceTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    layout.kinds[id] === :kinetic && layout.kinds[row] === :kinetic ||
        throw(ArgumentError("MaxwellianSourceTerm reads and adds to kinetic unknowns"))
    # the density is counted on the grid of `kinetic` and the source added on the row's
    for k in (id, row)
        _same_momentum_grids(term.grid, kinetic_grid(layout, k)) ||
            throw(DimensionMismatch("the term's grid does not match $(layout.unknowns[k].name)"))
    end
    if term.target_density isa AbstractVector &&
       length(term.target_density) != radial_cell_count(layout)
        throw(DimensionMismatch("target_density needs one value per radius"))
    end
    return nothing
end

function _target(term::MaxwellianSourceTerm, ir)
    term.target_density isa AbstractVector ? term.target_density[ir] : term.target_density
end

function rhs!(out, term::MaxwellianSourceTerm, u, layout, row, t)
    id = unknown_id(layout, term.kinetic)
    grid = kinetic_grid(layout, id)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        f = kinetic_view(u, layout, id, ir)
        n = zero(eltype(out))
        @inbounds for j in 1:pitch_cell_count(m), i in 1:term.bulk_cells[ir]

            n += jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] * f[i, j]
        end
        n /= grid.radial_grid.spatial_jacobian[ir]
        magnitude = term.rate * (_target(term, ir) - n)
        shape = term.shape[ir]
        added = kinetic_view(out, layout, row, ir)
        @inbounds for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)

            added[i, j] += magnitude * shape[i, j]
        end
    end
    return out
end

# explicit terms carry no Jacobian block
sparsity!(entries, ::MaxwellianSourceTerm, layout, row) = entries
