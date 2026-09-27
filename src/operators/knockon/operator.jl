# Knock-on (Møller Boltzmann) gain operator: the energy kernel `rates[i, k]` (S_ik) and the
# pitch kernel `δ[j, l](ξ*(p_i, p1_k))` applied to a primary distribution.
#
# Two equivalent evaluations:
# - `:loop`: DREAM `MollerBoltzmannOperator::SetSourceVector` and
#   `MollerDeltaAngleKernel::AccumulatePitch`, interpolating the pitch kernel per momentum
#   pair; cost O(nnz(S) · nξP · nξK) per radius.
# - `:blas`: contract each tabulated slice with the primary weights first,
#   `Y_m = D_m W[:, k ∈ K_m]` (one matrix product per slice m, restricted to the primary
#   momenta K_m whose pairs use slice m), then gather
#   `C[i, j] = Σ_k S_ik (w0 Y_{m0}[j, k] + w1 Y_{m0+1}[j, k])` for the two slices of a
#   single ξ* (`:midpoint`, `:centre`), or the sum over every slice of the pair's ξ* range
#   with `:interval` (`KnockOnKernels`).
#   Interpolation is linear, so both give the same result up to summation order.

using LinearAlgebra: mul!
using LinearOperators: LinearOperator

"""
    KnockOnKernels{T}

Energy and pitch kernels for one cutoff momentum, together with the production each
momentum pair `(i, k)` sends to each slice of the ξ* table. The entries of pair `(i, k)`
are `pair_start[i, k]:pair_stop[i, k]` (empty when the pair produces no knock-ons); entry
`t` puts the weight `entry_weight[t]` on table slice `entry_slice[t]`, and the weights of
a pair sum to `energy.rates[i, k]`, so the production is exactly the energy kernel's.

With `xi_star_mode = :interval` the weights come from the Møller flux
integrated over the receiving interval against the hat functions `φ_m` of the table,

    entry_weight[t] = β₁/Δp_i ∫ β(p) Σ(p, p1_k) φ_{m_t}(ξ*(p, p1_k)) dp,

which is the pitch kernel integrated over ξ*(p) instead of evaluated at one ξ*. With
`:midpoint` or `:centre` a pair has the two entries of the linear interpolation at its
single ξ* (one when the interpolation weight is 0 or 1).

For the `:blas` evaluation, the primary momenta whose pairs use slice m are
`slice_primaries[m]`, their products occupy columns
`slice_offsets[m] + 1 : slice_offsets[m] + length(slice_primaries[m])` of the operator's
`slice_products`, and `entry_column[t]` is the column of entry `t`.
"""
struct KnockOnKernels{T<:AbstractFloat}
    energy::MollerEnergyKernel{T}
    pitch::MollerPitchKernel{T}
    pair_start::Matrix{Int}     # (knock-on p cells, primary p cells); the range is empty if no knock-ons
    pair_stop::Matrix{Int}
    entry_slice::Vector{Int}
    entry_weight::Vector{T}
    entry_column::Vector{Int}
    slice_primaries::Vector{Vector{Int}}
    slice_offsets::Vector{Int}
    column_count::Int
end

# Weights of one pair over the table slices: the Møller flux of the receiving interval
# [p_lower, p_upper] distributed over the hat functions of the table. The interval is split
# at the momenta where ξ*(p) crosses a table value, each piece integrated by composite
# Gauss with `points` nodes, and the number of sub-intervals doubles until the weights
# have converged. The hat functions sum to one, so the total is a quadrature of the
# analytic flux `rate`, and agreement with it to 1e-12 ends the refinement early. That
# reference is a difference of two values of `S`, which loses its digits when the interval
# is narrow (a cutoff a few ulp below a face leaves a sliver of a cell), so the weights are
# otherwise converged against their previous refinement. Returns the slice range and the
# sum of the weights; throws if the weights do not converge or the ξ* range of the pair
# leaves the table.
function _pair_slice_weights!(acc::AbstractVector{T}, previous::AbstractVector{T}, xi_star_values,
                              p_lower, p_upper, p1, rate, nodes, weights) where {T}
    spacing = length(xi_star_values) > 1 ? xi_star_values[2] - xi_star_values[1] : zero(T)
    slice_count = length(xi_star_values)
    lowest = first(xi_star_values)
    position(x) = spacing > 0 ? clamp((x - lowest) / spacing, 0, slice_count - 1) : zero(T)
    xi_lower, xi_upper = xi_star(p_lower, p1), xi_star(p_upper, p1)
    # outside the table the clamped hat functions would pile the pair onto an edge slice
    margin = 1e-10 * max(spacing, eps(T))
    (lowest - margin <= xi_lower && xi_upper <= last(xi_star_values) + margin) ||
        throw(ErrorException("the ξ* range [$xi_lower, $xi_upper] of a knock-on pair leaves the table " *
                             "[$lowest, $(last(xi_star_values))]"))
    # a quadrature point at position s contributes to the slices floor(s) + 1 and + 2
    m_lower = floor(Int, position(xi_lower)) + 1
    m_upper = clamp(floor(Int, position(xi_upper)) + 2, m_lower, slice_count)
    # momenta where ξ*(p) meets a table value, as piece boundaries
    cuts = T[p_lower]
    for m in m_lower+1:m_upper-1
        p = momentum_at_xi_star(xi_star_values[m], p1)
        p_lower < p < p_upper && push!(cuts, p)
    end
    push!(cuts, p_upper)
    beta1 = p1 / sqrt(1 + p1^2)
    total = zero(T)
    for sub in (1, 2, 4, 8, 16, 32, 64, 128)
        sub > 1 && copyto!(view(previous, m_lower:m_upper), view(acc, m_lower:m_upper))
        fill!(view(acc, m_lower:m_upper), zero(T))
        total = zero(T)
        for q in 1:length(cuts)-1
            width = (cuts[q+1] - cuts[q]) / sub
            width > 0 || continue
            for s in 0:sub-1, (x, w) in zip(nodes, weights)
                p = cuts[q] + width * (s + (x + 1) / 2)
                value = w * width / 2 * beta1 * (p / sqrt(1 + p^2)) * moller_cross_section(p, p1)
                position_here = position(xi_star(p, p1))
                m = min(floor(Int, position_here), slice_count - 2) + 1
                upper = position_here - (m - 1)
                acc[m] += value * (1 - upper)
                acc[m+1] += value * upper
                total += value
            end
        end
        abs(total - rate) <= 1e-12 * abs(rate) && return m_lower, m_upper, total
        if sub > 1
            change = maximum(m -> abs(acc[m] - previous[m]), m_lower:m_upper)
            change <= 1e-12 * total && return m_lower, m_upper, total
        end
    end
    throw(ErrorException("the ξ* weights of a knock-on pair do not converge; raise xi_star_points"))
end

function KnockOnKernels(knockon::MomentumGrid{T}, primary::MomentumGrid{T}, p_cutoff::Real;
                        xi_star_points::Integer=4, pitch_options...) where {T}
    xi_star_points >= 1 || throw(ArgumentError("xi_star_points must be at least 1"))
    energy = MollerEnergyKernel(knockon, primary, p_cutoff)
    pitch = MollerPitchKernel(knockon, primary, p_cutoff; pitch_options...)
    np_knockon, np_primary = momentum_cell_count(knockon), momentum_cell_count(primary)
    slice_count = length(pitch.xi_star_values)
    integrated = pitch.xi_star_mode === :interval
    nodes, weights = gauss(T, xi_star_points)
    accumulator, previous = zeros(T, slice_count + 1), zeros(T, slice_count + 1)

    # pairs without knock-ons keep the empty range 1:0
    pair_start, pair_stop = ones(Int, np_knockon, np_primary), zeros(Int, np_knockon, np_primary)
    entry_slice, entry_weight = Int[], T[]
    used = falses(slice_count, np_primary)
    for k in 1:np_primary, i in 1:np_knockon
        rate = energy.rates[i, k]
        rate == 0 && continue
        first_entry = length(entry_slice) + 1
        if integrated
            p1 = primary.p_centers[k]
            p_lower = max(knockon.p_faces[i], T(p_cutoff))
            p_upper = min(knockon.p_faces[i+1], max_knockon_momentum(p1))
            # the rate of the energy kernel is Δp_i times the flux of the interval
            flux = rate * knockon.p_widths[i]
            m_lower, m_upper, total = _pair_slice_weights!(accumulator, previous, pitch.xi_star_values,
                                                           p_lower, p_upper, p1, flux, nodes, weights)
            total > 0 || throw(ErrorException("no knock-on production in the ξ* range of pair ($i, $k)"))
            # rescale to the analytic flux, so that the production stays exactly the
            # energy kernel's for any quadrature (and carries the same round-off in a sliver
            # of a cell)
            scale = rate / total
            for m in m_lower:m_upper
                accumulator[m] == 0 && continue
                push!(entry_slice, m)
                push!(entry_weight, accumulator[m] * scale)
            end
        else
            m = pitch.lower_slice[i, k]
            upper = pitch.upper_weight[i, k]
            if upper != 1
                push!(entry_slice, m)
                push!(entry_weight, rate * (1 - upper))
            end
            if upper != 0
                push!(entry_slice, m + 1)
                push!(entry_weight, rate * upper)
            end
        end
        pair_start[i, k], pair_stop[i, k] = first_entry, length(entry_slice)
        for t in first_entry:length(entry_slice)
            used[entry_slice[t], k] = true
        end
    end

    slice_primaries = [findall(view(used, m, :)) for m in 1:slice_count]
    slice_offsets = cumsum([0; length.(slice_primaries)[1:end-1]])
    column = zeros(Int, slice_count, np_primary)
    for m in 1:slice_count, (c, k) in enumerate(slice_primaries[m])
        column[m, k] = slice_offsets[m] + c
    end
    entry_column = zeros(Int, length(entry_slice))
    for k in 1:np_primary, i in 1:np_knockon, t in pair_start[i, k]:pair_stop[i, k]
        entry_column[t] = column[entry_slice[t], k]
    end
    return KnockOnKernels{T}(energy, pitch, pair_start, pair_stop, entry_slice, entry_weight,
                             entry_column, slice_primaries, slice_offsets, sum(length, slice_primaries))
end

"""
    KnockOnOperator{T}

Knock-on source on the grid `knockon` produced by primaries on the grid `primary`
(the same grid for in-grid knock-on) colliding with stationary targets of density
`n_tot` (the total electron density):

    C[i, j, r] = n_tot[r] / 𝒱'_K[i, j, r] Σ_k S[i, k] Σ_l W[k, l, r] δ[j, l](ξ*(p_i, p1_k)),
    W[k, l, r] = Δp_k Δξ_l 𝒱'_P[k, l, r] f[k, l, r].

`C` has the units of `∂f/∂t`. Its density moment is the total production rate:
`(1/V') Σ_ij 𝒱'_ij Δp_i Δξ_j C_ij = n_tot Σ_kl (𝒱'_kl Δp_k Δξ_l f_kl / V') total_rates[k]`,
with `total_rates` from the energy kernel. Only the ratio `𝒱'_P / 𝒱'_K` at equal radius
enters, so any constant normalization of `𝒱'` cancels (DREAM's cylindrical `Vprime` is
`4π²` times Runes2P's).

The cutoff momentum `p_cutoff[r]` may differ between radii and change in time
([`set_cutoff!`](@ref)); each radius uses the kernels of its cutoff, and radii with
equal cutoffs share them. Both grids must have the same radial grid. The operator is
linear in `f` and in `n_tot`; DREAM evaluates it explicitly, from `f` at the previous
time level, with one cutoff for all radii and times.
"""
mutable struct KnockOnOperator{T<:AbstractFloat}
    const knockon::KineticGrid{T}
    const primary::KineticGrid{T}
    const method::Symbol
    const pitch_options::NamedTuple
    const p_cutoff::Vector{T}
    const kernels::Vector{KnockOnKernels{T}}      # per radius
    const cache::Dict{Tuple{UInt,UInt,T},KnockOnKernels{T}}  # by (grids, cutoff), current ones only
    const primary_weights::Matrix{T}   # scratch W (primary ξ cells, primary p cells)
    const knockon_row::Vector{T}       # scratch C[i, :] (knock-on ξ cells)
    const gathered_weights::Matrix{T}  # primary weights gathered per slice (primary ξ cells, primary p cells)
    slice_products::Matrix{T}          # Y (knock-on ξ cells, largest number of (slice, k) combinations)
end

"""
    KnockOnOperator(knockon, primary, p_cutoff; method=:blas, table_size=80,
                    dream_compatible=false, primary_pitch, primary_pitch_points=4)

Build the energy and pitch kernels for knock-on grid `knockon` and primary grid
`primary` (both `KineticGrid`) with cutoff momentum `p_cutoff` (a number, or one value per
radius). `method` selects the evaluation (`:blas` or `:loop`, see the source file); the
other keyword arguments are passed to [`MollerPitchKernel`](@ref).
`dream_compatible = true, table_size = 80` (primary at the cell centre) with one cutoff
reproduces DREAM's `MollerBoltzmannOperator` in cylindrical geometry.
"""
function KnockOnOperator(knockon::KineticGrid{T}, primary::KineticGrid{T},
                         p_cutoff::Union{Real,AbstractVector}; method::Symbol=:blas,
                         pitch_options...) where {T}
    knockon.radial_grid.r_faces == primary.radial_grid.r_faces ||
        throw(ArgumentError("knock-on and primary grids must share the radial grid"))
    method in (:blas, :loop) || throw(ArgumentError("unknown method $method"))
    nr = radial_cell_count(knockon)
    radii = 1:nr
    max_nxi_knockon = maximum(ir -> pitch_cell_count(knockon, ir), radii)
    max_nxi_primary = maximum(ir -> pitch_cell_count(primary, ir), radii)
    max_np_primary = maximum(ir -> momentum_cell_count(primary, ir), radii)
    op = KnockOnOperator{T}(knockon, primary, method, NamedTuple(pitch_options), zeros(T, nr),
                            Vector{KnockOnKernels{T}}(undef, nr),
                            Dict{Tuple{UInt,UInt,T},KnockOnKernels{T}}(),
                            zeros(T, max_nxi_primary, max_np_primary),
                            zeros(T, max_nxi_knockon),
                            zeros(T, max_nxi_primary, method == :blas ? max_np_primary : 0),
                            zeros(T, max_nxi_knockon, 0))
    set_cutoff!(op, p_cutoff)
    return op
end

"""
    resolved_pitch_options(options::NamedTuple)

The kernel options of a [`KnockOnOperator`](@ref) with the defaults filled in: the
keywords of [`MollerPitchKernel`](@ref) and `xi_star_points` of [`KnockOnKernels`](@ref).
The operator keeps the keywords it was given, and the record of a run needs the values
used ([`describe`](@ref)).
"""
function resolved_pitch_options(options::NamedTuple)
    dream_compatible = get(options, :dream_compatible, false)
    return (table_size=get(options, :table_size, 80), dream_compatible=dream_compatible,
            xi_star_mode=get(options, :xi_star_mode, dream_compatible ? :centre : :interval),
            primary_pitch=get(options, :primary_pitch, dream_compatible ? :centre : :split),
            primary_pitch_points=get(options, :primary_pitch_points, 4),
            xi_star_points=get(options, :xi_star_points, 4))
end

# Kernels depend on the two momentum grids of a radius and on its cutoff.
_kernel_key(op::KnockOnOperator{T}, ir, cutoff) where {T} =
    (objectid(momentum_grid(op.knockon, ir)), objectid(momentum_grid(op.primary, ir)), T(cutoff))

"""
    set_cutoff!(op::KnockOnOperator, p_cutoff)

Set the cutoff momentum (a number for all radii, or one value per radius) and rebuild
the kernels of the radii whose cutoff changed. Kernels are cached by the momentum grids
of the radius and the cutoff value, so radii with the same grids and cutoff share one
set, and a combination already in use costs nothing. Returns the number of kernel sets
built. When a cutoff is invalid or a kernel cannot be built, the operator is unchanged.
A rebuild costs a few operator applications; snapping the cutoff
to momentum cell faces ([`choose_cutoff`](@ref)) keeps rebuilds rare when the cutoff
follows the plasma.

A `knockon_linear_operator` of `op` sees the new kernels.
"""
function set_cutoff!(op::KnockOnOperator{T}, p_cutoff::Union{Real,AbstractVector}) where {T}
    nr = radial_cell_count(op.knockon)
    p_cutoff isa AbstractVector && length(p_cutoff) != nr &&
        throw(DimensionMismatch("p_cutoff must be a number or have one entry per radius"))
    cutoffs = T[p_cutoff isa AbstractVector ? p_cutoff[ir] : p_cutoff for ir in 1:nr]
    all(>(0), cutoffs) || throw(ArgumentError("p_cutoff must be positive"))
    # the new kernels are built before the operator changes, so a failure (a cutoff or a
    # grid the kernels refuse) leaves it as it was
    built = Dict{keytype(op.cache),KnockOnKernels{T}}()
    kernels = map(1:nr) do ir
        key = _kernel_key(op, ir, cutoffs[ir])
        get(op.cache, key) do
            get!(built, key) do
                KnockOnKernels(momentum_grid(op.knockon, ir), momentum_grid(op.primary, ir),
                               cutoffs[ir]; op.pitch_options...)
            end
        end
    end
    merge!(op.cache, built)
    copyto!(op.p_cutoff, cutoffs)
    copyto!(op.kernels, kernels)
    # keep only the kernels in use
    in_use = Set(_kernel_key(op, ir, op.p_cutoff[ir]) for ir in 1:nr)
    for key in collect(keys(op.cache))
        key in in_use || delete!(op.cache, key)
    end
    if op.method === :blas
        column_count = maximum(k -> k.column_count, op.kernels)
        column_count > size(op.slice_products, 2) &&
            (op.slice_products = zeros(T, size(op.slice_products, 1), column_count))
    end
    return length(built)
end

"""
    energy_kernel(op::KnockOnOperator, ir=1)
    pitch_kernel(op::KnockOnOperator, ir=1)

Energy and pitch kernels used at radius index `ir`.
"""
energy_kernel(op::KnockOnOperator, ir::Integer=1) = op.kernels[ir].energy
pitch_kernel(op::KnockOnOperator, ir::Integer=1) = op.kernels[ir].pitch

"""
    apply!(df, op::KnockOnOperator, f, total_electron_density; scale=1)

Add `scale` times the knock-on source produced by the primary distribution `f[k, l, r]`
to `df[i, j, r]` on the knock-on grid; `total_electron_density[r]` is the target density.
`f` and `df` are 3D arrays, so both grids must have a shared momentum grid; for per-radius
grids use [`apply_at_radius!`](@ref). Allocation free; not safe to call concurrently on
the same operator (shared scratch).
"""
function apply!(df::AbstractArray{<:Any,3}, op::KnockOnOperator, f::AbstractArray{<:Any,3},
                total_electron_density::AbstractVector; scale=true)
    knockon, primary = op.knockon, op.primary
    size(df) == size(knockon) || throw(DimensionMismatch("df must have the knock-on grid size"))
    size(f) == size(primary) || throw(DimensionMismatch("f must have the primary grid size"))
    length(total_electron_density) == radial_cell_count(knockon) ||
        throw(DimensionMismatch("total_electron_density must have one entry per radius"))
    for ir in 1:radial_cell_count(knockon)
        apply_at_radius!(view(df, :, :, ir), op, view(f, :, :, ir), total_electron_density[ir], ir;
                         scale)
    end
    return df
end

"""
    apply_at_radius!(df, op::KnockOnOperator, f, total_electron_density, ir; scale=1)

Knock-on source at radius index `ir` alone: `f` is the primary distribution at that radius
(`(p cells, ξ cells)` of the primary momentum grid there), `df` the rate of change of the
knock-on distribution there, `total_electron_density` the target density there. Works with
a different momentum grid at each radius.
"""
function apply_at_radius!(df::AbstractMatrix, op::KnockOnOperator, f::AbstractMatrix,
                          total_electron_density::Real, ir::Integer; scale=true)
    size(df) == (momentum_cell_count(op.knockon, ir), pitch_cell_count(op.knockon, ir)) ||
        throw(DimensionMismatch("df must have the knock-on momentum grid size at radius $ir"))
    size(f) == (momentum_cell_count(op.primary, ir), pitch_cell_count(op.primary, ir)) ||
        throw(DimensionMismatch("f must have the primary momentum grid size at radius $ir"))
    primary_weights!(op.primary_weights, op.primary, f, ir)
    factor = total_electron_density * scale
    if op.method === :blas
        apply_radius_blas!(df, op, op.kernels[ir], factor, ir)
    else
        apply_radius_loop!(df, op, op.kernels[ir], factor, ir)
    end
    return df
end

function primary_weights!(weights, primary::KineticGrid, f::AbstractMatrix, ir)
    m = momentum_grid(primary, ir)
    jacobian = phase_space_jacobian(primary, ir)
    @inbounds for k in 1:momentum_cell_count(m), l in 1:pitch_cell_count(m)
        weights[l, k] = m.p_widths[k] * m.xi_widths[l] * jacobian[k, l] * f[k, l]
    end
    return weights
end

function apply_radius_loop!(df::AbstractMatrix, op::KnockOnOperator, kernels::KnockOnKernels, factor, ir)
    W, row = op.primary_weights, op.knockon_row
    jacobian = phase_space_jacobian(op.knockon, ir)
    nxi_knockon, nxi_primary = pitch_cell_count(op.knockon, ir), pitch_cell_count(op.primary, ir)
    table = kernels.pitch.table
    @inbounds for i in 1:momentum_cell_count(op.knockon, ir)
        fill!(row, 0)
        for k in 1:momentum_cell_count(op.primary, ir)
            for t in kernels.pair_start[i, k]:kernels.pair_stop[i, k]
                m = kernels.entry_slice[t]
                s = kernels.entry_weight[t]
                for l in 1:nxi_primary
                    a = s * W[l, k]
                    a == 0 && continue
                    for j in 1:nxi_knockon
                        row[j] += a * table[j, l, m]
                    end
                end
            end
        end
        for j in 1:nxi_knockon
            df[i, j] += factor * row[j] / jacobian[i, j]
        end
    end
    return df
end

function apply_radius_blas!(df::AbstractMatrix, op::KnockOnOperator, kernels::KnockOnKernels, factor, ir)
    pitch, W, row = kernels.pitch, op.primary_weights, op.knockon_row
    Y, gathered = op.slice_products, op.gathered_weights
    jacobian = phase_space_jacobian(op.knockon, ir)
    nxi_knockon, nxi_primary = pitch_cell_count(op.knockon, ir), pitch_cell_count(op.primary, ir)
    # Y_m = D_m W[:, K_m] for every slice
    for m in eachindex(kernels.slice_primaries)
        primaries = kernels.slice_primaries[m]
        n = length(primaries)
        n == 0 && continue
        @inbounds for (c, k) in enumerate(primaries), l in 1:nxi_primary
            gathered[l, c] = W[l, k]
        end
        offset = kernels.slice_offsets[m]
        mul!(view(Y, 1:nxi_knockon, offset+1:offset+n), view(pitch.table, :, :, m),
             view(gathered, 1:nxi_primary, 1:n))
    end
    # gather the slice products with the weights of each pair
    @inbounds for i in 1:momentum_cell_count(op.knockon, ir)
        fill!(row, 0)
        for k in 1:momentum_cell_count(op.primary, ir)
            for t in kernels.pair_start[i, k]:kernels.pair_stop[i, k]
                c = kernels.entry_column[t]
                a = kernels.entry_weight[t]
                for j in 1:nxi_knockon
                    row[j] += a * Y[j, c]
                end
            end
        end
        for j in 1:nxi_knockon
            df[i, j] += factor * row[j] / jacobian[i, j]
        end
    end
    return df
end

"""
    knockon_linear_operator(op::KnockOnOperator, total_electron_density; scale=1)

Matrix-free linear operator `scale ∂C/∂f` of size `(cell_count(knockon), cell_count(primary))`,
acting on `vec(f)` of the primary grid and returning `vec` of the knock-on grid: the
implicit (Jacobian) form of the knock-on term, for use in Krylov solvers and in sums
with other `LinearOperator`s or matrices. `C` is linear in `f`, so the Jacobian-vector
product is [`apply!`](@ref) itself. `total_electron_density` is referenced, not copied:
updating it in place (e.g. between Newton iterations) changes the operator. Both grids
need a shared momentum grid, as `apply!` does; with a momentum grid per radius the
products go through [`apply_at_radius!`](@ref), as in [`KnockOnTerm`](@ref).

The derivative with respect to the total electron density is `C(f)/n_tot` at each radius
(DREAM's only Jacobian block for this term, which it treats explicitly in `f`). Treating
`f` implicitly is a Runes2P departure; the explicit form stays available through `apply!`.

The product uses the operator's scratch arrays and is not safe to call concurrently.
No transpose product is provided.
"""
function knockon_linear_operator(op::KnockOnOperator{T}, total_electron_density::AbstractVector;
                                 scale=one(T)) where {T}
    knockon_size, primary_size = size(op.knockon), size(op.primary)
    length(total_electron_density) == radial_cell_count(op.knockon) ||
        throw(DimensionMismatch("total_electron_density must have one entry per radius"))
    function product!(result, v, a, b)
        if iszero(b)
            fill!(result, zero(eltype(result)))
        elseif !isone(b)
            result .*= b
        end
        apply!(reshape(result, knockon_size), op, reshape(v, primary_size), total_electron_density;
               scale=a * scale)
        return result
    end
    return LinearOperator{T,Vector{T}}(prod(knockon_size), prod(primary_size), false, false,
                                       product!, nothing, nothing)
end

"""
    pitch_kernel_matrix!(D, kernels::KnockOnKernels, i, k)

Effective pitch kernel of the momentum pair `(i, k)`, see the
[`MollerPitchKernel`](@ref) method.
"""
function pitch_kernel_matrix!(D::AbstractMatrix, kernels::KnockOnKernels, i, k)
    fill!(D, zero(eltype(D)))
    rate = kernels.energy.rates[i, k]
    rate == 0 && return D
    table = kernels.pitch.table
    for t in kernels.pair_start[i, k]:kernels.pair_stop[i, k]
        share = kernels.entry_weight[t] / rate
        @views D .+= share .* table[:, :, kernels.entry_slice[t]]
    end
    return D
end

"""
    knockon_kernels(op::KnockOnOperator, ir=1)

[`KnockOnKernels`](@ref) used at radius index `ir`: the energy and pitch kernels together
with the table slices and weights of every momentum pair.
"""
knockon_kernels(op::KnockOnOperator, ir::Integer=1) = op.kernels[ir]
