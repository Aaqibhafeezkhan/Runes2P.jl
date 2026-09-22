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
#   `C[i, j] = Σ_k S_ik (w0 Y_{m0}[j, k] + w1 Y_{m0+1}[j, k])`.
#   Interpolation is linear, so both give the same result up to summation order.

using LinearAlgebra: mul!
using LinearOperators: LinearOperator

"""
    KnockOnKernels{T}

Energy and pitch kernels for one cutoff momentum, with the column layout of the `:blas`
evaluation: the primary momenta whose pairs use slice m are `slice_primaries[m]`, and their
products occupy columns `slice_offsets[m] + 1 : slice_offsets[m] + length(slice_primaries[m])`
of the operator's `slice_products`.
"""
struct KnockOnKernels{T<:AbstractFloat}
    energy::MollerEnergyKernel{T}
    pitch::MollerPitchKernel{T}
    slice_primaries::Vector{Vector{Int}}
    slice_offsets::Vector{Int}
    lower_columns::Matrix{Int}  # slice_products column of (lower slice, k) for pair (i, k); 0 if no knock-ons
    upper_columns::Matrix{Int}  # column of (lower slice + 1, k); 0 if its weight is zero
    column_count::Int
end

function KnockOnKernels(knockon::MomentumGrid{T}, primary::MomentumGrid{T}, p_cutoff::Real;
                        pitch_options...) where {T}
    energy = MollerEnergyKernel(knockon, primary, p_cutoff)
    pitch = MollerPitchKernel(knockon, primary, p_cutoff; pitch_options...)
    np_knockon, np_primary = momentum_cell_count(knockon), momentum_cell_count(primary)
    slice_count = length(pitch.xi_star_values)
    # (slice, k) combinations that any pair with knock-ons needs
    used = falses(slice_count, np_primary)
    for k in 1:np_primary, i in 1:np_knockon
        energy.rates[i, k] == 0 && continue
        used[pitch.lower_slice[i, k], k] = true
        pitch.upper_weight[i, k] != 0 && (used[pitch.lower_slice[i, k]+1, k] = true)
    end
    slice_primaries = [findall(view(used, m, :)) for m in 1:slice_count]
    slice_offsets = cumsum([0; length.(slice_primaries)[1:end-1]])
    column = zeros(Int, slice_count, np_primary)
    for m in 1:slice_count, (c, k) in enumerate(slice_primaries[m])
        column[m, k] = slice_offsets[m] + c
    end
    lower_columns = zeros(Int, np_knockon, np_primary)
    upper_columns = zeros(Int, np_knockon, np_primary)
    for k in 1:np_primary, i in 1:np_knockon
        energy.rates[i, k] == 0 && continue
        lower_columns[i, k] = column[pitch.lower_slice[i, k], k]
        pitch.upper_weight[i, k] != 0 && (upper_columns[i, k] = column[pitch.lower_slice[i, k]+1, k])
    end
    return KnockOnKernels{T}(energy, pitch, slice_primaries, slice_offsets, lower_columns,
                             upper_columns, sum(length, slice_primaries))
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
    const cache::Dict{T,KnockOnKernels{T}}        # by cutoff value, current cutoffs only
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
    op = KnockOnOperator{T}(knockon, primary, method, NamedTuple(pitch_options), zeros(T, nr),
                            Vector{KnockOnKernels{T}}(undef, nr),
                            Dict{T,KnockOnKernels{T}}(),
                            zeros(T, pitch_cell_count(primary), momentum_cell_count(primary)),
                            zeros(T, pitch_cell_count(knockon)),
                            zeros(T, pitch_cell_count(primary),
                                  method == :blas ? momentum_cell_count(primary) : 0),
                            zeros(T, pitch_cell_count(knockon), 0))
    set_cutoff!(op, p_cutoff)
    return op
end

"""
    set_cutoff!(op::KnockOnOperator, p_cutoff)

Set the cutoff momentum (a number for all radii, or one value per radius) and rebuild
the kernels of the radii whose cutoff changed. Kernels are cached by cutoff value, so
radii with equal cutoffs share one set, and a value already in use costs nothing.
Returns the number of kernel sets built. A rebuild costs a few operator applications;
snapping the cutoff to momentum cell faces ([`choose_cutoff`](@ref)) keeps rebuilds rare
when the cutoff follows the plasma.

A `knockon_linear_operator` of `op` sees the new kernels.
"""
function set_cutoff!(op::KnockOnOperator{T}, p_cutoff::Union{Real,AbstractVector}) where {T}
    nr = radial_cell_count(op.knockon)
    p_cutoff isa AbstractVector && length(p_cutoff) != nr &&
        throw(DimensionMismatch("p_cutoff must be a number or have one entry per radius"))
    built = 0
    for ir in 1:nr
        cutoff = T(p_cutoff isa AbstractVector ? p_cutoff[ir] : p_cutoff)
        cutoff > 0 || throw(ArgumentError("p_cutoff must be positive"))
        kernels = get(op.cache, cutoff, nothing)
        if kernels === nothing
            kernels = KnockOnKernels(op.knockon.momentum_grid, op.primary.momentum_grid, cutoff;
                                     op.pitch_options...)
            op.cache[cutoff] = kernels
            built += 1
        end
        op.p_cutoff[ir] = cutoff
        op.kernels[ir] = kernels
    end
    # keep only the kernels in use
    for key in collect(keys(op.cache))
        key in op.p_cutoff || delete!(op.cache, key)
    end
    if op.method === :blas
        column_count = maximum(k -> k.column_count, op.kernels)
        column_count > size(op.slice_products, 2) &&
            (op.slice_products = zeros(T, pitch_cell_count(op.knockon), column_count))
    end
    return built
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
Allocation free; not safe to call concurrently on the same operator (shared scratch).
"""
function apply!(df::AbstractArray{<:Any,3}, op::KnockOnOperator, f::AbstractArray{<:Any,3},
                total_electron_density::AbstractVector; scale=true)
    knockon, primary = op.knockon, op.primary
    size(df) == size(knockon) || throw(DimensionMismatch("df must have the knock-on grid size"))
    size(f) == size(primary) || throw(DimensionMismatch("f must have the primary grid size"))
    length(total_electron_density) == radial_cell_count(knockon) ||
        throw(DimensionMismatch("total_electron_density must have one entry per radius"))
    for ir in 1:radial_cell_count(knockon)
        primary_weights!(op.primary_weights, primary, f, ir)
        if op.method === :blas
            apply_radius_blas!(df, op, op.kernels[ir], total_electron_density[ir] * scale, ir)
        else
            apply_radius_loop!(df, op, op.kernels[ir], total_electron_density[ir] * scale, ir)
        end
    end
    return df
end

function primary_weights!(weights, primary::KineticGrid, f, ir)
    m = primary.momentum_grid
    @inbounds for k in 1:momentum_cell_count(primary), l in 1:pitch_cell_count(primary)
        weights[l, k] = m.p_widths[k] * m.xi_widths[l] * primary.phase_space_jacobian[k, l, ir] * f[k, l, ir]
    end
    return weights
end

function apply_radius_loop!(df, op::KnockOnOperator, kernels::KnockOnKernels, factor, ir)
    knockon = op.knockon
    rates, pitch, W, row = kernels.energy.rates, kernels.pitch, op.primary_weights, op.knockon_row
    nxi_knockon, nxi_primary = pitch_cell_count(knockon), pitch_cell_count(op.primary)
    table = pitch.table
    @inbounds for i in 1:momentum_cell_count(knockon)
        fill!(row, 0)
        for k in 1:momentum_cell_count(op.primary)
            s = rates[i, k]
            s == 0 && continue
            m = pitch.lower_slice[i, k]
            w1 = pitch.upper_weight[i, k]
            w0 = 1 - w1
            for l in 1:nxi_primary
                a = s * W[l, k]
                a == 0 && continue
                for j in 1:nxi_knockon
                    row[j] += a * (w0 * table[j, l, m] + w1 * table[j, l, m+1])
                end
            end
        end
        for j in 1:nxi_knockon
            df[i, j, ir] += factor * row[j] / knockon.phase_space_jacobian[i, j, ir]
        end
    end
    return df
end

function apply_radius_blas!(df, op::KnockOnOperator, kernels::KnockOnKernels, factor, ir)
    knockon = op.knockon
    rates, pitch, W, row = kernels.energy.rates, kernels.pitch, op.primary_weights, op.knockon_row
    Y, gathered = op.slice_products, op.gathered_weights
    nxi_knockon = pitch_cell_count(knockon)
    # Y_m = D_m W[:, K_m] for every slice
    for m in eachindex(kernels.slice_primaries)
        primaries = kernels.slice_primaries[m]
        n = length(primaries)
        n == 0 && continue
        @inbounds for (c, k) in enumerate(primaries), l in axes(W, 1)
            gathered[l, c] = W[l, k]
        end
        offset = kernels.slice_offsets[m]
        mul!(view(Y, :, offset+1:offset+n), view(pitch.table, :, :, m), view(gathered, :, 1:n))
    end
    # gather with the energy kernel and the interpolation weights
    @inbounds for i in 1:momentum_cell_count(knockon)
        fill!(row, 0)
        for k in axes(rates, 2)
            c0 = kernels.lower_columns[i, k]
            c0 == 0 && continue
            s = rates[i, k]
            c1 = kernels.upper_columns[i, k]
            w1 = pitch.upper_weight[i, k]
            a0 = s * (1 - w1)
            if c1 == 0
                for j in 1:nxi_knockon
                    row[j] += a0 * Y[j, c0]
                end
            else
                a1 = s * w1
                for j in 1:nxi_knockon
                    row[j] += a0 * Y[j, c0] + a1 * Y[j, c1]
                end
            end
        end
        for j in 1:nxi_knockon
            df[i, j, ir] += factor * row[j] / knockon.phase_space_jacobian[i, j, ir]
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
updating it in place (e.g. between Newton iterations) changes the operator.

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
