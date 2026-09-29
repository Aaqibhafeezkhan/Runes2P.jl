# A preconditioner for the coupled system: the kinetic unknowns in blocks per
# radius, the fluid and scalar unknowns in one block that holds their coupling between the
# radii exactly through its Schur complement, and radial line solves for the radial
# transport of the distributions.

"""
    FluidSchurLU(; radial_lines=:auto)

The default preconditioner of [`GMRES`](@ref), for systems whose radii are coupled through the fluid
unknowns (Ampère's law, the plasma current and the other scalars) and through the radial
transport of the distributions. [`RadiusBlockLU`](@ref) leaves both
couplings out, and its Krylov count grows with the number of radii.

- The kinetic unknowns of each radius form one block, factorized exactly (UMFPACK), with
  the Woodbury update of low-rank terms as in [`RadiusBlockLU`](@ref).
- The fluid unknowns of all radii (every radial and scalar unknown) form one block. With
  the kinetic blocks eliminated, its Schur complement
  `S = A_ff − Σ_r A_fk,r A_kk,r⁻¹ A_kf,r` is a dense matrix of the size of the fluid
  unknowns (about 11 per radius and 3 scalars in a self-consistent run), factorized
  exactly. The columns `A_kf,r` (the electric field, the magnitude of the particle source)
  are few per radius. This stage is the exact inverse of the stored matrix without the
  kinetic couplings between radii.
- The radial lines are the tridiagonal systems along the radii of each kinetic cell: the
  diagonal of the stored matrix and the couplings of the cell to the same cell at the two
  neighbouring radii. The application is symmetric, lines, Schur stage, lines, each stage
  on the residual of the ones before: one block solve, two line solves and two products
  with the stored matrix and the low-rank terms of the blocks. The lines need one pitch
  grid for all radii (the same restriction as the four-point schemes in r).
  Without stored kinetic couplings between radii (no radial transport) they are left out.
  `radial_lines = :auto` (default) takes them where every kinetic unknown has one pitch
  grid at all radii and leaves them out otherwise, with a warning: the fluid coupling stays
  exact, and the radial transport of the distributions is left to the Krylov iteration.
  `true` refuses a grid whose pitch faces differ between radii, `false` never takes them.

The preconditioner is a fixed linear map for each linear solve. With
`preconditioner_refresh = :on_demand` the factors can be older than the stored matrix,
whose current values the residuals between the stages take: still a fixed linear map for
each solve, and a preconditioner of the current matrix. A kinetic row must not couple to a
radial unknown of another radius; `GMRES` refuses such a system.

Separation on a self-consistent system with an electric field unknown:
without radial transport the Schur stage alone is exact and GMRES
takes 1 to 2 iterations at 6 to 24 radii, against 8 to 22 with `RadiusBlockLU`. With
radial diffusion at a coupling typical of a disruption run (`D Δt/Δr² ≈ 5`) the symmetric
application takes 5 iterations at 12 and 24 radii, against 49 and 55.
"""
struct FluidSchurLU <: BlockPreconditioner
    radial_lines::Union{Bool,Symbol}
end

function FluidSchurLU(; radial_lines::Union{Bool,Symbol}=:auto)
    radial_lines isa Bool || radial_lines === :auto ||
        throw(ArgumentError("radial_lines is true, false or :auto, got $radial_lines"))
    return FluidSchurLU(radial_lines)
end

# whether every kinetic unknown has one pitch grid at all radii, as the lines need
_one_pitch_grid(layout::UnknownLayout) =
    all(layout.kinds[id] !== :kinetic ||
        (all(==(layout.sizes[id, 1]), layout.sizes[id, :]) && unknown.grid.radial_overlaps !== nothing &&
         all(has_identical_grids, unknown.grid.radial_overlaps))
        for (id, unknown) in enumerate(layout.unknowns))

# the tridiagonal systems along the radii of the kinetic cells
struct RadialLines{T}
    index::Matrix{Int}      # (line, radius): index in the state
    lower::Matrix{Int}      # position in nzval of the coupling to the radius below, 0 without
    diagonal::Matrix{Int}
    upper::Matrix{Int}
    a::Matrix{T}            # values: below, diagonal, above
    b::Matrix{T}
    c::Matrix{T}
    modified::Vector{T}     # scratch of the Thomas algorithm (per radius)
    right::Vector{T}
end

mutable struct FluidSchurFactors{T,F}
    const matrix::SparseMatrixCSC{T,Int}
    const kinetic_ranges::Vector{UnitRange{Int}}      # per radius
    const fluid::Vector{Int}                          # indices of the radial and scalar unknowns
    const blocks::Vector{SparseMatrixCSC{T,Int}}
    const block_positions::Vector{Vector{Int}}
    const factorizations::Vector{F}
    const low_rank::Vector{Union{Nothing,LowRankCorrection{T}}}
    const coupled::Vector{Vector{Int}}                # per radius: positions in `fluid` of the columns of A_kf
    const kf_entries::Vector{Vector{NTuple{3,Int}}}   # per radius: (kinetic row, column of `coupled`, nz position)
    const fk_entries::Vector{Vector{NTuple{3,Int}}}   # per radius: (fluid row, kinetic column, nz position)
    const fk_values::Vector{Vector{T}}
    const ff_entries::Vector{NTuple{3,Int}}           # (fluid row, fluid column, nz position)
    const solved_columns::Vector{Matrix{T}}           # per radius: A_kk⁻¹ A_kf
    const schur::Matrix{T}
    const schur_factors::Matrix{T}                    # the storage of the factorization of `schur`
    schur_factorization::Union{Nothing,LinearAlgebra.LU{T,Matrix{T},Vector{Int}}}
    const lines::Union{Nothing,RadialLines{T}}
    const low_rank_terms::Vector{Union{Nothing,Tuple{Matrix{T},Matrix{T}}}}   # per radius: the scaled left and right factors
    const low_rank_coefficients::Vector{Vector{T}}
    const kinetic_output::Vector{Vector{T}}
    const kinetic_input::Vector{Vector{T}}
    const coupled_values::Vector{Vector{T}}
    const fluid_work::Vector{T}
    const residual::Vector{T}
    const partial::Vector{T}
end

# position of the entry (row, column) in the nonzeros of `matrix`, 0 when it is not stored
function _entry_position(matrix::SparseMatrixCSC, row, column)
    range = nzrange(matrix, column)
    k = searchsortedfirst(matrix.rowval, row, first(range), last(range), Base.Order.Forward)
    return k <= last(range) && matrix.rowval[k] == row ? k : 0
end

function _kinetic_range(layout::UnknownLayout, ir)
    ranges = [unknown_range(layout, id, ir) for id in eachindex(layout.unknowns) if layout.kinds[id] === :kinetic]
    isempty(ranges) && return 1:0
    sort!(ranges; by=first)
    for k in 2:length(ranges)
        first(ranges[k]) == last(ranges[k-1]) + 1 ||
            throw(ArgumentError("the kinetic unknowns of radius $ir are not contiguous in the layout"))
    end
    return first(first(ranges)):last(last(ranges))
end

function _radial_lines(layout::UnknownLayout, matrix::SparseMatrixCSC{T}) where {T}
    nr = radial_cell_count(layout)
    lines = Vector{Vector{Int}}()
    for (id, unknown) in enumerate(layout.unknowns)
        layout.kinds[id] === :kinetic || continue
        all(==(layout.sizes[id, 1]), layout.sizes[id, :]) && unknown.grid.radial_overlaps !== nothing &&
            all(has_identical_grids, unknown.grid.radial_overlaps) ||
            throw(ArgumentError("the radial lines of FluidSchurLU need one pitch grid for all radii (issue #72): " *
                                "$(unknown.name) has pitch grids that differ between radii"))
        for q in 1:layout.sizes[id, 1]
            push!(lines, [first(unknown_range(layout, id, ir)) + q - 1 for ir in 1:nr])
        end
    end
    index = isempty(lines) ? zeros(Int, 0, nr) : permutedims(reduce(hcat, lines))
    count = size(index, 1)
    lower, diagonal, upper = zeros(Int, count, nr), zeros(Int, count, nr), zeros(Int, count, nr)
    for l in 1:count, ir in 1:nr
        row = index[l, ir]
        diagonal[l, ir] = _entry_position(matrix, row, row)
        ir > 1 && (lower[l, ir] = _entry_position(matrix, row, index[l, ir-1]))
        ir < nr && (upper[l, ir] = _entry_position(matrix, row, index[l, ir+1]))
    end
    return RadialLines(index, lower, diagonal, upper, zeros(T, count, nr), zeros(T, count, nr), zeros(T, count, nr),
                       zeros(T, nr), zeros(T, nr))
end

function preconditioner_factors(p::FluidSchurLU, matrix::SparseMatrixCSC{T}, layout::UnknownLayout) where {T}
    nr = radial_cell_count(layout)
    n = size(matrix, 1)
    kinetic_ranges = [_kinetic_range(layout, ir) for ir in 1:nr]
    radius = zeros(Int, n)
    for ir in 1:nr
        radius[radius_block_range(layout, ir)] .= ir
    end
    kinetic_radius = zeros(Int, n)
    for ir in 1:nr
        kinetic_radius[kinetic_ranges[ir]] .= ir
    end
    fluid = findall(==(0), kinetic_radius)
    fluid_position = zeros(Int, n)
    fluid_position[fluid] .= eachindex(fluid)
    extracted = [_extract_block(matrix, range) for range in kinetic_ranges]
    blocks, block_positions = first.(extracted), last.(extracted)
    # a radius without kinetic unknowns keeps a placeholder of size 1
    factorizations = [lu(isempty(range) ? SparseMatrixCSC{T,Int}(LinearAlgebra.I, 1, 1) : block; check=false)
                      for (range, block) in zip(kinetic_ranges, blocks)]
    kf = [NTuple{3,Int}[] for _ in 1:nr]
    fk = [NTuple{3,Int}[] for _ in 1:nr]
    ff = NTuple{3,Int}[]
    coupled_sets = [Int[] for _ in 1:nr]
    for column in 1:n, k in nzrange(matrix, column)
        row = matrix.rowval[k]
        row_radius, column_radius = kinetic_radius[row], kinetic_radius[column]
        if row_radius > 0 && column_radius == 0
            radius[column] == 0 || radius[column] == row_radius ||
                throw(ArgumentError("a kinetic row of radius $row_radius couples to a radial unknown of radius " *
                                    "$(radius[column]): FluidSchurLU eliminates the kinetic blocks radius by radius"))
            push!(coupled_sets[row_radius], fluid_position[column])
            push!(kf[row_radius], (row - first(kinetic_ranges[row_radius]) + 1, fluid_position[column], k))
        elseif row_radius == 0 && column_radius > 0
            push!(fk[column_radius], (fluid_position[row], column - first(kinetic_ranges[column_radius]) + 1, k))
        elseif row_radius == 0 && column_radius == 0
            push!(ff, (fluid_position[row], fluid_position[column], k))
        end
    end
    coupled = [sort!(unique(set)) for set in coupled_sets]
    # the columns of A_kf as positions in `coupled`
    for ir in 1:nr
        lookup = Dict(position => c for (c, position) in enumerate(coupled[ir]))
        kf[ir] = [(row, lookup[position], k) for (row, position, k) in kf[ir]]
    end
    nf = length(fluid)
    with_lines = nr > 1 && (p.radial_lines === true || (p.radial_lines === :auto && _one_pitch_grid(layout)))
    if p.radial_lines === :auto && !with_lines && nr > 1
        @warn "FluidSchurLU without radial lines: the pitch grids differ between radii (issue #72). The fluid " *
              "coupling stays exact, the radial transport of the distributions is left to the Krylov iteration" maxlog = 1
    end
    lines = with_lines ? _radial_lines(layout, matrix) : nothing
    # without kinetic couplings between radii the lines would be a diagonal step
    lines !== nothing && all(==(0), lines.lower) && all(==(0), lines.upper) && (lines = nothing)
    return FluidSchurFactors(matrix, kinetic_ranges, fluid, blocks, block_positions, factorizations,
                             Union{Nothing,LowRankCorrection{T}}[nothing for _ in 1:nr], coupled, kf, fk,
                             [zeros(T, length(entries)) for entries in fk], ff,
                             [zeros(T, length(kinetic_ranges[ir]), length(coupled[ir])) for ir in 1:nr],
                             zeros(T, nf, nf), zeros(T, nf, nf), nothing, lines,
                             Union{Nothing,Tuple{Matrix{T},Matrix{T}}}[nothing for _ in 1:nr], [T[] for _ in 1:nr],
                             [zeros(T, length(r)) for r in kinetic_ranges], [zeros(T, length(r)) for r in kinetic_ranges],
                             [zeros(T, length(c)) for c in coupled], zeros(T, nf), zeros(T, n), zeros(T, n))
end

# y = A_kk,r⁻¹ x with the Woodbury update of the block
function _kinetic_solve!(y, factors::FluidSchurFactors, ir, x)
    ldiv!(y, factors.factorizations[ir], x)
    factors.low_rank[ir] === nothing || _apply_low_rank_correction!(y, factors.low_rank[ir])
    return y
end

function factorize!(factors::FluidSchurFactors{T}, ::FluidSchurLU, matrix::SparseMatrixCSC, low_rank=nothing) where {T}
    nzval = matrix.nzval
    for ir in eachindex(factors.blocks)
        range = factors.kinetic_ranges[ir]
        isempty(range) && continue
        block, positions = factors.blocks[ir], factors.block_positions[ir]
        @inbounds for m in eachindex(positions)
            block.nzval[m] = nzval[positions[m]]
        end
        lu!(factors.factorizations[ir], block; check=false)
        issuccess(factors.factorizations[ir]) || return false
        factors.low_rank[ir] = nothing
        factors.low_rank_terms[ir] = nothing
        if low_rank !== nothing
            scaled = _scaled_low_rank(low_rank.blocks, range, low_rank.row_weights, low_rank.scales, T)
            if scaled !== nothing
                factors.low_rank_terms[ir] = scaled
                resize!(factors.low_rank_coefficients[ir], size(scaled[2], 2))
                factorization = factors.factorizations[ir]
                factors.low_rank[ir] = _low_rank_correction(scaled..., (y, x) -> ldiv!(y, factorization, x))
                factors.low_rank[ir] === nothing && return false
            end
        end
        # A_kk⁻¹ A_kf, column by column
        columns = factors.solved_columns[ir]
        input = factors.kinetic_input[ir]
        for c in axes(columns, 2)
            fill!(input, zero(T))
            for (row, column, k) in factors.kf_entries[ir]
                column == c && (input[row] += nzval[k])
            end
            _kinetic_solve!(view(columns, :, c), factors, ir, input)
        end
        for (e, (_, _, k)) in enumerate(factors.fk_entries[ir])
            factors.fk_values[ir][e] = nzval[k]
        end
    end
    # the Schur complement of the fluid unknowns
    schur = factors.schur
    fill!(schur, zero(T))
    for (row, column, k) in factors.ff_entries
        schur[row, column] += nzval[k]
    end
    for ir in eachindex(factors.blocks)
        columns, coupled = factors.solved_columns[ir], factors.coupled[ir]
        for (e, (row, kinetic_column, _)) in enumerate(factors.fk_entries[ir])
            value = factors.fk_values[ir][e]
            for c in eachindex(coupled)
                schur[row, coupled[c]] -= value * columns[kinetic_column, c]
            end
        end
    end
    if isempty(schur)
        factors.schur_factorization = nothing
    else
        copyto!(factors.schur_factors, schur)
        factorization = LinearAlgebra.lu!(factors.schur_factors; check=false)
        issuccess(factorization) || return false
        factors.schur_factorization = factorization
    end
    lines = factors.lines
    if lines !== nothing
        for ir in axes(lines.index, 2), l in axes(lines.index, 1)
            lines.a[l, ir] = lines.lower[l, ir] == 0 ? zero(T) : nzval[lines.lower[l, ir]]
            lines.b[l, ir] = lines.diagonal[l, ir] == 0 ? zero(T) : nzval[lines.diagonal[l, ir]]
            lines.c[l, ir] = lines.upper[l, ir] == 0 ? zero(T) : nzval[lines.upper[l, ir]]
            lines.b[l, ir] == 0 && return false
        end
    end
    return true
end

# the exact inverse of the stored matrix without the kinetic couplings between radii
function _schur_solve!(y, factors::FluidSchurFactors, x)
    fluid_work = factors.fluid_work
    @inbounds for (m, i) in enumerate(factors.fluid)
        fluid_work[m] = x[i]
    end
    for ir in eachindex(factors.blocks)
        range = factors.kinetic_ranges[ir]
        isempty(range) && continue
        input, output = factors.kinetic_input[ir], factors.kinetic_output[ir]
        copyto!(input, 1, x, first(range), length(range))
        _kinetic_solve!(output, factors, ir, input)
        values = factors.fk_values[ir]
        @inbounds for (e, (row, column, _)) in enumerate(factors.fk_entries[ir])
            fluid_work[row] -= values[e] * output[column]
        end
    end
    factors.schur_factorization === nothing || ldiv!(factors.schur_factorization, fluid_work)
    @inbounds for (m, i) in enumerate(factors.fluid)
        y[i] = fluid_work[m]
    end
    for ir in eachindex(factors.blocks)
        range = factors.kinetic_ranges[ir]
        isempty(range) && continue
        output, coupled_values = factors.kinetic_output[ir], factors.coupled_values[ir]
        @inbounds for (c, position) in enumerate(factors.coupled[ir])
            coupled_values[c] = fluid_work[position]
        end
        mul!(output, factors.solved_columns[ir], coupled_values, -1, 1)
        copyto!(y, first(range), output, 1, length(range))
    end
    return y
end

# the tridiagonal systems along the radii (Thomas algorithm), into y at the kinetic cells
function _line_solve!(y, lines::RadialLines{T}, x) where {T}
    nr = size(lines.index, 2)
    modified, right = lines.modified, lines.right
    @inbounds for l in axes(lines.index, 1)
        denominator = lines.b[l, 1]
        modified[1] = nr > 1 ? lines.c[l, 1] / denominator : zero(T)
        right[1] = x[lines.index[l, 1]] / denominator
        for ir in 2:nr
            denominator = lines.b[l, ir] - lines.a[l, ir] * modified[ir-1]
            modified[ir] = ir < nr ? lines.c[l, ir] / denominator : zero(T)
            right[ir] = (x[lines.index[l, ir]] - lines.a[l, ir] * right[ir-1]) / denominator
        end
        y[lines.index[l, nr]] = right[nr]
        for ir in nr-1:-1:1
            right[ir] -= modified[ir] * right[ir+1]
            y[lines.index[l, ir]] = right[ir]
        end
    end
    return y
end

# residual = x − A y with A the stored matrix and the low-rank terms of the blocks
function _stage_residual!(residual, factors::FluidSchurFactors, x, y)
    mul!(residual, factors.matrix, y)
    for ir in eachindex(factors.low_rank_terms)
        terms = factors.low_rank_terms[ir]
        terms === nothing && continue
        left, right = terms
        range = factors.kinetic_ranges[ir]
        coefficients = factors.low_rank_coefficients[ir]
        mul!(coefficients, transpose(right), view(y, range))
        mul!(view(residual, range), left, coefficients, 1, 1)
    end
    @. residual = x - residual
    return residual
end

# y += the line solve of x − A y, at the kinetic cells
function _line_stage!(y, factors::FluidSchurFactors, x)
    partial, residual = factors.partial, factors.residual
    _stage_residual!(residual, factors, x, y)
    fill!(partial, zero(eltype(partial)))
    _line_solve!(partial, factors.lines, residual)
    y .+= partial
    return y
end

function LinearAlgebra.ldiv!(y::AbstractVector, factors::FluidSchurFactors, x::AbstractVector)
    factors.lines === nothing && return _schur_solve!(y, factors, x)
    # lines, Schur stage, lines
    fill!(y, zero(eltype(y)))
    _line_stage!(y, factors, x)
    residual = _stage_residual!(factors.residual, factors, x, y)
    correction = factors.partial
    _schur_solve!(correction, factors, residual)
    y .+= correction
    _line_stage!(y, factors, x)
    return y
end
