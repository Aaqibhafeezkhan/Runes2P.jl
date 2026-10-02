# Momentum-conserving field-particle term of the electron-electron collisions. The
# test-particle operator of kinetic_terms.jl loses the parallel momentum that
# the electrons give to each other; a model field-particle term returns it to the bulk as
# a shifted Maxwellian, with heat-flux corrections. The term has rank 1 to 3 per radius, is
# applied matrix-free and enters the block preconditioners and the direct solver through a
# Woodbury update (linear_solver.jl, krylov.jl).
# Included after system/terms.jl, since it defines an EquationTerm.

"""
    LowRankBlock(rows, columns, left, right)

Part `left * right'` of a term's Jacobian, with rows `rows` and columns `columns` of the
state vector (contiguous ranges). `left` is `length(rows) × k`, `right` is
`length(columns) × k`, `k` the rank.
"""
struct LowRankBlock{T}
    rows::UnitRange{Int}
    columns::UnitRange{Int}
    left::Matrix{T}
    right::Matrix{T}
end

"""
    low_rank_jacobian(term, u, layout, row, t)

For a `:matrix_free` term whose Jacobian at the state `u` is a sum of low-rank blocks:
the vector of [`LowRankBlock`](@ref)s. `nothing` (the default) for any other term. The
block preconditioners of [`GMRES`](@ref) and the direct solver [`SparseLU`](@ref) add
the blocks by a Woodbury update, so the term needs no stored Jacobian.
"""
low_rank_jacobian(::EquationTerm, u, layout, row, t) = nothing

"""
    has_low_rank_jacobian(term)

Whether [`low_rank_jacobian`](@ref) returns the whole Jacobian of `term` (false by
default). A direct solver takes such matrix-free terms by a Woodbury update.
"""
has_low_rank_jacobian(::EquationTerm) = false

# electron-electron parts of the collision frequencies: the free (cold) electrons as
# targets, without the bound electrons and the ions, which take the momentum they receive
function _electron_electron_slowing_down_frequency(model::CollisionModel, ir, p)
    p == 0 && return zero(p * model.scaled_bessel_k0[ir])
    return COLLISION_PREFACTOR * (1 + p^2) / p^3 * _coulomb_logarithm(model, :electron_electron, p, ir) *
           _thermal_slowing_down_factor(model, ir, p) * model.plasma.cold_electron_density[ir]
end

function _electron_electron_deflection_frequency(model::CollisionModel, ir, p)
    p == 0 && return zero(p * model.scaled_bessel_k0[ir])
    return COLLISION_PREFACTOR * sqrt(1 + p^2) / p^3 * _coulomb_logarithm(model, :electron_electron, p, ir) *
           _thermal_deflection_factor(model, ir, p) * model.plasma.cold_electron_density[ir]
end

"""
    electron_electron_operator(grid, plasma, settings; scheme=:exponential_fitting)

Test-particle operator of the electron-electron collisions alone: slowing down, pitch-angle
scattering and energy diffusion `Θγν_s` from the free electrons of `plasma`, with the
thermal factors of `settings.mode = :full`. Bound electrons and ions are left out. Both
ends of the momentum grid have no flux (the friction points inwards at `p_max`, and the
full-mode slowing down vanishes at `p = 0`), so the operator conserves particles.
"""
function electron_electron_operator(grid::KineticGrid, plasma::PlasmaState, settings::CollisionSettings;
                                    scheme::Symbol=:exponential_fitting)
    settings.mode === :full ||
        throw(ArgumentError("the field-particle term needs collision mode :full (a Maxwellian bulk in f)"))
    op = AdvectionDiffusion(grid; scheme, p_upper_boundary=:outflow)
    model = CollisionModel(plasma, settings)
    m = momentum_grid(grid)
    np, nxi, nr = size(grid)
    for ir in 1:nr
        theta = _normalized_temperature(model, ir)
        for k in 1:np+1
            p = m.p_faces[k]
            nu = _electron_electron_slowing_down_frequency(model, ir, p)
            for j in 1:nxi
                op.advection_p[k, j, ir] -= p * nu
                op.diffusion_p_p[k, j, ir] += theta * sqrt(1 + p^2) * nu
            end
        end
        nxi == 1 && continue
        for i in 1:np
            nu = _electron_electron_deflection_frequency(model, ir, m.p_centers[i])
            for k in 1:nxi+1
                op.diffusion_xi_xi[i, k, ir] += nu * (1 - m.xi_faces[k]^2) / 2
            end
        end
    end
    return op
end

# Ratios N^{jk}/M^{jk} of the field-particle to the test-particle matrix elements of the
# linearized Landau operator for like species (equal mass and temperature), j, k = 0, 1, 2,
# in the Laguerre basis v∥ L_j^{(3/2)}(x²) f_M: Sugama et al., Phys. Plasmas 26 (2019),
# doi 10.1063/1.5115440, appendix C, eqs. (C2)-(C5) at α = 1. Row and column 0 are -1 by
# momentum conservation (N^{0k} = -M^{0k}). M¹¹ = -59/(16√2), N¹¹ = 27/(16√2),
# M¹² = -417/(128√2), N¹² = 225/(128√2), M²² = -8385/(1024√2), N²² = 2125/(1024√2).
const LIKE_SPECIES_FIELD_TO_TEST_RATIOS = [-1.0 -1.0 -1.0
                                           -1.0 -27/59 -225/417
                                           -1.0 -225/417 -2125/8385]

# generalized Laguerre polynomial L_j^{(3/2)}(y), j = 0, 1, 2
_laguerre(j, y) = j == 0 ? one(y) : j == 1 ? 5 / 2 - y : 35 / 8 - 7y / 2 + y^2 / 2

"""
    FieldParticleOperator(grid, plasma, settings=CollisionSettings();
                          modes=3, scheme=:exponential_fitting, pitch_rule=:cell_center)

Model field-particle term of the linearized electron-electron collisions, of rank `modes`
per radius. It conserves parallel momentum exactly, and it reproduces the matrix elements
of the Landau field-particle operator between the first `modes` Laguerre modes of the
`l = 1` part of `f`, as the improved Sugama operator does (Sugama et al., Phys. Plasmas
26 (2019), doi 10.1063/1.5115440):

    C_fp(f) = Σ_ab g_a A_ab (ℓ_b · f),   ℓ_b = C_eeᵀ (m ∘ L_b),   g_a = f_M ℓ_a / Δ𝒱,

per radius, `a, b = 0, …, modes - 1`.

- `C_ee` is the discrete [`electron_electron_operator`](@ref) with `scheme`.
- `m` is the parallel momentum of each cell, `m_ij = 𝒱'_ij Δp_i p_i w_j`, with the pitch
  weights `w_j` of [`pitch_moment_weights`](@ref)`(; pitch_rule)`.
- `L_b = L_b^{(3/2)}(x²)` are the Laguerre polynomials of the Braginskii expansion, `x² =
  (γ - 1)/Θ` at the cold-electron temperature (`L_0 = 1`, `L_1 = 5/2 - x²`).
  `ℓ_b · f` is the rate at which `C_ee` changes the moment `(m ∘ L_b) · f`, so `ℓ_0 · f` is
  the momentum `C_ee` removes.
- `g_a` is the Maxwellian `f_M ∝ exp(-(γ-1)/Θ)` times `ℓ_a` per unit phase-space volume
  `Δ𝒱_ij = 𝒱'_ij Δp_i Δξ_j`, made particle-free by subtracting its density times `f_M`.
  In the continuum limit `g_a ∝ C_ee(p ξ L_a f_M)`: `g_0` is the shifted Maxwellian of the
  model operator of Hirshman and Sigmar (1976), and `g_1` its heat-flux counterpart.
- `A` is fixed by the matrix elements `T_bd = ℓ_b · φ_d` of `C_ee` between the modes
  `φ_d = p ξ L_d f_M`: the field-particle elements become `R_bd T_bd`, with `R` the Landau
  ratios `N^{bd}/M^{bd}` for like species (`R_0d = R_b0 = -1`, `R_11 = -27/59`). Row 0 makes
  `m · (C_ee f + C_fp f) = 0` for every `f` to round-off.

With `modes = 1` this is the Hirshman-Sigmar (and Sugama 2009) momentum-restoring model.
It is exact for a shifted Maxwellian and gives 0.5172 σ_L at `Z = 1` against the
Spitzer-Härm 0.5816 σ_L. The heat-flux mode closes most of the gap, the
third mode the rest (`test/test_field_particle.jl`).

The ratios `R` are those of the nonrelativistic Landau operator, so the term is accurate
for a nonrelativistic bulk (`T ≪ mₑc²`). The default `pitch_rule = :cell_center` (`w_j =
ξ_j Δξ_j`) is the left eigenvector of the discrete pitch scattering, so the modes are
exact first Legendre modes on the grid and the conductivity does not depend on the pitch
resolution (to 1e-5). With `:linear`, the moment of [`current_density`](@ref), it
converges to the same value at second order in the pitch cell width. The normalization of
`f_M` cancels. On a pitch grid symmetric about `ξ = 0` the term vanishes on any `f` even
in `ξ`. Needs collision mode `:full`, where the bulk is part of `f`, and a shared momentum
grid.
"""
struct FieldParticleOperator{T}
    grid::KineticGrid{T}
    modes::Int
    scheme::Symbol
    pitch_rule::Symbol
    momentum_weights::Vector{Matrix{T}}   # m per radius
    loss_functionals::Vector{Matrix{T}}   # per radius: column b is ℓ_b (cells × modes)
    restoring_shapes::Vector{Matrix{T}}   # per radius: column b is Σ_a g_a A_ab (cells × modes)
    coefficient_source::Base.RefValue{Any}
end

function FieldParticleOperator(grid::KineticGrid{T}, plasma::PlasmaState,
                               settings::CollisionSettings=CollisionSettings(); modes::Integer=3,
                               scheme::Symbol=:exponential_fitting, pitch_rule::Symbol=:cell_center) where {T}
    has_shared_momentum_grid(grid) ||
        throw(ArgumentError("the field-particle operator needs a shared momentum grid"))
    1 <= modes <= size(LIKE_SPECIES_FIELD_TO_TEST_RATIOS, 1) ||
        throw(ArgumentError("modes must be 1, 2 or 3"))
    np, nxi, nr = size(grid)
    weights = Matrix{T}[]
    for ir in 1:nr
        m = momentum_grid(grid, ir)
        w = pitch_moment_weights(m; pitch_rule)
        V = phase_space_jacobian(grid, ir)
        push!(weights, [V[i, j] * m.p_widths[i] * m.p_centers[i] * w[j] for i in 1:np, j in 1:nxi])
    end
    op = FieldParticleOperator(grid, Int(modes), scheme, pitch_rule, weights,
                               [zeros(T, np * nxi, modes) for _ in 1:nr], [zeros(T, np * nxi, modes) for _ in 1:nr],
                               Ref{Any}(nothing))
    return set_field_particle_coefficients!(op, plasma, settings)
end

"""
    set_field_particle_coefficients!(op::FieldParticleOperator, plasma, settings=CollisionSettings())

Rebuild the functionals and the restoring shapes of `op` from `plasma` (the cold-electron
temperature and density, the Coulomb logarithm) for a plasma that changes with time.
"""
function set_field_particle_coefficients!(op::FieldParticleOperator{T}, plasma::PlasmaState,
                                          settings::CollisionSettings=CollisionSettings()) where {T}
    grid = op.grid
    radial_cell_count(plasma) == radial_cell_count(grid) ||
        throw(DimensionMismatch("plasma and grid radii differ"))
    np, nxi, nr = size(grid)
    K = op.modes
    collisions = operator_matrix(electron_electron_operator(grid, plasma, settings; scheme=op.scheme))
    ratios = LIKE_SPECIES_FIELD_TO_TEST_RATIOS[1:K, 1:K]
    # the moments m ∘ L_b of all radii, and the functionals ℓ_b = C_eeᵀ (m ∘ L_b)
    moments = zeros(T, np, nxi, nr, K)
    for ir in 1:nr
        m = momentum_grid(grid, ir)
        theta = plasma.cold_electron_temperature[ir] / Constants.ELECTRON_REST_ENERGY
        for b in 1:K, j in 1:nxi, i in 1:np
            x2 = m.p_centers[i]^2 / (1 + sqrt(1 + m.p_centers[i]^2)) / theta
            moments[i, j, ir, b] = op.momentum_weights[ir][i, j] * _laguerre(b - 1, x2)
        end
    end
    functionals = reshape(transpose(collisions) * reshape(moments, :, K), np, nxi, nr, K)
    for ir in 1:nr
        m = momentum_grid(grid, ir)
        V = phase_space_jacobian(grid, ir)
        theta = plasma.cold_electron_temperature[ir] / Constants.ELECTRON_REST_ENERGY
        volume = vec([V[i, j] * m.p_widths[i] * m.xi_widths[j] for i in 1:np, j in 1:nxi])
        x2 = vec([m.p_centers[i]^2 / (1 + sqrt(1 + m.p_centers[i]^2)) / theta for i in 1:np, j in 1:nxi])
        maxwellian = exp.(-x2)
        ell = reshape(view(functionals, :, :, ir, :), np * nxi, K)
        W = reshape(view(moments, :, :, ir, :), np * nxi, K)
        # the modes φ_d = p ξ L_d f_M at the cell centres
        pxi = vec([m.p_centers[i] * m.xi_centers[j] for i in 1:np, j in 1:nxi])
        phi = hcat((pxi .* _laguerre.(d - 1, x2) .* maxwellian for d in 1:K)...)
        shapes = maxwellian .* ell ./ volume
        for a in 1:K
            shapes[:, a] .-= (sum(volume .* shapes[:, a]) / sum(volume .* maxwellian)) .* maxwellian
        end
        test_elements = transpose(ell) * phi            # T_bd
        shape_elements = transpose(W) * shapes          # G_ca
        A = shape_elements \ ((ratios .* test_elements) / test_elements)
        # a DomainError: a rebuild in a run with a changing plasma reaches it, and the solvers
        # take that as a failed solve
        all(isfinite, A) ||
            throw(DomainError(plasma.cold_electron_temperature[ir],
                              "the restoring shapes are degenerate at radius $ir at this cold-electron temperature"))
        op.loss_functionals[ir] .= ell
        op.restoring_shapes[ir] .= shapes * A
    end
    op.coefficient_source[] = (; plasma, settings)
    return op
end

# df += scale Σ_b s_b (ℓ_b · f) at radius ir, for matrices f and df of that radius
function _add_field_particle!(df::AbstractMatrix, op::FieldParticleOperator, f::AbstractMatrix, ir; scale=1)
    ell, shapes = op.loss_functionals[ir], op.restoring_shapes[ir]
    for b in 1:op.modes
        rate = zero(promote_type(eltype(f), eltype(ell)))
        @inbounds for k in eachindex(f)
            rate += ell[k, b] * f[k]
        end
        @inbounds for k in eachindex(df)
            df[k] += scale * rate * shapes[k, b]
        end
    end
    return df
end

"""
    apply!(df, op::FieldParticleOperator, f; scale=1)

Add `scale` times the field-particle term of `f` (3D array on the operator's grid) to `df`.
"""
function apply!(df::AbstractArray{<:Any,3}, op::FieldParticleOperator, f::AbstractArray{<:Any,3}; scale=1)
    size(f) == size(df) == size(op.grid) || throw(DimensionMismatch("f and df must have the grid size"))
    for ir in 1:radial_cell_count(op.grid)
        _add_field_particle!(view(df, :, :, ir), op, view(f, :, :, ir), ir; scale)
    end
    return df
end

"""
    parallel_momentum(op::FieldParticleOperator, f)

Parallel momentum `Σ m_ij f_ij` of `f` per radius, in units of `mₑc` times the density
unit of `f` times the radial volume element (the moment the operator conserves).
"""
parallel_momentum(op::FieldParticleOperator, f::AbstractArray{<:Any,3}) =
    [sum(op.momentum_weights[ir] .* view(f, :, :, ir)) for ir in 1:radial_cell_count(op.grid)]

"""
    FieldParticleTerm(unknown, operator::FieldParticleOperator; plasma=nothing, settings=nothing)

The momentum-conserving field-particle term [`FieldParticleOperator`](@ref) applied to the
kinetic unknown `unknown`, added to its own equation. It couples all cells of a radius,
so it is `:matrix_free`, with a Jacobian of rank `modes` per radius
([`low_rank_jacobian`](@ref)) that the block preconditioner of [`GMRES`](@ref) and
[`SparseLU`](@ref) take by a Woodbury update. Linear in `f`.

`plasma` is the source of the plasma that the functionals and the restoring shapes follow:
a [`PlasmaSource`](@ref), a [`PlasmaState`](@ref) or a function of
time that returns one, as for the [`FokkerPlanckTerm`](@ref). They are rebuilt with
[`set_field_particle_coefficients!`](@ref) and `settings` (by default those the operator
was built with) when the plasma values of a radius change. The Woodbury factors of the
solvers are taken from [`low_rank_jacobian`](@ref) at each factorization, so they follow.

A plasma that reads unknowns ([`PlasmaFromUnknowns`](@ref)) is always taken
at the state at the start of the step ([`prepare_step!`](@ref)), the lagged treatment of
the [`FokkerPlanckTerm`](@ref). The derivative of the term with respect to the temperature
and the density at a radius is a dense column of all cells of that radius, which has no
place in the low-rank Jacobian of the term.
"""
struct FieldParticleTerm{T,P} <: EquationTerm
    unknown::Symbol
    operator::FieldParticleOperator{T}
    plasma::P                     # nothing, or the PlasmaSource of the coefficients
    settings::CollisionSettings
    plasma_key::PlasmaKey         # the plasma values of the last rebuild
end

function FieldParticleTerm(unknown::Symbol, operator::FieldParticleOperator; plasma=nothing,
                           settings::Union{Nothing,CollisionSettings}=nothing)
    if settings === nothing
        built = operator.coefficient_source[]
        settings = built === nothing ? CollisionSettings() : built.settings
    end
    source = plasma === nothing ? nothing : plasma_source(plasma)
    return FieldParticleTerm(unknown, operator, source, settings, PlasmaKey())
end

has_plasma_source(term::FieldParticleTerm) = term.plasma !== nothing

# a plasma from the unknowns is lagged: prepare_step! rebuilds the term
refresh_coefficients!(term::FieldParticleTerm, u, layout, t) =
    is_state_dependent(term.plasma) ? term.operator : _rebuild_coefficients!(term, u, layout, t)

function prepare_step!(term::FieldParticleTerm, u_old, layout, row, t)
    is_state_dependent(term.plasma) && _rebuild_coefficients!(term, u_old, layout, t)
    return nothing
end

function _rebuild_coefficients!(term::FieldParticleTerm, u, layout, t)
    term.plasma === nothing && return term.operator
    plasma = changed_plasma!(term.plasma_key, term.plasma, u, layout, t; electric_field=false)
    plasma === nothing && return term.operator
    try
        set_field_particle_coefficients!(term.operator, plasma, term.settings)
    catch
        reset_key!(term.plasma_key)
        rethrow()
    end
    return term.operator
end

dependencies(term::FieldParticleTerm) = [term.unknown; collect(Symbol, source_dependencies(term.plasma))]
coupling(::FieldParticleTerm) = :matrix_free
has_low_rank_jacobian(::FieldParticleTerm) = true
is_linear(term::FieldParticleTerm) = !is_state_dependent(term.plasma)

function validate(term::FieldParticleTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    layout.kinds[id] === :kinetic && id == row ||
        throw(ArgumentError("FieldParticleTerm acts on a kinetic unknown in its own equation"))
    _same_momentum_grids(term.operator.grid, kinetic_grid(layout, id)) ||
        throw(DimensionMismatch("the operator's grid does not match $(term.unknown)"))
    return nothing
end

# the term on the distribution of the state vector x, with the coefficients as they are
function _add_field_particle_term!(out, term::FieldParticleTerm, x, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        _add_field_particle!(kinetic_view(out, layout, row, ir), term.operator, kinetic_view(x, layout, id, ir), ir)
    end
    return out
end

function rhs!(out, term::FieldParticleTerm, u, layout, row, t)
    refresh_coefficients!(term, u, layout, t)
    return _add_field_particle_term!(out, term, u, layout, row)
end

# linear in f: the product is the term on v, with the coefficients of the state u
function jacobian_vector_product!(out, term::FieldParticleTerm, u, v, layout, row, t)
    refresh_coefficients!(term, u, layout, t)
    return _add_field_particle_term!(out, term, v, layout, row)
end

function low_rank_jacobian(term::FieldParticleTerm, u, layout, row, t)
    refresh_coefficients!(term, u, layout, t)
    id = unknown_id(layout, term.unknown)
    op = term.operator
    return [LowRankBlock(unknown_range(layout, row, ir), unknown_range(layout, id, ir),
                         op.restoring_shapes[ir], op.loss_functionals[ir])
            for ir in 1:radial_cell_count(layout)]
end
