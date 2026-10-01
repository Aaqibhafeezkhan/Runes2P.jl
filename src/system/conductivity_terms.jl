# The ohmic current next to the current of the distribution: the
# conductivity of the discrete kinetic operator by its linear response, the corrected
# conductivity of DREAM's fully kinetic mode, and the term `σ E` of Ohm's law.
using LinearAlgebra: lu
using SparseArrays: findnz, sparse

# tag of the dual numbers of the derivative with respect to the field
struct FieldResponseTag end

"""
    kinetic_conductivity(operator::AdvectionDiffusion; pitch_rule=:linear,
                         density=cold density of the plasma of the operator, field_particle=nothing)

Conductivity [S/m] of the discrete kinetic operator at every radius, from its linear
response to a small electric field. `operator` (the collisions, and the radiation when it
holds it) is `C`, and `∂L/∂E` the derivative of the operator with the field as an argument
at zero field, by dual numbers ([`apply!`](@ref)`(...; electric_field)`). Per radius the
two bordered systems

    [C  g; wᵀ 0] [f₀; λ₀] = [0; n],        [C  g; wᵀ 0] [f₁; λ₁] = [−(∂L/∂E) f₀; 0]

give the equilibrium `f₀` of the discrete operator with the density `n` and the response
`f₁` without a change of the density, with `w` the density weights of the cells and `g` a
particle source with the shape of a relativistic Maxwellian at the cold temperature
([`ParticleSourceTerm`](@ref)). This is the linearization of a run whose particle source
holds the density, so the conductivity is that which such a run with this operator, this
grid and these collision settings reaches in the steady state of a small field. The
current of the run exceeds it by a term of the order of the square of the field: 1.7e-7 of
it at a thousandth of the Dreicer field (5 critical fields at 10 eV), 1.7e-5 at a
hundredth.

`operator` holds coefficients built without a field, by `fokker_planck_operator(...;
electric_field = false)` or from a plasma without a field, knows its plasma
(`coefficient_source`) and has no radial transport.

- For a flux limiter, the operator is that with the limiter frozen at the Maxwellian, which
  is linear in `f`.
- The flux boundaries at `p_max` (`:constant_flux`, DREAM's default `BC_PHI_CONST`, and the
  extrapolated fluxes) leave the last momentum cell without a net flux in the steady state,
  so the bordered systems are singular with them. The response takes `f = 0` beyond `p_max`
  in their place. The distribution at `p_max` of a conductivity run is far below
  that of the bulk: DREAM's steady current on a production grid is the same with
  `BC_F_0` and `BC_PHI_CONST` to 1.1e-12.
- The boundary `:interface` of a hot-tail grid below a runaway grid takes `f = 0` beyond
  `p_max` as well: the response is that of the bulk on the hot-tail grid, and
  the flux into the runaway grid at a small field is far below the current of the bulk.
- `pitch_rule` is that of the current moment ([`current_density`](@ref)).
- `density` is the density of the equilibrium per radius, the electrons that carry the
  current, by default the cold density of the plasma of the operator. It differs from the
  cold density of the collision coefficients in the limit of the Lorentz gas, which has no
  electron-electron collisions. In a fully kinetic run the current of `f_hot` over all
  momenta is carried by `n_cold + n_hot`: the conductivity with the default density leaves
  out a part of the order `n_hot / n_cold`.
- `field_particle`, a [`FieldParticleOperator`](@ref) on the grid of `operator`, adds the
  momentum-conserving field-particle term to `C`.

The test-particle operator gives 0.296 of the Lorentz conductivity at `Z = 1`, against the
0.5816 of Spitzer and Härm, since it does not return the momentum of the electron-electron
collisions.
"""
function kinetic_conductivity(operator::AdvectionDiffusion{T}; pitch_rule::Symbol=:linear,
                              density=nothing, field_particle=nothing) where {T}
    source = operator.coefficient_source[]
    source === nothing &&
        throw(ArgumentError("the operator does not know its plasma: build it with fokker_planck_operator"))
    plasma = source.plasma
    get(source, :electric_field, true) && any(!iszero, plasma.electric_field) &&
        throw(ArgumentError("the coefficients of the operator hold an electric field: the linear response needs " *
                            "the operator without it (electric_field = false)"))
    (any(!iszero, operator.advection_r) || any(!iszero, operator.diffusion_r_r)) &&
        throw(ArgumentError("the operator holds radial transport: the linear response is that of the operator in " *
                            "momentum space, radius by radius"))
    any(!iszero, operator.unit_field_advection_p) || set_unit_field_coefficients!(operator)
    # DREAM's flux boundaries at p_max give the last momentum cell no net flux in the steady
    # state, which makes the bordered systems singular: f = 0 beyond p_max in
    # their place, with the coefficient arrays shared
    operator.p_upper_boundary in (:constant_flux, :extrapolated_flux, :extrapolated_flux_consistent, :interface) &&
        (operator = _with_upper_boundary(operator, :f_zero))
    grid = operator.grid
    np, nxi, nr = size(grid)
    cells = np * nxi * nr
    target = density === nothing ? plasma.cold_electron_density : density
    length(target) == nr || throw(DimensionMismatch("the density needs one value per radius"))
    index = LinearIndices((np, nxi, nr))
    # the relativistic Maxwellian at the cold temperature, with density 1: the shape of the
    # source and the state at which a limiter is frozen
    maxwellian = zeros(T, np, nxi, nr)
    for ir in 1:nr
        theta = plasma.cold_electron_temperature[ir] / Constants.ELECTRON_REST_ENERGY
        m = momentum_grid(grid, ir)
        for j in 1:nxi, (i, p) in enumerate(m.p_centers)
            maxwellian[i, j, ir] = exp(-p^2 / (sqrt(1 + p^2) + 1) / theta)
        end
    end
    _scale_to_density!(maxwellian, grid, ones(T, nr))
    limiter_state = is_linear(operator) ? nothing : maxwellian
    residual! = (df, f) -> (fill!(df, zero(eltype(df))); apply!(df, operator, f; limiter_state))
    collisions = jacobian(SparseJacobian(residual!, grid; backend=sparse_ad_backend(pattern=stencil_pattern(operator))),
                          zeros(T, np, nxi, nr))
    any(!iszero, collisions.nzval) || throw(ArgumentError("the operator has no coefficients"))
    # the border: a source column and a density row per radius
    rows, columns, values = findnz(collisions)
    for ir in 1:nr
        m = momentum_grid(grid, ir)
        jacobian_centers = phase_space_jacobian(grid, ir)
        for j in 1:nxi, i in 1:np
            k = index[i, j, ir]
            append!(rows, (k, cells + ir))
            append!(columns, (cells + ir, k))
            append!(values, (maxwellian[i, j, ir],
                             jacobian_centers[i, j] * m.p_widths[i] * m.xi_widths[j] / grid.radial_grid.spatial_jacobian[ir]))
        end
    end
    # the field-particle term, of rank `modes` per radius
    if field_particle !== nothing
        field_particle.grid === grid || _same_momentum_grids(field_particle.grid, grid) ||
            throw(DimensionMismatch("the field-particle operator lives on another grid"))
        per_radius = np * nxi
        for ir in 1:nr
            block = field_particle.restoring_shapes[ir] * transpose(field_particle.loss_functionals[ir])
            offset = (ir - 1) * per_radius
            for column in 1:per_radius, row in 1:per_radius
                block[row, column] == 0 && continue
                push!(rows, offset + row)
                push!(columns, offset + column)
                push!(values, block[row, column])
            end
        end
    end
    factors = lu(sparse(rows, columns, values, cells + nr, cells + nr))
    equilibrium = reshape((factors \ vcat(zeros(T, cells), T.(target)))[1:cells], np, nxi, nr)
    # the derivative of the operator with respect to the field at zero field, applied to it
    D = ForwardDiff.Dual{FieldResponseTag,T,1}
    field = fill(D(zero(T), ForwardDiff.Partials((one(T),))), nr)
    driven = zeros(D, np, nxi, nr)
    apply!(driven, operator, equilibrium; electric_field=field, limiter_state)
    response = factors \ vcat(-vec(ForwardDiff.partials.(driven, 1)), zeros(T, nr))
    return current_density(grid, reshape(response[1:cells], np, nxi, nr); pitch_rule)
end

# the operator with another boundary condition at p_max, sharing every coefficient array
_with_upper_boundary(op::AdvectionDiffusion{T}, boundary::Symbol) where {T} =
    AdvectionDiffusion{T}(op.grid, op.advection_p, op.diffusion_p_p, op.diffusion_p_xi, op.advection_xi,
                          op.diffusion_xi_xi, op.diffusion_xi_p, op.advection_r, op.diffusion_r_r,
                          op.unit_field_advection_p, op.unit_field_advection_xi, op.p_flux_at_lower_boundary,
                          op.advection_p_times_p2_at_zero, op.p_scheme, op.xi_scheme, op.r_scheme,
                          op.p_lower_boundary, boundary, op.r_boundary, op.momentum_grids, op.radial_overlaps,
                          op.jacobian_centers, op.jacobian_p_faces, op.jacobian_xi_faces, op.jacobian_r_faces,
                          op.jacobian_over_p2_at_zero, op.coefficient_source)

function _scale_to_density!(f, grid::KineticGrid, target)
    n = Runes2P.density(grid, f)
    for ir in axes(f, 3)
        view(f, :, :, ir) .*= target[ir] / n[ir]
    end
    return f
end

"""
    ConductivityModel(plasma; reference=:braams_karney, kinetic=:none, operator=nothing,
                      pitch_rule=:linear)

The conductivity `σ(r, t) = σ_reference − σ_kinetic` of a term `σ E` in the equation of
the ohmic current ([`ConductivityTerm`](@ref)). `plasma` is a [`PlasmaState`](@ref) or a
function of time that returns one. `model(t)` gives the conductivity per radius.

- `reference = :braams_karney`: [`braams_karney_conductivity`](@ref), which is DREAM's
  conductivity of Sauter and Redl in cylindrical geometry.
- `kinetic = :none`: nothing is subtracted, for Ohm's law of a fluid, `j_ohm = σ E`.
- `kinetic = :dream_fit`: DREAM's corrected conductivity of the fully kinetic mode. The
  ohmic current is the current of the distribution plus `E (σ − F σ)`, with the fit `F` of
  the conductivity of the kinetic operator
  ([`test_particle_conductivity_fraction`](@ref)).
- `kinetic = :discrete_operator`: the same with the conductivity of the discrete operator
  of the run in place of the fit ([`kinetic_conductivity`](@ref)). `operator` is the
  Fokker-Planck operator of the bulk or its term, and `pitch_rule` that of the current
  moment of the run: a system refuses a model whose rule differs from that of a
  [`CurrentMomentTerm`](@ref) in the same equation. The steady state of a small
  field then has the conductivity of the reference, whatever the grid and the collision
  settings are. On two grids the operator is that of the hot-tail grid. The equilibrium has
  the cold density of the plasma ([`kinetic_conductivity`](@ref)). With a plasma that changes
  in time the operator is a [`PrescribedFokkerPlanckTerm`](@ref), and every new time costs
  a sparse factorization of the bordered operator per evaluation (not covered by a test
  yet). The field-particle operator is not an option here: a system with it
  takes no correction.

The correction is a statement on the steady state. While the distribution relaxes to the
shape of the ohmic one, within collision times after a change of the field, the current of
the distribution is below its steady value and the sum is below `σ E`.

A run with the momentum-conserving field-particle term has the conductivity of
Spitzer in its distribution and takes no correction. The derivatives of the conductivity
with respect to the temperature and the densities are not part of the Jacobian: the plasma
is prescribed.
"""
struct ConductivityModel{P,O}
    plasma::P
    reference::Symbol
    kinetic::Symbol
    operator::O
    pitch_rule::Symbol
end
function ConductivityModel(plasma; reference::Symbol=:braams_karney, kinetic::Symbol=:none, operator=nothing,
                           pitch_rule::Symbol=:linear)
    reference === :braams_karney || throw(ArgumentError("reference must be :braams_karney, not $reference"))
    kinetic in (:none, :dream_fit, :discrete_operator) ||
        throw(ArgumentError("kinetic must be :none, :dream_fit or :discrete_operator, not $kinetic"))
    if kinetic === :discrete_operator
        operator isa Union{AdvectionDiffusion,FokkerPlanckTerm,PrescribedFokkerPlanckTerm} ||
            throw(ArgumentError("kinetic = :discrete_operator needs the Fokker-Planck operator of the bulk or its term"))
    else
        operator === nothing || throw(ArgumentError("an operator belongs to kinetic = :discrete_operator"))
    end
    return ConductivityModel(plasma, reference, kinetic, operator, pitch_rule)
end

_plasma_at(plasma::PlasmaState, t) = plasma
_plasma_at(plasma_at, t) = plasma_at(t)

# an operator rebuilt from a plasma of its own follows the time with a constant plasma of
# the model
is_time_dependent(model::ConductivityModel) =
    !(model.plasma isa PlasmaState) || model.operator isa PrescribedFokkerPlanckTerm

function (model::ConductivityModel)(t)
    plasma = _plasma_at(model.plasma, t)
    nr = radial_cell_count(plasma)
    sigma = [braams_karney_conductivity(plasma, ir) for ir in 1:nr]
    if model.kinetic === :dream_fit
        for ir in 1:nr
            sigma[ir] *= 1 - test_particle_conductivity_fraction(effective_charge(plasma, ir))
        end
    elseif model.kinetic === :discrete_operator
        sigma .-= kinetic_conductivity(_operator_at(model.operator, t); pitch_rule=model.pitch_rule)
    end
    return sigma
end

"""
    ConductivityTerm(electric_field, model::ConductivityModel; coefficient=1)

`coefficient` times `σ(r, t) E(r)` with the conductivity of `model` and the radial unknown
`electric_field`, added to the row of a radial unknown with one component: Ohm's law of a
fluid, `0 = σ E − j_ohm`, or the correction of the ohmic current of a fully kinetic run,
`0 = j(f, all p) − j_hot + E (σ − σ_kinetic) − j_ohm` (DREAM's `j_ohm` with
`correctedConductivity`). The conductivity is that of the time of the evaluation and is
kept between the evaluations of a time. The term is linear in the field.
"""
struct ConductivityTerm{M<:ConductivityModel} <: EquationTerm
    electric_field::Symbol
    model::M
    coefficient::Float64
    values::Vector{Float64}
    time::Base.RefValue{Float64}
end
ConductivityTerm(electric_field::Symbol, model::ConductivityModel; coefficient::Real=1.0) =
    ConductivityTerm(electric_field, model, Float64(coefficient), Float64[], Ref(NaN))

dependencies(term::ConductivityTerm) = [term.electric_field]
is_linear(::ConductivityTerm) = true

validate(term::ConductivityTerm, layout, row) = validate(term, layout, row, 0.0)

# the conductivity at the initial time of the system: a plasma of the model may be
# defined on the interval of the run alone
function validate(term::ConductivityTerm, layout, row, t)
    id = unknown_id(layout, term.electric_field)
    layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
        throw(ArgumentError("the electric field of a ConductivityTerm is a radial unknown with one component"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("ConductivityTerm adds to a radial unknown with one component"))
    length(conductivity_at(term, t)) == radial_cell_count(layout) ||
        throw(DimensionMismatch("the plasma of the conductivity has other radii than the layout"))
    return nothing
end

"""
    conductivity_at(term::ConductivityTerm, t)

The conductivity of the term per radius at the time `t` [S/m].
"""
function conductivity_at(term::ConductivityTerm, t)
    time = Float64(_value(t))
    if isnan(term.time[]) || (term.time[] != time && is_time_dependent(term.model))
        values = term.model(time)
        resize!(term.values, length(values))
        term.values .= values
        term.time[] = time
    end
    return term.values
end

function rhs!(out, term::ConductivityTerm, u, layout, row, t)
    sigma = conductivity_at(term, t)
    field = unknown_id(layout, term.electric_field)
    @inbounds for ir in 1:radial_cell_count(layout)
        out[first(unknown_range(layout, row, ir))] +=
            term.coefficient * sigma[ir] * u[first(unknown_range(layout, field, ir))]
    end
    return out
end

function sparsity!(entries, term::ConductivityTerm, layout, row)
    field = unknown_id(layout, term.electric_field)
    for ir in 1:radial_cell_count(layout)
        push!(entries, (first(unknown_range(layout, row, ir)), first(unknown_range(layout, field, ir))))
    end
    return entries
end

# A correction of the conductivity and the field-particle term are two ways to
# the conductivity of Spitzer: both in one system count the missing part twice. The
# conductivity of the discrete operator is that of a current moment with its pitch rule,
# and the rules of the two differ by a factor of the pitch grid.
function _check_conductivity_correction(equations)
    terms = [term for equation in equations for term in equation.terms]
    corrected = any(term -> term isa ConductivityTerm && term.model.kinetic !== :none, terms)
    corrected && any(term -> term isa FieldParticleTerm, terms) &&
        throw(ArgumentError("the system has a corrected conductivity and the field-particle term: the term gives " *
                            "the distribution the conductivity of Spitzer, and the correction would add the " *
                            "missing part again. Take one of the two"))
    for equation in equations, term in equation.terms
        term isa ConductivityTerm && term.model.kinetic === :discrete_operator || continue
        for moment in equation.terms
            moment isa CurrentMomentTerm && moment.pitch_rule !== term.model.pitch_rule &&
                throw(ArgumentError("the conductivity model of $(equation.unknown) takes the pitch rule " *
                                    "$(term.model.pitch_rule) and the current moment of $(moment.kinetic) in the " *
                                    "same equation $(moment.pitch_rule): give the model the rule of the moment " *
                                    "(ConductivityModel(...; pitch_rule = :$(moment.pitch_rule)))"))
        end
    end
    return nothing
end
