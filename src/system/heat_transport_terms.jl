# Heat transport of the cold electrons: radial diffusion of the
# temperature, `(1/V') ∂_r (V' χ ∂_r T)` in the row of the temperature (which holds the time
# derivative of the energy W_cold, in W/m³), with a prescribed diffusivity or the
# Rechester-Rosenbluth heat diffusivity of stochastic field lines (DREAM's
# HeatTransportDiffusion and HeatTransportRechesterRosenbluth).

"""
    HeatTransportTerm(; diffusion=nothing, diffusion_at=nothing, delta_b_over_b=nothing,
                      delta_b_over_b_at=nothing, major_radius=nothing, safety_factor=1.0,
                      temperature=:cold_electron_temperature, density=:cold_electron_density,
                      edge=:zero_flux, coefficient=1)

Radial heat diffusion of the cold electrons, `(1/V') ∂_r (V' χ ∂_r T)` [W/m³] with `T` in
eV, for the row of their temperature. The flux through the face between two cells is
`V' χ (T_k − T_{k−1})/Δr` with the distance `Δr` of the two centres, and `χ` at the face
takes the density and the temperature interpolated linearly between the two cells
(DREAM's `deltaRadialFlux`). No heat crosses the axis. One of two models:

- a diffusivity `D` [m²/s], `χ = 3/2 e D n` (DREAM's `HeatTransportDiffusion`): `diffusion`
  a radial profile ([`radial_values`](@ref) on the faces), or `diffusion_at` a function of
  time that returns one;
- the heat diffusivity of Rechester and Rosenbluth in stochastic field lines (DREAM's
  `HeatTransportRechesterRosenbluth`, cylinder),
  ```math
  χ = 3\\sqrt{2π}\\, e\\, c\\, q R_0 (δB/B)^2\\, n \\sqrt{Θ} (1 − 5Θ/8),\\qquad Θ = T/m_ec^2,
  ```
  the energy moment of the kinetic diffusion `π q R₀ |v_∥| (δB/B)²` of
  [`add_rechester_rosenbluth_diffusion!`](@ref) on a Maxwellian: `delta_b_over_b` a
  profile, or `delta_b_over_b_at` a function of time such as the
  [`PiecewiseConstant`](@ref) transport history of a [`RechesterRosenbluthUpdate`](@ref).
  The same object in both keeps the kinetic and the heat transport on one level. The
  major radius `R₀` and the safety factor `q` are inputs, as for the kinetic operator.

Its switch times must be time points of the run ([`forced_time_points`](@ref)), so that a
step does not straddle a change.

`edge` is the condition at `r = a`:

- `:zero_flux` (default, DREAM's `BC_CONSERVATIVE`, the setting of DREAM's disruption
  examples);
- `:absorbing`: `T = 0` on the edge face, over the distance from the last centre to the
  face;
- `:absorbing_dream`: DREAM's `BC_F_0`, `T = 0` at the distance of the last two centres;
- `:constant_gradient_dream`: DREAM's `BC_DF_CONST`, the gradient of the last two cells
  continued one cell outside (two radii at least).
"""
struct HeatTransportTerm{D,Q} <: EquationTerm
    temperature::Symbol
    density::Symbol
    model::Symbol                        # :prescribed or :rechester_rosenbluth
    level::D                             # D or δB/B: a profile, or a function of time
    level_in_time::Bool
    safety_factor::Q
    major_radius::Float64
    edge::Symbol
    coefficient::Float64
    faces::Base.RefValue{Vector{Float64}}   # the level on the faces, at `time`
    time::Base.RefValue{Float64}
    safety_factor_faces::Base.RefValue{Vector{Float64}}   # q on the faces, evaluated once
end

function HeatTransportTerm(; diffusion=nothing, diffusion_at=nothing, delta_b_over_b=nothing,
                           delta_b_over_b_at=nothing, major_radius::Union{Nothing,Real}=nothing,
                           safety_factor=1.0, temperature::Symbol=:cold_electron_temperature,
                           density::Symbol=:cold_electron_density, edge::Symbol=:zero_flux, coefficient::Real=1.0)
    given = [x !== nothing for x in (diffusion, diffusion_at, delta_b_over_b, delta_b_over_b_at)]
    count(given) == 1 ||
        throw(ArgumentError("give one of diffusion, diffusion_at, delta_b_over_b and delta_b_over_b_at"))
    edge in (:zero_flux, :absorbing, :absorbing_dream, :constant_gradient_dream) ||
        throw(ArgumentError("edge must be :zero_flux, :absorbing, :absorbing_dream or :constant_gradient_dream, " *
                            "not $edge"))
    model = given[1] || given[2] ? :prescribed : :rechester_rosenbluth
    if model === :rechester_rosenbluth
        major_radius === nothing && throw(ArgumentError("the Rechester-Rosenbluth model needs the major radius"))
        major_radius > 0 || throw(ArgumentError("the major radius must be positive"))
    end
    level = something(diffusion, diffusion_at, delta_b_over_b, delta_b_over_b_at)
    return HeatTransportTerm(temperature, density, model, level, given[2] || given[4], safety_factor,
                             major_radius === nothing ? NaN : Float64(major_radius), edge, Float64(coefficient),
                             Ref(Float64[]), Ref(NaN), Ref(Float64[]))
end

dependencies(term::HeatTransportTerm) = unique([term.temperature, term.density])
forced_time_points(term::HeatTransportTerm) = term.level_in_time ? forced_time_points(term.level) : Float64[]

function validate(term::HeatTransportTerm, layout, row)
    for id in (unknown_id(layout, term.temperature), unknown_id(layout, term.density), row)
        layout.kinds[id] === :radial && layout.sizes[id, 1] == 1 ||
            throw(ArgumentError("HeatTransportTerm takes and adds to radial unknowns with one component, not " *
                                "$(layout.unknowns[id].name)"))
    end
    term.edge === :constant_gradient_dream && radial_cell_count(layout) < 2 &&
        throw(ArgumentError("the edge :constant_gradient_dream needs two radii at least, as in DREAM"))
    term.level_in_time || _heat_level_faces(term, layout.radial_grid, 0.0)
    return nothing
end

# the level (D or δB/B) on the faces at time t: once for a profile, once per time for a
# function of time
function _heat_level_faces(term::HeatTransportTerm, rg::RadialGrid, t)
    if term.level_in_time
        term.time[] == t && !isempty(term.faces[]) && return term.faces[]
        values = collect(Float64, radial_values(term.level(t), rg.r_faces))
        term.time[] = t
    else
        isempty(term.faces[]) || return term.faces[]
        values = collect(Float64, radial_values(term.level, rg.r_faces))
    end
    all(>=(0), values) || throw(ArgumentError("the level of the heat transport must not be negative"))
    term.faces[] = values
    return values
end

# the safety factor on the faces, evaluated once: a profile is never evaluated inside the
# Newton iteration
function _heat_safety_factor_faces(term::HeatTransportTerm, rg::RadialGrid)
    isempty(term.safety_factor_faces[]) &&
        (term.safety_factor_faces[] = collect(Float64, radial_values(term.safety_factor, rg.r_faces)))
    return term.safety_factor_faces[]
end

# χ at a face with the level `level`, the density n and the temperature T [eV] there
function _heat_conductivity(term::HeatTransportTerm, level, q, n, T)
    term.model === :prescribed && return 1.5 * Constants.ELEMENTARY_CHARGE * level * n
    theta = T / Constants.ELECTRON_REST_ENERGY
    return 3 * sqrt(2π) * Constants.ELEMENTARY_CHARGE * Constants.SPEED_OF_LIGHT * q * term.major_radius *
           level^2 * n * sqrt(theta) * (1 - 5 * theta / 8)
end

# V' times the heat flux χ ∂_r T through face k (1 the axis, nr + 1 the edge)
function _heat_flux(term::HeatTransportTerm, rg::RadialGrid, levels, q, T, n, k)
    nr = radial_cell_count(rg)
    zero_flux = zero(eltype(T)) * zero(eltype(n))
    k == 1 && return zero_flux
    if k <= nr
        distance = rg.r_centers[k] - rg.r_centers[k-1]
        delta = rg.r_widths[k-1] / 2 / distance
        n_face = delta * n[k] + (1 - delta) * n[k-1]
        T_face = delta * T[k] + (1 - delta) * T[k-1]
        return rg.spatial_jacobian_faces[k] * _heat_conductivity(term, levels[k], q[k], n_face, T_face) *
               (T[k] - T[k-1]) / distance
    end
    term.edge === :zero_flux && return zero_flux
    if term.edge === :absorbing
        distance, outside = rg.r_faces[nr+1] - rg.r_centers[nr], zero(eltype(T))
    else
        distance = nr > 1 ? rg.r_centers[nr] - rg.r_centers[nr-1] : one(eltype(rg.r_centers))
        outside = if term.edge === :absorbing_dream
            zero(eltype(T))
        else
            ratio = rg.r_widths[nr] / rg.r_widths[nr-1]
            (1 + ratio) * T[nr] - ratio * T[nr-1]
        end
    end
    return rg.spatial_jacobian_faces[nr+1] * _heat_conductivity(term, levels[nr+1], q[nr+1], n[nr], T[nr]) *
           (outside - T[nr]) / distance
end

function rhs!(out, term::HeatTransportTerm, u, layout, row, t)
    rg = layout.radial_grid
    levels = _heat_level_faces(term, rg, t)
    q = _heat_safety_factor_faces(term, rg)
    T = radial_values(u, layout, unknown_id(layout, term.temperature))
    n = radial_values(u, layout, unknown_id(layout, term.density))
    for ir in 1:radial_cell_count(rg)
        divergence = (_heat_flux(term, rg, levels, q, T, n, ir + 1) - _heat_flux(term, rg, levels, q, T, n, ir)) /
                     (rg.spatial_jacobian[ir] * rg.r_widths[ir])
        out[first(unknown_range(layout, row, ir))] += term.coefficient * divergence
    end
    return out
end

# a row couples the temperature and the density of its radius and of both neighbours
function sparsity!(entries, term::HeatTransportTerm, layout, row)
    nr = radial_cell_count(layout)
    for ir in 1:nr, name in dependencies(term), neighbour in max(ir - 1, 1):min(ir + 1, nr)
        push!(entries, (first(unknown_range(layout, row, ir)), first(unknown_range(layout, unknown_id(layout, name), neighbour))))
    end
    return entries
end

"""
    heat_flux_through_edge(term::HeatTransportTerm, u, layout, t)

The heat that leaves through the edge of the plasma [W per unit of `∫ V' dr`, as the
volume integral of the term], `−V' χ ∂_r T` at `r = a`: zero with `edge = :zero_flux`.
The volume integral of the term is minus this flux.
"""
function heat_flux_through_edge(term::HeatTransportTerm, u, layout, t)
    rg = layout.radial_grid
    T = radial_values(u, layout, unknown_id(layout, term.temperature))
    n = radial_values(u, layout, unknown_id(layout, term.density))
    return -term.coefficient * _heat_flux(term, rg, _heat_level_faces(term, rg, t),
                                          _heat_safety_factor_faces(term, rg), T, n,
                                          radial_cell_count(rg) + 1)
end
