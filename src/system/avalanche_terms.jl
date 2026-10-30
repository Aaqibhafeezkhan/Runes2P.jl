# The Rosenbluth-Putvinski avalanche of a hot-tail grid without a runaway grid: DREAM's
# kinetic source of secondaries on f (AVALANCHE_MODE_KINETIC,
# src/Equations/Kinetic/AvalancheSourceRP.cpp) from a fluid runaway density above the grid,
# and the number of electrons it creates, which the particle source of the grid gives back.

"""
    rosenbluth_putvinski_rate(p_lower, p_upper=Inf)

The rate of the Rosenbluth-Putvinski avalanche source between the momenta `p_lower` and
`p_upper` (units of `mₑc`) per runaway and per electron of the total density, in m³/s:

```math
\\frac{1}{n_\\mathrm{re} n_\\mathrm{tot}} \\int_{p_\\mathrm{lower}}^{p_\\mathrm{upper}} S\\,\\mathrm{d}^3p
= 2π r_0^2 c \\left(\\frac{1}{γ_\\mathrm{lower} − 1} − \\frac{1}{γ_\\mathrm{upper} − 1}\\right),
```

the knock-ons of a runaway at the speed of light on electrons at rest: the Møller cross section
in the limit of an infinitely fast primary, `2π r₀²/(γ − 1)²` per unit `γ` (M. N.
Rosenbluth and S. V. Putvinski, Nucl. Fusion 37, 1355 (1997), doi:10.1088/0029-5515/37/10/I03,
eq. (4)). DREAM's `AvalancheSourceRP::EvaluateNormalizedTotalKnockOnNumber`
(`src/Equations/Kinetic/AvalancheSourceRP.cpp:96-115`), with its prefactor `e⁴/((4πε₀ mₑ
c)² c) = r₀² c`. Infinite at `p_lower = 0`.
"""
function rosenbluth_putvinski_rate(p_lower::Real, p_upper::Real = Inf)
    p_lower == 0 && return Inf
    inverse_energy(p) = isinf(p) ? 0.0 : (sqrt(1 + p^2) + 1) / p^2    # 1/(γ − 1)
    return 2π * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT *
           (inverse_energy(p_lower) - inverse_energy(p_upper))
end

"""
    RosenbluthPutvinskiSourceTerm(kinetic, grid; runaway_density=:runaway_density,
                                  total_density=:total_electron_density,
                                  electric_field=:electric_field, p_cutoff=0.01)

DREAM's kinetic Rosenbluth-Putvinski avalanche source (`AVALANCHE_MODE_KINETIC`,
`src/Equations/Kinetic/AvalancheSourceRP.cpp`, placed by `src/Settings/Equations/f_hot.cpp:116-127`)
in the equation of the kinetic unknown `kinetic` on `grid`: the knock-on electrons that a
fluid runaway density `n_re` above the grid creates on the electrons of the total density
`n_tot` (free and bound), at every momentum cell above `p_cutoff` and at the pitch
`ξ* = ±√((γ − 1)/(γ + 1))` that the momentum of a knock-on from a primary at the speed of
light fixes, in the cell that holds it:

```math
S_{ij} = n_\\mathrm{re} n_\\mathrm{tot}\\, r_0^2 c\\,
\\frac{1}{Δp_i}\\left(\\frac{1}{γ_{i−1/2}^* − 1} − \\frac{1}{γ_{i+1/2} − 1}\\right)
\\frac{2π V'}{\\mathcal{V}'_{ij} Δξ_j},
```

with `γ*_{i−1/2}` at the larger of the lower face and `p_cutoff` and `Δp_i` the whole cell.
Summed over the grid it is [`rosenbluth_putvinski_rate`](@ref)`(p_cutoff, p_max)` times
`n_re n_tot` exactly. The pitch is that of DREAM in a cylinder (`B_max = B_min`,
`fvm/Grid/FluxSurfaceAverager.avalancheDelta.cpp:114-140`): `ξ*` takes the sign of the
electric field of each radius (`RP_SOURCE_PITCH_ADAPTIVE`), or `+` without
`electric_field`, and a `ξ*` on a cell face gives no cell (DREAM's strict inequalities). The
term is bilinear in `n_re` and `n_tot`; the sign of the field enters without a derivative,
as in DREAM. The Jacobian is assembled directly.

The source creates electrons, which the particle source of the grid takes out of the cold
electrons: DREAM's row of the particle source, `S = dn_free/dt − internal − external`, holds
the internal number with the factor `−1` (`TotalElectronDensityFromKineticAvalanche`,
`f_hot.cpp:244-255, 362-374`). Written as `dn_free/dt = S + internal + external`
(`TimeDerivative(:free_electron_density)`), the term is
[`ProductTerm`](@ref)`(total_density, runaway_density; coefficient = +rosenbluth_putvinski_rate(p_cutoff, p_max))`. The value
`p_cutoff = 0.01` is that of DREAM's disruption example (`examples/Disruptions/generate.py:329`).
"""
struct RosenbluthPutvinskiSourceTerm{T} <: EquationTerm
    kinetic::Symbol
    grid::KineticGrid{T}
    runaway_density::Symbol
    total_density::Symbol
    electric_field::Union{Nothing, Symbol}
    p_cutoff::Float64
    positive::Vector{Matrix{T}}   # per radius: S/(n_re n_tot) [m³/s per phase-space volume] with ξ* > 0
    negative::Vector{Matrix{T}}   # the same with ξ* < 0
end

function RosenbluthPutvinskiSourceTerm(
        kinetic::Symbol, grid::KineticGrid{T}; runaway_density::Symbol = :runaway_density,
        total_density::Symbol = :total_electron_density,
        electric_field::Union{Nothing, Symbol} = :electric_field,
        p_cutoff::Real = 0.01) where {T}
    p_cutoff > 0 ||
        throw(ArgumentError("the cutoff of the Rosenbluth-Putvinski source must be positive"))
    rg = grid.radial_grid
    positive, negative = Matrix{T}[], Matrix{T}[]
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        V = phase_space_jacobian(grid, ir)
        shapes = (zeros(T, momentum_cell_count(m), pitch_cell_count(m)),
            zeros(T, momentum_cell_count(m), pitch_cell_count(m)))
        for i in 1:momentum_cell_count(m)
            lower, upper = m.p_faces[i], m.p_faces[i + 1]
            upper <= p_cutoff && continue
            momentum_part = rosenbluth_putvinski_rate(max(lower, p_cutoff), upper) /
                            (2π * (upper - lower))
            gamma = sqrt(1 + m.p_centers[i]^2)
            star = sqrt((gamma - 1) / (gamma + 1))
            for (shape, sign) in zip(shapes, (1, -1)), j in 1:pitch_cell_count(m)

                m.xi_faces[j] < sign * star < m.xi_faces[j + 1] || continue
                shape[i, j] = momentum_part * 2π * rg.spatial_jacobian[ir] /
                              (V[i, j] * m.xi_widths[j])
            end
        end
        push!(positive, shapes[1])
        push!(negative, shapes[2])
    end
    return RosenbluthPutvinskiSourceTerm(
        kinetic, grid, runaway_density, total_density, electric_field, Float64(p_cutoff),
        positive, negative)
end

function dependencies(term::RosenbluthPutvinskiSourceTerm)
    if term.electric_field === nothing
        return [term.runaway_density, term.total_density]
    else
        return [term.runaway_density, term.total_density, term.electric_field]
    end
end
is_linear(::RosenbluthPutvinskiSourceTerm) = false
jacobian_assembly(::RosenbluthPutvinskiSourceTerm) = :direct

function validate(term::RosenbluthPutvinskiSourceTerm, layout, row)
    id = unknown_id(layout, term.kinetic)
    id == row ||
        throw(ArgumentError("RosenbluthPutvinskiSourceTerm adds to the equation of its kinetic unknown $(term.kinetic)"))
    layout.kinds[id] === :kinetic ||
        throw(ArgumentError("RosenbluthPutvinskiSourceTerm needs a kinetic unknown"))
    _same_momentum_grids(term.grid, kinetic_grid(layout, id)) ||
        throw(DimensionMismatch("the grid of the Rosenbluth-Putvinski source does not match $(term.kinetic)"))
    for name in dependencies(term)
        k = unknown_id(layout, name)
        if !(layout.kinds[k] === :radial && layout.sizes[k, 1] == 1)
            throw(ArgumentError("the Rosenbluth-Putvinski source reads $name, a radial unknown with one component"))
        end
    end
    return nothing
end

# the shape of radius ir with the pitch of the sign of the field
function _rp_shape(term::RosenbluthPutvinskiSourceTerm, u, layout, ir)
    term.electric_field === nothing && return term.positive[ir]
    E = unknown_values(u, layout, unknown_id(layout, term.electric_field))[ir]
    return E >= 0 ? term.positive[ir] : term.negative[ir]
end

function rhs!(out, term::RosenbluthPutvinskiSourceTerm, u, layout, row, t)
    n_re = unknown_values(u, layout, unknown_id(layout, term.runaway_density))
    n_tot = unknown_values(u, layout, unknown_id(layout, term.total_density))
    for ir in 1:radial_cell_count(layout)
        kinetic_view(out, layout, row, ir) .+= n_re[ir] * n_tot[ir] .*
                                               _rp_shape(term, u, layout, ir)
    end
    return out
end

function add_jacobian!(J, term::RosenbluthPutvinskiSourceTerm, u, layout, row, t)
    re, tot = unknown_id(layout, term.runaway_density),
    unknown_id(layout, term.total_density)
    n_re, n_tot = unknown_values(u, layout, re), unknown_values(u, layout, tot)
    for ir in 1:radial_cell_count(layout)
        shape = _rp_shape(term, u, layout, ir)
        c_re, c_tot = first(unknown_range(layout, re, ir)),
        first(unknown_range(layout, tot, ir))
        for j in axes(shape, 2), i in axes(shape, 1)

            shape[i, j] == 0 && continue
            r = _kinetic_index(layout, row, ir, i, j)
            _add_entry!(J, r, c_re, n_tot[ir] * shape[i, j])
            _add_entry!(J, r, c_tot, n_re[ir] * shape[i, j])
        end
    end
    return J
end

# both pitches, so that the pattern holds whatever the sign of the field
function sparsity!(entries, term::RosenbluthPutvinskiSourceTerm, layout, row)
    re, tot = unknown_id(layout, term.runaway_density),
    unknown_id(layout, term.total_density)
    for ir in 1:radial_cell_count(layout)
        columns = (
            first(unknown_range(layout, re, ir)), first(unknown_range(layout, tot, ir)))
        for shape in (term.positive[ir], term.negative[ir]), j in axes(shape, 2),
            i in axes(shape, 1)
            shape[i, j] == 0 && continue
            r = _kinetic_index(layout, row, ir, i, j)
            push!(entries, (r, columns[1]), (r, columns[2]))
        end
    end
    return entries
end

"""
    AvalancheRateCache

The last evaluation of the matched rate of [`ExternalAvalancheTerm`](@ref) at one radius,
shared by the rows that hold the term and by the residual and the Jacobian at one state:
the local values `x` and the time `t` of the entry, the value of the rate times `n_re`, and
its derivatives by `x` once the Jacobian needed them. One entry per radius, so that a loop
over radii in threads writes separate entries. The entries take a state in `Float64`; a
state in another number type is evaluated without the cache.
"""
mutable struct AvalancheRateCache
    x::Vector{Float64}          # the local values (n_re, n_tot, E, T, n_cold, charge states) of the entry
    t::Float64
    value::Float64              # the rate times n_re
    has_value::Bool
    gradient::Vector{Float64}   # the derivatives of the value by x, empty until the Jacobian needs them
    evaluations::Int            # the solutions of the effective critical field so far
end
AvalancheRateCache() = AvalancheRateCache(Float64[], NaN, NaN, false, Float64[], 0)

"""
    ExternalAvalancheTerm(plasma; p_cutoff, runaway_density=:runaway_density,
                          total_density=:total_electron_density, electric_field=:electric_field,
                          temperature=:cold_electron_temperature, density=:cold_electron_density,
                          ions=nothing, settings=CollisionSettings(), exponent=-2)

DREAM's `ExternalAvalancheTerm` (`src/Equations/Fluid/ExternalAvalancheTerm.cpp`, placed by
`src/Settings/RunawaySourceTerms.cpp:36-58`) in the equation of the fluid runaway density
`n_re` above a hot-tail grid without a runaway grid: `dn_re/dt = Γ n_re` with

```math
Γ = \\left(Γ_\\mathrm{ext}^{a} + Γ_\\mathrm{fluid}^{a}\\right)^{1/a} \\text{ for } Γ_\\mathrm{fluid} > 0,
\\qquad Γ = Γ_\\mathrm{fluid} \\text{ otherwise},
```

`a = exponent = −2` (DREAM's value), the Rosenbluth-Putvinski rate of the knock-ons above
`p_cutoff`, `Γ_ext = n_tot` [`rosenbluth_putvinski_rate`](@ref)`(p_cutoff)`, and DREAM's fluid
growth rate with the effective critical field, [`fluid_avalanche_growth_rate`](@ref). The
match makes the external rate, positive at every field, follow the fluid rate below the
effective critical field, where it decays. DREAM takes `p_cutoff` as the upper momentum
face of the hot-tail grid, `p_max` (`RunawaySourceTerms.cpp:44-46`), not the cutoff of the
kinetic source ([`RosenbluthPutvinskiSourceTerm`](@ref)), and so must the case. One `p_cutoff`
serves every radius, as in DREAM ("assume same momentum grid at all radii",
`RunawaySourceTerms.cpp:41`): a grid whose `p_max` differs between radii needs a term per radius.

The plasma of the rate is that of `plasma` (a [`PlasmaState`](@ref) or a function of time:
the magnetic field, the composition) with the temperature `temperature`, the cold density
`density` and, with `ions`, the charge states of that radial unknown, as the other fluid terms
read them; `settings` gives the screening, the Coulomb logarithm and the tables of DREAM's
`RunawayFluid` ([`runaway_fluid_settings`](@ref)). The Jacobian is exact: the columns of
`n_re`, `n_tot`, the field, the temperature, the density and the charge states by forward
differentiation through the solutions of the effective critical field and of `p⋆` (one
Newton correction in dual numbers each). DREAM's own Jacobian takes the columns of the field
and of `n_tot` alone, with the approximation `Γ ∝ |E| − E_ceff` (`RunawayFluid.cpp:881-907`).
The solution of the effective critical field takes about 2.4 ms per radius on one core (the D
and Ar of DREAM's disruption example; 2.5 ms with the derivatives). The rate of a radius is
evaluated once per state: the rows that hold the term (the runaway density and the particle
source of a hot-tail grid with fluid runaways) and the residual and the Jacobian at one
state share the evaluation, and the Jacobian takes every partial in one pass of forward
differentiation.
"""
struct ExternalAvalancheTerm{S <: PrescribedPlasma} <: EquationTerm
    plasma::S
    p_cutoff::Float64
    runaway_density::Symbol
    total_density::Symbol
    electric_field::Symbol
    temperature::Symbol
    density::Symbol
    ions::Union{Nothing, Symbol}
    settings::CollisionSettings
    exponent::Float64
    cache::Vector{AvalancheRateCache}   # the last evaluation of each radius
end

function ExternalAvalancheTerm(
        plasma; p_cutoff::Real, runaway_density::Symbol = :runaway_density,
        total_density::Symbol = :total_electron_density, electric_field::Symbol = :electric_field,
        temperature::Symbol = :cold_electron_temperature, density::Symbol = :cold_electron_density,
        ions::Union{Nothing, Symbol} = nothing, settings::CollisionSettings = CollisionSettings(),
        exponent::Real = -2)
    plasma isa PlasmaSource && !(plasma isa PrescribedPlasma) &&
        throw(ArgumentError("ExternalAvalancheTerm takes a prescribed plasma and reads the temperature, the density " *
                            "and the ion densities from unknowns of its own: give it the prescribed part and " *
                            "the keywords temperature, density and ions"))
    p_cutoff > 0 ||
        throw(ArgumentError("the cutoff of the external avalanche must be positive"))
    exponent < 0 ||
        throw(ArgumentError("the exponent of the match must be negative (DREAM: -2)"))
    source = plasma isa PrescribedPlasma ? plasma : PrescribedPlasma(plasma)
    return ExternalAvalancheTerm(
        source, Float64(p_cutoff), runaway_density, total_density, electric_field,
        temperature, density, ions, settings, Float64(exponent), AvalancheRateCache[])
end

function dependencies(term::ExternalAvalancheTerm)
    [term.runaway_density; term.total_density; term.electric_field; term.temperature;
     term.density;
     term.ions === nothing ? Symbol[] : [term.ions]]
end
is_linear(::ExternalAvalancheTerm) = false
jacobian_assembly(::ExternalAvalancheTerm) = :direct

function validate(term::ExternalAvalancheTerm, layout, row, t)
    for name in (term.runaway_density, term.total_density, term.electric_field, term.temperature, term.density)
        k = unknown_id(layout, name)
        if !(layout.kinds[k] === :radial && layout.sizes[k, 1] == 1)
            throw(ArgumentError("ExternalAvalancheTerm reads $name, a radial unknown with one component"))
        end
    end
    if !(layout.kinds[row] === :radial && layout.sizes[row, 1] == 1)
        throw(ArgumentError("ExternalAvalancheTerm adds to a radial unknown with one component"))
    end
    if term.ions !== nothing
        _check_ion_unknown(layout, unknown_id(layout, term.ions),
            IonComposition(plasma_values(term.plasma, nothing, nothing, t)))
    end
    return nothing
end
validate(term::ExternalAvalancheTerm, layout, row) = validate(term, layout, row, 0.0)

# the matched rate and the runaways it acts on, at radius ir, from the local values
# x = (n_re, n_tot, E, T, n_cold, charge states...) of any number type
function _external_avalanche(term::ExternalAvalancheTerm, prescribed::PlasmaState, ir, x)
    n_re, n_tot, E, T, n = x[1], x[2], x[3], x[4], x[5]
    ions = if term.ions === nothing
        #! format: off
        [IonSpecies(s.name, s.nuclear_charge, reshape(s.charge_state_densities[:, ir] .+ zero(T), :, 1))
         for s in prescribed.ions]
        #! format: on
    else
        c = IonComposition(prescribed.ions)
        [IonSpecies(c.names[s], c.nuclear_charges[s], reshape(x[5 .+ species_components(c, s)], :, 1))
         for s in 1:length(c)]
    end
    local_plasma = PlasmaState(;
        cold_electron_temperature = [T], cold_electron_density = [n], ions,
        electric_field = [E], magnetic_field = [prescribed.magnetic_field[ir] + zero(T)])
    fluid, _, _ = fluid_avalanche_growth_rate(local_plasma, 1, n_tot; settings = term.settings)
    fluid > 0 || return fluid * n_re
    external = n_tot * rosenbluth_putvinski_rate(term.p_cutoff)
    a = term.exponent
    return (external^a + fluid^a)^(1 / a) * n_re
end

# the local values of radius ir and the entries of the state they come from
function _external_avalanche_inputs(term::ExternalAvalancheTerm, u, layout, ir)
    names = (term.runaway_density, term.total_density,
        term.electric_field, term.temperature, term.density)
    columns = [first(unknown_range(layout, unknown_id(layout, name), ir)) for name in names]
    if term.ions !== nothing
        append!(columns, unknown_range(layout, unknown_id(layout, term.ions), ir))
    end
    return u[columns], columns
end

# the cache entry of radius ir, grown on demand
function _cache_entry!(term::ExternalAvalancheTerm, ir)
    while length(term.cache) < ir
        push!(term.cache, AvalancheRateCache())
    end
    return term.cache[ir]
end

# The entry of radius ir at the local values x and the time t, with the value of the rate times
# n_re and, with `gradient`, its derivatives by x. An entry that holds the state is reused; else
# the value is evaluated in plain numbers, or, for the derivatives, value and derivatives in one
# pass of forward differentiation with every partial at once. A value evaluated in plain numbers
# is kept when the derivatives follow, so that the residual at a state does not depend on
# whether the Jacobian was taken there.
function _evaluate!(
        term::ExternalAvalancheTerm, prescribed::PlasmaState, ir, x, t; gradient::Bool)
    entry = _cache_entry!(term, ir)
    if !(entry.t == t && entry.x == x)
        entry.x, entry.t, entry.has_value = x, t, false
        empty!(entry.gradient)
    end
    if gradient && isempty(entry.gradient)
        f = y -> _external_avalanche(term, prescribed, ir, y)
        result = ForwardDiff.DiffResults.GradientResult(x)
        ForwardDiff.gradient!(result, f, x, ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{length(x)}()))
        entry.gradient = copy(ForwardDiff.DiffResults.gradient(result))
        if !entry.has_value
            entry.value = ForwardDiff.DiffResults.value(result)
            entry.has_value = true
        end
        entry.evaluations += 1
    elseif !entry.has_value
        entry.value, entry.has_value = _external_avalanche(term, prescribed, ir, x), true
        entry.evaluations += 1
    end
    return entry
end

function rhs!(out, term::ExternalAvalancheTerm, u, layout, row, t)
    prescribed = plasma_values(term.plasma, nothing, nothing, t)
    for ir in 1:radial_cell_count(layout)
        x, _ = _external_avalanche_inputs(term, u, layout, ir)
        value = if x isa Vector{Float64}
            _evaluate!(term, prescribed, ir, x, t; gradient = false).value
        else
            _external_avalanche(term, prescribed, ir, x)    # a state in dual numbers, without the cache
        end
        out[first(unknown_range(layout, row, ir))] += value
    end
    return out
end

function add_jacobian!(J, term::ExternalAvalancheTerm, u, layout, row, t)
    prescribed = plasma_values(term.plasma, nothing, nothing, t)
    for ir in 1:radial_cell_count(layout)
        x, columns = _external_avalanche_inputs(term, u, layout, ir)
        gradient = _evaluate!(term, prescribed, ir, x, t; gradient = true).gradient
        r = first(unknown_range(layout, row, ir))
        for (k, column) in enumerate(columns)
            _add_entry!(J, r, column, gradient[k])
        end
    end
    return J
end

function sparsity!(entries, term::ExternalAvalancheTerm, layout, row)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for name in dependencies(term)
            append!(entries, ((r, c)
            for c in unknown_range(layout, unknown_id(layout, name), ir)))
        end
    end
    return entries
end
