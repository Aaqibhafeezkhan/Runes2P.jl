# The densities of the ion charge states as unknowns: one radial unknown
# with a component per charge state and a group per species, the free-electron density as
# their weighted sum, the closure of the cold density, and the initial state with the
# deposition of neutrals at the start of a run.

"""
    WeightedSumTerm(unknown, weights)

`Σ_k w_k x_k` over the components `x_k` of the radial unknown `unknown` at each radius,
added to the row of a radial unknown with one component: the free-electron density
`n_free = Σ Z0 n_ij` of the ion densities with the weights of
[`free_electron_weights`](@ref). The term is linear, and its Jacobian entries are
assembled directly, since its row holds every component of a radius.
"""
struct WeightedSumTerm <: EquationTerm
    unknown::Symbol
    weights::Vector{Float64}
end
WeightedSumTerm(unknown::Symbol, weights::AbstractVector{<:Real}) = WeightedSumTerm(unknown, collect(Float64, weights))

dependencies(term::WeightedSumTerm) = [term.unknown]
is_linear(::WeightedSumTerm) = true
jacobian_assembly(::WeightedSumTerm) = :direct

function validate(term::WeightedSumTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    layout.kinds[id] === :radial && layout.sizes[id, 1] == length(term.weights) ||
        throw(DimensionMismatch("WeightedSumTerm needs a radial unknown with $(length(term.weights)) components, " *
                                "one per weight"))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("WeightedSumTerm adds to a radial unknown with one component"))
    return nothing
end

function rhs!(out, term::WeightedSumTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        total = zero(eltype(out))
        @inbounds for (k, i) in enumerate(unknown_range(layout, id, ir))
            total += term.weights[k] * u[i]
        end
        out[first(unknown_range(layout, row, ir))] += total
    end
    return out
end

function add_jacobian!(J, term::WeightedSumTerm, u, layout, row, t)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for (k, i) in enumerate(unknown_range(layout, id, ir))
            _add_entry!(J, r, i, term.weights[k])
        end
    end
    return J
end

function sparsity!(entries, term::WeightedSumTerm, layout, row)
    id = unknown_id(layout, term.unknown)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        append!(entries, ((r, i) for i in unknown_range(layout, id, ir)))
    end
    return entries
end

"""
    free_electron_weights(composition::IonComposition)

The charge `Z0` of every component of the ion densities: the weights of
[`WeightedSumTerm`](@ref) for the free-electron density `Σ Z0 n_ij`.
"""
free_electron_weights(c::IonComposition) = Float64[Z0 for Z in c.nuclear_charges for Z0 in 0:Z]

"""
    ion_densities_unknown(composition; name=:ion_densities, scale=1e19, limit_tolerance=1e-10)

The radial unknown of the densities [m⁻³] of the charge states of `composition`
([`IonComposition`](@ref)), with one group of components per species
([`component_groups`](@ref)). Its lower limit keeps every charge state at or above
`−limit_tolerance` times the density of its species (`LowerLimit(reference = :group)`): a
state that starts at 0 is not refused for a round-off below 0, and the limit scales with
each species, so a trace impurity has its own (DREAM limits only `Σ Z0² n` per radius,
DREAM issues #25 and #91). The default Newton tolerance of such an unknown is relative to
the density of each species as well ([`tolerance`](@ref)).
"""
function ion_densities_unknown(c::IonComposition; name::Symbol=:ion_densities, scale::Real=1e19,
                               limit_tolerance::Real=1e-10)
    groups = [species_components(c, s) for s in 1:length(c)]
    return RadialUnknown(name; components=component_count(c), scale, groups,
                         nonnegative=LowerLimit(tolerance=limit_tolerance, reference=:group))
end

"""
    set_ion_densities!(u, layout, ions; name=:ion_densities)

Set the unknown `name` of the state `u` to the charge-state densities of `ions`, a list of
[`IonSpecies`](@ref) or a [`PlasmaState`](@ref), whose species and order define the
composition of the unknown.
"""
set_ion_densities!(u::AbstractVector, layout::UnknownLayout, plasma::PlasmaState; name::Symbol=:ion_densities) =
    set_ion_densities!(u, layout, plasma.ions; name)
function set_ion_densities!(u::AbstractVector, layout::UnknownLayout, ions::AbstractVector{<:IonSpecies};
                            name::Symbol=:ion_densities)
    id = unknown_id(layout, name)
    composition = IonComposition(ions)
    _check_ion_unknown(layout, id, composition)
    for (s, species) in enumerate(ions), ir in 1:radial_cell_count(layout)
        size(species.charge_state_densities, 2) == radial_cell_count(layout) ||
            throw(DimensionMismatch("the densities of $(species.name) need one column per radius"))
        radial_view(u, layout, id, ir)[species_components(composition, s)] .= species.charge_state_densities[:, ir]
    end
    return u
end

"""
    ion_densities(u, layout, composition; name=:ion_densities)

The [`IonSpecies`](@ref) of `composition` with the charge-state densities of the unknown
`name` in the state `u`.
"""
function ion_densities(u::AbstractVector, layout::UnknownLayout, c::IonComposition; name::Symbol=:ion_densities)
    id = unknown_id(layout, name)
    _check_ion_unknown(layout, id, c)
    nr = radial_cell_count(layout)
    return [IonSpecies(c.names[s], c.nuclear_charges[s],
                       reduce(hcat, [radial_view(u, layout, id, ir)[species_components(c, s)] for ir in 1:nr]))
            for s in 1:length(c)]
end

"""
    deposit_neutrals!(u, layout, composition, deposits...; name=:ion_densities)

Add neutral atoms to the state `u` at the start of a run (DREAM's `IONS_DYNAMIC_NEUTRAL`):
each deposit `species => density` adds `density` [m⁻³], a number or one value per radius,
to the neutral state of the species of that name. The rate equations ionize them from the
first step on. DREAM's reference case deposits neutral deuterium and argon at 1 eV,
uniform in radius (`examples/Disruptions/generate.py`); a deposit of deuterium into the
species of the background deuterium gives the same densities as DREAM's separate species,
since the rates are linear in the densities.
"""
function deposit_neutrals!(u::AbstractVector, layout::UnknownLayout, c::IonComposition, deposits::Pair...;
                           name::Symbol=:ion_densities)
    id = unknown_id(layout, name)
    _check_ion_unknown(layout, id, c)
    nr = radial_cell_count(layout)
    for (species, density) in deposits
        density isa Real || length(density) == nr ||
            throw(DimensionMismatch("the deposit of $species needs a number or one value per radius"))
        all(>=(0), density) || throw(ArgumentError("the deposit of $species must not be negative"))
        neutral = first(species_components(c, String(species)))
        for ir in 1:nr
            radial_view(u, layout, id, ir)[neutral] += density isa Real ? density : density[ir]
        end
    end
    return u
end

"""
    population_equations(composition; ions=:ion_densities, cold=:cold_electron_density,
                         hot=nothing, runaway=nothing)

The closure of the cold-electron density, `0 = Σ Z0 n_ij − n_hot − n_re − n_cold`: the free
electrons of the ion densities `ions` ([`WeightedSumTerm`](@ref)) less those of the
distributions, given as the radial unknowns `hot` and `runaway` of their densities
(DREAM's `n_cold`). Without them the cold density is the
free-electron density. The cold density is the target of the e-e collisions, read by a
[`PlasmaFromUnknowns`](@ref) with `density = cold`.
"""
function population_equations(c::IonComposition; ions::Symbol=:ion_densities, cold::Symbol=:cold_electron_density,
                              hot::Union{Nothing,Symbol}=nothing, runaway::Union{Nothing,Symbol}=nothing)
    minus(name) = LinearTerm(name; coefficient=-1)
    terms = EquationTerm[WeightedSumTerm(ions, free_electron_weights(c))]
    for name in (hot, runaway)
        name === nothing || push!(terms, minus(name))
    end
    push!(terms, minus(cold))
    return [Equation(cold, Algebraic(), Tuple(terms))]
end

"""
    IonRateTerm(composition; ions=:ion_densities, temperature=:cold_electron_temperature,
                density=:cold_electron_density, kinetic=nothing, extrapolation=:dream,
                directory=adas_cache_directory())

The ionization and recombination of the charge states of `composition`
([`IonComposition`](@ref)) by the cold electrons, in the rows of the ion densities `ions`
(DREAM's `IonRateEquation`):

```math
\\frac{dn_j}{dt} = n_e \\left(I_{j-1} n_{j-1} - I_j n_j + R_{j+1} n_{j+1} - R_j n_j\\right)
```

for each species and charge state `j = 0, …, Z`, with the ionization coefficients `I`
(ADF11 SCD) and the recombination coefficients `R` (ACD) of OpenADAS (H. P. Summers, The ADAS User Manual, version 2.6 (2004), http://www.adas.ac.uk, with the data of OpenADAS, https://open.adas.ac.uk) at the electron
density `n_e` and temperature of the radial unknowns `density` and `temperature`
([`adas_coefficient`](@ref), with `extrapolation`, whose default `:dream` is a
decision of the author, 2026-10-04), as DREAM's
`src/Equations/Fluid/IonRateEquation.cpp`. The species are named by their element,
whose tables are read from `directory` when the term is built ([`read_adas`](@ref)).

The sum over the charge states of a species is zero, so the term moves particles between
the states and conserves each species. Its Jacobian comes from the colored AD of its
residual, with the derivatives of the coefficients through ForwardDiff in place of DREAM's
one-sided differences, and without DREAM's floors on the density (1 m⁻³) and the
coefficients (1e-40) in the columns of the temperature and the density.

`kinetic`, a [`KineticIonizationTerm`](@ref) of the same system, splits the ionization at
the threshold of its region: the electrons of `f` above the threshold
ionize through the cross sections of that term, and the cold electrons below it through
the fluid coefficient of [`fluid_ionization_coefficient`](@ref), the ADAS rate less the
part of the Maxwellian above the threshold. No electron is counted twice. The threshold is a
multiple of the thermal momentum of at least [`MINIMUM_SPLIT_THRESHOLD_FACTOR`](@ref).
Without
`kinetic` the fluid rate is the whole ADAS rate, which next to a kinetic ionization term
is DREAM's choice (`ions.cpp:280`): the Maxwellian tail above the threshold ionizes twice.
"""
struct IonRateTerm{K} <: EquationTerm
    composition::IonComposition
    ions::Symbol
    temperature::Symbol
    density::Symbol
    extrapolation::Symbol
    tables::Vector{NamedTuple}   # per species: the ADF11 tables of its element
    kinetic::K                   # nothing, or the KineticIonizationTerm of the electrons above the threshold
end

function IonRateTerm(c::IonComposition; ions::Symbol=:ion_densities, temperature::Symbol=:cold_electron_temperature,
                     density::Symbol=:cold_electron_density, kinetic=nothing, extrapolation::Symbol=:dream,
                     directory::AbstractString=adas_cache_directory())
    extrapolation in (:dream, :clamp_density) ||
        throw(ArgumentError("extrapolation must be :dream or :clamp_density, not $extrapolation"))
    kinetic === nothing || kinetic isa KineticIonizationTerm ||
        throw(ArgumentError("kinetic must be a KineticIonizationTerm or nothing"))
    kinetic === nothing || kinetic.composition == c ||
        throw(ArgumentError("the kinetic ionization of the split has another ion composition"))
    kinetic === nothing || _check_split_threshold(kinetic, "an IonRateTerm")
    tables = NamedTuple[_element_tables(name, Z, directory) for (name, Z) in zip(c.names, c.nuclear_charges)]
    return IonRateTerm(c, ions, temperature, density, extrapolation, tables, kinetic)
end

# the ADF11 tables of the element of a species, checked against its nuclear charge
function _element_tables(element::AbstractString, Z::Integer, directory)
    haskey(ADAS_FILES, element) ||
        throw(ArgumentError("the ion species $element is not an element with ADF11 tables " *
                            "($(join(sort!(collect(keys(ADAS_FILES))), ", ")))"))
    tables = read_adas(element; directory)
    tables.scd.nuclear_charge == Z ||
        throw(ArgumentError("the species $element has nuclear charge $Z, its ADF11 tables " *
                            "$(tables.scd.nuclear_charge)"))
    return tables
end

dependencies(term::IonRateTerm) =
    unique([term.ions; term.temperature; term.density;
            term.kinetic === nothing ? Symbol[] : collect(Symbol, _region_unknowns(term.kinetic.envelopes))])

function validate(term::IonRateTerm, layout, row)
    id = unknown_id(layout, term.ions)
    id == row || throw(ArgumentError("IonRateTerm adds to the rows of its ion densities $(term.ions)"))
    _check_ion_unknown(layout, id, term.composition)
    for name in (term.temperature, term.density)
        k = unknown_id(layout, name)
        layout.kinds[k] === :radial && layout.sizes[k, 1] == 1 ||
            throw(ArgumentError("the $name of an IonRateTerm is a radial unknown with one component"))
    end
    return nothing
end

# the rate of change of the charge states of species s at the electron density n_e and
# temperature T, added to `rows` (the entries of the species in `out`), from the densities
# `n` of its states
function _add_ion_rates!(out, rows, term::IonRateTerm, s, n, n_e, T, split=nothing)
    tables = term.tables[s]
    Z = term.composition.nuclear_charges[s]
    coefficient(table, Z0) = adas_coefficient(table, Z0, n_e, T; extrapolation=term.extrapolation)
    # the flow from state j to j + 1: ionization of j less recombination of j + 1
    for j in 0:Z-1
        ionization = split === nothing ? coefficient(tables.scd, j) :
                     _fluid_ionization(split, s, j, coefficient(tables.scd, j))
        flow = n_e * (ionization * n[j+1] - coefficient(tables.acd, j + 1) * n[j+2])
        out[rows[j+1]] -= flow
        out[rows[j+2]] += flow
    end
    return out
end

function rhs!(out, term::IonRateTerm, u, layout, row, t)
    T = radial_values(u, layout, unknown_id(layout, term.temperature))
    n_e = radial_values(u, layout, unknown_id(layout, term.density))
    c = term.composition
    for ir in 1:radial_cell_count(layout)
        n = radial_view(u, layout, row, ir)
        rows = unknown_range(layout, row, ir)
        split = term.kinetic === nothing ? nothing : threshold_split(term.kinetic, ir, u, layout, t)
        for s in 1:length(c)
            components = species_components(c, s)
            _add_ion_rates!(out, view(rows, components), term, s, view(n, components), n_e[ir], T[ir], split)
        end
    end
    return out
end

function sparsity!(entries, term::IonRateTerm, layout, row)
    T, n_e = unknown_id(layout, term.temperature), unknown_id(layout, term.density)
    c = term.composition
    for ir in 1:radial_cell_count(layout)
        rows = unknown_range(layout, row, ir)
        scalars = (first(unknown_range(layout, T, ir)), first(unknown_range(layout, n_e, ir)))
        for s in 1:length(c)
            states = rows[species_components(c, s)]
            for (k, r) in enumerate(states)
                for m in max(k - 1, 1):min(k + 1, length(states))
                    push!(entries, (r, states[m]))
                end
                push!(entries, (r, scalars[1]), (r, scalars[2]))
            end
        end
    end
    return entries
end

"""
    ion_rate_matrix(term::IonRateTerm, s, n_e, T)

The matrix `A` of the rates of species `s` at the electron density `n_e` [m⁻³] and
temperature `T` [eV], `dn/dt = A n` for the densities `n` of its charge states
([`IonRateTerm`](@ref)). Its columns sum to zero.
"""
function ion_rate_matrix(term::IonRateTerm, s::Integer, n_e, T)
    Z = term.composition.nuclear_charges[s]
    A = zeros(typeof(float(n_e * T)), Z + 1, Z + 1)
    for j in 0:Z
        unit = zeros(Z + 1)
        unit[j+1] = 1
        _add_ion_rates!(view(A, :, j + 1), 1:Z+1, term, s, unit, n_e, T)
    end
    return A
end

"""
    coronal_equilibrium(term::IonRateTerm, s, n_e, T; density=1)

The densities of the charge states of species `s` in coronal equilibrium at the electron
density `n_e` [m⁻³] and temperature `T` [eV], with the total `density`: the steady state of
the rates of [`IonRateTerm`](@ref), where the flow between neighbouring states vanishes,
`n_{j+1}/n_j = I_j/R_{j+1}`. The ratios are multiplied in log space, so states far from the
dominant one underflow to 0 and the large ratios at the extremes of the temperature do not
overflow (DREAM #241).
"""
function coronal_equilibrium(term::IonRateTerm, s::Integer, n_e::Real, T::Real; density::Real=1.0)
    tables = term.tables[s]
    Z = term.composition.nuclear_charges[s]
    coefficient(table, Z0) = adas_coefficient(table, Z0, n_e, T; extrapolation=term.extrapolation)
    logs = zeros(Z + 1)
    for j in 0:Z-1
        logs[j+2] = logs[j+1] + log(coefficient(tables.scd, j)) - log(coefficient(tables.acd, j + 1))
    end
    largest = maximum(logs)
    weights = exp.(logs .- largest)
    return density .* weights ./ sum(weights)
end

"""
    IonDiffusionTerm(composition; diffusion=nothing, diffusion_at=nothing, ions=:ion_densities,
                     states=:charged, edge=:zero_flux)

Radial diffusion of the densities of the ion charge states of `composition`,
`(1/V') ∂_r (V' D ∂_r n_j)` [m⁻³ s⁻¹] in the rows of the ion densities `ions`
(DREAM's `IonChargedPrescribedDiffusionTerm`). The diffusivity `D` [m²/s] is `diffusion`, a
radial profile on the faces ([`radial_values`](@ref)), or `diffusion_at`, a function of time
that returns one, such as the [`PiecewiseConstant`](@ref) transport history of
a disruption run (1 000 m²/s for the first milliseconds, then a hundredth): its switch times
must be time points of the run ([`forced_time_points`](@ref)). The flux through the face
between two cells is `V' D (n_k − n_{k−1})/Δr` with the distance of the centres, as for the
heat ([`HeatTransportTerm`](@ref)); nothing crosses the axis.

- `states = :charged` (default) diffuses the charge states `Z0 ≥ 1`, DREAM's charged
  diffusion; `:all` the neutrals too.
- `edge = :zero_flux` keeps the ions in the plasma; `:absorbing` sets the density to zero on
  the edge face, over the distance from the last centre to the face.

The term has the sign convention of every other row: a positive rate adds particles (DREAM
needs a negative coefficient for its ion transient term, its PR #145).
"""
struct IonDiffusionTerm{D} <: EquationTerm
    composition::IonComposition
    ions::Symbol
    level::D
    level_in_time::Bool
    states::Symbol
    edge::Symbol
    faces::Base.RefValue{Vector{Float64}}   # D on the faces at `time`
    time::Base.RefValue{Float64}
end

function IonDiffusionTerm(c::IonComposition; diffusion=nothing, diffusion_at=nothing, ions::Symbol=:ion_densities,
                          states::Symbol=:charged, edge::Symbol=:zero_flux)
    (diffusion === nothing) != (diffusion_at === nothing) ||
        throw(ArgumentError("give one of diffusion and diffusion_at"))
    states in (:charged, :all) || throw(ArgumentError("states must be :charged or :all, not $states"))
    edge in (:zero_flux, :absorbing) || throw(ArgumentError("edge must be :zero_flux or :absorbing, not $edge"))
    level = something(diffusion, diffusion_at)
    return IonDiffusionTerm(c, ions, level, diffusion_at !== nothing, states, edge, Ref(Float64[]), Ref(NaN))
end

dependencies(term::IonDiffusionTerm) = [term.ions]
is_linear(::IonDiffusionTerm) = true
forced_time_points(term::IonDiffusionTerm) = term.level_in_time ? forced_time_points(term.level) : Float64[]

function validate(term::IonDiffusionTerm, layout, row)
    id = unknown_id(layout, term.ions)
    id == row || throw(ArgumentError("IonDiffusionTerm adds to the rows of its ion densities $(term.ions)"))
    _check_ion_unknown(layout, id, term.composition)
    term.level_in_time || _diffusion_faces(term, layout.radial_grid, 0.0)
    return nothing
end

# D on the faces at time t: once for a profile, once per time for a function of time
function _diffusion_faces(term::IonDiffusionTerm, rg::RadialGrid, t)
    if term.level_in_time
        time = Float64(_value(t))
        term.time[] == time && !isempty(term.faces[]) && return term.faces[]
        values = collect(Float64, radial_values(level_over_step(term.level, time), rg.r_faces))
        term.time[] = time
    else
        isempty(term.faces[]) || return term.faces[]
        values = collect(Float64, radial_values(term.level, rg.r_faces))
    end
    all(>=(0), values) || throw(ArgumentError("the ion diffusivity must not be negative"))
    term.faces[] = values
    return values
end

# whether the component c of the ion densities diffuses
function _diffuses(term::IonDiffusionTerm, c)
    term.states === :all && return true
    composition = term.composition
    for s in 1:length(composition)
        components = species_components(composition, s)
        c in components && return c != first(components)
    end
    return false
end

function rhs!(out, term::IonDiffusionTerm, u, layout, row, t)
    rg = layout.radial_grid
    D = _diffusion_faces(term, rg, t)
    nr = radial_cell_count(rg)
    for c in 1:component_count(term.composition)
        _diffuses(term, c) || continue
        density(ir) = radial_view(u, layout, row, ir)[c]
        # V' D ∂_r n through face k
        function flux(k)
            k == 1 && return zero(eltype(u))
            if k <= nr
                distance = rg.r_centers[k] - rg.r_centers[k-1]
                return rg.spatial_jacobian_faces[k] * D[k] * (density(k) - density(k - 1)) / distance
            end
            term.edge === :zero_flux && return zero(eltype(u))
            return rg.spatial_jacobian_faces[nr+1] * D[nr+1] * (0 - density(nr)) / (rg.r_faces[nr+1] - rg.r_centers[nr])
        end
        for ir in 1:nr
            out[unknown_range(layout, row, ir)[c]] += (flux(ir + 1) - flux(ir)) / (rg.spatial_jacobian[ir] * rg.r_widths[ir])
        end
    end
    return out
end

function sparsity!(entries, term::IonDiffusionTerm, layout, row)
    nr = radial_cell_count(layout)
    for c in 1:component_count(term.composition), ir in 1:nr
        _diffuses(term, c) || continue
        for neighbour in max(ir - 1, 1):min(ir + 1, nr)
            push!(entries, (unknown_range(layout, row, ir)[c], unknown_range(layout, row, neighbour)[c]))
        end
    end
    return entries
end
