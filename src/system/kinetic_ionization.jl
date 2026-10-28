# The ionization of the ions by the electrons of a distribution: the rate
# of each charge state is a moment of f with the weight v σ(p) of the cross sections of
# physics/ionization_cross_sections.jl, above the threshold of a momentum region, and the
# split of the ionization at that threshold between this term and the fluid ADAS rate of
# the cold electrons below it.

"""
    KineticIonizationTerm(composition, kinetic, grid; ions=:ion_densities, region=nothing,
                          ionization_cross_section_form=:chianti)

The ionization of the charge states of `composition` ([`IonComposition`](@ref)) by the
electrons of the kinetic unknown `kinetic` on `grid`, in the rows of the ion densities
`ions` (DREAM's `IonKineticIonizationTerm`):

```math
\\frac{dn_j}{dt} = K_{j-1} n_{j-1} - K_j n_j, \\qquad
K_j = \\frac{1}{V'} \\sum_{\\text{cells}} e(p)\\, v σ_j(p)\\, 𝒱' Δp Δξ\\, f,
```

with the cross sections [`ionization_cross_section`](@ref) of the form
`ionization_cross_section_form` (`:chianti`, the default, CHIANTI's parameters continued
relativistically, with DREAM's form for the elements CHIANTI's table lacks; `:dream`,
DREAM's form and fits; or `:bethe`, DREAM's form with its relativistic term divided by `β²`,
refitted), tabulated at the cell centres of the momentum grid of each radius (DREAM
tabulates the grid of radius 0 alone), and the envelope `e(p)` of the
[`MomentumRegion`](@ref) `region`: the hot electrons above a threshold that may follow the
temperature unknown, or the whole grid without a region. Each species is conserved.

Paired with an [`IonRateTerm`](@ref) through its `kinetic_ionization` keyword, the electrons
below the threshold ionize at the fluid coefficient of
[`fluid_ionization_coefficient`](@ref), so a Maxwellian `f` ionizes at the ADAS rate for
every threshold. The term has no energy of its own: the friction on bound electrons of the
Fokker-Planck term already takes energy from `f` for the inelastic collisions. With
`bound_friction = :ionization_split` of [`CollisionSettings`](@ref) and its
`ionization_threshold` at the factor of this term's region, that friction holds the energy
`I_j + ⟨E_s⟩_j` of each ionization the term counts, up to the discretization. Without a
region the two agree to the discretization; at a threshold the friction takes a smooth step
of its own on the faces (`ionization_threshold_width`) while this term takes the envelope of
its region in the cells, so the two differ by the electrons within that width of the
threshold.

The Jacobian is assembled directly: the columns of `f` (every cell of a radius, the block
DREAM's reference case drops), of the charge states, and of the temperature through the
threshold of the region.
"""
struct KineticIonizationTerm{E} <: EquationTerm
    composition::IonComposition
    kinetic::Symbol
    grid::KineticGrid{Float64}
    ions::Symbol
    envelopes::E
    ionization_cross_section_form::Symbol  # the form of ionization_cross_section
    rates::Vector{Matrix{Float64}}         # per radius: v σ [m³/s] per momentum cell and component
    weights::Vector{Matrix{Float64}}       # per radius: 𝒱' Δp Δξ / V' per cell, the weight of a density moment
end

function KineticIonizationTerm(
        c::IonComposition, kinetic::Symbol, grid::KineticGrid{Float64};
        ions::Symbol = :ion_densities, region::Union{Nothing, MomentumRegion} = nothing,
        ionization_cross_section_form::Symbol = :chianti)
    ionization_cross_section_form in IONIZATION_CROSS_SECTION_FORMS ||
        throw(ArgumentError("ionization_cross_section_form must be :chianti, :dream or :bethe, not $ionization_cross_section_form"))
    states = [(name, Z0) for (name, Z) in zip(c.names, c.nuclear_charges) for Z0 in 0:Z]
    rates = Matrix{Float64}[]
    weights = Matrix{Float64}[]
    for ir in 1:radial_cell_count(grid)
        m = momentum_grid(grid, ir)
        speed(p) = Constants.SPEED_OF_LIGHT * p / sqrt(1 + p^2)
        push!(rates,
            [speed(p) *
             ionization_cross_section(name, Z0, p; form = ionization_cross_section_form)
             for p in m.p_centers, (name, Z0) in states])
        jacobian = phase_space_jacobian(grid, ir)
        push!(weights,
            [jacobian[i, j] * m.p_widths[i] * m.xi_widths[j] /
             grid.radial_grid.spatial_jacobian[ir]
             for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m)])
    end
    return KineticIonizationTerm(
        c, kinetic, grid, ions, RegionEnvelopes(region, grid),
        ionization_cross_section_form, rates,
        weights)
end

function dependencies(term::KineticIonizationTerm)
    unique([term.ions; term.kinetic; collect(Symbol, _region_unknowns(term.envelopes))])
end
jacobian_assembly(::KineticIonizationTerm) = :direct

function validate(term::KineticIonizationTerm, layout, row)
    id = unknown_id(layout, term.ions)
    id == row ||
        throw(ArgumentError("KineticIonizationTerm adds to the rows of its ion densities $(term.ions)"))
    _check_ion_unknown(layout, id, term.composition)
    kinetic = unknown_id(layout, term.kinetic)
    layout.kinds[kinetic] === :kinetic ||
        throw(ArgumentError("KineticIonizationTerm needs a kinetic unknown"))
    _same_momentum_grids(term.grid, kinetic_grid(layout, kinetic)) ||
        throw(DimensionMismatch("the grid of the kinetic ionization does not match $(term.kinetic)"))
    _check_envelopes(term.envelopes, term.grid)
    return nothing
end

# the momentum moment Σ_ξ w f per momentum cell of radius ir, in the number type of f
function _pitch_moment(term::KineticIonizationTerm, f, ir)
    w = term.weights[ir]
    return [sum(w[i, j] * f[i, j] for j in axes(w, 2)) for i in axes(w, 1)]
end

"""
    kinetic_ionization_rates(term::KineticIonizationTerm, u, layout, t, ir)

The rate coefficients `K` [1/s] of all charge states at radius `ir`, the moments of `f`
of the state `u` with `v σ` above the threshold of the region, one per component of the ion
densities.
"""
function kinetic_ionization_rates(term::KineticIonizationTerm, u, layout, t, ir)
    f = kinetic_view(u, layout, unknown_id(layout, term.kinetic), ir)
    g = _pitch_moment(term, f, ir)
    envelope = _moment_envelope(term.envelopes, term.grid, ir, u, layout, t)
    rates = term.rates[ir]
    return [sum(_envelope_value(envelope, i) * rates[i, c] * g[i] for i in axes(rates, 1))
            for c in axes(rates, 2)]
end

function rhs!(out, term::KineticIonizationTerm, u, layout, row, t)
    c = term.composition
    for ir in 1:radial_cell_count(layout)
        K = kinetic_ionization_rates(term, u, layout, t, ir)
        n = radial_view(u, layout, row, ir)
        rows = unknown_range(layout, row, ir)
        for s in 1:length(c)
            components = species_components(c, s)
            for k in components[1:(end - 1)]
                flow = K[k] * n[k]
                out[rows[k]] -= flow
                out[rows[k + 1]] += flow
            end
        end
    end
    return out
end

function add_jacobian!(J, term::KineticIonizationTerm, u, layout, row, t)
    c = term.composition
    kinetic = unknown_id(layout, term.kinetic)
    temperature = _region_unknowns(term.envelopes)
    for ir in 1:radial_cell_count(layout)
        m = momentum_grid(term.grid, ir)
        rows = unknown_range(layout, row, ir)
        n = radial_view(u, layout, row, ir)
        K = kinetic_ionization_rates(term, u, layout, t, ir)
        envelope = _envelope_at(term.envelopes, term.grid, ir, u, layout, t)
        rates, w = term.rates[ir], term.weights[ir]
        # the derivative of the rates in the temperature, through the threshold of the region
        dK = if isempty(temperature)
            nothing
        else
            D = ForwardDiff.Dual{PlasmaColumnTag, Float64, 1}
            T = unknown_values(u, layout, unknown_id(layout, only(temperature)))[ir]
            p0 = term.envelopes.region.limit.factor *
                 sqrt(2 * D(T, ForwardDiff.Partials((1.0,))) /
                      Constants.ELECTRON_REST_ENERGY)
            dual_envelope = momentum_envelope!(zeros(D, momentum_cell_count(m)), term.envelopes.region, m, p0)
            g = _pitch_moment(term, kinetic_view(u, layout, kinetic, ir), ir)
            [sum(ForwardDiff.partials(dual_envelope[i], 1) * rates[i, k] * g[i]
             for i in axes(rates, 1))
             for k in axes(rates, 2)]
        end
        T_column = isempty(temperature) ? 0 :
                   first(unknown_range(layout, unknown_id(layout, only(temperature)), ir))
        for s in 1:length(c), k in species_components(c, s)[1:(end - 1)]
            # the flow K_k n_k from state k to k + 1
            _add_entry!(J, rows[k], rows[k], -K[k])
            _add_entry!(J, rows[k + 1], rows[k], K[k])
            for j in 1:pitch_cell_count(m), i in 1:momentum_cell_count(m)

                value = n[k] * _envelope_value(envelope, i) * rates[i, k] * w[i, j]
                column = _kinetic_index(layout, kinetic, ir, i, j)
                _add_entry!(J, rows[k], column, -value)
                _add_entry!(J, rows[k + 1], column, value)
            end
            if dK !== nothing
                _add_entry!(J, rows[k], T_column, -n[k] * dK[k])
                _add_entry!(J, rows[k + 1], T_column, n[k] * dK[k])
            end
        end
    end
    return J
end

function sparsity!(entries, term::KineticIonizationTerm, layout, row)
    c = term.composition
    kinetic = unknown_id(layout, term.kinetic)
    temperature = _region_unknowns(term.envelopes)
    for ir in 1:radial_cell_count(layout)
        rows = unknown_range(layout, row, ir)
        cells = unknown_range(layout, kinetic, ir)
        T_column = isempty(temperature) ? 0 :
                   first(unknown_range(layout, unknown_id(layout, only(temperature)), ir))
        for s in 1:length(c)
            components = species_components(c, s)
            for k in components
                r = rows[k]
                k > first(components) && push!(entries, (r, rows[k - 1]))
                push!(entries, (r, rows[k]))
                append!(entries, ((r, cell) for cell in cells))
                T_column == 0 || push!(entries, (r, T_column))
            end
        end
    end
    return entries
end

"""
    threshold_split(term::KineticIonizationTerm, ir, u, layout, t)

The parts of a Maxwellian at the cold temperature of radius `ir` above the threshold of
the region of `term`, on its momentum grid, for [`fluid_ionization_coefficient`](@ref): the
fraction `f_>` of its density and the rate coefficient `A_>` [m³/s] of each charge state,
`Σ e(p) v σ(p) M w / Σ M w` with the envelope `e`, the cell weights `w` of a density moment
and the relativistic Maxwellian `M = exp(−(γ − 1)/Θ)` at the cell centres. The sums run over
the cells of the grid, the quadrature of the term itself. The temperature is that of the
region's unknown in the state `u`, in its number type.
"""
function threshold_split(term::KineticIonizationTerm, ir, u, layout, t)
    temperature = _region_unknowns(term.envelopes)
    isempty(temperature) &&
        throw(ArgumentError("the split of the ionization needs a region whose threshold follows the temperature unknown"))
    T = unknown_values(u, layout, unknown_id(layout, only(temperature)))[ir]
    m = momentum_grid(term.grid, ir)
    theta = T / Constants.ELECTRON_REST_ENERGY
    W = vec(sum(term.weights[ir]; dims = 2))
    maxwellian = [exp(-p^2 / (sqrt(1 + p^2) + 1) / theta) * W[i]
                  for (i, p) in enumerate(m.p_centers)]
    total = sum(maxwellian)
    envelope = _moment_envelope(term.envelopes, term.grid, ir, u, layout, t)
    above = sum(_envelope_value(envelope, i) * maxwellian[i]
    for i in eachindex(maxwellian)) / total
    rates = term.rates[ir]
    coefficients = [sum(_envelope_value(envelope, i) * rates[i, k] * maxwellian[i]
                    for i in eachindex(maxwellian)) / total
                    for k in axes(rates, 2)]
    return (term = term, fraction = above, coefficients = coefficients)
end

"""
    fluid_ionization_coefficient(adas, fraction, coefficient)

The ionization coefficient of the cold electrons below the threshold of a kinetic
ionization: `(I − A_>)/(1 − f_>)`, the ADAS coefficient `I` of the whole
Maxwellian less its part above the threshold, `A_>`, per electron below it, with the
fraction `f_>` of the Maxwellian above the threshold ([`threshold_split`](@ref)). With the
closure `n_cold = n_M (1 − f_>)` and the kinetic rate `n_M A_>` of a Maxwellian `f` of
density `n_M`, the two add up to `n_M I` for every threshold.

The coefficient has no floor. `A_>` comes from the cross sections of the kinetic ionization
and `I` from ADAS. The Maxwellian rate coefficient of DREAM's fits exceeds the ADAS
coefficient by up to a factor of about 1.6 for a state of argon, that of CHIANTI's cross
sections by up to 2.6 for a state of neon or argon (Ne²⁺ at 3 to 20 eV).
Where the threshold lies below the ionization energy of a state, almost all of its
Maxwellian ionization comes from above the threshold, so `A_> > I` and the coefficient is
negative. On the grid of the tests (`n_e = 5e19 m⁻³`, 5 to 50 eV), with DREAM's cross sections,
it is negative for the dominant argon states at threshold factors 2 and 3, by 3 to 18 % of `I`,
and for no state with a coronal fraction above 1e-17 at factors 5 and 7. From 1 eV to 10 keV,
for D, Ne and Ar, it is negative for no state with a coronal
fraction above 1e-3 with DREAM's cross sections at factors 5 to 7. With CHIANTI's it is
negative at factor 5 for three such states of a cold plasma: Ar⁺ at 1 eV (−0.19 `I`, coronal
fraction 0.087), Ar²⁺ at 1.6 eV (−0.50 `I`, 1.2e-3) and Ne²⁺ at 2.5 eV (−2.8 `I`, 0.025), and
for none at factors 6 and 7. With a Maxwellian `f` the sum with the
kinetic rate is `n_M I` whatever the sign. The split refuses thresholds below the factor of
[`MINIMUM_SPLIT_THRESHOLD_FACTORS`](@ref) of its form.
"""
function fluid_ionization_coefficient(adas, fraction, coefficient)
    (adas - coefficient) / (1 - fraction)
end

"""
    MINIMUM_SPLIT_THRESHOLD_FACTORS

The smallest factor of the thermal momentum at which the threshold of a
[`KineticIonizationTerm`](@ref) may split the ionization with a fluid term (the
`kinetic_ionization` keyword of [`IonRateTerm`](@ref) and [`RadiatedPowerTerm`](@ref)), per
form of its cross sections (a choice of this package; see the docs page *Choices of this
package*): 7 for CHIANTI's (`:chianti`, the default), and 5, the hot-region threshold of
DREAM's disruption example (`examples/Disruptions/generate.py:234`), for DREAM's (`:dream`)
and for the fits of DREAM's method (`:bethe`).

Runes2P's default hot region takes 7 for every form ([`ThermalMomentum`](@ref)).
"""
const MINIMUM_SPLIT_THRESHOLD_FACTORS = Dict(:chianti => 7.0, :dream => 5.0, :bethe => 5.0)

# refuses the split of the ionization at the threshold of `kinetic_ionization` below the minimum factor
# of its form, MINIMUM_SPLIT_THRESHOLD_FACTORS; `owner` names the fluid term
function _check_split_threshold(kinetic_ionization::KineticIonizationTerm, owner::AbstractString)
    envelopes = kinetic_ionization.envelopes
    region = envelopes === nothing ? nothing : envelopes.region
    limit = region === nothing ? nothing : region.limit
    limit isa ThermalMomentum{Symbol} ||
        throw(ArgumentError("the split of the ionization of $owner needs a kinetic ionization whose region has a " *
                            "threshold at a multiple of the thermal momentum of the temperature unknown " *
                            "(ThermalMomentum with the name of the unknown), so that the fluid coefficient " *
                            "(SCD − A_>)/(1 − f_>) follows the temperature"))
    form = kinetic_ionization.ionization_cross_section_form
    minimum = MINIMUM_SPLIT_THRESHOLD_FACTORS[form]
    limit.factor >= minimum ||
        throw(ArgumentError("the split of the ionization of $owner at $(limit.factor) thermal momenta is refused with the " *
                            "cross sections :$form: below $minimum thermal momenta the fluid coefficient " *
                            "(SCD − A_>)/(1 − f_>) is negative for populated states, since the cross sections exceed " *
                            "ADAS above a threshold that lies below the ionization energy (with DREAM's form by 3 to " *
                            "18 % of SCD for argon at factors 2 and 3; with CHIANTI's by up to 2.8 SCD " *
                            "at factor 5 in a plasma at 1 to 2.5 eV). A negative fluid flow drains sparse upper " *
                            "states and gives the cold electrons binding energy. Use a threshold factor of at least " *
                            "$minimum (Runes2P's hot region takes 7; DREAM's disruption example 5 with " *
                            "ionization_cross_section_form = :dream), or pair the kinetic ionization without the " *
                            "split as DREAM does."))
    return nothing
end

# the fluid coefficient of charge state Z0 of species s with the split of a radius
function _fluid_ionization(split, s, Z0, adas)
    k = species_components(split.term.composition, s)[Z0 + 1]
    return fluid_ionization_coefficient(adas, split.fraction, split.coefficients[k])
end

# the kinetic energy [eV] of the threshold of the region at the temperature T (or of a fixed
# limit), the cut of the secondaries that heat the cold electrons
function _threshold_energy(term::KineticIonizationTerm, T)
    limit = term.envelopes.region.limit
    p0 = limit isa ThermalMomentum ?
         limit.factor * sqrt(2 * T / Constants.ELECTRON_REST_ENERGY) : limit
    return (sqrt(1 + p0^2) - 1) * Constants.ELECTRON_REST_ENERGY
end

# the envelope of the region of a kinetic ionization at radius ir at the temperature T, in
# the number type of T
function _envelope_at_temperature(term::KineticIonizationTerm, ir, T)
    m = momentum_grid(term.grid, ir)
    # without a region the kinetic ionization covers the whole grid
    term.envelopes === nothing && return ones(typeof(float(T)), momentum_cell_count(m))
    limit = term.envelopes.region.limit
    p0 = limit isa ThermalMomentum ?
         limit.factor * sqrt(2 * T / Constants.ELECTRON_REST_ENERGY) : limit
    return momentum_envelope!(zeros(typeof(float(p0)), momentum_cell_count(m)), term.envelopes.region, m, p0)
end

"""
    SecondaryHeatingTerm(kinetic_ionization::KineticIonizationTerm;
                         widths=SECONDARY_SPECTRUM_WIDTHS,
                         temperature=:cold_electron_temperature, cut=:none, spectrum=:beb)

The heating [W/m³] of the cold electrons by the secondary electrons of the kinetic
ionization `kinetic_ionization`, for the row of their temperature:

```math
Q = e \\sum_{s,j} n_{s,j} \\frac{1}{V'} \\sum_{\\text{cells}} e(p)\\, v σ_j(p)\\,
⟨E_s⟩_j(E)\\, 𝒱' Δp Δξ\\, f,
```

with the rates of `kinetic_ionization`, the envelope `e(p)` of its region (1 without one)
and the mean energy `⟨E_s⟩_j` of the secondaries of the spectrum `spectrum` and the cut
`cut` below. A kinetically ionized electron joins the cold electrons through the closure of
the cold density; without this term it is born at zero energy.

- `cut = :none` (the default): the mean over every secondary, up
  to `(E − I)/2`. All secondaries join the cold electrons, also those born above the hot
  threshold, so this is the moment of the spectrum over the secondaries that the cold
  electrons receive, and the one that the friction with `bound_friction = :ionization_split`
  takes from `f` ([`CollisionSettings`](@ref)). A [`KnockOnTerm`](@ref) on the bound
  electrons also puts the secondaries above its cutoff into `f`, so next to it their energy
  is counted twice. Their share of the mean is large: with the spectrum of Opal et al. at 1.3
  MeV, the secondaries above 1 keV carry about 59 % of it.
- `cut = :threshold`: the mean below the hot threshold `E_c`, conditional on lying there,
  for every secondary.

`spectrum = :beb` (the default, a choice of this package; see the docs page *Choices of this
package*) takes the binary-encounter-Bethe spectrum of Kim and Rudd with the ionization
energy of each charge state as its binding energy ([`beb_mean_secondary_energy`](@ref)),
which needs no width; `spectrum = :opal` the spectrum of Opal et al. with the widths
`widths`, the neutral element's for every charge state. The friction of the Fokker-Planck
term on bound electrons takes the energy of the ionization from `f` already, so in the total
energy of the electrons the term moves the energy of the secondaries from that sink to the
cold electrons. That holds where the friction takes the energy. With the default screening
window of the collisions (`screening_window = :excitation_energy`) the friction on bound
electrons is zero from the ionization potential up to the mean excitation energy of the
state (about 190 eV for Ar⁰, more for the higher states): there `f` ionizes without paying,
and this term adds the energy of the secondaries on top. `screening_window = :dream` narrows
that band, and `bound_friction = :ionization_split` of [`CollisionSettings`](@ref) closes
it: `f` then pays the ionization potential and the energy of the secondaries for every
ionization above the threshold, and this term returns the energy of the secondaries to the
cold electrons. DREAM has no such term: `kinetic_ionization` alone is DREAM's model. The
region of `kinetic_ionization` sets `E_c`; its threshold follows the temperature unknown
`temperature` or is fixed.
"""
struct SecondaryHeatingTerm{K <: KineticIonizationTerm} <: EquationTerm
    kinetic_ionization::K
    temperature::Symbol
    cut::Symbol                       # :none, every secondary, or :threshold, the mean below E_c
    spectrum::Symbol                  # :opal or :beb
    widths::Vector{Float64}           # per species of the composition of `kinetic_ionization`
    energies::Vector{Vector{Float64}} # per radius: the kinetic energy [eV] of each momentum cell
end

function SecondaryHeatingTerm(
        kinetic_ionization::KineticIonizationTerm; widths = SECONDARY_SPECTRUM_WIDTHS,
        temperature::Symbol = :cold_electron_temperature, cut::Symbol = :none,
        spectrum::Symbol = :beb)
    cut in (:none, :threshold) ||
        throw(ArgumentError("cut must be :none or :threshold, not $cut"))
    spectrum in SECONDARY_SPECTRA ||
        throw(ArgumentError("spectrum must be :opal or :beb, not $spectrum"))
    cut === :threshold && kinetic_ionization.envelopes === nothing &&
        throw(ArgumentError("the heating by secondaries with cut = :threshold needs the region of the kinetic " *
                            "ionization: its threshold is the cut of the secondaries"))
    limit = kinetic_ionization.envelopes === nothing ? nothing :
            kinetic_ionization.envelopes.region.limit
    limit === nothing || limit isa Real || limit isa ThermalMomentum{Symbol} ||
        throw(ArgumentError("the heating by secondaries needs a threshold at a fixed momentum or at a multiple of the " *
                            "thermal momentum of the temperature unknown"))
    table = merge(SECONDARY_SPECTRUM_WIDTHS,
        Dict{String, Float64}(String(k) => Float64(v) for (k, v) in widths))
    names = atomic_element.(kinetic_ionization.composition.names)
    for name in names
        spectrum === :beb && break
        haskey(table, name) ||
            throw(ArgumentError("no width of the spectrum of the secondary electrons for the element $name: give it " *
                                "with widths = Dict(\"$name\" => Ē in eV)"))
        table[name] > 0 ||
            throw(ArgumentError("the width of the secondary spectrum of $name must be positive"))
    end
    energies = [[(sqrt(1 + p^2) - 1) * Constants.ELECTRON_REST_ENERGY
                 for p in momentum_grid(kinetic_ionization.grid, ir).p_centers]
                for ir in 1:radial_cell_count(kinetic_ionization.grid)]
    return SecondaryHeatingTerm(kinetic_ionization, temperature, cut, spectrum,
        [get(table, name, NaN) for name in names], energies)
end

function _region_temperature_name(term::SecondaryHeatingTerm)
    _region_unknowns(term.kinetic_ionization.envelopes)
end
function dependencies(term::SecondaryHeatingTerm)
    unique([term.kinetic_ionization.ions; term.kinetic_ionization.kinetic;
            collect(Symbol, _region_temperature_name(term))])
end
jacobian_assembly(::SecondaryHeatingTerm) = :direct

function validate(term::SecondaryHeatingTerm, layout, row)
    validate(term.kinetic_ionization, layout, unknown_id(layout, term.kinetic_ionization.ions))
    layout.kinds[row] === :radial && layout.sizes[row, 1] == 1 ||
        throw(ArgumentError("SecondaryHeatingTerm adds to a radial unknown with one component"))
    return nothing
end

# the weight e ⟨E_s⟩ v σ [J m³/s] of each momentum cell and component of radius ir at the
# temperature T
function _secondary_weights(term::SecondaryHeatingTerm, ir, T)
    kinetic_ionization = term.kinetic_ionization
    c = kinetic_ionization.composition
    # one type: Inf, and the threshold of a fixed-momentum region, in the type of T
    cut = oftype(float(T), term.cut === :threshold ?
                           _threshold_energy(kinetic_ionization, T) : Inf)
    rates = kinetic_ionization.rates[ir]
    energies = term.energies[ir]
    weights = zeros(typeof(float(cut)), size(rates))
    for s in 1:length(c), (j, k) in enumerate(species_components(c, s))

        j > c.nuclear_charges[s] && continue
        I = ionization_energy(c.names[s], j - 1)
        for i in axes(rates, 1)
            rates[i, k] == 0 && continue
            weights[i, k] = Constants.ELEMENTARY_CHARGE * rates[i, k] *
                            secondary_spectrum_mean(
                                term.spectrum, term.widths[s], energies[i], I, cut)
        end
    end
    return weights
end

# the heating at radius ir from the pitch moment g of f, the densities n of the charge states
# and the temperature T, in their number types
function _secondary_heating(term::SecondaryHeatingTerm, ir, g, n, T)
    weights = _secondary_weights(term, ir, T)
    envelope = _envelope_at_temperature(term.kinetic_ionization, ir, T)
    total = zero(eltype(weights)) * zero(eltype(g)) * zero(eltype(n))
    for k in axes(weights, 2)
        moment = zero(total)
        for i in axes(weights, 1)
            moment += envelope[i] * weights[i, k] * g[i]
        end
        total += n[k] * moment
    end
    return total
end

function _radius_temperature(term::SecondaryHeatingTerm, u, layout, ir)
    isempty(_region_temperature_name(term)) ? 0.0 :
    unknown_values(u, layout, unknown_id(layout, only(_region_temperature_name(term))))[ir]
end

function rhs!(out, term::SecondaryHeatingTerm, u, layout, row, t)
    kinetic_ionization = term.kinetic_ionization
    id = unknown_id(layout, kinetic_ionization.kinetic)
    for ir in 1:radial_cell_count(layout)
        g = _pitch_moment(kinetic_ionization, kinetic_view(u, layout, id, ir), ir)
        n = radial_view(u, layout, unknown_id(layout, kinetic_ionization.ions), ir)
        out[first(unknown_range(layout, row, ir))] += _secondary_heating(
            term, ir, g, n, _radius_temperature(term, u, layout, ir))
    end
    return out
end

function add_jacobian!(J, term::SecondaryHeatingTerm, u, layout, row, t)
    kinetic_ionization = term.kinetic_ionization
    id = unknown_id(layout, kinetic_ionization.kinetic)
    ions = unknown_id(layout, kinetic_ionization.ions)
    temperature = _region_temperature_name(term)
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        T = _radius_temperature(term, u, layout, ir)
        f = kinetic_view(u, layout, id, ir)
        g = _pitch_moment(kinetic_ionization, f, ir)
        n = radial_view(u, layout, ions, ir)
        weights = _secondary_weights(term, ir, T)
        envelope = _envelope_at_temperature(kinetic_ionization, ir, T)
        w = kinetic_ionization.weights[ir]
        # the cells of f: Σ_k n_k e(p) weight_k(p) times the cell weight
        for i in axes(weights, 1)
            per_cell = envelope[i] * sum(n[k] * weights[i, k] for k in axes(weights, 2))
            for j in axes(w, 2)
                _add_entry!(J, r, _kinetic_index(layout, id, ir, i, j), per_cell * w[i, j])
            end
        end
        # the charge states
        for (k, column) in enumerate(unknown_range(layout, ions, ir))
            #! format: off
            _add_entry!(J, r, column, sum(envelope[i] * weights[i, k] * g[i] for i in axes(weights, 1)))
            #! format: on
        end
        # the temperature, through the threshold and the cut of the secondaries
        if !isempty(temperature)
            derivative = ForwardDiff.derivative(x -> _secondary_heating(term, ir, g, n, x), T)
            _add_entry!(J, r,
                first(unknown_range(layout, unknown_id(layout, only(temperature)), ir)),
                derivative)
        end
    end
    return J
end

function sparsity!(entries, term::SecondaryHeatingTerm, layout, row)
    kinetic_ionization = term.kinetic_ionization
    for ir in 1:radial_cell_count(layout)
        r = first(unknown_range(layout, row, ir))
        for name in dependencies(term)
            append!(entries, ((r, c)
            for c in unknown_range(layout, unknown_id(layout, name), ir)))
        end
    end
    return entries
end

"""
    ionization_energy_flows(kinetic_ionization::KineticIonizationTerm, u, layout, t;
                            widths=SECONDARY_SPECTRUM_WIDTHS, spectrum=:beb)

The energy [W/m³ per radius] that the kinetic ionization `kinetic_ionization` takes from `f`
at the state `u`, in two parts:

- `potential`: the ionization potentials, `e Σ I_j K_j n_j`, stored in the ions and released
  at recombination (the gain of `RadiatedPowerTerm(...; radiation = :recombination)`);
- `secondaries`: the kinetic energy of the secondary electrons, which
  [`SecondaryHeatingTerm`](@ref) with its default `cut = :none` gives to the cold electrons; with
  `spectrum = :opal`, `nothing` when an element of the composition has no width of the
  secondary spectrum ([`SECONDARY_SPECTRUM_WIDTHS`](@ref)), since `potential` needs none.

Both are part of the sink of `f` through the friction on bound electrons
([`KineticEnergyLossTerm`](@ref) with `part = :sinks`), so an energy diagnostic that books
that sink as lost and the recombination radiation as radiated counts `potential` twice: it
is the share of the sink that goes to the potential energy of the ions.
"""
function ionization_energy_flows(
        kinetic_ionization::KineticIonizationTerm, u, layout, t; widths = SECONDARY_SPECTRUM_WIDTHS,
        spectrum::Symbol = :beb)
    c = kinetic_ionization.composition
    nr = radial_cell_count(layout)
    potential = zeros(nr)
    for ir in 1:nr
        K = kinetic_ionization_rates(kinetic_ionization, u, layout, t, ir)
        n = radial_view(u, layout, unknown_id(layout, kinetic_ionization.ions), ir)
        for s in 1:length(c), (j, k) in enumerate(species_components(c, s))

            j > c.nuclear_charges[s] && continue
            potential[ir] += Constants.ELEMENTARY_CHARGE *
                             ionization_energy(c.names[s], j - 1) * K[k] * n[k]
        end
    end
    table = merge(SECONDARY_SPECTRUM_WIDTHS,
        Dict{String, Float64}(String(k) => Float64(v) for (k, v) in widths))
    spectrum === :beb || all(name -> haskey(table, atomic_element(name)), c.names) ||
        return (potential = potential, secondaries = nothing)
    heating = SecondaryHeatingTerm(kinetic_ionization; widths, spectrum)
    row = unknown_id(layout, heating.temperature)
    secondaries = unknown_values(rhs!(zeros(length(u)), heating, u, layout, row, t), layout, row)
    return (potential = potential, secondaries = collect(secondaries))
end
