# Collision frequencies of the linearized relativistic test-particle operator, ported
# from DREAM (CollisionFrequency.cpp, SlowingDownFrequency.cpp,
# PitchScatterFrequency.cpp, ParallelDiffusionFrequency.cpp). Partial screening
# follows Hesslow et al., J. Plasma Phys. 84 (2018); the thermal electron terms
# follow Pike & Rose, Phys. Rev. E 89 (2014).

using SpecialFunctions: besselkx, erf
using QuadGK: quadgk

"""
    CollisionSettings(; mode, screening, coulomb_logarithm, screened_diffusion, ion_size_table,
                      screening_window, screening_window_start, screening_window_end, bremsstrahlung,
                      bremsstrahlung_diffusion, bremsstrahlung_constant, bound_friction, ionization_threshold,
                      ionization_threshold_width, secondary_widths, secondary_spectrum, ionization_cross_section_form)

The defaults of `mode`, `screening`, `coulomb_logarithm`, `screened_diffusion` and `bound_friction` are
those of DREAM's `CollisionHandler`. The defaults of `ion_size_table`, `screening_window`, `bremsstrahlung`,
`bremsstrahlung_diffusion` and `bremsstrahlung_constant` depart from DREAM (see the docs page
*Comparison with DREAM*).
The remaining options have no counterpart in DREAM.

- `mode`: `:full` (relativistic Maxwellian targets, adds energy diffusion),
  `:superthermal` (cold target electrons) or `:ultra_relativistic`.
- `screening`: `:partial` (Hesslow et al. 2018), `:complete` or `:none` (DREAM's
  `PARTIALLY_SCREENED`, `COMPLETELY_SCREENED`, `NON_SCREENED`).
- `coulomb_logarithm`: `:thermal`, `:energy_dependent` or `:relativistic` (see
  [`coulomb_logarithm`](@ref)).
- `screened_diffusion`: energy diffusion from the full `ν_s` (`:maxwellian`) or from
  the free-electron part only (`:zero`).
- `ion_size_table`: `:hesslow` uses Table 1 of Hesslow et al. 2018 as intended;
  `:dream` reproduces DREAM's lookup, whose charge list has one argon entry too few
  (affects Ar¹⁷⁺, Xe³⁺, Xe¹⁷⁺, W³⁺ and W⁶⁰⁺).
- `screening_window`: momenta where partial screening applies. With
  `:excitation_energy` the bound-electron parts of charge state `j` (the Bethe term of
  `ν_s`, the `g_j(p)` of `ν_D`) are weighted by a smooth step in the kinetic energy
  `(γ - 1) mₑc²`, 0 below `screening_window_start · I_j` and 1 above
  `screening_window_end · I_j`, with `I_j` the mean excitation energy. Below the window
  the frequencies are those of complete screening. The default factors 1 and 4
  (a choice of this package; see the docs page *Choices of this package*) place the
  window of Ar⁺ and Ar²⁺ at p ≈ 0.03 to 0.06 (Hesslow et al. PRL 2017 and JPP 2019).
  `:dream` applies partial screening at all momenta, as DREAM does. The weight depends only
  on `p` and `I_j`, so it holds unchanged when the densities become unknowns.
- `bremsstrahlung`: `:stopping_power`, the radiative energy loss of an electron on the
  nuclei of all charge states as a slowing down (DREAM's
  `BREMSSTRAHLUNG_MODE_STOPPING_POWER`): the non-screened formula 4BN of
  Koch and Motz (Rev. Mod. Phys. 31, 920, 1959), `ν_s = n Z² α r₀² c φ_rad(p)/β` per
  charge state ([`bremsstrahlung_cross_section`](@ref)), on the nuclei only, as in DREAM:
  there is no electron-electron bremsstrahlung, so the stopping lies below ESTAR for hydrogen
  by 1.21 at 212 keV and 1.55 at 1.11 MeV, tending to `Z(Z + 1)/Z²`. It is the default (a
  choice of this package; see the docs page *Choices of this package*). `:neglect` leaves the
  loss out, as DREAM does by default.
- `bremsstrahlung_diffusion`: whether the energy diffusion `ν_∥ = Θγν_s` of `mode = :full`
  with `screened_diffusion = :maxwellian` takes the radiative stopping of `bremsstrahlung`.
  `:none` (the default, a choice of this package; see the docs page *Choices of this package*):
  the radiative stopping is a pure drag, as in the model of M. Hoppe et al., Comput. Phys.
  Commun. 268, 108098 (2021), doi:10.1016/j.cpc.2021.108098, eqs. 8, 9, 14 and 15, so a
  Maxwellian at the temperature of the bulk loses its bremsstrahlung power, which tends to that
  of Gould's formula at low temperature ([`GOULD_BREMSSTRAHLUNG_PREFACTOR`](@ref)). `:dream`:
  the code of DREAM (`src/Equations/ParallelDiffusionFrequency.cpp:59-66`), whose `ν_∥`
  rescales the whole `ν_s`, so the radiative drag gets a thermal energy diffusion, a Maxwellian
  at the temperature of the bulk radiates nothing, and a tail at `Θ_eff` loses `1 − Θ/Θ_eff` of
  its bremsstrahlung. The DREAM comparisons with the stopping power on take `:dream`.
- `bremsstrahlung_constant`: the constant term of the formula 4BN in the stopping power,
  `:koch_motz` (the default), `4/3` as in Koch and Motz, or `:dream`, `1` as DREAM's integer
  division writes it (`src/Equations/SlowingDownFrequency.cpp:234`), for the comparisons with
  DREAM that run with the stopping power on ([`bremsstrahlung_cross_section`](@ref)).
- `bound_friction`: the slowing down on the bound electrons of each charge state with
  partial screening. `:bethe` (the default, DREAM's model) is the Bethe term of Hesslow et al.
  with the weight of `screening_window`. `:ionization_split` splits it into the energy loss of
  the ionization of the charge state, from the cross section of the kinetic ionization
  ([`ionization_cross_section`](@ref)), and the rest, the excitation, floored at zero:

  ```math
  ν_j = e(p)\\, ν_{\\mathrm{ion},j} + \\max\\left(w_j(p)\\, ν_{\\mathrm{Bethe},j} − e(p)\\, ν_{\\mathrm{ion},j},\\, 0\\right),
  \\qquad ν_{\\mathrm{ion},j} = \\frac{γ}{p^2} n_j σ_j(p) v \\frac{I_j + ⟨E_s⟩_j}{m_e c^2},
  ```

  with the ionization energy `I_j` of NIST ([`ionization_energy`](@ref)), the mean energy of
  every secondary `⟨E_s⟩_j` up to `(E − I_j)/2` in the spectrum `secondary_spectrum` (no cut:
  every secondary joins the cold electrons) and the envelope `e(p)` of the hot region of the
  kinetic ionization. `f` then pays `I_j + ⟨E_s⟩_j` for each ionization it causes, also from
  the ionization potential up to the mean excitation energy, where the default
  `screening_window` gives no Bethe friction. Elements need a fit of the cross section, and
  with `secondary_spectrum = :opal` a width of the secondary spectrum
  ([`SECONDARY_SPECTRUM_WIDTHS`](@ref), `secondary_widths`). With the momentum faces of each radius given to the model
  (`CollisionModel(plasma, settings; faces)`, as `set_fokker_planck_coefficients!` does), the
  ionization part is tabulated once per momentum grid and charge state at the faces.
- `ionization_threshold`: the factor of the thermal momentum `p₀ = factor √(2T/mₑc²)` at the
  threshold of the kinetic ionization, which the region of the
  [`KineticIonizationTerm`](@ref) of the system takes as well; 0 (the default) for a kinetic
  ionization over the whole grid. Below the threshold the ionization is that of the cold
  electrons, whose binding energy the cold electrons pay ([`RadiatedPowerTerm`](@ref) with
  `kinetic_ionization`), so `f` pays none there. The envelope `e(p)` of the faces is a cubic
  smooth step from `(1 − ionization_threshold_width) p₀` to
  `(1 + ionization_threshold_width) p₀`, a numerical width that keeps the coefficients smooth
  when the threshold crosses a face. The default is 0.1 (a choice of this package; see the docs
  page *Choices of this package*); DREAM has no counterpart. With `screened_diffusion =
  :maxwellian` the energy diffusion `Θγν_s` takes the split as well, as it takes the Bethe
  term.
- `secondary_widths`: widths `Ē` [eV] of the secondary spectrum per element, added to
  [`SECONDARY_SPECTRUM_WIDTHS`](@ref) for `:ionization_split`.
- `secondary_spectrum`: the spectrum of the secondaries of `⟨E_s⟩_j`: `:beb` (the default, a
  choice of this package; see the docs page *Choices of this package*), the
  binary-encounter-Bethe spectrum of Kim and Rudd with the ionization energy of each charge
  state as its binding energy ([`beb_mean_secondary_energy`](@ref)), which needs no width;
  `:opal`, Opal et al. with the width of the neutral element for every charge state
  ([`mean_secondary_energy`](@ref)).
- `ionization_cross_section_form`: the `form` of [`ionization_cross_section`](@ref) in the
  friction of `bound_friction = :ionization_split`: `:chianti` (the default),
  `:dream` or `:bethe`. A system pairs it with the `ionization_cross_section_form` of its
  [`KineticIonizationTerm`](@ref), so `f` pays for the ionizations that term counts.
"""
Base.@kwdef struct CollisionSettings
    mode::Symbol = :full
    screening::Symbol = :partial
    coulomb_logarithm::Symbol = :thermal
    screened_diffusion::Symbol = :maxwellian
    ion_size_table::Symbol = :hesslow
    screening_window::Symbol = :excitation_energy
    screening_window_start::Float64 = 1.0
    screening_window_end::Float64 = 4.0
    bremsstrahlung::Symbol = :stopping_power
    bremsstrahlung_diffusion::Symbol = :none
    bremsstrahlung_constant::Symbol = :koch_motz
    bound_friction::Symbol = :bethe
    ionization_threshold::Float64 = 0.0
    ionization_threshold_width::Float64 = 0.1
    secondary_widths::Dict{String, Float64} = Dict{String, Float64}()
    secondary_spectrum::Symbol = :beb
    ionization_cross_section_form::Symbol = :chianti
    function CollisionSettings(
            mode, screening, coulomb_logarithm, screened_diffusion, ion_size_table,
            screening_window, screening_window_start, screening_window_end, bremsstrahlung,
            bremsstrahlung_diffusion, bremsstrahlung_constant, bound_friction, ionization_threshold,
            ionization_threshold_width, secondary_widths, secondary_spectrum,
            ionization_cross_section_form)
        mode in (:superthermal, :full, :ultra_relativistic) ||
            throw(ArgumentError("unknown collision frequency mode $mode"))
        screening in (:complete, :none, :partial) ||
            throw(ArgumentError("unknown screening model $screening"))
        coulomb_logarithm in (:energy_dependent, :thermal, :relativistic) ||
            throw(ArgumentError("unknown Coulomb logarithm model $coulomb_logarithm"))
        screened_diffusion in (:maxwellian, :zero) ||
            throw(ArgumentError("unknown screened diffusion mode $screened_diffusion"))
        ion_size_table in (:hesslow, :dream) ||
            throw(ArgumentError("unknown ion size table $ion_size_table"))
        screening_window in (:excitation_energy, :dream) ||
            throw(ArgumentError("unknown screening window $screening_window"))
        0 <= screening_window_start < screening_window_end ||
            throw(ArgumentError("the screening window needs 0 ≤ start < end, got " *
                                "$screening_window_start and $screening_window_end"))
        bremsstrahlung in (:neglect, :stopping_power) ||
            throw(ArgumentError("unknown bremsstrahlung mode $bremsstrahlung"))
        bremsstrahlung_constant in (:koch_motz, :dream) ||
            throw(ArgumentError("unknown bremsstrahlung constant $bremsstrahlung_constant (:koch_motz or :dream)"))
        bremsstrahlung_diffusion in (:none, :dream) ||
            throw(ArgumentError("unknown bremsstrahlung diffusion $bremsstrahlung_diffusion (:none or :dream)"))
        bound_friction in (:bethe, :ionization_split) ||
            throw(ArgumentError("unknown bound-electron friction $bound_friction (:bethe or :ionization_split)"))
        bound_friction === :ionization_split && screening !== :partial &&
            throw(ArgumentError("bound_friction = :ionization_split splits the friction on bound electrons of " *
                                "partial screening, not of screening = :$screening"))
        ionization_threshold >= 0 ||
            throw(ArgumentError("the ionization threshold is a nonnegative factor of the thermal momentum"))
        0 < ionization_threshold_width < 1 ||
            throw(ArgumentError("the width of the ionization threshold must lie between 0 and 1"))
        #! format: off
        widths = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in secondary_widths)
        #! format: on
        all(>(0), values(widths)) ||
            throw(ArgumentError("the widths of the secondary spectrum must be positive"))
        if !(secondary_spectrum in SECONDARY_SPECTRA)
            throw(ArgumentError("unknown secondary spectrum $secondary_spectrum (:opal or :beb)"))
        end
        if !(ionization_cross_section_form in IONIZATION_CROSS_SECTION_FORMS)
            throw(ArgumentError("unknown form of the ionization cross section $ionization_cross_section_form " *
                                "(:chianti, :dream or :bethe)"))
        end
        return new(mode, screening, coulomb_logarithm, screened_diffusion, ion_size_table,
            screening_window, screening_window_start, screening_window_end, bremsstrahlung,
            bremsstrahlung_diffusion, bremsstrahlung_constant, bound_friction, Float64(ionization_threshold),
            Float64(ionization_threshold_width), widths,
            secondary_spectrum, ionization_cross_section_form)
    end
end

# equality by the values of the fields: `secondary_widths` is a Dict, which the default
# comparison of an immutable struct would compare by identity
function Base.:(==)(a::CollisionSettings, b::CollisionSettings)
    all(name -> getfield(a, name) == getfield(b, name), fieldnames(CollisionSettings))
end
function Base.isequal(a::CollisionSettings, b::CollisionSettings)
    all(name -> isequal(getfield(a, name), getfield(b, name)), fieldnames(CollisionSettings))
end
function Base.hash(s::CollisionSettings, h::UInt)
    foldl((h, name) -> hash(getfield(s, name), h),
        fieldnames(CollisionSettings); init = hash(:CollisionSettings, h))
end

const COLLISION_PREFACTOR = 4pi * Constants.CLASSICAL_ELECTRON_RADIUS^2 *
                            Constants.SPEED_OF_LIGHT

# ∫₀^X ln(1 + x)/x dx = −Li₂(−X): the series below X = 1/2, the inversion
# −Li₂(−X) = π²/6 + ½ ln²X + Li₂(−1/X) above X = 2, the quadrature between
function _bremsstrahlung_integral(X::Real)
    X <= 0.5 && return sum((-1)^(k + 1) * X^k / k^2 for k in 1:60)
    X >= 2 && return π^2 / 6 + log(X)^2 / 2 + sum((-1)^k * X^(-k) / k^2 for k in 1:60)
    return quadgk(x -> log1p(x) / x, 0, X; rtol = 1e-14)[1]
end

"""
    bremsstrahlung_cross_section(p; constant=:koch_motz)

The radiative energy-loss cross section of an electron of momentum `p` (units of `mₑc`) on a
bare nucleus, in units of `α r₀² Z²`: the formula 4BN of H. W. Koch and J. W. Motz, Rev. Mod.
Phys. 31, 920 (1959), doi:10.1103/RevModPhys.31.920, without screening, as DREAM's
`src/Equations/SlowingDownFrequency.cpp:194-237`. It tends to `16/3` for `p → 0` (H. Bethe and
W. Heitler, Proc. R. Soc. Lond. A 146, 83 (1934), doi:10.1098/rspa.1934.0140) and to
`4(ln 2γ − 1/3)` for `γ ≫ 1` (W. Heitler, The Quantum Theory of Radiation, 3rd ed., 1954).

DREAM's line 234 writes the constant `4/3` of the formula as an integer division, which is 1
in C++, so DREAM's stopping power is 6.3 to 6.5 % above this one for `0.1 ≤ p ≤ 1`, 3.1 %
at `p = 10` and 1.7 % at `p = 100`. Runes2P keeps `4/3`;
`constant = :dream` gives DREAM's value from `p ≈ 0.1`, within about 0.02 %, for the comparisons
with DREAM. Below it `constant = :dream` does not reproduce DREAM: DREAM takes the integral
term `∫₀^X ln(1 + x)/x dx` from a Steffen spline on 50 points in `0 ≤ X ≤ 10`
(`SlowingDownFrequency.cpp:66-80`), and its stopping power is 3.2 % below that of
`constant = :dream` at `p = 0.001`, 2.8 % above this one. DREAM
returns 0 at `p = 0`; the limit is `16/3`, and the slowing-down frequency is 0 there anyway.
"""
function bremsstrahlung_cross_section(p::Real; constant::Symbol = :koch_motz)
    p == 0 && return 16 / 3 * one(p)
    gamma = sqrt(1 + p^2)
    gp = gamma * p
    log_term = log(gamma + p)
    X = 2p * (gamma + p)
    c = constant === :dream ? 1.0 : 4 / 3
    return (4 / 3) * (3gamma^2 + 1) / gp * log_term -
           (8gamma + 6p) / (3gp * p) * log_term^2 - c +
           2 / gp * _bremsstrahlung_integral(X)
end

"""
    mean_excitation_energy(Z, Z0)

Mean excitation energy [eV] of charge state `Z0` of element `Z` (DREAM
`SlowingDownFrequency::GetAtomicParameter`). Up to Z = 18 the tabulated values of Sauer,
Sabin and Oddershede (2018). For Z = 19 to 86 DREAM's extended table, a semi-empirical
electron-density model (DREAM `tools/Generate_MeanExcitationEnergy.py`), which is not
hydrogenic for one and two bound electrons: 5123.6 eV for K¹⁸⁺ against `14.99 Z²` =
5411 eV. Beyond Z = 86 the analytic formula of Sauer, Sabin and Oddershede.
"""
function mean_excitation_energy(Z::Integer, Z0::Integer)
    Z == Z0 && return NaN
    Z <= size(MEAN_EXCITATION_ENERGY, 1) && return MEAN_EXCITATION_ENERGY[Z, Z0 + 1]
    Z <= size(MEAN_EXCITATION_ENERGY_EXTENDED, 1) &&
        return MEAN_EXCITATION_ENERGY_EXTENDED[Z, Z - Z0]
    bound = Z - Z0
    nmax = length(MEAN_EXCITATION_ENERGY_D)
    if bound <= nmax
        D = MEAN_EXCITATION_ENERGY_D[bound]
        S0 = MEAN_EXCITATION_ENERGY_S0[bound]
    else
        D = MEAN_EXCITATION_ENERGY_D[nmax]
        S0 = bound - sqrt(bound * 10.0 / 14.99)
    end
    A = (1 - D)^2
    B = 2 * (1 - D) * (bound * D - S0)
    C = (bound * D - S0)^2
    return 14.99 * (A * Z^2 + B * Z + C)
end

"""
    ion_size(Z, Z0; table=:hesslow)

Effective ion size `ā_j` in units of the reduced Compton wavelength (`2/α · a_j`):
DFT values from Hesslow et al. (2018) Table 1 where available, else their eq. 2.28.
"""
function ion_size(Z::Integer, Z0::Integer; table::Symbol = :hesslow)
    charges = table === :dream ? ION_SIZE_Z_DREAM : ION_SIZE_Z
    for n in eachindex(ION_SIZE_Z0)
        if charges[n] == Z && ION_SIZE_Z0[n] == Z0
            return 2 / Constants.FINE_STRUCTURE_CONSTANT * ION_SIZE_A[n]
        end
    end
    return 2 / Constants.FINE_STRUCTURE_CONSTANT * cbrt(9pi * (Z - Z0)^2) / (4Z)
end

# One ion charge state with its atomic parameters.
struct ChargeState
    nuclear_charge::Int
    charge::Int
    species_index::Int
    mean_excitation_energy::Float64  # in units of mₑc²
    effective_size::Float64          # ā_j, see ion_size
    element::String
    ionization_energy::Float64       # [eV], NaN unless bound_friction = :ionization_split
    secondary_width::Float64         # Ē [eV] of the secondary spectrum, NaN unless :ionization_split
end

"""
    CollisionModel(plasma, settings)

Collision frequencies for a background plasma. Radius-dependent quantities (free and
bound electron densities, Bessel functions, atomic parameters of the charge states) are
evaluated once; the frequencies are then evaluated at any momentum with
[`slowing_down_frequency`](@ref), [`deflection_frequency`](@ref) and
[`parallel_diffusion_frequency`](@ref). The Coulomb logarithm is evaluated in each call,
since the energy-dependent model depends on the momentum.
"""
struct CollisionModel{T, P <: PlasmaState}
    plasma::P
    settings::CollisionSettings
    charge_states::Vector{ChargeState}
    free_electron_density::Vector{T}
    bound_electron_density::Vector{T}
    scaled_bessel_k0::Vector{T}   # e^{1/Θ} K_n(1/Θ), Θ = T/(mₑc²)
    scaled_bessel_k1::Vector{T}
    scaled_bessel_k2::Vector{T}
    # per radius: the momentum faces and the ionization part of the slowing down per unit
    # density at each face and charge state (bound_friction = :ionization_split), or nothing
    ionization_tables::Vector{Union{Nothing, Tuple{Vector{Float64}, Matrix{Float64}}}}
end

# the tables of the ionization part of the slowing down, per momentum grid: keyed
# by the vector of its faces, and within it by the spectrum and the charge states. The weak
# keys let a table go with its grid.
const IONIZATION_FRICTION_TABLES = WeakKeyDict{Vector{Float64}, Dict{Any, Matrix{Float64}}}()
const IONIZATION_FRICTION_LOCK = ReentrantLock()

# the table of _ionization_slowing_down at the faces `faces` for the charge states, columns
# in the order of `states`, from the cache or built once
function _ionization_friction_table(
        faces::Vector{Float64}, states::Vector{ChargeState}, spectrum::Symbol,
        form::Symbol)
    key = (spectrum, form,
        [(s.element, s.charge, s.ionization_energy, s.secondary_width) for s in states])
    lock(IONIZATION_FRICTION_LOCK) do
        tables = get!(() -> Dict{Any, Matrix{Float64}}(), IONIZATION_FRICTION_TABLES, faces)
        get!(tables, key) do
            [_ionization_slowing_down(s, p, spectrum, form) for p in faces, s in states]
        end
    end
end

function CollisionModel(plasma::PlasmaState{T}, settings::CollisionSettings; faces = nothing) where {T}
    charge_states = ChargeState[]
    split = settings.bound_friction === :ionization_split
    widths = split ? merge(SECONDARY_SPECTRUM_WIDTHS, settings.secondary_widths) : nothing
    for (index, s) in enumerate(plasma.ions)
        if split
            _check_ionization_split_data(atomic_element(s.name), widths, settings.secondary_spectrum)
        end
        for Z0 in 0:s.nuclear_charge
            excitation = mean_excitation_energy(s.nuclear_charge, Z0) /
                         Constants.ELECTRON_REST_ENERGY
            effective_size = if Z0 == s.nuclear_charge
                NaN
            else
                ion_size(s.nuclear_charge, Z0; table = settings.ion_size_table)
            end
            ionization = if split && Z0 < s.nuclear_charge
                ionization_energy(s.name, Z0)
            else
                NaN
            end
            width = if split && settings.secondary_spectrum === :opal
                widths[atomic_element(s.name)]
            else
                NaN
            end
            push!(charge_states,
                ChargeState(
                    s.nuclear_charge, Z0, index, excitation, effective_size, s.name,
                    ionization, width))
        end
    end
    nr = radial_cell_count(plasma)
    free = [free_electron_density(plasma, ir) for ir in 1:nr]
    bound = [bound_electron_density(plasma, ir) for ir in 1:nr]
    theta = plasma.cold_electron_temperature ./ Constants.ELECTRON_REST_ENERGY
    k0 = besselkx.(0, 1 ./ theta)
    k1 = besselkx.(1, 1 ./ theta)
    k2 = besselkx.(2, 1 ./ theta)
    tables = Union{Nothing, Tuple{Vector{Float64}, Matrix{Float64}}}[nothing for _ in 1:nr]
    if split && faces !== nothing
        if length(faces) != nr
            throw(DimensionMismatch("the momentum faces and the plasma have different radii"))
        end
        for ir in 1:nr
            f = faces[ir]
            if f isa Vector{Float64}
                tables[ir] = (f,
                    _ionization_friction_table(
                        f, charge_states, settings.secondary_spectrum,
                        settings.ionization_cross_section_form))
            end
        end
    end
    return CollisionModel{eltype(k0), typeof(plasma)}(
        plasma, settings, charge_states, free,
        bound, k0, k1, k2, tables)
end

#! format: off
@inline _charge_state_density(model::CollisionModel, state::ChargeState, ir) =
    model.plasma.ions[state.species_index].charge_state_densities[state.charge+1, ir]
#! format: on

function _normalized_temperature(model::CollisionModel, ir)
    model.plasma.cold_electron_temperature[ir] / Constants.ELECTRON_REST_ENERGY
end

function _coulomb_logarithm(model::CollisionModel, collision, p, ir)
    coulomb_logarithm(collision, model.settings.coulomb_logarithm, p,
        model.free_electron_density[ir], model.plasma.cold_electron_temperature[ir])
end

# Density of the target electrons of electron-electron collisions: the cold electrons,
# plus the bound electrons without screening.
function _target_electron_density(model::CollisionModel, ir)
    n = model.plasma.cold_electron_density[ir]
    if model.settings.screening === :none
        n += model.bound_electron_density[ir]
    end
    return max(n, zero(n))
end

# Thermal integrals Ψ_n(p) = ∫₀ᵖ γ(s)ⁿ⁻¹ exp(-(γ(s) - 1)/Θ) ds, following DREAM
# evaluatePsi0/1/2: asymptotic expansions in the superthermal and low-energy limits,
# adaptive quadrature otherwise.
function _psi(n::Int, model::CollisionModel, ir, p)
    theta = _normalized_temperature(model, ir)
    gamma = sqrt(1 + p^2)
    gamma_minus_one = p^2 / (gamma + 1)
    threshold = 10theta
    if p^2 > threshold * (threshold + 2)  # superthermal expansion
        e = exp(-gamma_minus_one / theta)
        if n == 0
            return model.scaled_bessel_k0[ir] + theta * (-e / p) +
                   theta^2 * (gamma / p^3 * e)
        elseif n == 1
            return model.scaled_bessel_k1[ir] + theta * (-gamma / p * e) +
                   theta^2 * (e / p^3)
        else
            return model.scaled_bessel_k0[ir] +
                   theta * (model.scaled_bessel_k1[ir] - gamma^2 / p * e) +
                   theta^2 * (-gamma * (gamma^2 - 2) / p^3 * e)
        end
    elseif p < 0.15 || theta < 0.005  # low-energy expansion
        x = sqrt(gamma_minus_one / theta)
        order1 = 4n - 1.0
        order2 = 2n^2 - 3n + 0.375
        psi = (1 + theta * order1 / 8 + theta^2 * 0.1875 * order2) * sqrt(pi) / 2 * erf(x)
        psi -= theta * x * exp(-x^2) * (order1 + theta * order2 * (3 + 2x^2) / 2) / 8
        return psi * sqrt(2theta)
    else
        integrand = s -> begin
            gamma_s = sqrt(1 + s^2)
            exp(-(s^2 / (1 + gamma_s)) / theta) * gamma_s^(n - 1)
        end
        return first(quadgk(integrand, zero(p), p; rtol = 1e-8))
    end
end

# Thermal correction of the electron slowing down (1 unless mode = :full).
# In :full mode at p ≪ p_th the numerator is a difference of O(p) terms while it is ∝ p³,
# so relative round-off grows like (p_th/p)²; kept identical to DREAM for comparisons.
function _thermal_slowing_down_factor(model::CollisionModel, ir, p)
    model.settings.mode === :full || return one(p)
    p == 0 && return zero(p)
    theta = _normalized_temperature(model, ir)
    gamma = sqrt(1 + p^2)
    gamma_minus_one = p^2 / (gamma + 1)
    numerator = gamma^2 * _psi(1, model, ir, p) - theta * _psi(0, model, ir, p) +
                (theta * gamma - 1) * p * exp(-gamma_minus_one / theta)
    return numerator / (gamma^2 * model.scaled_bessel_k2[ir])
end

# Thermal correction of the electron pitch-angle deflection (1 unless mode = :full).
function _thermal_deflection_factor(model::CollisionModel, ir, p)
    model.settings.mode === :full || return one(p)
    p == 0 && return zero(p)
    theta = _normalized_temperature(model, ir)
    p2 = p^2
    gamma = sqrt(1 + p2)
    gamma_minus_one = p2 / (gamma + 1)
    numerator = (p2 * gamma^2 + theta^2) * _psi(0, model, ir, p)
    numerator += theta * (2p2^2 - 1) * _psi(1, model, ir, p)
    numerator += gamma * theta * (1 + theta * (2p2 - 1)) * p * exp(-gamma_minus_one / theta)
    return numerator / (p2 * gamma^2 * model.scaled_bessel_k2[ir])
end

# Bound-electron stopping power, matched Bethe formula (Hesslow et al. eq. 2.31 with
# DREAM's modification that keeps it positive).
function _bound_electron_stopping(model::CollisionModel, state::ChargeState, p)
    bound = state.nuclear_charge - state.charge
    bound == 0 && return zero(p)
    p2 = p^2
    gamma = sqrt(1 + p2)
    beta2 = p2 / (1 + p2)
    h = (p2 / sqrt(1 + gamma)) / state.mean_excitation_energy
    if model.settings.mode === :full
        k = COULOMB_LOGARITHM_INTERPOLATION_EXPONENT
        return bound * log(1 + (h * exp(-beta2))^k) / k
    else
        return bound * log(exp(1) + h * exp(-beta2))
    end
end

# Weight of the bound-electron parts of charge state `state` at momentum `p`: a cubic
# smooth step in the kinetic energy between the two factors of the mean excitation energy
# (CollisionSettings, `screening_window`). Exactly 0 below and exactly 1 above the window.
function _screening_weight(settings::CollisionSettings, state::ChargeState, p)
    settings.screening_window === :dream && return one(p)
    kinetic_energy = p^2 / (sqrt(1 + p^2) + 1)
    lower = settings.screening_window_start * state.mean_excitation_energy
    upper = settings.screening_window_end * state.mean_excitation_energy
    s = clamp((kinetic_energy - lower) / (upper - lower), zero(p), one(p))
    return s^2 * (3 - 2s)
end

# refuses an element without the data of bound_friction = :ionization_split
function _check_ionization_split_data(element, widths, spectrum)
    if !haskey(KINETIC_IONIZATION_FITS, element)
        known_fits = join(sort!(collect(keys(KINETIC_IONIZATION_FITS))), ", ")
        throw(ArgumentError("bound_friction = :ionization_split needs a fit of the ionization cross sections of " *
                            "$element, which DREAM's tables lack ($(known_fits))"))
    end
    spectrum === :beb || haskey(widths, element) ||
        throw(ArgumentError("bound_friction = :ionization_split needs the width of the secondary spectrum of " *
                            "$element: give it with secondary_widths = Dict(\"$element\" => Ē in eV)"))
    return nothing
end

# Envelope e(p) of the hot region of the kinetic ionization at a face: 1 without a
# threshold, otherwise a cubic smooth step in p around p₀ = factor √(2T/mₑc²)
# (CollisionSettings, `ionization_threshold`), in the number type of the temperature.
function _ionization_envelope(model::CollisionModel, ir, p)
    factor = model.settings.ionization_threshold
    T = model.plasma.cold_electron_temperature[ir]
    factor == 0 && return one(p * T)
    p0 = factor * sqrt(2 * T / Constants.ELECTRON_REST_ENERGY)
    width = model.settings.ionization_threshold_width
    s = clamp((p - (1 - width) * p0) / (2width * p0), zero(p0), one(p0))
    return s^2 * (3 - 2s)
end

# Slowing-down frequency [s⁻¹ per m⁻³ of the charge state] of the energy loss of the
# ionization of `state`, γ σ v (I + ⟨E_s⟩)/(mₑc² p²), with the mean energy of every secondary
# (bound_friction = :ionization_split). The energy
# loss rate n σ v (I + ⟨E_s⟩) of an electron equals mₑc² p² ν/γ.
function _ionization_slowing_down(state::ChargeState, p, spectrum, form)
    state.charge == state.nuclear_charge && return zero(p)
    sigma = ionization_cross_section(state.element, state.charge, p; form)
    sigma == 0 && return zero(p)
    gamma = sqrt(1 + p^2)
    energy = p^2 / (gamma + 1) * Constants.ELECTRON_REST_ENERGY      # kinetic energy [eV]
    #! format: off
    loss = state.ionization_energy +
           secondary_spectrum_mean(spectrum, state.secondary_width, energy, state.ionization_energy, Inf)
    #! format: on
    speed = Constants.SPEED_OF_LIGHT * p / gamma
    return gamma * sigma * speed * loss / (Constants.ELECTRON_REST_ENERGY * p^2)
end

# the slowing down on the bound electrons of charge state `state` per m⁻³ of it at radius
# ir, in units of the prefactor of ν_s: the Bethe term with its screening weight, or with
# bound_friction = :ionization_split the ionization part above the threshold and the rest,
# floored at zero. The Bethe path keeps the order of the arithmetic of the port of DREAM's
# formula, so that its values do not change when the split is added.
function _bound_slowing_down(
        model::CollisionModel, index, state::ChargeState, ir, p, prefactor)
    bethe = _screening_weight(model.settings, state, p) *
            _bound_electron_stopping(model, state, p)
    # one number type on both branches: the ionization part carries the partials of the plasma,
    # the Bethe term does not
    number_type = promote_type(typeof(bethe), eltype(model.plasma.cold_electron_temperature))
    model.settings.bound_friction === :ionization_split ||
        return convert(number_type, bethe)
    ionization = _ionization_envelope(model, ir, p) *
                 _ionization_part(model, index, state, ir, p) / prefactor
    return ionization + max(bethe - ionization, zero(ionization))
end

# the ionization part of the slowing down per unit density of charge state `index` at the
# momentum p: from the table of the grid where p is one of its faces, computed
# otherwise; both give the same number
function _ionization_part(model::CollisionModel, index, state::ChargeState, ir, p::Float64)
    table = model.ionization_tables[ir]
    if table !== nothing
        faces, values = table
        k = searchsortedfirst(faces, p)
        k <= length(faces) && faces[k] == p && return values[k, index]
    end
    return _ionization_slowing_down(state, p, model.settings.secondary_spectrum,
        model.settings.ionization_cross_section_form)
end
function _ionization_part(model::CollisionModel, index, state::ChargeState, ir, p)
    _ionization_slowing_down(state, p, model.settings.secondary_spectrum,
        model.settings.ionization_cross_section_form)
end

# Bound-electron contribution to pitch-angle scattering (Hesslow et al. eq. 2.25).
function _bound_electron_scattering(state::ChargeState, p)
    bound = state.nuclear_charge - state.charge
    bound == 0 && return zero(p)
    x = (p * state.effective_size)^1.5
    return 2 / 3 * bound *
           ((state.nuclear_charge + state.charge) * log(1 + x) - bound * x / (1 + x))
end

"""
    slowing_down_frequency(model, ir, p)

Slowing-down frequency `ν_s` [s⁻¹] at radius index `ir` and momentum `p`; the collisional
friction is `Aᵖ = -p ν_s`. With `bremsstrahlung = :stopping_power` it adds, per charge state
of density `n` and nuclear charge `Z`, `n Z² α r₀² c φ(p) γ/p` with the cross section `φ` of
[`bremsstrahlung_cross_section`](@ref), as DREAM's `src/Equations/SlowingDownFrequency.cpp:204-207`
and `:236` (where DREAM's `φ` differs by its integer division, see there).
"""
function slowing_down_frequency(model::CollisionModel, ir, p)
    _collisional_slowing_down(model, ir, p) + _radiative_slowing_down(model, ir, p)
end

# the collisional part of ν_s: the free and, with partial screening, the bound electrons
function _collisional_slowing_down(model::CollisionModel, ir, p)
    p == 0 && return zero(p * model.scaled_bessel_k0[ir])
    mode = model.settings.mode
    prefactor = if mode === :ultra_relativistic
        COLLISION_PREFACTOR / p
    else
        COLLISION_PREFACTOR * (1 + p^2) / p^3
    end
    nu = _coulomb_logarithm(model, :electron_electron, p, ir) *
         _thermal_slowing_down_factor(model, ir, p) * _target_electron_density(model, ir)
    if model.settings.screening === :partial
        for (index, state) in enumerate(model.charge_states)
            state.charge == state.nuclear_charge && continue
            nu += _bound_slowing_down(model, index, state, ir, p, prefactor) *
                  _charge_state_density(model, state, ir)
        end
    end
    return prefactor * nu
end

# the radiative part of ν_s with bremsstrahlung = :stopping_power, on the nuclei of all
# charge states, ν_s = n Z² α r₀² c φ_rad/β; zero otherwise
function _radiative_slowing_down(model::CollisionModel, ir, p)
    nu = zero(p * model.scaled_bessel_k0[ir])
    (p == 0 || model.settings.bremsstrahlung !== :stopping_power) && return nu
    radiative = COLLISION_PREFACTOR * Constants.FINE_STRUCTURE_CONSTANT / (4π) *
                sqrt(1 + p^2) / p *
                bremsstrahlung_cross_section(p; constant = model.settings.bremsstrahlung_constant)
    for state in model.charge_states
        nu += radiative * state.nuclear_charge^2 * _charge_state_density(model, state, ir)
    end
    return nu
end

"""
    deflection_frequency(model, ir, p)

Pitch-angle scattering (deflection) frequency `ν_D` [s⁻¹]; the collisional pitch diffusion is
`Dᵠᵠ = ν_D (1 - ξ²)/2`.
"""
function deflection_frequency(model::CollisionModel, ir, p)
    p == 0 && return zero(p * model.scaled_bessel_k0[ir])
    mode = model.settings.mode
    prefactor = if mode === :ultra_relativistic
        COLLISION_PREFACTOR / p^2
    else
        COLLISION_PREFACTOR * sqrt(1 + p^2) / p^3
    end
    nu = _coulomb_logarithm(model, :electron_electron, p, ir) *
         _thermal_deflection_factor(model, ir, p) * _target_electron_density(model, ir)
    electron_ion = _coulomb_logarithm(model, :electron_ion, p, ir)
    unscreened = model.settings.screening === :none
    for state in model.charge_states
        charge_squared = unscreened ? state.nuclear_charge^2 : state.charge^2
        nu += electron_ion * charge_squared * _charge_state_density(model, state, ir)
    end
    if model.settings.screening === :partial
        for state in model.charge_states
            state.charge == state.nuclear_charge && continue
            nu += _screening_weight(model.settings, state, p) *
                  _bound_electron_scattering(state, p) *
                  _charge_state_density(model, state, ir)
        end
    end
    return prefactor * nu
end

"""
    parallel_diffusion_frequency(model, ir, p)

Parallel (energy) diffusion frequency `ν_∥` [s⁻¹], `Dᵖᵖ = ν_∥`; nonzero only in `:full`
mode. With `screened_diffusion = :maxwellian` it is `Θγ` times the collisional part of `ν_s`,
and with `bremsstrahlung_diffusion = :dream` times the whole `ν_s` ([`CollisionSettings`](@ref)).
"""
function parallel_diffusion_frequency(model::CollisionModel, ir, p)
    z = zero(p * model.scaled_bessel_k0[ir])
    (model.settings.mode === :full && p != 0) || return z
    theta = _normalized_temperature(model, ir)
    nu_s = if model.settings.screened_diffusion === :maxwellian
        if model.settings.bremsstrahlung_diffusion === :dream
            slowing_down_frequency(model, ir, p)
        else
            _collisional_slowing_down(model, ir, p)
        end
    else  # free electrons only
        COLLISION_PREFACTOR * (1 + p^2) / p^3 *
        _coulomb_logarithm(model, :electron_electron, p, ir) *
        _thermal_slowing_down_factor(model, ir, p) *
        model.plasma.cold_electron_density[ir]
    end
    return theta * sqrt(1 + p^2) * nu_s
end

"""
    p3_slowing_down_frequency_at_zero(model, ir)

`lim_{p→0} p³ ν_s`, which sets the particle flux through `p = 0`. As DREAM's
`SlowingDownFrequency::GetP3NuSAtZero`, the value uses the prefactor `4π r₀² c` in every
mode. In `:ultra_relativistic` mode, where `ν_s ∝ 1/p` and the analytic limit is 0, it
therefore returns the `:superthermal` value. The friction of that mode is the constant
drift `Aᵖ = -4π r₀² c lnΛ n`, which brings electrons to `p = 0` in a finite time, and
DREAM's value lets them leave the grid there: the sink through `p = 0` is `1/Δp₁²` times
the inflow through the first interior face at equal `f`, so the first cell absorbs. With
the analytic limit the first cell would collect them without bound. DREAM runs the mode
on a runaway grid from `p = 0` (`examples/init_f_re`).
"""
function p3_slowing_down_frequency_at_zero(model::CollisionModel, ir)
    z = zero(model.scaled_bessel_k0[ir])
    nu = _coulomb_logarithm(model, :electron_electron, z, ir) *
         _thermal_slowing_down_factor(model, ir, z) * _target_electron_density(model, ir)
    if model.settings.screening === :partial
        for state in model.charge_states
            state.charge == state.nuclear_charge && continue
            nu += _screening_weight(model.settings, state, z) *
                  _bound_electron_stopping(model, state, z) *
                  _charge_state_density(model, state, ir)
        end
    end
    return COLLISION_PREFACTOR * nu
end
