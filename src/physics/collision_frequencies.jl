# Collision frequencies of the linearized relativistic test-particle operator, ported
# from DREAM (CollisionFrequency.cpp, SlowingDownFrequency.cpp,
# PitchScatterFrequency.cpp, ParallelDiffusionFrequency.cpp). Partial screening
# follows Hesslow et al., J. Plasma Phys. 84 (2018); the thermal electron terms
# follow Pike & Rose, Phys. Rev. E 89 (2014).

using SpecialFunctions: besselkx, erf
using QuadGK: quadgk

"""
    CollisionSettings(; mode, screening, coulomb_logarithm, screened_diffusion, ion_size_table)

Defaults are those of DREAM's `CollisionHandler`.

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
  the frequencies are those of complete screening. The Born basis of Hesslow et al. holds
  only for fast electrons, and the default factors 1 and 4 place the window of Ar⁺ and
  Ar²⁺ at p ≈ 0.03 to 0.06 (Hesslow et al. PRL 2017 and JPP 2019). `:dream` applies
  partial screening at all momenta, as DREAM does. The weight depends only on `p` and
  `I_j`, so it holds unchanged when the densities become unknowns.
- `bremsstrahlung`: `:stopping_power`, the radiative energy loss of an electron on the
  nuclei of all charge states as a slowing down (DREAM's
  `BREMSSTRAHLUNG_MODE_STOPPING_POWER`): the non-screened formula 4BN of
  Koch and Motz (Rev. Mod. Phys. 31, 920, 1959), `ν_s = n Z² α r₀² c φ_rad(p)/β` per
  charge state ([`bremsstrahlung_cross_section`](@ref)). It is the default (decision of
  the author). `:neglect` leaves the loss out, as DREAM does by default.
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
    function CollisionSettings(mode, screening, coulomb_logarithm, screened_diffusion, ion_size_table,
                               screening_window, screening_window_start, screening_window_end, bremsstrahlung)
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
        return new(mode, screening, coulomb_logarithm, screened_diffusion, ion_size_table,
                   screening_window, screening_window_start, screening_window_end, bremsstrahlung)
    end
end

const COLLISION_PREFACTOR = 4pi * Constants.CLASSICAL_ELECTRON_RADIUS^2 * Constants.SPEED_OF_LIGHT

# ∫₀^X ln(1 + x)/x dx = −Li₂(−X): the series below X = 1/2, the inversion
# −Li₂(−X) = π²/6 + ½ ln²X + Li₂(−1/X) above X = 2, the quadrature between
function _bremsstrahlung_integral(X::Real)
    X <= 0.5 && return sum((-1)^(k + 1) * X^k / k^2 for k in 1:60)
    X >= 2 && return π^2 / 6 + log(X)^2 / 2 + sum((-1)^k * X^(-k) / k^2 for k in 1:60)
    return quadgk(x -> log1p(x) / x, 0, X; rtol=1e-14)[1]
end

"""
    bremsstrahlung_cross_section(p)

The radiative energy-loss cross section of an electron of momentum `p` (units of `mₑc`) on a
bare nucleus, in units of `α r₀² Z²`: the formula 4BN of Koch and Motz (Rev. Mod. Phys. 31,
920, 1959) without screening, as DREAM's `SlowingDownFrequency`. It tends to `16/3` for
`p → 0` (Bethe and Heitler) and to `4(ln 2γ − 1/3)` for `γ ≫ 1` (Heitler).
"""
function bremsstrahlung_cross_section(p::Real)
    p == 0 && return 16 / 3 * one(p)
    gamma = sqrt(1 + p^2)
    gp = gamma * p
    log_term = log(gamma + p)
    X = 2p * (gamma + p)
    return (4 / 3) * (3gamma^2 + 1) / gp * log_term - (8gamma + 6p) / (3gp * p) * log_term^2 - 4 / 3 +
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
    Z <= size(MEAN_EXCITATION_ENERGY, 1) && return MEAN_EXCITATION_ENERGY[Z, Z0+1]
    Z <= size(MEAN_EXCITATION_ENERGY_EXTENDED, 1) && return MEAN_EXCITATION_ENERGY_EXTENDED[Z, Z-Z0]
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
function ion_size(Z::Integer, Z0::Integer; table::Symbol=:hesslow)
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
struct CollisionModel{T,P<:PlasmaState}
    plasma::P
    settings::CollisionSettings
    charge_states::Vector{ChargeState}
    free_electron_density::Vector{T}
    bound_electron_density::Vector{T}
    scaled_bessel_k0::Vector{T}   # e^{1/Θ} K_n(1/Θ), Θ = T/(mₑc²)
    scaled_bessel_k1::Vector{T}
    scaled_bessel_k2::Vector{T}
end

function CollisionModel(plasma::PlasmaState{T}, settings::CollisionSettings) where {T}
    charge_states = ChargeState[]
    for (index, s) in enumerate(plasma.ions), Z0 in 0:s.nuclear_charge
        excitation = mean_excitation_energy(s.nuclear_charge, Z0) / Constants.ELECTRON_REST_ENERGY
        effective_size = Z0 == s.nuclear_charge ? NaN :
                         ion_size(s.nuclear_charge, Z0; table=settings.ion_size_table)
        push!(charge_states, ChargeState(s.nuclear_charge, Z0, index, excitation, effective_size))
    end
    nr = radial_cell_count(plasma)
    free = [free_electron_density(plasma, ir) for ir in 1:nr]
    bound = [bound_electron_density(plasma, ir) for ir in 1:nr]
    theta = plasma.cold_electron_temperature ./ Constants.ELECTRON_REST_ENERGY
    k0 = besselkx.(0, 1 ./ theta)
    k1 = besselkx.(1, 1 ./ theta)
    k2 = besselkx.(2, 1 ./ theta)
    return CollisionModel{eltype(k0),typeof(plasma)}(plasma, settings, charge_states, free,
                                                      bound, k0, k1, k2)
end

@inline _charge_state_density(model::CollisionModel, state::ChargeState, ir) =
    model.plasma.ions[state.species_index].charge_state_densities[state.charge+1, ir]

_normalized_temperature(model::CollisionModel, ir) =
    model.plasma.cold_electron_temperature[ir] / Constants.ELECTRON_REST_ENERGY

_coulomb_logarithm(model::CollisionModel, collision, p, ir) =
    coulomb_logarithm(collision, model.settings.coulomb_logarithm, p,
                      model.plasma.cold_electron_temperature[ir], model.free_electron_density[ir])

# Density of the target electrons of electron-electron collisions: the cold electrons,
# plus the bound electrons without screening.
function _target_electron_density(model::CollisionModel, ir)
    n = model.plasma.cold_electron_density[ir]
    model.settings.screening === :none && (n += model.bound_electron_density[ir])
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
            return model.scaled_bessel_k0[ir] + theta * (-e / p) + theta^2 * (gamma / p^3 * e)
        elseif n == 1
            return model.scaled_bessel_k1[ir] + theta * (-gamma / p * e) + theta^2 * (e / p^3)
        else
            return model.scaled_bessel_k0[ir] + theta * (model.scaled_bessel_k1[ir] - gamma^2 / p * e) +
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
        return first(quadgk(integrand, zero(p), p; rtol=1e-8))
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

# Bound-electron contribution to pitch-angle scattering (Hesslow et al. eq. 2.25).
function _bound_electron_scattering(state::ChargeState, p)
    bound = state.nuclear_charge - state.charge
    bound == 0 && return zero(p)
    x = (p * state.effective_size)^1.5
    return 2 / 3 * bound * ((state.nuclear_charge + state.charge) * log(1 + x) - bound * x / (1 + x))
end

"""
    slowing_down_frequency(model, ir, p)

Slowing-down frequency `ν_s` [s⁻¹] at radius index `ir` and momentum `p`; the collisional
friction is `Aᵖ = -p ν_s`.
"""
function slowing_down_frequency(model::CollisionModel, ir, p)
    p == 0 && return zero(p * model.scaled_bessel_k0[ir])
    mode = model.settings.mode
    prefactor = mode === :ultra_relativistic ? COLLISION_PREFACTOR / p :
                COLLISION_PREFACTOR * (1 + p^2) / p^3
    nu = _coulomb_logarithm(model, :electron_electron, p, ir) *
         _thermal_slowing_down_factor(model, ir, p) * _target_electron_density(model, ir)
    if model.settings.screening === :partial
        for state in model.charge_states
            state.charge == state.nuclear_charge && continue
            nu += _screening_weight(model.settings, state, p) * _bound_electron_stopping(model, state, p) *
                  _charge_state_density(model, state, ir)
        end
    end
    nu = prefactor * nu
    if model.settings.bremsstrahlung === :stopping_power
        # the radiative stopping on the nuclei of all charge states, ν_s = n Z² α r₀² c φ_rad/β
        radiative = COLLISION_PREFACTOR * Constants.FINE_STRUCTURE_CONSTANT / (4π) * sqrt(1 + p^2) / p *
                    bremsstrahlung_cross_section(p)
        for state in model.charge_states
            nu += radiative * state.nuclear_charge^2 * _charge_state_density(model, state, ir)
        end
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
    prefactor = mode === :ultra_relativistic ? COLLISION_PREFACTOR / p^2 :
                COLLISION_PREFACTOR * sqrt(1 + p^2) / p^3
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
            nu += _screening_weight(model.settings, state, p) * _bound_electron_scattering(state, p) *
                  _charge_state_density(model, state, ir)
        end
    end
    return prefactor * nu
end

"""
    parallel_diffusion_frequency(model, ir, p)

Parallel (energy) diffusion frequency `ν_∥` [s⁻¹], `Dᵖᵖ = ν_∥`; nonzero only in `:full`
mode.
"""
function parallel_diffusion_frequency(model::CollisionModel, ir, p)
    z = zero(p * model.scaled_bessel_k0[ir])
    (model.settings.mode === :full && p != 0) || return z
    theta = _normalized_temperature(model, ir)
    if model.settings.screened_diffusion === :maxwellian
        nu_s = slowing_down_frequency(model, ir, p)
    else  # free electrons only
        nu_s = COLLISION_PREFACTOR * (1 + p^2) / p^3 *
               _coulomb_logarithm(model, :electron_electron, p, ir) *
               _thermal_slowing_down_factor(model, ir, p) * model.plasma.cold_electron_density[ir]
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
            nu += _screening_weight(model.settings, state, z) * _bound_electron_stopping(model, state, z) *
                  _charge_state_density(model, state, ir)
        end
    end
    return COLLISION_PREFACTOR * nu
end
