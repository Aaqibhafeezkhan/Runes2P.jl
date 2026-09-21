# Collision frequencies of the linearized relativistic test-particle operator, ported
# from DREAM (CollisionFrequency.cpp, SlowingDownFrequency.cpp,
# PitchScatterFrequency.cpp, ParallelDiffusionFrequency.cpp). Partial screening
# follows Hesslow et al., J. Plasma Phys. 84 (2018); the thermal electron terms
# follow Pike & Rose, Phys. Rev. E 89 (2014).

using SpecialFunctions: besselkx, erf
using QuadGK: quadgk

"""
    CollisionSettings(; mode, screening, lnlambda, screened_diffusion, ion_size_table)

Defaults are those of DREAM's `CollisionHandler`.

- `mode`: `:full` (relativistic Maxwellian targets, adds energy diffusion),
  `:superthermal` (cold target electrons) or `:ultra_relativistic`.
- `screening`: `:partially` (Hesslow et al. 2018), `:completely` or `:non`.
- `lnlambda`: `:thermal`, `:energy_dependent` or `:constant`.
- `screened_diffusion`: energy diffusion from the full `ν_s` (`:maxwellian`) or from
  the free-electron part only (`:zero`).
- `ion_size_table`: `:hesslow` uses Table 1 of Hesslow et al. 2018 as intended;
  `:dream` reproduces DREAM's lookup, whose charge list has one argon entry too few
  (affects Ar¹⁷⁺, Xe³⁺, Xe¹⁷⁺, W³⁺ and W⁶⁰⁺).
"""
Base.@kwdef struct CollisionSettings
    mode::Symbol = :full
    screening::Symbol = :partially
    lnlambda::Symbol = :thermal
    screened_diffusion::Symbol = :maxwellian
    ion_size_table::Symbol = :hesslow
    function CollisionSettings(mode, screening, lnlambda, screened_diffusion, ion_size_table)
        mode in (:superthermal, :full, :ultra_relativistic) ||
            throw(ArgumentError("unknown collision frequency mode $mode"))
        screening in (:completely, :non, :partially) ||
            throw(ArgumentError("unknown screening model $screening"))
        lnlambda in (:energy_dependent, :thermal, :constant) ||
            throw(ArgumentError("unknown lnΛ model $lnlambda"))
        screened_diffusion in (:maxwellian, :zero) ||
            throw(ArgumentError("unknown screened diffusion mode $screened_diffusion"))
        ion_size_table in (:hesslow, :dream) ||
            throw(ArgumentError("unknown ion size table $ion_size_table"))
        return new(mode, screening, lnlambda, screened_diffusion, ion_size_table)
    end
end

const COLLISION_PREFACTOR = 4π * Constants.r0^2 * Constants.c

"""
    mean_excitation_energy(Z, Z0)

Mean excitation energy [eV] of charge state `Z0` of element `Z` (DREAM
`SlowingDownFrequency::GetAtomicParameter`): tabulated up to Z = 86, analytic
formula of Sauer, Sabin and Oddershede (2018) beyond.
"""
function mean_excitation_energy(Z::Integer, Z0::Integer)
    Z == Z0 && return NaN
    Z <= size(MEAN_EXCITATION_ENERGY, 1) && return MEAN_EXCITATION_ENERGY[Z, Z0+1]
    Z <= size(MEAN_EXCITATION_ENERGY_EXTENDED, 1) && return MEAN_EXCITATION_ENERGY_EXTENDED[Z, Z-Z0]
    Ne = Z - Z0
    nmax = length(MEAN_EXCITATION_ENERGY_D)
    if Ne <= nmax
        D = MEAN_EXCITATION_ENERGY_D[Ne]
        S0 = MEAN_EXCITATION_ENERGY_S0[Ne]
    else
        D = MEAN_EXCITATION_ENERGY_D[nmax]
        S0 = Ne - sqrt(Ne * 10.0 / 14.99)
    end
    A = (1 - D)^2
    B = 2 * (1 - D) * (Ne * D - S0)
    C = (Ne * D - S0)^2
    return 14.99 * (A * Z^2 + B * Z + C)
end

"""
    ion_size(Z, Z0; table=:hesslow)

Effective ion size `ā_j` in units of the reduced Compton wavelength (`2/α · a_j`):
DFT values from Hesslow et al. (2018) Table 1 where available, else their eq. 2.28.
"""
function ion_size(Z::Integer, Z0::Integer; table::Symbol=:hesslow)
    Zs = table === :dream ? ION_SIZE_Z_DREAM : ION_SIZE_Z
    for n in eachindex(ION_SIZE_Z0)
        if Zs[n] == Z && ION_SIZE_Z0[n] == Z0
            return 2 / Constants.alpha * ION_SIZE_A[n]
        end
    end
    return 2 / Constants.alpha * cbrt(9π * (Z - Z0)^2) / (4Z)
end

# One ion charge state with its atomic parameters.
struct ChargeState
    Z::Int
    Z0::Int
    species::Int
    I::Float64   # mean excitation energy / mc²
    a::Float64   # effective ion size
end

"""
    CollisionModel(plasma, settings)

Collision frequencies for a background plasma. Radius-dependent quantities
(Coulomb logarithms, Bessel functions) are evaluated once; the frequencies are then
evaluated at any momentum with [`nu_s`](@ref), [`nu_D`](@ref), [`nu_par`](@ref).
"""
struct CollisionModel{T,P<:PlasmaState}
    plasma::P
    settings::CollisionSettings
    states::Vector{ChargeState}
    n_free::Vector{T}
    n_bound::Vector{T}
    K0::Vector{T}   # e^{1/Θ} K_n(1/Θ)
    K1::Vector{T}
    K2::Vector{T}
end

function CollisionModel(plasma::PlasmaState{T}, settings::CollisionSettings) where {T}
    states = ChargeState[]
    for (k, s) in enumerate(plasma.ions), Z0 in 0:s.Z
        I = mean_excitation_energy(s.Z, Z0) / Constants.mc2inEV
        a = Z0 == s.Z ? NaN : ion_size(s.Z, Z0; table=settings.ion_size_table)
        push!(states, ChargeState(s.Z, Z0, k, I, a))
    end
    nr_ = n_radii(plasma)
    n_free = [free_electron_density(plasma, ir) for ir in 1:nr_]
    n_bound = [bound_electron_density(plasma, ir) for ir in 1:nr_]
    Θ = plasma.T_cold ./ Constants.mc2inEV
    K0 = besselkx.(0, 1 ./ Θ)
    K1 = besselkx.(1, 1 ./ Θ)
    K2 = besselkx.(2, 1 ./ Θ)
    return CollisionModel{eltype(K0),typeof(plasma)}(plasma, settings, states, n_free,
                                                      n_bound, K0, K1, K2)
end

@inline _density(cm::CollisionModel, st::ChargeState, ir) =
    cm.plasma.ions[st.species].n[st.Z0+1, ir]

_lnlambda(cm::CollisionModel, kind, p, ir) =
    lnlambda(kind, cm.settings.lnlambda, p, cm.plasma.T_cold[ir], cm.n_free[ir])

function _ntarget(cm::CollisionModel, ir)
    n = cm.plasma.n_cold[ir]
    cm.settings.screening === :non && (n += cm.n_bound[ir])
    return max(n, zero(n))
end

# Thermal integrals Ψ_n(p) = ∫₀ᵖ γ(s)ⁿ⁻¹ exp(-(γ(s) - 1)/Θ) ds, following DREAM
# evaluatePsi0/1/2: asymptotic expansions in the superthermal and low-energy limits,
# adaptive quadrature otherwise.
function _psi(n::Int, cm::CollisionModel, ir, p)
    Θ = cm.plasma.T_cold[ir] / Constants.mc2inEV
    γ = sqrt(1 + p^2)
    γm1 = p^2 / (γ + 1)
    thr = 10Θ
    if p^2 > thr * (thr + 2)  # superthermal expansion
        e = exp(-γm1 / Θ)
        if n == 0
            return cm.K0[ir] + Θ * (-e / p) + Θ^2 * (γ / p^3 * e)
        elseif n == 1
            return cm.K1[ir] + Θ * (-γ / p * e) + Θ^2 * (e / p^3)
        else
            return cm.K0[ir] + Θ * (cm.K1[ir] - γ^2 / p * e) + Θ^2 * (-γ * (γ^2 - 2) / p^3 * e)
        end
    elseif p < 0.15 || Θ < 0.005  # low-energy expansion
        x = sqrt(γm1 / Θ)
        O1 = 4n - 1.0
        O2 = 2n^2 - 3n + 0.375
        ψ = (1 + Θ * O1 / 8 + Θ^2 * 0.1875 * O2) * sqrt(π) / 2 * erf(x)
        ψ -= Θ * x * exp(-x^2) * (O1 + Θ * O2 * (3 + 2x^2) / 2) / 8
        return ψ * sqrt(2Θ)
    else
        integrand = s -> begin
            gs = sqrt(1 + s^2)
            exp(-(s^2 / (1 + gs)) / Θ) * gs^(n - 1)
        end
        return first(quadgk(integrand, zero(p), p; rtol=1e-8))
    end
end

# In :full mode at p ≪ p_th the numerator is a difference of O(p) terms while M ∝ p³,
# so relative round-off grows like (p_th/p)²; kept identical to DREAM for comparisons.
function _electron_term_s(cm::CollisionModel, ir, p)
    cm.settings.mode === :full || return one(p)
    p == 0 && return zero(p)
    Θ = cm.plasma.T_cold[ir] / Constants.mc2inEV
    γ = sqrt(1 + p^2)
    γm1 = p^2 / (γ + 1)
    M = γ^2 * _psi(1, cm, ir, p) - Θ * _psi(0, cm, ir, p) + (Θ * γ - 1) * p * exp(-γm1 / Θ)
    return M / (γ^2 * cm.K2[ir])
end

function _electron_term_D(cm::CollisionModel, ir, p)
    cm.settings.mode === :full || return one(p)
    p == 0 && return zero(p)
    Θ = cm.plasma.T_cold[ir] / Constants.mc2inEV
    p2 = p^2
    γ = sqrt(1 + p2)
    γm1 = p2 / (γ + 1)
    M = (p2 * γ^2 + Θ^2) * _psi(0, cm, ir, p)
    M += Θ * (2p2^2 - 1) * _psi(1, cm, ir, p)
    M += γ * Θ * (1 + Θ * (2p2 - 1)) * p * exp(-γm1 / Θ)
    return M / (p2 * γ^2 * cm.K2[ir])
end

# Bound-electron stopping power, matched Bethe formula (Hesslow et al. eq. 2.31 with
# DREAM's modification that keeps it positive).
function _stopping(cm::CollisionModel, st::ChargeState, p)
    NB = st.Z - st.Z0
    NB == 0 && return zero(p)
    p2 = p^2
    γ = sqrt(1 + p2)
    β2 = p2 / (1 + p2)
    h = (p2 / sqrt(1 + γ)) / st.I
    if cm.settings.mode === :full
        return NB * log(1 + (h * exp(-β2))^LNLAMBDA_K) / LNLAMBDA_K
    else
        return NB * log(exp(1) + h * exp(-β2))
    end
end

# Bound-electron contribution to pitch-angle scattering (Hesslow et al. eq. 2.25).
function _screened(st::ChargeState, p)
    NB = st.Z - st.Z0
    NB == 0 && return zero(p)
    x = (p * st.a)^1.5
    return 2 / 3 * NB * ((st.Z + st.Z0) * log(1 + x) - NB * x / (1 + x))
end

"""
    nu_s(cm, ir, p)

Slowing-down frequency [s⁻¹] at radius index `ir` and momentum `p`; the collisional
friction is `Aᵖ = -p ν_s`.
"""
function nu_s(cm::CollisionModel, ir, p)
    p == 0 && return zero(p * cm.K0[ir])
    mode = cm.settings.mode
    pre = mode === :ultra_relativistic ? COLLISION_PREFACTOR / p :
          COLLISION_PREFACTOR * (1 + p^2) / p^3
    ν = _lnlambda(cm, :ee, p, ir) * _electron_term_s(cm, ir, p) * _ntarget(cm, ir)
    if cm.settings.screening === :partially
        for st in cm.states
            ν += _stopping(cm, st, p) * _density(cm, st, ir)
        end
    end
    return pre * ν
end

"""
    nu_D(cm, ir, p)

Pitch-angle scattering frequency [s⁻¹]; the collisional pitch diffusion is
`Dᵠᵠ = ν_D (1 - ξ²)/2`.
"""
function nu_D(cm::CollisionModel, ir, p)
    p == 0 && return zero(p * cm.K0[ir])
    mode = cm.settings.mode
    pre = mode === :ultra_relativistic ? COLLISION_PREFACTOR / p^2 :
          COLLISION_PREFACTOR * sqrt(1 + p^2) / p^3
    ν = _lnlambda(cm, :ee, p, ir) * _electron_term_D(cm, ir, p) * _ntarget(cm, ir)
    lnLei = _lnlambda(cm, :ei, p, ir)
    nonscreened = cm.settings.screening === :non
    for st in cm.states
        Zfact = nonscreened ? st.Z^2 : st.Z0^2
        ν += lnLei * Zfact * _density(cm, st, ir)
    end
    if cm.settings.screening === :partially
        for st in cm.states
            ν += _screened(st, p) * _density(cm, st, ir)
        end
    end
    return pre * ν
end

"""
    nu_par(cm, ir, p)

Parallel (energy) diffusion frequency [s⁻¹], `Dᵖᵖ = ν_∥`; nonzero only in `:full` mode.
"""
function nu_par(cm::CollisionModel, ir, p)
    z = zero(p * cm.K0[ir])
    (cm.settings.mode === :full && p != 0) || return z
    Θ = cm.plasma.T_cold[ir] / Constants.mc2inEV
    if cm.settings.screened_diffusion === :maxwellian
        νs = nu_s(cm, ir, p)
    else  # free electrons only
        νs = COLLISION_PREFACTOR * (1 + p^2) / p^3 * _lnlambda(cm, :ee, p, ir) *
             _electron_term_s(cm, ir, p) * cm.plasma.n_cold[ir]
    end
    return Θ * sqrt(1 + p^2) * νs
end

"""
    p3nu_s_at_zero(cm, ir)

`lim_{p→0} p³ ν_s`, which sets the particle flux through `p = 0`.
"""
function p3nu_s_at_zero(cm::CollisionModel, ir)
    z = zero(cm.K0[ir])
    ν = _lnlambda(cm, :ee, z, ir) * _electron_term_s(cm, ir, z) * _ntarget(cm, ir)
    if cm.settings.screening === :partially
        for st in cm.states
            ν += _stopping(cm, st, z) * _density(cm, st, ir)
        end
    end
    return COLLISION_PREFACTOR * ν
end
