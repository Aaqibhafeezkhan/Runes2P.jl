# Coulomb logarithms following DREAM `CoulombLogarithm.cpp` (Hesslow et al.,
# J. Plasma Phys. 84 (2018), eqs. 2.7-2.10).

const LNLAMBDA_K = 5  # interpolation exponent between thermal and relativistic limits

"""
    lnlambda_thermal(T, n_free)

Thermal Coulomb logarithm `14.9 - ½ ln(n/10²⁰) + ln(T/keV)`, zero for `n_free = 0`.
"""
lnlambda_thermal(T, n) = n == 0 ? zero(T) : 14.9 - 0.5 * log(n / 1e20) + log(T / 1e3)

"""
    lnlambda_relativistic(T, n_free)

Relativistic Coulomb logarithm `14.6 + ½ ln(T/(n/10²⁰))` (DREAM `lnLambda_c`).
"""
lnlambda_relativistic(T, n) = n == 0 ? zero(T) : 14.6 + 0.5 * log(T / (n / 1e20))

"""
    lnlambda(kind, model, p, T, n_free)

Coulomb logarithm for electron-electron (`kind = :ee`) or electron-ion (`:ei`)
collisions at momentum `p` (units of mc). `model` is `:energy_dependent`, `:thermal` or
`:constant` (the relativistic value).
"""
function lnlambda(kind::Symbol, model::Symbol, p, T, n)
    model === :constant && return lnlambda_relativistic(T, n)
    lnT = lnlambda_thermal(T, n)
    model === :thermal && return lnT
    model === :energy_dependent || throw(ArgumentError("unknown lnΛ model $model"))
    pTe = sqrt(2 * T / Constants.mc2inEV)
    x = if kind === :ee
        γm1 = p^2 / (sqrt(1 + p^2) + 1)
        sqrt(2 * γm1) / pTe
    elseif kind === :ei
        2p / pTe
    else
        throw(ArgumentError("kind must be :ee or :ei"))
    end
    return lnT + log(1 + x^LNLAMBDA_K) / LNLAMBDA_K
end
