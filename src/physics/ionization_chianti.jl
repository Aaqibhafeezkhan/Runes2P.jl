# The cross sections of the ionization of the ions by electron impact from the parameters of
# the CHIANTI database (K. P. Dere, Astron. Astrophys. 466, 771 (2007)), continued
# relativistically by the Bethe theory (M. Inokuti, Rev. Mod. Phys. 43, 297 (1971)).
# The parameters are in the generated table of physics/ionization_chianti_fits.jl.

"""
    ChiantiDirectShell(scale, threshold, knots, values)

The fit of the direct ionization of one shell of an ion in CHIANTI's `.diparams` files (K. P.
Dere, Astron. Astrophys. 466, 771 (2007), doi:10.1051/0004-6361:20066728, Sect. 2.1): the
Burgess-Tully parameter `f` (`scale`), the ionization energy of the shell [eV]
(`threshold`), and the spline nodes of the scaled cross section `Σ` (`values`, in 1e-14 cm²
eV², the unit of the files) at the scaled energies `U` (`knots`, from 0 to 1). Dere's eqs.
(2) and (3):

```math
U = 1 - \\frac{\\ln f}{\\ln(u - 1 + f)}, \\qquad Σ = \\frac{u σ I^2}{\\ln u + 1},
```

with `u = E/I`. `Σ(0) = 0`, and `Σ(U = 1)` is the Bethe limit of the shell. That limit is not
fitted: Dere computes it from the photoionization cross sections of Verner and Yakovlev (1995)
by Younger's formula, his eq. (1), and notes that measured cross sections "have often been
found to approach a high-energy limit that is a factor of 1.4 to 2.0 higher than the Bethe
limit" (his Sect. 2.1). The spline is the
cubic interpolating spline with not-a-knot ends, SciPy's `splrep(s = 0)`, with which the fits
were made (the comment of ChiantiPy's `Ion.diCross`, the CHIANTI package for Python; fiasco
interpolates the same nodes with a monotone cubic, `PchipInterpolator`). `curvatures` holds its
second derivatives at the nodes.
"""
struct ChiantiDirectShell
    scale::Float64
    threshold::Float64
    knots::Vector{Float64}
    values::Vector{Float64}
    curvatures::Vector{Float64}
end
ChiantiDirectShell(scale, threshold, knots, values) =
    ChiantiDirectShell(scale, threshold, knots, values, _not_a_knot_curvatures(knots, values))

"""
    ChiantiAutoionization(kind, oscillator_strength, energy, scale, values)

One transition of the excitation-autoionization of an ion in CHIANTI's `.easplom` files (K. P.
Dere, Astron. Astrophys. 466, 771 (2007), Sect. 2.2): the Burgess-Tully type of the
transition (`kind`), its weighted oscillator strength, its excitation energy [Ry, the unit of
the files] (`energy`), the Burgess-Tully parameter `C` (`scale`) and the five spline nodes of
the scaled collision strength at equally spaced scaled energies from 0 to 1 (`values`). With
`X = E/ΔE`, the collision strength `Ω` is, by type, as ChiantiPy's `util.splomDescale`
evaluates it:

- type 1: `Ω = Ω_s(1 − ln C/ln(X − 1 + C)) ln(X − 1 + e)`;
- type 2: `Ω = Ω_s((X − 1)/(X − 1 + C))`;
- type 3: `Ω = Ω_s((X − 1)/(X − 1 + C))/X²`;
- type 4: `Ω = Ω_s(1 − ln C/ln(X − 1 + C)) ln(X − 1 + C)`.

The cross section is `s π a₀² Ω/(E/R_y)` with the scale `s` of the ion
([`ChiantiIonization`](@ref)); the statistical weight of the ground level is in `Ω`
(ChiantiPy's `Ion.eaCross`). Types 1 and 4, the allowed transitions, grow as `ln X` and add
`s Ω_s(1)/4` to the Bethe coefficient `M²` of the ion.
"""
struct ChiantiAutoionization
    kind::Int
    oscillator_strength::Float64
    energy::Float64
    scale::Float64
    values::Vector{Float64}
    curvatures::Vector{Float64}
end
ChiantiAutoionization(kind, oscillator_strength, energy, scale, values) =
    ChiantiAutoionization(kind, oscillator_strength, energy, scale, values,
                          _not_a_knot_curvatures(range(0.0, 1.0; length=length(values)), values))

"""
    ChiantiIonization(ion, direct, autoionization_scale, autoionization)

The CHIANTI parameters of the ionization of one ion (`ion`, CHIANTI's name, `"ar_4"` for Ar IV,
the charge state 3): the shells of its direct ionization ([`ChiantiDirectShell`](@ref)), and
its excitation-autoionization ([`ChiantiAutoionization`](@ref)) with the scale of the last line
of its `.diparams` file (1 for every ion of the table). The constructor derives the
coefficients of the relativistic Bethe form that continues the fit above
[`CHIANTI_MATCHING_ENERGY`](@ref) ([`chianti_cross_section`](@ref)): `bethe_m2`, the
coefficient `M²` of the high-energy limit of the fit, the sum of `Σ(1)/(4π a₀² R_y I)` over the
shells and `s Ω_s(1)/4` over the allowed transitions of the autoionization; `bethe_c`, the
constant `C` that makes the Bethe form equal the continued fit at the matching energy.
"""
struct ChiantiIonization
    ion::String
    direct::Vector{ChiantiDirectShell}
    autoionization_scale::Float64
    autoionization::Vector{ChiantiAutoionization}
    bethe_m2::Float64
    bethe_c::Float64
end

function ChiantiIonization(ion, direct, autoionization_scale, autoionization)
    m2 = sum(_bethe_m2(s) for s in direct) + sum((_bethe_m2(t, autoionization_scale) for t in autoionization); init=0.0)
    fit = ChiantiIonization(ion, direct, autoionization_scale, autoionization, m2, NaN)
    p = _momentum_of_energy(CHIANTI_MATCHING_ENERGY)
    beta2 = p^2 / (1 + p^2)
    c = _fano_ordinate(fit, p) - m2 * (log(p^2) - beta2)
    return ChiantiIonization(ion, direct, autoionization_scale, autoionization, m2, c)
end

"""
    CHIANTI_MATCHING_ENERGY

The kinetic energy [eV], 100 keV, above which the cross section of
[`chianti_cross_section`](@ref) follows the relativistic Bethe form, a decision of the
author (2026-10-13): "above about 100 keV". There
`β² = 0.30`, and the fits rest on measurements below a few keV and on distorted-wave
calculations to about 16 times the threshold (R. Bao and B. Li, At. Data Nucl. Data Tables
159, 101663 (2024), Figs. 19 and 20 for Ar¹³⁺ and Ar¹⁴⁺). For the neutral atoms the slope of
the Fano plot of the fit there is 0.94 (H), 1.03 (Ar) and 1.04 (Ne) of its `M²`, so the switch
to the Bethe form changes little. Since the
spline is pinned to its Bethe limit at `U = 1`, this says nothing about whether `M²` is right;
measurements check it for H and Ar only ([`chianti_cross_section`](@ref)).
"""
const CHIANTI_MATCHING_ENERGY = 1e5

"""
    CHIANTI_MATCHING_WIDTH

The factor, 2, by which the weight of [`chianti_cross_section`](@ref) passes from the continued
fit to the Bethe form, from `CHIANTI_MATCHING_ENERGY/2` to `2 CHIANTI_MATCHING_ENERGY`, with
the cubic `w = x²(3 − 2x)` in `x = ln(2E/E_m)/ln 4`. The weight and its slope are continuous at
both ends, so the cross section and its slope are continuous. The window is short against
the decade over which the two forms agree within the spread of the fits (0.94 to 1.04 in the
Fano slope of the neutral atoms).
"""
const CHIANTI_MATCHING_WIDTH = 2.0

# the momentum [mₑc] of an electron of kinetic energy E [eV]
_momentum_of_energy(E) = sqrt((1 + E / Constants.ELECTRON_REST_ENERGY)^2 - 1)

# β² of an electron of kinetic energy E [eV]
function _beta_squared_of_energy(E)
    gamma = 1 + E / Constants.ELECTRON_REST_ENERGY
    return 1 - 1 / gamma^2
end

# the second derivatives at the nodes of the cubic interpolating spline with not-a-knot ends
# (the third derivative continuous at the second and the second-to-last node)
function _not_a_knot_curvatures(x, y)
    n = length(x)
    h = diff(collect(Float64, x))
    A = zeros(n, n)
    b = zeros(n)
    for i in 2:n-1
        A[i, i-1], A[i, i], A[i, i+1] = h[i-1], 2(h[i-1] + h[i]), h[i]
        b[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    end
    A[1, 1], A[1, 2], A[1, 3] = -1 / h[1], 1 / h[1] + 1 / h[2], -1 / h[2]
    A[n, n-2], A[n, n-1], A[n, n] = -1 / h[n-2], 1 / h[n-2] + 1 / h[n-1], -1 / h[n-1]
    return A \ b
end

# the spline with the nodes (x, y) and the curvatures M at t, in the number type of t
function _spline_value(x, y, M, t)
    i = clamp(searchsortedlast(x, t), 1, length(x) - 1)
    h = x[i+1] - x[i]
    a = (x[i+1] - t) / h
    b = (t - x[i]) / h
    return a * y[i] + b * y[i+1] + ((a^3 - a) * M[i] + (b^3 - b) * M[i+1]) * h^2 / 6
end

# the coefficient M² of the high-energy limit of a shell, Σ(1) = 4π a₀² R_y I M², and of an
# allowed transition of the autoionization, s Ω_s(1)/4
_bethe_m2(shell::ChiantiDirectShell) =
    shell.values[end] * 1e-18 / (4π * Constants.BOHR_RADIUS^2 * Constants.RYDBERG_ENERGY * shell.threshold)
_bethe_m2(t::ChiantiAutoionization, scale) = t.kind in (1, 4) ? scale * t.values[end] / 4 : 0.0

# The Fano ordinate T σ/(4π a₀² R_y), with T = mₑc²β²/2, of a shell and of a transition of the
# autoionization: the fit at the velocity variable u = β²/β_I² (the ratio of the squared speeds
# of the electron and of an electron of the threshold energy), which is E/I at low energies,
# and its relativistic rise M² [−ln(1 − β²) − β²], weighted by 1 − 1/u so that it starts at
# zero at the threshold. Zero below the threshold.
function _fano_ordinate(shell::ChiantiDirectShell, beta2, rise)
    u = beta2 / _beta_squared_of_energy(shell.threshold)
    u > 1 || return zero(beta2)
    U = 1 - log(shell.scale) / log(u - 1 + shell.scale)
    sigma = _spline_value(shell.knots, shell.values, shell.curvatures, U) * 1e-18   # Σ [m² eV²]
    # T σ = T Σ (ln u + 1)/(u I²), with T/(u I) = mₑc² β_I²/(2I)
    T_over_uI = Constants.ELECTRON_REST_ENERGY * _beta_squared_of_energy(shell.threshold) / (2 * shell.threshold)
    return T_over_uI * sigma * (log(u) + 1) / (shell.threshold * 4π * Constants.BOHR_RADIUS^2 * Constants.RYDBERG_ENERGY) +
           (1 - 1 / u) * _bethe_m2(shell) * rise
end

function _fano_ordinate(t::ChiantiAutoionization, scale, beta2, rise)
    delta = t.energy * Constants.RYDBERG_ENERGY
    X = beta2 / _beta_squared_of_energy(delta)
    X > 1 || return zero(beta2)
    knots = range(0.0, 1.0; length=length(t.values))
    if t.kind == 1 || t.kind == 4
        s = 1 - log(t.scale) / log(X - 1 + t.scale)
        omega = _spline_value(knots, t.values, t.curvatures, s) * log(X - 1 + (t.kind == 1 ? ℯ : t.scale))
    else
        s = (X - 1) / (X - 1 + t.scale)
        omega = _spline_value(knots, t.values, t.curvatures, s) / (t.kind == 3 ? X^2 : one(X))
    end
    # σ = s π a₀² Ω R_y/(X ΔE), so T σ/(4π a₀² R_y) = s Ω T/(4 X ΔE) with T/(X ΔE) = mₑc² β_Δ²/(2ΔE)
    T_over_XD = Constants.ELECTRON_REST_ENERGY * _beta_squared_of_energy(delta) / (2 * delta)
    return scale * omega * T_over_XD / 4 + (1 - 1 / X) * _bethe_m2(t, scale) * rise
end

# the Fano ordinate of the continued fit of an ion at the momentum p
function _fano_ordinate(fit::ChiantiIonization, p)
    beta2 = p^2 / (1 + p^2)
    rise = log(1 + p^2) - beta2
    g = zero(beta2)
    for shell in fit.direct
        g += _fano_ordinate(shell, beta2, rise)
    end
    for t in fit.autoionization
        g += _fano_ordinate(t, fit.autoionization_scale, beta2, rise)
    end
    return g
end

"""
    chianti_cross_section(fit::ChiantiIonization, p)

The cross section [m²] of the ionization of the ion of `fit` by an electron of momentum `p`
(in units of `mₑc`): CHIANTI's direct ionization and excitation-autoionization (K. P. Dere,
Astron. Astrophys. 466, 771 (2007), doi:10.1051/0004-6361:20066728), continued relativistically
by the Bethe theory (decision of the author, 2026-10-13).

The relativistic Bethe cross section of M. Inokuti, Rev. Mod. Phys. 43, 297 (1971),
doi:10.1103/RevModPhys.43.297, eq. (4.26), for an electron (z = 1),

```math
σ_B = \\frac{4π a_0^2 R_y}{T} \\left\\{M^2 \\left[\\ln\\frac{β^2}{1 - β^2} - β^2\\right] + C\\right\\},
\\qquad T = \\frac{m_e v^2}{2},
```

splits, by Fano's analysis (Inokuti, p. 328), into the nonrelativistic Bethe form (his eq.
(4.18)) at the energy `T = mₑv²/2` and the relativistic rise `(4π a₀² R_y/T) M² [−ln(1 − β²) −
β²]`. The fits of CHIANTI are nonrelativistic: their high-energy limit is the nonrelativistic
Bethe limit (Dere, Sect. 2.1). They are continued in the same way:

- each shell and each transition of the autoionization is evaluated at the speed of the
  electron, `u = β²/β_I²` with `β_I` the speed of an electron of the threshold energy, which is
  `E/I` at low energies and keeps the threshold at `E = I`;
- the relativistic rise is added with the coefficient `M²` of the high-energy limit of each
  shell and allowed transition ([`ChiantiIonization`](@ref)), weighted by `1 − 1/u` so that
  it starts at zero at the threshold of each (the weight is 0.94 or more at 100 keV for
  every state of the table).

Above [`CHIANTI_MATCHING_ENERGY`](@ref) the cross section is `σ_B` with the `M²` of the fit and
the `C` at which `σ_B` equals the continued fit at that energy, joined to it by the weight of
[`CHIANTI_MATCHING_WIDTH`](@ref). Below about 10 keV the continuation changes the fit by less
than the spread of the data: by 0.3 % at 1 keV and about 3 % at 10 keV for the neutral atoms.
For the states of high charge the fits come from distorted-wave calculations, which may hold
the relativistic kinematics already: up to 7 % at 10 times the threshold of Ar¹⁶⁺ and Ar¹⁷⁺ may
be counted twice.

Implied by the fits: `M² = 0.2842` for H (0.2823 from the fit of M. B. Shah, D. S. Elliott, H.
B. Gilbody, J. Phys. B 20, 3501 (1987)), 4.180 for Ar (4.22, F. F. Rieke, W. Prepejchal, Phys.
Rev. A 6, 1507 (1972), Table II) and 1.705 for Ne (2.02, the same table, 16 % low, within
Dere's caveat of the Bethe limit, [`ChiantiDirectShell`](@ref)); `C = 3.92` for H (4.007) and
40.8 for Ar (37.93). No ion is checked above about 16 times its threshold. For Ar⁺ and Ar²⁺
the fit does not support its limit: the last node of Ar⁺ falls from 19.19 to 7.69, `M²` is
0.58 and 0.27, and `C/M²` at 100 keV is 21.4 and 18.8, against 8.3 to 13.7 for every other
state and about 9 in Rieke and Prepejchal; at 1 MeV Ar⁺ is 0.59 of DREAM's form. Whether these
states take another `M²` is an open question.

The threshold is CHIANTI's ionization energy, which differs from that of
[`ionization_energy`](@ref) (NIST) by up to 1.4 eV for the argon ions; the energy loss of the
friction of `bound_friction = :ionization_split` takes NIST's. Generic in the number type of
`p`.
"""
function chianti_cross_section(fit::ChiantiIonization, p)
    p2 = p^2
    beta2 = p2 / (1 + p2)
    T = Constants.ELECTRON_REST_ENERGY * beta2 / 2
    E = p2 / (sqrt(1 + p2) + 1) * Constants.ELECTRON_REST_ENERGY
    low = CHIANTI_MATCHING_ENERGY / CHIANTI_MATCHING_WIDTH
    high = CHIANTI_MATCHING_ENERGY * CHIANTI_MATCHING_WIDTH
    prefactor = 4π * Constants.BOHR_RADIUS^2 * Constants.RYDBERG_ENERGY / T
    bethe() = prefactor * (fit.bethe_m2 * (log(p2) - beta2) + fit.bethe_c)
    E >= high && return bethe()
    g = _fano_ordinate(fit, p)
    # below every threshold, where T may be 0
    iszero(g) && return zero(g)
    continued = prefactor * g
    E <= low && return continued
    x = log(E / low) / log(high / low)
    w = x^2 * (3 - 2x)
    return (1 - w) * continued + w * bethe()
end
