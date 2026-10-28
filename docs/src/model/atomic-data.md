# Atomic data and ionization

```@meta
CurrentModule = Runes2P
```

The charge states of the ions evolve by rate equations whose coefficients come from three sources.
The cold electrons ionize and recombine at the rate coefficients of OPEN-ADAS.
The electrons of a distribution `f` ionize through cross sections of electron-impact ionization.
The binding energy of each charge state is the ionization energy of NIST.
A species takes the data of its element, the part of its name before an underscore ([`atomic_element`](@ref)).
The ionization energies, the atomic weights, the parameters of the cross sections and the table of ADF11 files are TOML files of `data/`, which the package reads when it is precompiled.

## ADAS rate coefficients

The fluid rates and the radiation of the ions are the ADF11 coefficients of ADAS (H. P. Summers, The ADAS User Manual, version 2.6 (2004)), with the data of OPEN-ADAS.
[`read_adf11`](@ref) reads one unresolved ADF11 file into an [`ADF11Table`](@ref) in SI units.
The classes are ionization (SCD), recombination (ACD), charge exchange (CCD), line radiation (PLT) and recombination radiation with bremsstrahlung (PRB).
[`ADAS_FILES`](@ref) names the file of each element and class, as DREAM's `tools/elements.json` does, and [`adf11_file_name`](@ref) builds its name.
[`read_adas`](@ref) reads the five tables of an element from the directory of [`adas_cache_directory`](@ref).
The terms of OPEN-ADAS do not allow the data to be redistributed with a modelling code.
No file is part of the repository, and `scripts/fetch_adas.py` fills the cache on a machine with internet access.
No equation term takes the charge-exchange coefficients.

[`adas_coefficient`](@ref) evaluates a coefficient at the free-electron density ``n`` and the temperature ``T`` with DREAM's interpolation (`src/ADASRateInterpolator.cpp`).
Inside the table it is the bicubic interpolation of ``\log_{10}`` of the coefficient in ``\log_{10} n`` and ``\log_{10} T``, with the node derivatives of natural cubic splines, as GSL's bicubic interpolation.
Outside the table, `extrapolation = :dream` continues the bilinear interpolation of the edge cell linearly.
The value jumps at the table edge between nodes, where the bicubic and the bilinear interpolations differ.
`extrapolation = :clamp_density` holds the density at the nearest edge and extrapolates in the temperature alone, since at low density the coefficients tend to their coronal limit, which does not depend on the density.
The default `:dream`, a choice of this package, matches DREAM, a code used by researchers in the field.
The derivatives in ``n`` and ``T`` are exact through ForwardDiff, where DREAM takes one-sided differences.

Recombination and its radiation belong to the recombining ion ``Z_0 \ge 1``, ionization and line radiation to the states ``Z_0 \le Z - 1``.
Charge exchange takes the index of ionization, as in DREAM.
A coefficient is zero for a state that its class lacks.

Checks:

- `test/test_adas.jl`: the reader on synthetic tables, the interpolation against values of GSL, the extrapolations and the derivatives;
- `test/test_adas_cache.jl`: the reader against the tables that DREAM compiles from the same files, where the cache exists.

## Ionization energies

[`IONIZATION_ENERGIES`](@ref) holds the ionization energy of every charge state from the NIST Atomic Spectra Database, as DREAM's `src/Atomics/nistdata_ionization.cpp`.
[`ionization_energy`](@ref) reads it.
Deuterium and tritium take the energies of hydrogen.
These energies are the binding energy of the radiation terms and of the BEB spectrum of the secondaries.
They are also the energy that the friction on bound electrons takes per ionization with `bound_friction = :ionization_split`.

## Rate equations of the charge states

The ion densities are a radial unknown with one group of components per species ([`ion_densities_unknown`](@ref), [`IonComposition`](@ref)).
[`IonRateTerm`](@ref) is the ionization and recombination by the cold electrons, as DREAM's `IonRateEquation`:

```math
\frac{dn_j}{dt} = n_e \left(I_{j-1} n_{j-1} - I_j n_j + R_{j+1} n_{j+1} - R_j n_j\right),
```

with the SCD coefficients ``I_j`` and the ACD coefficients ``R_j`` at the cold density and temperature.
The sum over the charge states of a species is zero, so each species is conserved.
[`ion_rate_matrix`](@ref) gives the matrix of these rates and [`coronal_equilibrium`](@ref) its steady state.
[`population_equations`](@ref) closes the cold-electron density as the free electrons of the ions less the electrons of the distributions.
[`deposit_neutrals!`](@ref) adds neutral atoms at the start of a run, and the rate equations ionize them from the first step on.

Checks:

- `test/test_ion_rates.jl`: the conservation, the Jacobian and the coronal equilibrium computed three ways;
- `test/test_ion_rates_dream.jl`: the charge states of DREAM's fluid mode with deposited deuterium and argon at prescribed temperatures;
- `test/test_independent_references.jl`: the coronal balance of argon against published balances with other atomic data:
  - P. Mazzotta et al., Astronomy and Astrophysics Supplement Series 133, 403 (1998), [doi:10.1051/aas:1998330](https://doi.org/10.1051/aas:1998330);
  - P. Bryans et al., The Astrophysical Journal 691, 1540 (2009), [doi:10.1088/0004-637X/691/2/1540](https://doi.org/10.1088/0004-637X/691/2/1540).

## Kinetic ionization

[`KineticIonizationTerm`](@ref) is the ionization by the electrons of a kinetic unknown, as DREAM's `IonKineticIonizationTerm`.
The ionization rate per ion of charge state ``j`` is the moment of ``f`` with the weight ``v σ_j(p)``:

```math
\frac{dn_j}{dt} = K_{j-1} n_{j-1} - K_j n_j, \qquad
K_j = \frac{1}{V'} \sum_{\text{cells}} e(p)\, v σ_j(p)\, \mathcal{V}' Δp\, Δξ\, f ,
```

with the spatial and phase-space Jacobians ``V'`` and ``\mathcal{V}'`` ([Grids](grids.md)).

The envelope ``e(p)`` of a [`MomentumRegion`](@ref) restricts the moment to the electrons above a threshold.
Without a region the moment covers the whole grid.
The cross sections are tabulated at the cell centres of the momentum grid of each radius.
[`kinetic_ionization_rates`](@ref) gives the rates ``K_j`` of a state.

### Cross sections

[`ionization_cross_section`](@ref) has three forms, chosen by `ionization_cross_section_form` of the term and of [`CollisionSettings`](@ref).

`:dream` is DREAM's form with DREAM's parameters [`KINETIC_IONIZATION_FITS`](@ref), fitted to the SCD coefficients at zero density.
At low energies it is the formula of A. Burgess and M. C. Chidichimo, Monthly Notices of the Royal Astronomical Society 203, 1269 (1983), [doi:10.1093/mnras/203.4.1269](https://doi.org/10.1093/mnras/203.4.1269).
At high energies it is DREAM's form of the relativistic Bethe formula of N. A. Garland et al., Physics of Plasmas 27, 040702 (2020), [doi:10.1063/5.0003638](https://doi.org/10.1063/5.0003638).
That form lacks the factor ``1/β^2`` of a Bethe cross section.
`:bethe` adds that factor, and its parameters [`KINETIC_IONIZATION_BETHE_FITS`](@ref) come from DREAM's fitting tool, run unchanged on this form.
The docstring of [`ionization_cross_section`](@ref) gives the formulas of both.

`:chianti` takes the parameters of the CHIANTI database ([`CHIANTI_IONIZATION_FITS`](@ref); K. P. Dere, Astronomy and Astrophysics 466, 771 (2007), [doi:10.1051/0004-6361:20066728](https://doi.org/10.1051/0004-6361:20066728)).
Each ion has the direct ionization of its shells ([`ChiantiDirectShell`](@ref)) and its excitation-autoionization ([`ChiantiAutoionization`](@ref)).
Both are fits in the scaled variables of A. Burgess and J. A. Tully, Astronomy and Astrophysics 254, 436 (1992).
The table holds hydrogen, neon and argon, and deuterium and tritium take the parameters of hydrogen ([`chianti_element`](@ref)).
The other elements, tungsten among them, take DREAM's form.
The one- and two-electron ions take the parameters of CHIANTI's files, as the Python package fiasco does.
Dere's text takes for them the fits of C. J. Fontes, D. H. Sampson and H. L. Zhang, Physical Review A 59, 1329 (1999), [doi:10.1103/PhysRevA.59.1329](https://doi.org/10.1103/PhysRevA.59.1329).
The threshold is CHIANTI's ionization energy, which differs from NIST's, and the energy of the friction takes NIST's.
`:chianti` is the default, a choice of this package ([Choices of this package](../choices.md)).

### Relativistic continuation of CHIANTI's fits

CHIANTI's fits are nonrelativistic, and their high-energy limit is the nonrelativistic Bethe limit.
[`chianti_cross_section`](@ref) continues them by the relativistic Bethe cross section (M. Inokuti, Reviews of Modern Physics 43, 297 (1971), [doi:10.1103/RevModPhys.43.297](https://doi.org/10.1103/RevModPhys.43.297), eq. (4.26)):

```math
σ_B = \frac{4π a_0^2 \mathrm{Ry}}{E_v} \left\{M^2 \left[\ln\frac{β^2}{1 - β^2} - β^2\right] + C_B\right\},
\qquad E_v = \frac{m_e v^2}{2} .
```

Each shell and each transition of the autoionization is evaluated at ``u = β^2/β_I^2``, with ``β_I`` the speed of an electron of kinetic energy ``I``.
Each gains the relativistic rise ``(4π a_0^2 \mathrm{Ry}/E_v)\, M^2 [-\ln(1 - β^2) - β^2]`` of ``σ_B`` with the weight ``1 - 1/u``, which is zero at the threshold.
[`ChiantiIonization`](@ref) derives ``M^2`` from the fit, and ``C_B`` from the condition that ``σ_B`` equals the continued fit at ``E_m``, [`CHIANTI_MATCHING_ENERGY`](@ref), 100 keV.
Above ``2 E_m`` the cross section is ``σ_B``; a cubic weight in ``\ln E`` joins the two between ``E_m/2`` and ``2 E_m`` ([`CHIANTI_MATCHING_WIDTH`](@ref), 2), so the cross section and its slope are continuous.
The matching energy, the width and the weight ``1 - 1/u`` are choices of this package.
The ``M^2`` of the direct shells is Dere's computed Bethe limit, not fitted to measurements; the allowed transitions of the autoionization add the limit of their collision strengths.

Checks:

- `test/test_chianti_ionization.jl`:
  - the table against CHIANTI's files, the threshold, the continuity at the matching, the derivative through ForwardDiff and the fallback to DREAM's form;
  - the argon ions against the relativistic distorted-wave calculations of R. Bao and B. Li, Atomic Data and Nuclear Data Tables 159, 101663 (2024), [doi:10.1016/j.adt.2024.101663](https://doi.org/10.1016/j.adt.2024.101663);
- `test/test_independent_references.jl`:
  - hydrogen against the measurements of M. B. Shah et al., Journal of Physics B 20, 3501 (1987), [doi:10.1088/0022-3700/20/14/022](https://doi.org/10.1088/0022-3700/20/14/022);
  - argon against the measurements of H. C. Straub et al., Physical Review A 52, 1115 (1995), [doi:10.1103/PhysRevA.52.1115](https://doi.org/10.1103/PhysRevA.52.1115);
  - against relativistic Bethe asymptotes from 0.1 to 10 MeV: Shah's fit for hydrogen, F. F. Rieke and W. Prepejchal, Physical Review A 6, 1507 (1972), [doi:10.1103/PhysRevA.6.1507](https://doi.org/10.1103/PhysRevA.6.1507), for argon and neon;
- `test/test_kinetic_ionization.jl`: the Maxwellian averages of the cross sections against the SCD coefficients at low density, and the relativistic term of `:bethe` against the large-energy limit of the Burgess–Chidichimo formula;
- `test/test_kinetic_ionization_dream.jl`: the rates of DREAM's fully kinetic mode, with `:dream`, a threshold of 3 thermal momenta and the fluid rate of DREAM's pairing.

Dere's fit of hydrogen rests on Shah's data, and Straub's data are among those of his fit of argon.
These rows check the implementation of the fits.
The MeV asymptotes and the calculations of Bao and Li check the model.

### Split at the hot threshold

A kinetic ionization above a threshold next to the full ADAS rate of the cold electrons counts the Maxwellian tail above the threshold twice.
That is DREAM's pairing (`src/Settings/Equations/ions.cpp`), and [`IonRateTerm`](@ref) without `kinetic_ionization` reproduces it.
With `kinetic_ionization`, the term splits the ionization at the threshold of the [`KineticIonizationTerm`](@ref).
[`Runes2P.threshold_split`](@ref) takes the Maxwellian at the cold temperature on the grid of the term and gives the fraction ``f_>`` of its density above the threshold and its rate coefficient ``A_>`` from the cross sections.
The cold electrons below the threshold ionize at the coefficient of [`fluid_ionization_coefficient`](@ref):

```math
I_{\mathrm{fluid}} = \frac{I - A_>}{1 - f_>} .
```

With the closure ``n_{\mathrm{cold}} = n_M (1 - f_>)`` and the kinetic rate ``n_M A_>`` of a Maxwellian ``f`` of density ``n_M``, the two add up to the ADAS rate ``n_M I`` for every threshold.

The coefficient has no floor, since a floor would break that sum.
It is negative where the cross sections give more ionization above the threshold than ADAS gives for the whole Maxwellian.
It can turn negative when the threshold lies below the ionization energy of a populated state: then all of its ionization lies above the threshold.
The split takes a threshold ``p_0 = k\, p_{\mathrm{th}}``, with the thermal momentum ``p_{\mathrm{th}} = \sqrt{2T/m_e c^2}`` (``T`` in energy units) of the temperature unknown ([`ThermalMomentum`](@ref)), so that the fluid coefficient follows the temperature.
It refuses factors ``k`` below [`MINIMUM_SPLIT_THRESHOLD_FACTORS`](@ref): 7 with `:chianti`, 5 with `:dream` and `:bethe`, a choice of this package.
[`RadiatedPowerTerm`](@ref) with `kinetic_ionization` takes the binding energy of the cold electrons at the same split coefficient.

Checks:

- `test/test_kinetic_ionization.jl`: a Maxwellian ``f`` ionizes at the ADAS rate for every threshold, and DREAM's pairing counts the tail twice;
- `test/test_independent_references.jl`: the split of a Maxwellian on the grid against the continuous integrals.

### Energy of the kinetic ionization

[`KineticIonizationTerm`](@ref) has no energy of its own.
The friction of the Fokker-Planck term on the bound electrons takes the energy of the inelastic collisions from ``f`` ([Collision models](collisions.md)).
With `bound_friction = :ionization_split` of [`CollisionSettings`](@ref), that friction holds the energy ``I_j + ⟨E_s⟩_j`` of each ionization the term counts.
``⟨E_s⟩_j`` is the mean kinetic energy of the secondary electrons up to ``(E - I_j)/2``.
The system refuses this friction unless its threshold factor and its cross-section form are those of the kinetic ionization.

The secondaries join the cold electrons through the closure of the cold density.
[`SecondaryHeatingTerm`](@ref) gives their energy to the cold electrons; without it they are born at zero energy.
Two spectra of the secondaries are available:

- `:beb` ([`beb_mean_secondary_energy`](@ref)), the binary-encounter-Bethe spectrum of Y.-K. Kim and M. E. Rudd, Physical Review A 50, 3954 (1994), [doi:10.1103/PhysRevA.50.3954](https://doi.org/10.1103/PhysRevA.50.3954).
  It takes the ionization energy of each state as its binding energy and has no fitted parameter;
- `:opal` ([`mean_secondary_energy`](@ref)), the spectrum of C. B. Opal, W. K. Peterson and E. C. Beaty, The Journal of Chemical Physics 55, 4100 (1971), [doi:10.1063/1.1676707](https://doi.org/10.1063/1.1676707).
  It takes the widths of [`SECONDARY_SPECTRUM_WIDTHS`](@ref), measured on neutral gases.

`:beb` is the default, a choice of this package.
[`ionization_energy_flows`](@ref) splits the energy that the kinetic ionization takes from ``f`` into the ionization potentials and the energy of the secondaries.
DREAM has no heating by secondaries: [`KineticIonizationTerm`](@ref) alone is DREAM's model.

## Radiation and binding energy

[`RadiatedPowerTerm`](@ref) is the power that the cold electrons lose to the ions, in the energy equation of the cold electrons:

```math
P = n_e \sum_{i, Z_0} n_{i, Z_0} \left(L_{i, Z_0} + B_{i, Z_0}\right),
```

with the coefficients ``L`` of radiation and ``B`` of binding energy [W m³], from the ADAS coefficients and the ionization energies of NIST.
The ion densities come from a prescribed [`PlasmaState`](@ref) or from the ion unknown.

- `radiation = :recombination`, the default and a choice of this package: ``L`` is PLT + PRB, the line radiation with the recombination radiation and bremsstrahlung.
  ``B`` is the energy ``I(Z_0)\,\mathrm{SCD}`` that ionization takes, less the energy ``I(Z_0 - 1)\,\mathrm{ACD}`` that recombination returns.
- `radiation = :dream`, DREAM's default: ``L`` is PLT with the bremsstrahlung of R. J. Gould, The Astrophysical Journal 238, 1026 (1980), [doi:10.1086/158068](https://doi.org/10.1086/158068), and ``B`` the ionization loss alone.
  `bremsstrahlung_corrections` selects the relativistic corrections of the bremsstrahlung, `:gould` or `:dream`.
  `:gould` takes the values after Gould's erratum and by his eqs. (18) and (22) ([`GOULD_ION_CORRECTION`](@ref), [`GOULD_ELECTRON_CORRECTION`](@ref)).
  The erratum is The Astrophysical Journal 243, 677 (1981), [doi:10.1086/158634](https://doi.org/10.1086/158634).
  `:dream` takes DREAM's values ([`DREAM_GOULD_ION_CORRECTION`](@ref), [`DREAM_GOULD_ELECTRON_CORRECTION`](@ref)).

DREAM's mode with recombination takes ``I(Z_0)`` instead of ``I(Z_0 - 1)`` in the recombination gain from its second radial cell on, which `:recombination` does not reproduce.
[`radiated_power`](@ref) gives the radiation alone: it leaves the plasma, while the binding energy stays in it.
The losses of the electrons of ``f`` by bremsstrahlung and synchrotron radiation are terms of the Fokker-Planck operator ([Collision models](collisions.md)).

Checks:

- `test/test_radiation.jl`:
  - the radiated power against sums of the tabulated coefficients of synthetic tables in both modes;
  - Gould's bremsstrahlung against the NRL Plasma Formulary (A. S. Richardson, NRL/PU/6770--19-652, Naval Research Laboratory, 2019);
- `test/test_ion_energy_dream.jl`: the charge states and the temperature of DREAM's fluid mode with dynamic ions, where the radiated power and the binding energy read the charge states of the run;
- `test/test_independent_references.jl`:
  - the energy of a closed neon plasma with ionizing charge states;
  - Gould's relativistic correction against the cross section 4BN of H. W. Koch and J. W. Motz, Reviews of Modern Physics 31, 920 (1959), [doi:10.1103/RevModPhys.31.920](https://doi.org/10.1103/RevModPhys.31.920).
