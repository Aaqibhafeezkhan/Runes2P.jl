# Collision models

```@meta
CurrentModule = Runes2P
```

The distribution ``f(p, ξ, r)`` of the electrons obeys a Fokker-Planck equation in the momentum ``p`` (units of ``m_e c``) and the pitch ``ξ = p_\parallel/p``.
The equation holds the electric field, the synchrotron loss and, for small-angle collisions, the linearized relativistic test-particle operator of DREAM.
Large-angle collisions enter as a separate knock-on source, and a field-particle term restores the momentum the test-particle operator loses.
The geometry is a cylinder in a homogeneous magnetic field, so the bounce averages are trivial: each coefficient takes its local value.

## The Fokker-Planck operator

[`fokker_planck_operator`](@ref) fills the face coefficients of an [`AdvectionDiffusion`](@ref) operator in the flux form

```math
\frac{\partial f}{\partial t} = -\frac{1}{\mathcal{V}'} \frac{\partial}{\partial z^k}
\left[\mathcal{V}' \left(A^k f - D^{kl} \frac{\partial f}{\partial z^l}\right)\right],
\qquad z = (p, ξ),
```

with the phase-space Jacobian ``\mathcal{V}'`` ([`phase_space_jacobian`](@ref)).
Each physical process adds to the coefficients:

| Process | Coefficients | Function |
|---|---|---|
| electric field | ``A^p = \frac{eE}{m_e c} ξ``, ``A^ξ = \frac{eE}{m_e c} \frac{1 - ξ^2}{p}`` | [`add_electric_field!`](@ref) |
| slowing down | ``A^p = -p\, ν_s(p)`` | [`add_slowing_down!`](@ref) |
| pitch-angle scattering | ``D^{ξξ} = ν_D(p)\, \frac{1 - ξ^2}{2}`` | [`add_pitch_scattering!`](@ref) |
| energy diffusion | ``D^{pp} = ν_\parallel(p)`` | [`add_energy_diffusion!`](@ref) |
| synchrotron loss | ``A^p = -C B^2 p γ (1 - ξ^2)``, ``A^ξ = C B^2 \frac{ξ (1 - ξ^2)}{γ}`` | [`add_synchrotron!`](@ref) |

The collision frequencies ``ν_s``, ``ν_D`` and ``ν_\parallel`` come from a [`CollisionModel`](@ref) of the plasma and the [`CollisionSettings`](@ref).
The flux through ``p = 0`` follows from ``\lim_{p \to 0} p^3 ν_s`` ([`p3_slowing_down_frequency_at_zero`](@ref)).
The interpolation of ``f`` to the faces is the scheme of the operator (see [Positivity and flux limiters](positivity.md)).

An electric field that is an unknown of the system does not enter the coefficients.
The operator then holds the coefficients of a unit field ([`set_unit_field_coefficients!`](@ref)), and the [`FokkerPlanckTerm`](@ref) multiplies them by the field of each radius.
A plasma that changes in time rebuilds the coefficients through [`set_fokker_planck_coefficients!`](@ref), with the scheme and the boundary conditions kept ([`prescribed_fokker_planck_term`](@ref)).
With a plasma read from the unknowns ([`PlasmaFromUnknowns`](@ref)) the term is nonlinear, and its keyword `lagged` selects the part of the plasma taken at the start of the step.
The default `lagged = :ions` is a choice of this package (see [Choices of this package](../choices.md)).

## Collision frequencies

The frequencies are those of DREAM's `CollisionFrequency` classes, with the partial screening of L. Hesslow et al., J. Plasma Phys. 84, 905840605 (2018), doi:10.1017/S0022377818001113.
The `mode` of [`CollisionSettings`](@ref) sets the target electrons:

- `:full`: relativistic Maxwellian targets at the cold-electron temperature, with the thermal factors of Pike and Rose and the energy diffusion ``ν_\parallel``;
- `:superthermal`: cold targets, no energy diffusion;
- `:ultra_relativistic`: cold targets, no energy diffusion, and the limit ``v \to c`` of the prefactors.

The thermal factors are those of O. J. Pike and S. J. Rose, Phys. Rev. E 89, 053107 (2014), doi:10.1103/PhysRevE.89.053107.

With ``Θ = T/(m_e c^2)``, the cold-electron density ``n_c`` and the density ``n_j`` of charge state ``j`` (charge ``Z_{0j}``, nuclear charge ``Z_j``), the frequencies of the `:full` and `:superthermal` modes are

```math
\begin{aligned}
ν_s &= 4π r_0^2 c\, \frac{γ^2}{p^3} \Big[\ln Λ_{ee}\, ψ_s(p)\, n_c + \sum_j n_j\, w_j(p)\, h_j(p)\Big] + ν_{s,\mathrm{rad}}, \\
ν_D &= 4π r_0^2 c\, \frac{γ}{p^3} \Big[\ln Λ_{ee}\, ψ_D(p)\, n_c + \ln Λ_{ei} \sum_j n_j Z_{0j}^2 + \sum_j n_j\, w_j(p)\, g_j(p)\Big], \\
ν_\parallel &= Θ γ\, ν_s^{\mathrm{coll}},
\end{aligned}
```

[`slowing_down_frequency`](@ref), [`deflection_frequency`](@ref) and [`parallel_diffusion_frequency`](@ref) evaluate them.
With `screening = :none`, ``n_c`` includes the bound electrons and ``Z_{0j}^2`` becomes ``Z_j^2``.
``ψ_s`` and ``ψ_D`` are the thermal factors of the `:full` mode and one otherwise.
``h_j`` and ``g_j`` are the bound-electron terms of partial screening, ``w_j`` their screening weight, and ``ν_{s,\mathrm{rad}}`` the bremsstrahlung stopping.
``ν_s^{\mathrm{coll}}`` is ``ν_s`` without ``ν_{s,\mathrm{rad}}``, unless `bremsstrahlung_diffusion = :dream`.
With `screened_diffusion = :zero` the energy diffusion takes the free-electron part alone.

### Coulomb logarithms

[`coulomb_logarithm`](@ref) gives the logarithm of electron-electron and electron-ion collisions in three models:

- `:thermal`: ``\ln Λ_T = 14.9 - \tfrac12 \ln\bigl(n/(10^{20}\,\mathrm{m^{-3}})\bigr) + \ln(T/\mathrm{keV})`` ([`thermal_coulomb_logarithm`](@ref));
- `:relativistic`: the momentum-independent ``\ln Λ_c = 14.6 + \tfrac12 \ln\bigl((T/\mathrm{eV})/(n/(10^{20}\,\mathrm{m^{-3}}))\bigr)`` ([`relativistic_coulomb_logarithm`](@ref));
- `:energy_dependent`: ``\ln Λ = \ln Λ_T + \tfrac{1}{k} \ln(1 + x^k)`` with ``k = 5``, ``x = \sqrt{2(γ - 1)}/p_T`` for electrons and ``x = 2p/p_T`` for ions, ``p_T = \sqrt{2Θ}``.

These are eqs. 2.7 to 2.10 of Hesslow et al. (2018), as in DREAM's `CoulombLogarithm.cpp`.
``n`` is the free-electron density.

### Screening

The `screening` of [`CollisionSettings`](@ref) sets how the bound electrons of partially ionized ions act on a fast electron:

- `:complete`: the bound electrons are invisible, and an ion scatters with its net charge ``Z_{0j}``;
- `:none`: the bound electrons are targets like free ones, and an ion scatters with its nuclear charge ``Z_j``;
- `:partial`: the model of Hesslow et al. (2018), between the two.

Partial screening adds the Bethe stopping on the bound electrons, ``h_j``, with the mean excitation energy ``I_j`` of the charge state ([`mean_excitation_energy`](@ref)).
It adds the deflection ``g_j`` by the screened nucleus, with the effective ion size of the charge state ([`ion_size`](@ref)).
The sizes are the DFT values of Table 1 of Hesslow et al. where tabulated and their eq. 2.28 otherwise.
This is `ion_size_table = :hesslow` (the default); `ion_size_table = :dream` takes DREAM's lookup of that table.

With `screening_window = :excitation_energy` (the default) the bound-electron parts are weighted by ``w_j``, which is a cubic smooth step in the kinetic energy ``E_k = (γ - 1) m_e c^2``:

```math
w_j = s^2 (3 - 2s), \qquad s = \operatorname{clamp}\!\left(\frac{E_k - a I_j}{(b - a) I_j},\, 0,\, 1\right),
```

with ``a`` and ``b`` the settings `screening_window_start` and `screening_window_end`.
Below the window the frequencies are those of complete screening.
The default factors are a choice of this package (see [Choices of this package](../choices.md)).
`screening_window = :dream` takes ``w_j = 1`` at all momenta, as DREAM does.

With `bound_friction = :ionization_split` the stopping on the bound electrons splits into two parts.
The ionization part is the energy of the ionizations that ``f`` causes.
The excitation part is the remainder, floored at zero.
The ionization part uses the cross sections of the kinetic ionization ([`ionization_cross_section`](@ref)), so that ``f`` pays the energy of the ionizations that the [`KineticIonizationTerm`](@ref) counts (see [Atomic data and ionization](atomic-data.md)).

### Bremsstrahlung

With `bremsstrahlung = :stopping_power` the radiative energy loss on the nuclei of all charge states enters ``ν_s`` as a drag:

```math
ν_{s,\mathrm{rad}} = \sum_j n_j Z_j^2\, α r_0^2 c\, \frac{γ}{p}\, φ_\mathrm{rad}(p),
```

with the unscreened cross section 4BN of H. W. Koch and J. W. Motz, Reviews of Modern Physics 31, 920 (1959), doi:10.1103/RevModPhys.31.920 ([`bremsstrahlung_cross_section`](@ref)).
There is no electron-electron bremsstrahlung, as in DREAM.
The stopping is on by default, where DREAM leaves it out, a choice of this package (see [Choices of this package](../choices.md)).
`bremsstrahlung_constant` selects the constant term of the formula: ``4/3`` (Koch and Motz) with `:koch_motz` (the default), or ``1``, DREAM's integer value, with `:dream`.

`bremsstrahlung_diffusion` decides whether the energy diffusion of the `:full` mode takes the radiative stopping as well:

- `:none` (the default): the stopping is a pure drag, as in the model of M. Hoppe et al., Comput. Phys. Commun. 268, 108098 (2021), doi:10.1016/j.cpc.2021.108098, eqs. 8, 9, 14 and 15;
- `:dream`: DREAM's code, whose ``ν_\parallel`` rescales the whole ``ν_s``.

With `:none` a Maxwellian at the bulk temperature loses its bremsstrahlung power, which tends to Gould's formula at low temperature ([`GOULD_BREMSSTRAHLUNG_PREFACTOR`](@ref)).
The formula is that of R. J. Gould, The Astrophysical Journal 238, 1026 (1980), doi:10.1086/158068.
With `:dream` a Maxwellian at the bulk temperature radiates nothing, because the energy diffusion rescaled with the whole ``ν_s`` balances the radiative drag on it.
The default `:none` is a choice of this package (see [Choices of this package](../choices.md)).

## Synchrotron loss

[`add_synchrotron!`](@ref) adds the radiation reaction of the gyration in the magnetic field ``B`` of each radius, with

```math
C = \frac{e^4}{6π ε_0 m_e^3 c^3}
```

([`SYNCHROTRON_RATE_PER_FIELD_SQUARED`](@ref)).
It is on by default in [`fokker_planck_operator`](@ref), [`set_fokker_planck_coefficients!`](@ref) and the [`FokkerPlanckTerm`](@ref), where DREAM leaves it out.
The default is chosen for accuracy.
`synchrotron = false` leaves the loss out.

## Knock-on collisions

The small-angle operator holds the collisions of ``f`` with the thermal electrons.
Close collisions of a fast primary with an electron at rest produce secondaries at large momentum transfer, the knock-ons.
A [`KnockOnOperator`](@ref) adds their source, the Boltzmann gain term of the cross section of C. Møller, Annalen der Physik 406, 531 (1932), doi:10.1002/andp.19324060506, above a cutoff momentum ``p_\mathrm{cut}``.
It is ported from DREAM's `MollerBoltzmannOperator`.
The target density is the total electron density ``n_\mathrm{tot}``, free and bound.
The operator is the source of the knock-ons alone.
It has no loss term for the primaries.

The source in a knock-on cell is ``n_\mathrm{tot}`` times a sum over the primary cells of three factors: an energy kernel, a pitch kernel and the number of primaries in the cell (the formula is in [`KnockOnOperator`](@ref)).

- The energy kernel ``S_{ik}`` ([`MollerEnergyKernel`](@ref)) is the Møller production rate from primary momentum cell ``k``, averaged over knock-on cell ``i``.
  It integrates the cross section ([`moller_cross_section`](@ref), [`moller_flux`](@ref)) from the cutoff to half the kinetic energy of the primary ([`max_knockon_momentum`](@ref)).
- The pitch kernel ``δ_{jl}`` ([`MollerPitchKernel`](@ref)) is the distribution in knock-on pitch of the knock-ons from primary pitch cell ``l``, normalized so that ``\sum_j Δξ_j δ_{jl} = 1``, from the gyrophase-averaged fraction of each cell ([`pitch_cell_fraction`](@ref)).
  It is taken at the scattering cosine ``ξ^*`` that the kinematics fixes ([`xi_star`](@ref)).

The pitch kernel is tabulated in the scattering cosine and interpolated per momentum pair ([`KnockOnKernels`](@ref)).
By default the primary pitch is averaged over its cell, and the scattering cosine is integrated over the momentum interval that receives the knock-ons.
With `dream_compatible = true` both are taken at the cell centres, as DREAM does.
The primary and the knock-on grids may differ, with the same radial grid, and the momentum grid may differ between radii ([`apply_at_radius!`](@ref)).

A [`KnockOnTerm`](@ref) places the source in the equation of a kinetic unknown.
Its `treatment` takes three values.
`:explicit_primary` (the default) takes the primary at the start of the step, as DREAM's Boltzmann operator does.
`:explicit` takes the whole term, target density included, at the previous time level.
`:implicit` takes the whole term at the new time level, matrix-free, since it couples all cells of a radius.
The operator's own matrix-free Jacobian is [`knockon_linear_operator`](@ref).

### The cutoff

The cutoff separates the knock-ons counted by the source from the collisions of the small-angle operator.
It may differ between radii and change in time: [`set_cutoff!`](@ref) takes any positive cutoff per radius, and the kernels ([`MollerEnergyKernel`](@ref), [`MollerPitchKernel`](@ref)) take the part of a cell above a cutoff inside it.
A [`KnockOnCutoffUpdate`](@ref) sets it after every accepted step from the plasma of each radius, on a face of the momentum grid, so that no cell is cut.
Its default placement ([`cutoff_target`](@ref)) is the logarithmic midpoint in energy between the temperature ``T`` and the kinetic energy ``E_\mathrm{crit}`` at the critical momentum:

```math
E_\mathrm{cut} = \sqrt{T\, E_\mathrm{crit}}.
```

Where no critical momentum exists (the electric field below the critical field), or its kinetic energy is not above the temperature, the cutoff energy is `no_ceiling_factor` times the temperature.
A face is kept while the target stays between its neighbouring faces ([`choose_cutoff_face`](@ref)), so the kernels are rebuilt rarely.
The placement and `no_ceiling_factor` are choices of this package (see [Choices of this package](../choices.md)).
The threshold is [`total_critical_momentum`](@ref) by default.
Its critical field takes the Coulomb logarithm named by the keyword `coulomb_logarithm` of [`KnockOnCutoffUpdate`](@ref), as DREAM's critical fields do.
The default `:thermal` gives the thermal logarithm, and the other models the relativistic one.
A run whose collisions take another model passes it, and `:relativistic` is allowed in any run.
DREAM keeps one cutoff for all radii and times.

With a hot-tail and a runaway grid the pairings of the two grids share one cutoff, through the keyword `linked` of [`KnockOnCutoffUpdate`](@ref), so that every knock-on above it is counted once.

### Fluid avalanche

A fluid runaway density above a grid creates knock-ons as well.
[`RosenbluthPutvinskiSourceTerm`](@ref) is DREAM's kinetic Rosenbluth-Putvinski source, the knock-ons of a primary at the speed of light ([`rosenbluth_putvinski_rate`](@ref)).
It is the source of M. N. Rosenbluth and S. V. Putvinski, Nucl. Fusion 37, 1355 (1997), doi:10.1088/0029-5515/37/10/I03.
[`ExternalAvalancheTerm`](@ref) is the avalanche growth rate of the fluid runaways that DREAM uses above the grid.
The rate is that of L. Hesslow et al., Nucl. Fusion 59, 084004 (2019), doi:10.1088/1741-4326/ab26c2, with the [`fluid_avalanche_growth_rate`](@ref) of the [`effective_critical_field`](@ref) and the [`avalanche_critical_momentum`](@ref).
The fluid growth rate takes the collision settings of DREAM's `RunawayFluid` ([`runaway_fluid_settings`](@ref)), not those of the kinetic equation.

## Field-particle term

The test-particle operator treats the thermal electrons as a fixed Maxwellian.
It removes the parallel momentum that the electrons give to each other, which lowers the conductivity.
A [`FieldParticleTerm`](@ref) returns that momentum to the bulk with a model field-particle operator ([`FieldParticleOperator`](@ref)):

```math
C_\mathrm{fp}(f) = \sum_{a,b} g_a A_{ab} (\ell_b \cdot f), \qquad a, b = 0, \dots, N - 1,
```

per radius, with ``N`` = `modes`.
``\ell_b \cdot f`` is the rate at which the discrete electron-electron operator ([`electron_electron_operator`](@ref)) changes the parallel momentum weighted by the Laguerre polynomial ``L_b^{(3/2)}(x^2)``, ``x = v/v_T``.
The polynomials are those of the expansion of S. I. Braginskii, Rev. Plasma Phys. 1, 205 (1965).
``g_a \propto v_\parallel L_a^{(3/2)}(x^2) f_M`` is the shape of mode ``a`` on the Maxwellian ``f_M``.
``A`` fixes the matrix elements to those of the Landau operator between the first `modes` Laguerre modes.
The improved Sugama operator does the same (H. Sugama et al., Phys. Plasmas 26, 102108 (2019), doi:10.1063/1.5115440).
Row 0 of ``A`` makes the parallel momentum conserved to round-off.
With `modes = 1` the term is the momentum-restoring model of S. P. Hirshman and D. J. Sigmar, Phys. Fluids 19, 1532 (1976), doi:10.1063/1.861356, and the higher modes add the heat-flux corrections.

The term needs the `:full` mode, where the bulk is part of ``f``.
Its matrix elements are those of the nonrelativistic Landau operator, so it holds for a nonrelativistic bulk.
It couples all cells of a radius and is applied matrix-free, with a Jacobian of rank `modes` per radius ([`low_rank_jacobian`](@ref)) that the linear solvers take by a Woodbury update.
[`set_field_particle_coefficients!`](@ref) rebuilds it for a plasma that changes in time.

## Ion energy exchange

The thermal exchange between the electrons and the ion species is the [`IonEnergyExchangeTerm`](@ref) (see [Energy and current balance](energy-current.md)).
Its electron-ion logarithm is the thermal one, as in DREAM.
Among the ions, `ion_coulomb_logarithm = :nrl` (the default) takes the mixed ion-ion logarithm of the NRL Plasma Formulary (A. S. Richardson, NRL/PU/6770--19-652, 2019).
It takes the temperature of each species and a screening sum over all ion species and the electrons.
It is cut at zero so that the exchange keeps its sign.
`:dream` takes DREAM's form, built from the electron temperature and the same for every pair.

## Comparisons

- The collision frequencies against DREAM: `test/test_collisions_dream.jl`.
- One step of the small-angle kinetic operator against DREAM: `test/test_kinetic_operator_dream.jl`.
- The knock-on operator in the DREAM-compatible configuration against DREAM's `MollerBoltzmannOperator`: `test/test_knockon_operator.jl`.
- The default knock-on operator against a quadrature reference, and its physics limits: `test/test_knockon_reference.jl`, `test/test_knockon_physics.jl`.
- An avalanche run against DREAM's Boltzmann knock-on mode: `test/test_avalanche_dream.jl`.
- The Rosenbluth-Putvinski source and the fluid growth rate against DREAM: `test/test_avalanche_rp_dream.jl`.
- The field-particle term, `test/test_field_particle.jl`: momentum conservation, the matrix elements of the test-particle operator against Braginskii's, and the field-particle ratios against the like-species sums.
  The same file checks the Lorentz limit and the Spitzer-Härm conductivity.

The conductivity is that of L. Spitzer and R. Härm, Phys. Rev. 89, 977 (1953), doi:10.1103/PhysRev.89.977.
