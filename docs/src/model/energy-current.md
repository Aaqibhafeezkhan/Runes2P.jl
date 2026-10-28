# Energy and current balance

The electrons exchange energy with the electric field, the ions, the radiation and each other.
Their currents set the field through the induction equations.
Each term of these balances is a building block of an [`Equation`](@ref), and [`EquationSystem`](@ref) checks that the pieces match.

## The energy of the cold electrons

The energy of the cold electrons is an unknown of its own, as in DREAM:

```math
W_\mathrm{cold} = \tfrac{3}{2}\, e\, n_\mathrm{cold} T_\mathrm{cold},
```

the algebraic row of [`thermal_energy_equation`](@ref) with the factor [`THERMAL_ENERGY_FACTOR`](@ref).
The row of the temperature holds the time derivative of this energy, `TimeDerivative(:cold_thermal_energy)` ([`TimeDerivative`](@ref)).
Its terms are powers in W/m³, and the temperature follows from the algebraic row.
The energy takes its initial value from the temperature and the density ([`set_thermal_energy!`](@ref)).

The channels of this row:

| Channel | Term |
|---|---|
| ohmic heating ``E j_\mathrm{ohm}`` | [`ohmic_heating_term`](@ref), a [`ProductTerm`](@ref) |
| line and recombination radiation, bremsstrahlung, binding energy | [`RadiatedPowerTerm`](@ref) |
| radial heat transport | [`HeatTransportTerm`](@ref) |
| exchange with the ions | [`IonEnergyExchangeTerm`](@ref) with `side = :electrons` |
| secondaries of the kinetic ionization | [`SecondaryHeatingTerm`](@ref) |
| friction of fluid runaways | [`RunawayFluidEnergyTransferTerm`](@ref) |
| collisions of the runaway grid with the cold electrons | [`free_electron_energy_transfer_term`](@ref) |

[`IonEnergyExchangeTerm`](@ref) with `side = :ions` is the same exchange in the row of the ion energies, so the two sides are antisymmetric.
[`RunawayFluidEnergyTransferTerm`](@ref) is DREAM's heating by runaways without a runaway grid, the power ``e c\, E_c\, n_\mathrm{re}``: runaways at the speed of light against the friction of the critical field ``E_c``, with the Coulomb logarithm at ``p = 20\, m_e c``.
With a runaway grid, [`free_electron_energy_transfer_term`](@ref) takes its place: the energy moment of the electron-electron collisions of that grid.

The channels other than [`SecondaryHeatingTerm`](@ref) are those of DREAM's equation of ``T_\mathrm{cold}`` (`src/Settings/Equations/T_cold.cpp`, DREAM v26.5), and [`RadiatedPowerTerm`](@ref) takes DREAM's radiation with `radiation = :dream` and `bremsstrahlung_corrections = :dream`.
With a hot-tail grid, DREAM adds the collisional transfer from the hot-tail distribution, except in its fully kinetic mode (`collfreq_mode` FULL), where it leaves the transfer out.
`test/test_cold_energy_dream.jl` compares the row with ohmic heating, radiation and heat diffusion with DREAM in its fluid mode.

## The total energy in the fully kinetic mode

In the fully kinetic mode the hot-tail distribution `f` covers the bulk.
The electrons above the threshold of the hot region (a [`MomentumRegion`](@ref) with a [`ThermalMomentum`](@ref) limit) hold the energy

```math
W_\mathrm{hot} = \frac{1}{V'} \sum_{ij} w_i\, m_e c^2 (\gamma_i - 1)\, f_{ij}\, \mathcal{V}'_{ij} \Delta p_i\, \Delta\xi_j,
```

the [`EnergyMomentTerm`](@ref) over that region, with the envelope ``w`` of the region above the threshold ``p_0``.
[`total_energy_equation`](@ref) adds the unknown

```math
W_\mathrm{tot} = W_\mathrm{cold} + W_\mathrm{hot}
```

with its algebraic row.
The row of the temperature then holds ``\mathrm{d}W_\mathrm{tot}/\mathrm{d}t``, `TimeDerivative(:total_electron_energy)`, and the temperature follows from ``W_\mathrm{cold} = W_\mathrm{tot} - W_\mathrm{hot}``.

The row lists only the channels by which energy enters or leaves the electrons of the hot-tail grid:

- the work of the field, ``E j_\mathrm{ohm} + E j_\mathrm{hot}`` (two [`ProductTerm`](@ref)s), where the ohmic current already leaves out the hot current, so each electron is counted once;
- the radiation, the heat transport and the exchange with the ions, as in the row of the cold energy;
- the heating by the secondaries of a kinetic ionization ([`SecondaryHeatingTerm`](@ref)), which moves their energy from the sinks to the cold electrons;
- the sinks of the collisions of `f`: [`KineticEnergyLossTerm`](@ref) with `part = :sinks` and the hot region;
- the outflow at ``p_\mathrm{max}``: [`MomentumBoundaryFluxTerm`](@ref) with `weight = :energy` and `coefficient = -1`;
- the electrons that cross to a runaway grid: [`InterfaceEnergyFluxTerm`](@ref).

The collisional transfer from the hot electrons to the cold ones does not appear in the row.
Neither do the electrons that cross the threshold, nor the threshold that moves with the temperature.
The time derivative acts on the energy of the hot region itself, so these exchanges inside ``W_\mathrm{tot}`` are exact for any time stepper.

### What the system enforces

Every change of ``W_\mathrm{hot}`` that the row does not list is charged to the cold electrons.
[`EquationSystem`](@ref) therefore checks the row against the equation of `f`; its docstring gives the conditions.

- Collisions with sinks (synchrotron loss, bremsstrahlung stopping power, partial or no screening) remove energy from `f`, so the row needs the sinks term.
- The electrons that leave a Fokker-Planck term with collisions at ``p_\mathrm{max}``, or cross its interface to a runaway grid, take their energy with them.
  The row needs the energy flux of that boundary or interface.
- Radial transport of `f` and [`RadialEdgeLossTerm`](@ref) are refused, since no term carries the energy moment of radial transport.
- A [`KnockOnTerm`](@ref) on the grid is refused, since no term carries the energy of the secondaries.
- A [`RosenbluthPutvinskiSourceTerm`](@ref) is refused, since its primaries have no energy unknown.

Each refused channel would take its energy from the cold electrons of the radius where the electrons are moved or created.

### Why the hot-tail grid only

The runaway grid stays outside ``W_\mathrm{tot}``.
A relative error ``\varepsilon`` of ``W_\mathrm{tot}`` becomes a relative error of the cold energy of

```math
\varepsilon\, \frac{W_\mathrm{tot}}{W_\mathrm{cold}},
```

so the temperature is lost to cancellation when ``W_\mathrm{tot}`` is much larger than ``W_\mathrm{cold}``.
Keeping the runaways out keeps this ratio small.
The runaway grid enters the row through two channels only: [`free_electron_energy_transfer_term`](@ref) for its collisions with the cold electrons, and [`InterfaceEnergyFluxTerm`](@ref) for the electrons that it receives.

### The field work on the hot region

The row charges ``E j_\mathrm{hot}`` for the work of the field on the hot region.
The energy moment of the discrete field term over that region differs from it.
The difference includes the energy ``E_k(p_0)\, \Gamma``, with ``E_k(p_0) = m_e c^2 (\gamma(p_0) - 1)`` and ``\Gamma`` the flux of electrons that the field moves across the threshold ``p_0``, which the total energy takes from the cold electrons.
`test/test_energy_budget.jl` compares the two in a driven fully kinetic plasma, next to the transfer from a slowing-down tail and the checks of the row.

## Energy transfer between the distributions

[`KineticEnergyLossTerm`](@ref) is the energy moment of a part of the collisions of a [`FokkerPlanckTerm`](@ref):

```math
L = -\frac{1}{V'} \sum_{ij} w_i\, m_e c^2 (\gamma_i - 1)\, \left[C_a f - C_b f\right]_{ij} \mathcal{V}'_{ij} \Delta p_i \Delta\xi_j,
```

with the envelope ``w`` of the region.

- `part = :free_electrons`: ``C_a`` the electron-electron collisions ([`electron_electron_operator`](@ref)), ``C_b = 0``.
  This is the energy that `f` gives to the free cold electrons, [`free_electron_energy_transfer_term`](@ref).
- `part = :sinks`: ``C_a`` all collisions without the field, ``C_b`` their electron-electron part.
  This is the energy that leaves the electrons by the friction on bound electrons, bremsstrahlung and synchrotron radiation.

The sinks take no flux through ``p_\mathrm{max}``: the energy of the electrons that leave there belongs to [`MomentumBoundaryFluxTerm`](@ref) alone, so it is not counted twice.
With the sinks charged to ``W_\mathrm{tot}``, the cold electrons receive exactly the free-electron part of what the hot electrons lose.

[`MomentumBoundaryFluxTerm`](@ref) and [`InterfaceEnergyFluxTerm`](@ref) weight the flux at the top of the grid with the kinetic energy of the last momentum cell.
That is the energy that the discrete energy moment of the grid loses with the electrons.
The electrons that cross to the runaway grid take their energy with them and do not heat the cold ones.
`test/test_energy_transfer.jl` and `test/test_energy_moment.jl` test these terms and the energy moment.

## The energy of the kinetic ionization

A kinetic ionization ([`KineticIonizationTerm`](@ref)) has no energy of its own: the friction on bound electrons of the Fokker-Planck term takes the energy of the ionization from `f`.
With `bound_friction = :ionization_split` of [`CollisionSettings`](@ref), that friction charges `f` the energy ``I_k + \langle E_s \rangle_k`` of each ionization of the charge state ``k``, with ``I_k`` its ionization potential and ``\langle E_s \rangle_k`` the mean energy of its secondary.
The potential is stored in the ions and released at recombination ([`RadiatedPowerTerm`](@ref) with `radiation = :recombination`).
The secondaries join the cold electrons through the closure of the cold density, and [`SecondaryHeatingTerm`](@ref) with `cut = :none` gives them their energy.
[`ionization_energy_flows`](@ref) separates the two parts.

[`EquationSystem`](@ref) requires the heating by secondaries where the sinks of `f` are booked with this friction, with the same spectrum and `cut = :none`.
With DREAM's `bound_friction = :bethe`, the friction does not hold the energy of each ionization, so the requirement does not apply ([`KineticEnergyLossTerm`](@ref)).
Next to a [`KnockOnTerm`](@ref) on the bound electrons, [`SecondaryHeatingTerm`](@ref) with `cut = :none` counts the energy of the secondaries above the knock-on cutoff twice, since the knock-on term also puts them into `f`.
Under the total energy the system refuses the knock-on term; in the row of the cold energy it does not check the pair.

## Currents and conductivity

The total current density is the sum

```math
j_\mathrm{tot} = j_\mathrm{ohm} + j_\mathrm{hot} + j_\mathrm{re},
```

each an algebraic unknown.
The hot current is the [`CurrentMomentTerm`](@ref) of `f` over the hot region.
The runaway current is the current moment of the runaway grid, or ``e c\, n_\mathrm{re}`` of a fluid runaway density.

The ohmic current takes a [`ConductivityTerm`](@ref) with a [`ConductivityModel`](@ref):

- fluid electrons: ``j_\mathrm{ohm} = \sigma E``, with the conductivity of Braams and Karney ([`braams_karney_conductivity`](@ref)), which is DREAM's conductivity of Sauter and Redl in a cylinder;
- fully kinetic: ``j_\mathrm{ohm} = j(f) - j_\mathrm{hot} + E (\sigma - \sigma_\mathrm{kin})``, DREAM's corrected conductivity.

The conductivity of Braams and Karney: B. J. Braams, C. F. F. Karney, Phys. Fluids B 1, 1355 (1989), doi:10.1063/1.858966.
The conductivity of Sauter: O. Sauter, C. Angioni, Y. R. Lin-Liu, Phys. Plasmas 6, 2834 (1999), doi:10.1063/1.873240.
The formulae of Redl: A. Redl et al., Phys. Plasmas 28, 022502 (2021), doi:10.1063/5.0012664.

The test-particle operator does not conserve the momentum of the electron-electron collisions, so its conductivity is below ``\sigma``; the correction adds the difference.
Two models of ``\sigma_\mathrm{kin}``:

- `kinetic = :dream_fit`: DREAM's fit of the conductivity of its kinetic operator ([`test_particle_conductivity_fraction`](@ref));
- `kinetic = :discrete_operator`: the conductivity of the discrete operator of the run ([`kinetic_conductivity`](@ref)), by its linear response to a small field.

With `:discrete_operator`, the steady state of a small field has the reference conductivity whatever the grid and the collision settings are.
The correction is a statement on the steady state: for some collision times after a change of the field, the current stays below ``\sigma E``.
With its heat-flux modes, the momentum-conserving [`FieldParticleTerm`](@ref) brings the conductivity of the distribution close to that of Spitzer and Härm, for a nonrelativistic bulk.
The conductivity of Spitzer and Härm: L. Spitzer, R. Härm, Phys. Rev. 89, 977 (1953), doi:10.1103/PhysRev.89.977.
The system refuses it next to a correction.
The system also refuses a `:discrete_operator` model whose pitch rule differs from that of the [`CurrentMomentTerm`](@ref) in the same equation.

The ohmic heating [`ohmic_heating_term`](@ref) reads the same ohmic current, so the heating and the current agree by construction.
`test/test_conductivity.jl` tests the conductivity against Spitzer and Härm and the Lorentz gas, the corrected conductivity in steady state and transient, and the resistive decay of a kinetic run against a fluid one.

## Induction

[`induction_unknowns`](@ref) and [`induction_equations`](@ref) give the induction equations in cylindrical geometry, with DREAM's pairing of rows and unknowns.
The poloidal flux ``\psi`` is divided by the major radius, as in DREAM.

| Unknown | Equation |
|---|---|
| electric field ``E`` | ``\mathrm{d}\psi/\mathrm{d}t = 2\pi E`` (Faraday) |
| poloidal flux ``\psi`` | ``0 = \frac{1}{r} \partial_r (r\, \partial_r \psi) - 2\pi \mu_0 j_\mathrm{tot}`` (Ampère) |
| plasma current ``I_p`` | ``0 = \sum 2\pi r\, \Delta r\, j_\mathrm{tot} - I_p`` |
| flux at the edge ``\psi_\mathrm{edge}`` | ``0 = \psi_\mathrm{wall} - \mu_0 \ln(b/a)\, I_p - \psi_\mathrm{edge}`` |
| flux at the wall ``\psi_\mathrm{wall}`` | ``\mathrm{d}\psi_\mathrm{wall}/\mathrm{d}t = V_\mathrm{loop,wall}`` |

The row of the field holds the time derivative of the flux ([`TimeDerivative`](@ref) with `of`).
Ampère's law is [`PoloidalFluxDiffusionTerm`](@ref), with the gradient at the edge from ``\psi_\mathrm{edge}``.
The plasma current is [`CrossSectionIntegralTerm`](@ref), and the inductance between the edge and the wall [`edge_to_wall_inductance`](@ref).
A loop voltage of zero at the wall is a perfectly conducting wall.
The caller gives the row of ``j_\mathrm{tot}``, the sum of the currents of the run.
[`initial_poloidal_flux`](@ref) solves Ampère's law for the initial current density, as DREAM does.

`test/test_induction.jl` tests Ampère's law on analytic profiles and the resistive decay with the wall at the edge and at a distance.
`test/test_self_consistent_dream.jl` compares a self-consistent field on both kinetic grids with DREAM.
