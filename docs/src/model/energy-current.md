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
| friction of the hot-tail distribution on the cold electrons, without the full collision frequencies | [`FrictionEnergyTransferTerm`](@ref) |
| electron-electron work on the electrons that cross a momentum interface to a runaway grid (optional) | [`InterfaceCollisionWorkTerm`](@ref) |

[`IonEnergyExchangeTerm`](@ref) with `side = :ions` is the same exchange in the row of the ion energies, so the two sides are antisymmetric.
[`RunawayFluidEnergyTransferTerm`](@ref) is DREAM's heating by runaways without a runaway grid, the power ``e c\, E_c\, n_\mathrm{re}``: runaways at the speed of light against the friction of the critical field ``E_c``, with the Coulomb logarithm at ``p = 20\, m_e c``.
With a runaway grid, [`free_electron_energy_transfer_term`](@ref) takes its place: the energy moment of the electron-electron collisions of that grid.

The channels other than [`SecondaryHeatingTerm`](@ref) are those of DREAM's equation of ``T_\mathrm{cold}`` (`src/Settings/Equations/T_cold.cpp`, DREAM v26.5).
With `radiation = :dream` and `bremsstrahlung_corrections = :dream`, [`RadiatedPowerTerm`](@ref) takes DREAM's radiation.
DREAM adds the collisional transfer from a runaway grid in every mode, and from a hot-tail grid in every mode but one:

| DREAM's mode | Transfer from the hot-tail distribution | Transfer from the runaway grid |
|---|---|---|
| fully kinetic (`collfreq_mode` FULL) | left out | added |
| every other mode | added | added |

[`FrictionEnergyTransferTerm`](@ref) is that transfer from the hot-tail distribution: the moment ``\frac{1}{V'} \sum_{ij} m_e c\, v_i p_i\, \nu_s(p_i) f_{ij} \mathcal{V}'_{ij} \Delta p_i \Delta\xi_j`` at the cell centres, with the superthermal frequencies on the cold electrons alone.
As a moment of `f` it takes any scheme of the operator, and it counts the energy of an electron until it leaves the grid, also through ``p = 0`` under `p_zero_boundary = :outflow`.
For a runaway grid it is DREAM's form of the transfer, the option next to Runes2P's [`free_electron_energy_transfer_term`](@ref) ([Comparison with DREAM](../dream-comparison.md#Defaults-that-differ)).
`test/test_cold_energy_dream.jl` compares the row with ohmic heating, radiation and heat diffusion with DREAM in its fluid mode.

## The total energy in the fully kinetic mode

In the fully kinetic mode the hot-tail distribution `f` covers the bulk.
The hot region is the part of the grid above a threshold ``p_0``, a [`MomentumRegion`](@ref) with a [`ThermalMomentum`](@ref) limit, so that ``p_0`` follows ``T_\mathrm{cold}`` at each radius.
Its electrons hold the energy

```math
W_\mathrm{hot} = \frac{1}{V'} \sum_{ij} w_i\, m_e c^2 (\gamma_i - 1)\, f_{ij}\, \mathcal{V}'_{ij} \Delta p_i\, \Delta\xi_j,
```

the [`EnergyMomentTerm`](@ref) over the hot region, with the envelope ``w`` of that region.
[`total_energy_equation`](@ref) adds the unknown

```math
W_\mathrm{tot} = W_\mathrm{cold} + W_\mathrm{hot}
```

with its algebraic row.
The row of the temperature then holds ``\mathrm{d}W_\mathrm{tot}/\mathrm{d}t``, `TimeDerivative(:total_electron_energy)`, and the temperature follows from ``W_\mathrm{cold} = W_\mathrm{tot} - W_\mathrm{hot}``.

The row lists only the channels by which energy enters or leaves the electrons of the hot-tail grid:

- The work of the field is ``E j_\mathrm{ohm} + E j_\mathrm{hot}`` (two [`ProductTerm`](@ref)s), where the ohmic current already leaves out the hot current, so each electron is counted once.
- The radiation, the heat transport and the exchange with the ions are as in the row of the cold energy, except for the Rechester-Rosenbluth heat transport next to radial transport of `f` ([below](#Radial-transport-of-the-kinetic-electrons)).
- The heating by the secondaries of a kinetic ionization ([`SecondaryHeatingTerm`](@ref)) moves their energy from the sinks to the cold electrons.
- The sinks of the collisions of `f` are [`KineticEnergyLossTerm`](@ref) with `part = :sinks` over the hot region.
- The outflow at ``p_\mathrm{max}`` is [`MomentumBoundaryFluxTerm`](@ref) with `weight = :energy` and `coefficient = -1`.
- The channel of the electrons that cross to a runaway grid is [`InterfaceEnergyFluxTerm`](@ref), with the work of the electron-electron collisions on them, [`InterfaceCollisionWorkTerm`](@ref) (below).
- The energy of the knock-ons that the fluid runaways above the grid create ([`RosenbluthPutvinskiSourceTerm`](@ref)) is the [`KineticEnergyChangeTerm`](@ref) of that source, over the whole grid.
- The energy that radial transport of `f` moves at each radius is the [`KineticEnergyChangeTerm`](@ref) of the [`FokkerPlanckTerm`](@ref) of the transport, over the whole grid ([below](#Radial-transport-of-the-kinetic-electrons)).

The collisional transfer from the hot electrons to the cold ones does not appear in the row.
Neither do the electrons that cross the threshold in momentum at one radius, nor the threshold that moves with the temperature.
The time derivative acts on the energy of the hot region itself, so these exchanges inside ``W_\mathrm{tot}`` at one radius are exact for any time stepper.
Radial transport moves electrons between radii, so it is not among these exchanges, and the row holds it as a term of its own ([below](#Radial-transport-of-the-kinetic-electrons)).

### What the system enforces

Every change of ``W_\mathrm{hot}`` that the row does not list is charged to the cold electrons.
[`EquationSystem`](@ref) therefore checks the row against the equation of `f`.
The docstring of [`total_energy_equation`](@ref) gives the conditions.

- Collisions with sinks (synchrotron loss, bremsstrahlung stopping power, partial or no screening) remove energy from `f`, so the row needs the sinks term.
- The electrons that leave a Fokker-Planck term with collisions at ``p_\mathrm{max}``, or cross its interface to a runaway grid, take their energy with them.
  The row needs the energy flux of that boundary or interface.
  At the interface it also needs one [`InterfaceCollisionWorkTerm`](@ref) of the same interface, with a negative coefficient (below).
  The row refuses the term of an interface whose lower distribution, that of the grid below the face, is not in ``W_\mathrm{hot}``.
  A second [`InterfaceCollisionWorkTerm`](@ref) of one interface, which would count the work twice, is refused in every row, the row of the cold thermal energy alone included.
- Radial transport of `f` changes the energy of `f` at each radius.
  A [`FokkerPlanckTerm`](@ref) of `f` whose operator holds nonzero radial coefficients requires one [`KineticEnergyChangeTerm`](@ref) of that very term in the row, without a region.
  A Fokker-Planck term whose operator declares radial transport but holds no radial coefficients yet, as one built for a [`RechesterRosenbluthUpdate`](@ref), does not yet require it.
  A run checks the radial coefficients again before its first step, so the update passes only when the row holds the term from the build of the system.
- The system checks that the coefficient of that energy term is positive, and a coefficient of 1 closes the budget.
  A second energy term of one Fokker-Planck term is refused, and so is one whose Fokker-Planck term is not in the equation of a distribution in ``W_\mathrm{hot}``.
  Next to it, a [`HeatTransportTerm`](@ref) with the Rechester-Rosenbluth diffusivity is refused, since it would move the heat of the bulk a second time.
- A [`FokkerPlanckTerm`](@ref) of `f` without collisions, with no plasma and no coefficient source, is refused when its operator holds coefficients in momentum that change the energy of `f`.
  These are the advection and diffusion in ``p``, the cross diffusion of the flux in ``p``, and a flux through ``p_\mathrm{min}`` or ``p = 0``.
  No term of the row carries that energy, since the sinks and the outflow at ``p_\mathrm{max}`` belong to the terms with collisions.
  Pitch coefficients move no energy, and the coefficients of a unit field are the work ``E j_\mathrm{hot}``, so both stay allowed.
  A run checks the coefficients again before its first step.
- A [`KnockOnTerm`](@ref) on the grid is refused, since no term carries the energy of the secondaries.
- A [`RosenbluthPutvinskiSourceTerm`](@ref) creates knock-ons whose energy no other term carries.
  The row needs one [`KineticEnergyChangeTerm`](@ref) of that very source, without a region (below).
  The system checks that the coefficient is positive, and a coefficient of 1 closes the budget.
  The row refuses a [`KineticEnergyChangeTerm`](@ref) whose source is not in the equation of a distribution in ``W_\mathrm{hot}``, since the energy that it counts is outside ``W_\mathrm{tot}``.
- A [`LossRateTerm`](@ref) and a [`MaxwellianSourceTerm`](@ref) are refused, since no term carries the energy of the electrons that they remove or add.
- A [`LinearTerm`](@ref) with a nonzero coefficient in the equation of `f` is refused, since it moves electrons into `f` from a kinetic unknown or out of it.

Each refused channel would take its energy from the cold electrons, or give it to them, at the radius where the electrons are moved, created or removed.

A [`ParticleSourceTerm`](@ref) is accepted.
The cold density is the closure ``n_\mathrm{cold} = n_\mathrm{free} - n_\mathrm{hot} - n_\mathrm{re}``.
Here ``n_\mathrm{hot}`` is the density of the hot region ([`DensityMomentTerm`](@ref) over it), and ``n_\mathrm{re}`` the runaway density where the system has one.
With the constraint ``0 = n(f) - n_\mathrm{cold} - n_\mathrm{hot}`` for its magnitude, which keeps ``n(f)`` at ``n_\mathrm{free} - n_\mathrm{re}``, the source moves electrons between the cold population and `f`.
The cold electrons therefore rightly pay the energy that its electrons bring into ``W_\mathrm{hot}``.
A [`MaxwellianSourceTerm`](@ref) adds or removes electrons from outside the system, so no population of the system should pay or gain their energy.

The row creates the energy of the knock-ons of the Rosenbluth-Putvinski source in the system, a choice of this package ([Choices of this package](../choices.md#Total-energy-of-the-electrons)).
The primaries of the source are the fluid runaways above the grid, which have no energy unknown, so no population pays that energy.
The alternative, which the package rejects, charges it to the cold electrons, as a row without [`KineticEnergyChangeTerm`](@ref) would.
The term takes no region, since every knock-on is in ``W_\mathrm{tot}``.
A secondary below the threshold of the hot region heats the cold electrons, and one above it joins ``W_\mathrm{hot}``.

The term overlaps with the heating by fluid runaways, [`RunawayFluidEnergyTransferTerm`](@ref).
That heating gives the cold electrons ``e c\, E_c\, n_\mathrm{re}``, the whole collisional stopping power of the runaways on the free electrons.
The knock-ons between the cutoff ``p_\mathrm{cut}`` of the source and ``p_\mathrm{max}`` take their energy from that loss, so with both terms in the row ``W_\mathrm{tot}`` gains their energy twice.
DREAM has both terms too, in separate reservoirs: its equation of ``T_\mathrm{cold}`` holds ``e c\, E_c\, n_\mathrm{re}`` and no term of the source.

Where all free electrons are cold and ``n_\mathrm{tot}`` equals their density, the ratio of the power ``P`` of the knock-on term to the heating is, in the limit of fine momentum cells,

```math
\frac{P}{e c\, E_c\, n_\mathrm{re}} = \frac{\ln[(\gamma_\mathrm{max} - 1)/(\gamma_\mathrm{cut} - 1)]}{2 \ln\Lambda_{ee}(p = 20)},
```

with ``\gamma_\mathrm{cut}`` and ``\gamma_\mathrm{max}`` the Lorentz factors at ``p_\mathrm{cut}`` and ``p_\mathrm{max}``, and ``\ln\Lambda_{ee}(p = 20)`` the Coulomb logarithm of the heating.
At ``p_\mathrm{cut} = 0.01``, ``p_\mathrm{max} = 0.8``, 10 eV and ``10^{20}\, \mathrm{m^{-3}}``, with the `:energy_dependent` logarithm, the ratio is 0.251.

The term does not remove the thermal energy of the target electron.
Take the closure of the cold density and the constraint of a [`ParticleSourceTerm`](@ref) (above).
Each knock-on above the threshold of the hot region then takes one electron out of the cold population, while ``W_\mathrm{cold}`` keeps its thermal energy ``\tfrac{3}{2} T``.
The cold electrons that remain therefore gain ``\tfrac{3}{2} T`` per such knock-on.
That is 0.62 % of the term at 10 eV, ``p_\mathrm{cut} = 0.04`` and ``p_\mathrm{max} = 0.8``, where every knock-on lies above the threshold.
A bound target, which ``n_\mathrm{tot}`` counts, does not pay its ionization energy either, and DREAM behaves the same in both respects.

### Radial transport of the kinetic electrons

Radial transport of `f`, such as the diffusion along stochastic field lines of [`add_rechester_rosenbluth_diffusion!`](@ref), is a [`FokkerPlanckTerm`](@ref) of its own.
Under the total energy its operator holds radial and pitch coefficients alone: no term of the row carries the energy that coefficients in momentum would move, and the system refuses them (below).
Its [`KineticEnergyChangeTerm`](@ref) is the power density ``P_k`` [W/m³] that the radial coefficients of the operator give the electrons of `f` at radius ``k``, over the whole grid:

```math
P_k = -\frac{1}{V'_k \Delta r_k} \sum_{ij} m_e c^2 (\gamma_i - 1)\, \Delta p_i\, \Delta\xi_j \left(\Phi_{ij,k+1} - \Phi_{ij,k}\right).
```

``P_k`` is positive where the electrons at radius ``k`` gain energy.
``V'_k`` is the spatial Jacobian of the radius (``V' = r`` in a cylinder) and ``\Delta r_k`` its width.
``\Phi_{ij,k} = \mathcal{V}'_{ij,k} \Gamma^r_{ij,k}`` is the radial flux of the cell ``(i, j)`` through face ``k``, the inner face of radius ``k``, positive outward.
``\mathcal{V}'_{ij,k}`` is the phase-space Jacobian on that face ([`phase_space_jacobian_r_faces`](@ref)).
``\Gamma^r = A^r f - D^{rr} \partial_r f`` is the radial flux density of [`AdvectionDiffusion`](@ref), and face ``N + 1`` is the edge of the plasma, with ``N`` the number of radii.

The fluxes are those of the discretised equation of `f`.
``P_k`` is therefore the change that the transport makes to the energy of `f` at radius ``k``, to round-off.
The term takes the whole grid, a choice of this package ([Choices of this package](../choices.md#Total-energy-of-the-electrons)).
The transport only moves energy between the radii and out through the edge.
The volume integral ``\sum_k V'_k \Delta r_k P_k`` is minus the energy flux out through the edge, the energy that the plasma loses there.

That integral is zero for an operator with a closed edge, `r_boundary = :zero_flux` of [`AdvectionDiffusion`](@ref), whatever the threshold of the hot region at each radius.
The row holds ``P_k``.
``W_\mathrm{hot}`` changes by the same moment over the hot region alone, with the envelope ``w_{i,k}`` of that region, from 0 below the threshold to 1 above it.
Since ``W_\mathrm{cold} = W_\mathrm{tot} - W_\mathrm{hot}``, ``W_\mathrm{cold}`` changes by the moment with the weight ``1 - w_{i,k}``: the energy that the transport moves with the electrons below the threshold.

An electron that the transport carries from radius ``k``, where it lies above the threshold, to radius ``k + 1``, where it lies below it, shows how this works:

1. At radius ``k`` the electron leaves ``W_\mathrm{hot}``, and ``W_\mathrm{tot}`` loses the same energy through ``P_k``, so ``W_\mathrm{cold}`` there receives nothing from it.
2. At radius ``k + 1`` it lies below the threshold, so ``W_\mathrm{hot}`` does not change, while ``P_{k+1}`` adds its energy to ``W_\mathrm{tot}``.
3. Its energy therefore enters ``W_\mathrm{cold}`` at radius ``k + 1``.

The cold density does not follow the electrons below the threshold, since it is the closure of the cold density (above), which reads ``n_\mathrm{hot}`` and not ``n(f)``.
``T_\mathrm{cold}`` therefore rises where the transport brings their energy and falls where it takes it.
Where the transport changes ``n_\mathrm{hot}``, ``n_\mathrm{cold}`` changes by the opposite amount, and that part of the change brings no energy to ``W_\mathrm{cold}``, as for a knock-on (above).

A [`ParticleSourceTerm`](@ref) adds electrons to `f` with its shape, by default a Maxwellian at ``T_\mathrm{cold}``, at the rate of its magnitude, and the row has no term for their energy.
Its part above the threshold changes ``W_\mathrm{hot}`` and is charged to the cold electrons, and its part below the threshold changes no term of the row.
With the constraint above for its magnitude, the source refills `f` where the transport takes electrons out of it and drains it where the transport brings them.

[`HeatTransportTerm`](@ref) is a separate model of the heat transport of the cold electrons, the diffusion of ``T_\mathrm{cold}``.
Its Rechester-Rosenbluth diffusivity is the energy moment of the kinetic diffusion of [`add_rechester_rosenbluth_diffusion!`](@ref) on a Maxwellian of uniform density.
Next to the energy term of the transport of `f`, that diffusivity would move the heat of the bulk a second time, so the row of the total energy refuses the pair, whatever its level.
On a Maxwellian bulk of uniform density at a temperature profile on 8 radii, the two agree to 1 % at the six inner radii and to 13 % at the seventh, near the edge (`test/test_heat_transport.jl`).
A prescribed diffusivity of [`HeatTransportTerm`](@ref) stays allowed, as a heat transport of the user's own on top of the kinetic transport.

`test/test_energy_transfer.jl` compares the term with the moment of the operator applied to `f`, and `test/test_energy_budget.jl` closes the budget of ``W_\mathrm{tot}`` at each radius and in the volume with it.

### Why the hot-tail grid only

The runaway grid stays outside ``W_\mathrm{tot}``.
A relative error ``\varepsilon`` of ``W_\mathrm{tot}`` becomes a relative error of the cold energy of

```math
\varepsilon\, \frac{W_\mathrm{tot}}{W_\mathrm{cold}},
```

so the temperature is lost to cancellation when ``W_\mathrm{tot}`` is much larger than ``W_\mathrm{cold}``.
Keeping the runaways out keeps this ratio small.
The runaway grid enters the row through three channels only.
They are [`free_electron_energy_transfer_term`](@ref) for its collisions with the cold electrons, [`InterfaceEnergyFluxTerm`](@ref) for the electrons that it receives, and [`InterfaceCollisionWorkTerm`](@ref) for the work of the collisions on those electrons.

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

Neither part takes a flux through ``p_\mathrm{max}``, whatever the upper boundary of the Fokker-Planck term.
The energy of the electrons that cross there, that of the last momentum cell, belongs to the boundary term alone, so it is not counted twice, and the cold electrons do not exchange it.
That term is [`MomentumBoundaryFluxTerm`](@ref) at an outer boundary and [`InterfaceEnergyFluxTerm`](@ref) at the interface to a runaway grid.
With the sinks charged to ``W_\mathrm{tot}``, the cold electrons receive exactly the free-electron part of what the hot electrons lose.

The momentum face values of the parts take the upwind side of the whole advection of the Fokker-Planck term, the sum of its collisions and its electric field.
This holds under every momentum scheme whose face value has a side: `:upwind`, `:quick`, `:upwind_second_order` and the flux limiters.
The four-point schemes among them have a side on a grid of at least 3 momentum cells, and take the central face value on fewer.
The field part is the flux of the electric field alone, the flux of the whole less those of the free-electron part and the sinks, and it takes the same side.
That face value depends on ``f`` and on the side alone, so the field part, the free-electron part and the sinks sum to the momentum flux of the term face by face.

With each operator of a part on the side of its own advection, the parts would miss that flux at every face where the advection of the operator and the whole advection have opposite signs.
This happens where the field beats the friction.
It also happens at the face ``p = 0`` under a four-point scheme with `p_zero_boundary = :outflow`, where the friction vanishes and the field alone sets the side.
In a test of a 100 eV plasma at 2 V/m (`test/test_energy_transfer.jl`), their sum then differs from the flux of the whole by up to 0.18 of the largest face flux of the whole under the flux limiters.
Under the linear schemes with a side it differs by up to 0.44 of that flux.

The central scheme and exponential fitting in ``p`` have no side.
With linear schemes in ``\xi`` and ``r`` the parts then take the moment of their operators applied to ``f``.
The central face value is the same on either side, so the parts sum to the flux under it as well.

Exponential fitting in ``p`` is the exception to that sum: its face value follows the coefficients of each operator, so the parts do not sum to the flux there.
This holds with or without a limiter in ``\xi`` or ``r``, and for the faces inside a grid.
At the interface to a runaway grid every part takes the one face value of the whole flux, so the parts sum to the flux there under every scheme ([The energy step at the interface](#The-energy-step-at-the-interface)).

Under a momentum scheme with a side, and under a flux limiter in any direction, the term takes the moment in the form of the momentum faces, since the pitch fluxes sum to zero over a momentum row:

```math
L = -\frac{1}{V'} \sum_j \Delta\xi_j \sum_k \left(\varepsilon_k - \varepsilon_{k-1}\right) F_{kj},
```

with ``\varepsilon_k`` the energy ``w_k m_e c^2 (\gamma_k - 1)`` of momentum cell ``k``, ``\varepsilon_0 = 0``, and ``F_{kj}`` the momentum flux of ``C_a - C_b`` through the lower face of cell ``(k, j)``, multiplied by ``\mathcal{V}'``.
The term reads the sign of the whole advection, not of the field alone, and has no column of the field unknown.
Where a face changes its side, the whole advection passes through zero, so the momentum flux of the Fokker-Planck term, the sum of the parts, stays continuous.
The advection of each part does not vanish there, so each part jumps as its face value moves from one cell to the other.

With a row that feeds back on the side, Newton's method may not converge on an implicit step that straddles a change of side.
An example is the temperature of the cold electrons heated by a runaway grid, since the friction that sets the side follows that temperature.
Newton's iteration then alternates between the two sides, the step control halves the step, and the run continues with the result of the smaller steps.
`test/test_runaway_heating_dream.jl` shows this under `:upwind` and under the flux limiter `:tcdf`.

[`MomentumBoundaryFluxTerm`](@ref) with `weight = :energy` and [`InterfaceEnergyFluxTerm`](@ref) weight the flux at the top of the grid with the kinetic energy of the last momentum cell.
That is the energy that the discrete energy moment of the grid loses with the electrons.
The electrons that cross to the runaway grid take that energy with them and do not heat the cold ones with it.
`test/test_energy_transfer.jl` and `test/test_energy_moment.jl` test these terms and the energy moment, and `test/test_friction_energy_transfer.jl` tests [`FrictionEnergyTransferTerm`](@ref).

### The side on a coarse momentum grid

This subsection shows how the shared side changes the free-electron part on a coarse momentum grid, and how both choices of the side converge.

The shared side is the upwind side of the whole advection, which all parts take.
Where the field beats the friction at a face, the face value of every part there is that of the cell below the face.
The friction of the free electrons then carries electrons of that cell down through the face, and the term counts that flux with the step of the energy between the two cells.
On its own side the friction would take the face value of the cell above the face, and would carry the electrons of a cell down through the face below it instead.
On a coarse momentum grid the two give different energies to the cold electrons.

The table holds the free-electron part divided by the friction moment at the cell centres.
That moment is [`FrictionEnergyTransferTerm`](@ref) with the collision settings of the grid, on the same grid and state, and it is not an exact value.
The state is the start of the test case `runaway_heating_example` at `compat = :runes2p`, at 100 eV and 1.5 V/m.
In it ``f = 1`` in the momentum cells below ``p = 0.5\, m_e c`` of the last pitch cell and zero elsewhere, and only the width of the momentum cells varies.
The column "Each part on its own side" is the moment of the operator applied to ``f``, each face on the side of the advection of its own operator, which the term does not take.

| Momentum cell width [``m_e c``] | Shared side | Each part on its own side |
|---|---|---|
| 0.5 | 2.127 | 0.000 |
| 0.25 | 1.543 | 0.486 |
| 0.125 | 1.267 | 0.740 |
| 0.0625 | 1.133 | 0.869 |

- Both columns converge to 1 at first order in the cell width, from above and from below, and their mean at second order (`test/test_runaway_heating_dream.jl`).
- At the width 0.5 the electrons fill the first momentum cell alone.
  On their own side the friction carries them only through ``p = 0``, which the grid of the case closes, so that column is zero there.
- The ratio 2.127 needs ``f`` in the forward pitch cell.
  With ``f = 1`` in every pitch cell below ``p = 0.5\, m_e c`` it is 0.978 at the width 0.5, since the field does not beat the friction at that face in every pitch cell.

### The energy step at the interface

An electron that crosses the interface leaves the hot-tail grid with the kinetic energy ``E_{\mathrm{low},N}`` of its last momentum cell.
It enters the runaway grid with the energy ``E_{\mathrm{up},1}`` of the first momentum cell there.
The step ``E_{\mathrm{up},1} - E_{\mathrm{low},N}`` is about ``m_e c^2 \beta\, \delta p``, with ``\beta = v/c`` at the face and ``\delta p`` the distance of the two cell centres in units of ``m_e c``.
It therefore vanishes as the cells at the face are refined.

Neither the energy moments of the two operators nor the sinks term take a flux through the face, so no operator pays the step.
Each part of the flux through the face pays it for its electrons.
A payer loses the step when the flux of its part goes up through the face, and gains it when that flux goes down.

| Part of the flux through the face | Pays the step | Term of the row |
|---|---|---|
| the field | the field | no term of its own: ``E j_\mathrm{hot}`` holds the part up to the face (below) |
| the electron-electron collisions: friction and energy diffusion | the cold electrons | [`InterfaceCollisionWorkTerm`](@ref) |
| the friction on bound electrons, the energy diffusion by bound electrons under partial screening, bremsstrahlung and synchrotron radiation | the sinks | none |

The ions scatter the electrons in pitch only, so they move none across the face.
The step of the sinks changes the energy of the runaway grid alone, which is outside ``W_\mathrm{tot}``, so the row takes no term for it.

The row credits ``W_\mathrm{tot}`` with the field work ``E j_\mathrm{hot}``, and the hot current ``j_\mathrm{hot}`` covers the cells of `f` up to the face.
``W_\mathrm{hot}`` itself gains the work of the discrete field term of `f` only up to the centre of its last cell.
At the face it loses ``\Gamma E_{\mathrm{low},N}``, with ``\Gamma`` the flux through the face.
So ``W_\mathrm{tot}`` holds about ``\Gamma_E (E_\mathrm{face} - E_{\mathrm{low},N})`` that ``W_\mathrm{hot}`` never receives, with ``\Gamma_E`` the field part of the flux and ``E_\mathrm{face}`` the kinetic energy at the face.
Since ``W_\mathrm{cold} = W_\mathrm{tot} - W_\mathrm{hot}``, the cold electrons receive it as heat.

The runaway grid receives the whole step with the electron, so that part of the step is counted twice.
The part is positive where the field drives electrons up through the face, and negative where it drives them down.
It is roughly the fraction ``\Delta p_\mathrm{low}/(\Delta p_\mathrm{low} + \Delta p_\mathrm{up})`` of the step of the field, with ``\Delta p_\mathrm{low}`` and ``\Delta p_\mathrm{up}`` the widths of the two cells at the face, and it falls with the width of these cells.
The row has no term for it, as for the gap at the threshold of the hot region ([The field work on the hot region](#The-field-work-on-the-hot-region)).

Each part takes the face value of the whole flux: the upwind side, the linear interpolation or the weight of exponential fitting that the coefficients of all processes set.
The parts therefore add up to the flux through the face, as the docstring of [`InterfaceCollisionWorkTerm`](@ref) shows.

The term charges the work of the electron-electron collisions to the cold electrons, a choice of this package ([Choices of this package](../choices.md#Total-energy-of-the-electrons)).
The cold electrons are the targets of these collisions, and inside each grid they gain the energy that the collisions take from the electrons.
Where the friction pulls electrons down through the face, the cold electrons gain the step of each of them, and where the energy diffusion pushes electrons up, they pay it.

`test/test_energy_budget.jl` closes the energy of the two grids across the face under a field, with the whole step of the field at the face written out as a channel of its own and without the row of the total energy.
Without a field, it conserves the total energy of a two-grid system through the row of the total energy with the term.
Both tests also show what they miss without the term.
With a field, the row of the total energy counts the part of the step of the field up to the face twice, as described above.

## The energy of the kinetic ionization

A kinetic ionization ([`KineticIonizationTerm`](@ref)) has no energy of its own: the friction on bound electrons of the Fokker-Planck term takes the energy of the ionization from `f`.
With `bound_friction = :ionization_split` of [`CollisionSettings`](@ref), that friction charges `f` the energy ``I_k + \langle E_s \rangle_k`` of each ionization of the charge state ``k``, with ``I_k`` its ionization potential and ``\langle E_s \rangle_k`` the mean energy of its secondary.
The potential is stored in the ions and released at recombination ([`RadiatedPowerTerm`](@ref) with `radiation = :recombination`).
The secondaries join the cold electrons through the closure of the cold density, and [`SecondaryHeatingTerm`](@ref) with `cut = :none` gives them their energy.
[`ionization_energy_flows`](@ref) separates the two parts.

[`EquationSystem`](@ref) requires the heating by secondaries where the sinks of `f` are booked with this friction, with the same spectrum and `cut = :none` or `cut = :knockon`.
With DREAM's `bound_friction = :bethe`, the friction does not hold the energy of each ionization, so the requirement does not apply ([`KineticEnergyLossTerm`](@ref)).

Next to a [`KnockOnTerm`](@ref) on the bound electrons, [`SecondaryHeatingTerm`](@ref) with `cut = :none` counts the energy of the secondaries above the knock-on cutoff twice, since the knock-on term also puts them into a distribution.
With `cut = :knockon` and the knock-on term as `knockon_term`, the heating gives the cold electrons only the energy of the secondaries below the knock-on cutoff at each radius ([`secondary_energy_below`](@ref), [`beb_secondary_energy_below`](@ref)).
The heating reads the cutoff at each evaluation, so it follows [`KnockOnCutoffUpdate`](@ref).

The system refuses `cut = :none` and `cut = :threshold` next to a knock-on term of the same primary whose target density is the unknown `:total_electron_density`, which counts the bound electrons.
With `cut = :knockon`, `knockon_term` must then be such a term.
The system refuses `cut = :knockon` with a knock-on term on `:cold_electron_density` or `:runaway_density`, which count free electrons only, since no term then puts the secondaries above the cutoff into a distribution.
A target density given as values or under another name is not checked, since the system cannot tell which electrons it counts.

Under the total energy the system refuses the knock-on term.

## Currents and conductivity

The total current density is the sum

```math
j_\mathrm{tot} = j_\mathrm{ohm} + j_\mathrm{hot} + j_\mathrm{re},
```

each an algebraic unknown.
The hot current is the [`CurrentMomentTerm`](@ref) of `f` over the hot region.
The runaway current is the current moment of the runaway grid, or ``e c\, n_\mathrm{re}`` of a fluid runaway density.

The ohmic current takes a [`ConductivityTerm`](@ref) with a [`ConductivityModel`](@ref):

- fluid electrons: ``j_\mathrm{ohm} = \sigma E``, with the conductivity of Braams and Karney ([`braams_karney_conductivity`](@ref)), which is DREAM's conductivity of Sauter and Redl in a cylinder.
- fully kinetic: ``j_\mathrm{ohm} = j(f) - j_\mathrm{hot} + E (\sigma - \sigma_\mathrm{kin})``, DREAM's corrected conductivity.

The conductivity of Braams and Karney: B. J. Braams, C. F. F. Karney, Phys. Fluids B 1, 1355 (1989), doi:10.1063/1.858966.
The conductivity of Sauter: O. Sauter, C. Angioni, Y. R. Lin-Liu, Phys. Plasmas 6, 2834 (1999), doi:10.1063/1.873240.
The formulae of Redl: A. Redl et al., Phys. Plasmas 28, 022502 (2021), doi:10.1063/5.0012664.

The test-particle operator does not conserve the momentum of the electron-electron collisions, so its conductivity is below ``\sigma``.
The correction adds the difference.
Two models of ``\sigma_\mathrm{kin}``:

- `kinetic = :dream_fit`: DREAM's fit of the conductivity of its kinetic operator ([`test_particle_conductivity_fraction`](@ref)).
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
`test/test_first_step_field.jl` checks the field of one short step of a system that holds its current against Braginskii's, independently of DREAM ([Collision models](collisions.md#The-field-of-the-first-step)).
