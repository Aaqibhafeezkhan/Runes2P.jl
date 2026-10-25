# Runes2P

```@docs
Runes2P
```

Runes2P solves the relativistic Fokker-Planck equation of the electrons of a cylindrical plasma together with the fluid quantities that couple to it.
The unknowns are laid out by an [`UnknownLayout`](@ref): distribution functions on a [`KineticGrid`](@ref), whose momentum grid may differ between radii, radial profiles and scalars.
Each unknown has an [`Equation`](@ref) built from [`EquationTerm`](@ref)s.
The equations form an [`EquationSystem`](@ref), ``M \, \mathrm{d}u/\mathrm{d}t = \mathrm{rhs}(u, t)``.
[`integrate`](@ref) advances it with backward Euler steps, or BDF2 steps with [`StepController`](@ref), each solved by Newton's method ([`NewtonSolver`](@ref)), and [`write_output`](@ref) writes the result.

## Citing and acknowledgements

These are requests, not conditions of the licence.

- Parts of Runes2P are ported from DREAM and compared with it.
  Please cite the DREAM paper: M. Hoppe, O. Embréus and T. Fülöp, Computer Physics Communications 268, 108098 (2021), [doi:10.1016/j.cpc.2021.108098](https://doi.org/10.1016/j.cpc.2021.108098).
- For hydrogen and its isotopes, neon and argon, the `:chianti` form of [`ionization_cross_section`](@ref) uses the fit parameters of the CHIANTI atomic database, version 11.0.2.
  CHIANTI is a collaborative project involving George Mason University, the University of Michigan (USA), the University of Cambridge (UK) and NASA Goddard Space Flight Center (USA).
  Work that uses these data should cite:
  - K. P. Dere et al., Astronomy and Astrophysics Supplement Series 125, 149 (1997), [doi:10.1051/aas:1997368](https://doi.org/10.1051/aas:1997368),
  - R. P. Dufresne et al., The Astrophysical Journal 974, 71 (2024), [doi:10.3847/1538-4357/ad6765](https://doi.org/10.3847/1538-4357/ad6765),
  - for the ionization data, K. P. Dere, Astronomy and Astrophysics 466, 771 (2007), [doi:10.1051/0004-6361:20066728](https://doi.org/10.1051/0004-6361:20066728).
- The ADF11 rate coefficients that [`read_adas`](@ref) reads are taken from OPEN-ADAS (<https://open.adas.ac.uk>) under its terms and conditions.
  For H, D and T, every rate coefficient except charge exchange comes from the high-density files of DREAM's repository instead.
  No ADAS data file is distributed with Runes2P.
  ADAS: H. P. Summers, The ADAS User Manual, version 2.6 (2004), <https://www.adas.ac.uk>.
- The ionization energies of [`IONIZATION_ENERGIES`](@ref) are those of the NIST Atomic Spectra Database (Kramida, Ralchenko, Reader and the NIST ASD Team, [doi:10.18434/T4W30F](https://doi.org/10.18434/T4W30F)), through DREAM's table.
- NIST ESTAR is the reference of the radiative stopping power that the docstring of [`CollisionSettings`](@ref) names.
  ESTAR: M. J. Berger, J. S. Coursey, M. A. Zucker and J. Chang, NIST Standard Reference Database 124 (2017), [doi:10.18434/T4NC7P](https://doi.org/10.18434/T4NC7P).
- The DREAM reference data of the tests come from DREAM's public examples and from synthetic cases, generated with DREAM release v26.5 and DREAM's public branch `boltzmann-eqterm`.
  A further set holds GSL 2.8's bicubic interpolation as DREAM uses it, on a synthetic table (see [Reference data](reference-data.md)).

The licences of the code and data that Runes2P builds on are in the file `THIRD_PARTY_NOTICES.md` of the repository.

## Installation

Runes2P is added from its repository, as described on the page [Getting started](getting-started.md#Installation).

Three parts are package extensions, enabled by loading their packages next to Runes2P (see [Extensions](extensions.md)):

- `HDF5`: the output files and checkpoints ([`write_output`](@ref), [`read_output`](@ref)),
- `SciMLBase`: the SciML form of an equation system ([`ode_function`](@ref)),
- `Interpolations`: interpolators of Interpolations.jl as radial profiles.

Questions and bug reports go to the issue tracker of the repository, <https://github.com/osrep/Runes2P.jl/issues>.

## Map of the pages

- [Getting started](getting-started.md): installation, a first run with the examples, the settings of a run and its output files.
- Model and numerics:
  - [Unknowns and the equation system](model/unknowns.md): the unknown layout, equations and terms, the equation system and its pairing checks.
  - [Grids](model/grids.md): momentum, pitch and radial grids, the hot-tail and runaway grids and their interface.
  - [Collision models](model/collisions.md): the Fokker-Planck operator, screening, bremsstrahlung, synchrotron, knock-on and field-particle collisions.
  - [Atomic data and ionization](model/atomic-data.md): ADAS rates, the kinetic ionization cross sections, radiation.
  - [Energy and current balance](model/energy-current.md): the energy of the electrons, the energy transfer between the distributions, conductivity and induction.
  - [Positivity and flux limiters](model/positivity.md): the interpolation schemes of the advection, the limiters, the lower limits of the Newton step.
  - [Time stepping and solvers](model/time-stepping.md): backward Euler, the step controllers, Newton, the linear solvers, checkpoints and the stop.
  - [Output files](model/output.md): the HDF5 output and checkpoint format, reading and joining segments.
- [Choices of this package](choices.md): the values and defaults that are choices of this package, with their reasons.
- [Comparison with DREAM](dream-comparison.md): the features of DREAM and their state in Runes2P, the departures.
- [Reference data](reference-data.md): the reference sets of the tests and how to regenerate them.
- API:
  - [Overview](api-overview.md): how the API reference is organized and built.
  - [Grids](api/grids.md), [Physics](api/physics.md), [Operators](api/operators.md), [Equation system](api/system.md), [Solvers and runs](api/solvers.md): the docstrings by category.
  - [Extensions](extensions.md): the docstrings of the package extensions.
