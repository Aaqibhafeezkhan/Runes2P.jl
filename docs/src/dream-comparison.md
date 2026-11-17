# Comparison with DREAM

```@meta
CurrentModule = Runes2P
```

This page compares Runes2P with DREAM, the code whose design it follows (M. Hoppe, O. Embréus and T. Fülöp, Comput. Phys. Commun. 268, 108098 (2021), [doi:10.1016/j.cpc.2021.108098](https://doi.org/10.1016/j.cpc.2021.108098)).
The comparison is with DREAM release v26.5 and its branch `boltzmann-eqterm` at commit `93d27e1`, the builds of the [Reference data](reference-data.md).
The file and line citations of DREAM's source refer to commit `93d27e1` unless they name another version.

Agreement with DREAM shows that the two codes use the same method, not that the method is right.
Each piece of physics therefore also has a check that does not rely on DREAM: against the literature, an analytic limit, a manufactured solution or a second operator of the code.
The model pages name these checks.
The collision models collect theirs in one section, next to the comparisons with DREAM ([Collision models](model/collisions.md#Comparisons)).

## Features of DREAM

| ✅ implemented | 🚧 partial | ➖ not implemented | features |
|---|---|---|---|
| 76 | 17 | 47 | 140 |

The tables follow the groups of DREAM's settings (`py/DREAM/Settings`).
The column "DREAM" names the setting, class or unknown.
The column "Runes2P" gives the state, the names that implement the feature and, for a partial feature, the part that is in the code.
A cell that ends with "(note)" has a note under its table, which gives the detail.

| State | Meaning |
|---|---|
| ✅ implemented | DREAM's feature is in the code, possibly next to other options |
| 🚧 partial | a part of the feature is in the code, and the cell names that part |
| ➖ not implemented | the code has no counterpart |

Runes2P is cylindrical: a straight magnetic field, constant on each flux surface, so no trapping and no flux-surface averages ([`cylindrical_grid`](@ref)).
Every toroidal feature of DREAM is therefore not implemented.

### Run modes

Runes2P has no switch of the run mode.
A run is an [`EquationSystem`](@ref) of the terms the caller chooses, so a mode is a choice of unknowns and terms.

| Mode | DREAM | Runes2P |
|---|---|---|
| fully kinetic | hot-tail grid with `COLLFREQ_MODE_FULL`, the bulk inside `f_hot` | ✅ `mode = :full` of [`CollisionSettings`](@ref), [`total_energy_equation`](@ref) |
| superthermal | hot-tail grid with `COLLFREQ_MODE_SUPERTHERMAL`, a fluid cold population | ✅ `mode = :superthermal` of [`CollisionSettings`](@ref), [`ConductivityTerm`](@ref), [`FrictionEnergyTransferTerm`](@ref) (note) |
| isotropic | one pitch cell, the pitch-averaged equation | ➖ a grid of one pitch cell is refused by [`fokker_planck_operator`](@ref) |
| fluid | no kinetic grid | 🚧 the field, the cold electrons and the ions, but no Dreicer or avalanche term that feeds ``n_\mathrm{re}`` |

- **Superthermal.**
  [`ConductivityTerm`](@ref) is for Ohm's law of the cold electrons.
  [`FrictionEnergyTransferTerm`](@ref) is for their heating by ``f_\mathrm{hot}``.
  The disruption case in this mode is compared with DREAM end to end ([The disruption case](disruption-case.md)).

### Grids

| Feature | DREAM | Runes2P |
|---|---|---|
| cylindrical radial grid, custom faces | `RadialGrid.TYPE_CYLINDRICAL`, `setCustomGridPoints` | ✅ [`RadialGrid`](@ref), [`uniform_r_faces`](@ref) |
| minor radius | `setMinorRadius` | ✅ the last face of a [`RadialGrid`](@ref) |
| major radius | `setMajorRadius` | ➖ (note) |
| inner radius | `setInnerRadius` | 🚧 a [`RadialGrid`](@ref) may start above the axis, with no comparison with DREAM's inner boundary (note) |
| magnetic field on the axis | `setB0` | ✅ the magnetic field of a [`PlasmaState`](@ref), one value per radius |
| wall radius | `setWallRadius` | ✅ `wall_radius` of [`induction_equations`](@ref) |
| analytic toroidal equilibrium, shaping | `TYPE_ANALYTIC_TOROIDAL`, `setShaping`, `setShapeParameter` | ➖ |
| numerical equilibrium | `TYPE_NUMERICAL`, `FILE_FORMAT_LUKE`, `setNumerical` | ➖ |
| poloidal-angle resolution of the averages | `setNtheta`, `setNthetaOut` | ➖ |
| momentum grid in ``(p, \xi)`` | `MomentumGrid.TYPE_PXI` | ✅ [`MomentumGrid`](@ref) |
| momentum grid in ``(p_\parallel, p_\perp)`` | `TYPE_PPARPPERP` | ➖ |
| uniform momentum grid | `PGrid.TYPE_UNIFORM` | ✅ [`uniform_p_faces`](@ref) |
| biuniform momentum grid | `PGrid.TYPE_BIUNIFORM` | ✅ [`biuniform_p_faces`](@ref) |
| custom momentum faces, ``p_\mathrm{min}`` | `PGrid.TYPE_CUSTOM`, `setPmin` | ✅ [`MomentumGrid`](@ref) from its faces |
| uniform pitch grid | `XiGrid.TYPE_UNIFORM` | ✅ [`uniform_xi_faces`](@ref) |
| biuniform in the pitch angle | `XiGrid.TYPE_BIUNIFORM_THETA` | ✅ [`biuniform_theta_xi_faces`](@ref) |
| biuniform in ``\xi``, uniform in the pitch angle | `XiGrid.TYPE_BIUNIFORM`, `TYPE_UNIFORM_THETA` | ✅ as custom faces of a [`MomentumGrid`](@ref) |
| custom pitch faces | `XiGrid.TYPE_CUSTOM` | ✅ [`MomentumGrid`](@ref) from its faces |
| trapped-passing boundary layer | `XiGrid.TYPE_TRAPPED`, `setTrappedPassingBoundaryLayerGrid` | ➖ |
| hot-tail and runaway grids, the flux between them | `hottailgrid`, `runawaygrid` | ✅ [`runaway_grid`](@ref), [`MomentumInterfaceTerm`](@ref) |

- **Major radius.**
  A cylinder has none, as in DREAM's cylindrical grid.
  `major_radius` is only an input of the Rechester–Rosenbluth coefficient ([`add_rechester_rosenbluth_diffusion!`](@ref), [`RechesterRosenbluthUpdate`](@ref), [`HeatTransportTerm`](@ref)).
- **Inner radius.**
  A [`RadialGrid`](@ref) that starts above the axis has no flux through its first face.

### Collisions

| Feature | DREAM | Runes2P |
|---|---|---|
| collision frequency mode | `COLLFREQ_MODE_SUPERTHERMAL`, `FULL`, `ULTRA_RELATIVISTIC` | ✅ `mode` of [`CollisionSettings`](@ref) |
| screening | `COLLFREQ_TYPE_COMPLETELY_SCREENED`, `NON_SCREENED`, `PARTIALLY_SCREENED` | ✅ `screening` |
| Walkowiak screening | `COLLFREQ_TYPE_WALKOWIAK` | ➖ |
| Coulomb logarithm | `LNLAMBDA_CONSTANT`, `ENERGY_DEPENDENT`, `THERMAL` | ✅ [`coulomb_logarithm`](@ref) (`:relativistic` is `CONSTANT`) |
| energy diffusion with screening | `SCREENED_DIFFUSION_MODE_ZERO`, `MAXWELLIAN` | ✅ `screened_diffusion` |
| bremsstrahlung stopping power | `BREMSSTRAHLUNG_MODE_NEGLECT`, `STOPPING_POWER` | ✅ `bremsstrahlung`, [`bremsstrahlung_cross_section`](@ref) |
| critical momentum ``p_\star`` without and with trapping | `PSTAR_MODE_COLLISIONAL`, `COLLISIONLESS` | ✅ [`avalanche_critical_momentum`](@ref), the two modes coincide in a cylinder |
| linearized test-particle operator | `SlowingDownTerm`, `PitchScatterTerm`, `EnergyDiffusionTerm` | ✅ [`fokker_planck_operator`](@ref) |

### Electric field and current

| Feature | DREAM | Runes2P |
|---|---|---|
| prescribed field | `ElectricField.TYPE_PRESCRIBED` | ✅ the field of a [`PlasmaState`](@ref) |
| self-consistent field, poloidal flux, plasma current | `TYPE_SELFCONSISTENT`, `psi_p`, `I_p` | ✅ [`induction_equations`](@ref), [`induction_unknowns`](@ref) |
| initial current profile | `OhmicCurrent.setInitialProfile` | ✅ [`initial_poloidal_flux`](@ref) |
| field from a prescribed ohmic current | `TYPE_PRESCRIBED_OHMIC_CURRENT` | ➖ |
| loop voltage prescribed at the wall | `BC_TYPE_PRESCRIBED` | ✅ `loop_voltage_at_wall` |
| resistive wall with a wall time | `BC_TYPE_SELFCONSISTENT`, `inverse_wall_time` | ➖ |
| transformer, initial wall current | `BC_TYPE_TRANSFORMER`, `setInitialWallCurrent` | ➖ |
| hyperresistivity | `HYPERRESISTIVITY_MODE_PRESCRIBED`, `ADAPTIVE`, `ADAPTIVE_LOCAL` | ➖ |
| Braams–Karney conductivity | `CONDUCTIVITY_MODE_BRAAMS` | ✅ [`braams_karney_conductivity`](@ref), [`ConductivityModel`](@ref) |
| Sauter conductivity | `CONDUCTIVITY_MODE_SAUTER_COLLISIONLESS`, `SAUTER_COLLISIONAL` | ✅ in a cylinder, where it is the Braams–Karney conductivity |
| corrected conductivity of the fully kinetic mode | `CORRECTED_CONDUCTIVITY_ENABLED` | ✅ `kinetic = :dream_fit` of [`ConductivityModel`](@ref) |
| current profile of a torus | `PROFILE_TYPE_J_PARALLEL`, `J_DOT_GRADPHI`, `JTOR_OVER_R`, `CORSICA` | ➖ |
| bootstrap current | `BOOTSTRAP_MODE_REDL` | ➖ |
| current of a distribution | unknowns `j_hot`, `j_re` | ✅ [`CurrentMomentTerm`](@ref), [`current_density`](@ref) |

### Temperature and energy

| Feature | DREAM | Runes2P |
|---|---|---|
| prescribed cold temperature | `ColdElectronTemperature.TYPE_PRESCRIBED` | ✅ [`PrescribedPlasma`](@ref) |
| self-consistent cold temperature | `TYPE_SELFCONSISTENT` | ✅ [`thermal_energy_equation`](@ref), [`PlasmaFromUnknowns`](@ref) |
| ohmic heating | `Tcold_ohmic` | ✅ [`ohmic_heating_term`](@ref) |
| collisional heating by ``f_\mathrm{hot}`` and ``f_\mathrm{re}`` | `Tcold_fhot_coll`, `Tcold_fre_coll` | ✅ DREAM's moment, [`FrictionEnergyTransferTerm`](@ref) (note) |
| collisional heating by fluid runaways | `CollisionalEnergyTransferREFluidTerm` | ✅ [`RunawayFluidEnergyTransferTerm`](@ref) |
| line radiation, ionization energy, Gould's bremsstrahlung | `RECOMBINATION_RADIATION_NEGLECTED` | ✅ `radiation = :dream` of [`RadiatedPowerTerm`](@ref), with `bremsstrahlung_corrections = :dream` |
| recombination radiation | `RECOMBINATION_RADIATION_INCLUDED` | ✅ `radiation = :recombination` of [`RadiatedPowerTerm`](@ref), with the energy of the recombined state at every radius |
| halo region losses | `HALO_REGION_LOSSES_INCLUDED` | ➖ |
| neutral beam heating | `setNBI`, `NBI_PROFILE_*` | ➖ |
| ion temperatures, energy exchange | `IONS_T_I_INCLUDE` | ✅ [`IonEnergyExchangeTerm`](@ref) |

- **Collisional heating by ``f_\mathrm{hot}`` and ``f_\mathrm{re}``.**
  For the hot tail, Runes2P takes DREAM's moment.
  For the runaway grid, Runes2P's default is [`free_electron_energy_transfer_term`](@ref), the energy change of the electron-electron operator ([`KineticEnergyLossTerm`](@ref)).
  DREAM's moment is the option for the runaway grid ([Defaults that differ](#Defaults-that-differ)).
  The moment is compared with DREAM's saved heating of both grids (`test_collisional_heating_dream.jl`).
  It is also compared in a run of the runaway grid with the cold temperature self-consistent against DREAM's run (`test_runaway_heating_dream.jl`).

### Ions and ionization

| Feature | DREAM | Runes2P |
|---|---|---|
| prescribed charge states | `IONS_PRESCRIBED`, `_NEUTRAL`, `_FULLY_IONIZED` | ✅ the ions of a [`PlasmaState`](@ref) |
| dynamic charge states | `IONS_DYNAMIC` | ✅ [`IonRateTerm`](@ref), [`ion_densities_unknown`](@ref) |
| dynamic, neutral or fully ionized at the start | `IONS_DYNAMIC_NEUTRAL`, `_FULLY_IONIZED` | ✅ [`deposit_neutrals!`](@ref), [`set_ion_densities!`](@ref) |
| start in coronal equilibrium | `init_equil` | ✅ [`fill_coronal_equilibrium!`](@ref), [`coronal_equilibrium`](@ref) (note) |
| charge states held in coronal equilibrium | `IONS_EQUILIBRIUM` | ➖ |
| ADAS rate coefficients | `Atomics`, ADF11 | ✅ [`read_adas`](@ref), [`adas_coefficient`](@ref) |
| ADAS interpolation | `ADAS_INTERP_BICUBIC`, `ADAS_INTERP_BILINEAR` | 🚧 the bicubic interpolation, DREAM's default |
| fluid ionization | `IONIZATION_MODE_FLUID` | ✅ [`IonRateTerm`](@ref) |
| kinetic ionization by ``f`` | `IONIZATION_MODE_KINETIC` | ✅ [`KineticIonizationTerm`](@ref) |
| kinetic ionization with an approximate Jacobian | `IONIZATION_MODE_KINETIC_APPROX_JAC` | ➖ the Jacobian is exact |
| ionization by fluid runaways | `IONIZATION_MODE_FLUID_RE`, `reioniz_scale` | ➖ |
| DREAM's kinetic cross sections | `IonKineticIonizationTerm`, Burgess–Chidichimo fits | ✅ `form = :dream` of [`ionization_cross_section`](@ref) |
| opacity to line radiation | `ION_OPACITY_MODE_GROUND_STATE_OPAQUE` | ➖ |
| prescribed ion source, volumetric source | `ION_SOURCE_PRESCRIBED`, `PRESCRIBED_VOLUMETRIC` | ➖ |
| shattered pellet injection | `SPI` | ➖ |
| isotopes | `isotope` of `IonSpecies` | 🚧 H, D and T as elements of an [`IonSpecies`](@ref), [`ion_mass`](@ref) |

- **Start in coronal equilibrium.**
  [`fill_coronal_equilibrium!`](@ref) is DREAM's fixed point of the free-electron density over the species started in equilibrium.
  It takes the charge states of [`coronal_equilibrium`](@ref).
  It is compared with DREAM's `examples/ionequil` (`test_initial_state_dream.jl`).

### Runaway generation

| Feature | DREAM | Runes2P |
|---|---|---|
| fluid runaway density | unknown `n_re` | 🚧 a radial unknown, fed by [`ExternalAvalancheTerm`](@ref), [`MomentumBoundaryFluxTerm`](@ref) or [`InterfaceParticleFluxTerm`](@ref) (note) |
| Connor–Hastie Dreicer rate | `DREICER_RATE_CONNOR_HASTIE`, `_NOCORR` | 🚧 the rate as a function ([`connor_hastie_rate`](@ref)), no source term |
| Dreicer rate of the neural network | `DREICER_RATE_NEURAL_NETWORK` | ➖ |
| extrapolation of the Dreicer rate | `setExtrapolateDreicer` | ➖ |
| fluid avalanche | `AVALANCHE_MODE_FLUID` | 🚧 the rates as functions ([`fluid_avalanche_growth_rate`](@ref), [`avalanche_growth_rate`](@ref)), no term in a run without a kinetic grid (note) |
| fluid avalanche with DREAM's alternative ``p_\star`` | `AVALANCHE_MODE_FLUID_HESSLOW`, `pStarFunctionAlt` | ➖ |
| kinetic Rosenbluth–Putvinski source, external avalanche | `AVALANCHE_MODE_KINETIC`, `pCutAvalanche` | ✅ [`RosenbluthPutvinskiSourceTerm`](@ref), [`ExternalAvalancheTerm`](@ref) |
| Boltzmann knock-on operator | `AVALANCHE_MODE_BOLTZMANN_KNOCK_ON` (`boltzmann-eqterm`) | ✅ [`KnockOnOperator`](@ref), [`KnockOnTerm`](@ref) |
| effective critical field | `COLLQTY_ECEFF_MODE_FULL`, `SIMPLE`, `CYLINDRICAL`, `EC_TOT` | 🚧 mode `FULL` in a cylinder, [`effective_critical_field`](@ref) |
| Compton source | `COMPTON_MODE_FLUID`, `KINETIC`, `COMPTON_RATE_ITER_DMS` | ➖ |
| tritium source | `TRITIUM_MODE_FLUID`, `KINETIC` | ➖ |
| hot-tail rate | `HOTTAIL_MODE_ANALYTIC`, `ANALYTIC_ALT_PC` | ➖ |
| losses through the last closed flux surface | `LCFS_LOSS_MODE_FLUID`, `KINETIC`, `setLCFSLossTime`, `setPsiEdgeFromOutput` | 🚧 a prescribed loss rate of ``f`` ([`LossRateTerm`](@ref)), not the edge flux from an output file (`setPsiEdgeFromOutput`) |
| negative runaway density | `setNegativeRunaways` | ➖ |

- **Fluid runaway density.**
  Above a hot-tail grid without a runaway grid, it is a radial unknown fed by [`ExternalAvalancheTerm`](@ref) and the flux through ``p_\mathrm{max}`` ([`MomentumBoundaryFluxTerm`](@ref)).
  With a runaway grid, it is fed by the particle flux through the interface ([`InterfaceParticleFluxTerm`](@ref)).
- **Fluid avalanche.**
  [`avalanche_growth_rate`](@ref) is the Rosenbluth–Putvinski rate.
  The fluid rate enters ``n_\mathrm{re}`` only above a hot-tail grid, through [`ExternalAvalancheTerm`](@ref).

### Transport

| Feature | DREAM | Runes2P |
|---|---|---|
| Rechester–Rosenbluth transport of ``f`` | `TRANSPORT_RECHESTER_ROSENBLUTH` | ✅ [`add_rechester_rosenbluth_diffusion!`](@ref), [`RechesterRosenbluthUpdate`](@ref) |
| prescribed transport coefficients of ``f`` | `TRANSPORT_PRESCRIBED` | 🚧 `advection_r` and `diffusion_r_r` of [`AdvectionDiffusion`](@ref), constant in time (note) |
| Svensson transport | `TRANSPORT_SVENSSON` | ➖ |
| MHD-like transport | `TRANSPORT_MHD_LIKE`, `MHD_LIKE_LOCAL` | ➖ |
| frozen-current transport | `TRANSPORT_FROZEN_CURRENT` | ➖ |
| interpolation of the coefficients in time | `INTERP1D_*`, `INTERP3D_*` | 🚧 the coefficients at the start of each step, from a [`PiecewiseConstant`](@ref) or a function of time |
| boundary of the transport of ``f`` | `BC_CONSERVATIVE`, `BC_F_0` | ✅ `r_boundary = :zero_flux`, `:absorbing_dream` |
| boundary with a constant gradient | `BC_DF_CONST` | 🚧 for the heat transport alone, `edge = :constant_gradient_dream` of [`HeatTransportTerm`](@ref) |
| Kiramov boundary | `BC_KIRAMOV` | ➖ |
| heat transport of the cold electrons | `ColdElectronTemperature.transport` | ✅ [`HeatTransportTerm`](@ref) |
| transport of the fluid runaways | `RunawayElectrons.transport` | ➖ |
| diffusion of the charge states | `ION_CHARGED_DIFFUSION_MODE_PRESCRIBED`, `ION_NEUTRAL_DIFFUSION_MODE_PRESCRIBED` | ✅ [`IonDiffusionTerm`](@ref), `states = :charged` or `:all` |
| advection of the charge states | `ION_CHARGED_ADVECTION_MODE_PRESCRIBED`, `ION_NEUTRAL_ADVECTION_MODE_PRESCRIBED` | ➖ |

- **Prescribed transport coefficients of ``f``.**
  The caller sets them on the faces of the grid.
  There are no coefficients given in time, radius and momentum and interpolated to the grid.

### Kinetic options

| Feature | DREAM | Runes2P |
|---|---|---|
| interpolation of the advection | `AD_INTERP_CENTRED`, `UPWIND`, `UPWIND_2ND_ORDER`, `QUICK`, `SMART`, `MUSCL`, `OSPRE`, `TCDF` | ✅ `scheme` of [`AdvectionDiffusion`](@ref) |
| downwind interpolation | `AD_INTERP_DOWNWIND` | ➖ |
| a scheme per direction | `ad_int_r`, `ad_int_p1`, `ad_int_p2` | ✅ `r_scheme`, `p_scheme`, `xi_scheme` |
| Jacobian of a flux limiter | `AD_INTERP_JACOBIAN_LINEAR`, `FULL`, `UPWIND` | ✅ `jacobian` of [`FokkerPlanckTerm`](@ref) |
| boundary at ``p_\mathrm{max}`` | `BC_F_0`, `BC_PHI_CONST`, `BC_DPHI_CONST` | ✅ `p_upper_boundary` of [`AdvectionDiffusion`](@ref) |
| boundary at ``p = 0`` | `AD_BC_MIRRORED` with the internal boundary condition of ``f_\mathrm{hot}``, `AD_BC_DIRICHLET` on ``f_\mathrm{re}`` | ✅ `p_zero_boundary` of [`AdvectionDiffusion`](@ref), `:outflow` and `:zero_flux` |
| threshold of the hot region | `HOT_REGION_P_MODE_THERMAL`, `THERMAL_SMOOTH`, `MC` | ✅ [`ThermalMomentum`](@ref), [`MomentumRegion`](@ref) |
| particle source | `PARTICLE_SOURCE_ZERO`, `IMPLICIT`, `EXPLICIT` | 🚧 zero and implicit, [`ParticleSourceTerm`](@ref) with a [`Constraint`](@ref) (note) |
| shape of the particle source | `PARTICLE_SOURCE_SHAPE_MAXWELLIAN`, `DELTA` | ✅ `shape` of [`ParticleSourceTerm`](@ref) |
| synchrotron loss | `SYNCHROTRON_MODE_INCLUDE` | ✅ [`add_synchrotron!`](@ref) |
| ion columns in the Jacobian of ``f`` | `enableIonJacobian` | ✅ `lagged` of [`FokkerPlanckTerm`](@ref) |
| magnetic ripple | `RIPPLE_MODE_BOX`, `GAUSSIAN` | ➖ |
| time-varying magnetic field | `TIME_VARYING_B_MODE_INCLUDE` | ➖ |
| analytic or prescribed distribution | `DISTRIBUTION_MODE_ANALYTICAL`, `PRESCRIBED` | ➖ |
| non-relativistic hot-tail equation | `F_HOT_DIST_MODE_NONREL` | ➖ |
| initial distribution | `setInitialProfiles`, `setInitialValue` | 🚧 a state array of the caller or DREAM's Maxwellian ([`fill_maxwellian!`](@ref)), no interpolation from another grid (note) |
| initial runaway distributions | `INIT_FORWARD`, `XI_NEGATIVE`, `XI_POSITIVE`, `ISOTROPIC`, `AVALANCHE`, `PRESCRIBED` | 🚧 four of DREAM's shapes ([`fill_runaway_distribution!`](@ref)) or a state array of the caller, not the analytic avalanche shape (note) |

- **Particle source.**
  The explicit source has no packaged form.
  The disruption case builds it from an unknown of the source with its own row (`test/example_disruption.jl`).
- **Initial distribution.**
  A run takes any state array of the caller on the grid of the run.
  DREAM's Maxwellian is built from profiles of ``n`` and ``T``.
- **Initial runaway distributions.**
  [`fill_runaway_distribution!`](@ref) gives the four shapes in the first momentum cell.
  They are compared with DREAM's (`test_initial_state_dream.jl`).

### Solver and time stepping

| Feature | DREAM | Runes2P |
|---|---|---|
| linearly implicit solver | `Solver.LINEAR_IMPLICIT` | ➖ |
| Newton solver | `Solver.NONLINEAR`, `setMaxIterations` | ✅ [`NewtonSolver`](@ref) |
| tolerances per unknown | `ToleranceSettings` | ✅ [`Tolerance`](@ref) |
| residual check | `setCheckResidual` | ✅ as a convergence criterion, with two exceptions (note) |
| damped Newton step | `MaximalPhysicalStepLength` | ✅ [`LowerLimit`](@ref), `max_reduction` |
| direct linear solvers | `LINEAR_SOLVER_LU`, `MUMPS`, `MKL`, `SUPERLU` | 🚧 UMFPACK ([`SparseLU`](@ref)) and MKL PARDISO ([`MKLPardiso`](@ref) with `using Pardiso`) |
| GMRES | `LINEAR_SOLVER_GMRES` | ✅ [`GMRES`](@ref) |
| backup solver | `setBackupSolver` | ➖ |
| scaling of the unknowns | `Preconditioner` | ✅ [`unknown_scales`](@ref) |
| constant time step | `TimeStepper.TYPE_CONSTANT`, `setNt`, `setDt` | ✅ [`FixedSteps`](@ref) |
| saved steps | `setNumberOfSaveSteps` | ✅ `save_times` of [`RunOptions`](@ref) |
| adaptive time step by step doubling | `TYPE_ADAPTIVE`, `checkevery` | ✅ [`StepDoubling`](@ref) |
| smallest time between saved steps | `setMinSaveTimestep` | ➖ the saved times are prescribed |
| ionization time stepper | `TYPE_IONIZATION` | ➖ |
| termination function | `setTerminationFunction` | ➖ a segment stops only on a wall-time budget or a stop file ([`StopCondition`](@ref)) |
| restart from an output file | `fromOutput` | ✅ [`read_state`](@ref), [`read_checkpoint`](@ref) |
| solver statistics | `setSaveStatistics`, `setSaveConvergenceInfo` | ✅ [`SolverStatistics`](@ref) |
| verbose output | `setVerbose` | ✅ [`ProgressLog`](@ref) |

- **Residual check.**
  The residual is a convergence criterion except for the linear steps that `linear_acceptance` of [`NewtonSolver`](@ref) accepts on the update.
  The steps at the round-off floor of the residual are the other exception.

### Output

| Feature | DREAM | Runes2P |
|---|---|---|
| HDF5 output of the unknowns | `Output.setFilename` | ✅ [`write_output`](@ref), [`read_output`](@ref) |
| settings in the output | `setSaveSettings` | ✅ [`run_settings`](@ref) |
| timing | `setTiming` | ✅ the wall times of [`SolverStatistics`](@ref) |
| other quantities | `OtherQuantities` | 🚧 computed from a saved state, such as [`radiated_power`](@ref) and [`interface_flux`](@ref) |
| DREAM's settings files | `DREAMSettings` | ➖ |
| Python interface and plotting | `py/DREAM` | ➖ the Julia API |

## The disruption case

The page [The disruption case](disruption-case.md) compares the two codes in whole runs of DREAM's disruption case of `examples/Disruptions/generate.py`.
It covers the reduced size of the tests and the whole quenches of scenarios 1 and 4 at full size.

## What Runes2P has that DREAM does not

- Exponential fitting, the default scheme of [`fokker_planck_operator`](@ref) ([Positivity and flux limiters](model/positivity.md#Exponential-fitting)).
- The flux limiter van Albada, `scheme = :van_albada` of [`AdvectionDiffusion`](@ref).
- A smooth momentum grid, [`tanh_p_faces`](@ref).
- A pitch grid per radius, with the same cell counts and momentum faces at all radii, and conservative radial fluxes between different pitch grids ([`PitchOverlap`](@ref), [`radial_face_overlap`](@ref)).
- A conservative flux between the hot-tail and the runaway grid by overlap cells, `pitch_mapping = :overlap` of [`MomentumInterfaceTerm`](@ref).
- The momentum-conserving field-particle term of the electron-electron collisions, [`FieldParticleTerm`](@ref).
- The conductivity of the discrete kinetic operator, [`kinetic_conductivity`](@ref), as the correction of the ohmic current (`kinetic = :discrete_operator` of [`ConductivityModel`](@ref)).
- The total energy of the electrons in the fully kinetic mode, [`total_energy_equation`](@ref), which closes the energy of the cold electrons.
- A smooth window of partial screening above the excitation energy, `screening_window` of [`CollisionSettings`](@ref).
- The friction of the ionization on ``f``, `bound_friction = :ionization_split` of [`CollisionSettings`](@ref).
- The heating of the cold electrons by the secondaries of the kinetic ionization, [`SecondaryHeatingTerm`](@ref).
- The split of the ionization at the threshold of the kinetic ionization, with the fluid coefficient of the cold electrons below it ([`fluid_ionization_coefficient`](@ref), `kinetic_ionization` of [`IonRateTerm`](@ref)).
- CHIANTI's cross sections of the kinetic ionization continued by the relativistic Bethe form, and DREAM's form with the factor ``1/β^2`` (`form = :chianti`, `:bethe` of [`ionization_cross_section`](@ref)).
- The knock-on pitch integrated over each pair of momentum cells and averaged over the primary pitch cell ([`MollerPitchKernel`](@ref)).
- An implicit knock-on term, `treatment = :implicit` of [`KnockOnTerm`](@ref).
- A knock-on cutoff per radius that follows the plasma during a run, [`KnockOnCutoffUpdate`](@ref).
- The extrapolated flux at ``p_\mathrm{max}`` by DREAM's stated definition, `p_upper_boundary = :extrapolated_flux_consistent`.
- Absorbing edges with ``f = 0`` or ``T = 0`` on the edge face: `r_boundary = :absorbing` of [`AdvectionDiffusion`](@ref), `edge = :absorbing` of [`HeatTransportTerm`](@ref).
- A safety factor and a major radius in the Rechester–Rosenbluth coefficient ([`add_rechester_rosenbluth_diffusion!`](@ref)).
- Lower limits per unknown, relative to a reference unknown or to the density of a species ([`LowerLimit`](@ref)), and the temperature floor of [`PlasmaFromUnknowns`](@ref).
- Tolerances of the charge states relative to the density of their species, `group` of [`Tolerance`](@ref).
- A step controller with an error estimate from a predictor, and variable-step BDF2 ([`StepController`](@ref)).
- A backtracking line search of the Newton iteration, `line_search` of [`NewtonSolver`](@ref).
- Preconditioners of [`GMRES`](@ref) for systems coupled through the fluid unknowns: [`FluidSchurLU`](@ref), [`RadiusBlockLU`](@ref), [`RadiusBlockILU`](@ref).
- Exact Jacobians by automatic differentiation, the columns of the plasma unknowns included ([`jacobian!`](@ref), [`jacobian_correction_product!`](@ref)).
- A schedule of legs with prescribed steps or a controller ([`Leg`](@ref), [`time_points`](@ref)).
- Checkpoints and segments of a run ([`CheckpointSettings`](@ref), [`read_output`](@ref)).
  With [`SparseLU`](@ref) a resumed run repeats the uninterrupted one bit for bit.
- The system as an ODE function of SciML ([`ode_function`](@ref)).

## Departures from DREAM

The sections below list the defaults that differ from DREAM's and the behaviour without an option for DREAM's.
The last three give the details of the extrapolated flux at the highest momentum, the approximate Jacobians and the line search.

### Defaults that differ

Each row names the default of Runes2P and the option that gives DREAM's behaviour.
The reasons of the defaults that are choices of this package are on [Choices of this package](choices.md).

| Topic | Runes2P default | DREAM | Option for DREAM's behaviour |
|---|---|---|---|
| interpolation of ``f`` in ``p`` and ``\xi`` | exponential fitting in [`fokker_planck_operator`](@ref) | centred interpolation | `scheme = :central` |
| boundary at ``p_\mathrm{max}`` | `p_upper_boundary = :f_zero` in [`fokker_planck_operator`](@ref) | `BC_PHI_CONST` | `p_upper_boundary = :constant_flux` |
| synchrotron loss | on | off | `synchrotron = false` |
| bremsstrahlung stopping power | on | off | `bremsstrahlung = :neglect` |
| bremsstrahlung in the energy diffusion | a pure drag | the whole ``ν_s`` diffused | `bremsstrahlung_diffusion = :dream` |
| constant of the stopping power | ``4/3`` of the formula 4BN of Koch and Motz | ``1`` | `bremsstrahlung_constant = :dream` |
| ion sizes of partial screening | Table 1 of Hesslow et al. (note) | a lookup that shifts the sizes from Ar¹⁷⁺ on | `ion_size_table = :dream` |
| momenta of partial screening | above the excitation energy of each charge state | all momenta | `screening_window = :dream` |
| knock-on pitch | integrated over the cell pair, primary averaged over its cell | both at the cell centres | `dream_compatible = true` of [`KnockOnOperator`](@ref) |
| current of a distribution | ``f`` linear across a pitch cell | ``f`` constant over a pitch cell | `pitch_rule = :cell_center` |
| flux between the two grids | by the overlap cells of the pitch grids | interpolated in pitch | `pitch_mapping = :dream` |
| flux through ``p = 0`` on a runaway grid from ``p = 0`` | `p_zero_boundary = :outflow` of [`fokker_planck_operator`](@ref) (note) | none | `p_zero_boundary = :zero_flux`, set by the caller |
| Rechester–Rosenbluth coefficient ``D = \pi q R_0 v_\parallel (\delta B/B)^2`` | safety factor and major radius as inputs | ``q = 1`` and ``R_0 = 1`` m in a cylinder | `safety_factor = 1`, `major_radius = 1` |
| heating of the cold electrons by a runaway grid | [`free_electron_energy_transfer_term`](@ref) (note) | the moment ``m_e c\, v p\, \nu_s f`` at the cell centres | [`FrictionEnergyTransferTerm`](@ref) in place of the default |
| radiated power of the ions | line and recombination radiation, the binding energy of ionization and recombination | line radiation, the ionization loss and Gould's bremsstrahlung | `radiation = :dream` of [`RadiatedPowerTerm`](@ref), with `bremsstrahlung_corrections = :dream` for DREAM's corrections |
| relativistic corrections of Gould's bremsstrahlung, the coefficients of ``T/m_ec^2`` | ``11/24`` and ``5(44 - 3\pi^2)/(24\sqrt{2})`` ([`GOULD_ION_CORRECTION`](@ref), [`GOULD_ELECTRON_CORRECTION`](@ref)) (note) | ``19/24`` and ``5(44 - 3\pi^2)/(8\sqrt{2})`` | `bremsstrahlung_corrections = :dream` |
| cross sections of the kinetic ionization | CHIANTI's for H, D, T, Ne and Ar, DREAM's for the other elements | DREAM's fits | `form = :dream` |
| ionization at the hot threshold | the fluid coefficient below the threshold, `kinetic_ionization` of [`IonRateTerm`](@ref) | the whole ADAS rate next to the kinetic term | [`IonRateTerm`](@ref) without `kinetic_ionization` |
| Coulomb logarithm among the ions | the ion-ion logarithm of the NRL Plasma Formulary, with the temperature of each species | one logarithm with ``T_e`` | `ion_coulomb_logarithm = :dream` of [`IonEnergyExchangeTerm`](@ref) |
| ion masses | nuclear masses of H, D and T, standard atomic weights otherwise | standard atomic weights up to ``Z = 40``, ``2.3\,Z`` u beyond | `mass_table = :dream` |
| charge states in the collisions of ``f`` | those of the start of the step, `lagged = :ions` | those of the iterate with their Jacobian columns | `lagged = ()` |
| Newton matrix of a linear system | assembled and factorized once per step | every iteration | `reuse_linear_matrix = false` of [`NewtonSolver`](@ref) |
| negative entries of a kinetic unknown at convergence | one more Newton iteration for the schemes of [`POLISHED_SCHEMES`](@ref) | none | `polish_negatives = false` of [`NewtonSolver`](@ref) |
| iterative refinement of MKL PARDISO | none, `refinement_steps = 0` of [`MKLPardiso`](@ref) | up to 10 steps per solve (`fvm/Solvers/MIMKL.cpp`) | `MKLPardiso(refinement_steps = 10)` |
| floor of a temperature unknown | [`DEFAULT_TEMPERATURE_FLOOR`](@ref) ([`fluid_temperature_floors`](@ref)) | damped towards zero | a lower floor of [`PlasmaFromUnknowns`](@ref) or a [`LowerLimit`](@ref) |

The sources of the values of Runes2P in the table are these.

| Row | Source |
|---|---|
| constant of the stopping power | the formula 4BN of H. W. Koch and J. W. Motz, Rev. Mod. Phys. 31, 920 (1959), [doi:10.1103/RevModPhys.31.920](https://doi.org/10.1103/RevModPhys.31.920) |
| ion sizes of partial screening | Table 1 of L. Hesslow et al., J. Plasma Phys. 84, 905840605 (2018), [doi:10.1017/S0022377818001113](https://doi.org/10.1017/S0022377818001113) |
| relativistic corrections of Gould's bremsstrahlung, electron-electron | the ratio of eqs. (22) and (18) of R. J. Gould, Astrophys. J. 238, 1026 (1980), [doi:10.1086/158068](https://doi.org/10.1086/158068) |
| relativistic corrections of Gould's bremsstrahlung, electron-ion | the erratum of the paper of 1980, Astrophys. J. 243, 677 (1981), [doi:10.1086/158634](https://doi.org/10.1086/158634) |
| Coulomb logarithm among the ions | the NRL Plasma Formulary, A. S. Richardson, NRL/PU/6770--19-652, Naval Research Laboratory, 2019 |

**Ion sizes of partial screening.**
DREAM's lookup has a list of nuclear charges with 17 argon entries for 18 charge states.
This shifts the sizes from Ar¹⁷⁺ on.

**Flux through ``p = 0`` on a runaway grid from ``p = 0``.**
Runes2P takes the operator's default, `p_zero_boundary = :outflow` of [`fokker_planck_operator`](@ref), as DREAM's hot-tail grid.
The friction carries electrons out by ``\lim_{p \to 0} p^3 ν_s`` ([`p3_slowing_down_frequency_at_zero`](@ref)).
With it the runaway grid of the case `heating` of `reference/dream/cases/initial_state.py` loses 1.55e-4 of the density of ``f_\mathrm{re}`` in the first step, where DREAM's stays constant to round-off.

The caller sets the option `p_zero_boundary = :zero_flux` for a runaway grid from ``p = 0``, as both `compat` values of `runaway_heating_example` do (`test_runaway_heating_dream.jl`).
In DREAM's source:

- ``f_\mathrm{re}`` has no internal boundary condition (`src/Settings/Equations/f_re.cpp:87`).
- ``f_\mathrm{re}`` keeps the Dirichlet lower boundary of its advection in ``p`` (`fvm/Equation/AdvectionTerm.cpp:36`).
- Unlike ``f_\mathrm{re}``, ``f_\mathrm{hot}`` has an internal boundary condition (`src/Settings/Equations/f_hot.cpp:96`).
  With it, the lower boundary of its advection in ``p`` is mirrored (`AD_BC_MIRRORED`) in place of Dirichlet (`src/Settings/Equations/distribution.cpp:234-237`).

**Heating of the cold electrons by a runaway grid.**
Runes2P's default is the energy change of the electron-electron operator of the grid, with a linear scheme of the operator.
DREAM's moment, [`FrictionEnergyTransferTerm`](@ref), also counts the energy that the friction carries out of the grid through ``p = 0`` where that flux is applied (`p_zero_boundary = :outflow`).
The test case `runaway_heating_example` (`test/example_cases.jl`) takes the option at `compat = dream`.
It is compared with DREAM's run in `test_runaway_heating_dream.jl`.

**Relativistic corrections of Gould's bremsstrahlung.**
The coefficients of ``T/m_ec^2`` are these.

| Coefficient | Runes2P | DREAM |
|---|---|---|
| electron-ion | ``11/24`` ([`GOULD_ION_CORRECTION`](@ref)) | ``19/24`` |
| electron-electron | ``5(44 - 3\pi^2)/(24\sqrt{2})`` ([`GOULD_ELECTRON_CORRECTION`](@ref)) | ``5(44 - 3\pi^2)/(8\sqrt{2})`` |

### Behaviour without an option for DREAM's

In these points Runes2P differs from DREAM, and no option gives DREAM's behaviour.

- Newton's method converges on both the update and the scaled residual, except for the linear steps that `linear_acceptance` accepts on the update ([`NewtonSolver`](@ref)).
  A stiff step at the round-off floor of its residual is accepted on the update too, within the round-off bound of its rows or where the residual stopped decreasing.
  DREAM converges on the update.
  With `setCheckResidual` on, its default, DREAM warns when the residual is not converged (`src/Solver/SolverNonLinear.cpp:305-310`, `py/DREAM/Settings/Solver.py:28`).
- A failed Newton step is reported as a status of [`NewtonResult`](@ref).
  The time stepper retries it with a shorter step.
- Each charge state has a lower limit relative to the density of its species ([`ion_densities_unknown`](@ref)).
  In DREAM the charge states have no limit of their own (`src/Solver/SolverNonLinear.cpp:441-508`).
  DREAM's `MaximalPhysicalStepLength` damps these quantities (`src/Solver/SolverNonLinear.cpp:441-508`):
  - ``T_\mathrm{cold}``, ``n_\mathrm{tot}`` and ``n_\mathrm{cold}``.
  - ``W_\mathrm{cold}``, ``W_i`` and the species densities ``N_i``, where they are unknowns.
  - ``\sum Z_0^2 n`` per radius.
- A lower limit shortens only steps towards the limit, never growth ([`LowerLimit`](@ref)).
- A doubled step can fail on an unknown at zero with no absolute tolerance while every other unknown has no error.
  Such a step is retried with half its length ([`StepDoubling`](@ref)).
  DREAM's next step is unbounded there (`src/TimeStepper/TimeStepperAdaptive.cpp:458-482`).
- The ion rows take the sign convention of every other row, so [`IonDiffusionTerm`](@ref) takes a positive coefficient (DREAM: `src/Equations/Fluid/IonTransientTerm.cpp:78`).
- [`coronal_equilibrium`](@ref) multiplies the ratios of the rates in log space, where DREAM multiplies them directly (`src/Settings/IonEquilibrium.cpp:100-137`).
- The derivatives of the ADAS coefficients are exact, without DREAM's floors in the Jacobian ([`adas_coefficient`](@ref), [`IonRateTerm`](@ref)).
- The Jacobian of [`ExternalAvalancheTerm`](@ref) holds every column of the rate, where DREAM's holds those of the field and of ``n_\mathrm{tot}``.
- The temperature column of [`ParticleSourceTerm`](@ref) holds the derivative of the normalization of its shape, which DREAM's column omits (`src/Equations/Kinetic/ParticleSourceTerm.cpp`).
- The recombination gain of `radiation = :recombination` takes the ionization energy of the recombined state at every radius.
  DREAM's takes that of the recombining state from its second radial cell on (`src/Equations/Fluid/RadiatedPowerTerm.cpp`).
  `test/test_adas_cache.jl` checks the radiated power in coronal equilibrium, which detects this mismatch.
- [`RechesterRosenbluthUpdate`](@ref) takes the level of the perturbation at the start of each step.
  DREAM interpolates its table linearly in time and takes the value at the end of the step.
- [`StepDoubling`](@ref) reads the tolerances by unknown, for the acceptance and for the next step.
  DREAM's step update reads them by position in its list of non-trivial unknowns (`GetErrorScale`, `src/ConvergenceChecker.cpp`).
- `check_every` of [`StepDoubling`](@ref) takes the steps between two checked steps at the current length, the documented intent of DREAM's `checkevery`.
  DREAM's code takes them from a stale start time, with other lengths and one step more (`src/TimeStepper/TimeStepperAdaptive.cpp`).
- The `:smooth` envelope of a [`MomentumRegion`](@ref) whose limit lies at or above the last cell centre takes the width of the last cell.
  DREAM takes the limit minus the last centre (`FindThresholdStep`, `fvm/Equation/MomentQuantity.cpp`).
  That gives zero width and a NaN at that centre, and a jump of the envelope as the limit crosses it.

### Extrapolated flux at the highest momentum

DREAM's `BC_DPHI_CONST` is defined by a constant ``\mathrm{d}Φ/\mathrm{d}p`` at ``p_\mathrm{max}``, with ``Φ = \mathcal{V}' Γ^p``, the phase-space Jacobian ``\mathcal{V}'`` times the momentum flux ``Γ^p``.
Its code (`fvm/Equation/BoundaryConditions/PXiExternalLoss.cpp`) shares the ``\mathcal{V}'`` of the last interior face between the two terms of the extrapolation.
It multiplies both terms by the advection coefficient ``A^p`` of that face.
The interpolation and the diffusion of the second term are those of the face before it.
`p_upper_boundary = :extrapolated_flux` of [`AdvectionDiffusion`](@ref) ports this code literally, and the comparisons with DREAM use it.
`:extrapolated_flux_consistent` extrapolates with the ``\mathcal{V}'`` and the ``Γ^p`` of each face, as the definition states.
`test/test_pmax_boundary.jl` checks both.
`test/test_pmax_boundary_dream.jl` compares the literal port with DREAM.

### Approximate Jacobians

The residual of a [`FokkerPlanckTerm`](@ref) is always that of its scheme.
`jacobian` selects the stored block alone:

- `:exact` differentiates the limiter.
- `:frozen_limiter` is DREAM's `AD_INTERP_JACOBIAN_LINEAR`.
- A linear scheme name gives that scheme's Jacobian ([`with_scheme`](@ref)).

`jacobian = :upwind` with a central residual is DREAM's `AD_INTERP_JACOBIAN_UPWIND`, which makes some of DREAM's runs at high momentum resolution converge (DREAM issue [#126](https://github.com/chalmersplasmatheory/DREAM/issues/126)).
A term with an approximate block declares [`jacobian_kind`](@ref)`(term) === :approximate`.
Newton's method on the stored matrix is then a quasi-Newton iteration.
With [`GMRES`](@ref)`(operator = :exact)` the stored block only preconditions the exact Newton operator ([`jacobian_correction_product!`](@ref)).
`test/test_jacobian_strategies.jl` checks the modes.

### Line search

DREAM's nonlinear solver (`src/Solver/SolverNonLinear.cpp`) damps the Newton step.
It has no line search.
With `line_search = true`, [`NewtonSolver`](@ref) backtracks on ``\tfrac{1}{2}\lVert F \rVert^2`` of the scaled residual ``F`` with the Armijo condition.
It starts from the step length that the lower limits allow.
The default is off, DREAM's behaviour.
`test/test_time_stepping.jl` checks it.
