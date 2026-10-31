# Comparison with DREAM

```@meta
CurrentModule = Runes2P
```

This page compares Runes2P with DREAM, the code whose design it follows (M. Hoppe, O. Embréus and T. Fülöp, Comput. Phys. Commun. 268, 108098 (2021), [doi:10.1016/j.cpc.2021.108098](https://doi.org/10.1016/j.cpc.2021.108098)).
The comparison is with DREAM release v26.5 and its branch `boltzmann-eqterm` at commit `93d27e1`, the builds of the [Reference data](reference-data.md).
The file and line citations of DREAM's source refer to commit `93d27e1` unless they name another version.

## Features of DREAM

| ✅ implemented | 🚧 partial | ➖ not implemented | features |
|---|---|---|---|
| 73 | 19 | 47 | 139 |

The tables follow the groups of DREAM's settings (`py/DREAM/Settings`).
The column "DREAM" names the setting, class or unknown, the column "Runes2P" its state and the name that implements it.

| State | Meaning |
|---|---|
| ✅ implemented | DREAM's feature is in the code, possibly next to other options |
| 🚧 partial | a part of the feature is in the code; the entry names the part |
| ➖ not implemented | the code has no counterpart |

Runes2P is cylindrical: a straight magnetic field, constant on each flux surface, so no trapping and no flux-surface averages ([`cylindrical_grid`](@ref)).
Every toroidal feature of DREAM is therefore not implemented.

### Run modes

Runes2P has no switch of the run mode.
A run is an [`EquationSystem`](@ref) of the terms the caller chooses, so a mode is a choice of unknowns and terms.

| Mode | DREAM | Runes2P |
|---|---|---|
| fully kinetic | hot-tail grid with `COLLFREQ_MODE_FULL`, the bulk inside `f_hot` | ✅ implemented: `mode = :full` of [`CollisionSettings`](@ref), [`total_energy_equation`](@ref) |
| superthermal | hot-tail grid with `COLLFREQ_MODE_SUPERTHERMAL`, a fluid cold population | 🚧 partial: the terms, `mode = :superthermal` of [`CollisionSettings`](@ref), [`MomentumRegion`](@ref), [`ConductivityTerm`](@ref); no self-consistent superthermal case is packaged or tested |
| isotropic | one pitch cell, the pitch-averaged equation | ➖ not implemented: a grid of one pitch cell is refused by [`fokker_planck_operator`](@ref) |
| fluid | no kinetic grid | 🚧 partial: the field, the cold electrons and the ions; without a kinetic grid no Dreicer or avalanche term feeds ``n_\mathrm{re}`` |

### Grids

| Feature | DREAM | Runes2P |
|---|---|---|
| cylindrical radial grid, custom faces | `RadialGrid.TYPE_CYLINDRICAL`, `setCustomGridPoints` | ✅ implemented: [`RadialGrid`](@ref), [`uniform_r_faces`](@ref) |
| minor radius | `setMinorRadius` | ✅ implemented: the last face of a [`RadialGrid`](@ref) |
| major radius | `setMajorRadius` | ➖ not implemented: a cylinder has none, as in DREAM's cylindrical grid; `major_radius` is only an input of the Rechester–Rosenbluth coefficient ([`add_rechester_rosenbluth_diffusion!`](@ref), [`RechesterRosenbluthUpdate`](@ref), [`HeatTransportTerm`](@ref)) |
| inner radius | `setInnerRadius` | 🚧 partial: a [`RadialGrid`](@ref) may start above the axis, with no flux through its first face; no comparison with DREAM's inner boundary |
| magnetic field on the axis | `setB0` | ✅ implemented: the magnetic field of a [`PlasmaState`](@ref), one value per radius |
| wall radius | `setWallRadius` | ✅ implemented: `wall_radius` of [`induction_equations`](@ref) |
| analytic toroidal equilibrium, shaping | `TYPE_ANALYTIC_TOROIDAL`, `setShaping`, `setShapeParameter` | ➖ not implemented |
| numerical equilibrium | `TYPE_NUMERICAL`, `FILE_FORMAT_LUKE`, `setNumerical` | ➖ not implemented |
| poloidal-angle resolution of the averages | `setNtheta`, `setNthetaOut` | ➖ not implemented |
| momentum grid in ``(p, \xi)`` | `MomentumGrid.TYPE_PXI` | ✅ implemented: [`MomentumGrid`](@ref) |
| momentum grid in ``(p_\parallel, p_\perp)`` | `TYPE_PPARPPERP` | ➖ not implemented |
| uniform momentum grid | `PGrid.TYPE_UNIFORM` | ✅ implemented: [`uniform_p_faces`](@ref) |
| biuniform momentum grid | `PGrid.TYPE_BIUNIFORM` | ✅ implemented: [`biuniform_p_faces`](@ref) |
| custom momentum faces, ``p_\mathrm{min}`` | `PGrid.TYPE_CUSTOM`, `setPmin` | ✅ implemented: [`MomentumGrid`](@ref) from its faces |
| uniform pitch grid | `XiGrid.TYPE_UNIFORM` | ✅ implemented: [`uniform_xi_faces`](@ref) |
| biuniform in the pitch angle | `XiGrid.TYPE_BIUNIFORM_THETA` | ✅ implemented: [`biuniform_theta_xi_faces`](@ref) |
| biuniform in ``\xi``, uniform in the pitch angle | `XiGrid.TYPE_BIUNIFORM`, `TYPE_UNIFORM_THETA` | ✅ implemented: as custom faces of a [`MomentumGrid`](@ref) |
| custom pitch faces | `XiGrid.TYPE_CUSTOM` | ✅ implemented: [`MomentumGrid`](@ref) from its faces |
| trapped-passing boundary layer | `XiGrid.TYPE_TRAPPED`, `setTrappedPassingBoundaryLayerGrid` | ➖ not implemented |
| hot-tail and runaway grids, the flux between them | `hottailgrid`, `runawaygrid` | ✅ implemented: [`runaway_grid`](@ref), [`MomentumInterfaceTerm`](@ref) |

### Collisions

| Feature | DREAM | Runes2P |
|---|---|---|
| collision frequency mode | `COLLFREQ_MODE_SUPERTHERMAL`, `FULL`, `ULTRA_RELATIVISTIC` | ✅ implemented: `mode` of [`CollisionSettings`](@ref) |
| screening | `COLLFREQ_TYPE_COMPLETELY_SCREENED`, `NON_SCREENED`, `PARTIALLY_SCREENED` | ✅ implemented: `screening` |
| Walkowiak screening | `COLLFREQ_TYPE_WALKOWIAK` | ➖ not implemented |
| Coulomb logarithm | `LNLAMBDA_CONSTANT`, `ENERGY_DEPENDENT`, `THERMAL` | ✅ implemented: [`coulomb_logarithm`](@ref) (`:relativistic` is `CONSTANT`) |
| energy diffusion with screening | `SCREENED_DIFFUSION_MODE_ZERO`, `MAXWELLIAN` | ✅ implemented: `screened_diffusion` |
| bremsstrahlung stopping power | `BREMSSTRAHLUNG_MODE_NEGLECT`, `STOPPING_POWER` | ✅ implemented: `bremsstrahlung`, [`bremsstrahlung_cross_section`](@ref) |
| critical momentum ``p_\star`` without and with trapping | `PSTAR_MODE_COLLISIONAL`, `COLLISIONLESS` | ✅ implemented: [`avalanche_critical_momentum`](@ref); the two modes coincide in a cylinder |
| linearized test-particle operator | `SlowingDownTerm`, `PitchScatterTerm`, `EnergyDiffusionTerm` | ✅ implemented: [`fokker_planck_operator`](@ref) |

### Electric field and current

| Feature | DREAM | Runes2P |
|---|---|---|
| prescribed field | `ElectricField.TYPE_PRESCRIBED` | ✅ implemented: the field of a [`PlasmaState`](@ref) |
| self-consistent field, poloidal flux, plasma current | `TYPE_SELFCONSISTENT`, `psi_p`, `I_p` | ✅ implemented: [`induction_equations`](@ref), [`induction_unknowns`](@ref) |
| initial current profile | `OhmicCurrent.setInitialProfile` | ✅ implemented: [`initial_poloidal_flux`](@ref) |
| field from a prescribed ohmic current | `TYPE_PRESCRIBED_OHMIC_CURRENT` | ➖ not implemented |
| loop voltage prescribed at the wall | `BC_TYPE_PRESCRIBED` | ✅ implemented: `loop_voltage_at_wall` |
| resistive wall with a wall time | `BC_TYPE_SELFCONSISTENT`, `inverse_wall_time` | ➖ not implemented |
| transformer, initial wall current | `BC_TYPE_TRANSFORMER`, `setInitialWallCurrent` | ➖ not implemented |
| hyperresistivity | `HYPERRESISTIVITY_MODE_PRESCRIBED`, `ADAPTIVE`, `ADAPTIVE_LOCAL` | ➖ not implemented |
| Braams–Karney conductivity | `CONDUCTIVITY_MODE_BRAAMS` | ✅ implemented: [`braams_karney_conductivity`](@ref), [`ConductivityModel`](@ref) |
| Sauter conductivity | `CONDUCTIVITY_MODE_SAUTER_COLLISIONLESS`, `SAUTER_COLLISIONAL` | ✅ implemented in a cylinder, where it is the Braams–Karney conductivity |
| corrected conductivity of the fully kinetic mode | `CORRECTED_CONDUCTIVITY_ENABLED` | ✅ implemented: `kinetic = :dream_fit` of [`ConductivityModel`](@ref) |
| current profile of a torus | `PROFILE_TYPE_J_PARALLEL`, `J_DOT_GRADPHI`, `JTOR_OVER_R`, `CORSICA` | ➖ not implemented |
| bootstrap current | `BOOTSTRAP_MODE_REDL` | ➖ not implemented |
| current of a distribution | unknowns `j_hot`, `j_re` | ✅ implemented: [`CurrentMomentTerm`](@ref), [`current_density`](@ref) |

### Temperature and energy

| Feature | DREAM | Runes2P |
|---|---|---|
| prescribed cold temperature | `ColdElectronTemperature.TYPE_PRESCRIBED` | ✅ implemented: [`PrescribedPlasma`](@ref) |
| self-consistent cold temperature | `TYPE_SELFCONSISTENT` | ✅ implemented: [`thermal_energy_equation`](@ref), [`PlasmaFromUnknowns`](@ref) |
| ohmic heating | `Tcold_ohmic` | ✅ implemented: [`ohmic_heating_term`](@ref) |
| collisional heating by ``f_\mathrm{hot}`` and ``f_\mathrm{re}`` | `Tcold_fhot_coll`, `Tcold_fre_coll` | ✅ implemented: [`free_electron_energy_transfer_term`](@ref), [`KineticEnergyLossTerm`](@ref) |
| collisional heating by fluid runaways | `CollisionalEnergyTransferREFluidTerm` | ✅ implemented: [`RunawayFluidEnergyTransferTerm`](@ref) |
| line radiation, ionization energy, Gould's bremsstrahlung | `RECOMBINATION_RADIATION_NEGLECTED` | ✅ implemented: `radiation = :dream` of [`RadiatedPowerTerm`](@ref), with `bremsstrahlung_corrections = :dream` |
| recombination radiation | `RECOMBINATION_RADIATION_INCLUDED` | ✅ implemented: `radiation = :recombination` of [`RadiatedPowerTerm`](@ref), with the energy of the recombined state at every radius |
| halo region losses | `HALO_REGION_LOSSES_INCLUDED` | ➖ not implemented |
| neutral beam heating | `setNBI`, `NBI_PROFILE_*` | ➖ not implemented |
| ion temperatures, energy exchange | `IONS_T_I_INCLUDE` | ✅ implemented: [`IonEnergyExchangeTerm`](@ref) |

### Ions and ionization

| Feature | DREAM | Runes2P |
|---|---|---|
| prescribed charge states | `IONS_PRESCRIBED`, `_NEUTRAL`, `_FULLY_IONIZED` | ✅ implemented: the ions of a [`PlasmaState`](@ref) |
| dynamic charge states | `IONS_DYNAMIC` | ✅ implemented: [`IonRateTerm`](@ref), [`ion_densities_unknown`](@ref) |
| dynamic, neutral or fully ionized at the start | `IONS_DYNAMIC_NEUTRAL`, `_FULLY_IONIZED` | ✅ implemented: [`deposit_neutrals!`](@ref), [`set_ion_densities!`](@ref) |
| start in coronal equilibrium | `init_equil` | 🚧 partial: the charge states of [`coronal_equilibrium`](@ref); no run that starts from them is packaged or tested |
| charge states held in coronal equilibrium | `IONS_EQUILIBRIUM` | ➖ not implemented |
| ADAS rate coefficients | `Atomics`, ADF11 | ✅ implemented: [`read_adas`](@ref), [`adas_coefficient`](@ref) |
| ADAS interpolation | `ADAS_INTERP_BICUBIC`, `ADAS_INTERP_BILINEAR` | 🚧 partial: the bicubic interpolation, DREAM's default |
| fluid ionization | `IONIZATION_MODE_FLUID` | ✅ implemented: [`IonRateTerm`](@ref) |
| kinetic ionization by ``f`` | `IONIZATION_MODE_KINETIC` | ✅ implemented: [`KineticIonizationTerm`](@ref) |
| kinetic ionization with an approximate Jacobian | `IONIZATION_MODE_KINETIC_APPROX_JAC` | ➖ not implemented: the Jacobian is exact |
| ionization by fluid runaways | `IONIZATION_MODE_FLUID_RE`, `reioniz_scale` | ➖ not implemented |
| DREAM's kinetic cross sections | `IonKineticIonizationTerm`, Burgess–Chidichimo fits | ✅ implemented: `form = :dream` of [`ionization_cross_section`](@ref) |
| opacity to line radiation | `ION_OPACITY_MODE_GROUND_STATE_OPAQUE` | ➖ not implemented |
| prescribed ion source, volumetric source | `ION_SOURCE_PRESCRIBED`, `PRESCRIBED_VOLUMETRIC` | ➖ not implemented |
| shattered pellet injection | `SPI` | ➖ not implemented |
| isotopes | `isotope` of `IonSpecies` | 🚧 partial: H, D and T as elements of an [`IonSpecies`](@ref), [`ion_mass`](@ref) |

### Runaway generation

| Feature | DREAM | Runes2P |
|---|---|---|
| fluid runaway density | unknown `n_re` | 🚧 partial: a radial unknown above a hot-tail grid without a runaway grid, fed by [`ExternalAvalancheTerm`](@ref) and the flux through ``p_\mathrm{max}`` ([`MomentumBoundaryFluxTerm`](@ref)) |
| Connor–Hastie Dreicer rate | `DREICER_RATE_CONNOR_HASTIE`, `_NOCORR` | 🚧 partial: [`connor_hastie_rate`](@ref), no source term |
| Dreicer rate of the neural network | `DREICER_RATE_NEURAL_NETWORK` | ➖ not implemented |
| extrapolation of the Dreicer rate | `setExtrapolateDreicer` | ➖ not implemented |
| fluid avalanche | `AVALANCHE_MODE_FLUID` | 🚧 partial: [`fluid_avalanche_growth_rate`](@ref) and the Rosenbluth–Putvinski rate [`avalanche_growth_rate`](@ref) as functions; the fluid rate enters ``n_\mathrm{re}`` only above a hot-tail grid, through [`ExternalAvalancheTerm`](@ref); no term in a run without a kinetic grid |
| fluid avalanche with DREAM's alternative ``p_\star`` | `AVALANCHE_MODE_FLUID_HESSLOW`, `pStarFunctionAlt` | ➖ not implemented |
| kinetic Rosenbluth–Putvinski source, external avalanche | `AVALANCHE_MODE_KINETIC`, `pCutAvalanche` | ✅ implemented: [`RosenbluthPutvinskiSourceTerm`](@ref), [`ExternalAvalancheTerm`](@ref) |
| Boltzmann knock-on operator | `AVALANCHE_MODE_BOLTZMANN_KNOCK_ON` (`boltzmann-eqterm`) | ✅ implemented: [`KnockOnOperator`](@ref), [`KnockOnTerm`](@ref) |
| effective critical field | `COLLQTY_ECEFF_MODE_FULL`, `SIMPLE`, `CYLINDRICAL`, `EC_TOT` | 🚧 partial: mode `FULL` in a cylinder, [`effective_critical_field`](@ref) |
| Compton source | `COMPTON_MODE_FLUID`, `KINETIC`, `COMPTON_RATE_ITER_DMS` | ➖ not implemented |
| tritium source | `TRITIUM_MODE_FLUID`, `KINETIC` | ➖ not implemented |
| hot-tail rate | `HOTTAIL_MODE_ANALYTIC`, `ANALYTIC_ALT_PC` | ➖ not implemented |
| losses through the last closed flux surface | `LCFS_LOSS_MODE_FLUID`, `KINETIC`, `setLCFSLossTime`, `setPsiEdgeFromOutput` | 🚧 partial: a prescribed loss rate of ``f``, [`LossRateTerm`](@ref); the edge flux from an output file (`setPsiEdgeFromOutput`) is not implemented |
| negative runaway density | `setNegativeRunaways` | ➖ not implemented |

### Transport

| Feature | DREAM | Runes2P |
|---|---|---|
| Rechester–Rosenbluth transport of ``f`` | `TRANSPORT_RECHESTER_ROSENBLUTH` | ✅ implemented: [`add_rechester_rosenbluth_diffusion!`](@ref), [`RechesterRosenbluthUpdate`](@ref) |
| prescribed transport coefficients of ``f`` | `TRANSPORT_PRESCRIBED` | 🚧 partial: `advection_r` and `diffusion_r_r` of [`AdvectionDiffusion`](@ref), set by the caller on the faces of the grid and constant in time; no coefficients given in time, radius and momentum and interpolated to the grid |
| Svensson transport | `TRANSPORT_SVENSSON` | ➖ not implemented |
| MHD-like transport | `TRANSPORT_MHD_LIKE`, `MHD_LIKE_LOCAL` | ➖ not implemented |
| frozen-current transport | `TRANSPORT_FROZEN_CURRENT` | ➖ not implemented |
| interpolation of the coefficients in time | `INTERP1D_*`, `INTERP3D_*` | 🚧 partial: the level at the start of each step, [`PiecewiseConstant`](@ref) or a function of time |
| boundary of the transport of ``f`` | `BC_CONSERVATIVE`, `BC_F_0` | ✅ implemented: `r_boundary = :zero_flux`, `:absorbing_dream` |
| boundary with a constant gradient | `BC_DF_CONST` | 🚧 partial: for the heat transport alone, `edge = :constant_gradient_dream` of [`HeatTransportTerm`](@ref) |
| Kiramov boundary | `BC_KIRAMOV` | ➖ not implemented |
| heat transport of the cold electrons | `ColdElectronTemperature.transport` | ✅ implemented: [`HeatTransportTerm`](@ref) |
| transport of the fluid runaways | `RunawayElectrons.transport` | ➖ not implemented |
| diffusion of the charge states | `ION_CHARGED_DIFFUSION_MODE_PRESCRIBED`, `ION_NEUTRAL_DIFFUSION_MODE_PRESCRIBED` | ✅ implemented: [`IonDiffusionTerm`](@ref), `states = :charged` or `:all` |
| advection of the charge states | `ION_CHARGED_ADVECTION_MODE_PRESCRIBED`, `ION_NEUTRAL_ADVECTION_MODE_PRESCRIBED` | ➖ not implemented |

### Kinetic options

| Feature | DREAM | Runes2P |
|---|---|---|
| interpolation of the advection | `AD_INTERP_CENTRED`, `UPWIND`, `UPWIND_2ND_ORDER`, `QUICK`, `SMART`, `MUSCL`, `OSPRE`, `TCDF` | ✅ implemented: `scheme` of [`AdvectionDiffusion`](@ref) |
| downwind interpolation | `AD_INTERP_DOWNWIND` | ➖ not implemented |
| a scheme per direction | `ad_int_r`, `ad_int_p1`, `ad_int_p2` | ✅ implemented: `r_scheme`, `p_scheme`, `xi_scheme` |
| Jacobian of a flux limiter | `AD_INTERP_JACOBIAN_LINEAR`, `FULL`, `UPWIND` | ✅ implemented: `jacobian` of [`FokkerPlanckTerm`](@ref) |
| boundary at ``p_\mathrm{max}`` | `BC_F_0`, `BC_PHI_CONST`, `BC_DPHI_CONST` | ✅ implemented: `p_upper_boundary` of [`AdvectionDiffusion`](@ref) |
| threshold of the hot region | `HOT_REGION_P_MODE_THERMAL`, `THERMAL_SMOOTH`, `MC` | ✅ implemented: [`ThermalMomentum`](@ref), [`MomentumRegion`](@ref) |
| particle source | `PARTICLE_SOURCE_ZERO`, `IMPLICIT`, `EXPLICIT` | 🚧 partial: zero and implicit, [`ParticleSourceTerm`](@ref) with a [`Constraint`](@ref) |
| shape of the particle source | `PARTICLE_SOURCE_SHAPE_MAXWELLIAN`, `DELTA` | ✅ implemented: `shape` of [`ParticleSourceTerm`](@ref) |
| synchrotron loss | `SYNCHROTRON_MODE_INCLUDE` | ✅ implemented: [`add_synchrotron!`](@ref) |
| ion columns in the Jacobian of ``f`` | `enableIonJacobian` | ✅ implemented: `lagged` of [`FokkerPlanckTerm`](@ref) |
| magnetic ripple | `RIPPLE_MODE_BOX`, `GAUSSIAN` | ➖ not implemented |
| time-varying magnetic field | `TIME_VARYING_B_MODE_INCLUDE` | ➖ not implemented |
| analytic or prescribed distribution | `DISTRIBUTION_MODE_ANALYTICAL`, `PRESCRIBED` | ➖ not implemented |
| non-relativistic hot-tail equation | `F_HOT_DIST_MODE_NONREL` | ➖ not implemented |
| initial distribution | `setInitialProfiles`, `setInitialValue` | 🚧 partial: any state array of the caller on the grid of the run, or DREAM's Maxwellian from profiles of ``n`` and ``T`` ([`fill_maxwellian!`](@ref)); no interpolation from another grid |
| initial runaway distributions | `INIT_FORWARD`, `XI_NEGATIVE`, `XI_POSITIVE`, `ISOTROPIC`, `AVALANCHE`, `PRESCRIBED` | 🚧 partial: any state array of the caller, no packaged shapes |

### Solver and time stepping

| Feature | DREAM | Runes2P |
|---|---|---|
| linearly implicit solver | `Solver.LINEAR_IMPLICIT` | ➖ not implemented |
| Newton solver | `Solver.NONLINEAR`, `setMaxIterations` | ✅ implemented: [`NewtonSolver`](@ref) |
| tolerances per unknown | `ToleranceSettings` | ✅ implemented: [`Tolerance`](@ref) |
| residual check | `setCheckResidual` | ✅ implemented as a convergence criterion, except for the linear steps that `linear_acceptance` of [`NewtonSolver`](@ref) accepts on the update |
| damped Newton step | `MaximalPhysicalStepLength` | ✅ implemented: [`LowerLimit`](@ref), `max_reduction` |
| direct linear solvers | `LINEAR_SOLVER_LU`, `MUMPS`, `MKL`, `SUPERLU` | 🚧 partial: UMFPACK, [`SparseLU`](@ref) |
| GMRES | `LINEAR_SOLVER_GMRES` | ✅ implemented: [`GMRES`](@ref) |
| backup solver | `setBackupSolver` | ➖ not implemented |
| scaling of the unknowns | `Preconditioner` | ✅ implemented: [`unknown_scales`](@ref) |
| constant time step | `TimeStepper.TYPE_CONSTANT`, `setNt`, `setDt` | ✅ implemented: [`FixedSteps`](@ref) |
| saved steps | `setNumberOfSaveSteps` | ✅ implemented: `save_times` of [`RunOptions`](@ref) |
| adaptive time step by step doubling | `TYPE_ADAPTIVE`, `checkevery` | ✅ implemented: [`StepDoubling`](@ref) |
| smallest time between saved steps | `setMinSaveTimestep` | ➖ not implemented: the saved times are prescribed |
| ionization time stepper | `TYPE_IONIZATION` | ➖ not implemented |
| termination function | `setTerminationFunction` | ➖ not implemented: a segment stops only on a wall-time budget or a stop file ([`StopCondition`](@ref)) |
| restart from an output file | `fromOutput` | ✅ implemented: [`read_state`](@ref), [`read_checkpoint`](@ref) |
| solver statistics | `setSaveStatistics`, `setSaveConvergenceInfo` | ✅ implemented: [`SolverStatistics`](@ref) |
| verbose output | `setVerbose` | ✅ implemented: [`ProgressLog`](@ref) |

### Output

| Feature | DREAM | Runes2P |
|---|---|---|
| HDF5 output of the unknowns | `Output.setFilename` | ✅ implemented: [`write_output`](@ref), [`read_output`](@ref) |
| settings in the output | `setSaveSettings` | ✅ implemented: [`run_settings`](@ref) |
| timing | `setTiming` | ✅ implemented: the wall times of [`SolverStatistics`](@ref) |
| other quantities | `OtherQuantities` | 🚧 partial: computed from a saved state, such as [`radiated_power`](@ref) and [`interface_flux`](@ref) |
| DREAM's settings files | `DREAMSettings` | ➖ not implemented |
| Python interface and plotting | `py/DREAM` | ➖ not implemented: the Julia API |

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
- Checkpoints and segments of a run; with a direct solver a resumed run repeats the uninterrupted one bit for bit ([`CheckpointSettings`](@ref), [`read_output`](@ref)).
- The system as an ODE function of SciML ([`ode_function`](@ref)).

## Departures from DREAM

### Defaults that differ

Each row names the default of Runes2P and the option that gives DREAM's behaviour.
The reasons of the defaults that are choices of this package are on [Choices of this package](choices.md).

| Topic | Runes2P default | DREAM's behaviour, the option for it |
|---|---|---|
| interpolation of ``f`` in ``p`` and ``\xi`` | exponential fitting in [`fokker_planck_operator`](@ref) | centred interpolation, `scheme = :central` |
| boundary at ``p_\mathrm{max}`` | `p_upper_boundary = :f_zero` in [`fokker_planck_operator`](@ref) | `BC_PHI_CONST`, `p_upper_boundary = :constant_flux` |
| synchrotron loss | on | off, `synchrotron = false` |
| bremsstrahlung stopping power | on | off, `bremsstrahlung = :neglect` |
| bremsstrahlung in the energy diffusion | a pure drag | the whole ``ν_s`` diffused, `bremsstrahlung_diffusion = :dream` |
| constant of the stopping power | ``4/3`` of the formula 4BN of H. W. Koch and J. W. Motz, Rev. Mod. Phys. 31, 920 (1959), [doi:10.1103/RevModPhys.31.920](https://doi.org/10.1103/RevModPhys.31.920) | ``1``, `bremsstrahlung_constant = :dream` |
| ion sizes of partial screening | Table 1 of L. Hesslow et al., J. Plasma Phys. 84, 905840605 (2018), [doi:10.1017/S0022377818001113](https://doi.org/10.1017/S0022377818001113) | a lookup whose list of nuclear charges has 17 argon entries for 18 charge states, which shifts the sizes from Ar¹⁷⁺ on, `ion_size_table = :dream` |
| momenta of partial screening | above the excitation energy of each charge state | all momenta, `screening_window = :dream` |
| knock-on pitch | integrated over the cell pair, primary averaged over its cell | both at the cell centres, `dream_compatible = true` of [`KnockOnOperator`](@ref) |
| current of a distribution | ``f`` linear across a pitch cell | ``f`` constant over a pitch cell, `pitch_rule = :cell_center` |
| flux between the two grids | by the overlap cells of the pitch grids | interpolated in pitch, `pitch_mapping = :dream` |
| Rechester–Rosenbluth coefficient ``D = \pi q R_0 v_\parallel (\delta B/B)^2`` | safety factor and major radius as inputs | ``q = 1`` and ``R_0 = 1`` m in a cylinder, `safety_factor = 1`, `major_radius = 1` |
| radiated power of the ions | line and recombination radiation, the binding energy of ionization and recombination | line radiation, the ionization loss and Gould's bremsstrahlung, `radiation = :dream` of [`RadiatedPowerTerm`](@ref) (with `bremsstrahlung_corrections = :dream` for DREAM's corrections) |
| relativistic corrections of Gould's bremsstrahlung, the coefficients of ``T/m_ec^2`` | electron-ion ``11/24`` of the erratum ([`GOULD_ION_CORRECTION`](@ref)), electron-electron ``5(44 - 3\pi^2)/(24\sqrt{2})``, the ratio of eqs. (22) and (18) ([`GOULD_ELECTRON_CORRECTION`](@ref)), of R. J. Gould, Astrophys. J. 238, 1026 (1980), [doi:10.1086/158068](https://doi.org/10.1086/158068), erratum Astrophys. J. 243, 677 (1981), [doi:10.1086/158634](https://doi.org/10.1086/158634) | ``19/24`` and ``5(44 - 3\pi^2)/(8\sqrt{2})``, `bremsstrahlung_corrections = :dream` |
| cross sections of the kinetic ionization | CHIANTI's for H, D, T, Ne and Ar, DREAM's for the other elements | DREAM's fits, `form = :dream` |
| ionization at the hot threshold | the fluid coefficient below the threshold, `kinetic_ionization` of [`IonRateTerm`](@ref) | the whole ADAS rate next to the kinetic term, [`IonRateTerm`](@ref) without `kinetic_ionization` |
| Coulomb logarithm among the ions | the ion-ion logarithm of the NRL Plasma Formulary (A. S. Richardson, NRL/PU/6770--19-652, Naval Research Laboratory, 2019) with the temperature of each species | one logarithm with ``T_e``, `ion_coulomb_logarithm = :dream` of [`IonEnergyExchangeTerm`](@ref) |
| ion masses | nuclear masses of H, D and T, standard atomic weights otherwise | standard atomic weights up to ``Z = 40``, ``2.3\,Z`` u beyond, `mass_table = :dream` |
| charge states in the collisions of ``f`` | those of the start of the step, `lagged = :ions` | those of the iterate with their Jacobian columns, `lagged = ()` |
| Newton matrix of a linear system | assembled and factorized once per step | every iteration, `reuse_linear_matrix = false` of [`NewtonSolver`](@ref) |
| floor of a temperature unknown | [`DEFAULT_TEMPERATURE_FLOOR`](@ref) ([`fluid_temperature_floors`](@ref)) | damped towards zero, a lower floor of [`PlasmaFromUnknowns`](@ref) or a [`LowerLimit`](@ref) |

### Behaviour without an option for DREAM's

- Newton's method converges on both the update and the scaled residual, except for the linear steps that `linear_acceptance` accepts on the update ([`NewtonSolver`](@ref)).
  DREAM converges on the update; with `setCheckResidual` on, its default, it warns when the residual is not converged (`src/Solver/SolverNonLinear.cpp:305-310`, `py/DREAM/Settings/Solver.py:28`).
- A failed Newton step is reported as a status of [`NewtonResult`](@ref), and the time stepper retries it with a shorter step.
- Each charge state has a lower limit relative to the density of its species ([`ion_densities_unknown`](@ref)).
  DREAM's `MaximalPhysicalStepLength` damps ``T_\mathrm{cold}``, ``n_\mathrm{tot}``, ``n_\mathrm{cold}``, where they are unknowns ``W_\mathrm{cold}``, ``W_i`` and the species densities ``N_i``, and ``\sum Z_0^2 n`` per radius; the charge states have no limit of their own (`src/Solver/SolverNonLinear.cpp:441-508`).
- A lower limit shortens only steps towards the limit, never growth ([`LowerLimit`](@ref)).
- A doubled step that an unknown at zero with no absolute tolerance fails, while every other unknown has no error, is retried with half its length ([`StepDoubling`](@ref)).
  DREAM's next step is unbounded there (`src/TimeStepper/TimeStepperAdaptive.cpp:458-482`).
- The ion rows take the sign convention of every other row, so [`IonDiffusionTerm`](@ref) takes a positive coefficient (DREAM: `src/Equations/Fluid/IonTransientTerm.cpp:78`).
- [`coronal_equilibrium`](@ref) multiplies the ratios of the rates in log space, where DREAM multiplies them directly (`src/Settings/IonEquilibrium.cpp:100-137`).
- The derivatives of the ADAS coefficients are exact, without DREAM's floors in the Jacobian ([`adas_coefficient`](@ref), [`IonRateTerm`](@ref)).
- The Jacobian of [`ExternalAvalancheTerm`](@ref) holds every column of the rate, where DREAM's holds those of the field and of ``n_\mathrm{tot}``.
- The temperature column of [`ParticleSourceTerm`](@ref) holds the derivative of the normalization of its shape, which DREAM's column omits (`src/Equations/Kinetic/ParticleSourceTerm.cpp`).
- The recombination gain of `radiation = :recombination` takes the ionization energy of the recombined state at every radius.
  DREAM's takes that of the recombining state from its second radial cell on (`src/Equations/Fluid/RadiatedPowerTerm.cpp`).
  `test/test_adas_cache.jl` checks the radiated power in coronal equilibrium, which detects this mismatch.
- [`RechesterRosenbluthUpdate`](@ref) takes the level of the perturbation at the start of each step; DREAM interpolates its table linearly in time and takes the value at the end of the step.
- [`StepDoubling`](@ref) reads the tolerances by unknown, for the acceptance and for the next step; DREAM's step update reads them by position in its list of non-trivial unknowns (`GetErrorScale`, `src/ConvergenceChecker.cpp`).
- `check_every` of [`StepDoubling`](@ref) takes the steps between two checked steps at the current length, the documented intent of DREAM's `checkevery`.
  DREAM's code takes them from a stale start time, with other lengths and one step more (`src/TimeStepper/TimeStepperAdaptive.cpp`).

### Extrapolated flux at the highest momentum

DREAM's `BC_DPHI_CONST` is defined by a constant ``\mathrm{d}Φ/\mathrm{d}p`` at ``p_\mathrm{max}``, with ``Φ = \mathcal{V}' Γ^p``, the phase-space Jacobian ``\mathcal{V}'`` times the momentum flux ``Γ^p``.
Its code (`fvm/Equation/BoundaryConditions/PXiExternalLoss.cpp`) shares the ``\mathcal{V}'`` of the last interior face between the two terms of the extrapolation and multiplies both by the advection coefficient ``A^p`` of that face.
The interpolation and the diffusion of the second term are those of the face before it.
`p_upper_boundary = :extrapolated_flux` of [`AdvectionDiffusion`](@ref) ports this code literally, and the comparisons with DREAM use it.
`:extrapolated_flux_consistent` extrapolates with the ``\mathcal{V}'`` and the ``Γ^p`` of each face, as the definition states.
`test/test_pmax_boundary.jl` checks both, and `test/test_pmax_boundary_dream.jl` compares the literal port with DREAM.

### Approximate Jacobians

The residual of a [`FokkerPlanckTerm`](@ref) is always that of its scheme, and `jacobian` selects the stored block alone.
`:exact` differentiates the limiter, `:frozen_limiter` is DREAM's `AD_INTERP_JACOBIAN_LINEAR`, and a linear scheme name gives that scheme's Jacobian ([`with_scheme`](@ref)).
`jacobian = :upwind` with a central residual is DREAM's `AD_INTERP_JACOBIAN_UPWIND`, which makes some of DREAM's runs at high momentum resolution converge (DREAM issue [#126](https://github.com/chalmersplasmatheory/DREAM/issues/126)).
A term with an approximate block declares [`jacobian_kind`](@ref)`(term) === :approximate`.
Newton's method on the stored matrix is then a quasi-Newton iteration, and with [`GMRES`](@ref)`(operator = :exact)` the stored block only preconditions the exact Newton operator ([`jacobian_correction_product!`](@ref)).
`test/test_jacobian_strategies.jl` checks the modes.

### Line search

DREAM's nonlinear solver (`src/Solver/SolverNonLinear.cpp`) damps the Newton step and has no line search.
With `line_search = true`, [`NewtonSolver`](@ref) backtracks on ``\tfrac{1}{2}\lVert F \rVert^2`` of the scaled residual ``F`` with the Armijo condition.
It starts from the step length that the lower limits allow.
The default is off, DREAM's behaviour.
`test/test_time_stepping.jl` checks it.
