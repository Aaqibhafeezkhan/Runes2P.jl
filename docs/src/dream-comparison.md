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
| 75 | 17 | 47 | 139 |

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
| superthermal | hot-tail grid with `COLLFREQ_MODE_SUPERTHERMAL`, a fluid cold population | ✅ implemented: `mode = :superthermal` of [`CollisionSettings`](@ref), [`ConductivityTerm`](@ref) for Ohm's law of the cold electrons, [`FrictionEnergyTransferTerm`](@ref) for their heating by ``f_\mathrm{hot}``; the disruption case in this mode is compared with DREAM end to end (below) |
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
| collisional heating by ``f_\mathrm{hot}`` and ``f_\mathrm{re}`` | `Tcold_fhot_coll`, `Tcold_fre_coll` | ✅ implemented: [`FrictionEnergyTransferTerm`](@ref), DREAM's moment, for the hot tail; for the runaway grid Runes2P's default is [`free_electron_energy_transfer_term`](@ref), the energy change of the electron-electron operator ([`KineticEnergyLossTerm`](@ref)), and DREAM's moment is the option ([Defaults that differ](#Defaults-that-differ)); the moment is compared with DREAM's saved heating of both grids (`test_collisional_heating_dream.jl`), and in a run of the runaway grid with the cold temperature self-consistent against DREAM's run (`test_runaway_heating_dream.jl`) |
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
| start in coronal equilibrium | `init_equil` | ✅ implemented: [`fill_coronal_equilibrium!`](@ref), DREAM's fixed point of the free-electron density over the species started in equilibrium, with the charge states of [`coronal_equilibrium`](@ref); compared with DREAM's `examples/ionequil` (`test_initial_state_dream.jl`) |
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
| fluid runaway density | unknown `n_re` | 🚧 partial: a radial unknown above a hot-tail grid without a runaway grid, fed by [`ExternalAvalancheTerm`](@ref) and the flux through ``p_\mathrm{max}`` ([`MomentumBoundaryFluxTerm`](@ref)); with a runaway grid, fed by the particle flux through the interface ([`InterfaceParticleFluxTerm`](@ref)) |
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
| boundary at ``p = 0`` | `AD_BC_MIRRORED` with the internal boundary condition of ``f_\mathrm{hot}``, `AD_BC_DIRICHLET` on ``f_\mathrm{re}`` | ✅ implemented: `p_zero_boundary` of [`AdvectionDiffusion`](@ref), `:outflow` and `:zero_flux` |
| threshold of the hot region | `HOT_REGION_P_MODE_THERMAL`, `THERMAL_SMOOTH`, `MC` | ✅ implemented: [`ThermalMomentum`](@ref), [`MomentumRegion`](@ref) |
| particle source | `PARTICLE_SOURCE_ZERO`, `IMPLICIT`, `EXPLICIT` | 🚧 partial: zero and implicit, [`ParticleSourceTerm`](@ref) with a [`Constraint`](@ref); the explicit source has no packaged form, and the disruption case builds it from an unknown of the source with its own row (`test/example_disruption.jl`) |
| shape of the particle source | `PARTICLE_SOURCE_SHAPE_MAXWELLIAN`, `DELTA` | ✅ implemented: `shape` of [`ParticleSourceTerm`](@ref) |
| synchrotron loss | `SYNCHROTRON_MODE_INCLUDE` | ✅ implemented: [`add_synchrotron!`](@ref) |
| ion columns in the Jacobian of ``f`` | `enableIonJacobian` | ✅ implemented: `lagged` of [`FokkerPlanckTerm`](@ref) |
| magnetic ripple | `RIPPLE_MODE_BOX`, `GAUSSIAN` | ➖ not implemented |
| time-varying magnetic field | `TIME_VARYING_B_MODE_INCLUDE` | ➖ not implemented |
| analytic or prescribed distribution | `DISTRIBUTION_MODE_ANALYTICAL`, `PRESCRIBED` | ➖ not implemented |
| non-relativistic hot-tail equation | `F_HOT_DIST_MODE_NONREL` | ➖ not implemented |
| initial distribution | `setInitialProfiles`, `setInitialValue` | 🚧 partial: any state array of the caller on the grid of the run, or DREAM's Maxwellian from profiles of ``n`` and ``T`` ([`fill_maxwellian!`](@ref)); no interpolation from another grid |
| initial runaway distributions | `INIT_FORWARD`, `XI_NEGATIVE`, `XI_POSITIVE`, `ISOTROPIC`, `AVALANCHE`, `PRESCRIBED` | 🚧 partial: the four shapes in the first momentum cell ([`fill_runaway_distribution!`](@ref)), compared with DREAM's (`test_initial_state_dream.jl`), and any state array of the caller; not the analytic avalanche shape |

### Solver and time stepping

| Feature | DREAM | Runes2P |
|---|---|---|
| linearly implicit solver | `Solver.LINEAR_IMPLICIT` | ➖ not implemented |
| Newton solver | `Solver.NONLINEAR`, `setMaxIterations` | ✅ implemented: [`NewtonSolver`](@ref) |
| tolerances per unknown | `ToleranceSettings` | ✅ implemented: [`Tolerance`](@ref) |
| residual check | `setCheckResidual` | ✅ implemented as a convergence criterion, except for the linear steps that `linear_acceptance` of [`NewtonSolver`](@ref) accepts on the update and the steps at the round-off floor of the residual |
| damped Newton step | `MaximalPhysicalStepLength` | ✅ implemented: [`LowerLimit`](@ref), `max_reduction` |
| direct linear solvers | `LINEAR_SOLVER_LU`, `MUMPS`, `MKL`, `SUPERLU` | 🚧 partial: UMFPACK, [`SparseLU`](@ref); MKL PARDISO, [`MKLPardiso`](@ref) with `using Pardiso` |
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

## The disruption case

The reference case is DREAM's disruption of `examples/Disruptions/generate.py`, scenario 1, in a cylinder.
Its case script is `reference/dream/cases/disruption_kinetic.py` ([Reference data](reference-data.md)).
The tests run the case at a reduced size, fully kinetic and in DREAM's superthermal mode.
The whole quenches of scenarios 1 and 4 at full size are measurements that no test repeats.

### The runs of the case script

The case script runs DREAM's chain of `generate.py` as six runs, 0 to 5.
Run 5 needs the option `--window-start`.

| Run | What it does |
|---|---|
| 0 | one step: the conductivity of the initial plasma and the field of the prescribed current profile |
| 1 | 1 s in 3 steps from the Maxwellian with that field |
| 2 | run 1 again, with the field rescaled to the plasma current |
| 3 | the ionization phase (phase 1): the ionization of the injected D and Ar |
| 4 | the thermal and current quench (phase 2) |
| 5 | a window of phase 2: run 4 stops at a saved step, and run 5 takes the next steps from DREAM's state there |

Runs 0 to 2 are the initialization.

### Parts of the comparison

The builder of the case is `build` of `test/example_disruption.jl`.
Its keyword `compat = :dream` takes DREAM's settings of the case where the codes differ, and `compat = :runes2p` takes Runes2P's defaults ([Defaults that differ](#Defaults-that-differ)).
The synchrotron loss is off at both values.

| Part | DREAM runs | Runes2P starts from or builds | Test |
|---|---|---|---|
| initialization | runs 0 to 2 | the same runs (notes below) | `test_disruption_initialization_dream.jl` |
| ionization phase | run 3 | DREAM's state at the start of the phase | `test_disruption_dream.jl` |
| thermal and current quench | run 4 | DREAM's state at the start of the phase | `test_disruption_dream.jl` |
| the chain end to end | runs 0 to 4 | Runes2P's own initialization, then each phase from Runes2P's end state of the run before | `test_disruption_chain_dream.jl` |
| the chain end to end in the superthermal mode | runs 0 to 4 with the case script's `--mode superthermal` (DREAM's `generate.py --superthermal`) | the same with `mode = :superthermal`, and, separately, each phase from DREAM's state | `test_disruption_superthermal_dream.jl` |
| quench with a runaway grid | runs 1 to 4 with the runaway grid to step 170 of 429 of phase 2, then 10 steps (run 5) | DREAM's state at the start of run 5 | `test_disruption_runaway_dream.jl` |
| the same from `f_re` zero | not compared with DREAM | the start of phase 2, with `f_re` zero | `test_disruption_runaway.jl` |
| whole quench of scenario 1 at full size | runs 0 to 4, the runaway grid in runs 1 to 4 | DREAM's state at the start of phase 2, 429 steps in one run | none: a measurement |
| the same in scenario 4 | runs 0 to 4 of scenario 4, the runaway grid in runs 1 to 4 | DREAM's state at the start of phase 2, 1600 steps in one run | none: a measurement |

DREAM's run 4 without the runaway grid holds no `f_re`, so the test from `f_re` zero has nothing to compare with.

The reduced sizes of the tests, as momentum × pitch cells:

| Test | Radii | Hot-tail grid | Runaway grid | Steps |
|---|---|---|---|---|
| initialization | 2 | 30 × 4 | none | runs 1 and 2: 3 steps over 1 s each |
| phases, chain | 2 | 30 × 4 | none | phase 1: 10 of 4e-8 s, phase 2: 10 of 1e-7 s |
| superthermal mode | 2 | 20 × 4, uniform, without the fine cells below p = 0.05 | none | as in the two rows above |
| quench with a runaway grid | 2 | 30 × 4 | 32 × 8, from p = 0.8 to 20 | 10 of 6.53e-7 s, from step 170 of 429 of phase 2 |
| the same from `f_re` zero | 2 | 30 × 4 | 32 × 8 | 3 of 1e-7 s |

The momenta are in units of ``m_e c``.
The hot-tail grid ends at `p_max = 0.8`, where the runaway grid starts.

The initialization in Runes2P:

- Runes2P runs the initialization itself, as runs 0 to 2.
- The field and the cold temperature stay at their prescribed values in runs 1 and 2.
- Runes2P builds the state at the start of phase 1 from the end of run 2 and the ions of the settings.

### The runaway grid

The runaway grid is the case script's `--runaway-grid` ([Choices](choices.md#Reference-cases)).
`build` adds it with `runaway = true`.

- The flux of `f_hot` through the hot-tail `p_max` enters `f_re` and `n_re` ([`MomentumInterfaceTerm`](@ref), [`InterfaceParticleFluxTerm`](@ref)).
- The kinetic avalanche source acts on `f_re`, on the runaway grid.
- The fluid avalanche acts in `n_re`, which counts every electron above the hot-tail `p_max`.
  On the runaway grid the fluid avalanche equals the kinetic source of `f_re`, and in `n_re` it adds the tail beyond the runaway grid's `p_max`.
- `f_re` carries the runaway current, ionizes the ions and heats the cold electrons.
- In the window of run 5, the runaway current is 5.9 % of the plasma current at the start.

### Balances checked without DREAM

The tests of the phases, of the chains and of the runaway grid also check balances of the assembled system that do not need DREAM.
They check them at both `compat` values.
The test of the initialization checks none.
`field` is one value over the whole run, from the first state to the last.
Every other value is the worst over the steps.

| Balance | What is checked | Relative to | Bound |
|---|---|---|---|
| `species` | the density of each ion species stays fixed, since the case has no ion transport | the largest density | 1e-13 |
| `exchange` | the energy that the electrons give the ions is the energy the ions take | the largest electron side | 1e-13 |
| `diffusion` | the heat diffusion has no net flux, since no heat leaves at the edge | its absolute sum | 1e-13 |
| `closure` | the change of `W_cold` equals the sum of its energy channels | dW/dt | 1e-12 |
| `field` | the change of the poloidal field energy equals the work of the field on the currents | the change | 1e-8 |
| field energy | the magnetic energy falls over the run | | its change below 0 |
| `drift` | the density of `f_hot` less n_cold and n_hot stays fixed, fully kinetic mode only | n_cold | 1e-6 in phase 1, 1e-13 in phase 2 |
| `interface` | the electrons the interface takes from `f_hot` are those it gives to `f_re` and to `n_re` | that influx at the radius in the step | 2e-14 |
| `avalanche` | n_re − ∫f_re grows by the avalanche beyond the runaway grid and the outflow there | the larger of dn_re/dt and d∫f_re/dt of the step at that radius | 1e-11 in the window, 1e-9 from `f_re` zero |
| influx | electrons cross from `f_hot` into `f_re` at every step and radius | | the smallest influx above 0 |

- In the fully kinetic mode `f_hot` covers the cold bulk as well, and the particle source keeps the balance of the `drift` row.
- In phase 1 the drift follows the convergence of Newton's method on the particle source, 2.7e-8 measured, so its bound is larger there.
- The rows `interface`, `avalanche` and influx hold for the tests with the runaway grid.
- The test from `f_re` zero also checks that the steps are accepted.
  It checks that `build` without the runaway grid has no `f_re`, and that the runaway grid adds `f_re` alone to the unknowns.
  It checks that the start state takes `S_particle` from a DREAM output that holds it.

### Agreement at the size of the tests

This section gives the largest differences against DREAM in the tests, all at `compat = :dream`.
Each difference is relative: |Runes2P − DREAM| divided by a reference value of DREAM.
Each number is the largest such difference over the saved steps after the first.
`phase_differences` of `test/example_disruption.jl` takes the reference by the kind of quantity:

| Short name | Reference | Used for |
|---|---|---|
| over the radii | DREAM's largest value over the radii at the same step | the radial quantities, such as `E_field`, the currents, `n_hot` and `psi_p` |
| per radius and step | DREAM's largest value over the cells, charge states or species at the same radius and step | `f_hot`, `f_re`, `n_i`, `W_i` |
| `I_p` | DREAM's `I_p` at the same step | `I_p` |
| per radius over the run | DREAM's largest value at the same radius over the run | `n_re` and `j_re`, where a test compares them per radius |
| per unknown | DREAM's largest entry of the unknown in its first state of phase 1 | the hand-over from the initialization to phase 1 |

The initialization compares `n_re` and `j_re` over the radii only.
The chain also compares `n_re` and `j_re` at the first radius, per radius over the run.
The values are measured at Julia's default optimisation level, unless marked, and change slightly with it.
The functions `phase_differences`, `read_dream_state` and `run_window` belong to `test/example_disruption.jl`, not to the package's API.

#### The initialization

`test_disruption_initialization_dream.jl` compares runs 1 and 2, and the state that they hand to phase 1.

| Quantity | Largest relative difference | Reference |
|---|---|---|
| `j_hot` | 7.9e-10 | over the radii |
| `I_p` | 7.8e-10 | `I_p` |
| `E_field` | 7.7e-10 | over the radii |
| `j_ohm` | 7.5e-10 | over the radii |
| `j_tot` | 7.5e-10 | over the radii |
| `psi_p` | 4.9e-10 | over the radii |
| `f_hot` | 2.7e-11 | per radius and step |
| `n_hot` | 1.7e-11 | over the radii |
| every unknown, at the hand-over to phase 1 | 7.7e-10 | per unknown |

DREAM's Newton tolerance of these runs is 1e-7, so the differences lie far below it.

#### The phases

`test_disruption_dream.jl` starts each phase from DREAM's state.
`test_disruption_chain_dream.jl` runs the chain end to end from Runes2P's own initialization.

| Quantity | From DREAM's state | The chain end to end | Reference |
|---|---|---|---|
| `f_hot` | 4.4e-5 | 4.4e-5 | per radius and step |
| `E_field` | 4.1e-7 | 4.1e-7 | over the radii |
| `psi_p` | 6.8e-13 | 2.8e-10 | over the radii |
| `I_p` | 1.3e-12 | 1.3e-11 | `I_p` |
| `n_re` at the first radius | 5.5e-12, at `-O0` | 3.6e-11 | per radius over the run |

- In the chain, `psi_p`, `I_p` and `n_re` carry the differences of the initialization.
- The table gives `n_re` at the first radius only.
  At the second radius `n_re` is 2e-15 of the first.
- From DREAM's state the test bounds `n_re` per radius at the second radius too, where it differs by 1.5e-11 at `-O0`.
- At the start of phase 1, DREAM's `n_re` at the second radius is 6.8e-3 m⁻³.
  DREAM's output holds it with too few significant bits for a comparison.
  So the chain compares the growth of `n_re` over phase 2 there, which differs by 1.9e-3 of DREAM's largest growth.

#### The superthermal mode

`test_disruption_superthermal_dream.jl` runs the chain in the superthermal mode, and each phase from DREAM's state.
In the first four rows the value is the largest difference over all compared quantities, each in its own reference.
These rows take `n_re` and `j_re` of the chain over the radii, and the last row takes them per radius.

| Comparison | Largest in | Largest relative difference | Reference |
|---|---|---|---|
| initialization, runs 1 and 2 | `j_tot`, `j_ohm`, `j_hot` | 1.5e-9 | over the radii |
| hand-over to phase 1 | `n_cold` | 1.3e-8 | per unknown |
| phases of the chain | `n_i` | 3.5e-6 | per radius and step |
| phases from DREAM's state | `n_i` | 3.2e-6 | per radius and step |
| phases of the chain, `n_re` and `j_re` at the second radius | | 3.8e-4 | per radius over the run |

- `n_cold` and `W_cold` come next to `n_i`, at 2.9e-6 and 2.8e-6 in both kinds of run.
- The 3.8e-4 comes from the initialization.
  At the second radius `n_re` is 4e-7 of the first radius, fed by the far tail of f.

#### The window with the runaway grid

`test_disruption_runaway_dream.jl` starts from DREAM's state at the start of run 5 and takes its 10 steps.

| Quantity | Largest relative difference | Reference |
|---|---|---|
| `f_re` | 1.1e-9 | per radius and step |
| `n_re` | 1.1e-9 | per radius over the run |
| `j_re` | 1.2e-9 | per radius over the run |
| `n_i`, the largest of the other quantities | 2.2e-8 | per radius and step |

- Each value is the larger of those at `-O0` and at the default optimisation level.
- Every other quantity differs by less than 2.2e-8 in its own reference.
- At the start, DREAM's `j_re` equals the moment of its `f_re` on Runes2P's grid to 1.1e-15.
  This is a check of DREAM's state, not a difference between the codes.

### The whole quench of scenario 1

This measurement runs phase 2 of scenario 1 at the full size of DREAM's example, in 429 steps of 6.53e-7 s.

#### The run

The table gives the set-up of the measurement.
The page [Performance and accuracy](performance.md#The-disruption-case) gives the same case in more detail.

| | |
|---|---|
| Case | scenario 1 of DREAM's `examples/Disruptions/generate.py`, thermal and current quench |
| DREAM | runs 0 to 4 of the case script (the command line below) |
| Size | 15 radii, hot-tail grid 140 × 15, runaway grid 32 × 15 (`--re-nxi 15 --re-nxi-sep 10`) |
| Start | DREAM's state at the start of phase 2, not Runes2P's own chain |
| Steps | DREAM's 429 fixed steps of 6.53e-7 s |
| Settings of Runes2P | `runaway = true` and `compat = :dream` of `build` |
| Solver | Runes2P's default direct solver, UMFPACK ([`SparseLU`](@ref)), at Julia's default optimisation level, DREAM with `--linear-solver mkl` |

DREAM runs the case script with its default sizes, which are the full size of DREAM's example:

```sh
disruption_kinetic.py --nt-ioniz 500 --t-ioniz 2e-5 --nt-quench 429 --t-quench 3e-4 --linear-solver mkl --runaway-grid --re-nxi 15 --re-nxi-sep 10
```

`--t-quench` is the end time of phase 2, counted from the start of phase 1.
So phase 2 lasts 3e-4 s − 2e-5 s, and each of its 429 steps is 6.53e-7 s.

#### Newton iterations

The table counts the Newton iterations of each code over the whole quench.

| Code | Accepted steps | Newton iterations, total over the 429 steps |
|---|---|---|
| Runes2P | 429 of 429 | 1822 |
| DREAM | 429 fixed steps | 1935 |

#### Agreement with DREAM

Each difference is relative: |Runes2P − DREAM| divided by the reference of its row.
Each number is the largest such difference over the 429 steps of the quench.
1e5 m⁻³ is DREAM's absolute tolerance of `n_re`.

| Quantity | Largest relative difference | Reference |
|---|---|---|
| `n_re` | 4.4e-6 | DREAM's largest `n_re` at the same radius over the quench |
| `j_re` | 3.4e-6 | DREAM's largest `j_re` at the same radius over the quench |
| `j_hot` | 2.0e-6 | DREAM's largest value over the radii at the same step |
| `E_field` | 1.1e-6 | DREAM's largest value over the radii at the same step |
| `j_ohm` | 1.1e-6 | DREAM's largest value over the radii at the same step |
| `j_tot` | 2.2e-7 | DREAM's largest value over the radii at the same step |
| `f_hot` | 5.3e-8 | DREAM's largest `f_hot` at the same radius and step |
| `I_p` | 5.0e-9 | DREAM's `I_p` at the same step |
| `f_re`, where DREAM's `n_re` is above 1e5 m⁻³ | at most 1.3e-5 | DREAM's largest `f_re` at the same radius and step |
| every other quantity | 5e-8 | the reference of its kind (table of references above) |

Over all radii and steps, `phase_differences` gives a larger value for `f_re`.
The larger differences of `f_re` are at the two outermost radii:

| Radius | Steps | Largest relative difference of `f_re` |
|---|---|---|
| 0.45 m | 1 to 4 | 0.51 |
| 0.48 m | 15 to 25 | 4.6e-2 |

In these steps DREAM's `n_re` at the two radii is at most 6.1e-7 m⁻³.
These cells hold almost no runaways: DREAM's `n_re` there lies far below its absolute tolerance.

At the end of the quench the runaway current is 67.8 % of the plasma current in both codes.

#### How to repeat it

1. DREAM runs the case script with the command line of "The run" above.
2. `read_dream_state` reads DREAM's output of run 4.
3. `run_window` runs Runes2P over the saved times of a DREAM run, here all 429 steps of the quench.
4. `phase_differences` returns the largest difference of each quantity.

### The whole quench of scenario 4

Scenario 4 of `generate.py` has a quench of 4.99e-3 s in 1600 steps.
Its heat diffusivity is 1000 m²/s, against the 4000 m²/s of scenario 1.
The quench is a measurement at full size.

#### The run

| | |
|---|---|
| Case | scenario 4 of DREAM's `examples/Disruptions/generate.py`, thermal and current quench |
| DREAM | v26.5, runs 0 to 4 of the case script (the command line below) |
| Size | 15 radii, hot-tail grid 140 × 15, runaway grid 52 × 15 to `p_max = 200` (12 fine and 40 coarse momentum cells), 43803 unknowns of Runes2P |
| Start | DREAM's state at the start of the quench |
| Steps | DREAM's 1600 fixed steps of 3.11875e-6 s, to 4.99e-3 s |
| Settings of Runes2P | `runaway = true`, `compat = :dream` and `diffusivity = 1000.0` of the builder of the case |
| Solver | MKL PARDISO on one thread in both codes, Runes2P at `-O2` with one Julia thread |

DREAM runs the case script with its default sizes and these options:

```sh
disruption_kinetic.py --scenario 4 --linear-solver mkl --runaway-grid --re-pmax 200 --re-np-coarse 40 --re-nxi 15 --re-nxi-sep 10
```

The runaway grid is a choice of this package ([Choices](choices.md#Reference-cases)).
The runs of Runes2P with fixed steps started with the particle source at zero, in place of DREAM's `S_particle`.
`initial_state` of `test/example_disruption.jl` takes `S_particle` where DREAM's output holds it.
With fixed steps the first step solves the source.
With DREAM's `S_particle` at the start, the differences against DREAM in the first steps are the same to the digits of the tables below.

#### Newton iterations

| Code | Accepted steps | Newton iterations | Most in one step |
|---|---|---|---|
| Runes2P | 1600 of 1600 | 6356 | 6 |
| DREAM v26.5 | 1600 | 7899 | 14 |

#### Agreement with DREAM

Each difference is relative: |Runes2P − DREAM| divided by the reference of its row.
Each number is the largest such difference over the 1600 saved steps.

| Quantity | Largest relative difference | Reference |
|---|---|---|
| `I_p` | 6.67e-8 | DREAM's `I_p` at the same step |
| `j_hot` | 2.80e-6 | DREAM's largest value over the radii at the same step |
| `j_ohm` | 1.55e-6 | DREAM's largest value over the radii at the same step |
| `E_field` | 9.31e-7 | DREAM's largest value over the radii at the same step |
| `n_hot` | 4.77e-7 | DREAM's largest value over the radii at the same step |
| `T_cold` | 3.48e-7 | DREAM's largest value over the radii at the same step |
| `j_tot` | 2.95e-7 | DREAM's largest value over the radii at the same step |
| `n_cold` | 3.44e-8 | DREAM's largest value over the radii at the same step |
| `n_re`, radii 1 to 11 | 4.3e-7, at radius 4 | DREAM's largest `n_re` at the same radius over the quench |
| `j_re`, radii 1 to 11 | 4.4e-7, at radius 4 | DREAM's largest `j_re` at the same radius over the quench |
| `n_re` and `j_re`, radii 12 and 13 | 5.6e-7 | DREAM's largest value at the same radius over the quench |
| `n_re`, radii 14 and 15 | 2.2e-2 | DREAM's largest `n_re` at the same radius over the quench |
| `j_re`, radii 14 and 15 | 3.4e-2 | DREAM's largest `j_re` at the same radius over the quench |
| `f_re` | 1.91e-6 | DREAM's largest `f_re` at the same radius and step |
| `f_hot` | 1.12e-5 | DREAM's largest `f_hot` at the same radius and step |

- At radii 1 to 11 DREAM's `n_re` is above its absolute tolerance of 1e5 m⁻³.
- At radii 12 and 13 it is below that tolerance.
  Its largest values there are 4.9e3 and 8.8 m⁻³.
- At radii 14 and 15 it stays below 2e-3 m⁻³.
- The `f_re` row counts only the radii and steps where DREAM's largest `f_re` is above 1e5, its absolute tolerance in DREAM's runs of the case.
- At 4.99e-3 s `I_p` is 2.543120e5 A in both codes.
- At 4.99e-3 s the share of `n_re` above the runaway grid is 4.0e-5 to 4.1e-5.
  This holds at radii 1 to 11, and the two codes give it equal to four digits.

#### The difference of `f_hot`

- The largest difference, 1.12e-5, is in the first step, in the lowest momentum cell.
- It falls to 1.5e-8 by step 20.
- At the time of the first step, the runs of 3200 steps differ by 1.10e-5 in the same measure.
- The 2-norm of the difference is 3.8e-7 of the 2-norm of `f_hot`.
- Its cause is not established.

#### Negative cells

The cells are counted over the 1601 saved states, at the radii where DREAM's largest value is above its absolute tolerance.
A cell that is negative in 10 states counts 10 times.

| Distribution | Negative in Runes2P | Negative in DREAM | Negative in both | Most negative value, relative to the largest value at its radius |
|---|---|---|---|---|
| `f_re` | 77146 | 77762 | 77146 | −3.9e-6 in both codes |
| `f_hot` | 947945 | 1034261 | 947620 | −1.2e-16 in both codes |

- Every negative cell of `f_re` in Runes2P is negative in DREAM too.
- The most negative cell of `f_re` is in the first momentum cell of the runaway grid.
- The negative cells of `f_hot` are at round-off.
  The most negative one is the same cell in both codes.
- A cell of `f_hot` negative in one code only holds at most 3.3e-26 of DREAM's largest value at its radius and step.
- DREAM's start state holds 2411 negative cells, in `f_hot` and `f_re` together and at all radii.
- `polish_negatives = :auto` of [`NewtonSolver`](@ref) stops polishing after 1.56e-5 s.
  By then five polished solves in a row have left negative entries.

#### The time-step error of the reference

The agreement above is at equal steps, so it does not show the error of the step length.
Runes2P was run with the three step counts of the table and the runaway grid to `p_max = 200`.
The limit is the Richardson extrapolation of these three runs.

| Run | `I_p` at 4.99e-3 s (A) | (`I_p` − limit) / limit | `E_field` at the first radius at 4.00135625e-3 s (V/m) | (`E_field` − limit) / limit |
|---|---|---|---|---|
| 1600 steps (the reference) | 2.5431201e5 | +1.00e-2 | 1.1793628 | −7.2e-2 |
| 3200 steps | 2.5306491e5 | +5.1e-3 | 1.2218632 | −3.9e-2 |
| 6400 steps | 2.5243459e5 | +2.6e-3 | 1.2447340 | −2.1e-2 |
| Richardson limit | 2.5179047e5 | | 1.2713809 | |

- The observed order is 0.98 in `I_p` and 0.89 in the field.
- An extrapolation of order 1 gives 1.268 V/m for the field.

DREAM v26.5 was run with the same three step counts and `p_max = 200`.
The limits of the two codes agree:

| Quantity | Runes2P limit | DREAM limit | Observed order (both) | (DREAM − Runes2P) / Runes2P limit |
|---|---|---|---|---|
| `I_p` at 4.99e-3 s | 2.5179047e5 A | 2.5179052e5 A | 0.98 | +1.7e-7 |
| `E_field` at the first radius at 4.00135625e-3 s | 1.2713809 V/m | 1.2713786 V/m | 0.89 | −1.8e-6 |

The table of the runs above uses Runes2P's limit.

At 6400 steps and `p_max = 200`, DREAM's values at the times of that table are close to Runes2P's:

| Quantity | DREAM | (DREAM − Runes2P) / Runes2P |
|---|---|---|
| `I_p` at 4.99e-3 s | 2.5243461e5 A | +6.6e-8 |
| `E_field` at the first radius at 4.00135625e-3 s | 1.2447333 V/m | −5.5e-7 |

The agreement at the other step counts takes the references of the agreement table.

| Steps | `p_max` | Compared times | Radial quantities | `I_p` | `n_re` and `j_re`, radii 1 to 11 |
|---|---|---|---|---|---|
| 3200 | 200 in both codes | 1613 of DREAM's 3201 | 6.8e-7 | 4.0e-8 | 2.5e-7, at radius 4 |
| 6400 | 200 in both codes | 1639 of DREAM's 6401 | 1.1e-6, largest in `j_ohm` | 6.6e-8 | 3.9e-7, at radius 4 |
| 6400 | 60 in both codes | all 6400 | 2.4e-6 | 6.6e-8 | 3.9e-7, at radius 4 |

- The run of Runes2P with 3200 steps saved every second step and the steps around 4e-3 s.
  That gives the 1613 compared times.
- The run of Runes2P with 6400 steps and `p_max = 200` saved every fourth step and the steps around 4e-3 s.
  That gives the 1639 compared times.

#### The step controller against fixed steps

[`StepController`](@ref) starts from the same state as the fixed steps.
Its start takes the particle source from DREAM's `S_particle` through `initial_state`.
Its `relative_tolerance` is 1e-3 (the default) or 1e-4, and its other settings are the defaults.
The runs save only at 4.00135625e-3 s and 4.99e-3 s, where a step ends.
The Newton iterations count every attempt.

| Run | Accepted steps | Newton iterations | (`I_p` − limit) / limit | (`E_field` − limit) / limit | (`T_cold` − value at 6400 fixed steps) / that value |
|---|---|---|---|---|---|
| fixed, 1600 steps | 1600 | 6356 | +1.00e-2 | −7.2e-2 | +2.1e-5 |
| fixed, 6400 steps | 6400 | 20743 | +2.6e-3 | −2.1e-2 | 0 |
| `StepController`, `relative_tolerance = 1e-3` | 1107 | 4681 | +6.3e-3 | −2.8e-2 | +1.3e-3 |
| `StepController`, `relative_tolerance = 1e-4` | 3239 | 11898 | +2.2e-3 | −1.2e-2 | +4.6e-4 |

- `T_cold` is that of the first radius at 4.99e-3 s.
- At 1e-3 the controller is closer to the limit in `I_p` and the field than 1600 fixed steps, with fewer Newton iterations.
- At 1e-4 it is closer than 6400 fixed steps, with fewer Newton iterations.
- In `T_cold` it is further off than the fixed steps.
  The runs with fixed steps differ there by at most 2.1e-5 of the value at 6400 fixed steps.
- At 6400 fixed steps with `p_max = 200`, DREAM v26.5 takes 25809 Newton iterations, at most 14 in a step.
  Runes2P takes 20743, at most 5 in a step.

The start of the controller:

| | `relative_tolerance = 1e-3` | `relative_tolerance = 1e-4` |
|---|---|---|
| Initial step | 3.11875e-6 s, rejected | 3.11875e-6 s, rejected |
| Steps rejected on the error estimate, the initial step among them | 3 | 3 |
| Steps repeated after a Newton failure | 1, at 1.03e-3 s | 0 |
| First accepted step | 4.05e-7 s | 9.4e-8 s |
| Smallest accepted step | 2.7e-7 s | 7.2e-8 s |

#### How to repeat it

The default `diffusivity` of `build` and `run_window` of `test/example_disruption.jl` is that of scenario 1.
So the run of scenario 4 needs `diffusivity = 1000.0`.

1. `read_dream_state` reads DREAM's output of run 4.
2. `run_window(start, :dream; diffusivity = 1000.0, solver)` runs the quench, with a [`NewtonSolver`](@ref) on [`MKLPardiso`](@ref).
3. `phase_differences` returns the largest difference of each quantity.

`phase_differences` gives larger values for three quantities of this run:

| Quantity | Value of `phase_differences` |
|---|---|
| `n_re` | 2.23e-2 |
| `j_re` | 3.35e-2 |
| `f_re` | 18.0 |

These come from the outer radii, below DREAM's tolerances.
The table above takes `n_re` and `j_re` per group of radii.
It takes `f_re` only where DREAM's largest value at the radius and step is above 1e5.

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
- Checkpoints and segments of a run; with [`SparseLU`](@ref) a resumed run repeats the uninterrupted one bit for bit ([`CheckpointSettings`](@ref), [`read_output`](@ref)).
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
| flux through ``p = 0`` on a runaway grid from ``p = 0`` | the operator's default, `p_zero_boundary = :outflow` of [`fokker_planck_operator`](@ref), as DREAM's hot-tail grid: the friction carries electrons out by ``\lim_{p \to 0} p^3 ν_s`` ([`p3_slowing_down_frequency_at_zero`](@ref)); with it the runaway grid of the case `heating` of `reference/dream/cases/initial_state.py` loses 1.55e-4 of the density of ``f_\mathrm{re}`` in the first step, where DREAM's stays constant to round-off | none, the option `p_zero_boundary = :zero_flux`, which the caller sets for a runaway grid from ``p = 0``: DREAM's ``f_\mathrm{re}`` has no internal boundary condition (`src/Settings/Equations/f_re.cpp:87`) and keeps the Dirichlet lower boundary of its advection in ``p`` (`fvm/Equation/AdvectionTerm.cpp:36`), while ``f_\mathrm{hot}`` has one (`src/Settings/Equations/f_hot.cpp:96`), which mirrors it (`src/Settings/Equations/distribution.cpp:234-237`); both `compat` values of `runaway_heating_example` (`test_runaway_heating_dream.jl`) |
| Rechester–Rosenbluth coefficient ``D = \pi q R_0 v_\parallel (\delta B/B)^2`` | safety factor and major radius as inputs | ``q = 1`` and ``R_0 = 1`` m in a cylinder, `safety_factor = 1`, `major_radius = 1` |
| heating of the cold electrons by a runaway grid | the energy change of the electron-electron operator of the grid, [`free_electron_energy_transfer_term`](@ref), with a linear scheme of the operator | the moment ``m_e c\, v p\, \nu_s f`` at the cell centres, [`FrictionEnergyTransferTerm`](@ref), which also counts the energy that the friction carries out of the grid through ``p = 0`` where that flux is applied (`p_zero_boundary = :outflow`); `compat = dream` of the test case `runaway_heating_example` (`test/example_cases.jl`), compared with DREAM's run in `test_runaway_heating_dream.jl` |
| radiated power of the ions | line and recombination radiation, the binding energy of ionization and recombination | line radiation, the ionization loss and Gould's bremsstrahlung, `radiation = :dream` of [`RadiatedPowerTerm`](@ref) (with `bremsstrahlung_corrections = :dream` for DREAM's corrections) |
| relativistic corrections of Gould's bremsstrahlung, the coefficients of ``T/m_ec^2`` | electron-ion ``11/24`` of the erratum ([`GOULD_ION_CORRECTION`](@ref)), electron-electron ``5(44 - 3\pi^2)/(24\sqrt{2})``, the ratio of eqs. (22) and (18) ([`GOULD_ELECTRON_CORRECTION`](@ref)), of R. J. Gould, Astrophys. J. 238, 1026 (1980), [doi:10.1086/158068](https://doi.org/10.1086/158068), erratum Astrophys. J. 243, 677 (1981), [doi:10.1086/158634](https://doi.org/10.1086/158634) | ``19/24`` and ``5(44 - 3\pi^2)/(8\sqrt{2})``, `bremsstrahlung_corrections = :dream` |
| cross sections of the kinetic ionization | CHIANTI's for H, D, T, Ne and Ar, DREAM's for the other elements | DREAM's fits, `form = :dream` |
| ionization at the hot threshold | the fluid coefficient below the threshold, `kinetic_ionization` of [`IonRateTerm`](@ref) | the whole ADAS rate next to the kinetic term, [`IonRateTerm`](@ref) without `kinetic_ionization` |
| Coulomb logarithm among the ions | the ion-ion logarithm of the NRL Plasma Formulary (A. S. Richardson, NRL/PU/6770--19-652, Naval Research Laboratory, 2019) with the temperature of each species | one logarithm with ``T_e``, `ion_coulomb_logarithm = :dream` of [`IonEnergyExchangeTerm`](@ref) |
| ion masses | nuclear masses of H, D and T, standard atomic weights otherwise | standard atomic weights up to ``Z = 40``, ``2.3\,Z`` u beyond, `mass_table = :dream` |
| charge states in the collisions of ``f`` | those of the start of the step, `lagged = :ions` | those of the iterate with their Jacobian columns, `lagged = ()` |
| Newton matrix of a linear system | assembled and factorized once per step | every iteration, `reuse_linear_matrix = false` of [`NewtonSolver`](@ref) |
| negative entries of a kinetic unknown at convergence | one more Newton iteration for the schemes of [`POLISHED_SCHEMES`](@ref) | none, `polish_negatives = false` of [`NewtonSolver`](@ref) |
| iterative refinement of MKL PARDISO | none, `refinement_steps = 0` of [`MKLPardiso`](@ref) | up to 10 steps per solve (`fvm/Solvers/MIMKL.cpp`), `MKLPardiso(refinement_steps = 10)` |
| floor of a temperature unknown | [`DEFAULT_TEMPERATURE_FLOOR`](@ref) ([`fluid_temperature_floors`](@ref)) | damped towards zero, a lower floor of [`PlasmaFromUnknowns`](@ref) or a [`LowerLimit`](@ref) |

### Behaviour without an option for DREAM's

- Newton's method converges on both the update and the scaled residual, except for the linear steps that `linear_acceptance` accepts on the update ([`NewtonSolver`](@ref)).
  A stiff step at the round-off floor of its residual is accepted on the update too, within the round-off bound of its rows or where the residual stopped decreasing.
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
- The `:smooth` envelope of a [`MomentumRegion`](@ref) whose limit lies at or above the last cell centre takes the width of the last cell.
  DREAM takes the limit minus the last centre (`FindThresholdStep`, `fvm/Equation/MomentQuantity.cpp`): zero width and a NaN at that centre, and a jump of the envelope as the limit crosses it.

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
