# Unknowns and the equation system

```@meta
CurrentModule = Runes2P
```

A run of Runes2P solves one system of equations for one state vector ``u``.
The [`UnknownLayout`](@ref) places the unknowns in ``u``.
Each unknown has one [`Equation`](@ref), whose right-hand side is a sum of [`EquationTerm`](@ref)s.
The [`EquationSystem`](@ref) collects the equations into

```math
M \dot u = \mathrm{rhs}(u, t),
```

with a constant mass matrix ``M``, checks that the parts fit together, and supplies the residual and the Jacobians to the solvers.

## Unknowns

An unknown has a name, a kind, a reference scale and optionally a [`LowerLimit`](@ref).

| Kind | Values | Example |
|---|---|---|
| [`ScalarUnknown`](@ref) | one for the whole system | plasma current, poloidal flux at the wall |
| [`RadialUnknown`](@ref) | `components` per radius | electric field, ion charge-state densities |
| [`KineticUnknown`](@ref) | one per momentum cell of its [`KineticGrid`](@ref) at each radius | hot-tail or runaway distribution |

The momentum grid of a kinetic unknown may differ between radii (see [Grids](grids.md)).
The components of a radial unknown split into consecutive groups ([`component_groups`](@ref)).
For the ion densities of [`ion_densities_unknown`](@ref) a group holds the charge states of one species.
Lower limits and tolerances can be relative to the sum over a group, so a trace impurity is converged on its own density.

### Layout of the state vector

Each radius owns one contiguous block of ``u``.
Within it come the radial unknowns, then the kinetic unknowns, each in the order of declaration.
A kinetic unknown is stored with ``p`` fastest, then ``\xi``.
The scalar unknowns form one block after the radius blocks.
`UnknownLayout(...; scalars = :first)` places that block first, which serves only to check that no code depends on the order.

Code reaches the entries only through the layout, never through offsets of its own:

- [`unknown_id`](@ref) turns a name into the integer id used in inner loops;
- [`radial_view`](@ref), [`kinetic_view`](@ref) and [`scalar_view`](@ref) give type-stable views, also of vectors of dual numbers;
- [`kinetic_array`](@ref) and [`unknown_values`](@ref) present an unknown as a 3D array or as its values over the radii to the operators;
- [`unknown_range`](@ref), [`radius_block_range`](@ref) and [`scalar_block_range`](@ref) give index ranges, which the block preconditioners use.

The layout is immutable.
[`copy_unknowns!`](@ref) carries a state by name to a layout with another order or other unknowns, as a resumed run does.
[`unknown_scales`](@ref) gives the reference scale of every entry, which the tolerances and the row scales build on.

## Equations

An [`Equation`](@ref) names its unknown ``x``, its kind and its terms.

| Kind | Row | Determines its unknown |
|---|---|---|
| [`TimeDerivative`](@ref)`()` | ``\dot x = \sum \text{terms}`` | through the time derivative |
| [`TimeDerivative`](@ref)`(of)` | ``\dot y = \sum \text{terms}`` for another unknown ``y`` | through its terms, or through the equation of ``y`` |
| [`Algebraic`](@ref)`()` | ``0 = \sum \text{terms}`` | through an implicit term that reads ``x`` |
| [`Constraint`](@ref)`()` | ``0 = \sum \text{terms}`` | through the equations of other unknowns |

### Pairing of rows and unknowns

`TimeDerivative(of = y)` puts ``\dot y`` in the row of ``x``, so the row of ``y`` can be algebraic and each row still holds one unknown.
The induction equations of [`induction_equations`](@ref) follow the pairing of [DREAM](../dream-comparison.md).
The row of the parallel electric field ``E`` holds Faraday's law, ``\dot \psi = 2\pi E``, with ``\psi`` the poloidal flux divided by the major radius (V s/m), as in DREAM.
The row of the flux holds Ampère's law, which is algebraic in ``\psi``.
The plasma current, the flux at the edge and the flux at the wall are the scalar unknowns of [`induction_unknowns`](@ref).
The equation of the total current density, the sum of the currents of the run, comes from the caller.

The cold-electron temperature is paired in the same way.
Its row holds the time derivative of the thermal energy ``W_\mathrm{cold}``, whose own row is the algebraic ``0 = \tfrac32 e\, n_\mathrm{cold} T_\mathrm{cold} - W_\mathrm{cold}``, with ``T_\mathrm{cold}`` in eV ([`thermal_energy_equation`](@ref)).
In the fully kinetic mode the row holds the derivative of the total energy instead ([`total_energy_equation`](@ref), see [Energy and current balance](energy-current.md)).

A [`Constraint`](@ref) fixes a rate that appears in other equations.
The magnitude of a particle source is of this kind: its row is the density that the distribution must keep, and the source stands in the kinetic equation.
The source enters only through ``\dot f``, and the constraint does not contain it, so the system has differential index 2 in that unknown, which backward Euler solves.

### Scales and initial values

The scale of a row is the magnitude of the quantity it balances.
By default it is the scale of the unknown whose time derivative the row holds, or of the row's own unknown.
`Equation(...; scale)` sets it where the row balances another quantity: Ampère's law, and every [`Constraint`](@ref).
[`residual_scales`](@ref) divides differential rows by the `time_scale` of the system.

`Equation(...; initial_value = :from_equation)` sets an algebraic unknown from its row at the start of a run, as DREAM's `INITRULE_EVAL_EQUATION` does ([`fill_initial_values!`](@ref)).
The thermal energy of the cold electrons takes its initial value this way.
[`fill_maxwellian!`](@ref) fills a distribution with a Maxwellian from profiles of the density and the temperature, the start of a fully kinetic run in DREAM.

## Terms

A term adds its contribution to the rows of one unknown, the row unknown, and may read any unknowns ([`dependencies`](@ref)).
[`rhs!`](@ref) evaluates it as a pure function of the state and the time.
A few properties decide how a term enters the system and its Jacobian:

| Property | Values | Effect |
|---|---|---|
| [`treatment`](@ref) | `:implicit`, `:explicit` | an explicit term is evaluated at the previous time level and is absent from the Jacobian |
| [`coupling`](@ref) | `:sparse`, `:matrix_free` | a sparse block is stored ([`sparsity!`](@ref)); a matrix-free one enters through [`jacobian_vector_product!`](@ref) |
| [`jacobian_assembly`](@ref) | `:automatic`, `:direct` | colored forward-mode AD of `rhs!`, or the term's own entries ([`add_jacobian!`](@ref)) |
| [`jacobian_kind`](@ref) | `:exact`, `:approximate` | an approximate stored block comes with [`jacobian_correction_product!`](@ref) |
| [`is_linear`](@ref) | `false`, `true` | a linear system with an exact Newton matrix accepts a Newton step on the update test alone, without the residual test (see [Time stepping and solvers](time-stepping.md)) |
| [`prepare_step!`](@ref) | a method or none | evaluates a part lagged to the start of the step |
| [`low_rank_jacobian`](@ref) | `nothing` or [`LowRankBlock`](@ref)s | the linear solvers add the blocks by a Woodbury update |

A row that holds every cell of a radius, such as a density moment, would take one color per cell in the colored AD.
Such terms assemble their block directly, and the coloring sees the other terms alone.
A term whose rows read all charge states of a radius builds its block by one dual-number pass over a reduced layout ([`Runes2P.add_local_jacobian!`](@ref), [`Runes2P.local_layout`](@ref)).
The knock-on term with an implicit primary ([`KnockOnTerm`](@ref)) is matrix-free, since its block is dense per radius; with a lagged primary it assembles a sparse block directly.

A term defined outside the package adds methods to these functions under their qualified names, `Runes2P.rhs!(out, term::MyTerm, ...)`.

## The system

[`EquationSystem`](@ref) sorts the terms into implicit sparse, implicit matrix-free and explicit terms.
[`mass_matrix`](@ref) has an entry 1 in each row with a time derivative, in the column of the unknown whose derivative the row holds.
It is diagonal when every such row differentiates its own unknown.

### Residual and Jacobian

A backward Euler step solves

```math
F(u) = M \frac{u - u_\mathrm{ref}}{h} - \mathrm{rhs}(u, t) = 0
```

with [`residual!`](@ref).
The explicit terms are evaluated at the start state, and [`prepare_step!`](@ref) sees the start state before every evaluation.
For backward Euler ``h = \Delta t`` and ``u_\mathrm{ref}`` is the start state; for the variable-step BDF2 of [`StepController`](@ref) ``h = c\,\Delta t`` and ``u_\mathrm{ref}`` is a combination of the two last states, with ``c`` set by the ratio of the last two steps.
Newton's method ([`NewtonSolver`](@ref)) uses the Jacobian ``M/h - \partial \mathrm{rhs}/\partial u`` (see [Time stepping and solvers](time-stepping.md)).

The stored part of ``\partial \mathrm{rhs}/\partial u`` ([`jacobian!`](@ref)) has three sources:

- the terms with automatic assembly, by colored forward-mode AD ([`jacobian_colors`](@ref) counts the passes);
- the terms with direct assembly, by their own entries;
- the plasma columns of the terms whose coefficients follow unknowns ([`add_plasma_columns!`](@ref)).

[`jacobian_operator`](@ref) adds the matrix-free terms and the corrections of approximate blocks, so the operator is the exact Jacobian.
The structure ([`jacobian_sparsity`](@ref), [`jacobian_prototype`](@ref)) is fixed when the system is built.
The construction evaluates the sparse terms at `initial_state` and `initial_time`.
A term that is defined only within the run's interval or at a physical state therefore builds without error.

[`rhs!`](@ref)`(du, u, system, t)` is the ODE form for a general integrator.
There a lagged part sees the state itself, so [`ode_function`](@ref) passes no Jacobian for a system with a lagged part ([`Runes2P.lags_in_ode_form`](@ref)).

### Tolerances

A [`Tolerance`](@ref) passes an entry when

```math
|\delta| \le a + r|x| + g \sum_{k \in \text{group}} |x_k|,
```

with ``g = 0`` without a group part.
[`tolerance`](@ref) gives the default values; a radial unknown with several groups, such as the ion densities, takes a group part for the Newton iteration, so a trace species is converged on its own density.
The system holds separate tolerances for Newton and for the error control of the time steps.

### Construction checks

The constructor refuses a system in which

- an unknown has no equation or more than one;
- a term reads a name that is not in the layout, or fails its own check against the layout;
- an algebraic row has no implicit term that reads its unknown;
- a constraint has no scale, or no other equation reads its unknown implicitly;
- a `TimeDerivative(of)` pairs unknowns of different kind or sizes, or nothing determines the row's unknown;
- the time derivative of one unknown appears in two rows;
- a tolerance with a group part belongs to an unknown other than a radial one.

The algebraic block is not checked for numerical regularity.

## Coefficients that follow the unknowns

Terms read the plasma through one interface, a [`PlasmaSource`](@ref) evaluated by [`plasma_values`](@ref).
A [`PrescribedPlasma`](@ref) is prescribed in time.
A [`PlasmaFromUnknowns`](@ref) takes the cold-electron temperature and optionally the density and the ion charge states from radial unknowns, and the rest from a prescribed plasma.
A source that reads the ions reads the density as well.
[`is_state_dependent`](@ref) tells the two apart.

### Caches and rebuilds

The values of a [`PlasmaFromUnknowns`](@ref) carry no derivative parts, so the coefficients of a term hold plain numbers.
A cache of coefficients keeps the plasma values per radius it was built from in a [`PlasmaKey`](@ref).
[`changed_plasma!`](@ref) compares them with [`update_key!`](@ref) and returns the plasma only when the values of some radius differ.
A [`FokkerPlanckTerm`](@ref) then rebuilds its coefficients at all radii ([`refresh_coefficients!`](@ref)).
A prescribed source is compared once per time, so the evaluations of a term at one time share one comparison.

### Plasma columns of the Jacobian

The derivatives with respect to the plasma unknowns enter the stored Jacobian as columns of their own ([`has_plasma_columns`](@ref)).
[`dual_plasma`](@ref) seeds the temperature as partial 1, the density as partial 2 and the charge states after them.
The coefficients at a radius depend only on the plasma of that radius, so one evaluation gives the derivatives at every radius.
[`Runes2P.plasma_columns`](@ref) maps the partials to entries of the state, and [`plasma_column_sparsity!`](@ref) lists them in the pattern.
[`add_plasma_columns!`](@ref) applies the dual coefficients to ``f`` with the scheme's own face weights, so the plasma columns are exact also when the stored block in ``f`` is approximate.

### Lagged parts

The keyword `lagged` of [`FokkerPlanckTerm`](@ref) names the part of the plasma taken at the start of the step:

| `lagged` | Taken at the start of the step | Stored Jacobian |
|---|---|---|
| `:all` | the whole plasma | no plasma columns; with the plasma fixed for the step, the block in ``f`` is the term's whole derivative (exact unless `jacobian` selects an approximation) |
| `:ions` (default) | the ion charge states | the columns of the temperature and, when the source reads it, the density |
| `()` | nothing | the columns of the temperature, the density when the source reads it, and the charge states |
| `(:ion_densities,)` | the named ion unknown, as `:ions` | the columns of the temperature and, when the source reads it, the density |

The default `:ions` is a choice of this package (see [Choices of this package](../choices.md)).
A term that lags the ions is of first order under BDF2.
The kinetic ionization and a corrected conductivity read the charge states of the iterate, so within a step the ions enter at two states, a difference of first order in the step.

[`ConductivityModel`](@ref)`(...; kinetic = :discrete_operator)` takes a prescribed plasma.
Its correction comes from the steady state of the discrete operator, and following the unknowns would cost a factorization of a bordered operator per radius at every rebuild.

### Temperature floor

A [`PlasmaFromUnknowns`](@ref) raises a `DomainError` for a temperature at or below its floor.
The Newton iteration reports it as a failed solve, and the stepper answers with a shorter step.
The floor is also the lower limit of the temperature unknown in the Newton iteration ([`temperature_floors`](@ref)).
A temperature that only fluid terms read takes the default floor [`DEFAULT_TEMPERATURE_FLOOR`](@ref) unless it declares a limit of its own ([`fluid_temperature_floors`](@ref)).
The value of the floor is a choice of this package (see [Choices of this package](../choices.md)).
How the Newton iteration keeps an unknown above its limit is on the page [Positivity and flux limiters](positivity.md).

## Pairing checks

Terms that describe one physical process from two sides must agree on their plasma, temperature, treatment, region, ion unknown, composition or tables.
A mismatch gives a result that is wrong without an error, so [`EquationSystem`](@ref) refuses it at construction.

### Plasma and treatment

| Terms | Must agree on | Otherwise |
|---|---|---|
| the terms of the cold bulk | one temperature: all read the unknown once one does | collisions and fluid rows belong to different temperatures |
| [`FokkerPlanckTerm`](@ref) and [`FieldParticleTerm`](@ref) of one unknown | the kind of plasma, the [`CollisionSettings`](@ref), `lagged = :all` when the plasma follows unknowns | the field-particle part does not restore the momentum of the e-e part |
| a corrected [`ConductivityTerm`](@ref) and a [`FieldParticleTerm`](@ref) | not both | the missing part of the conductivity is added twice |
| a corrected conductivity that follows the temperature and the Fokker-Planck terms | none lagged with `:all` | the correction and the current of ``f`` belong to different temperatures |
| `:discrete_operator` conductivity and the [`CurrentMomentTerm`](@ref) of its row | the pitch rule | the correction and the current of ``f`` take different pitch rules |
| both sides of [`IonEnergyExchangeTerm`](@ref) | the same unknowns and plasma | energy is not conserved |

### Shared coefficients

| Terms | Rule | Otherwise |
|---|---|---|
| a term that reads the operator of a Fokker-Planck term with a plasma source | takes the term, not its bare operator | it reads coefficients of another plasma |
| two Fokker-Planck terms with a plasma source | an operator each | the coefficients of one rebuild pass for those of the other |
| a Fokker-Planck term with a plasma source | no radial transport in its operator | the rebuild clears it |
| energy terms, and boundary fluxes or interfaces whose plasma follows unknowns | read the term of the system, not a copy | a copy is never prepared for the step, and its flux is wrong |

### Hot region and kinetic ionization

| Terms | Must agree on | Otherwise |
|---|---|---|
| [`DensityMomentTerm`](@ref), [`CurrentMomentTerm`](@ref), [`EnergyMomentTerm`](@ref), [`KineticIonizationTerm`](@ref), [`KineticEnergyLossTerm`](@ref) over the hot region of one temperature | the threshold factor, the envelope and its width | they count different electrons as hot |
| `bound_friction = :ionization_split` and the [`KineticIonizationTerm`](@ref) of its unknown | the term exists, the threshold factor, the cross-section form | ``f`` pays for other ionizations than it causes |
| the same, with the sinks of ``f`` booked by [`KineticEnergyLossTerm`](@ref) | a [`SecondaryHeatingTerm`](@ref) of that ionization | the energy of the secondaries is lost |
| [`SecondaryHeatingTerm`](@ref) and the split friction | `cut = :none`, the spectrum, the widths unless `:beb` | the heating differs from what ``f`` pays |
| a sinks term and an energy moment over a thermal region | the region | the sinks of the bulk are charged as well |

### Ions

| Terms | Must agree on | Otherwise |
|---|---|---|
| [`IonRateTerm`](@ref) or [`RadiatedPowerTerm`](@ref) with a split ionization | its [`KineticIonizationTerm`](@ref) is in the system, with the same ion unknown and temperature | the share of the electrons above the threshold is left out |
| [`IonRateTerm`](@ref) and [`RadiatedPowerTerm`](@ref) of one ion unknown | the split, the extrapolation, the ADF11 files of the ionization | the binding energy is charged at another rate than the rate equations take |
| [`KineticIonizationTerm`](@ref) and the closure of the cold density | a [`WeightedSumTerm`](@ref) closure reads its ions ([`population_equations`](@ref)) | no electron appears |
| every term that reads ions, where the system evolves charge states | the ion unknown | it reads prescribed or other ions |
| the terms of one ion unknown | one [`IonComposition`](@ref) in one order | components map to other species |
| [`SecondaryHeatingTerm`](@ref) and its kinetic ionization | the temperature unknown | the threshold follows another temperature |

### Rows of the total energy

A row that holds the time derivative of ``W_\mathrm{tot} = W_\mathrm{cold} + W_\mathrm{hot}`` charges every unlisted change of ``W_\mathrm{hot}`` to the cold electrons.
For each distribution in ``W_\mathrm{hot}`` the system therefore requires in that row:

- the sinks of its collisions where they have any ([`KineticEnergyLossTerm`](@ref) with `part = :sinks`), over the region of the energy moment;
- the energy that leaves through ``p_\mathrm{max}``, as [`MomentumBoundaryFluxTerm`](@ref) with `weight = :energy` or as [`InterfaceEnergyFluxTerm`](@ref) of the interface to the runaway grid.

Each of these terms takes a negative coefficient.

The system refuses radial transport of that distribution, [`RadialEdgeLossTerm`](@ref), [`KnockOnTerm`](@ref) and [`RosenbluthPutvinskiSourceTerm`](@ref) on it, since no term of the row carries their energy.
The energy balance itself is on the page [Energy and current balance](energy-current.md).
