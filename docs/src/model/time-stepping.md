# Time stepping and solvers

```@meta
CurrentModule = Runes2P
```

A run is one call of [`integrate`](@ref) on an [`EquationSystem`](@ref), the system ``M\,\mathrm{d}u/\mathrm{d}t = \mathrm{rhs}(u, t)`` of the page [Unknowns and the equation system](unknowns.md).
Three levels nest in it.
A [`TimeStepper`](@ref) sets the length of every step and the number of solves it takes.
Each solve is one call of [`newton_solve!`](@ref) with the settings of a [`NewtonSolver`](@ref).
Each Newton iteration is one linear solve by a [`LinearSolver`](@ref).
Between two accepted steps the run saves states, tests the stop, writes checkpoints and applies its interval updates.
The settings of a run can be passed as keywords or as one [`RunOptions`](@ref).

## The step

Every step is a backward-Euler step from the state ``u_n`` at ``t_n`` to ``u`` at ``t = t_n + h``, the zero of the residual of [`residual!`](@ref):

```math
F(u) = \frac{M\,(u - u_\mathrm{ref})}{h} - \mathrm{rhs}(u, t),
```

with ``u_\mathrm{ref} = u_n``.
The two-step BDF2 of the step controller passes another reference and an effective step ([Step controller](#Step-controller)).
The terms of the system enter the step in three groups:

- implicit terms with a stored Jacobian, evaluated at ``u``;
- implicit matrix-free terms, evaluated at ``u`` and entering the Jacobian only through products ([`jacobian_operator`](@ref));
- explicit terms, evaluated at ``u_n`` and absent from the Jacobian.

A term can also lag part of its residual to the start of the step, which [`prepare_step!`](@ref) sets from ``u_n``.
The ion densities in the collisions of a [`FokkerPlanckTerm`](@ref) (`lagged`) and the primary distribution of a [`KnockOnTerm`](@ref) (`treatment = :explicit_primary`) are lagged this way.
A lagged part is first-order accurate in the step, also under BDF2 ([`FokkerPlanckTerm`](@ref), [`StepController`](@ref)).

The same system can be driven by OrdinaryDiffEq or NonlinearSolve through [`ode_function`](@ref).
In that form there is no start of the step, so a lagged part is evaluated at the state itself, and `ode_function` passes no Jacobian for a system with such a term ([`Runes2P.lags_in_ode_form`](@ref)).

## Newton iteration

[`newton_solve!`](@ref) iterates on ``F(u) = 0`` from the guess ``u = u_n``.
The linear system of an iteration is scaled by rows and columns:

```math
A = W \left(\frac{M}{h} - J\right) S, \qquad A\,\tilde\delta = -W F, \qquad \delta = S\,\tilde\delta,
```

with ``J`` the Jacobian of the implicit terms, ``S`` the scales of the unknowns and ``W`` the row weights.
A row with a time derivative has the weight ``h/s_i``, an algebraic row ``1/s_i``, with ``s_i`` the scale of the row.
On a row of a kinetic unknown with a time derivative, ``s_i`` is the larger of the scale of the row and the entry whose time derivative the row holds, taken at the start of the step.
The collision fluxes of the bulk cancel in its rows, and the round-off of that cancellation grows with the entry, not with the scale of the tail.
On a row with a time derivative the scaled residual ``|W F|`` is then the change that the residual causes in one step relative to ``s_i``.
On an algebraic row it is the residual relative to ``s_i``.

### Convergence

An iterate is accepted when two tests pass:

- the update: every entry satisfies ``|\delta_i| \le a_i + r_i\,|u_i|`` with the Newton [`Tolerance`](@ref) of its unknown ([`tolerance`](@ref));
- the residual: ``\max_i |W F|_i`` is at most `residual_tolerance`.

The `group` part of a [`Tolerance`](@ref) adds `group` times the sum of the charge-state densities of the species at the radius, so that a trace impurity is converged on its own scale.
The unknown of a [`Constraint`](@ref) is a rate, and its update test admits the round-off of the constrained quantity divided by ``h``.
An unknown with an infinite absolute tolerance is left out of both tests.

Two waivers of the residual test exist, each flagged in the [`NewtonResult`](@ref) and the [`SolverStatistics`](@ref):

- the residual floor: on a stiff step the terms of a row cancel to many digits, and round-off holds the residual above `residual_tolerance`.
  A step whose update passes is accepted when every row above `residual_tolerance` is within its round-off bound ``|F_i| \le c\,\varepsilon \sum_j |(\partial F/\partial u)_{ij}\, u_j|``, scaled by the row's weight, with ``c = 1`` ([Choices of this package](../choices.md)).
  Each term of a row carries a rounding error of about ``\varepsilon`` times its size, so no iterate brings the row lower.
  The bound comes from the stored Jacobian and is used only when that holds every implicit term exactly.
  The bound estimates the size of each term by its linearization and leaves the explicit terms out, so it errs low.
  Where the bound is not available or a row is above it, the step is accepted once the residual stops decreasing, as before; that can take a few iterations more, since the residual at the floor wanders.
  Both are flagged as `residual_floor`.
- the linear step: a system linear in the state ([`is_linear`](@ref)) is solved by the first iteration and accepted at the second, on the update test alone (`linear_step`, `linear_acceptance`).

Both waivers need an exact Newton matrix and a linear solve accurate to `floor_linear_tolerance`.
An inexact solve or an approximate Jacobian block ([`jacobian_kind`](@ref)) converges linearly, with an update that understates the error.
Such a quasi-Newton iteration is held by the residual test alone.

An iterate that passes both tests but has negative entries in a polished kinetic unknown is not accepted (`polish_negatives` of [`NewtonSolver`](@ref), by default for the schemes of [`POLISHED_SCHEMES`](@ref)).
Newton takes one more iteration and accepts the next iterate that passes them, flagged as `polished` and `polished_for_negatives`.
A limiter has then converged further in the tail cells near zero ([Positivity](positivity.md#Convergence-of-the-limiter)).

For a system linear in the state the stored Jacobian depends only on the end time and the length of the step.
With `reuse_linear_matrix` it is assembled and factorized once per step and kept for the later iterations and for a repeated step.
An update of the setup discards it ([`invalidate_jacobian!`](@ref)).

### Lower limits and line search

A [`LowerLimit`](@ref) declared on an unknown shortens the whole Newton step, so that no limited entry loses more than the fraction `max_reduction` of its distance to its limit in one iteration.
Steps away from the limit are never shortened.
The default of `max_reduction` and its origin in DREAM are on the page [Positivity and flux limiters](positivity.md#Lower-limits-of-the-Newton-step).
The temperature of a [`PlasmaFromUnknowns`](@ref) has the floor of that source as its limit.
A temperature that only fluid terms read takes the default floor [`DEFAULT_TEMPERATURE_FLOOR`](@ref) as its limit, unless it declares a [`LowerLimit`](@ref) of its own ([`fluid_temperature_floors`](@ref)).
A converged state below a limit is refused with the status `:below_lower_limit`.
The limiters of the advection are on the page [Positivity and flux limiters](positivity.md).

`line_search = true` adds backtracking on ``\tfrac12 \|W F\|^2`` with the Armijo condition, starting from the step length the lower limits allow.
It is off by default.

A failed iteration never throws.
Its reason is the status of the [`NewtonResult`](@ref), and the stepper retries the step with a shorter length.

## Linear solvers and preconditioners

| Solver | Method | Implicit matrix-free terms |
|---|---|---|
| [`SparseLU`](@ref) | sparse LU (UMFPACK) of the whole matrix, the symbolic analysis reused | low-rank ones only, by a Woodbury update |
| [`MKLPardiso`](@ref) | sparse LU (Intel MKL PARDISO) of the whole matrix, the ordering and the matching reused, threaded | low-rank ones only, by a Woodbury update |
| [`GMRES`](@ref) | restarted GMRES (Krylov.jl), right preconditioned | all, through their products |

[`SparseLU`](@ref) is the default.
It takes the field-particle term ([`low_rank_jacobian`](@ref)) exactly and refuses a system with other implicit matrix-free terms, such as the implicit knock-on term.

[`MKLPardiso`](@ref) takes the same systems.
It needs the package Pardiso.jl: `using Pardiso` in the script loads it, and MKL comes with Pardiso.jl as an artifact.
Then `NewtonSolver(linear_solver = MKLPardiso())` solves with PARDISO on one thread, and `MKLPardiso(threads = 8)` on eight.
Its pivot order differs from UMFPACK's, so its converged states differ from those of [`SparseLU`](@ref) within the Newton tolerance.
PARDISO perturbs a small pivot instead of failing, so a solve whose relative residual exceeds 1e-6 fails with `:singular_matrix`, as a zero pivot does with [`SparseLU`](@ref), and the step is retried shorter ([Choices](../choices.md#Linear-solvers)).
After a failed solve the next factorization computes the ordering and the matching again from its own matrix.
Its timings against UMFPACK on the disruption case are on the page [Performance and accuracy](../performance.md#The-linear-solvers).

[`GMRES`](@ref) solves with the operator ``W (M/h - J) S`` in which ``J`` holds the stored Jacobian and the matrix-free terms.
With right preconditioning the residual it minimizes is the true residual of the scaled system.
Its `relative_tolerance` must be at or below the `floor_linear_tolerance` of the [`NewtonSolver`](@ref), or neither waiver can apply; the Newton workspace warns otherwise.
`operator = :exact` adds the corrections of approximate Jacobian blocks to every product ([`jacobian_correction_product!`](@ref)), so the iteration is Newton and the approximate block only preconditions.
`operator = :stored` keeps the stored matrix, a quasi-Newton iteration.
The default restart length of 50 is a choice of this package ([Choices of this package](../choices.md)).

The preconditioners are factorizations of the stored matrix in blocks ([`BlockPreconditioner`](@ref)):

- [`RadiusBlockLU`](@ref) and [`RadiusBlockILU`](@ref): one block per radius ([`radius_block_range`](@ref)) and one for the scalars ([`scalar_block_range`](@ref)), each factorized exactly or incompletely.
  The couplings between the blocks, the radial transport and the couplings to the scalars, are left out of the preconditioner and left to the Krylov iteration.
- [`FluidSchurLU`](@ref), the default: the kinetic unknowns of each radius as blocks, and the fluid unknowns of all radii in one block through its Schur complement.
  Where every kinetic unknown has one pitch grid at all radii, tridiagonal line solves along the radii take the radial transport of the distributions (`radial_lines`).
  Without them the radial transport is left to the Krylov iteration.
  It holds the coupling through Ampère's law and the scalars exactly.

`preconditioner_refresh` sets when the factors are renewed: in every Newton iteration, in the first iteration of every step, or on demand when the step changes or the Krylov count grows.
A solve that fails with older factors is repeated once with fresh ones.

## Time steppers

The stepper sets the length of every step.
Every time point of the run is the end of a step, and every saved state is a computed one.

| Stepper | Error estimate | Step length | Save times |
|---|---|---|---|
| [`FixedSteps`](@ref) (default) | none | one step per interval of the time points, halved on failure | time points |
| [`StepDoubling`](@ref) | two half steps against one full step | ``\text{safety}\cdot h/\text{error}^{\text{exponent}}`` | time points |
| [`StepController`](@ref) | the predictor of the last accepted states, no extra solve | PI filter with a capped growth | any, each ends a step |

A step that fails is retried with a shorter step, up to a set number of times, after which the run throws [`TimeStepFailure`](@ref).
The state a stepper carries between steps is in every checkpoint ([`stepper_state`](@ref)).

### Fixed steps and step doubling

[`FixedSteps`](@ref) is DREAM's constant stepper.
A failed step is retried with half the time step, the shorter steps continue to the end of the interval, and the next interval starts again with its full length.

[`StepDoubling`](@ref) is DREAM's adaptive stepper, the compatibility option of the step control.
The error of an unknown is ``\|u_\mathrm{full} - u_\mathrm{half}\|_2 / (a + r\,\|u_\mathrm{full}\|_2)``, with ``u_\mathrm{half}`` the result of the two half steps and the norms taken over all entries of the unknown at every radius.
The error of the step is the largest over the unknowns, and the step is accepted at an error of at most 1.
An unknown at zero with no absolute tolerance stays out of that largest error, but fails the step when it moves between the full and the half steps, as in DREAM.
The next step follows from the largest error of the other unknowns, as in DREAM; when that error is zero, the step is retried with half its length ([Choices of this package](../choices.md)).
Its tolerances are the `time_step_tolerances` of the [`EquationSystem`](@ref); an unknown without one gets `relative_tolerance` and no absolute part, and an unknown with both tolerances zero is left out.

### Step controller

[`StepController`](@ref) estimates the local error of a backward-Euler step from the linear extrapolation through the two last accepted states:

```math
P = u_n + \frac{h}{h_{n-1}}\,(u_n - u_{n-1}), \qquad \mathrm{err} = \tfrac12\,(u - P).
```

On the rows with a time derivative, where ``M\,(u_n - u_{n-1}) = h_{n-1}\,\mathrm{rhs}(u_n)``, ``P`` is the forward-Euler predictor, and ``u - P \approx h^2 u''`` is twice the local error.
The first step of a run, and the first after an update that changed the setup by a jump, has no history and takes its estimate from one step doubling.
An update whose change is continuous in time keeps the history and discards only the stored Jacobian: a [`RechesterRosenbluthUpdate`](@ref) whose level has no [`forced_time_points`](@ref), any function other than a [`PiecewiseConstant`](@ref).

The estimate is taken on every entry of every unknown, also on an unknown whose row holds no time derivative of its own.
That is an unknown of an [`Algebraic`](@ref) or a [`Constraint`](@ref) equation, or one whose row holds the time derivative of another unknown (`TimeDerivative(of)`).
For such an unknown ``P`` is the linear extrapolation of its own two last values, and the estimate is

```math
\mathrm{err} = \tfrac12\,(u - P) \approx \tfrac14\,h\,(h + h_{n-1})\,u''.
```

It measures the curvature of the path of the unknown in time: the local error that backward Euler would make on a differential unknown with that path, ``h^2 u''/2`` at constant steps.
It is not the error of the equation of the unknown, which the Newton solve meets to its tolerance.
Nor is it the error that the differential unknowns carry into it.
The doubling of a first step measures that carried error on the unknown of an [`Algebraic`](@ref) equation: the difference of the values its equation gives at the end states of the full and the half steps.
The unknown of a [`Constraint`](@ref) is the mean rate over a step.
Its doubling difference is the change of that rate between the full step and the second half step, ``h\,S'/4`` for a rate ``S``, of first order in the step.
The estimate is twice that difference, ``h\,S'/2``.
With `order = 2` the estimate of such an unknown measures the third derivative of its path.
Such an unknown counts in the largest error and can set the step.
A [`Tolerance`](@ref) with all parts zero in `time_step_tolerances` leaves it out.

The error of an unknown is the root mean square of the estimate over ``a + r\,|u|`` per entry:

- an unknown without its own [`Tolerance`](@ref) gets `relative_tolerance` and the absolute part `relative_tolerance` times its scale, a floor for entries near zero;
- the charge states of an ion species are measured against the density of their species ([`component_groups`](@ref));
- a kinetic unknown also holds its density and current density above the momentum `tail_cut` at every radius ([`density`](@ref), [`current_density`](@ref) with a [`MomentumRegion`](@ref)).

The values of a distribution in its hot tail lie far below the absolute tolerance of the cell error, `relative_tolerance` times the scale, so the cells alone leave the tail uncontrolled; the tail moments control it.
Their absolute tolerance is `tail_floor` times the density of the electrons on all kinetic grids at the radius, which stays a floor for a runaway grid whose electrons all lie above the cut.

The next step follows the PI controller of K. Gustafsson, M. Lundh and G. Söderlind, BIT 28, 270 (1988), doi:10.1007/BF01934091:

```math
h_\mathrm{new} = h \cdot \mathrm{clamp}\!\left(\text{safety}\cdot \mathrm{err}^{-k_1}\,\mathrm{err}_\mathrm{old}^{\,k_2},\ \text{min\_shrink},\ \text{max\_growth}\right),
```

with the gains ``(k_1, k_2) = (0.7, 0.4)/(q + 1)`` for a method of order ``q``, those of K. Gustafsson, ACM Transactions on Mathematical Software 17, 533 (1991), doi:10.1145/210232.210242.
Both are in section IV.2 of E. Hairer and G. Wanner, Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems, 2nd ed., Springer (1996), doi:10.1007/978-3-642-05221-7.
A rejected step is repeated with a shorter step and keeps the memory of the filter.
A step cut short by a time point or a save time sets the next length from the step taken.

With `order = 2` the steps are those of the variable-step BDF2, with ``\omega = h/h_{n-1}``:

```math
M\,(u - a\,u_n + b\,u_{n-1}) = c\,h\,\mathrm{rhs}(u, t), \qquad a = \frac{(1+\omega)^2}{1+2\omega}, \quad b = \frac{\omega^2}{1+2\omega}, \quad c = \frac{1+\omega}{1+2\omega}.
```

The solve is the backward-Euler residual with ``u_\mathrm{ref} = a\,u_n - b\,u_{n-1}`` and the step ``c\,h``.
The error estimate is Milne's, from the quadratic extrapolation through the three last accepted states.
The first two steps of a run, and the first two after an update that jumps the setup, are backward-Euler steps.
BDF2 is zero-stable for ``\omega < 1 + \sqrt2``, which bounds `max_growth`.
It keeps neither the positivity that backward Euler has with an M-matrix operator nor the second order for explicit and lagged terms.
The default `order = 1` is backward Euler, which DREAM uses.

The estimate assumes that every term is implicit.
An explicit or lagged term contributes ``h\,h_{n-1}\,u''/2`` in place of ``h^2 u''/2``, a factor ``h_{n-1}/h`` that the limits of the step ratio bound.

## Legs

[`Leg`](@ref)s describe the time points of a run in pieces, each with its steps and its saved states.
[`time_points`](@ref) turns a list of legs into the `times` and `save_times` of [`integrate`](@ref) with the same floating-point sums, so that every save time of a prescribed-step leg is exactly a time point.
A leg with `save_dt` alone has no prescribed steps and serves a [`StepController`](@ref).

## Interval updates

An [`IntervalUpdate`](@ref) changes the setup of the run at the start and after every accepted step, at the state and the time where that step ends.
The step that follows uses the new setup.

- [`KnockOnCutoffUpdate`](@ref) moves the knock-on cutoff of each radius with the plasma.
- [`RechesterRosenbluthUpdate`](@ref) sets the Rechester–Rosenbluth diffusion ([`add_rechester_rosenbluth_diffusion!`](@ref)) from the prescribed ``\delta B/B`` at the start of each step.
  The diffusion is that of A. B. Rechester and M. N. Rosenbluth, Phys. Rev. Lett. 40, 38 (1978), doi:10.1103/PhysRevLett.40.38.

An update that reports a changed setup discards the stored Jacobian.
A jump also discards the history of the step controller; a change continuous in time (`changes_continuously`, a `RechesterRosenbluthUpdate` without forced time points) keeps it.
A time at which an update or a term switches ([`forced_time_points`](@ref)) must be a time point of the run, and [`check_update`](@ref) refuses an update that does not fit the system.
The state of every update is saved with every saved state and checkpoint, and restored on resume.

## Checkpoints, segments and the stop

A run that does not fit one allocation is a chain of segments.
After each accepted step the loop does, in order:

1. the saves at the end of the step;
2. the stop test of the [`StopCondition`](@ref): the wall time left or a stop file ([`stop_reason`](@ref));
3. a checkpoint ([`CheckpointSettings`](@ref)) when its interval has passed, at a stop and at the last time point;
4. the interval updates for the next step.

A checkpoint writes the output of the segment first and the checkpoint file after it, each by [`write_atomically`](@ref).
It holds the state, the states of the updates, the record of the stepper and a summary of the run ([`StoredState`](@ref)).
`resume` continues from a stored state ([`read_checkpoint`](@ref), [`read_state`](@ref)), and `resume = true` lets one script serve every segment.
On the same layout and with [`SparseLU`](@ref) a resumed run repeats the uninterrupted one bit for bit.
With [`MKLPardiso`](@ref) the resumed run computes the ordering and the matching of PARDISO from its own first matrix, which acts as another pivot order.
Its states differ from those of the uninterrupted run within the Newton tolerance, and within round-off where the Newton iterations of the steps are the same.
With `GMRES(preconditioner_refresh = :on_demand)` the preconditioner factors are not stored, so the Krylov solves after a resume differ within the Newton tolerance.
After either, the step lengths of a [`StepController`](@ref) may differ.
A run resumed onto another layout starts the step controller with a doubling, and fills only the unknowns with `initial_value = :from_equation` that the stored state does not have ([`fill_initial_values!`](@ref)).

[`run_status`](@ref) and [`exit_status`](@ref) tell a batch script whether another segment follows.
[`read_output`](@ref) joins the segments, and the file format is on the page [Output files](output.md).
A [`ProgressLog`](@ref) reports the run while it goes: a header, entries at milestones of the simulated time, and the events of failed steps, checkpoints and stops.

## Comparison with DREAM

`test/test_time_stepping_dream.jl` makes two comparisons with DREAM:

- multi-step backward Euler of the Fokker-Planck operator with DREAM's explicit knock-on term on a prescribed constant plasma, over DREAM's time points.
  It covers two collision variants (superthermal and full), each with [`SparseLU`](@ref) and with [`GMRES`](@ref).
- [`StepDoubling`](@ref) against DREAM's adaptive stepper: the accepted steps, the errors and the rejected attempts, and the moments `n_hot`, `j_hot` and `n_re` at DREAM's accepted times.
  They are compared over the stretch before DREAM's step update reads the tolerances of the wrong unknowns.
  The errors of a second DREAM run, with the check of the runaway unknowns off, are compared over the same stretch.

The departures of the step control and of the Newton iteration from DREAM are on the page [Comparison with DREAM](../dream-comparison.md).
