# Positivity and flux limiters

```@meta
CurrentModule = Runes2P
```

Negative values arise in two places, and Runes2P treats them separately.
The discretization of the kinetic advection can make a distribution negative: that is a property of the interpolation scheme.
A Newton iterate can overshoot below a physical bound of a fluid unknown: the lower limits of the Newton step prevent that.
A distribution ([`KineticUnknown`](@ref)) is normally declared without a lower limit, so its negative values show the scheme and are not hidden by the solver.

## Interpolation schemes of the advection

The operator [`AdvectionDiffusion`](@ref) writes the flux through a face between two cell centres a distance ``\Delta`` apart as

```math
\Gamma = A f_\mathrm{face} - D \frac{\partial f}{\partial x},
```

with the advection ``A`` and the diffusion ``D`` given on the face.
The scheme sets the face value ``f_\mathrm{face}`` of the advection.
Every scheme except exponential fitting takes the central difference for the diffusion, and the off-diagonal entries of the diffusion tensor are central in all of them.
The cell Péclet number ``\mathrm{Pe} = |A| \Delta / D`` measures how much the advection dominates on the scale of a cell.

A backward-Euler step solves ``(I/\Delta t - L) f^{n+1} = f^n / \Delta t``.
The flux form conserves particles, so the volume-weighted column sums of ``L`` are zero inside and at most zero with outflow or sinks.
When in addition every off-diagonal coupling of ``L`` is non-negative, ``I/\Delta t - L`` is an M-matrix.
Its inverse is then non-negative, so the step maps a non-negative ``f^n`` to a non-negative ``f^{n+1}`` for every ``\Delta t``.
This is the positivity property of the schemes below.

| Scheme | Face value | Linear in ``f`` | Positivity of the step |
|---|---|---|---|
| `:central` | linear between the two centres | yes | lost where ``\mathrm{Pe} > 2`` |
| `:upwind` | the upstream cell | yes | every ``\mathrm{Pe}`` |
| `:exponential_fitting` | the exact face flux for constant ``A`` and ``D`` | yes | every ``\mathrm{Pe}`` |
| `:quick`, `:upwind_second_order` | three cells, fixed weights | yes | not guaranteed |
| `:tcdf`, `:muscl`, `:smart`, `:ospre`, `:van_albada` | three cells, limited | no | not guaranteed |

The central scheme is second order.
At ``\mathrm{Pe} > 2`` its downstream coupling turns negative, so it oscillates and goes negative in steep profiles.
Upwind is first order and positive.
For pure advection a linear scheme of higher than first order cannot be monotone (S. K. Godunov, Matematicheskii Sbornik 47(89), 271 (1959)), which is why `:quick` and `:upwind_second_order` can go negative as well.
The flux limiters escape that bound by being nonlinear in ``f``.

The scheme can differ between directions (`p_scheme`, `xi_scheme`, `r_scheme`).
[`with_scheme`](@ref) gives the same operator with other schemes in ``p`` and ``\xi``.

### Exponential fitting

`:exponential_fitting` is the Scharfetter-Gummel flux (D. L. Scharfetter and H. K. Gummel, IEEE Transactions on Electron Devices 16, 64 (1969), doi:10.1109/T-ED.1969.16566).
In plasma kinetics it is known as the Chang-Cooper scheme (J. S. Chang and G. Cooper, J. Comput. Phys. 6, 1 (1970), doi:10.1016/0021-9991(70)90001-X):

```math
\Gamma = \frac{D}{\Delta} \left[ B(-w) f_L - B(w) f_R \right],
\qquad w = \frac{A \Delta}{D},
\qquad B(x) = \frac{x}{e^x - 1}.
```

It is the exact flux of the steady one-dimensional face problem with constant ``A`` and ``D`` between the two centres.
Its zero-flux solution ``e^{\int A/D}`` is exact for constant ``A/D``.
Both couplings ``(D/\Delta) B(\pm w)`` are non-negative, so the step is positive at every Péclet number.
At small ``\mathrm{Pe}`` it reduces to a central flux with midpoint weights and is second order.
At large ``\mathrm{Pe}`` it tends to upwind.
Where ``D \le 0`` it takes upwind advection with central diffusion.
The collisional Maxwellian, whose ``A/D`` varies slowly over a cell, is steady up to that variation.

Exponential fitting is the default scheme of [`fokker_planck_operator`](@ref) in every direction, a choice of this package (see [Choices of this package](../choices.md)).
DREAM's default is central, and most comparisons with DREAM set `scheme = :central` (the cooling plasma also runs TCDF).
Where the electric field dominates the diffusion (large ``\mathrm{Pe}``), as in the slide-away loss through ``p_\mathrm{max}`` or the hot-tail seed of a fast quench, it is first order, like upwind.
Quantities set by such a tail need a convergence check in the number of momentum cells.

## Four-point schemes and flux limiters

The schemes of [`FOUR_POINT_SCHEMES`](@ref) build the face value from the upstream cell ``u``, the cell ``uu`` one further upstream and the downstream cell ``d``.
They are ported from DREAM, with van Albada added by Runes2P.
`:quick` takes the quadratic through the three cells and `:upwind_second_order` the second-order upwind weights.
Both are linear in ``f``.

The limiters correct the upstream value by a limited slope:

```math
f_\mathrm{face} = f_u + \psi(r) \, (x_\mathrm{face} - x_u) \, s_u,
\qquad s_u = \frac{f_u - f_{uu}}{x_u - x_{uu}},
\qquad r = \frac{s_d}{s_u}, \quad s_d = \frac{f_d - f_u}{x_d - x_u}.
```

The slope ratio is clamped at ``|r| = 10^5``, as in DREAM, also where ``s_u`` vanishes.
The limiter ``\psi(r)`` makes the scheme nonlinear in ``f``:

- `:tcdf`: the piecewise limiter TCDF of D. Zhang, C. Jiang, D. Liang and L. Cheng (J. Comput. Phys. 302, 114 (2015), doi:10.1016/j.jcp.2015.08.042), in DREAM's form, also on its branch ``r < 0``.
- `:muscl`, `:smart`: piecewise linear, the generalized ``\kappa`` scheme with ``\kappa = 0``, ``M = 2`` and ``\kappa = 1/2``, ``M = 4`` (N. P. Waterson and H. Deconinck, J. Comput. Phys. 224, 182 (2007), doi:10.1016/j.jcp.2007.01.021).
- `:ospre`: the OSPRE limiter of Waterson and Deconinck, ``\psi = 1.5\, r (r + 1)/(r^2 + r + 1)``, applied for every ``r`` as DREAM writes it, which makes it negative for ``-1 < r < 0``.
- `:van_albada`: ``\psi = (r^2 + r)/(r^2 + 1)`` for ``r > 0`` and zero otherwise (G. D. van Albada, B. van Leer and W. W. Roberts, Astronomy and Astrophysics 108, 76 (1982), bibcode 1982A&A...108...76V).

`:muscl`, `:smart` and `:van_albada` have ``\psi = 0`` for ``r \le 0``, so at an extremum their face value is upwind and first order; `:tcdf` and `:ospre` keep a nonzero ``\psi`` there.
Where ``f`` is smooth the limiters are second order.
Its step has no M-matrix property, so non-negativity is not guaranteed.
The tests check it in particular runs.

Outside the grid the stencil uses ghost cells as DREAM sets them for the hot-tail distribution ([`AdvectionDiffusion`](@ref) lists them and the boundary faces).
In the radial direction a four-point scheme sets the face value of the advection and needs one pitch grid at all radii.

`test/test_physics.jl` checks the order of the residual of a Maxwellian under the collision operator, central and van Albada, in the bulk of the distribution.

`:muscl` and `:smart` are piecewise linear, and the ``\psi'`` of `:tcdf` jumps at ``r = 0``.
Where the Jacobian jumps between pieces, a Newton iteration on a steady problem can cycle between them.
Time stepping reaches the same steady state, which `test/test_pitch_accuracy.jl` checks with `:smart` for the pitch equilibrium ``f \propto e^{c \xi}``, with ``c`` the ratio of the pitch advection to the pitch scattering.

## Jacobians of the limiters

A limiter makes the Fokker-Planck term nonlinear in ``f``, and the Newton iteration needs its Jacobian.
The `jacobian` option of [`FokkerPlanckTerm`](@ref) selects what the stored block holds, as DREAM's `ad_jac` setting does:

- `:exact`: the derivative of the residual, including ``\psi'(r)``.
- `:frozen_limiter`: the limiter weights frozen at the iterate and differentiated as constants.
- A linear scheme name: the Jacobian of the operator with that scheme ([`with_scheme`](@ref)), such as `:upwind` with a central residual.

The residual is always the operator's own, so every mode converges to the same solution.
For a scheme linear in ``f`` the frozen limiter is the exact Jacobian.
For a limiter `:frozen_limiter`, and for any scheme a scheme name other than its own, make the block approximate ([`jacobian_kind`](@ref)).
The Newton iteration on the stored matrix is then quasi-Newton and converges linearly.
Its residual falls by a constant factor per iteration, which the stall test of [`NewtonSolver`](@ref) would mistake for the round-off floor, so such an iteration must pass the residual test.
With [`GMRES`](@ref)`(operator = :exact)` the approximate block only preconditions the exact Newton operator ([`jacobian_correction_product!`](@ref)).

A linear scheme with coefficients fixed within the step (no electric field or plasma read from unknowns) makes the term linear in ``f`` ([`is_linear`](@ref)).
Its stored Jacobian is then exact, and [`NewtonSolver`](@ref) accepts the step on the update test alone (`linear_acceptance`).

## Lower limits of the Newton step

A [`LowerLimit`](@ref) declared on an unknown bounds every entry ``u_k`` from below by

```math
\ell_k = v - \epsilon \, R_k,
```

with the absolute value ``v``, the tolerance ``\epsilon`` and the reference value ``R_k``.
``R_k`` is the unknown's scale, a reference unknown at the same radius, or the sum over the group of the entry.
[`NewtonSolver`](@ref) shortens the whole update ``\delta`` to the step length

```math
\alpha = \min\left(1, \; \min_{k:\, \delta_k < 0} \mu \, \frac{u_k - \ell_k}{-\delta_k}\right),
```

so that no limited entry loses more than the fraction ``\mu`` = `max_reduction` of its distance to the limit in one iteration.
The default ``\mu = 0.9`` is DREAM's value: DREAM keeps each damped entry above a tenth of its value at the start of the iteration (`threshold = 0.1`, `src/Solver/SolverNonLinear.cpp:443`).
Only updates towards the limit are shortened, not those away from it.
An iterate approaches its limit and does not cross it.
The optional line search starts from ``\alpha``.

The Newton solve fails with `:below_lower_limit` ([`NewtonResult`](@ref)) in two cases.
An entry at or below its limit would be pushed further down, or the converged state lies below a limit.
The time stepper ([`TimeStepper`](@ref)) then repeats the step with a shorter length.
A limit with tolerance 0 on an unknown that can sit at exactly 0 fails on a round-off negative update, and a shorter step does not help, so such an unknown takes a positive tolerance.

The limits the system sets:

- Ion densities ([`ion_densities_unknown`](@ref)): each charge state stays at or above ``-\epsilon`` times the density of its species, with ``\epsilon`` = `limit_tolerance` (`reference = :group`), so a trace impurity has a limit of its own.
- The temperature that a [`PlasmaFromUnknowns`](@ref) reads: the floor of that source ([`temperature_floors`](@ref)).
- A temperature that only fluid terms read: [`DEFAULT_TEMPERATURE_FLOOR`](@ref), unless the unknown declares a limit ([`fluid_temperature_floors`](@ref)).

A step whose solution lies below a temperature floor is refused and repeated shorter.
DREAM applies the same rule with limit 0 and repeats no step.
It damps ``T_\mathrm{cold}``, ``n_\mathrm{tot}``, ``n_\mathrm{cold}``, where they are unknowns ``W_\mathrm{cold}``, ``W_i`` and the species densities ``N_i``, and the sum ``\sum_{i, Z_0} Z_0^2 n_{i, Z_0}`` at each radius (`src/Solver/SolverNonLinear.cpp:441-508`).
The charge states have no limit of their own there (see [Comparison with DREAM](../dream-comparison.md)).

## Tests

- `test/test_exponential_fitting.jl`: no negative off-diagonal coupling of exponential fitting at any Péclet number, against the negative couplings of the central scheme, and the collisional Maxwellian as a steady state.
- `test/test_flux_limiters.jl`: the orders of the four-point schemes by manufactured solutions, their Jacobian against dense forward-mode AD, the frozen limiter, and the sign of ``f`` in a relaxation run.
- `test/test_pitch_accuracy.jl`: the pitch equilibrium per scheme and the sign of ``f`` in ``\xi``.
- `test/test_cooling_dream.jl`: a cooling plasma with a slide-away field against DREAM, for central interpolation and TCDF, comparing the distribution and the number of cells where it is negative.
  The test requires that number to be zero with TCDF.
- `test/test_jacobian_strategies.jl`: the Jacobian modes, quasi-Newton on the stored block, and exact Newton with GMRES.
- `test/test_time_stepping.jl`: lower limits that shorten only steps towards the limit, limits relative to a reference, and the temperature floor.
