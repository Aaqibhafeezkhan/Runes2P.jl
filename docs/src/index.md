# Runes2P

```@docs
Runes2P
```

This is the API reference of the package, generated from the docstrings in `src/` and `ext/`.
The status and the conventions are in the `README.md` of the repository.

## Pages

- [Grids](api/grids.md): momentum, pitch and radial grids, the kinetic grid, radial profiles, momentum regions.
- [Physics](api/physics.md): plasma state, Coulomb logarithm, collision frequencies, conductivity, runaway rates, constants.
- [Operators](api/operators.md): the advection–diffusion operator, Fokker–Planck coefficients, radial transport, knock-on and field–particle collisions.
- [Equation system](api/system.md): the unknown layout, equation terms, equations and the system, Jacobians.
- [Solvers and runs](api/solvers.md): linear solvers, Newton, time stepping, interval updates, checkpoints, the progress log, run settings.
- [Extensions](extensions.md): HDF5 output and checkpoints, SciML functions, interpolated radial profiles.

Each category has an overview page and subpages that group its source files.
A subpage has one section per source file, with the exported names first and the internal ones after them.
The grouping is a table in `docs/make.jl`.
A new source file that the table does not list appears on the subpage "Other" of its category.

## Building and viewing

The build runs one Julia process and is strict: every exported name needs a docstring, and every `@ref` link must resolve.

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The site is written to `docs/build/`.
Open `docs/build/index.html` in a browser, or serve it with live reload locally:

```
julia --project=docs -e 'using Revise, LiveServer; servedocs(include_dirs=["src", "ext"], skip_dirs=[joinpath("docs", "src", "api")], launch_browser=true)'
```

LiveServer serves the site at `http://localhost:8000`.
It rebuilds the site when a page in `docs/src` or a file in `src/` or `ext/` changes.
Revise loads the changed docstrings into the running session.
The build writes the API pages into `docs/src/api/`, so LiveServer skips that directory, or it would rebuild without end.
