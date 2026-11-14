# Examples

This folder holds basic runs of the methods in the code, one script each.
Each script takes `name=value` switches on the command line, and its defaults run as they are.
The environment is `examples/Project.toml`, with Runes2P from the parent directory and HDF5 for the output.
From the root of a checkout:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=examples examples/relaxation.jl
julia --project=examples examples/cooling.jl scheme=van_albada solver=gmres
julia --project=examples examples/avalanche.jl treatment=implicit solver=gmres
```

| Script | What it integrates | What it reports |
|---|---|---|
| `relaxation.jl` | a Maxwell-Jüttner distribution in a fixed plasma and field | the density fraction above `p_star=`, when given |
| `cooling.jl` | a thermal quench from the slide-away regime | the saves with negative cells |
| `avalanche.jl` | knock-on multiplication of a Maxwell-Jüttner seed | the mean growth rate of the density |

The docs page [Getting started](../docs/src/getting-started.md) describes the scripts and what they print.
It lists the switches of `common.jl` and of `avalanche.jl`.
It has notes on the grids, the schemes and the solver.
It also covers the lower limit, the checkpoints, the stop of a segment and the output files.
A script's own defaults are in its header.
