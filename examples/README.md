# Examples

This folder holds basic runs of the methods in the code, one script each.
Each script takes `name=value` switches on the command line, and its defaults run as they are.
The environment is `examples/Project.toml`, with Runes2P from the parent directory and HDF5 for the output:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=examples examples/relaxation.jl
julia --project=examples examples/cooling.jl scheme=van_albada solver=gmres
julia --project=examples examples/avalanche.jl treatment=implicit solver=gmres
```

| Script | Run | Report |
|---|---|---|
| `relaxation.jl` | a Maxwell-Jüttner distribution under a fixed plasma and field (Dreicer regime by default) | with `p_star=`, the fraction above that momentum |
| `cooling.jl` | a temperature decay from `T=` to `T_final=` over `t_end=` | where a scheme goes negative |
| `avalanche.jl` | knock-on multiplication of a 100 keV seed on DREAM's avalanche example | the growth rate |

The docs page [Getting started](https://osrep.github.io/Runes2P.jl/dev/getting-started/) describes the scripts and what they print.
It has the [switches](https://osrep.github.io/Runes2P.jl/dev/getting-started/#Switches) of `common.jl` and of `avalanche.jl`, and notes on the grids, the schemes and the solver.
It also covers the [lower limit](https://osrep.github.io/Runes2P.jl/dev/getting-started/#Lower-limits), the checkpoints, the Slurm jobs and the [output files](https://osrep.github.io/Runes2P.jl/dev/getting-started/#Output-files).
A script's own defaults are in its header.
