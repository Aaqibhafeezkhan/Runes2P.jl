<p align="center"><img alt="Runes2P" src="docs/src/assets/logo.svg" width="160"></p>

# Runes2P

Runes2P is a relativistic electron kinetic solver for runaway electrons in the style of [DREAM](https://github.com/chalmersplasmatheory/DREAM), written in Julia.
It solves the Fokker-Planck equation of the electrons on momentum grids per radius.
The equation is coupled to the fluid quantities of a cylindrical plasma and solved implicitly.

Runes2P is not yet released.
The version is 0.1.0 until the first tagged release.

Runes2P was developed with heavy use of LLM-based coding tools, under the direction and review of its author.
Every physics value is cited next to it in the code, and the results are checked against DREAM and against independent references by the test suite.

## History of the repository

The development began in a private repository.
The history here keeps one commit per development step; the earlier commits do not pass their tests, since the reference data, the documentation and the final naming arrive with the last commit.

## Installation

Runes2P needs Julia 1.13.
It is not registered in the General registry and is added from its repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/osrep/Runes2P.jl")
```

The HDF5 output, the SciML interface and the Interpolations.jl profiles are package extensions (see the docs page [Extensions](docs/src/extensions.md)).

## A first run

The examples run from a checkout of the repository, in the environment `examples/Project.toml`:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=examples examples/relaxation.jl
julia --project=examples examples/cooling.jl scheme=van_albada solver=gmres
julia --project=examples examples/avalanche.jl treatment=implicit solver=gmres
```

The scripts and their switches are described in [`examples/README.md`](examples/README.md) and on the docs page [Getting started](docs/src/getting-started.md).

## Tests

From the root of a checkout:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

The groups, the tiers, the reference data and the environment variables of the suite are in [`test/README.md`](test/README.md).

## Documentation

The documentation is at <https://osrep.github.io/Runes2P.jl/dev/>.
Its sources are in `docs/src/`, and [`CONTRIBUTING.md`](CONTRIBUTING.md) tells how to build it.
The changes that users see, such as renamed functions and keywords, are listed in [`CHANGELOG.md`](CHANGELOG.md).

## Citing

These are requests, not conditions of the licence.

- Parts of Runes2P are ported from DREAM and compared with it.
  Please cite the DREAM paper: M. Hoppe, O. Embréus and T. Fülöp, Computer Physics Communications 268, 108098 (2021), [doi:10.1016/j.cpc.2021.108098](https://doi.org/10.1016/j.cpc.2021.108098).
- Work that uses the CHIANTI, OPEN-ADAS or NIST data of Runes2P should cite them as the [home page of the documentation](docs/src/index.md) lists.

## Licence

Runes2P is released under the MIT licence, see [`LICENSE`](LICENSE).
The licences of the code and data it builds on are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Questions and bug reports

Questions and bug reports go to the issue tracker, <https://github.com/osrep/Runes2P.jl/issues>.
How to contribute is in [`CONTRIBUTING.md`](CONTRIBUTING.md).
