# Extensions

Runes2P loads these parts only together with their trigger packages, so that Runes2P itself does not depend on them.

| Extension | Loaded with | Content |
|---|---|---|
| `Runes2PHDF5Ext` | `HDF5` | output files and checkpoints of a run, their readers |
| `Runes2PInterpolationsExt` | `Interpolations` | interpolators of Interpolations.jl as radial profiles |
| `Runes2PSciMLBaseExt` | `SciMLBase` | the SciML form of an equation system |

```@autodocs
Modules = [Base.get_extension(Runes2P, :Runes2PHDF5Ext),
           Base.get_extension(Runes2P, :Runes2PInterpolationsExt),
           Base.get_extension(Runes2P, :Runes2PSciMLBaseExt)]
```
