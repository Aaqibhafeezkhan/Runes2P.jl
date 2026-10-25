# Output files

```@meta
CurrentModule = Runes2P
```

A run writes HDF5 files through the package extension `Runes2PHDF5Ext`, which loads with HDF5 ([Extensions](../extensions.md)).
Without `using HDF5` the writers and readers below have no methods.
A run that does not fit one allocation is a chain of segments; when a segment stops and how segments are chained: [Time stepping and solvers](time-stepping.md).

## Files of a run

[`write_output`](@ref) writes one [`TimeSolution`](@ref) to a file of any name, in place unless `atomic = true`.
A run whose [`CheckpointSettings`](@ref) have the name `name` writes three kinds of file to its directory:

| File | Content |
|---|---|
| `<name>.segment<k>.h5` | the output of segment `k` (four digits, [`segment_file`](@ref)), rewritten with every checkpoint |
| `<name>.checkpoint.h5` | the last checkpoint ([`checkpoint_file`](@ref)) |
| `<name>.checkpoint.prev.h5` | the checkpoint before it ([`previous_checkpoint_file`](@ref)) |

The segment files and the checkpoints are written by [`write_atomically`](@ref): to a temporary file that is synced and renamed.
A path therefore holds either its old content or the new one, and a failed write changes neither the path nor the previous checkpoint.
The last checkpoint is renamed to the previous one just before the new file is renamed onto its path.
Between the two renames the previous checkpoint is the only one, which is why [`read_checkpoint`](@ref) falls back on it.
A last checkpoint that is not HDF5 does not replace the previous one.

At each checkpoint the segment file is written first, then the checkpoint.
A checkpoint then never names states that its segment file lacks.
An attempt killed between the two writes leaves a segment file that the checkpoint does not name.
A run with `resume = true` replaces that file.
Any other run refuses to replace an existing segment file of its number unless `overwrite = true`.
A run refuses to start while segment files of its name with higher numbers than its segment exist, with `overwrite = true` as well, and it deletes no file: the reason is at the end of [Joining segments](@ref).

## Root of a file

Output and checkpoint files share one root:

| Entry | Kind | Content |
|---|---|---|
| `format_version` | attribute | the format, the string `"major.minor.patch"` of [`OUTPUT_FORMAT_VERSION`](@ref) |
| `runes2p_commit` | attribute | the commit of the Runes2P checkout, `"unknown"` outside a checkout ([`code_version`](@ref)) |
| `runes2p_version` | attribute | the package version |
| `runes2p_tree_clean` | attribute | `1` when the tracked files of the checkout were unchanged |
| `settings` | group | the record of the run ([`run_settings`](@ref)) |
| `layout` | group | the layout of the unknowns, with its grids |
| `time` | dataset | the saved times |
| `unknowns` | group | one entry per unknown |
| `solver_statistics` | group | the statistics of the attempted steps (output files only) |
| `update_states` | group | the states of the interval updates (output files only) |
| `checkpoint` | group | the stepper and the counters (checkpoint files only) |

The version counts up with every change of the format.
[`read_output`](@ref) and [`read_state`](@ref) refuse a file whose `format_version` is older than 0.0.1 or newer than [`OUTPUT_FORMAT_VERSION`](@ref).
A segment without a saved state has no `time`, no `unknowns` and no datasets under `update_states`.

## Records

The `settings` group, the `layout` group and the record parts of a checkpoint are plain records ([`SettingsRecord`](@ref)) written as HDF5 objects:

- a record becomes a group with one entry per key;
- a list of records becomes a group with one subgroup per entry, named by the entry's `name`, and the dataset `order` with the names;
- a list of arrays becomes a group with the datasets `1`, `2`, …;
- a boolean becomes a byte with the attribute `bool`, read back as a boolean;
- strings, numbers, arrays of numbers and lists of strings become datasets.

The reader returns every group as a record, so a list of records reads back as a record keyed by the names and a list of arrays as one keyed `"1"`, `"2"`, ….
[`read_settings`](@ref) returns the record of [`run_settings`](@ref) as written, type for type, and [`settings_hash`](@ref) of it equals the hash written.

### The settings group

[`run_settings`](@ref) builds the record of a run from the [`describe`](@ref) records of its objects.
Its sections:

| Section | Content | In the setup hash |
|---|---|---|
| `grids` | the kinetic grids by name (`grid_1`, …), with their faces | yes |
| `layout` | the unknowns, their order, sizes, scales and lower limits, with the grids by name | yes |
| `system` | every equation with its terms and their options, the Newton and time-step tolerances | yes |
| `updates` | the interval updates by name ([`IntervalUpdate`](@ref)) | yes |
| `prescribed` | the tables of prescribed functions of time, sampled at the time points ([`describe_prescribed_plasma`](@ref)) | yes |
| `solver` | the Newton and linear solver settings | yes |
| `time` | the time points, the save times and the stepper | yes |
| `run` | the `description` that the caller adds | yes |
| `hashes` | the hash of each of the eight sections above | no |
| `setup_hash` | the hash of the eight sections together | no |
| `complete` | false when some object of the run has no record of its own | no |
| `code` | [`code_version`](@ref) | no |
| `environment` | [`run_environment`](@ref) | no |
| `checkpoint` | the [`CheckpointSettings`](@ref), or the type `none` | no |
| `stop` | the [`StopCondition`](@ref), or the type `none` | no |
| `log` | the settings of the [`ProgressLog`](@ref), or the type `none` | no |
| `segment` | the record of the segment | no |

The eight sections of the setup are [`SETUP_SECTIONS`](@ref).
`code`, `environment`, `checkpoint`, `stop`, `log` and `segment` stay outside the hashes, so the same setup has the same hash on another machine and another commit.
The tables of `prescribed` are a section of their own, so the hash of `system` does not change with the time points.
A term refers to its table by name (`table_1`, …), and a function used by several terms has one table.

The `segment` section holds `index`, `start_time`, `saved_before`, `resumed`, `status`, `end_time`, `steps`, `stop_reason` after a stop, `checkpoint_written` and `attempts_before`.
`saved_before` counts the states saved before the segment, `end_time` is the time reached, and `attempts_before` counts the steps attempted before the segment.
`attempts_before` is missing for a segment resumed from an output file.
A resumed segment adds `resumed_from` (the file), `resumed_from_segment`, `previous_commit`, `commit_changed`, `previous_setup_hash`, `settings_changed`, and `changed_sections` when some section of the setup changed.

### The layout group

The root group `layout` is the [`describe`](@ref) record of the [`UnknownLayout`](@ref) with its grids inline under `grids`, so it does not depend on the settings group; the record carries `layout_version = 2`.
A reader rebuilds the layout from it and the kinetic grids with [`cylindrical_grid`](@ref) from the stored faces.
A resume matches the unknowns by name with [`copy_unknowns!`](@ref), which refuses a stored unknown on other momentum grids: the distribution is not remapped between grids.

Some entries are recorded only when they are set, and the reader takes a default where one is missing:

| Entry | Recorded when | Reader without it |
|---|---|---|
| `lower_limit_tolerance` of an unknown | the unknown has a [`LowerLimit`](@ref) | no lower limit |
| `lower_limit_value` | the value is not zero | `0.0` |
| `lower_limit_reference` | the limit has a reference | no reference |
| `group_starts` of a radial unknown | it has more than one component group | one group |

Two entries of the settings record follow the same rule: `ions` of the `plasma_unknowns` of a [`PlasmaFromUnknowns`](@ref), absent for prescribed ions, and `group` of a [`Tolerance`](@ref), absent for group 0.

## Unknowns

The group `unknowns` has one entry per unknown, named by it, with the saved times as the last dimension:

| Kind | Entry | Shape |
|---|---|---|
| scalar | dataset | `(t)` |
| radial | dataset | `(components, r, t)` |
| kinetic, one momentum grid at all radii | dataset | `(p, xi, r, t)` |
| kinetic, per-radius momentum grids | group with one dataset `radius_<ir>` per radius | `(p, xi, t)` |

The shapes are those Julia reads; a C-order reader such as h5py or h5dump sees the dimensions reversed, with time first.
The values are those of the state vector, in the units of each unknown, not divided by its `scale` (a reference magnitude); `time` is in seconds.
In memory, [`unknown_view`](@ref) views one unknown of a saved state of a [`TimeSolution`](@ref).

## Solver statistics and update states

The group `solver_statistics` has one dataset per field of [`SolverStatistics`](@ref).
Its `time` array has one entry per attempted step, rejected ones included, and is not the `time` of the saved states.
The arrays of the Newton iterations have one entry per iteration and name their step by `iteration_attempt`.
A field without entries has no dataset, and the reader takes it as empty.

The group `update_states` has one dataset per interval update, named by the update, of shape `(n, t)`, reversed for a C-order reader, with `n` the length of the update's state.
Column `t` is the state of the update in force when saved state `t` was computed.

## Checkpoint files

[`write_checkpoint`](@ref) writes the root of an output file with one state, and the group `checkpoint`:

| Entry | Content |
|---|---|
| `time` | the time of the state |
| `segment` | the number of the segment that wrote it |
| `saved_count` | the count of states the run had saved up to the time |
| `summary` | attempted and accepted steps, Newton and Krylov iterations, wall time of the run so far |
| `stepper` | the record of the time stepper: its step length and what it carries between steps |
| `update_states` | the state of each interval update |

With a direct solver the stepper record lets a resumed run repeat the steps of the uninterrupted one; the preconditioner factors of [`GMRES`](@ref) with on-demand refresh are not stored ([`StepController`](@ref)).
A state read from an output file has no stepper record, so its time must be a time point of the resumed run.

## Reading

- [`read_output`](@ref) returns the [`TimeSolution`](@ref) of an output file, or of the segments of a run joined; it refuses a checkpoint.
- [`read_settings`](@ref) returns the `settings` group alone.
- [`read_state`](@ref) returns one saved state of an output file, the last by default, or the state of a checkpoint, as a [`StoredState`](@ref) to resume from.
- [`read_checkpoint`](@ref) with the [`CheckpointSettings`](@ref) of a run reads its last checkpoint and falls back on the previous one.
  It returns `nothing` when neither file exists, and throws when files exist and none can be read, so a run with unreadable checkpoints stops instead of starting from the beginning and overwriting its files.

### Joining segments

`read_output(settings)` joins the segment files that exist, in the order of their numbers ([`segment_files`](@ref)).
Files given by name are joined in the order given.
The join refuses files that are not the segments of one run:

- a segment that starts before the one given before it;
- a segment whose `resumed_from_segment` is not the `index` of the one before it, where both records hold them;
- a segment whose `previous_setup_hash` is not the `setup_hash` of the one before it, where both are known;
- a segment with other unknown names or another state length;
- a segment with saved states and other interval updates;
- joined times that are not strictly increasing.

A segment that went on after its last checkpoint is repeated from there by the following one.
Its states after the start time of the following segment are dropped.
Its statistics are cut at the count of attempts the following segment resumed from (`attempts_before` of both), or at the start time where a count is missing.
The count matters because a rejected attempt is recorded at the time it aimed at, which may lie beyond the checkpoint.
The joined solution carries the settings of the last file.

The higher segments of an earlier run under the same name and with the same setup pass these checks.
This is why a run refuses to start while segment files of its name with higher numbers than its segment exist ([`CheckpointSettings`](@ref)).
