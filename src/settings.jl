# Records of a run's settings. Every object
# a run is built from has a `describe` method that returns a plain record: a dictionary
# with string keys whose values are strings, booleans, integers, floating-point numbers,
# arrays of numbers, lists of strings, or records. `run_settings` collects them into the
# `settings` group of the output file, with the code version, the environment and a hash
# per section.

using SHA: sha256
using LibGit2: LibGit2
using LinearAlgebra: BLAS

"""
    OUTPUT_FORMAT_VERSION

Version of the output file format, written as the string `"major.minor"` in the root
attribute `format_version`. The minor number counts up (0.1, 0.2, …, 0.10, …) with every
change of the file's content until the first release of Runes2P, which starts 1.0. Files
with the integer 1 in the attribute were written before the numbering.
"""
const OUTPUT_FORMAT_VERSION = v"0.29"

"The format version as written to files, `\"major.minor\"`."
output_format_version() = "$(OUTPUT_FORMAT_VERSION.major).$(OUTPUT_FORMAT_VERSION.minor)"

"A plain record: string keys, plain values (see the head of `src/settings.jl`)."
const SettingsRecord = Dict{String,Any}

# ---------------------------------------------------------------------------------------
# plain values

_plain(x::Bool) = x
_plain(x::Integer) = Int(x)
_plain(x::AbstractFloat) = Float64(x)
_plain(x::Symbol) = String(x)
_plain(x::AbstractString) = String(x)
_plain(x::VersionNumber) = string(x)
function _plain(x::AbstractArray{<:Real})
    isempty(x) && throw(ArgumentError("a settings record cannot hold an empty array"))
    return eltype(x) <: Integer ? Array{Int}(x) : Array{Float64}(x)
end
function _plain(x::AbstractVector{<:Union{Symbol,AbstractString}})
    isempty(x) && throw(ArgumentError("a settings record cannot hold an empty list"))
    return String[String(v) for v in x]
end
function _plain(x::Union{AbstractDict,NamedTuple})
    record = SettingsRecord()
    for (key, value) in pairs(x)
        name = String(key isa Symbol ? key : string(key))
        isempty(name) && throw(ArgumentError("a settings record cannot have an empty key"))
        occursin('/', name) &&
            throw(ArgumentError("the key \"$name\" of a settings record must not contain '/'"))
        record[name] = _plain(value)
    end
    return record
end
_plain(x) = throw(ArgumentError("a settings record cannot hold a value of type $(typeof(x)); " *
                                "use strings, numbers, arrays of numbers, lists of strings or dictionaries of them"))

const PlainValue = Union{Bool,Integer,AbstractFloat,Symbol,AbstractString,VersionNumber,AbstractArray,
                         AbstractDict,NamedTuple}

_type_name(x) = String(nameof(typeof(x)))

# record of an object without a `describe` method: named, and marked as not described,
# which makes the run's record incomplete
_undescribed(x, reason::AbstractString="no describe method") =
    SettingsRecord("type" => _type_name(x), "described" => false, "reason" => String(reason))

_describe_value(x::PlainValue) = _plain(x)
_describe_value(x) = hasmethod(describe, Tuple{typeof(x)}) ? describe(x) : _undescribed(x)

# record of a settings struct: its type and every field
function _describe_fields(x)
    record = SettingsRecord("type" => _type_name(x))
    for name in fieldnames(typeof(x))
        record[String(name)] = _describe_value(getfield(x, name))
    end
    return record
end

"""
    settings_fields(T)

`(described, derived)`: the fields of the type `T` that its [`describe`](@ref) record
holds, and those it leaves out because they follow from the described ones (work arrays,
caches, Jacobians of a grid). Every field is in one of the two, which the tests check, so
a field added to a type must be classified. Types without a method describe every field.
"""
settings_fields(T::Type) = (described=fieldnames(T), derived=())

# ---------------------------------------------------------------------------------------
# context of a description: the grids referred to by name, the time points at which
# prescribed functions of time are sampled, and the tables of those samples

const MAX_PRESCRIBED_TABLE_BYTES = 64 * 1024^2

"""
    DescriptionContext(; times=Float64[], max_table_bytes=$MAX_PRESCRIBED_TABLE_BYTES)

Context shared by the [`describe`](@ref) calls of one run. Kinetic grids are described
once and referred to by name (`grid_1`, `grid_2`, … in the order of first use; grids with
equal faces share a name). `times` are the run's time points, at which prescribed
functions of time are sampled. The tables of the samples are kept in the context and
referred to by name (`table_1`, …), each at most `max_table_bytes` large. A function used
by several terms or updates has one table.
"""
struct DescriptionContext
    grids::Vector{KineticGrid}
    times::Vector{Float64}
    max_table_bytes::Int
    tables::Vector{SettingsRecord}
    table_sources::Vector{Any}
end
DescriptionContext(; times=Float64[], max_table_bytes::Integer=MAX_PRESCRIBED_TABLE_BYTES) =
    DescriptionContext(KineticGrid[], collect(Float64, times), Int(max_table_bytes), SettingsRecord[], Any[])

"Name of `grid` in the context, registering it at its first use."
function grid_name!(context::DescriptionContext, grid::KineticGrid)
    k = findfirst(g -> _same_momentum_grids(g, grid), context.grids)
    if k === nothing
        push!(context.grids, grid)
        k = length(context.grids)
    end
    return "grid_$k"
end

"Records of the grids registered in the context, by name."
describe(context::DescriptionContext) =
    SettingsRecord("grid_$k" => describe(grid) for (k, grid) in enumerate(context.grids))

"Name of the table of the prescribed plasma `plasma_at` in the context, which builds it at the first use."
function prescribed_table!(context::DescriptionContext, plasma_at)
    k = findfirst(source -> source === plasma_at, context.table_sources)
    if k === nothing
        push!(context.table_sources, plasma_at)
        push!(context.tables, describe_prescribed_plasma(plasma_at, context))
        k = length(context.tables)
    end
    return "table_$k"
end

"Records of the tables registered in the context, by name."
describe_tables(context::DescriptionContext) =
    SettingsRecord("table_$k" => table for (k, table) in enumerate(context.tables))

# ---------------------------------------------------------------------------------------
# grids

settings_fields(::Type{<:MomentumGrid}) =
    (described=(:p_faces, :xi_faces), derived=(:p_centers, :p_widths, :xi_centers, :xi_widths))
settings_fields(::Type{<:RadialGrid}) =
    (described=(:r_faces,), derived=(:r_centers, :r_widths, :spatial_jacobian, :spatial_jacobian_faces))
settings_fields(::Type{<:KineticGrid}) =
    (described=(:radial_grid, :momentum_grids, :shared, :jacobian_centers),
     derived=(:jacobian_p_faces, :jacobian_xi_faces, :jacobian_r_faces, :jacobian_over_p2_at_zero,
              :same_p_faces, :same_cell_counts, :radial_overlaps))

"""
    describe(object)
    describe(object, context::DescriptionContext)

Plain record of `object`: a dictionary with string keys holding its type under `"type"`
and its settings as strings, numbers, arrays of numbers, lists of strings and nested
records. Defined for the grids, the layout, the plasma and collision settings, the
operators, the terms, the equation system and the solvers. With a `context`, kinetic
grids are referred to by name and prescribed functions of time are sampled at the run's
time points. An object without a method gets a record with `"described" => false`.

A term defined outside the package adds `Runes2P.describe(term::MyTerm, context::Runes2P.DescriptionContext)`.
"""
describe(grid::MomentumGrid) =
    SettingsRecord("type" => "MomentumGrid", "p_faces" => _plain(grid.p_faces), "xi_faces" => _plain(grid.xi_faces))

describe(grid::RadialGrid) = SettingsRecord("type" => "RadialGrid", "r_faces" => _plain(grid.r_faces))

# whether the phase-space Jacobian at the cell centres is the cylindrical 2π p² V'
function _is_cylindrical(grid::KineticGrid)
    volume = grid.radial_grid.spatial_jacobian
    return all(1:radial_cell_count(grid)) do ir
        m = momentum_grid(grid, ir)
        jacobian = phase_space_jacobian(grid, ir)
        all(jacobian[i, j] == 2pi * m.p_centers[i]^2 * volume[ir]
            for i in 1:momentum_cell_count(m), j in 1:pitch_cell_count(m))
    end
end

function describe(grid::KineticGrid)
    record = SettingsRecord("type" => "KineticGrid", "r_faces" => _plain(grid.radial_grid.r_faces),
                            "shared_momentum_grid" => has_shared_momentum_grid(grid),
                            "radial_cell_count" => radial_cell_count(grid))
    if _is_cylindrical(grid)
        record["geometry"] = "cylindrical"
    else
        record["geometry"] = "other"
        record["described"] = false
        record["reason"] = "the phase-space Jacobian is not the cylindrical one and is not stored"
    end
    if has_shared_momentum_grid(grid)
        record["momentum_grid"] = describe(momentum_grid(grid))
    else
        record["momentum_grids"] = SettingsRecord("radius_$ir" => describe(momentum_grid(grid, ir))
                                                  for ir in 1:radial_cell_count(grid))
    end
    return record
end

# ---------------------------------------------------------------------------------------
# layout

settings_fields(::Type{<:UnknownLayout}) =
    (described=(:radial_grid, :unknowns, :scalars),
     derived=(:ids, :kinds, :offsets, :radius_offsets, :sizes, :shapes, :kinetic_grids, :scalar_offsets,
              :radius_blocks, :scalar_block, :length))

"""
    describe(layout)
    describe(layout, context::DescriptionContext)

Plain record of the layout: the order of the unknowns and per unknown its kind, scale,
position, sizes per radius, components, grid and lower limit. With a `context` the grids
are named in it (the `layout` section of the settings, next to their `grids` section).
Without one, the record holds the records of its grids under `grids` and is
self-contained: it is the root group `layout` of the output and checkpoint files, which
a resume reads to match the unknowns by name ([`copy_unknowns!`](@ref)) and to rebuild
the grids (`layout_version` 2; version 1 held the faces per unknown).
"""
function describe(layout::UnknownLayout)
    context = DescriptionContext()
    record = describe(layout, context)
    record["grids"] = describe(context)
    return record
end

function describe(layout::UnknownLayout, context::DescriptionContext)
    unknowns = SettingsRecord()
    for (id, u) in enumerate(layout.unknowns)
        entry = SettingsRecord("kind" => String(layout.kinds[id]), "scale" => u.scale, "position" => id,
                               "sizes" => u isa ScalarUnknown ? [1] : layout.sizes[id, :])
        u isa RadialUnknown && (entry["components"] = u.components)
        # the first component of each group, recorded only for more than one group
        u isa RadialUnknown && length(u.groups) > 1 && (entry["group_starts"] = first.(u.groups))
        u isa KineticUnknown && (entry["grid"] = grid_name!(context, u.grid))
        if u.lower_limit !== nothing
            entry["lower_limit_tolerance"] = u.lower_limit.tolerance
            # recorded only when set, so that the records of earlier limits keep their content
            iszero(u.lower_limit.value) || (entry["lower_limit_value"] = u.lower_limit.value)
            u.lower_limit.reference === nothing ||
                (entry["lower_limit_reference"] = String(u.lower_limit.reference))
        end
        unknowns[String(u.name)] = entry
    end
    return SettingsRecord("type" => "UnknownLayout", "layout_version" => 2,
                          "scalars" => String(layout.scalars),
                          "order" => String[String(u.name) for u in layout.unknowns],
                          "r_faces" => _plain(layout.radial_grid.r_faces),
                          "unknown_count" => length(layout), "unknowns" => unknowns)
end

# ---------------------------------------------------------------------------------------
# plasma and collisions

describe(settings::CollisionSettings) = _describe_fields(settings)
describe(species::IonSpecies) = _describe_fields(species)

function describe(plasma::PlasmaState)
    record = SettingsRecord("type" => "PlasmaState")
    for field in (:cold_electron_temperature, :cold_electron_density, :electric_field, :magnetic_field)
        record[String(field)] = _plain(getfield(plasma, field))
    end
    record["ions"] = SettingsRecord("species_$k" => describe(s) for (k, s) in enumerate(plasma.ions))
    return record
end

function _callable_name(f)
    name = String(nameof(f isa Function ? f : typeof(f)))
    return startswith(name, '#') ? "anonymous function" : name
end

function _sample_indices(count::Integer, max_samples::Integer)
    count <= max_samples && return collect(1:count)
    return unique(round.(Int, range(1, count; length=max(max_samples, 2))))
end

# numbers in the record of one plasma state
_plasma_size(plasma::PlasmaState) =
    4 * radial_cell_count(plasma) + sum(s -> length(s.charge_state_densities), plasma.ions; init=0)

"""
    describe_prescribed_plasma(plasma_at, context)

Table of a plasma state prescribed as a function of time, `plasma_at(t)`: the states at
the time points of `context`, stacked along a last time axis (`(radii, times)` for the
profiles, `(charge states, radii, times)` for the ion densities). These are the states
the steps of the run use, except the steps shortened after a failure. A function cannot
be stored, so the table is the record of it. A table that would exceed the context's
`max_table_bytes` is thinned: sampled evenly over the time points at as many as fit, and
marked `thinned`. Without time points, or when the function fails at one, the record is
marked as not described.
"""
function describe_prescribed_plasma(plasma_at, context::DescriptionContext)
    record = SettingsRecord("type" => "prescribed PlasmaState", "source" => _callable_name(plasma_at))
    function incomplete(reason)
        record["described"] = false
        record["reason"] = reason
        return record
    end
    isempty(context.times) && return incomplete("no time points to sample the function at")
    first_state = try
        plasma_at(first(context.times))
    catch error
        return incomplete("the function failed at a time point: " * sprint(showerror, error))
    end
    first_state isa PlasmaState || return incomplete("the function does not return a PlasmaState")
    max_samples = context.max_table_bytes ÷ (8 * _plasma_size(first_state))
    indices = _sample_indices(length(context.times), max_samples)
    times = context.times[indices]
    # the first sample is the state above: one call of the function per sample
    states = try
        [k == 1 ? first_state : plasma_at(t) for (k, t) in enumerate(times)]
    catch error
        return incomplete("the function failed at a time point: " * sprint(showerror, error))
    end
    all(s -> s isa PlasmaState, states) || return incomplete("the function does not return a PlasmaState")
    reference = first_state
    same_species(s) = length(s.ions) == length(reference.ions) &&
                      all(k -> s.ions[k].name == reference.ions[k].name &&
                               s.ions[k].nuclear_charge == reference.ions[k].nuclear_charge,
                          eachindex(s.ions))
    all(same_species, states) || return incomplete("the ion species change in time")
    record["times"] = times
    record["time_count"] = length(context.times)
    for field in (:cold_electron_temperature, :cold_electron_density, :electric_field, :magnetic_field)
        record[String(field)] = _plain(stack(getfield(s, field) for s in states))
    end
    ions = SettingsRecord()
    for (k, species) in enumerate(reference.ions)
        ions["species_$k"] = SettingsRecord(
            "type" => "IonSpecies", "name" => species.name, "nuclear_charge" => species.nuclear_charge,
            "charge_state_densities" => _plain(stack(s.ions[k].charge_state_densities for s in states)))
    end
    record["ions"] = ions
    record["thinned"] = length(indices) < length(context.times)
    return record
end

# ---------------------------------------------------------------------------------------
# operators

const COEFFICIENT_FIELDS = (:advection_p, :diffusion_p_p, :diffusion_p_xi, :advection_xi, :diffusion_xi_xi,
                            :diffusion_xi_p, :advection_r, :diffusion_r_r, :advection_p_times_p2_at_zero)
const OPERATOR_OPTION_FIELDS = (:p_scheme, :xi_scheme, :r_scheme, :p_lower_boundary, :p_upper_boundary,
                                :r_boundary)

settings_fields(::Type{<:AdvectionDiffusion}) =
    (described=(:grid, COEFFICIENT_FIELDS..., :p_flux_at_lower_boundary, OPERATOR_OPTION_FIELDS...,
                :coefficient_source),
     derived=(:momentum_grids, :radial_overlaps, :jacobian_centers, :jacobian_p_faces, :jacobian_xi_faces,
              :jacobian_r_faces, :jacobian_over_p2_at_zero, :unit_field_advection_p, :unit_field_advection_xi))

"""
    describe(op::AdvectionDiffusion, context=DescriptionContext(); coefficients=true)

Record of the operator: its grid, schemes and boundary conditions, the boundary flux at
`p_min` when it is not zero, and with `coefficients = true` the coefficient arrays. An
array that is zero everywhere is named in `zero_coefficients` and not stored. When the
coefficients were built by [`set_fokker_planck_coefficients!`](@ref), the plasma and the
collision settings they were built from are recorded under `coefficient_source`. The
arrays are the definition of the operator, the source is what explains them. Arrays
changed by hand afterwards leave the source as it was.
"""
function describe(op::AdvectionDiffusion, context::DescriptionContext=DescriptionContext();
                  coefficients::Bool=true)
    record = SettingsRecord("type" => "AdvectionDiffusion", "grid" => grid_name!(context, op.grid))
    for field in OPERATOR_OPTION_FIELDS
        record[String(field)] = String(getfield(op, field))
    end
    any(!iszero, op.p_flux_at_lower_boundary) &&
        (record["p_flux_at_lower_boundary"] = _plain(op.p_flux_at_lower_boundary))
    coefficients || return record
    values = SettingsRecord()
    zero_names = String[]
    for field in COEFFICIENT_FIELDS
        array = getfield(op, field)
        any(!iszero, array) ? (values[String(field)] = _plain(array)) : push!(zero_names, String(field))
    end
    record["coefficients"] = values
    isempty(zero_names) || (record["zero_coefficients"] = zero_names)
    source = op.coefficient_source[]
    if source !== nothing
        record["coefficient_source"] = SettingsRecord("plasma" => describe(source.plasma),
                                                      "collision_settings" => describe(source.settings),
                                                      "synchrotron" => source.synchrotron)
        # the field of the plasma is left out when the field is an unknown of the system
        get(source, :electric_field, true) || (record["coefficient_source"]["electric_field"] = false)
    end
    return record
end

settings_fields(::Type{<:KnockOnOperator}) =
    (described=(:knockon, :primary, :method, :pitch_options, :p_cutoff),
     derived=(:kernels, :cache, :primary_weights, :knockon_row, :gathered_weights, :slice_products))

"""
    describe(op::KnockOnOperator, context=DescriptionContext())

Record of the operator: the knock-on and primary grids, the evaluation method, the cutoff
momentum per radius as it is now, and the options of the kernels with their defaults
filled in ([`resolved_pitch_options`](@ref)).
"""
function describe(op::KnockOnOperator, context::DescriptionContext=DescriptionContext())
    return SettingsRecord("type" => "KnockOnOperator", "knockon" => grid_name!(context, op.knockon),
                          "primary" => grid_name!(context, op.primary), "method" => String(op.method),
                          "p_cutoff" => _plain(op.p_cutoff),
                          "pitch_options" => _plain(resolved_pitch_options(op.pitch_options)))
end

# ---------------------------------------------------------------------------------------
# terms

function _term_record(term::EquationTerm)
    record = SettingsRecord("type" => _type_name(term), "treatment" => String(treatment(term)),
                            "coupling" => String(coupling(term)), "linear" => is_linear(term),
                            "jacobian_kind" => String(jacobian_kind(term)))
    names = dependencies(term)
    isempty(names) || (record["dependencies"] = String[String(name) for name in names])
    return record
end

# a term without a record of its own
function describe(term::EquationTerm, context::DescriptionContext)
    record = _term_record(term)
    record["described"] = false
    record["reason"] = "no describe method"
    return record
end
describe(term::EquationTerm) = describe(term, DescriptionContext())

settings_fields(::Type{<:FokkerPlanckTerm}) =
    (described=(:unknown, :operator, :jacobian, :electric_field, :plasma, :settings, :synchrotron, :coefficients,
                :lagged),
     derived=(:stored_operator, :values, :correction, :dual_input, :dual_output, :field_values, :dual_field,
              :plasma_key, :rebuilds, :dual_operator, :dual_columns, :ion_state))

function describe(term::FokkerPlanckTerm, context::DescriptionContext)
    record = _term_record(term)
    record["unknown"] = String(term.unknown)
    record["jacobian"] = String(term.jacobian)
    term.electric_field === nothing || (record["electric_field"] = String(term.electric_field))
    if term.plasma === nothing
        record["operator"] = describe(term.operator, context)
    else
        # the coefficients follow the plasma, so the plasma is their record: the function
        # by its name here, its values in a table of the context
        record["operator"] = describe(term.operator, context; coefficients=false)
        record["collision_settings"] = describe(term.settings)
        record["synchrotron"] = term.synchrotron
        is_state_dependent(term.plasma) && (record["coefficients"] = String(term.coefficients))
        # recorded only when set, so that the records of other terms keep their content
        isempty(term.lagged) || (record["lagged"] = String.(collect(term.lagged)))
        _describe_plasma_source!(record, term.plasma, context)
    end
    return record
end

settings_fields(::Type{<:PrescribedPlasma}) = (described=(:plasma_at,), derived=(:time, :state))
settings_fields(::Type{<:PlasmaFromUnknowns}) =
    (described=(:prescribed, :temperature, :density, :temperature_floor, :ions), derived=(:composition,))

# a plasma from the unknowns: their names, the floor, and the prescribed rest
function _describe_plasma_source!(record, source::PlasmaFromUnknowns, context::DescriptionContext)
    record["plasma_unknowns"] = SettingsRecord("temperature" => String(source.temperature),
                                               "density" => source.density === nothing ? "prescribed" :
                                                            String(source.density),
                                               "temperature_floor" => source.temperature_floor)
    # recorded only when set, so that the records of plasmas with prescribed ions keep their content
    source.ions === nothing || (record["plasma_unknowns"]["ions"] = String(source.ions))
    return _describe_plasma_source!(record, source.prescribed, context)
end

# the plasma of a source in the record of a term: a constant state in full, a function of
# time by its name, with its values in a table of the context shared by the terms that use it
function _describe_plasma_source!(record, source::PrescribedPlasma, context::DescriptionContext)
    if source.plasma_at isa PlasmaState
        record["plasma"] = describe(source.plasma_at)
    else
        record["plasma"] = _callable_name(source.plasma_at)
        record["plasma_table"] = prescribed_table!(context, source.plasma_at)
    end
    return record
end

function describe(term::KnockOnTerm, context::DescriptionContext)
    record = _term_record(term)
    # the treatment asked for; `treatment` above is where the system places the term
    record["knock_on_treatment"] = String(term.treatment)
    record["primary"] = String(term.primary)
    record["target_density"] = term.target_density isa Symbol ? String(term.target_density) :
                               _plain(term.target_density)
    record["operator"] = describe(term.operator, context)
    return record
end

settings_fields(::Type{<:PrescribedValueTerm}) = (described=(:values,), derived=(:points, :evaluated))

function describe(term::PrescribedValueTerm, context::DescriptionContext)
    record = _term_record(term)
    points = isempty(term.points[]) ? nothing : term.points[]
    record["values"] = term.values isa Function ? describe_prescribed_levels(term.values, context; points) :
                       _plain(term.values)
    return record
end

function describe(term::LinearTerm, context::DescriptionContext)
    record = _term_record(term)
    record["unknown"] = String(term.unknown)
    record["coefficient"] = _plain(term.coefficient)
    return record
end

settings_fields(::Type{<:RadiatedPowerTerm}) =
    (described=(:plasma, :temperature, :density, :radiation, :extrapolation, :ions, :kinetic), derived=(:tables,))

function describe(term::RadiatedPowerTerm, context::DescriptionContext)
    record = _term_record(term)
    record["temperature"] = String(term.temperature)
    record["density"] = String(term.density)
    record["radiation"] = String(term.radiation)
    record["extrapolation"] = String(term.extrapolation)
    term.ions === nothing || (record["ions"] = String(term.ions))
    term.kinetic === nothing || (record["kinetic_ionization"] = describe(term.kinetic, context))
    # the ADF11 files of each element, by name: the cache itself is not part of the record
    record["adf11_files"] = SettingsRecord(element => SettingsRecord(String(class) => basename(getproperty(tables, class).source)
                                                                     for class in ADF11_CLASSES)
                                           for (element, tables) in term.tables)
    return _describe_plasma_source!(record, term.plasma, context)
end

settings_fields(::Type{<:IonRateTerm}) =
    (described=(:composition, :ions, :temperature, :density, :extrapolation, :kinetic), derived=(:tables,))

function describe(term::IonRateTerm, context::DescriptionContext)
    record = _term_record(term)
    c = term.composition
    record["species"] = c.names
    record["nuclear_charges"] = c.nuclear_charges
    for field in (:ions, :temperature, :density, :extrapolation)
        record[String(field)] = String(getfield(term, field))
    end
    term.kinetic === nothing || (record["kinetic_ionization"] = describe(term.kinetic, context))
    # the ADF11 files of each element, by name: the cache itself is not part of the record
    record["adf11_files"] = SettingsRecord(name => SettingsRecord(String(class) => basename(getproperty(tables, class).source)
                                                                  for class in (:scd, :acd))
                                           for (name, tables) in zip(c.names, term.tables))
    return record
end

settings_fields(::Type{<:KnockOnTerm}) =
    (described=(:primary, :operator, :target_density, :treatment),
     derived=(:lagged, :prepared, :prepared_from, :preparations))

settings_fields(::Type{<:IonDiffusionTerm}) =
    (described=(:composition, :ions, :level, :level_in_time, :states, :edge), derived=(:faces, :time))

function describe(term::IonDiffusionTerm, context::DescriptionContext)
    record = _term_record(term)
    record["species"] = term.composition.names
    record["nuclear_charges"] = term.composition.nuclear_charges
    record["ions"] = String(term.ions)
    record["states"] = String(term.states)
    record["edge"] = String(term.edge)
    record["diffusion"] = _describe_radial_level(term.level, term.level_in_time, term.faces[], context)
    return record
end

# the record of the level of a radial transport term: a number or a vector as given, a
# history in time by its levels at the time points of the context (a `PiecewiseConstant` by
# its values and switch times), a profile by its source and its values on the faces
# (the name alone lost the levels of a transport history)
function _describe_radial_level(level, in_time::Bool, faces, context)
    level isa Union{Real,AbstractVector} && return _plain(level)
    in_time && return describe_prescribed_levels(level, context)
    record = SettingsRecord("profile" => _describe_profile(level))
    isempty(faces) || (record["faces"] = _plain(faces))
    return record
end

settings_fields(::Type{<:SecondaryHeatingTerm}) =
    (described=(:kinetic, :temperature, :widths), derived=(:energies,))

function describe(term::SecondaryHeatingTerm, context::DescriptionContext)
    record = _term_record(term)
    record["kinetic_ionization"] = describe(term.kinetic, context)
    record["temperature"] = String(term.temperature)
    record["widths"] = SettingsRecord(name => width for (name, width) in zip(term.kinetic.composition.names, term.widths))
    return record
end

settings_fields(::Type{<:KineticIonizationTerm}) =
    (described=(:composition, :kinetic, :grid, :ions, :envelopes), derived=(:rates, :weights))

function describe(term::KineticIonizationTerm, context::DescriptionContext)
    record = _term_record(term)
    record["species"] = term.composition.names
    record["nuclear_charges"] = term.composition.nuclear_charges
    record["kinetic"] = String(term.kinetic)
    record["grid"] = grid_name!(context, term.grid)
    record["ions"] = String(term.ions)
    record["cross_sections"] = "Burgess-Chidichimo with Garland's extension, DREAM's fits"
    return _describe_region!(record, term.envelopes, context)
end

settings_fields(::Type{<:HeatTransportTerm}) =
    (described=(:temperature, :density, :model, :level, :level_in_time, :safety_factor, :major_radius, :edge,
                :coefficient),
     derived=(:faces, :time, :safety_factor_faces))

function describe(term::HeatTransportTerm, context::DescriptionContext)
    record = _term_record(term)
    record["temperature"] = String(term.temperature)
    record["density"] = String(term.density)
    record["model"] = String(term.model)
    record["edge"] = String(term.edge)
    record["coefficient"] = term.coefficient
    name = term.model === :prescribed ? "diffusion" : "delta_b_over_b"
    record[name] = _describe_radial_level(term.level, term.level_in_time, term.faces[], context)
    if term.model === :rechester_rosenbluth
        record["major_radius"] = term.major_radius
        record["safety_factor"] = _describe_radial_level(term.safety_factor, false, term.safety_factor_faces[], context)
    end
    return record
end

settings_fields(::Type{<:IonEnergyExchangeTerm}) =
    (described=(:plasma, :side, :ion_energy, :temperature, :density, :ion_ion, :masses, :ions), derived=())

function describe(term::IonEnergyExchangeTerm, context::DescriptionContext)
    record = _term_record(term)
    record["side"] = String(term.side)
    record["ion_energy"] = String(term.ion_energy)
    record["temperature"] = String(term.temperature)
    record["density"] = String(term.density)
    record["ion_ion"] = term.ion_ion
    record["masses"] = term.masses
    term.ions === nothing || (record["ions"] = String(term.ions))
    return _describe_plasma_source!(record, term.plasma, context)
end

function describe(term::RunawayFluidEnergyTransferTerm, context::DescriptionContext)
    record = _term_record(term)
    record["runaway_density"] = String(term.runaway_density)
    record["temperature"] = String(term.temperature)
    record["density"] = String(term.density)
    record["coulomb_logarithm"] = String(term.coulomb_logarithm)
    term.ions === nothing || (record["ions"] = String(term.ions))
    return _describe_plasma_source!(record, term.plasma, context)
end

settings_fields(::Type{<:KineticEnergyLossTerm}) =
    (described=(:kinetic, :source, :part, :coefficient, :envelopes), derived=(:lagged_plasma,))

function describe(term::KineticEnergyLossTerm, context::DescriptionContext)
    record = _term_record(term)
    record["kinetic"] = String(term.kinetic)
    record["part"] = String(term.part)
    record["coefficient"] = term.coefficient
    record["source"] = describe(term.source, context)
    return _describe_region!(record, term.envelopes, context)
end

function describe(term::InterfaceEnergyFluxTerm, context::DescriptionContext)
    record = _term_record(term)
    record["coefficient"] = term.coefficient
    record["interface"] = describe(term.interface, context)
    return record
end

function describe(term::WeightedSumTerm, context::DescriptionContext)
    record = _term_record(term)
    record["unknown"] = String(term.unknown)
    record["weights"] = term.weights
    return record
end

function describe(term::ProductTerm, context::DescriptionContext)
    record = _term_record(term)
    record["first"] = String(term.first)
    record["second"] = String(term.second)
    record["coefficient"] = term.coefficient
    return record
end

"""
    describe(region::MomentumRegion, context)

Record of a momentum region: the side, the envelope and the limit. A limit that follows
the temperature is recorded with its factor and the temperatures at the time points of
`context`, another function of time with its values at the time points.
"""
function describe(region::MomentumRegion, context::DescriptionContext=DescriptionContext())
    record = SettingsRecord("type" => "MomentumRegion", "side" => String(region.side),
                            "envelope" => String(region.envelope))
    region.envelope === :smooth && (record["smooth_width"] = region.smooth_width)
    record["limit"] = _describe_limit(region.limit, context)
    return record
end

_describe_limit(limit::Union{Real,AbstractVector{<:Real}}, context) = _plain(limit)
_describe_limit(limit, context) = describe_prescribed_levels(limit, context)
function _describe_limit(limit::ThermalMomentum, context)
    record = SettingsRecord("type" => "ThermalMomentum", "factor" => limit.factor)
    if limit.temperature isa Symbol
        record["temperature"] = SettingsRecord("unknown" => String(limit.temperature))
        return record
    end
    record["temperature"] = limit.temperature isa AbstractVector ? _plain(limit.temperature) :
                            describe_prescribed_levels(t -> _temperatures(limit.temperature, t), context)
    limit.temperature isa AbstractVector || (record["temperature"]["source"] = _callable_name(limit.temperature))
    return record
end

# the region of a moment term in its record
function _describe_region!(record, envelopes, context)
    envelopes === nothing || (record["region"] = describe(envelopes.region, context))
    return record
end

settings_fields(::Type{<:DensityMomentTerm}) = (described=(:kinetic, :coefficient, :envelopes), derived=())

function describe(term::DensityMomentTerm, context::DescriptionContext)
    record = _term_record(term)
    record["kinetic"] = String(term.kinetic)
    record["coefficient"] = term.coefficient
    return _describe_region!(record, term.envelopes, context)
end

function describe(term::EnergyMomentTerm, context::DescriptionContext)
    record = _term_record(term)
    record["kinetic"] = String(term.kinetic)
    record["coefficient"] = term.coefficient
    return _describe_region!(record, term.envelopes, context)
end

settings_fields(::Type{<:CurrentMomentTerm}) =
    (described=(:kinetic, :grid, :coefficient, :pitch_rule, :envelopes), derived=(:pitch_weights,))

function describe(term::CurrentMomentTerm, context::DescriptionContext)
    record = _term_record(term)
    record["kinetic"] = String(term.kinetic)
    record["grid"] = grid_name!(context, term.grid)
    record["coefficient"] = term.coefficient
    record["pitch_rule"] = String(term.pitch_rule)
    return _describe_region!(record, term.envelopes, context)
end

settings_fields(::Type{<:ParticleSourceTerm}) =
    (described=(:source, :grid, :temperature, :shape_kind), derived=(:shape, :temperatures, :time))

function describe(term::ParticleSourceTerm, context::DescriptionContext)
    record = _term_record(term)
    record["source"] = String(term.source)
    record["grid"] = grid_name!(context, term.grid)
    record["shape"] = String(term.shape_kind)
    record["temperature"] = if term.temperature isa Symbol
        SettingsRecord("unknown" => String(term.temperature))
    elseif term.temperature isa AbstractVector
        _plain(term.temperature)
    else
        describe_prescribed_levels(t -> _temperatures(term.temperature, t), context)
    end
    term.temperature isa Union{Symbol,AbstractVector} ||
        (record["temperature"]["source"] = _callable_name(term.temperature))
    return record
end

function describe(term::MomentumBoundaryFluxTerm, context::DescriptionContext)
    record = _term_record(term)
    record["kinetic"] = String(term.kinetic)
    record["weight"] = String(term.weight)
    record["coefficient"] = term.coefficient
    operator = _operator_of(term.source)
    record["operator"] = describe(operator, context; coefficients=false)
    return record
end

settings_fields(::Type{<:ConductivityTerm}) =
    (described=(:electric_field, :model, :coefficient), derived=(:values, :plasma_key, :operator_rebuilds))

"""
    describe(model::ConductivityModel, context)

Record of a conductivity model: the reference, what is subtracted for the kinetic
operator, and the conductivity per radius at the time points of `context` (at the first
one for a plasma that is constant in time).
"""
function describe(model::ConductivityModel, context::DescriptionContext=DescriptionContext())
    record = SettingsRecord("type" => "ConductivityModel", "reference" => String(model.reference),
                            "kinetic" => String(model.kinetic))
    if model.kinetic === :discrete_operator
        record["pitch_rule"] = String(model.pitch_rule)
        record["operator"] = describe(_operator_of(model.operator), context; coefficients=false)
    end
    _describe_plasma_source!(record, model.plasma, context)
    # the first time of the run need not be 0, and an operator rebuilt in time makes the
    # model time dependent with a constant plasma
    first_time = isempty(context.times) ? 0.0 : first(context.times)
    record["conductivity"] = if model.plasma isa PlasmaFromUnknowns
        SettingsRecord("described" => false, "reason" => "the conductivity follows the temperature unknown")
    elseif !is_time_dependent(model)
        _plain(model(first_time))
    elseif model.kinetic === :discrete_operator && isempty(context.times)
        SettingsRecord("described" => false, "reason" => "no time points to sample the model at")
    elseif model.kinetic === :discrete_operator
        # each time costs a sparse factorization of the bordered operator per radius
        SettingsRecord("time" => first_time, "values" => _plain(model(first_time)), "sampled" => "first time point",
                       "reason" => "the conductivity of the discrete operator costs a factorization per time")
    else
        describe_prescribed_levels(model, context)
    end
    return record
end

function describe(term::ConductivityTerm, context::DescriptionContext)
    record = _term_record(term)
    record["electric_field"] = String(term.electric_field)
    record["coefficient"] = term.coefficient
    record["model"] = describe(term.model, context)
    return record
end

function describe(term::PoloidalFluxDiffusionTerm, context::DescriptionContext)
    record = _term_record(term)
    record["poloidal_flux"] = String(term.poloidal_flux)
    record["edge_poloidal_flux"] = String(term.edge_poloidal_flux)
    record["coefficient"] = term.coefficient
    return record
end

function describe(term::CrossSectionIntegralTerm, context::DescriptionContext)
    record = _term_record(term)
    record["unknown"] = String(term.unknown)
    record["coefficient"] = term.coefficient
    return record
end

settings_fields(::Type{<:MaxwellianSourceTerm}) =
    (described=(:kinetic, :grid, :target_density, :rate, :temperature, :bulk_momentum),
     derived=(:shape, :bulk_cells))

function describe(term::MaxwellianSourceTerm, context::DescriptionContext)
    record = _term_record(term)
    record["kinetic"] = String(term.kinetic)
    record["grid"] = grid_name!(context, term.grid)
    record["target_density"] = _plain(term.target_density)
    record["rate"] = term.rate
    record["temperature"] = term.temperature
    record["bulk_momentum"] = term.bulk_momentum
    return record
end

function describe(term::RadialEdgeLossTerm, context::DescriptionContext)
    record = _term_record(term)
    record["kinetic"] = String(term.kinetic)
    record["coefficient"] = term.coefficient
    # the coefficients belong to the Fokker-Planck term of the radial transport, or to an
    # update of the run, which hold their record
    record["operator"] = describe(term.operator, context; coefficients=false)
    return record
end

function describe(term::LossRateTerm, context::DescriptionContext)
    record = _term_record(term)
    record["unknown"] = String(term.unknown)
    record["rate"] = _plain(term.rate)
    return record
end

settings_fields(::Type{<:MomentumInterfaceTerm}) =
    (described=(:lower, :upper, :lower_term, :lower_grid, :upper_grid, :pitch_mapping, :scheme),
     derived=(:overlaps, :interpolation_cells, :interpolation_weights))

function describe(term::MomentumInterfaceTerm, context::DescriptionContext)
    record = _term_record(term)
    record["lower"] = String(term.lower)
    record["upper"] = String(term.upper)
    # the coefficients are those of the Fokker-Planck term of the lower unknown, which
    # holds their record
    record["coefficients"] = "last p face of the Fokker-Planck operator of $(term.lower)"
    record["lower_grid"] = grid_name!(context, term.lower_grid)
    record["upper_grid"] = grid_name!(context, term.upper_grid)
    record["pitch_mapping"] = String(term.pitch_mapping)
    record["scheme"] = String(term.scheme)
    return record
end

settings_fields(::Type{<:FieldParticleOperator}) =
    (described=(:grid, :modes, :scheme, :pitch_rule, :coefficient_source),
     derived=(:momentum_weights, :loss_functionals, :restoring_shapes))

"""
    describe(op::FieldParticleOperator, context=DescriptionContext())

Record of the field-particle operator: the grid, the number of modes, the
scheme of its electron-electron operator, the pitch rule of the momentum weights, and the
plasma and the collision settings it was built from. The functionals and the shapes
follow from them.
"""
function describe(op::FieldParticleOperator, context::DescriptionContext=DescriptionContext())
    record = SettingsRecord("type" => "FieldParticleOperator", "grid" => grid_name!(context, op.grid),
                            "modes" => op.modes, "scheme" => String(op.scheme), "pitch_rule" => String(op.pitch_rule))
    source = op.coefficient_source[]
    if source !== nothing
        record["coefficient_source"] = SettingsRecord("plasma" => describe(source.plasma),
                                                      "collision_settings" => describe(source.settings))
    end
    return record
end

settings_fields(::Type{<:FieldParticleTerm}) =
    (described=(:unknown, :operator, :plasma, :settings), derived=(:plasma_key,))

function describe(term::FieldParticleTerm, context::DescriptionContext)
    record = _term_record(term)
    record["unknown"] = String(term.unknown)
    record["operator"] = describe(term.operator, context)
    if term.plasma !== nothing
        # the functionals and the shapes follow the plasma, recorded as for a Fokker-Planck
        # term, and the plasma of the last rebuild is not part of the record
        delete!(record["operator"], "coefficient_source")
        record["collision_settings"] = describe(term.settings)
        _describe_plasma_source!(record, term.plasma, context)
    end
    return record
end

# ---------------------------------------------------------------------------------------
# equation system and solvers

settings_fields(::Type{<:EquationSystem}) =
    (described=(:layout, :equations, :time_scale, :newton_tolerances, :time_step_tolerances),
     derived=(:sparse_terms, :matrix_free_terms, :explicit_terms, :correction_terms, :direct_terms, :column_terms,
              :automatic_jacobian, :automatic_positions, :stored_jacobian_exact,
              :linear, :mass, :mass_columns, :row_scales, :pattern, :sparse_rhs!, :preparation, :backend, :prototype, :time, :buffer,
              :product))

function describe(tolerance::Tolerance)
    record = SettingsRecord("absolute" => tolerance.absolute, "relative" => tolerance.relative)
    # recorded only when set, so that the records of other tolerances keep their content
    iszero(tolerance.group) || (record["group"] = tolerance.group)
    return record
end

"""
    describe(system::EquationSystem, context=DescriptionContext())

Record of the system: per unknown the kind of its equation and its terms (`term_1`,
`term_2`, … in the order given), the time scale, and the Newton and time-step tolerances
of every unknown as they apply (the defaults filled in). The layout has its own record.
"""
function describe(system::EquationSystem, context::DescriptionContext=DescriptionContext())
    equations = SettingsRecord()
    for equation in system.equations
        terms = SettingsRecord("term_$k" => describe(term, context) for (k, term) in enumerate(equation.terms))
        record = SettingsRecord("kind" => equation.kind isa TimeDerivative ? "time_derivative" :
                                          equation.kind isa Constraint ? "constraint" : "algebraic",
                                "term_count" => length(equation.terms), "terms" => terms)
        # the unknown whose time derivative the row holds, when it is another one
        equation.kind isa TimeDerivative && equation.kind.of !== nothing &&
            (record["time_derivative_of"] = String(equation.kind.of))
        equation.scale === nothing || (record["scale"] = equation.scale)
        equation.initial_value === :given || (record["initial_value"] = String(equation.initial_value))
        equations[String(equation.unknown)] = record
    end
    tolerances(kind) = SettingsRecord(String(u.name) => describe(tolerance(system, u.name; kind))
                                      for u in system.layout.unknowns)
    return SettingsRecord("type" => "EquationSystem", "equations" => equations,
                          "time_scale" => system.time_scale, "newton_tolerances" => tolerances(:newton),
                          "time_step_tolerances" => tolerances(:time_step))
end

describe(solver::LinearSolver) = _describe_fields(solver)
describe(preconditioner::BlockPreconditioner) = _describe_fields(preconditioner)
describe(solver::NewtonSolver) = _describe_fields(solver)

# ---------------------------------------------------------------------------------------
# code version and environment

const _CODE_VERSION = Ref{Union{Nothing,SettingsRecord}}(nothing)

"""
    code_version()

Record of the code a run uses: the Runes2P package version, the commit of the Runes2P checkout
with its branch, whether the tracked files of the checkout were unchanged (`tree_clean`),
and the Julia version. The checkout is read once per session, at the first call. Outside
a checkout (an installed package) the commit is `"unknown"` and `commit_known` is false.
"""
function code_version()
    cached = _CODE_VERSION[]
    cached === nothing || return deepcopy(cached)
    version = pkgversion(@__MODULE__)
    record = SettingsRecord("package" => "Runes2P", "package_version" => version === nothing ? "unknown" : string(version),
                            "commit" => "unknown", "commit_known" => false, "tree_clean" => false,
                            "branch" => "unknown", "julia_version" => string(VERSION))
    root = pkgdir(@__MODULE__)
    if root !== nothing
        try
            repository = LibGit2.GitRepo(root)
            try
                record["commit"] = string(LibGit2.head_oid(repository))
                record["commit_known"] = true
                record["tree_clean"] = !LibGit2.isdirty(repository)
                record["branch"] = LibGit2.headname(repository)
            finally
                close(repository)
            end
        catch
            # not a checkout, or one the library cannot read: the record says unknown
        end
    end
    _CODE_VERSION[] = record
    return deepcopy(record)
end

const _CPU_MODEL = Ref{Union{Nothing,String}}(nothing)

# Read once per session: `Sys.cpu_info()` queries every core and takes a noticeable time on a
# machine with many cores, which every call of `integrate` would pay.
function _cpu_model()
    cached = _CPU_MODEL[]
    cached === nothing || return cached
    model = try
        String(first(Sys.cpu_info()).model)
    catch
        "unknown"
    end
    _CPU_MODEL[] = model
    return model
end

"""
    run_environment()

Record of where a run runs: host, CPU model, thread counts of the machine, Julia and
BLAS, the BLAS libraries, the start time, and the Slurm job when there is one. It is not
part of the setup hash. The CPU model is read once per session.
"""
function run_environment()
    libraries = join((basename(library.libname) for library in BLAS.get_config().loaded_libs), ", ")
    record = SettingsRecord("hostname" => gethostname(), "cpu_model" => _cpu_model(),
                            "cpu_threads" => Sys.CPU_THREADS, "julia_threads" => Threads.nthreads(),
                            "blas_threads" => BLAS.get_num_threads(),
                            "blas_libraries" => isempty(libraries) ? "unknown" : libraries,
                            "start_time" => Libc.strftime("%Y-%m-%dT%H:%M:%S%z", time()))
    for name in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_RESTART_COUNT")
        value = get(ENV, name, "")
        isempty(value) || (record[lowercase(name)] = value)
    end
    return record
end

# ---------------------------------------------------------------------------------------
# hashes

const LITTLE_ENDIAN = ENDIAN_BOM == 0x04030201
_write_numbers(io, x::Array) = LITTLE_ENDIAN ? write(io, x) : write(io, htol.(x))

function _canonical!(io::IO, record::AbstractDict)
    write(io, UInt8('d'), htol(Int64(length(record))))
    for key in sort!(collect(keys(record)))
        _canonical!(io, String(key))
        _canonical!(io, record[key])
    end
    return io
end
_canonical!(io::IO, x::AbstractString) = (write(io, UInt8('s'), htol(Int64(ncodeunits(x)))); write(io, x); io)
_canonical!(io::IO, x::Bool) = (write(io, UInt8('b'), UInt8(x)); io)
_canonical!(io::IO, x::Integer) = (write(io, UInt8('i'), htol(Int64(x))); io)
_canonical!(io::IO, x::AbstractFloat) = (write(io, UInt8('f'), htol(reinterpret(UInt64, Float64(x)))); io)
function _canonical!(io::IO, x::AbstractArray{<:Real})
    write(io, UInt8(eltype(x) <: Integer ? 'I' : 'F'), htol(Int64(ndims(x))))
    foreach(n -> write(io, htol(Int64(n))), size(x))
    _write_numbers(io, eltype(x) <: Integer ? Array{Int64}(x) : Array{Float64}(x))
    return io
end
function _canonical!(io::IO, x::AbstractVector{<:AbstractString})
    write(io, UInt8('S'), htol(Int64(length(x))))
    foreach(s -> _canonical!(io, s), x)
    return io
end

"""
    settings_hash(record)

SHA-256 of a plain record, as 64 hexadecimal digits. The hash does not depend on the
order of the keys. It covers the type of every value (an integer 1 and a floating-point
1.0 differ) and the bits of floating-point numbers, so records read back from a file hash
as the records written.
"""
function settings_hash(record::AbstractDict)
    io = IOBuffer()
    _canonical!(io, record)
    return bytes2hex(sha256(take!(io)))
end

# ---------------------------------------------------------------------------------------
# the record of a run

"Sections of the settings that define the setup of a run, covered by `setup_hash`."
const SETUP_SECTIONS = ("grids", "layout", "system", "updates", "prescribed", "solver", "time", "run")

_has_undescribed(record::AbstractDict) =
    get(record, "described", true) === false || any(_has_undescribed, values(record))
_has_undescribed(value) = false

function _describe_time(times, save_times, stepper)
    record = SettingsRecord("time_count" => length(times), "save_count" => length(save_times),
                            "stepper" => describe(stepper))
    isempty(times) || (record["times"] = collect(Float64, times))
    isempty(save_times) || (record["save_times"] = collect(Float64, save_times))
    return record
end

"""
    run_settings(system; solver=NewtonSolver(), stepper=FixedSteps(), times=Float64[],
                 save_times=times, description=Dict(), updates=(), checkpoint=nothing,
                 stop=nothing, log=nothing)
    run_settings(system, times, options::RunOptions)

The record of a run, written by [`write_output`](@ref) as the `settings` group. Sections:

- `grids`: the kinetic grids by name, with their faces;
- `layout`: the unknowns, their order, sizes, scales and lower limits;
- `system`: the equations with every term and its options, the tolerances;
- `updates`: the interval updates by name ([`IntervalUpdate`](@ref));
- `prescribed`: the tables of the prescribed functions of time, sampled at the time
  points ([`describe_prescribed_plasma`](@ref)), which the terms refer to by name. They
  are a section of their own so that the hash of `system` does not change with the time
  points;
- `solver`: the Newton and linear solver settings;
- `time`: the time points, the save times and the stepper ([`TimeStepper`](@ref);
  `max_step_halvings = n` stands for `stepper = FixedSteps(max_halvings=n)`);
- `run`: `description`, what the caller adds (a name, the origin of the initial state,
  the parameters a hand-built coefficient came from), as plain values;
- `code`: [`code_version`](@ref), with the commit of Runes2P;
- `environment`: [`run_environment`](@ref);
- `checkpoint`: the checkpoint settings, or the type `none`;
- `stop`: the stop condition, or the type `none`;
- `log`: the settings of the progress log, or the type `none`;
- `hashes`: [`settings_hash`](@ref) of each of the first eight sections, and
  `setup_hash` of the eight together. `code`, `environment`, `checkpoint`, `stop` and
  `log` are outside the hashes, so the same setup has the same hash on another machine
  and commit;
- `complete`: false when some object of the run has no record of its own (a term defined
  outside the package, a prescribed function that fails at a time point).

[`integrate`](@ref) builds the record at its start and returns it in the solution, with
the record of the segment added under `segment`: its number, its start time, the number
of states saved before it, its status (`running`, `stopped` or `finished`) with the time
reached, the intervals done and the reason of a stop, and for a resumed segment the file
it resumed from, the commit and setup hash of the run that wrote it and the sections whose
hash changed.
"""
function run_settings(system::EquationSystem; solver::NewtonSolver=NewtonSolver(),
                      times::AbstractVector=Float64[], save_times::AbstractVector=times,
                      stepper=nothing, max_step_halvings::Union{Nothing,Integer}=nothing,
                      description=SettingsRecord(),
                      updates=(), checkpoint=nothing, stop=nothing, log=nothing,
                      max_table_bytes::Integer=MAX_PRESCRIBED_TABLE_BYTES)
    if max_step_halvings !== nothing
        stepper === nothing || throw(ArgumentError("give max_step_halvings or stepper, not both"))
        stepper = FixedSteps(; max_halvings=max_step_halvings)
    end
    stepper === nothing && (stepper = FixedSteps())
    context = DescriptionContext(; times, max_table_bytes)
    record = SettingsRecord()
    record["layout"] = describe(system.layout, context)
    record["system"] = describe(system, context)
    record["updates"] = SettingsRecord(update_name(update) => describe(update, context)
                                       for update in _checked_updates(updates))
    # after the layout, the terms and the updates have named their grids and tables
    record["grids"] = describe(context)
    record["prescribed"] = describe_tables(context)
    record["solver"] = describe(solver)
    record["time"] = _describe_time(times, save_times, stepper)
    record["run"] = _plain(description)
    record["complete"] = !_has_undescribed(record)
    record["hashes"] = SettingsRecord(section => settings_hash(record[section]) for section in SETUP_SECTIONS)
    record["setup_hash"] = settings_hash(SettingsRecord(section => record[section] for section in SETUP_SECTIONS))
    record["code"] = code_version()
    record["environment"] = run_environment()
    record["checkpoint"] = checkpoint === nothing ? SettingsRecord("type" => "none") : describe(checkpoint)
    record["stop"] = stop === nothing ? SettingsRecord("type" => "none") : describe(stop)
    record["log"] = log === nothing ? SettingsRecord("type" => "none") : describe(log)
    return record
end

"""
    read_settings(filename)

The `settings` group of an output file as a plain record, equal to the record written
([`run_settings`](@ref)); empty for a file without the group. Defined by the package
extension `Runes2PHDF5Ext`, which loads with HDF5 (`using HDF5`).
"""
function read_settings end
