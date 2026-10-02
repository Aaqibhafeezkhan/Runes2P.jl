# API reference of Runes2P, built and viewed locally:
#   julia --project=docs docs/make.jl
#   julia --project=docs -e 'using Revise, LiveServer; servedocs(include_dirs=["src", "ext"], skip_dirs=[joinpath("docs", "src", "api")], launch_browser=true)'
# The build is strict: every exported name needs a docstring, and a broken `@ref` link
# fails it.
using Documenter
using Runes2P
# the extensions load with their trigger packages, and their docstrings are part of the API
using HDF5, Interpolations, SciMLBase

const EXTENSIONS = [Base.get_extension(Runes2P, name)
                    for name in (:Runes2PHDF5Ext, :Runes2PInterpolationsExt, :Runes2PSciMLBaseExt)]
any(isnothing, EXTENSIONS) && error("an extension of Runes2P did not load")

# checkdocs = :exports only checks that the existing docstrings appear on a page: an
# exported name without any docstring passes it unnoticed
let undocumented = Docs.undocumented_names(Runes2P)
    isempty(undocumented) || error("exported names without a docstring: ", join(sort(string.(undocumented)), ", "))
end

const SOURCE = joinpath(dirname(@__DIR__), "src")
# generated pages, gitignored
const API = joinpath(@__DIR__, "src", "api")

# Categories of the API reference: a landing page each, with subpages that group source
# files (paths relative to src/). A source file of a category directory that no subpage
# lists lands on the subpage "Other" of its category, so a new file is never left out.
const CATEGORIES = [
    (slug = "grids", title = "Grids", directory = "grids",
     summary = "Momentum, pitch and radial grids, the kinetic grid that combines them, radial profiles and momentum regions.",
     subpages = [
         ("momentum", "Momentum and pitch grids", ["grids/momentum.jl", "grids/pitch_overlap.jl"]),
         ("radial", "Radial grid and profiles", ["grids/radial.jl", "grids/radial_profiles.jl"]),
         ("kinetic", "Kinetic grid and momentum regions", ["grids/kinetic.jl", "grids/momentum_region.jl"]),
     ]),
    (slug = "physics", title = "Physics", directory = "physics",
     summary = "The plasma state, the Coulomb logarithm, collision frequencies, conductivity, runaway rates and the physical constants.",
     subpages = [
         ("plasma", "Plasma state and atomic data", ["physics/plasma.jl", "physics/plasma_source.jl", "physics/atomic_data.jl"]),
         ("collisions", "Collisions", ["physics/coulomb_logarithm.jl", "physics/collision_frequencies.jl"]),
         ("conductivity", "Conductivity and runaway rates", ["physics/conductivity.jl", "physics/runaway_rates.jl"]),
         ("constants", "Physical constants", ["physics/constants.jl"]),
     ]),
    (slug = "operators", title = "Operators", directory = "operators",
     summary = "The flux-form advection–diffusion operator and the physics that fills its coefficients: Fokker–Planck terms, radial transport, knock-on and field–particle collisions.",
     subpages = [
         ("advection_diffusion", "Advection–diffusion operator", ["operators/advection_diffusion.jl", "operators/flux_limiters.jl"]),
         ("fokker_planck", "Fokker–Planck coefficients", ["operators/kinetic_terms.jl"]),
         ("radial_transport", "Radial transport", ["operators/radial_transport.jl"]),
         ("knockon", "Knock-on collisions", ["operators/knockon/kinematics.jl", "operators/knockon/energy_kernel.jl",
                                             "operators/knockon/pitch_kernel.jl", "operators/knockon/operator.jl",
                                             "operators/knockon/cutoff.jl"]),
         ("field_particle", "Field–particle collisions", ["operators/field_particle.jl"]),
     ]),
    (slug = "system", title = "Equation system", directory = "system",
     summary = "The layout of the unknowns, the equation terms, the equations and the system `M du/dt = rhs(u, t)` with its Jacobians.",
     subpages = [
         ("layout", "Unknown layout", ["system/layout.jl"]),
         ("terms", "Equation terms", ["system/terms.jl", "system/interface_term.jl", "system/transport_terms.jl",
                                      "system/population_terms.jl", "system/energy_terms.jl"]),
         ("conductivity_induction", "Conductivity and induction", ["system/conductivity_terms.jl", "system/induction_terms.jl"]),
         ("equation_system", "Equations and the system", ["system/equation_system.jl"]),
         ("jacobian", "Jacobians", ["system/jacobian.jl"]),
     ]),
    (slug = "solvers", title = "Solvers and runs", directory = "solvers",
     summary = "Linear solvers and preconditioners, Newton, time stepping, updates between intervals, checkpoints, the progress log and the settings of a run.",
     subpages = [
         ("linear_newton", "Linear solvers and Newton", ["solvers/linear_solver.jl", "solvers/krylov.jl",
                                                         "solvers/fluid_schur_preconditioner.jl", "solvers/newton.jl"]),
         ("time_stepping", "Time stepping", ["solvers/time_stepping.jl"]),
         ("interval_updates", "Updates between intervals", ["solvers/interval_updates.jl", "solvers/transport_update.jl"]),
         ("checkpoints", "Checkpoints and progress log", ["solvers/checkpoint.jl", "solvers/progress_log.jl"]),
         ("settings", "Run settings", ["settings.jl"]),
     ]),
]

source_files(directory) = sort([relpath(joinpath(root, file), SOURCE)
                                for (root, _, files) in walkdir(joinpath(SOURCE, directory))
                                for file in files if endswith(file, ".jl")])

# Number of docstrings of Runes2P per source file, exported and internal, so that a page
# leaves out empty sections. Documenter's `Public` makes the same test of the binding.
# Methods of functions of other modules (`Base.size(::KineticGrid)`) are documented in Runes2P
# but not listed by `@autodocs`; they get a `@docs` block of their own, by signature.
function docstring_counts()
    counts = Dict{String,Vector{Int}}()
    foreign = Dict{String,Vector{String}}()
    for (binding, multidoc) in Docs.meta(Runes2P), docstring in values(multidoc.docs)
        path = get(docstring.data, :path, nothing)
        path === nothing && continue
        file = relpath(realpath(path), realpath(SOURCE))
        if binding.mod !== Runes2P
            arguments = join(("::$(type)" for type in docstring.data[:typesig].parameters), ", ")
            push!(get!(foreign, file, String[]), "$(binding.mod).$(binding.var)($(arguments))")
            continue
        end
        entry = get!(counts, file, [0, 0])
        entry[Base.isexported(binding.mod, binding.var) ? 1 : 2] += 1
    end
    return counts, foreign
end
const COUNTS, FOREIGN = docstring_counts()

autodocs_block(file, public) = """
    ```@autodocs
    Modules = [Runes2P]
    Pages = ["$(file)"]
    Public = $(public)
    Private = $(!public)
    ```

    """

# a file without docstrings (a data table) gets no heading
function file_section(file)
    exported, internal = get(COUNTS, file, [0, 0])
    methods = get(FOREIGN, file, String[])
    exported + internal + length(methods) == 0 && return ""
    text = "## `src/$(file)`\n\n"
    exported > 0 && (text *= "### Exported\n\n" * autodocs_block(file, true))
    isempty(methods) || (text *= "### Methods of other modules\n\n```@docs\n" * join(methods, "\n") * "\n```\n\n")
    internal > 0 && (text *= "### Internal\n\n" * autodocs_block(file, false))
    return text
end

# the submodule of physical constants has a docstring of its own and no exports
const CONSTANTS_SECTION = """
    ## `Runes2P.Constants`

    ```@autodocs
    Modules = [Runes2P.Constants]
    ```

    """

function write_page(path, text)
    mkpath(dirname(path))
    write(path, text)
    return nothing
end

function write_subpage(category, slug, title, files)
    text = "# $(title)\n\n```@meta\nCurrentModule = Runes2P\n```\n\n"
    category.slug == "physics" && slug == "constants" && (text *= CONSTANTS_SECTION)
    text *= join(file_section.(files))
    write_page(joinpath(API, category.slug, "$(slug).md"), text)
    return title => "api/$(category.slug)/$(slug).md"
end

function write_category(category)
    listed = reduce(vcat, (files for (_, _, files) in category.subpages))
    for file in listed
        isfile(joinpath(SOURCE, file)) || error("$(file) in the API table does not exist")
    end
    unlisted = setdiff(source_files(category.directory), listed)
    subpages = isempty(unlisted) ? category.subpages : [category.subpages; ("other", "Other", unlisted)]
    pages = [write_subpage(category, slug, title, files) for (slug, title, files) in subpages]

    links = join(("- [$(title)]($(category.slug)/$(slug).md): " *
                  join(("`src/$(file)`" for file in files), ", ") for (slug, title, files) in subpages), "\n")
    write_page(joinpath(API, "$(category.slug).md"), "# $(category.title)\n\n$(category.summary)\n\n$(links)\n")
    return category.title => ["Overview" => "api/$(category.slug).md", pages...]
end

rm(API; recursive = true, force = true)
const API_PAGES = [write_category(category) for category in CATEGORIES]

# every source file with docstrings is on a page (the module docstring of Runes2P.jl is on
# the home page)
let listed = reduce(vcat, [source_files(category.directory) for category in CATEGORIES]; init = ["settings.jl", "Runes2P.jl"])
    files = union(keys(COUNTS), keys(FOREIGN))
    missing_files = setdiff(filter(file -> !startswith(file, ".."), collect(files)), listed)
    isempty(missing_files) || error("source files with docstrings on no page: ", join(sort(missing_files), ", "))
end

makedocs(;
    sitename = "Runes2P",
    modules = [Runes2P, Runes2P.Constants, EXTENSIONS...],
    format = Documenter.HTML(; prettyurls = false, size_threshold = 200 * 2^10),
    pages = ["Home" => "index.md", "API" => [API_PAGES..., "Extensions" => "extensions.md"]],
    checkdocs = :exports,
    warnonly = false,
)
