# Test suite of Runes2P.
#
# Every test file runs in a module of its own. A file may use Runes2P, Test, `Diagonal`, the
# setups of `example_cases.jl` and the shared helpers of `helpers.jl`, and nothing that
# another test file defines, so that any file and any group runs on its own.
#
# The files are listed in groups in `groups.toml`, which balance the run time. Without
# arguments every group runs. Arguments:
#   <group> ...            the named groups
#   --index <n>            the n-th group of groups.toml
#   --tier quick           the groups of the unit tests (tier = "quick" in groups.toml)
#   --files <file>,...     the named files
#   --results <path>       write a TOML summary of the run
# from `Pkg.test(test_args=[...])` or `julia --project=test test/runtests.jl ...`.
using Runes2P
using Test
using TOML
using LinearAlgebra: Diagonal

const TEST_DIRECTORY = @__DIR__
# the shared helpers, loaded once into Main and imported by every test module
include(joinpath(TEST_DIRECTORY, "helpers.jl"))

# the groups of groups.toml as (name, files) pairs, in the order of the file, and their tiers
const GROUP_TABLE = TOML.parsefile(joinpath(TEST_DIRECTORY, "groups.toml"))["group"]
read_groups() = [(g["name"], String.(g["files"])) for g in GROUP_TABLE]
group_tier(name) = only(g["tier"] for g in GROUP_TABLE if g["name"] == name)

# every test file of the directory: test_*.jl without the helper files they include
function test_files_on_disk()
    sort([f
          for f in readdir(TEST_DIRECTORY)
          if startswith(f, "test_") && endswith(f, ".jl") && !endswith(f, "_helpers.jl")])
end

function parse_arguments(arguments)
    groups, files, results, index, tier = String[], String[], nothing, nothing, nothing
    k = 1
    while k <= length(arguments)
        a = arguments[k]
        if a == "--files"
            append!(files, split(arguments[k + 1], ','; keepempty = false))
            k += 2
        elseif a == "--tier"
            tier = arguments[k + 1]
            k += 2
        elseif a == "--index"
            index = parse(Int, arguments[k + 1])
            k += 2
        elseif a == "--results"
            results = arguments[k + 1]
            k += 2
        else
            push!(groups, a)
            k += 1
        end
    end
    return (; groups, files, results, index, tier)
end

# A fresh module with what every test file may use: Runes2P, Test, `Diagonal`, the helpers and
# the setups of example_cases.jl. `include` inside it resolves paths relative to the
# including file, as at the top level.
function test_module(file)
    m = Module(Symbol(replace(file, ".jl" => "")))
    Core.eval(m, :(using Runes2P, Test))
    Core.eval(m, :(using LinearAlgebra: Diagonal))
    Core.eval(m, :(using Main.TestHelpers))
    Core.eval(m, :(include(path) = Base.include($m, path)))
    Core.eval(m, :(eval(x) = Core.eval($m, x)))
    Base.include(m, joinpath(TEST_DIRECTORY, "example_cases.jl"))
    return m
end

const FILE_TIMES = Pair{String, Float64}[]

function run_file(file)
    elapsed = @elapsed Base.include(test_module(file), joinpath(TEST_DIRECTORY, file))
    push!(FILE_TIMES, file => elapsed)
    total = sum(last, FILE_TIMES)
    println("== ", file, ": ", round(elapsed; digits = 1), " s, ",
        round(total / 60; digits = 1), " minutes so far")
    flush(stdout)
    return nothing
end

function print_file_times()
    total = sum(last, FILE_TIMES; init = 0.0)
    println("Time per test file, ", round(total / 60; digits = 1), " minutes in all:")
    for (file, elapsed) in sort(FILE_TIMES; by = last, rev = true)
        println(rpad(file, 44), lpad(round(elapsed; digits = 1), 8), " s",
            lpad(round(100 * elapsed / total; digits = 1), 7), " %")
    end
end

# passed, failed, errored and broken tests of a finished test set, with its children
function test_counts(ts::Test.DefaultTestSet)
    counts = Test.get_test_counts(ts)
    return (passed = counts.cumulative_passes + counts.passes,
        failed = counts.cumulative_fails + counts.fails,
        errored = counts.cumulative_errors + counts.errors, broken = counts.cumulative_broken +
                                                                     counts.broken)
end

git(arguments...) =
    try
        readchomp(Cmd(`git -C $TEST_DIRECTORY $arguments`))
    catch
        ""
    end

function write_results(path, label, files, counts, elapsed)
    dirty = !isempty(git("status", "--porcelain", "--untracked-files=no"))
    load = isfile("/proc/loadavg") ? join(split(read("/proc/loadavg", String))[1:3], " ") :
           ""
    summary = Dict(
        "label" => label, "commit" => git("rev-parse", "HEAD"), "dirty" => dirty,
        "node" => gethostname(), "load" => load, "julia" => string(VERSION),
        "elapsed_seconds" => elapsed, "files" => files,
        "file_seconds" => Dict(f => t for (f, t) in FILE_TIMES),
        "passed" => counts.passed, "failed" => counts.failed, "errored" => counts.errored,
        "broken" => counts.broken)
    mkpath(dirname(abspath(path)))
    open(io -> TOML.print(io, summary; sorted = true), path, "w")
    return nothing
end

options = parse_arguments(ARGS)
groups = read_groups()
known = Dict(groups)
options.index === nothing || push!(options.groups, first(groups[options.index]))
options.tier === nothing || append!(options.groups, [name
                         for (name, _) in groups if group_tier(name) == options.tier])
options.tier === nothing || !isempty(options.groups) ||
    error("no group of tier $(options.tier) in groups.toml")
unknown = filter(g -> !haskey(known, g), options.groups)
isempty(unknown) ||
    error("unknown test groups $(join(unknown, ", ")); groups.toml has $(join(first.(groups), ", "))")
selected_groups = isempty(options.groups) && isempty(options.files) ? first.(groups) :
                  options.groups
files = vcat([known[g] for g in selected_groups]..., options.files)
label = isempty(options.files) ? join(selected_groups, ",") :
        join(vcat(selected_groups, options.files), ",")

counts, elapsed = let start = time()
    ts = try
        @testset "Runes2P" begin
            # the manifest itself: every test file in exactly one group, every listed file on disk
            @testset "groups.toml" begin
                listed = vcat(last.(groups)...)
                @test sort(listed) == test_files_on_disk()
                @test allunique(listed)
                @test all(g -> get(g, "tier", "") in ("quick", "full"), GROUP_TABLE)
            end
            for file in files
                run_file(file)
            end
            print_file_times()
        end
    catch err
        err isa Test.TestSetException || rethrow()
        err
    end
    counts = ts isa Test.TestSetException ?
             (passed = ts.pass, failed = ts.fail, errored = ts.error, broken = ts.broken) :
             test_counts(ts)
    counts, time() - start
end
options.results === nothing || write_results(options.results, label, files, counts, elapsed)
println("Summary: ", counts.passed, " passed, ", counts.failed,
    " failed, ", counts.errored, " errored, ",
    counts.broken, " broken in ", round(elapsed / 60; digits = 1), " minutes")
# a failing run ends with an error, as Pkg.test expects
counts.failed + counts.errored == 0 ||
    error("Some tests did not pass: $(counts.failed) failed, $(counts.errored) errored.")
