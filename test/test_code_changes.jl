# The answer of scripts/code_changes.jl, which decides whether a branch changes code: the
# Component tests and the Full suite of CI are skipped when it answers `code=false`, so a
# wrong `false` would merge a code change untested. The script runs on a scratch git
# repository, one file per rule of its header:
#
# - branch `nocode`: every kind of change that is not code, with the default reference
#   `origin/dev`, which has moved on since the fork;
# - branch `code`: the same plus one file per kind of change that is code.
#
# Each file of `code` must be printed and each file of `nocode` must not, so a rule removed
# from the script shows up as a file in the wrong list. The repository and the script run
# with neither the global nor the system git configuration, and without the configuration,
# author and committer that environment variables would pass to git.

using Test

const SCRIPT = joinpath(dirname(@__DIR__), "scripts", "code_changes.jl")

# the file of `DOCSTRINGS_READ_BY_TESTS` in the script
const DOCSTRING_READ_BY_A_TEST = "src/physics/ionization_chianti_fits.jl"

function git(repo, arguments...)
    identity_flags = ["-c", "user.name=Test", "-c", "user.email=test@example.com",
        "-c", "commit.gpgsign=false"]
    run(Cmd(Cmd(["git", identity_flags..., arguments...]); dir = repo))
    return nothing
end

function put(repo, path, text)
    mkpath(dirname(joinpath(repo, path)))
    write(joinpath(repo, path), text)
    return nothing
end

function commit(repo, message)
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message", message)
    return nothing
end

# the script run in `repo` with `arguments`, `GITHUB_OUTPUT` set to `output` (unset for
# nothing): whether it succeeded, the lines it printed and its error output
function code_changes(repo, arguments...; output = nothing)
    printed = IOBuffer()
    errors = IOBuffer()
    command = Cmd(`$(Base.julia_cmd()) --startup-file=no $SCRIPT $(collect(arguments))`;
        dir = repo)
    process = withenv("GITHUB_OUTPUT" => output) do
        run(pipeline(ignorestatus(command); stdout = printed, stderr = errors))
    end
    lines = split(String(take!(printed)), '\n'; keepempty = false)
    return (; success = success(process), lines, errors = String(take!(errors)))
end

# the files at the fork of both branches
const FORK = Dict(
    "README.md" => "# Package\n\nThe first line.\n",
    "docs/src/example.jl" => "x = 1\n",
    "docs/make.jl" => "# the build\nmakedocs()\n",
    "src/docstring.jl" => "\"\"\"\n    f(x)\n\nThe input.\n\"\"\"\nf(x) = x\n",
    "src/comment.jl" => "# twice the input\ng(x) = 2x\n",
    "src/layout.jl" => "a = 1\n@inline h(x) = x + 1\nk(x) = max(x, 0)\n",
    "src/raw_docstring.jl" => "@doc raw\"\"\"\n    r(x)\n\nThe input, \\alpha.\n\"\"\"\nr(x) = x\n",
    "src/doc_macro.jl" => "@doc \"The value.\" v() = 1\n",
    "src/moved_on.jl" => "# the offset\noffset() = 1\n",
    "ext/Ext.jl" => "module Ext\n# the extension\nend\n",
    "scripts/tool.jl" => "\"The tool.\"\ntool() = nothing\n",
    "test/test_a.jl" => "\"\"\"\n    helper()\n\nA helper.\n\"\"\"\nhelper() = 1\n",
    "Project.toml" => "# the package\nname = \"Package\"\n",
    "data/table.toml" => "# the table\nvalue = 1.0\n",
    ".github/workflows/CI.yml" => "# the checks\nname: CI\non: push\n",
    "src/deleted.jl" => "d() = 1\n",
    "src/renamed.jl" => "n() = 1\n",
    "src/broken.jl" => "# a broken file\nb(x = \n",
    "src/interpolated.jl" => "const NAME = \"p\"\n\"\"\"\n    i(x)\n\nThe input of \$NAME.\n\"\"\"\ni(x) = x\n",
    "src/field.jl" => "struct S\n    \"the first field\"\n    a::Int\nend\n",
    DOCSTRING_READ_BY_A_TEST => "\"The fits.\"\nconst FITS = 1\n",
    "src/mixed.jl" => "# the scale\nscale() = 1\n")

# no code: markdown, docs/src/, and comments, docstrings and layout of Julia files outside
# test/; src/moved_on.jl has a comment change here and a code change on `origin/dev`
const NO_CODE = Dict(
    "README.md" => "# Package\n\nThe first line, changed.\n",
    "docs/src/example.jl" => "x = 2\n",
    "docs/make.jl" => "# the build of the docs\nmakedocs()\n",
    "src/docstring.jl" => "\"\"\"\n    f(x)\n\nThe input, unchanged.\n\"\"\"\nf(x) = x\n",
    "src/comment.jl" => "# two times the input\ng(x) = 2x\n",
    "src/layout.jl" => "a = 1\n\n\n@inline h(x) = x + 1\nk(x) = max(x,\n    0)\n",
    "src/raw_docstring.jl" => "@doc raw\"\"\"\n    r(x)\n\nThe input, \\beta.\n\"\"\"\nr(x) = x\n",
    "src/doc_macro.jl" => "@doc \"The constant value.\" v() = 1\n",
    "src/moved_on.jl" => "# the offset of the grid\noffset() = 1\n",
    "ext/Ext.jl" => "module Ext\n# the extension of the package\nend\n",
    "scripts/tool.jl" => "\"The tool of the package.\"\ntool() = nothing\n")

# code: test/, non-Julia files outside docs/src/ (a comment change, which the parse of
# Project.toml and of the data file as Julia would drop), a Julia file that does not parse,
# an interpolating docstring, the docstring of a field, a file whose docstring a test reads,
# and code next to a comment change
const CODE = Dict(
    "test/test_a.jl" => "\"\"\"\n    helper()\n\nA helper of the tests.\n\"\"\"\nhelper() = 1\n",
    "Project.toml" => "# the package itself\nname = \"Package\"\n",
    "data/table.toml" => "# the table of values\nvalue = 1.0\n",
    ".github/workflows/CI.yml" => "# the checks of a push\nname: CI\non: push\n",
    "src/added.jl" => "added() = 1\n",
    "src/broken.jl" => "# a broken file, still broken\nb(x = \n",
    "src/interpolated.jl" => "const NAME = \"p\"\n\"\"\"\n    i(x)\n\nThe value of \$NAME.\n\"\"\"\ni(x) = x\n",
    "src/field.jl" => "struct S\n    \"the only field\"\n    a::Int\nend\n",
    DOCSTRING_READ_BY_A_TEST => "\"The fits of the rates.\"\nconst FITS = 1\n",
    "src/mixed.jl" => "# the scale factor\nscale() = 2\n")

# the files of `code` that change code: those of CODE, the deleted file, and both sides of
# the rename, which the script sees as a deletion and an addition
const CODE_FILES = sort(vcat(collect(keys(CODE)),
    ["src/deleted.jl", "src/renamed.jl", "src/renamed_to.jl"]))

@testset "code_changes.jl" begin
    mktempdir() do scratch
        config = joinpath(scratch, "gitconfig")
        touch(config)
        repo = joinpath(scratch, "repo")
        mkpath(repo)
        # git reads GIT_CONFIG_KEY_n and GIT_CONFIG_VALUE_n only below GIT_CONFIG_COUNT, and
        # the author and committer of the commits come from the user.* settings of `git`
        no_identity = ["GIT_$(role)_$(part)" => nothing
                       for role in ("AUTHOR", "COMMITTER")
                       for part in ("NAME", "EMAIL", "DATE")]
        withenv("GIT_CONFIG_GLOBAL" => config, "GIT_CONFIG_NOSYSTEM" => "1",
            "GIT_CONFIG_COUNT" => nothing, "GIT_CONFIG_PARAMETERS" => nothing,
            "GIT_DIR" => nothing, "GIT_WORK_TREE" => nothing,
            "GIT_INDEX_FILE" => nothing, no_identity...) do
            git(repo, "init", "--quiet", "--initial-branch=fork")
            for (path, text) in FORK
                put(repo, path, text)
            end
            commit(repo, "fork")

            # `origin/dev` moves on with a code change of src/moved_on.jl and a new file
            git(repo, "checkout", "--quiet", "-b", "upstream")
            put(repo, "src/moved_on.jl", "# the offset\noffset() = 2\n")
            put(repo, "src/upstream.jl", "upstream() = 1\n")
            commit(repo, "upstream")
            git(repo, "update-ref", "refs/remotes/origin/dev", "upstream")

            git(repo, "checkout", "--quiet", "-b", "nocode", "fork")
            for (path, text) in NO_CODE
                put(repo, path, text)
            end
            commit(repo, "no code")

            git(repo, "checkout", "--quiet", "-b", "code")
            for (path, text) in CODE
                put(repo, path, text)
            end
            git(repo, "rm", "--quiet", "src/deleted.jl")
            git(repo, "mv", "src/renamed.jl", "src/renamed_to.jl")
            commit(repo, "code")

            @testset "no code against origin/dev" begin
                git(repo, "checkout", "--quiet", "nocode")
                output = joinpath(scratch, "output_nocode")
                write(output, "earlier=1\n")
                result = code_changes(repo; output)
                @test result.success
                @test result.lines == ["no file changes code"]
                # appended to the file, not written over it
                @test read(output, String) == "earlier=1\ncode=false\n"
            end

            @testset "code against origin/dev" begin
                git(repo, "checkout", "--quiet", "code")
                output = joinpath(scratch, "output_code")
                result = code_changes(repo; output)
                @test result.success
                @test sort(result.lines) == ["changes code: " * path for path in CODE_FILES]
                @test read(output, String) == "code=true\n"
            end

            @testset "another reference, without GITHUB_OUTPUT" begin
                # against `nocode`, the branch `code` changes the files of CODE_FILES only
                result = code_changes(repo, "nocode")
                @test result.success
                @test sort(result.lines) == ["changes code: " * path for path in CODE_FILES]
                result = code_changes(repo, "code")
                @test result.success
                @test result.lines == ["no file changes code"]
            end

            @testset "errors" begin
                output = joinpath(scratch, "output_errors")
                touch(output)
                result = code_changes(repo, "nocode", "fork"; output)
                @test !result.success
                @test occursin("ArgumentError", result.errors)
                @test occursin("at most one reference", result.errors)
                result = code_changes(repo, "no-such-reference"; output)
                @test !result.success
                @test occursin("failed process", result.errors)
                @test isempty(read(output, String))
            end
        end
    end
end
