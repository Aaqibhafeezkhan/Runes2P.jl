# Whether a branch changes code, for the checks of CI.yml and FullSuite.yml: the tests run
# only when it does.
#
#     julia scripts/code_changes.jl                the branch against origin/dev
#     julia scripts/code_changes.jl origin/main    the same against another reference
#
# The changes are those of `git diff <base>...HEAD`, the branch since its merge base. A file
# does not change code when it is
#
# - a markdown file (`*.md`);
# - a file under `docs/src/`, which the documentation build checks;
# - a Julia file outside `test/` that is in both trees and whose parsed code is the same in
#   both once its docstrings are removed: the parser drops comments and line numbers, so a
#   change of comments, docstrings or layout does not count.
#
# A Julia file whose docstrings a test reads (`DOCSTRINGS_READ_BY_TESTS`) changes code with
# any change. Every other file changes code, among them a Julia file that is added, deleted
# or renamed, or that does not parse. The script prints the files that change code, and
# writes `code=true` or `code=false` to the file `$GITHUB_OUTPUT` when it is set. The exit
# code is 0 unless git fails. The jobs "Code changes" of CI.yml and FullSuite.yml run it,
# and are kept the same.

# the files whose docstrings a test reads: test_chianti_ionization.jl checks the source that
# the docstring of CHIANTI_IONIZATION_FITS cites
const DOCSTRINGS_READ_BY_TESTS = ["src/physics/ionization_chianti_fits.jl"]

# the expression without docstrings and line numbers; a docstring is the macro call
# `Core.@doc` (or `@doc`) with the string and the documented expression
function _without_docs(x)
    x isa Expr || return x
    if x.head === :macrocall && _is_doc_macro(x.args[1]) && length(x.args) == 4
        return _without_docs(x.args[4])
    end
    args = Any[]
    for (k, arg) in enumerate(x.args)
        # the line number of a macro call is its second argument
        if arg isa LineNumberNode
            if x.head === :macrocall && k == 2
                push!(args, nothing)
            end
            continue
        end
        push!(args, _without_docs(arg))
    end
    return Expr(x.head, args...)
end

function _is_doc_macro(name)
    name == GlobalRef(Core, Symbol("@doc")) || name === Symbol("@doc")
end

# the parsed code of a Julia source without its docstrings, or nothing when it does not parse
function _code_of(text::AbstractString, filename::AbstractString)
    parsed = try
        Meta.parseall(text; filename)
    catch
        return nothing
    end
    has_error(x) = x isa Expr && (x.head in (:error, :incomplete) || any(has_error, x.args))
    has_error(parsed) && return nothing
    return _without_docs(parsed)
end

_git(args...) = read(Cmd(["git", args...]), String)

_file_at(ref, path) = _git("show", "$ref:$path")

function changes_code(base::AbstractString, status::AbstractString, path::AbstractString)
    endswith(path, ".md") && return false
    startswith(path, "docs/src/") && return false
    startswith(path, "test/") && return true
    (status == "M" && endswith(path, ".jl")) || return true
    path in DOCSTRINGS_READ_BY_TESTS && return true
    merge_base = strip(_git("merge-base", base, "HEAD"))
    old = _code_of(_file_at(merge_base, path), path)
    new = _code_of(_file_at("HEAD", path), path)
    return old === nothing || new === nothing || old != new
end

function main(arguments)
    base = isempty(arguments) ? "origin/dev" : only(arguments)
    lines = split(_git("diff", "--name-status", "--no-renames", "$base...HEAD"), '\n';
        keepempty = false)
    code = String[]
    for line in lines
        status, path = split(line, '\t'; limit = 2)
        if changes_code(base, status, path)
            push!(code, path)
        end
    end
    for path in code
        println("changes code: $path")
    end
    isempty(code) && println("no file changes code")
    output = get(ENV, "GITHUB_OUTPUT", "")
    if !isempty(output)
        open(output, "a") do io
            println(io, "code=", !isempty(code))
        end
    end
    return nothing
end

main(ARGS)
