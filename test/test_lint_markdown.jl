# The markdown lint of `scripts/lint_markdown.py` on small files written here: a period after
# an abbreviation of its list does not end a sentence, and a word such as "approval.", which
# ends like "al.", keeps its period. The expected findings are worked out by applying the
# rules of the section "Markdown" of CONTRIBUTING.md by hand to each line.
using Test

const LINT = joinpath(dirname(@__DIR__), "scripts", "lint_markdown.py")

# the findings of the lint on one file as (line, rule) pairs, and its exit code; the script
# checks files below the folder above its own, so a copy of it runs in `directory`
function lint_findings(directory, name, text)
    path = joinpath(directory, name)
    write(path, text)
    script = joinpath(directory, "scripts", "lint_markdown.py")
    output = IOBuffer()
    process = run(pipeline(ignorestatus(`python3 $script $path`); stdout = output))
    findings = Tuple{Int, String}[]
    for line in eachline(IOBuffer(take!(output)))
        found = match(r":(\d+): (\w+): ", line)
        if found !== nothing
            push!(findings, (parse(Int, found[1]), found[2]))
        end
    end
    return findings, process.exitcode
end

@testset "markdown lint of abbreviations" begin
    if Sys.which("python3") === nothing
        @info "no python3 on the path: the test of scripts/lint_markdown.py is skipped"
        @test_skip false
    else
        mktempdir() do directory
            mkpath(joinpath(directory, "scripts"))
            cp(LINT, joinpath(directory, "scripts", "lint_markdown.py"))
            # a sentence that ends in "approval." at the end of a line ends there
            findings, code = lint_findings(directory, "approval.md",
                "The change waits for the approval.\nThe next sentence starts on its own line.\n")
            @test isempty(findings)
            @test code == 0

            text = """
            The rate follows Smith et al. Their values are used here.
            The rate follows the approval. Their values are used here.
            The rates agree, e.g. Smith finds the same values.
            The rates agree, _e.g._ Smith finds the same values.
            The rates agree with Fig. A and with Figs. B and C.
            The values are taken from the journal. Their errors are not.
            The rate follows Smith et al.
            in its second part.
            """
            findings, code = lint_findings(directory, "abbreviations.md", text)
            # lines 1 and 3 to 5: an abbreviation before a capital does not end the sentence,
            # also in emphasis (line 4) and where two of them share a prefix (line 5);
            # lines 2 and 6: "approval." and "journal." do; line 7: the sentence after
            # "et al." goes on in line 8
            @test findings == [(2, "sentences"), (6, "sentences"), (7, "wrap")]
            @test code == 1
        end
    end
end
