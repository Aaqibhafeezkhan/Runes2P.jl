# The markdown lint of `scripts/lint_markdown.py` on small files written here. The findings
# of the section "Markdown" of CONTRIBUTING.md: a period after an abbreviation of its list
# does not end a sentence, and a word such as "approval.", which ends like "al.", keeps its
# period. The warnings of the section "Writing": each rule on a sample that breaks it and on
# near misses that must stay quiet (code, math, citations, conditions, equation numbers, the
# alternatives of a table cell). The mode for the text of GitHub, the exit codes and the
# annotations of GitHub Actions. The expected lines are worked out by applying the rules of
# the script's docstring by hand to each line.
using Test

const LINT = joinpath(dirname(@__DIR__), "scripts", "lint_markdown.py")

# the lint run with `arguments` from the copy of the script in `directory`: its findings and
# its warnings as (line, rule) pairs, its exit code and its output. GITHUB_ACTIONS is set by
# hand, since CI sets it for the tests too.
function lint(directory, arguments...; annotate = false, input = nothing)
    script = joinpath(directory, "scripts", "lint_markdown.py")
    command = addenv(`python3 $script $arguments`, "GITHUB_ACTIONS" =>
        annotate ? "true" : "false")
    output = IOBuffer()
    process = run(pipeline(
        ignorestatus(command); stdin = input === nothing ? devnull : IOBuffer(input),
        stdout = output, stderr = devnull))
    text = String(take!(output))
    findings = Tuple{Int, String}[]
    warnings = Tuple{Int, String}[]
    for line in eachline(IOBuffer(text))
        found = match(r":(\d+): (warning: )?(\w+): ", line)
        if found !== nothing
            push!(found[2] === nothing ? findings : warnings, (
                parse(Int, found[1]), found[3]))
        end
    end
    return (; findings, warnings, code = process.exitcode, text)
end

# the lint of one file written into `directory`, below the copy of the script
function lint_text(directory, name, text, arguments...; annotate = false)
    path = joinpath(directory, name)
    write(path, text)
    return lint(directory, arguments..., path; annotate)
end

@testset "markdown lint" begin
    if Sys.which("python3") === nothing
        @info "no python3 on the path: the test of scripts/lint_markdown.py is skipped"
        @test_skip false
    else
        mktempdir() do directory
            # the script checks files below the folder above its own, so a copy of it runs here
            mkpath(joinpath(directory, "scripts"))
            cp(LINT, joinpath(directory, "scripts", "lint_markdown.py"))

            @testset "abbreviations" begin
                # a sentence that ends in "approval." at the end of a line ends there
                result = lint_text(directory, "approval.md",
                    "The change waits for the approval.\nThe next sentence starts on its own line.\n")
                @test isempty(result.findings)
                @test result.code == 0
                @test occursin("0 findings in 1 file\n", result.text)

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
                result = lint_text(directory, "abbreviations.md", text)
                # lines 1 and 3 to 5: an abbreviation before a capital does not end the sentence,
                # also in emphasis (line 4) and where two of them share a prefix (line 5);
                # lines 2 and 6: "approval." and "journal." do; line 7: the sentence after
                # "et al." goes on in line 8
                @test result.findings == [(2, "sentences"), (6, "sentences"), (7, "wrap")]
                # the eight lines are one paragraph of more than five sentences
                @test result.warnings == [(1, "paragraph")]
                @test result.code == 1
            end

            @testset "warnings in prose" begin
                text = """
                The scheme is central; the boundary is fixed.

                The call `build(start; diffusivity = 1000.0)` runs the quench.

                The diffusion is that of A. B. Rechester and M. N. Rosenbluth, Phys. Rev. Lett. 40, 38 (1978); doi:10.1103/PhysRevLett.40.38.

                The weight is ``a; b`` in math, and &amp; is an entity.

                The runs take 1.2 s, 3.4 s and 5.6 s.

                At 120 eV, the time is 6.89e-7 s and the field is 1.030 V/m.

                The forms follow eqs. (2.5e), (2.7) and (4.6), built on 2026-11-12 with v26.5 and 1.13.1.

                The ratios are 1.5e-3, 2.5e-3 and 3.5e-3.

                The rates follow eqs. 2.22, 2.25 and 2.31 of the paper, from 0.1 to 10 MeV.

                The moment is that of a part of the collisions of the term.

                The sum of the moments and of the rates of the grid stays.
                """
                result = lint_text(directory, "prose.md", text)
                # line 1: a semicolon joins two clauses; line 3: in code; line 5: before a
                # citation; line 7: in math and in an entity; line 9: three values in s; line
                # 11: a condition and two values of other units; line 13: equation numbers,
                # a date and versions; line 15: three numbers in scientific notation; line 17:
                # equation numbers and a range; line 19: a chain of three "of"; line 21: "and"
                # breaks the chain
                @test isempty(result.findings)
                @test result.warnings ==
                      [(1, "semicolon"), (9, "values"), (15, "values"), (19, "of_chain")]
                @test result.code == 0
                @test occursin("0 findings in 1 file, 4 warnings\n", result.text)
                # with --strict a warning fails the run
                @test lint(directory, "--strict", joinpath(directory, "prose.md")).code == 1
            end

            @testset "warnings in tables, lists and paragraphs" begin
                long = "the reference data of the comparison tests, one lazy artifact per set, written by the cases of the folder of the cases, with the values of the library of GSL"
                text = """
                | Switch | Values |
                |---|---|
                | `grid=` | `uniform` (default); `biuniform`, `tanh` |
                | `sets=` | $long |
                | `link=` | [short](https://example.org/$(repeat("a", 200))) |

                - `HDF5`: the output files,
                - `SciMLBase`: the SciML form;
                - `Interpolations`: the interpolators.

                One.
                Two.
                Three.
                Four.
                Five.

                One.
                Two.
                Three.
                Four.
                Five.
                Six.
                """
                result = lint_text(directory, "blocks.md", text)
                # line 3: alternatives in a cell; line 4: a cell of more than 150 characters;
                # line 5: a link counts by its text; lines 7 and 8: items that end with "," and
                # ";"; line 11: five sentences; line 17: six
                @test length(long) > 150
                @test isempty(result.findings)
                @test result.warnings ==
                      [(4, "cell"), (7, "list_end"), (8, "list_end"), (17, "paragraph")]
            end

            @testset "the start of a page" begin
                result = lint_text(directory, "list-first.md", "# Title\n\n- A list comes first.\n")
                @test result.warnings == [(3, "page_start")]
                result = lint_text(directory, "meta-first.md",
                    "# Title\n\n```@meta\nCurrentModule = X\n```\n\nThe page holds this.\n")
                @test isempty(result.warnings)
            end

            @testset "text for GitHub" begin
                mktempdir() do outside
                    paragraph = "The change adds a rule. It is tested. Its docs are in CONTRIBUTING.md.\n\n## Tests\n\n- One item. Another sentence.\n"
                    path = joinpath(outside, "pr.md")
                    write(path, paragraph)
                    # a file outside the repository is refused without --github
                    @test lint(directory, path).code == 2
                    # several sentences on a line are the rule there
                    result = lint(directory, "--github", path)
                    @test isempty(result.findings) && isempty(result.warnings)
                    @test result.code == 0
                    # a paragraph broken over lines shows as line breaks on GitHub; a sentence
                    # of the paragraph longer than 200 characters is too long
                    broken = "The change adds a rule\nand tests it.\n\nThe rule; " *
                             repeat("long ", 41) * "ends. Short.\n"
                    write(path, broken)
                    result = lint(directory, "--github", path)
                    @test result.findings == [(1, "wrap"), (4, "length")]
                    @test result.warnings == [(4, "semicolon")]
                    @test result.code == 1
                    # the same text from standard input
                    result = lint(directory, "--github", "-"; input = broken)
                    @test result.findings == [(1, "wrap"), (4, "length")]
                end
            end

            @testset "annotations of GitHub Actions" begin
                result = lint_text(
                    directory, "a,b.md", "The scheme is central; the boundary\nis fixed.\n";
                    annotate = true)
                @test result.findings == [(1, "wrap")]
                @test result.warnings == [(1, "semicolon")]
                @test occursin(
                    "::error file=a%2Cb.md,line=1,title=wrap::the sentence continues on the next line\n",
                    result.text)
                @test occursin(
                    "::warning file=a%2Cb.md,line=1,title=semicolon::a semicolon joins two clauses\n",
                    result.text)
                @test !occursin("::", lint_text(directory, "a,b.md", "The scheme; it is set.\n").text)
            end
        end
    end
end
