# The markdown lint of `scripts/lint_markdown.py` on small files written here. The findings
# of the section "Markdown" of CONTRIBUTING.md: a period after an abbreviation of its list
# does not end a sentence, and a word such as "approval.", which ends like "al.", keeps its
# period. A sentence that starts with an issue reference ends the one before, and a line
# that starts with #n is text, not a heading. The warnings of the section "Writing": each
# rule on a sample that breaks it and on near misses that must stay quiet (code, math,
# citations, conditions, equation numbers, the alternatives of a table cell). Each near miss
# of the rule values is a sentence that would warn without its exemption. The mode for the
# text of GitHub, the exit codes, the lines that a branch adds and the annotations of GitHub
# Actions. The expected lines are worked out by applying the rules of the script's docstring
# by hand to each line.
using Test

const LINT = joinpath(dirname(@__DIR__), "scripts", "lint_markdown.py")
# git apart from the configuration of the user: no global or system file, so that no hook,
# signing key or default branch of the machine applies
const GIT_ENV = ("GIT_CONFIG_GLOBAL" => "/dev/null", "GIT_CONFIG_NOSYSTEM" => "1")

# the lint run with `arguments` from the copy of the script in `directory`: its findings and
# its warnings as (line, rule) pairs, its exit code, its output and its messages on stderr.
# GITHUB_ACTIONS is set by hand, since CI sets it for the tests too.
function lint(directory, arguments...; annotate = false, input = nothing)
    script = joinpath(directory, "scripts", "lint_markdown.py")
    environment = ("GITHUB_ACTIONS" => annotate ? "true" : "false", GIT_ENV...)
    command = addenv(`python3 $script $arguments`, environment...)
    output = IOBuffer()
    errors = IOBuffer()
    process = run(pipeline(
        ignorestatus(command); stdin = input === nothing ? devnull : IOBuffer(input),
        stdout = output, stderr = errors))
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
    return (; findings, warnings, code = process.exitcode, text,
        message = String(take!(errors)))
end

# git in the repository `directory`, with a name, no signing and a default branch of its own
function git_in(directory, arguments...)
    command = `git -C $directory -c user.name=Test -c user.email=test@example.org
               -c commit.gpgsign=false -c init.defaultBranch=main $arguments`
    run(pipeline(addenv(command, GIT_ENV...); stdout = devnull, stderr = devnull))
    return nothing
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
                # - lines 1 and 3 to 5: an abbreviation before a capital does not end the
                #   sentence, also in emphasis (line 4) and where two of them share a prefix
                #   (line 5)
                # - lines 2 and 6: "approval." and "journal." end it
                # - line 7: the sentence after "et al." goes on in line 8
                @test result.findings == [(2, "sentences"), (6, "sentences"), (7, "wrap")]
                # the eight lines are one paragraph of more than six sentences
                @test result.warnings == [(1, "paragraph")]
                @test result.code == 1
            end

            @testset "sentences that start with an issue reference" begin
                text = "The step is done. #12 is merged.\n" *
                       "The step is done. osrep/Runes2P.jl#12 is merged.\n" *
                       "The values are those of eq. 2.28 of the paper.\n" *
                       "#12 is merged, and " * repeat("long ", 37) * "ends.\n" *
                       "The step is done (it is merged. #12 has it).\n" *
                       "The palette uses red. #1f77b4 is its first color.\n"
                result = lint_text(directory, "references.md", text)
                # - lines 1 and 2: a sentence that starts with a reference ends the one before
                # - line 3: a digit after a period does not, since it follows abbreviations
                #   such as "eq." in citations
                # - line 4: a line that starts with #12 is text, not a heading, and the line
                #   of 209 characters is too long
                # - line 5: the reference starts a second sentence, also inside the parentheses
                # - line 6: a color such as #1f77b4 is no issue reference
                expected = [(1, "sentences"), (2, "sentences"), (4, "length"),
                    (5, "sentences"), (5, "parenthesis")]
                @test result.findings == expected
                @test result.code == 1
                mktempdir() do outside
                    # for GitHub: two sentences of 107 and 104 characters, which the old rule
                    # read as one of 212
                    path = joinpath(outside, "comment.md")
                    write(path,
                        "The step is done, and " * repeat("long ", 16) *
                        "ends. #12 is merged, and " *
                        repeat("long ", 16) * "ends.\n")
                    result = lint(directory, "--github", path)
                    @test isempty(result.findings)
                    @test result.code == 0
                end
            end

            @testset "warnings in prose" begin
                text = """
                The scheme is central; the boundary is fixed.

                The call `build(start; diffusivity = 1000.0)` runs the quench.

                The diffusion is that of A. B. Rechester and M. N. Rosenbluth, Phys. Rev. Lett. 40, 38 (1978); doi:10.1103/PhysRevLett.40.38.

                The weight is ``a; b`` in math, and &amp; is an entity.

                The runs take 1.2 s, 3.4 s and 5.6 s.

                At 10 eV, the losses are 2 eV and 4 eV per step.

                The forms follow (2.5e), (2.7) and (4.6).

                The build uses 1.13.1, 1.12.4 and 1.11.2.

                The ratios are 1.5e-3, 2.5e-3 and 3.5e-3.

                The rates follow eqs. 2.22, 2.25 and 2.31 of the paper.

                The window spans 0.5 s to 1.5 s, and the step takes 2.5 s.

                The moment is that of a part of the collisions of the term.

                The sum of the moments and of the rates of the grid stays.

                The weights are ``a = 1.5``, ``b = 2.5`` and ``c = 3.5`` in math.

                The sum is 1.5 + 2.5 = 4.0 of the total.

                The tolerances are set to 1e-6, 1e-8 and 1e-10.

                The run at 3 eV, 5 eV and 7 eV converges.

                The tests run on Julia 1.10, 1.11 and 1.12.

                The run at 1.5, 2.5 and 3.5 converges.

                Julia 1.10 runs the cases with 1.5, 2.5 and 3.5.

                The code is version 1.10, 1.11 or 1.12.

                In DREAM 1.5 s, 2.5 s and 3.5 s pass.

                Between 0.5 s and 1.5 s the step takes 2.5 s.
                """
                result = lint_text(directory, "prose.md", text)
                # - line 1: a semicolon joins two clauses
                # - line 3: a semicolon in code
                # - line 5: a semicolon before a doi
                # - line 7: semicolons in math and in an entity
                # - line 9: three values in s
                # - line 11: a condition and two values in eV
                # - line 13: numbers alone in parentheses
                # - line 15: versions of three parts
                # - line 17: three numbers in scientific notation
                # - line 19: equation numbers
                # - line 21: the first number of a range and two values in s
                # - line 23: a chain of three "of"
                # - line 25: "and" breaks the chain
                # - line 27: numbers in math
                # - line 29: a formula
                # - line 31: "to" before a list of three values states no range
                # - line 33: a condition word before a list of three values in eV
                # - line 35: a list of versions after "Julia"
                # - line 37: a condition word before a list of three numbers
                # - line 39: the version after "Julia" and three numbers after it
                # - line 41: a list of versions after "version"
                # - line 43: values with a unit after "DREAM" are no versions
                # - line 45: a range "between a and b" and one more value in s
                @test isempty(result.findings)
                @test result.warnings ==
                      [(1, "semicolon"), (9, "values"), (17, "values"), (23, "of_chain"),
                    (31, "values"), (33, "values"), (37, "values"), (39, "values"),
                    (43, "values")]
                @test result.code == 0
                @test occursin("0 findings in 1 file, 9 warnings\n", result.text)
                # with --strict a warning fails the run
                @test lint(directory, "--strict", joinpath(directory, "prose.md")).code == 1
            end

            @testset "semicolons between citations" begin
                text = """
                The scheme is central; the boundary follows Smith (2019).

                The rates are tabulated ([`FITS`](@ref); K. P. Dere, Astronomy and Astrophysics 466, 771 (2007), [doi:10.1051/0004-6361:20066728](https://doi.org/10.1051/0004-6361:20066728)).

                The rates are those of A. B. Rechester and M. N. Rosenbluth, Phys. Rev. Lett. 40, 38 (1978); L. Hesslow et al., Nucl. Fusion 59, 084004 (2019).

                The model follows Smith (2019); the boundary follows Jones (2020).
                """
                result = lint_text(directory, "citations.md", text)
                # - lines 1 and 7: a clause after the semicolon, even with a year in it
                # - line 3: names and a journal up to its volume, page and year after it
                # - line 5: a list of two journal references
                @test isempty(result.findings)
                @test result.warnings == [(1, "semicolon"), (7, "semicolon")]
            end

            @testset "math blocks, link definitions and footnotes" begin
                text = """
                \$\$
                a = b; c
                \$\$

                \$\$ a = b; c \$\$

                [a]: https://example.org/a "A title"
                [b]: <https://example.org/b>

                \$\$a\$\$ is the weight; it holds.

                [^1]: A footnote; it holds.

                [Note]: some text; it holds.
                """
                result = lint_text(directory, "math.md", text)
                # - lines 1 to 8: blocks of math and definitions of links are skipped
                # - line 10: a line that starts with math and goes on is prose
                # - lines 12 and 14: a footnote and a label without a single target are prose
                @test isempty(result.findings)
                @test result.warnings ==
                      [(10, "semicolon"), (12, "semicolon"), (14, "semicolon")]
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
                - The parts:
                  - `A`: the first part.

                One.
                Two.
                Three.
                Four.
                Five.
                Six.

                One.
                Two.
                Three.
                Four.
                Five.
                Six.
                Seven.
                """
                result = lint_text(directory, "blocks.md", text)
                # - line 3: alternatives in a cell
                # - line 4: a cell of more than 150 characters
                # - line 5: a link counts by its text
                # - lines 7, 8 and 10: items that end with ",", ";" and ":"
                # - line 13: a paragraph of six sentences
                # - line 20: a paragraph of seven sentences
                @test length(long) > 150
                @test isempty(result.findings)
                @test result.warnings == [(4, "cell"), (8, "list_end"), (20, "paragraph")]
            end

            @testset "the start of a page" begin
                result = lint_text(directory, "list-first.md", "# Title\n\n- A list comes first.\n")
                @test result.warnings == [(3, "page_start")]
                result = lint_text(directory, "meta-first.md",
                    "# Title\n\n```@meta\nCurrentModule = X\n```\n\nThe page holds this.\n")
                @test isempty(result.warnings)
                # a comment below the title renders nothing
                result = lint_text(directory, "comment-first.md",
                    "# Title\n\n<!-- a note -->\n\nThe page holds this.\n")
                @test isempty(result.warnings)
                # a line that starts with #12 is text, not a heading
                result = lint_text(directory, "reference-first.md",
                    "# Title\n\n#12 holds the rows, and the page holds this.\n")
                @test isempty(result.warnings)
                result = lint_text(directory, "long-comment-first.md",
                    "# Title\n\n<!--\na note\n-->\n\n- A list comes first.\n")
                @test result.warnings == [(7, "page_start")]
                # a line of a code block that starts with "# " is no title
                result = lint_text(directory, "code-first.md",
                    "```bash\n# install\n```\n\nThe page holds this.\n")
                @test isempty(result.warnings)
            end

            @testset "text for GitHub" begin
                mktempdir() do outside
                    description = "The change adds a rule. It is tested. Its docs are in CONTRIBUTING.md.\n\n## Tests\n\n- One item. Another sentence.\n"
                    path = joinpath(outside, "pr.md")
                    write(path, description)
                    # a file outside the repository is refused without --github
                    @test lint(directory, path).code == 2
                    # several sentences on a line are the rule there
                    result = lint(directory, "--github", path)
                    @test isempty(result.findings) && isempty(result.warnings)
                    @test result.code == 0
                    # a paragraph broken over lines shows as line breaks on GitHub: a warning.
                    # A sentence of the paragraph longer than 200 characters is too long.
                    broken = "The change adds a rule\nand tests it.\n\nThe rule; " *
                             repeat("long ", 41) * "ends. Short.\n"
                    write(path, broken)
                    result = lint(directory, "--github", path)
                    @test result.findings == [(4, "length")]
                    @test result.warnings == [(1, "wrap"), (4, "semicolon")]
                    @test result.code == 1
                    # the same text from standard input
                    result = lint(directory, "--github", "-"; input = broken)
                    @test result.findings == [(4, "length")]
                    # a line break alone is a warning: the run passes, and fails with --strict
                    write(path, "The change adds a rule\nand tests it.\n")
                    result = lint(directory, "--github", path)
                    @test isempty(result.findings)
                    @test result.warnings == [(1, "wrap")]
                    @test result.code == 0
                    @test lint(directory, "--strict", "--github", path).code == 1
                    # the rules of "Writing" on a paragraph over several lines: a sentence
                    # that goes on over a line break is checked whole, at its first line
                    write(path, """
                    The scheme is central;
                    the boundary is fixed.

                    The runs take 1.2 s,
                    3.4 s and 5.6 s.

                    The first rule holds
                    for the grid. The second rule holds. The third rule holds.
                    The fourth rule holds. The fifth rule holds. The sixth rule holds.

                    The first rule holds
                    for the grid. The second rule holds. The third rule holds.
                    The fourth rule holds. The fifth rule holds. The sixth rule holds.
                    The seventh rule holds.

                    The rule $(repeat("long ", 25))
                    $(repeat("long ", 20))ends.
                    """)
                    result = lint(directory, "--github", path)
                    # - line 1: the semicolon and the clause after it in line 2
                    # - line 4: three values in s over two lines
                    # - line 7: six sentences over three lines, the first one over two, which
                    #   count as seven line by line
                    # - line 11: seven sentences
                    # - line 16: a sentence of 239 characters over two lines, each under 200
                    @test result.findings == [(16, "length")]
                    wraps = filter(warning -> warning[2] == "wrap", result.warnings)
                    @test first.(wraps) == [1, 4, 7, 8, 11, 12, 13, 16]
                    @test filter(warning -> warning[2] != "wrap", result.warnings) ==
                          [(1, "semicolon"), (4, "values"), (11, "paragraph")]
                    # HTML lines, math blocks and definitions of reference links do not
                    # render as paragraphs
                    write(path, """
                    <details>
                    <summary>The log</summary>

                    The log is long.

                    </details>

                    \$\$
                    a = b; c
                    d = e
                    \$\$

                    [a]: https://example.org/a
                    [b]: https://example.org/b
                    """)
                    result = lint(directory, "--github", path)
                    @test isempty(result.findings) && isempty(result.warnings)
                end
            end

            @testset "usage errors" begin
                # a file that does not exist, with and without --github
                missing = joinpath(directory, "missing.md")
                for arguments in ((missing,), ("--github", missing))
                    result = lint(directory, arguments...)
                    @test result.code == 2
                    @test occursin("missing.md: no such file", result.message)
                end
                # --github without a file
                result = lint(directory, "--github")
                @test result.code == 2
                @test occursin("--github takes the files to check", result.message)
                # a file that is no UTF-8 text
                write(joinpath(directory, "latin.md"), UInt8[0x54, 0xe9, 0x0a])
                result = lint(directory, joinpath(directory, "latin.md"))
                @test result.code == 2
                @test occursin("latin.md: not UTF-8 text", result.message)
            end

            @testset "the lines that a branch adds" begin
                if Sys.which("git") === nothing
                    @info "no git on the path: the test of the lines that a branch adds is skipped"
                    @test_skip false
                else
                    mktempdir() do repository
                        mkpath(joinpath(repository, "scripts"))
                        cp(LINT, joinpath(repository, "scripts", "lint_markdown.py"))
                        page = joinpath(repository, "page.md")
                        git_in(repository, "init", "-q")
                        write(page, """
                        # Title

                        The first sentence.
                        The second sentence.
                        The third sentence.
                        The fourth sentence.

                        The scheme is old; it stays.
                        """)
                        git_in(repository, "add", "page.md")
                        git_in(repository, "commit", "-q", "-m", "base")
                        git_in(repository, "checkout", "-q", "-b", "topic")
                        write(page, """
                        # Title

                        The first sentence.
                        The second sentence.
                        The third sentence.
                        The fourth sentence.
                        The fifth sentence.
                        The sixth sentence.
                        The seventh sentence.

                        The scheme is old; it stays.

                        The scheme is new; it is added.
                        """)
                        git_in(repository, "commit", "-q", "-a", "-m", "topic")
                        result = lint(repository, "--base", "main")
                        # - line 7: the first added line of a paragraph of seven sentences
                        #   that starts on line 3
                        # - line 11: a semicolon on a line of the base, not reported
                        # - line 13: a semicolon on an added line
                        @test isempty(result.findings)
                        @test result.warnings == [(7, "paragraph"), (13, "semicolon")]
                        @test result.code == 0
                        # a reference that git does not know
                        result = lint(repository, "--base", "nosuch")
                        @test result.code == 2
                        @test occursin("--base nosuch", result.message)
                    end
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
