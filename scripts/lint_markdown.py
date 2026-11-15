#!/usr/bin/env python3
"""Check the rules of CONTRIBUTING.md for markdown text that a script can decide.

    python3 scripts/lint_markdown.py                  the lines the branch adds against origin/dev
    python3 scripts/lint_markdown.py --base main      the same against another reference
    python3 scripts/lint_markdown.py <file> ...       every line of the given files
    python3 scripts/lint_markdown.py --github <file>  a text for GitHub, at any path, or - for stdin
    python3 scripts/lint_markdown.py --strict ...     warnings count as findings for the exit code

Findings, from the section "Markdown" of CONTRIBUTING.md:

    sentences   one sentence per line
    wrap        no line break inside a sentence (with --github a warning, below)
    length      a sentence of at most 200 characters
    parenthesis no second sentence inside parentheses

Warnings, from the section "Writing" of CONTRIBUTING.md:

    semicolon   a semicolon followed by more text of the sentence, in prose outside code, math, URLs
                and HTML entities. One that separates the entries of a list of citations is not
                reported: the text after it starts with a doi, or with names and a journal up to
                its volume, page and year ("; K. P. Dere, Astronomy and Astrophysics 466, 771
                (2007)"). Table cells, where it may separate alternatives, are not checked.
    values      three or more values of one unit in a sentence. Numbers with a unit of the list
                UNITS count by their unit. Numbers in scientific notation or with a decimal point
                count as one kind. These do not count:
                - integers without a unit, among them years and the parts of a date
                - a number that follows a letter or a period, such as v26.5
                - a version of three parts (1.13.1), and the list of versions after Julia, Python,
                  DREAM or "version" ("Julia 1.10, 1.11 and 1.12")
                - a doi, and the volume, page and year of a journal reference
                - a number alone in parentheses, with an optional letter, such as (2.7) or (2.5e)
                - the numbers after "eq.", "fig.", "table", "sec." and their plurals, and a file and line
                - a formula: numbers joined by + - − = or ×, with a space after the sign
                - a number after a word of a condition ("at 120 eV"), unless further values of
                  its unit follow it as a list ("at 3 eV, 5 eV and 7 eV" counts three)
                - the first number of a range ("a to b", "a – b", "between a and b")
    cell        a table cell of more than 150 characters as it renders, with links reduced to their
                text and backticks removed
    list_end    a list item that ends with a semicolon. One that ends with a comma or a colon is
                not reported.
    paragraph   a paragraph or a list item of more than MOST_SENTENCES sentences, six. The rule of
                "Writing" is five sentences, and the lint warns from seven on.
    of_chain    three "of" in a row, each joined to the next by one or two words other than
                "and", "or", "is", "are", "was", "were", "than" and "as"
    page_start  a page whose first block below its title is no paragraph, past Documenter's blocks
                `@meta` and `@setup`. It does not apply with --github.

Code blocks, `$$` math blocks, headings, quotes, comments, lines that start with an HTML tag,
definitions of reference links and horizontal rules are skipped, and so are tables except for the
rule cell.
The files of .github/ are skipped, since they keep one paragraph per line, and so is
THIRD_PARTY_NOTICES.md, whose attributions and citations keep the wording of their sources.
A period after an abbreviation of the list below that does not follow a letter does not end a sentence.

With --github the text is that of a pull request, an issue or a comment, where GitHub renders a
newline as a line break.
There sentences and page_start do not apply.
wrap is a warning there, on each line of a paragraph or a list item that goes on in the next line.
A sentence that goes on over a line break is checked whole, at its first line.
length holds for each sentence, and paragraph counts the sentences of the whole paragraph.
The other rules hold as in a file.
The files may lie outside the repository, and none is skipped.

Each finding and each warning is one line, `file:line: rule: message` and `file:line: warning: rule: message`.
Under GitHub Actions (GITHUB_ACTIONS=true) each one of a file is also written as a workflow command,
which GitHub shows as an annotation on the line.
GitHub shows at most 10 warnings and 10 errors of a step as annotations ("REST API endpoints for
check runs" of the GitHub Docs), and the log of the job lists them all.
The exit code is 1 when a line breaks a rule of "Markdown", or a rule of "Writing" with --strict,
and 0 otherwise.
A line break inside a paragraph with --github is a warning, so it fails the run only with --strict.
It is 2 on a usage error, with a message: a file that does not exist or cannot be read as UTF-8
text, a file outside the repository without --github, --github without a file, or a --base that
git does not know.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIPPED = (".github/", "THIRD_PARTY_NOTICES.md")
LENGTH = 200
CELL = 150
# the most sentences of a paragraph or a list item without a warning. The rule of "Writing" is five,
# and the lint leaves a margin of one.
MOST_SENTENCES = 6
VALUES = 3
# a period after one of these does not end a sentence
ABBREVIATIONS = ("e.g.", "i.e.", "cf.", "vs.", "al.", "approx.", "Fig.", "Figs.", "Eq.", "Eqs.", "Ref.", "Refs.",
                 "Sec.", "No.", "Phys.", "Rev.", "Lett.", "Nucl.", "Comput.", "Commun.", "J.")
# an abbreviation as a whole word: not after a letter, so that "approval." keeps its period
ABBREVIATION_PATTERN = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(a[:-1]) for a in ABBREVIATIONS) + r")\.")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(?:\[[ xX]\]\s+)?")
SENTENCE_END = re.compile(r"[.!?][\"'”’)\]*_]*\s+(?=[\"'“‘(\[*_]*[A-ZÀ-Ý§])")
ENDS_SENTENCE = re.compile(r"[.!?:][\"'”’)\]*_]*$")
# the units whose numbers count as values, each as written in the text
UNITS = ("eV", "keV", "MeV", "GeV", "K", "s", "ms", "µs", "μs", "ns", "min", "h", "m", "cm", "mm", "m⁻³", "m^-3",
         "m⁻¹", "m²", "m³", "T", "V", "V/m", "A", "kA", "MA", "W", "MW", "J", "MJ", "Pa", "%", "GB", "MB",
         "kB", "Hz", "rad")
NUMBER = r"(?<![\w.^⁻])[-−+]?\d+(?:\.\d+)?(?:[eE][-−+]?\d+|\s?[×x]\s?10[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?"
UNIT = (r"(?:" + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
        + r")(?![\w⁻⁰¹²³⁴⁵⁶⁷⁸⁹/])")
VALUE_PATTERN = NUMBER + r"(?:\s?(" + UNIT + r"))?"
VALUE = re.compile(VALUE_PATTERN)
# numbers joined by an operator are a formula, not values
FORMULA = re.compile(VALUE_PATTERN + r"(?:\s*[-+=−×]\s+" + VALUE_PATTERN + ")+")
# a number that follows one of these words states a condition, not a value, unless further values of
# its unit follow as a list. After "between" the next value is the end of a range, not of a list.
CONDITION = re.compile(r"\b([Aa]t|[Aa]bove|[Bb]elow|[Bb]eyond|[Uu]nder|[Oo]ver|[Ff]rom|[Bb]etween|[Ww]ithin"
                       r"|[Uu]p to|[Ff]or|[Aa]fter|[Bb]efore|than|[Nn]ear)\s+" + VALUE_PATTERN)
LIST_NEXT = re.compile(r"(?:,\s*|\s+(?:and|or)\s+)" + VALUE_PATTERN)
# the first number of a range "a to b"
RANGE = re.compile(VALUE_PATTERN + r"\s+(?:to|–)\s+(?=[-−+]?\d)")
# a version without a unit after it, and a list of them after the name of a program or "version"
VERSION = r"v?\d+(?:\.\d+)*(?!\w|\.\d)(?!\s?" + UNIT + ")"
VERSIONS = (r"\b(?:Julia|Python|DREAM|[Vv]ersions?)\s+" + VERSION
            + r"(?:(?:,\s*|\s+(?:and|or|to)\s+)" + VERSION + ")*")
# what is not a value: versions, a number alone in parentheses (an equation number), a file and line,
# a doi, and the numbers of equations, figures, tables and sections
NOT_VALUES = re.compile(r"\bv?\d+(?:\.\d+){2,}\b|" + VERSIONS + r"|\(\s*[\d.]+[a-z]?\s*\)"
                        r"|\S+\.\w+:\d+|\bdoi:\S+"
                        r"|\b(?:[Ee]qs?|[Ee]quations?|[Ff]igs?|[Ff]igures?|[Tt]ables?|[Ss]ecs?|[Ss]ections?)[.․]?\s+"
                        r"[\w.()]+(?:(?:,\s*|\s+and\s+|\s+to\s+)[\w.()]+)*")
CITATION = re.compile(r"\(\d{4}[a-z]?\)|\bdoi:")
# an entry of a list of citations as `prose` gives it: a doi, or names and a journal, each word
# capitalized or a joining word, up to the volume, the page and the year
CITATION_ENTRY = re.compile(r"doi:|(?:(?:[A-ZÀ-Ý][^\s,;()]*|and|et|al․|of|de|van|von|der),?\s+)+"
                            r"\d+(?:,\s*[A-Z]?\d+)?\s+\(\d{4}[a-z]?\)")
FENCE = re.compile(r"^(`{3,}|~{3,})")
HTML_LINE = re.compile(r"^</?[A-Za-z][\w-]*(?:\s[^>]*)?/?>")
# a definition of a reference link: a label that is no footnote, one target and an optional title
LINK_DEFINITION = re.compile(r"^\[(?!\^)[^\]]+\]:\s*(?:<[^>]*>|\S+)(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*$")
# three "of" in a row, each pair joined by one or two words that are not a conjunction or a verb of the list
LINK_WORD = r"(?!(?:and|or|is|are|was|were|than|as)\b)[^\s,;:()]+"
OF_CHAIN = re.compile(rf"\bof(?:\s+{LINK_WORD}){{1,2}}\s+of(?:\s+{LINK_WORD}){{1,2}}\s+of\b")


class UsageError(Exception):
    """A call that the lint cannot run, reported with exit code 2."""


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise UsageError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def added_lines(base):
    """Markdown files of the branch and the numbers of the lines it adds, against the merge base."""
    try:
        git("rev-parse", "--verify", "--quiet", base + "^{commit}")
    except UsageError:
        raise UsageError(f"--base {base}: git knows no such commit") from None
    start = git("merge-base", base, "HEAD").strip()
    files = {}
    for name in git("diff", "--name-only", "--diff-filter=AMR", start, "--", "*.md").split("\n"):
        if not name:
            continue
        numbers = set()
        for line in git("diff", "-U0", start, "--", name).split("\n"):
            hunk = re.match(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            if hunk:
                first, count = int(hunk.group(1)), int(hunk.group(2) or 1)
                numbers.update(range(first, first + count))
        files[name] = numbers
    for name in git("ls-files", "--others", "--exclude-standard", "--", "*.md").split("\n"):
        if name:
            files[name] = None      # an untracked file: every line
    return files


def plain(text):
    """The text of a line as it reads: no list marker, code spans and link targets collapsed."""
    text = LIST_ITEM.sub("", text.strip(), count=1)
    text = re.sub(r"``+.*?``+|`[^`]*`", "Code", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<https?://[^>]*>|https?://\S+", "Link", text)
    text = ABBREVIATION_PATTERN.sub("\\1\u2024", text)                      # a dot that ends nothing
    text = re.sub(r"\b([A-Z])\.(?=\s+[A-Z])", "\\1\u2024", text)            # initials of a name
    return text


def prose(text):
    """The plain text of a line without HTML entities and `$` math, for the rules of "Writing"."""
    text = re.sub(r"&#?\w+;", " ", plain(text))
    return re.sub(r"\$[^$]+\$", "Math", text)


def sentences(text):
    """The sentences of the plain text of a line."""
    parts, start = [], 0
    for end in SENTENCE_END.finditer(text):
        parts.append(text[start:end.end()].strip())
        start = end.end()
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def blocks(lines):
    """The paragraphs, list items and table rows of a text, each as a kind and its line numbers (from 1)."""
    found, fence, comment, math = [], None, False, False
    previous = None     # the kind of the line before: "prose", "row" or None
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        marker = FENCE.match(stripped)
        if fence:
            if marker and marker.group(1).startswith(fence):
                fence = None
            continue
        if marker:
            fence, previous = marker.group(1), None
            continue
        if math:
            math = "$$" not in stripped
            continue
        if stripped.startswith("$$") and stripped.count("$$") == 1:
            math, previous = True, None     # a block of math opens
            continue
        if stripped.startswith("$$") and stripped.endswith("$$") and stripped.count("$$") == 2:
            previous = None                 # a block of math on one line
            continue
        if comment or stripped.startswith("<!--"):
            comment, previous = "-->" not in stripped, None
            continue
        if HTML_LINE.match(stripped) or LINK_DEFINITION.match(stripped):
            previous = None
            continue
        if stripped.startswith("|"):
            found.append(("row", [number]))
            previous = "row"
            continue
        if not stripped or stripped.startswith(("#", ">")) or re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            previous = None
            continue
        if LIST_ITEM.match(line):
            found.append(("item", [number]))
        elif previous == "prose":
            found[-1][1].append(number)
        else:
            found.append(("paragraph", [number]))
        previous = "prose"
    return found


def cells(line):
    """The cells of a table row as they render: code spans and escaped bars masked before the split on `|`."""
    spans = re.findall(r"``+.*?``+|`[^`]*`", line)
    masked = line.strip().strip("|")
    for index, span in enumerate(spans):
        masked = masked.replace(span, f"\0{index}\0", 1)
    masked = masked.replace("\\|", "\1")        # an escaped bar does not split
    spans = [span.strip("`") for span in spans]
    rendered = []
    for cell in masked.split("|"):
        cell = re.sub(r"\0(\d+)\0", lambda m: spans[int(m.group(1))], cell)
        cell = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", cell)
        rendered.append(cell.replace("\1", "|").strip())
    return rendered


def values(sentence):
    """The units of which a sentence holds three or more values, with their counts."""
    text = CITATION.sub(" ", re.sub(r"\d+, [A-Z]?\d+ \(\d{4}[a-z]?\)", " ", sentence))
    text = FORMULA.sub(" ", NOT_VALUES.sub(" ", text))

    def condition(found):
        following = LIST_NEXT.match(text, found.end())
        if found.group(1).lower() != "between" and following and following.group(1) == found.group(2):
            return found.group(0)       # the first value of a list
        return " "

    text = RANGE.sub(" ", CONDITION.sub(condition, text))
    counts = {}
    for found in VALUE.finditer(text):
        unit = found.group(1)
        number = found.group(0)
        if unit is None and not re.search(r"[.eE×x]", number):
            continue        # an integer without a unit: a count, a name or a numeral
        kind = unit or "number"
        counts[kind] = counts.get(kind, 0) + 1
    return {kind: count for kind, count in counts.items() if count >= VALUES}


def semicolons(sentence):
    """Whether a semicolon joins more text of the sentence, other than the next entry of a list of citations."""
    for found in re.finditer(r";\s+(?=\S)", sentence):
        if not CITATION_ENTRY.match(sentence, found.end()):
            return True
    return False


def page_start(lines):
    """The number of the first line below the title of a page when it is no paragraph, or None.

    The title is the first line outside code blocks that starts with "# ".
    Documenter's blocks `@meta` and `@setup` and HTML comments, which render nothing, are passed over.
    """
    title, fence = None, None
    for index, line in enumerate(lines):
        marker = FENCE.match(line.strip())
        if fence:
            if marker and marker.group(1).startswith(fence):
                fence = None
        elif marker:
            fence = marker.group(1)
        elif line.startswith("# "):
            title = index
            break
    if title is None:
        return None
    index, hidden, comment = title + 1, False, False
    while index < len(lines):
        stripped = lines[index].strip()
        if hidden:
            hidden = not stripped.startswith("```")
        elif comment:
            comment = "-->" not in stripped
        elif stripped.startswith(("```@meta", "```@setup")):
            hidden = True
        elif stripped.startswith("<!--"):
            comment = "-->" not in stripped
        elif stripped:
            break
        index += 1
    if index == len(lines):
        return None
    if LIST_ITEM.match(lines[index]) or lines[index].strip().startswith(("#", "|", ">", "`" * 3, "~" * 3)):
        return index + 1
    return None


def check(lines, wanted, github=False):
    """The findings of a text as (line, level, rule, message), level "error" or "warning"."""
    findings = []

    def report(number, level, rule, message):
        if wanted is None or number in wanted:
            findings.append((number, level, rule, message))

    start = None if github else page_start(lines)
    if start is not None:
        report(start, "warning", "page_start", "the page does not start with a sentence on what it holds")

    for kind, numbers in blocks(lines):
        if kind == "row":
            number = numbers[0]
            if re.match(r"^\|?[\s:|-]+$", lines[number - 1].strip()):
                continue    # the separator below the header
            for cell in cells(lines[number - 1]):
                if len(cell) > CELL:
                    report(number, "warning", "cell", f"a table cell of {len(cell)} characters, the limit is {CELL}")
            continue
        # the pieces whose sentences are checked, each as its line numbers: a line, or with --github
        # the lines of one sentence that goes on over a line break
        pieces = []
        for position, number in enumerate(numbers):
            if pieces and github and not ENDS_SENTENCE.search(plain(lines[pieces[-1][-1] - 1])):
                pieces[-1].append(number)
            else:
                pieces.append([number])
            # the next line belongs to the same paragraph or list item
            if position == len(numbers) - 1:
                continue
            text = plain(lines[number - 1])
            if github:
                report(number, "warning", "wrap", "the paragraph goes on in the next line, "
                       "which GitHub shows as a line break in a description or a comment")
            elif text and not ENDS_SENTENCE.search(text):
                report(number, "error", "wrap", "the sentence continues on the next line")
        count = 0
        for piece in pieces:
            number = piece[0]
            text = " ".join(plain(lines[n - 1]) for n in piece)
            if github:
                for sentence in sentences(text):
                    if len(sentence) > LENGTH:
                        report(number, "error", "length", f"a sentence of {len(sentence)} characters, the limit is {LENGTH}")
            else:
                if SENTENCE_END.search(text):
                    report(number, "error", "sentences", "several sentences on one line")
                if len(text) > LENGTH:
                    report(number, "error", "length", f"{len(text)} characters, the limit is {LENGTH}")
            if re.search(r"\([^()]*[.!?]\s+[A-ZÀ-Ý][^()]*\)", text):
                report(number, "error", "parenthesis", "a second sentence inside parentheses")
            parts = sentences(" ".join(prose(lines[n - 1]) for n in piece))
            count += len(parts)
            for sentence in parts:
                if semicolons(sentence):
                    report(number, "warning", "semicolon", "a semicolon joins two clauses")
                for unit, many in values(sentence).items():
                    report(number, "warning", "values", f"{many} values in {unit} in one sentence: a table holds them")
                chain = OF_CHAIN.search(sentence)
                if chain:
                    report(number, "warning", "of_chain", f"a chain of three \"of\": \"{chain.group(0)}\"")
        if kind == "item" and lines[numbers[-1] - 1].strip().endswith(";"):
            report(numbers[-1], "warning", "list_end", "the list item ends with \";\"")
        if count > MOST_SENTENCES:
            anchor = next((n for n in numbers if wanted is None or n in wanted), numbers[0])
            report(anchor, "warning", "paragraph", f"a {'list item' if kind == 'item' else 'paragraph'} of {count} sentences, "
                   f"the lint warns above {MOST_SENTENCES}")
    return sorted(findings, key=lambda finding: finding[0])     # by line, stable within a line


def escape(text, as_property=False):
    """Text escaped for a workflow command of GitHub Actions, as data or as a property."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if as_property:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="files to check in full (default: the lines the branch adds)")
    parser.add_argument("--base", default="origin/dev", help="the reference of the branch (default: origin/dev)")
    parser.add_argument("--github", action="store_true",
                        help="text for GitHub, where a newline is a line break, at any path, or - for stdin")
    parser.add_argument("--strict", action="store_true", help="exit with 1 on warnings too")
    arguments = parser.parse_args()
    if arguments.github:
        if not arguments.files:
            parser.error("--github takes the files to check, or - for stdin")
        targets = {name: None for name in arguments.files}
    elif arguments.files:
        targets = {}
        for name in arguments.files:
            path = Path(name).resolve()
            if not path.is_relative_to(ROOT):
                parser.error(f"{name} lies outside the repository: check a draft for GitHub with --github")
            targets[str(path.relative_to(ROOT))] = None
    else:
        try:
            targets = added_lines(arguments.base)
        except UsageError as error:
            parser.error(str(error))
    # the texts, read before any output, so that a file that cannot be read stops the run with a message
    texts = {}
    for name in sorted(targets):
        if not arguments.github and name.startswith(SKIPPED):
            continue
        path = Path(name) if arguments.github else ROOT / name
        if not arguments.files and not path.exists():
            continue        # a file of the diff that the working tree no longer holds
        try:
            texts[name] = sys.stdin.read() if arguments.github and name == "-" else path.read_text(encoding="utf-8")
        except FileNotFoundError:
            parser.error(f"{name}: no such file")
        except UnicodeDecodeError:
            parser.error(f"{name}: not UTF-8 text")
        except OSError as error:
            parser.error(f"{name}: cannot be read: {error.strerror}")
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    errors, warnings = 0, 0
    for name, text in texts.items():
        for number, level, rule, message in check(text.split("\n"), targets[name], arguments.github):
            if level == "error":
                print(f"{name}:{number}: {rule}: {message}")
                errors += 1
            else:
                print(f"{name}:{number}: warning: {rule}: {message}")
                warnings += 1
            if annotate and name != "-":
                print(f"::{level} file={escape(name, True)},line={number},title={escape(rule, True)}::{escape(message)}")
    summary = f"{errors} finding{'s' if errors != 1 else ''} in {len(targets)} file{'s' if len(targets) != 1 else ''}"
    if warnings:
        summary += f", {warnings} warning{'s' if warnings != 1 else ''}"
    print(summary)
    sys.exit(1 if errors or (arguments.strict and warnings) else 0)
