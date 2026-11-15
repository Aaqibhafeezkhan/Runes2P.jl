#!/usr/bin/env python3
"""Check the rules of CONTRIBUTING.md for markdown text that a script can decide.

    python3 scripts/lint_markdown.py                  the lines the branch adds against origin/dev
    python3 scripts/lint_markdown.py --base main      the same against another reference
    python3 scripts/lint_markdown.py <file> ...       every line of the given files
    python3 scripts/lint_markdown.py --github <file>  a text for GitHub, at any path, or - for stdin
    python3 scripts/lint_markdown.py --strict ...     warnings count as findings for the exit code

Findings, from the section "Markdown" of CONTRIBUTING.md:

    sentences   one sentence per line
    wrap        no line break inside a sentence
    length      a sentence of at most 200 characters
    parenthesis no second sentence inside parentheses

Warnings, from the section "Writing" of CONTRIBUTING.md:

    semicolon   a semicolon followed by more text of the sentence, in prose outside code, math, URLs
                and HTML entities; one before a citation (a year in parentheses or a doi in the text
                after it) is not reported, and table cells, where it may separate alternatives, are
                not checked
    values      three or more values of one unit in a sentence: numbers with a unit of the list
                UNITS, and numbers in scientific notation or with a decimal point as one kind;
                integers without a unit, citations, dates, versions, numbers in parentheses such as
                equation numbers, and a number after a word of a condition ("at 120 eV") do not count
    cell        a table cell of more than 150 characters as it renders: links reduced to their text,
                backticks removed
    list_end    a list item that ends with a semicolon or a comma
    paragraph   a paragraph or a list item of more than five sentences
    of_chain    three "of" in a row, each joined to the next by one or two words other than
                "and", "or", "is", "are", "was", "were", "than" and "as"
    page_start  a page whose first block below its title is no paragraph, past Documenter's blocks
                `@meta` and `@setup`; not with --github

Code blocks, headings, quotes, comments and horizontal rules are skipped, and so are tables except
for the rule cell.
The files of .github/ are skipped, since they keep one paragraph per line, and so is
THIRD_PARTY_NOTICES.md, whose attributions and citations keep the wording of their sources.
A period after an abbreviation of the list below that does not follow a letter does not end a sentence.

With --github the text is that of a pull request, an issue or a comment, where a paragraph is one line,
since GitHub renders a newline as a line break.
There sentences does not apply, wrap reports a paragraph or a list item that goes on to the next line,
and length holds for each sentence of a line; the other rules hold as in a file.
The files may lie outside the repository, and none is skipped.

Each finding and each warning is one line, `file:line: rule: message` and `file:line: warning: rule: message`.
Under GitHub Actions (GITHUB_ACTIONS=true) each one of a file is also written as a workflow command,
which GitHub shows as an annotation on the line.
The exit code is 1 when a line breaks a rule of "Markdown", or a rule of "Writing" with --strict,
and 0 otherwise.
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
SENTENCES = 5
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
VALUE = re.compile(NUMBER + r"(?:\s?(" + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
                   + r")(?![\w⁻⁰¹²³⁴⁵⁶⁷⁸⁹/]))?")
# a number that follows one of these words states a condition, not a value
CONDITION = re.compile(r"\b(?:[Aa]t|[Aa]bove|[Bb]elow|[Bb]eyond|[Uu]nder|[Oo]ver|[Ff]rom|to|[Bb]etween|[Ww]ithin"
                       r"|[Uu]p to|[Ff]or|[Aa]fter|[Bb]efore|than|[Nn]ear)\s+" + NUMBER)
# what is not a value: dates, versions, numbers in parentheses (equation numbers, years), a file and
# line, a doi, and the numbers of equations, figures, tables and sections
NOT_VALUES = re.compile(r"\d{4}-\d{2}-\d{2}|\bv?\d+(?:\.\d+){2,}\b|\bv\d+(?:\.\d+)*|\(\s*[\d.]+[a-z]?\s*\)"
                        r"|\S+\.\w+:\d+|\bdoi:\S+"
                        r"|\b(?:[Ee]qs?|[Ee]quations?|[Ff]igs?|[Ff]igures?|[Tt]ables?|[Ss]ecs?|[Ss]ections?)[.․]?\s+"
                        r"[\w.()]+(?:(?:,\s*|\s+and\s+|\s+to\s+)[\w.()]+)*")
CITATION = re.compile(r"\(\d{4}[a-z]?\)|\bdoi:")
# three "of" in a row, each pair joined by one or two words that are not a conjunction or a verb of the list
LINK_WORD = r"(?!(?:and|or|is|are|was|were|than|as)\b)[^\s,;:()]+"
OF_CHAIN = re.compile(rf"\bof(?:\s+{LINK_WORD}){{1,2}}\s+of(?:\s+{LINK_WORD}){{1,2}}\s+of\b")


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout


def added_lines(base):
    """Markdown files of the branch and the numbers of the lines it adds, against the merge base."""
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
    found, fence, comment = [], None, False
    previous = None     # the kind of the line before: "prose", "row" or None
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence:
            if marker and marker.group(1).startswith(fence):
                fence = None
            continue
        if marker:
            fence, previous = marker.group(1), None
            continue
        if comment or stripped.startswith("<!--"):
            comment, previous = "-->" not in stripped, None
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
    """The cells of a table row as they render: code and math masked before the split on `|`."""
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
    text = NOT_VALUES.sub(" ", CONDITION.sub(" ", text))
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
    """Whether a semicolon joins more text of the sentence, other than a citation after it."""
    for found in re.finditer(r";\s+(?=\S)", sentence):
        after = sentence[found.end():].split(";")[0]
        if not CITATION.search(after):
            return True
    return False


def page_start(lines):
    """The number of the first line below the title of a page when it is no paragraph, or None.

    Documenter's blocks `@meta` and `@setup`, which render nothing, are passed over.
    """
    title = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    if title is None:
        return None
    index, hidden = title + 1, False
    while index < len(lines):
        stripped = lines[index].strip()
        if hidden:
            hidden = not stripped.startswith("```")
        elif stripped.startswith(("```@meta", "```@setup")):
            hidden = True
        elif stripped:
            break
        index += 1
    if index == len(lines):
        return None
    if LIST_ITEM.match(lines[index]) or lines[index].strip().startswith(("#", "|", ">", "`" * 3, "~" * 3, "<!--")):
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
        count = 0
        for position, number in enumerate(numbers):
            line = lines[number - 1]
            text = plain(line)
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
            # the next line belongs to the same paragraph or list item
            continues = position < len(numbers) - 1
            if github and continues:
                report(number, "error", "wrap", "the paragraph goes on in the next line, which GitHub shows as a line break")
            elif continues and text and not ENDS_SENTENCE.search(text):
                report(number, "error", "wrap", "the sentence continues on the next line")
            parts = sentences(prose(line))
            count += len(parts)
            for sentence in parts:
                if semicolons(sentence):
                    report(number, "warning", "semicolon", "a semicolon joins two clauses")
                for unit, many in values(sentence).items():
                    report(number, "warning", "values", f"{many} values in {unit} in one sentence: a table holds them")
                chain = OF_CHAIN.search(sentence)
                if chain:
                    report(number, "warning", "of_chain", f"a chain of three \"of\": \"{chain.group(0)}\"")
        if kind == "item" and re.search(r"[;,]$", lines[numbers[-1] - 1].strip()):
            report(numbers[-1], "warning", "list_end", f"the list item ends with \"{lines[numbers[-1] - 1].strip()[-1]}\"")
        if count > SENTENCES:
            anchor = next((n for n in numbers if wanted is None or n in wanted), numbers[0])
            report(anchor, "warning", "paragraph", f"a {'list item' if kind == 'item' else 'paragraph'} of {count} sentences, "
                   f"the limit is {SENTENCES}")
    return findings


def escape(text, property=False):
    """Text escaped for a workflow command of GitHub Actions, as data or as a property."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if property:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="files to check in full (default: the lines the branch adds)")
    parser.add_argument("--base", default="origin/dev", help="the reference of the branch (default: origin/dev)")
    parser.add_argument("--github", action="store_true",
                        help="text for GitHub, one paragraph per line, at any path, or - for stdin")
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
        targets = added_lines(arguments.base)
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    errors, warnings = 0, 0
    for name in sorted(targets):
        if arguments.github:
            text = sys.stdin.read() if name == "-" else Path(name).read_text(encoding="utf-8")
        elif name.startswith(SKIPPED) or not (ROOT / name).exists():
            continue
        else:
            text = (ROOT / name).read_text(encoding="utf-8")
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
