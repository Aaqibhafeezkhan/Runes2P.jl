#!/usr/bin/env python3
"""Check the markdown rules of CONTRIBUTING.md that a script can decide.

    python3 scripts/lint_markdown.py                  the lines the branch adds against origin/dev
    python3 scripts/lint_markdown.py --base main      the same against another reference
    python3 scripts/lint_markdown.py <file> ...       every line of the given files

Rules:

    sentences   one sentence per line
    wrap        no line break inside a sentence
    length      a sentence of at most 200 characters
    parenthesis no second sentence inside parentheses

Code blocks, tables, headings and comments are skipped, and so are the files of .github/,
which keep one paragraph per line, and THIRD_PARTY_NOTICES.md,
whose attributions and citations keep the wording of their sources.
A period after an abbreviation of the list below does not end a sentence, also where the letters
end a longer word, so a sentence end after such a word is missed.
The exit code is 1 when a line breaks a rule, and 0 otherwise.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIPPED = (".github/", "THIRD_PARTY_NOTICES.md")
LENGTH = 200
# a period after one of these does not end a sentence
ABBREVIATIONS = ("e.g.", "i.e.", "cf.", "vs.", "al.", "approx.", "Fig.", "Figs.", "Eq.", "Eqs.", "Ref.", "Refs.",
                 "Sec.", "No.", "Phys.", "Rev.", "Lett.", "Nucl.", "Comput.", "Commun.", "J.")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(?:\[[ xX]\]\s+)?")
SENTENCE_END = re.compile(r"[.!?][\"'”’)\]*_]*\s+(?=[\"'“‘(\[*_]*[A-ZÀ-Ý§])")
ENDS_SENTENCE = re.compile(r"[.!?:][\"'”’)\]*_]*$")


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
    for abbreviation in ABBREVIATIONS:
        text = text.replace(abbreviation, abbreviation[:-1] + "\u2024")     # a dot that ends nothing
    text = re.sub(r"\b([A-Z])\.(?=\s+[A-Z])", "\\1\u2024", text)            # initials of a name
    return text


def prose_lines(lines):
    """Numbers (from 1) of the lines that hold prose."""
    numbers, fence, comment = [], None, False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence:
            if marker and marker.group(1).startswith(fence):
                fence = None
            continue
        if marker:
            fence = marker.group(1)
            continue
        if comment or stripped.startswith("<!--"):
            comment = "-->" not in stripped
            continue
        if not stripped or stripped.startswith(("#", "|", ">")) or re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            continue
        numbers.append(number)
    return numbers


def check(name, wanted):
    lines = (ROOT / name).read_text(encoding="utf-8").split("\n")
    prose = prose_lines(lines)
    is_prose = set(prose)
    findings = []
    for number in prose:
        if wanted is not None and number not in wanted:
            continue
        text = plain(lines[number - 1])
        if SENTENCE_END.search(text):
            findings.append((number, "sentences", "several sentences on one line"))
        if len(text) > LENGTH:
            findings.append((number, "length", f"{len(text)} characters, the limit is {LENGTH}"))
        if re.search(r"\([^()]*[.!?]\s+[A-ZÀ-Ý][^()]*\)", text):
            findings.append((number, "parenthesis", "a second sentence inside parentheses"))
        # the next line belongs to the same paragraph or list item
        continues = number + 1 in is_prose and not LIST_ITEM.match(lines[number])
        if continues and text and not ENDS_SENTENCE.search(text):
            findings.append((number, "wrap", "the sentence continues on the next line"))
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="files to check in full (default: the lines the branch adds)")
    parser.add_argument("--base", default="origin/dev", help="the reference of the branch (default: origin/dev)")
    arguments = parser.parse_args()
    if arguments.files:
        targets = {str(Path(name).resolve().relative_to(ROOT)): None for name in arguments.files}
    else:
        targets = added_lines(arguments.base)
    count = 0
    for name in sorted(targets):
        if name.startswith(SKIPPED) or not (ROOT / name).exists():
            continue
        for number, rule, message in check(name, targets[name]):
            print(f"{name}:{number}: {rule}: {message}")
            count += 1
    print(f"{count} finding{'s' if count != 1 else ''} in {len(targets)} file{'s' if len(targets) != 1 else ''}")
    sys.exit(1 if count else 0)
