#!/usr/bin/env python3
"""Check the simplest patterns of "Code style" in CONTRIBUTING.md, in the Julia files of src/ and ext/.

    python3 scripts/check_code_style.py                  the lines the branch adds against origin/dev
    python3 scripts/check_code_style.py --base main      the same against another reference
    python3 scripts/check_code_style.py <file> ...       every line of the given files

Rules:

    short-circuit  `cond && action` or `cond || action` whose action is an assignment, an assignment
                   in parentheses or a call of a function whose name ends in `!`;
                   `return`, `continue` and `break` after the operator are allowed
    throw          `cond && throw(...)` or `cond || throw(...)`, or `error(...)` in place of `throw`,
                   that is not a check at the start of its function
    conditional    a conditional expression nested in another on one line,
                   or one whose branches continue on the next line
    semicolon      `;` between two statements on one line
    composition    `|>` or `∘` as an operator

Comments, docstrings and the contents of strings are skipped.
A statement continued over several lines is checked as one, and a finding is reported at the line
of its operator; against a base it counts when one of its lines is added.
A check is at the start of its function when, between it and the line that opens the function
(a `function` line, a `do` block or a one-line definition), there are only other checks, `for` loops
that hold only checks and the assignments they read, `if` blocks that end in a `throw` after
assignments only, and assignments of a name that one of the checks of the function's start reads.
A check outside any function is not reported.
The rules decide on patterns, not on parsed Julia: a form they do not recognise is not reported.
The exit code is 1 when a line breaks a rule, and 0 otherwise.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATHSPECS = ("src/*.jl", "ext/*.jl")
# the end of a line that continues the statement on the next one
CONTINUES = re.compile(r"(\|\||&&|[=+\-*/\\^÷%,?<>|&~∘]|(?<=\s):|->|\b(?:in|isa)\b)\s*$")
# an assignment operator at its `=`: not `==`, `<=`, `>=`, `!=`, `=>`, `===`, `!==`
ASSIGNMENT = re.compile(r"(?<![=!<>≤≥:])=(?![=>])")
IDENTIFIER = re.compile(r"[^\W\d]\w*!?")
OPENERS = re.compile(r"^(?:@\w+\s+)*(function|macro|for|while|if|elseif|else|let|begin|try|catch|finally|"
                     r"struct|mutable\s+struct|module|baremodule|quote)\b")


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout


def added_lines(base):
    """Julia files of src/ and ext/ that the branch changes and the numbers of the lines it adds."""
    start = git("merge-base", base, "HEAD").strip()
    files = {}
    for name in git("diff", "--name-only", "--diff-filter=AMR", start, "--", *PATHSPECS).split("\n"):
        if not name:
            continue
        numbers = set()
        for line in git("diff", "-U0", start, "--", name).split("\n"):
            hunk = re.match(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            if hunk:
                first, count = int(hunk.group(1)), int(hunk.group(2) or 1)
                numbers.update(range(first, first + count))
        files[str(ROOT / name)] = numbers
    for name in git("ls-files", "--others", "--exclude-standard", "--", *PATHSPECS).split("\n"):
        if name:
            files[str(ROOT / name)] = None      # an untracked file: every line
    return files


def mask(text):
    """The code of a file with comments and string contents blanked; also, per line, whether its end
    lies inside a string."""
    out = list(text)
    n = len(text)
    in_string_at_end = []
    stack = []          # ["code", depth] inside an interpolation, ["str", delimiter, raw]

    def blank(start, stop):
        for k in range(start, stop):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while i < n:
        ch = text[i]
        inside = bool(stack)        # inside a string, also in its interpolations
        if ch == "\n":
            in_string_at_end.append(any(c[0] == "str" for c in stack))
            if stack and stack[-1][0] == "str":
                i += 1
                continue
        if not stack or stack[-1][0] == "code":
            if text.startswith("#=", i):
                level, j = 1, i + 2
                while j < n and level:
                    if text.startswith("#=", j):
                        level, j = level + 1, j + 2
                    elif text.startswith("=#", j):
                        level, j = level - 1, j + 2
                    else:
                        if text[j] == "\n":
                            in_string_at_end.append(False)
                        j += 1
                blank(i, j)
                i = j
                continue
            if ch == "#":
                j = text.find("\n", i)
                j = n if j < 0 else j
                blank(i, j)
                i = j
                continue
            if ch in "\"`":
                delimiter = ch * 3 if text.startswith(ch * 3, i) else ch
                raw = i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_") and ch == '"'
                if inside:
                    blank(i, i + len(delimiter))
                stack.append(["str", delimiter, raw])
                i += len(delimiter)
                continue
            if ch == "'":
                previous = text[i - 1] if i > 0 else ""
                if previous and (previous.isalnum() or previous in "_)]}'\"!."):
                    i += 1          # the adjoint
                    continue
                j = i + 1
                if j < n and text[j] == "\\":
                    j += 2
                    while j < n and text[j] not in "'\n":
                        j += 1
                else:
                    j += 1
                if j < n and text[j] == "'":
                    blank(i + 1 if not inside else i, j if not inside else j + 1)
                    i = j + 1
                    continue
                i += 1
                continue
            if stack:               # an interpolation
                if ch == "(":
                    stack[-1][1] += 1
                elif ch == ")":
                    stack[-1][1] -= 1
                    if stack[-1][1] == 0:
                        stack.pop()
                blank(i, i + 1)
            i += 1
            continue
        # inside a string
        _, delimiter, raw = stack[-1]
        if ch == "\\":
            blank(i, min(i + 2, n))
            if i + 1 < n and text[i + 1] == "\n":
                in_string_at_end.append(True)
            i += 2
            continue
        if text.startswith(delimiter, i):
            stack.pop()
            if stack:
                blank(i, i + len(delimiter))
            i += len(delimiter)
            continue
        if not raw and text.startswith("$(", i):
            blank(i, i + 2)
            stack.append(["code", 1])
            i += 2
            continue
        blank(i, i + 1)
        i += 1
    in_string_at_end.append(bool(stack))
    return "".join(out).split("\n"), in_string_at_end


class Statement:
    """Physical lines joined into one statement: the text with newlines, the number of its first line,
    the bracket depth before each character and the indentation of its first line."""

    def __init__(self, first, lines):
        self.first = first
        self.text = "\n".join(lines)
        self.indent = len(lines[0]) - len(lines[0].lstrip())
        self.depth = []
        self.curly = []
        depth = curly = 0
        for ch in self.text:
            if ch in ")]}":
                depth = max(depth - 1, 0)
                if ch == "}":
                    curly = max(curly - 1, 0)
            self.depth.append(depth)
            self.curly.append(curly)
            if ch in "([{":
                depth += 1
                if ch == "{":
                    curly += 1
        self.stripped = self.text.strip()

    def line_of(self, position):
        return self.first + self.text.count("\n", 0, position)

    @property
    def last(self):
        return self.first + self.text.count("\n")

    def top_level(self, pattern, start=0, stop=None):
        """Matches of a pattern at bracket depth 0."""
        stop = len(self.text) if stop is None else stop
        return [m for m in re.finditer(pattern, self.text[:stop]) if m.start() >= start and self.depth[m.start()] == 0]

    def assignment(self):
        """The position of the first assignment operator at depth 0, or None."""
        found = self.top_level(ASSIGNMENT)
        return found[0].start() if found else None

    def opener(self):
        """The keyword of the block this statement opens or continues, or None."""
        match = OPENERS.match(self.stripped)
        if match:
            return match.group(1).split()[-1]
        if self.top_level(r"\bdo\b") or re.search(r"->\s*begin$", self.stripped):
            return "do"
        position = self.assignment()
        if position is not None:
            match = re.match(r"\s*(if|let|begin|try|for|while)\b", self.text[position + 1:])
            if match:
                return match.group(1)
        return None

    def is_end(self):
        return re.fullmatch(r"end\b.*", self.stripped) is not None

    def definition(self):
        """Whether the statement is a one-line definition `name(...) = ...`."""
        position = self.assignment()
        if position is None:
            return False
        left = self.text[:position].strip()
        return re.match(r"^(?:@\w+\s+)*[^\W\d][\w.!]*(?:\{.*\})?\(", left) is not None and \
            re.search(r"[+\-*/^÷%|&⊻\\.]$", left) is None

    def assigned_names(self):
        """The names an assignment `a = ...` or `a, b = ...` gives values, or None for another statement."""
        position = self.assignment()
        if position is None:
            return None
        left = self.text[:position].strip().rstrip("+-*/^÷%|&⊻\\").strip().strip("()")
        names = set()
        for part in left.split(","):
            part = re.sub(r"::.*", "", part).strip()
            if not re.fullmatch(r"[^\W\d]\w*", part):
                return None
            names.add(part)
        return names

    def check(self):
        """The condition of a check `cond || throw(...)` that makes up the whole statement, or None."""
        operators = self.top_level(r"&&|\|\|")
        if not operators or self.assignment() is not None:
            return None
        last = operators[-1]
        rest = self.text[last.end():]
        match = re.match(r"\s*(?:throw|error)\(", rest)
        if not match:
            return None
        close = closing(self, last.end() + match.end() - 1)
        if close is None or self.text[close + 1:].strip():
            return None
        return self.text[:last.start()]


def closing(statement, position):
    """The position of the bracket that closes the one at `position`, or None."""
    depth = statement.depth[position]
    for k in range(position + 1, len(statement.text)):
        if statement.text[k] in ")]}" and statement.depth[k] == depth:
            return k
    return None


def statements(lines, in_string_at_end):
    """The statements of a file, as Statement objects, blank ones left out."""
    result, current, first, depth = [], [], None, 0
    for number, line in enumerate(lines, start=1):
        if not current and not line.strip():
            continue
        if not current:
            first = number
        current.append(line)
        for ch in line:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(depth - 1, 0)
        ends_in_string = in_string_at_end[number - 1] if number - 1 < len(in_string_at_end) else False
        if depth > 0 or ends_in_string or CONTINUES.search(line):
            continue
        result.append(Statement(first, current))
        current, depth = [], 0
    if current:
        result.append(Statement(first, current))
    return result


def names_in(text):
    return set(IDENTIFIER.findall(text))


def block(stmts, k):
    """The statements of the block that the opener at index k starts, and the index of its `end`."""
    body = []
    for j in range(k + 1, len(stmts)):
        if stmts[j].indent <= stmts[k].indent:
            return body, j
        body.append(stmts[j])
    return body, None


def check_block(stmts, k):
    """For an opener at index k: the names its header and checks read when it is a `for` loop that
    holds only checks and the assignments they read, or an `if` that ends in a `throw` after
    assignments only; None for another block."""
    kind = stmts[k].opener()
    body, end = block(stmts, k)
    if end is None or not stmts[end].is_end() or not body:
        return None
    if kind == "for":
        read = set().union(*(names_in(s.check()) for s in body if s.check() is not None))
        if read and all(s.check() is not None or (s.assigned_names() or set()) & read for s in body):
            return read | names_in(stmts[k].text)
        return None
    if kind == "if" and re.fullmatch(r"(?:throw|error)\(.*\)", body[-1].stripped, re.S) and \
            all(s.assigned_names() for s in body[:-1]):
        return names_in(stmts[k].text)
    return None


def at_start_of_function(stmts, k, condition):
    """Whether the check at index k comes at the start of its function, or the walk cannot tell."""
    names = names_in(condition)
    indent = stmts[k].indent
    # the checks that follow, which may read an assignment before this one
    for statement in stmts[k + 1:]:
        if statement.indent != indent:
            break
        if statement.check() is not None:
            names |= names_in(statement.check())
        elif not statement.assigned_names():
            break
    j = k - 1
    while j >= 0:
        statement = stmts[j]
        if statement.indent > indent:
            j -= 1
            continue
        if statement.indent < indent:
            kind = statement.opener()
            if kind in ("function", "macro", "do", "quote", "struct", "module", "baremodule") or kind is None:
                return True
            if kind == "for":
                read = check_block(stmts, j)
                if read is None:
                    return False
                names |= read
                indent = statement.indent
                j -= 1
                continue
            return False
        if statement.is_end():
            # the opener of the block that this `end` closes
            o = j - 1
            while o >= 0 and stmts[o].indent > indent:
                o -= 1
            if o < 0 or stmts[o].indent < indent:
                return True
            read = check_block(stmts, o)
            if read is None:
                return False
            names |= read
            j = o - 1
            continue
        if statement.opener() is not None:
            return False
        condition = statement.check()
        if condition is not None:
            names |= names_in(condition)
            j -= 1
            continue
        assigned = statement.assigned_names()
        if assigned and assigned & names:
            names |= names_in(statement.text[statement.assignment() + 1:])
            j -= 1
            continue
        return False
    return True


def operand(statement, start):
    """The text of the operand that begins at `start`: up to the next `&&`, `||`, `,`, `;`, `?` or
    keyword `for` or `if` at the same depth, or a closing bracket."""
    depth = statement.depth[start] if start < len(statement.text) else 0
    k = start
    text = statement.text
    while k < len(text):
        if statement.depth[k] < depth:
            break
        if statement.depth[k] == depth:
            if text.startswith(("&&", "||"), k) or text[k] in ",;?":
                break
            if re.match(r"\s(?:for|if)\b", text[k:]):
                break
        k += 1
    return text[start:k]


def short_circuits(statement, stmts, index):
    findings = []
    assigned_at = statement.assignment()
    for match in statement.top_level(r"&&|\|\|"):
        if assigned_at is not None and assigned_at < match.start():
            continue
        start = match.end()
        while start < len(statement.text) and statement.text[start] in " \n":
            start += 1
        action = operand(statement, start)
        span = (statement.line_of(match.start()), statement.line_of(start + len(action)))
        if re.match(r"(?:return|continue|break)\b", action):
            continue
        if re.match(r"(?:throw|error)\(", action):
            if statement.definition():
                continue
            condition = statement.text[:match.start()]
            if not at_start_of_function(stmts, index, condition):
                findings.append((span, "throw", f"`{match.group()} {action.split('(')[0]}(...)` that is not a check "
                                                "at the start of its function: use `if`"))
            continue
        inner = Statement(0, [action])
        if inner.assignment() is not None:
            findings.append((span, "short-circuit", f"an assignment after `{match.group()}`: use `if`"))
        elif action.startswith("(") and (close := closing(inner, 0)) is not None and not action[close + 1:].strip() \
                and ASSIGNMENT.search("".join(c for c, d in zip(action, inner.depth) if d == 1)):
            findings.append((span, "short-circuit", f"an assignment in parentheses after `{match.group()}`: use `if`"))
        elif (call := re.match(r"[^\W\d][\w.]*!\(", action)) and (close := closing(inner, call.end() - 1)) is not None \
                and not action[close + 1:].strip():
            name = call.group()[:-1]
            findings.append((span, "short-circuit", f"a call of `{name}` after `{match.group()}`: use `if`"))
    return findings


def ternary_extent(statement, position):
    """The end of the conditional expression whose `?` is at `position`."""
    text, depth = statement.text, statement.depth[position]
    k = position + 1
    while k < len(text):
        if statement.depth[k] < depth or (statement.depth[k] == depth and text[k] in ",;"):
            break
        if statement.depth[k] == depth and re.match(r"\s(?:for|if)\b", text[k:]):
            break
        k += 1
    return k


def conditionals(statement):
    findings = []
    text = statement.text
    marks = [k for k, ch in enumerate(text) if ch == "?" and statement.curly[k] == 0]
    for position in marks:
        line = statement.line_of(position)
        end = ternary_extent(statement, position)
        nested = [m for m in marks if position < m < end and statement.line_of(m) == line]
        if nested:
            findings.append(((line, line), "conditional", "a conditional expression nested in another: use `if`"))
        line_end = text.find("\n", position)
        if 0 <= line_end < end and text[line_end:end].strip():
            findings.append(((line, statement.line_of(end)), "conditional",
                             "a conditional expression that continues on the next line: use `if`"))
    return findings


def check(name, wanted):
    lines, in_string_at_end = mask(Path(name).read_text(encoding="utf-8"))
    stmts = statements(lines, in_string_at_end)
    findings = []
    for index, statement in enumerate(stmts):
        findings += short_circuits(statement, stmts, index)
        findings += conditionals(statement)
        for match in statement.top_level(r";"):
            k = match.start()
            line_start = statement.text.rfind("\n", 0, k) + 1
            line_end = statement.text.find("\n", k)
            line_end = len(statement.text) if line_end < 0 else line_end
            if statement.text[line_start:k].strip() and statement.text[k + 1:line_end].strip():
                line = statement.line_of(k)
                findings.append(((line, line), "semicolon", "two statements on one line"))
        for match in re.finditer(r"\|>|(?<!\()∘(?!\))", statement.text):
            line = statement.line_of(match.start())
            findings.append(((line, line), "composition", f"`{match.group()}` as an operator: name the steps"))
    result = []
    for (first, last), rule, message in findings:
        if wanted is None or any(number in wanted for number in range(first, last + 1)):
            result.append((first, rule, message))
    return sorted(set(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="files to check in full (default: the lines the branch adds)")
    parser.add_argument("--base", default="origin/dev", help="the reference of the branch (default: origin/dev)")
    arguments = parser.parse_args()
    if arguments.files:
        targets = {name: None for name in arguments.files if name.endswith(".jl")}
    else:
        targets = added_lines(arguments.base)
    count = 0
    for name in sorted(targets):
        if not Path(name).exists():
            continue
        shown = name
        if Path(name).is_absolute():
            try:
                shown = str(Path(name).resolve().relative_to(ROOT))
            except ValueError:
                pass
        for number, rule, message in check(name, targets[name]):
            print(f"{shown}:{number}: {rule}: {message}")
            count += 1
    print(f"{count} finding{'s' if count != 1 else ''} in {len(targets)} file{'s' if len(targets) != 1 else ''}")
    sys.exit(1 if count else 0)
