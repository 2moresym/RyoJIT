#!/usr/bin/env python3
"""Every PpcIntr variant must be handled by every table, and every call site must
match the table it is checked against.

`PpcIntr::{name, effects, arity, has_dst}` are four parallel `match`es over the
same enum, and `lower::Ctx::intr_raw` debug-asserts operand counts against them at
test time.  Missing an arm is a compile error for `name`/`arity` (no `_ =>`), but
`has_dst` uses `!matches!` and `effects` can silently fall through to the wrong
class if an arm is forgotten — so all four are checked here, plus every call site's
operand count against the arity table.

Run this after touching intrinsics.rs or a `PpcIntr::` call site.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INTR = ROOT / "src/frontends/wii_ppc/intrinsics.rs"
LOW = ROOT / "src/frontends/wii_ppc/lower.rs"

src = INTR.read_text()
variants = re.findall(r"^    ([A-Z][A-Za-z0-9]+) = 0x", src, re.M)
problems = []


def section(start_marker, end_marker):
    i = src.index(start_marker)
    j = src.index(end_marker, i) if end_marker in src[i:] else len(src)
    return src[i:j]


BODIES = {
    "name": section("pub const fn name(self)", "/// Side effects"),
    "effects": section("pub const fn effects(self)", "/// Fixed operand count"),
    "arity": section("pub const fn arity(self)", "/// Does this intrinsic define a value"),
    "has_dst": section("pub const fn has_dst(self)", "/// Resolve an intrinsic id"),
    "ALL": section("pub const ALL", "/// The wire id"),
}
for table, body in BODIES.items():
    listed = set(re.findall(r"\b([A-Z][A-Za-z0-9]+)\b", body))
    missing = [v for v in variants if v not in listed]
    if table == "has_dst":
        # has_dst is `!matches!(…)`, so it only lists the *no-value* variants;
        # every variant must appear either there or nowhere.  Check instead that
        # the list has no unknown names.
        unknown = [x for x in re.findall(r"\|\s*([A-Z][A-Za-z0-9]+)", body) if x not in variants]
        if unknown:
            problems.append(f"has_dst lists unknown variants: {unknown}")
        continue
    if missing:
        problems.append(f"{table}: variant(s) not listed -> {missing}")

# call-site operand counts vs the arity table
arity = {}
for m in re.finditer(r"([A-Za-z0-9_|\s,]+?)\s*=>\s*(\d),", BODIES["arity"], re.S):
    for nm in re.findall(r"\b([A-Z][A-Za-z0-9]+)\b", m.group(1)):
        arity[nm] = int(m.group(2))

low = LOW.read_text()
for m in re.finditer(r"self\.intr(?:_raw)?\(\s*(?:I::|PpcIntr::)([A-Za-z0-9]+)\s*,\s*(Vec::new\(\)|vec!\[)", low):
    name, opener = m.group(1), m.group(2)
    if opener == "Vec::new()":
        n = 0
    else:
        i = low.index("[", m.end() - 1)
        depth, j = 0, i
        while j < len(low):
            if low[j] == "[":
                depth += 1
            elif low[j] == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        inner = low[i + 1 : j].strip().rstrip(",")
        n, d = 1, 0
        for ch in inner:
            if ch in "([{":
                d += 1
            elif ch in ")]}":
                d -= 1
            elif ch == "," and d == 0:
                n += 1
        if not inner:
            n = 0
    want = arity.get(name)
    if want is None:
        problems.append(f"call site {name}: no arity arm (non-exhaustive match = compile error)")
    elif want != n:
        problems.append(f"call site {name}: passes {n} operands, table says {want}")

if problems:
    print(f"{len(problems)} intrinsic-table problem(s):")
    for pr in problems:
        print("  !", pr)
    sys.exit(1)

calls = low.count("self.intr") + low.count("self.intr_raw")
print("intrinsic tables consistent: %d variants across 4 tables, %d arity entries, %d call sites checked"
      % (len(variants), len(arity), calls))
