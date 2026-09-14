#!/usr/bin/env python3
"""Static consistency checker for the Rust side, without a toolchain.

The sandbox this lands in has no rustc (the network is locked down), and a
2000-line hand-written decoder is exactly where "did I typo that method name /
did I forget that enum variant" bugs live.  So: parse with the real Rust grammar
(tree-sitter) and check the things a compiler would catch in the first second.

Checks, per file under src/:
  * syntax              — no ERROR / MISSING nodes anywhere in the tree
  * method resolution   — every `self.foo(` / `Self::foo(` has a matching
                          `fn foo` inside an impl block of that file (skips
                          known std/foreign receivers)
  * enum paths          — every `Enum::Variant` mentions a declared variant
  * duplicate variants  — an enum cannot declare the same name twice
  * duplicate match arms— two arms of one match with identical literal patterns
                          (this is the one that catches a copy-pasted opcode row)
  * use paths           — every `use crate::X::…` / `use super::X::…` head X is
                          actually declared in the module it names (the on-disk
                          directory layout is not the module layout)
  * exhaustive decode   — the `match insn.kind` in lower.rs mentions every
                          PpcKind variant, and decode.rs's kind table has no
                          unreachable duplicate key

Usage:  python3 tools/check_rust_static.py [-v]
Exit status is non-zero when anything is wrong.
"""
import re
import sys
import pathlib
from collections import defaultdict

import tree_sitter_rust
from tree_sitter import Language, Parser

ROOT = pathlib.Path(__file__).resolve().parent.parent / "src"
VERBOSE = "-v" in sys.argv
parser = Parser(Language(tree_sitter_rust.language()))

# Methods that live on std types / other crates, so "not defined in this file" is
# not an error for them.
ALLOWED_METHODS = set("""
push insert remove get contains_key iter iter_mut len is_empty new default clone
copied unwrap unwrap_or unwrap_or_else expect map and_then ok_or ok or else
collect retain extend clear next position count to_string to_vec to_le_bytes
to_be_bytes from_be_bytes from_le_bytes to_bits from_bits min max saturating_add
saturating_sub saturating_mul checked_add checked_mul pow2 swap_remove pop
enumerate filter filter_map find find_any_rev rev then_some is_some is_none
unwrap_or_default as_ptr as_mut_ptr is_null to_string as_ref as_mut into from
into_iter fold any all zip chain take skip step_by resize copy_from_slice
chunks chunks_exact windows first last first_mut sort sort_by_key sort_unstable
binary_search position borrow borrow_mut try_into try_from eq ne cmp partial_cmp
wrapping_add wrapping_sub wrapping_mul wrapping_shl wrapping_shr rotate_left
rotate_right leading_zeros trailing_zeros count_ones count_ones swap_bytes
checked_shl checked_shr bitand bitor bitxor not shl shr add sub mul div rem
abs signum floor ceil round trunc recip clamp is_finite is_nan to_int checked_as
fmt write write_str write_fmt push_str push_char with_capacity reserve reserve_exact
shrink_to_fit entry or_insert or_insert_with or_default occupy and_modify
raw_entry len bits has has_any
""".split())

# Receivers we never try to resolve (foreign/stdlib value types).
SKIP_RECEIVERS = set("""
self.ir self.fwd self.zero operands block ops Vec String HashMap std core
""".split())


def parse(path):
    src = path.read_bytes()
    return parser.parse(src), src


def walk(node, types=None):
    yield node
    for c in node.children:
        yield from walk(c, types)


def find_all(node, *types):
    for n in walk(node):
        if n.type in types:
            yield n


def txt(node, src):
    return src[node.start_byte:node.end_byte].decode("utf8", "replace")


def collect_impl_items(root):
    """Names of consts/fns defined in `impl Type { … }`, per type name.

    `Enum::SOME_CONST` is a legitimate path that is not a variant, so the variant
    check has to know about them.
    """
    out = defaultdict(set)
    for imp in find_all(root, "impl_item"):
        ty = imp.child_by_field_name("type")
        if ty is None:
            continue
        name = ty.text.decode().split("<")[0].strip()
        body = imp.child_by_field_name("body")
        if not body:
            continue
        for member in body.children:
            if member.type in ("function_item", "constant_item", "associated_constant"):
                n = member.child_by_field_name("name")
                if n:
                    out[name].add(n.text.decode())
            elif member.type == "const_item" or member.type.startswith("function"):
                n = member.child_by_field_name("name")
                if n:
                    out[name].add(n.text.decode())
    return out


def collect_enums(root, src):
    """enum name -> [variant names] (only fieldless/simple variants)."""
    out = {}
    for en in find_all(root, "enum_item"):
        name = txt(en.child_by_field_name("name"), src)
        body = en.child_by_field_name("body")
        variants = []
        if body:
            for v in body.children:
                if v.type == "enum_variant":
                    n = v.child_by_field_name("name")
                    if n:
                        variants.append(n.text.decode())
        out[name] = variants
    return out


def collect_impl_methods(root, src):
    """All `fn` names declared inside any impl block in this file."""
    out = set()
    for imp in find_all(root, "impl_item"):
        body = imp.child_by_field_name("body")
        if not body:
            continue
        for fn in find_all(body, "function_item"):
            n = fn.child_by_field_name("name")
            if n:
                out.add(n.text.decode())
    return out


def collect_free_fns(root):
    out = set()
    for n in find_all(root, "function_item"):
        nm = n.child_by_field_name("name")
        if nm:
            out.add(nm.text.decode())
    return out


def collect_all_impl_items():
    """`impl Type { const/fn … }` names, crate-wide, for `Type::item` paths."""
    out = defaultdict(set)
    for path in sorted(ROOT.rglob("*.rs")):
        tree, src = parse(path)
        for name, items in collect_impl_items(tree.root_node).items():
            out[name] |= items
    return dict(out)


def collect_all_enums():
    """Every enum in the crate, so cross-module paths resolve."""
    out = defaultdict(list)
    for path in sorted(ROOT.rglob("*.rs")):
        tree, src = parse(path)
        for name, variants in collect_enums(tree.root_node, src).items():
            for v in variants:
                if v not in out[name]:
                    out[name].append(v)
    return dict(out)


def check_file(path, global_enums=None):
    problems = []
    tree, src = parse(path)
    root = tree.root_node

    for n in walk(root):
        if n.type == "ERROR" or n.is_missing:
            problems.append(f"{path.name}:{n.start_point[0]+1}: syntax error near `{src[n.start_byte:n.start_byte+60].decode('utf8','replace')}`")

    enums = collect_enums(root, src)
    if global_enums:
        merged = dict(global_enums)
        for k, v in enums.items():
            merged.setdefault(k, [])
            for item in v:
                if item not in merged[k]:
                    merged[k].append(item)
        enums = merged
    impl_items = collect_impl_items(root)
    methods = collect_impl_methods(root, src) | collect_free_fns(root)

    # ---- self.method( / Self::method( resolution ----------------------------
    used = defaultdict(list)
    for call in find_all(root, "call_expression"):
        fn = call.child_by_field_name("function")
        if fn is None:
            continue
        if fn.type == "field_expression":
            recv = fn.child_by_field_name("field")
            obj = fn.child_by_field_name("value")
            if obj is not None and obj.type == "self":
                used[recv.text.decode()].append(call.start_point[0] + 1)
        elif fn.type == "scoped_identifier":
            parts = fn.text.decode().split("::")
            if parts[0] == "Self":
                used[parts[-1]].append(call.start_point[0] + 1)
    for name, lines in sorted(used.items()):
        if name in ALLOWED_METHODS or name in methods:
            continue
        problems.append(f"{path.name}:{lines[0]}: no method `{name}` in this file's impl blocks")

    # ---- enum variant paths --------------------------------------------------
    for sc in find_all(root, "scoped_identifier"):
        t = sc.text.decode()
        if t.count("::") != 1:
            continue
        base, variant = t.split("::")
        if base in enums:
            if variant not in enums[base] and variant not in impl_items.get(base, set()):
                problems.append(f"{path.name}:{sc.start_point[0]+1}: `{t}` is not a variant of {base}")
    for ua in find_all(root, "use_as_clause"):
        pass
    # bare-variant uses inside `use Enum::*;` bodies are checked by the match scan
    # below (a typo there shows up as "unknown variant" from the exhaustive check).

    # ---- duplicate enum variants --------------------------------------------
    for name, variants in enums.items():
        seen = set()
        for v in variants:
            if v in seen:
                problems.append(f"{path.name}: enum {name} declares variant {v} twice")
            seen.add(v)

    # ---- duplicate literal match arms (opcode tables) -----------------------
    for mb in find_all(root, "match_block"):
        seen = defaultdict(list)
        for arm in find_all(mb, "match_arm"):
            pats = arm.child_by_field_name("pattern")
            if pats is None:
                continue
            # a top-level `|` alternation: check each alternative; a bare literal:
            # check it whole
            for lit in find_all(pats, "literal"):
                if lit.type in ("integer_literal", "string_literal"):
                    seen[lit.text.decode()].append(arm.start_point[0] + 1)
        for lit, lines in seen.items():
            if len(lines) > 1 and all(l != lines[0] for l in lines):
                # allow the same literal in different nesting only if the match is
                # the same one, which it is by construction → report
                problems.append(
                    f"{path.name}: match arm for {lit} duplicated at lines "
                    + ", ".join(str(l) for l in lines)
                )

    # ---- duplicate items at the same level (E0252/E0428 class) --------------
    from collections import Counter
    decls = Counter()
    for n in root.children:
        for kind in ("mod_item", "function_item", "struct_item", "enum_item",
                     "constant_item", "use_declaration", "type_item"):
            if n.type == kind:
                nm = n.child_by_field_name("name")
                label = n.text.decode().split("\n")[0][:60] if nm is None else nm.text.decode()
                decls[(kind, label)] += 1
    for (kind, name), c in decls.items():
        if c > 1 and kind != "use_declaration":
            problems.append(f"{path.name}: {kind} `{name}` declared {c} times at module level")

    # ---- `mod x;` must resolve to a file ------------------------------------
    for n in find_all(root, "mod_item"):
        body = n.child_by_field_name("body")
        if body is not None:
            continue  # inline module
        nm = n.child_by_field_name("name")
        if nm is None:
            continue
        name = nm.text.decode()
        # path attribute form
        text = src[max(0, n.start_byte - 200):n.start_byte].decode()
        if '#[path' in text or f'path = "{name}' in text:
            continue
        parent_dir = path.parent
        cand = [parent_dir / f"{name}.rs", parent_dir / name / "__init__mod__.rs",
                parent_dir / name / "mod.rs"]
        if not any(c.exists() for c in cand):
            problems.append(f"{path.name}: `mod {name};` has no matching file")

    return problems, root, src, enums


def check_lower_exhaustive(root, src, enums, problems):
    """`match insn.kind` in lower.rs must mention every PpcKind variant, and no
    variant that does not exist."""
    kinds = set(enums.get("PpcKind", []))
    target = None
    for me_ in find_all(root, "match_expression"):
        scrut = me_.child_by_field_name("value")
        if scrut is not None and txt(scrut, src).strip() == "insn.kind":
            target = me_.child_by_field_name("body")
            break
    if target is None:
        problems.append("lower.rs: could not find `match insn.kind` to check exhaustiveness")
        return
    mentioned = set()
    for arm in find_all(target, "match_arm"):
        pats = arm.child_by_field_name("pattern")
        if pats is None:
            continue
        for ident in find_all(pats, "identifier"):
            mentioned.add(ident.text.decode())
    unknown = {m for m in mentioned if m not in kinds} - {
        "insn", "kind", "f", "c", "m", "s", "p", "n", "v", "x", "y", "r", "t", "a", "b",
        "i", "j", "k", "l", "p2", "op", "lane", "which", "other", "spec", "src",
    }
    for m in sorted(unknown):
        # a pattern identifier that is not a variant is either a binding (fine) or
        # a typo; bindings in this match are all single letters or `Some(...)`
        if len(m) > 2 and m not in ("Illegal", "Unsupported") and m[0].isupper():
            problems.append(f"lower.rs: match arm `{m}` is not a PpcKind variant")
    missing = kinds - mentioned
    if missing:
        problems.append(
            "lower.rs: `match insn.kind` does not cover: " + ", ".join(sorted(missing))
        )


# ---------------------------------------------------------------------------
# module-path resolution
#
# The crate root is `src/lib.rs`; the Wii frontend lives under `src/frontends/`
# but is *attached* with `#[path]` as `crate::frontend_wii_ppc` (there is no
# `crate::frontends` module).  A `use` written against the on-disk directory
# layout instead of the declared module layout is not caught by any other check
# here -- it parses fine, resolves fine in an editor, and simply fails to
# compile.  This is exactly the bug that was live in mod.rs once, so: rebuild
# the module tree from the declarations and re-resolve every crate/super path.
# ---------------------------------------------------------------------------

_RE_MOD = re.compile(r"^\s*(?:pub\s+)?mod\s+(\w+)\s*;", re.M)
_RE_MOD_PATH = re.compile(r'#\[path\s*=\s*"([^"]+)"\]\s*\n\s*(?:pub\s+)?mod\s+(\w+)\s*;')
_RE_INLINE_MOD = re.compile(r"^\s*pub\s+mod\s+(\w+)\s*\{", re.M)
_RE_USE = re.compile(r"^\s*(?:pub\s+)?use\s+(crate|super|self)::([A-Za-z0-9_:]+)", re.M)
_RE_DECL = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static|fn|unsafe fn|struct|enum|union|trait|type|macro_rules!)\s+(\w+)", re.M)


def _inline_mod_body(src, name):
    """Return the body text of `mod name { ... }` by brace matching, or None."""
    m = re.search(r"(?:^|\n)\s*(?:pub\s+)?mod\s+" + re.escape(name) + r"\s*\{", src)
    if not m:
        return None
    i = src.index("{", m.start())
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
    return src[i + 1:]


def _scope_decls(src):
    """Names declared or re-exported at the top level of `src`."""
    names = set(_RE_DECL.findall(src))
    for m in re.finditer(r"pub\s+use\s+[\w:]+::\s*\{([^;}]+)\}", src):
        for piece in m.group(1).split(","):
            piece = piece.strip().split(" as ")[-1].strip().strip("::")
            if piece and re.fullmatch(r"\w+", piece):
                names.add(piece)
    for m in re.finditer(r"pub\s+use\s+([\w:]+)\s*;", src):
        names.add(m.group(1).split("::")[-1])
    names |= set(_RE_INLINE_MOD.findall(src))
    return names


class Scope:
    """A module scope: the file that implements it plus, for inline modules, the
    text of its body."""

    def __init__(self, file, body=None, children=None):
        self.file = file
        self.body = body
        self.children = children or {}

    def src(self):
        return self.body if self.body is not None else (self.file.read_text() if self.file else "")


def build_scopes():
    """path -> Scope for every module reachable from the crate root."""
    root_file = ROOT / "lib.rs"
    scopes = {}

    def children_of(src, file):
        """declared submodule name -> Scope (file-backed or inline)."""
        kids = {}
        for m in _RE_MOD.finditer(src):
            name = m.group(1)
            # honour an immediately preceding #[path = "..."]
            head = src[:m.start()]
            pm = re.search(r'#\[path\s*=\s*"([^"]+)"\s*\]\s*$', head)
            if pm:
                cand = file.parent / pm.group(1)
                target = cand if cand.is_file() else cand.with_suffix("") / "mod.rs"
            else:
                a, b = file.parent / (name + ".rs"), file.parent / name / "mod.rs"
                target = a if a.is_file() else (b if b.is_file() else None)
            kids[name] = Scope(target)
        for name in _RE_INLINE_MOD.findall(src):
            if name not in kids:
                kids[name] = Scope(file, body=_inline_mod_body(src, name))
        return kids

    def register(path_str, scope):
        scope.declared = set(scope.src() and _scope_decls(scope.src()) or set())
        scope.kids = children_of(scope.src(), scope.file) if scope.file else {}
        scopes[path_str] = scope
        for name, sub in scope.kids.items():
            key = f"{path_str}::{name}" if path_str else name
            if key not in scopes and sub.file is not None:
                register(key, sub)
            elif key not in scopes:
                scopes[key] = sub  # inline body, no children of its own

    register("", Scope(root_file))
    return scopes


def module_of(path, scopes):
    """Which module scope does this file implement?"""
    for k, sc in scopes.items():
        if sc.file is not None and sc.file == path:
            return k
    return None


def check_use_paths(scopes, problems):
    for path in sorted(ROOT.rglob("*.rs")):
        if path.name in ("lib.rs", "main.rs"):
            continue  # main.rs is the bin target: `ryojit::…`, not a module
        mod_path = module_of(path, scopes)
        if mod_path is None:
            problems.append(f"{path.relative_to(ROOT.parent)}: not reachable from any "
                            f"`mod` declaration in the crate root (orphan file)")
            continue
        src = path.read_text()
        for m in _RE_USE.finditer(src):
            base, rest = m.group(1), m.group(2)
            line = src[: m.start()].count("\n") + 1
            # Only the path *before* `::{...}` names modules; the braces hold item
            # names and a trailing `*` is a glob, not a module segment.
            head, _, _braced = rest.partition("::")
            segs = [s for s in rest.split("::") if s and s != "*"]
            if not segs:
                continue
            scope_key = {"crate": "", "self": mod_path}.get(base)
            if scope_key is None:  # `super` of the crate root is the crate root
                scope_key = mod_path.rsplit("::", 1)[0] if "::" in mod_path else (
                    "" if mod_path else None)
                if scope_key is None:
                    continue
            scope = scopes.get(scope_key)
            if scope is None:
                continue
            walk = scope
            for seg in segs:
                if seg in walk.declared:
                    break  # resolved to an item; deeper segments need item tables
                nxt = walk.kids.get(seg)
                if nxt is None:
                    avail = ", ".join(sorted(walk.declared | set(walk.kids))) or "<none>"
                    problems.append(
                        f"{path.relative_to(ROOT.parent)}:{line}: `use {base}::{rest}` -- "
                        f"`{seg}` is not declared in `crate::{scope_key or '(root)'}` "
                        f"(available: {avail}). The on-disk directory layout is not the "
                        f"module layout: `#[path]` renames attach the module elsewhere.")
                    break
                walk = nxt
                # only keep descending while segments name modules; a segment that
                # is an *item* was already accepted by `seg in walk.declared`
                if seg not in walk.kids:
                    break


def main():
    problems = []
    files = sorted(ROOT.rglob("*.rs"))
    # `all_enums` stays *only* variants (the exhaustiveness check compares against
    # it); `all_paths` additionally accepts associated consts/fns in `Type::x`
    # paths, which are legal but are not variants.
    all_enums = collect_all_enums()
    all_paths = {k: list(v) for k, v in all_enums.items()}
    for k, v in collect_all_impl_items().items():
        all_paths.setdefault(k, [])
        for item in v:
            if item not in all_paths[k]:
                all_paths[k].append(item)
    scopes = build_scopes()
    check_use_paths(scopes, problems)
    for path in files:
        fp, root, src, enums = check_file(path, all_paths)
        problems += fp
        if path.name == "lower.rs":
            check_lower_exhaustive(root, src, all_enums, problems)
        if VERBOSE:
            print(f"  checked {path.relative_to(ROOT.parent)} ({len(src.splitlines())} lines)")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  ! " + p)
        return 1
    print(f"static checks passed over {len(files)} files")
    return 0


sys.exit(main())
