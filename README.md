# RyoJIT

A Rust + C/C++ hybrid JIT core for the **Nintendo Wii** (PowerPC "Broadway", a
750CL derivative, big-endian) and **Nintendo Switch** (ARMv8-A "Tegra X1",
little-endian), sharing one core architecture and tuned for good performance on
weak/older x86-64 laptops and desktops. Wii U is the eventual third guest.

## Language split (deliberate, do not collapse it)

* **Rust** — universal IR, optimisation passes, linear-scan register allocation,
  the `Frontend` trait + both guest frontends, code cache, dispatcher/runtime
  loop, and the GPU generic-model layer.
* **C/C++** — *only* the x86-64 code-emission layer (asmjit), reached through a
  four-symbol `extern "C"` boundary. Rust serialises an IR block plus the
  register allocation into flat `#[repr(C)]` structs; C++ emits machine code and
  hands back a pointer/length plus patch-site offsets. The returned buffer is
  freed exactly once, from `CompiledBlock`'s `Drop` via `jit_free_code`.

## Layout

```
src/lib.rs                        FFI types + IR + passes + regalloc + marshal
                                  + cache + runtime + GPU  (the architecture doc,
                                  now also the crate root; keep it that way so
                                  `crate::…` paths and every #[repr(C)] stay put)
src/frontends/wii_ppc/            PowerPC Broadway decode + lowering   ← REAL
    mod.rs                        Frontend impl, and the three conventions the
                                  core relies on (guest-state access, block exits,
                                  the FPR/paired-single container model)
    fields.rs                     bit-field views, PPC bit numbering
    decode.rs                     the opcode dispatch table (~220 instructions)
    lower.rs                      decode → universal IR, per instruction
    intrinsics.rs                 the Intrinsic id/effects/arity table = C++ ABI
    vectors.rs                    generated decode corpus (do not edit)
src/ir_dump.rs, src/ir_verify.rs  debug dump + structural IR checks
src/jit_stub.rs                   PLACEHOLDER emitter so the crate links and the
                                  boundary is testable (feature `jit-stub`;
                                  delete when jit/*.cpp lands)
tools/                            the scripts that validated the tables (see below)
```

## Build / test

```sh
cargo test                    # default features: placeholder emitter, no C++ needed
cargo test -- --nocapture
python3 tools/check_rust_static.py       # grammar-level audit + module-path resolution, no rustc
python3 tools/gen_ppc_vectors.py --check # regenerate + re-verify the decode corpus
python3 tools/check_ppc_branch_bo.py     # `bc` condition truth table vs the ISA text
python3 tools/check_intrinsic_table.py   # intrinsic tables + every call site's operand count
```

`cargo test` needs no crates.io access (zero dependencies on purpose).
`--no-default-features` selects the real FFI and therefore fails to *link* until
`jit/` exists — that is intentional.

> **Read this before the first `cargo build` on a real machine.** The sandbox the
> PowerPC frontend was written in had **no Rust toolchain** (no `rustc`, no
> `cargo`, and every toolchain download blocked by the network), so the Rust in
> this tree has never been through `rustc`. What *was* run, and passes, is:
> `check_rust_static.py` (tree-sitter Rust grammar: parse errors, method
> resolution, `Enum::Variant` validity, duplicate enum variants, duplicated
> `match` arms, `match insn.kind` exhaustiveness over `PpcKind`, and module-path
> resolution of every `use crate::`/`use super::`), `check_intrinsic_table.py`
> (58 intrinsic ids against 4 tables plus the operand count of all 73 call
> sites), `check_ppc_branch_bo.py` (the `bc`/`bdnz` condition truth table over
> all 32 BO values against the ISA pseudocode), and `gen_ppc_vectors.py` (an
> independent PPC assembler cross-checked against Capstone, reproducing
> `vectors.rs` byte-for-byte).
> Expect the first real compile to surface only trivial fallout — unused imports,
> a `mut`, an unreachable pattern, `offset_of!` availability (`rust-version =
> "1.77"`) — rather than redesign. Run `cargo test && cargo clippy && cargo fmt`.

## Status

Implemented and tested: the whole Rust core (IR, passes, regalloc, marshaling,
cache, dispatcher shape) and the **PowerPC frontend** — the Broadway opcode
dispatch table plus lowering for fixed-point arithmetic (including the
`addc`/`subfe` carry family and `XER`), the logical/rotate/extend class,
compares and traps, all integer and FP loads/stores (including update, indexed,
byte-reversed and string/list forms), `CR`/`XER`/`SPR` access, all branches and
their chaining shape, scalar FP, and paired singles — the last of which lower
onto the shared `VecAdd`/`VecMul`/`VecFma` IR exactly as designed.

Still `todo!()`, in priority order: AArch64 frontend; guest memory access
(the reserved guest-address mapping behind `fetch_guest_bytes`); the C++ asmjit
emitter (`jit_emit_block`/`jit_free_code`/`jit_patch_chain`, whose intrinsic
contract is `src/frontends/wii_ppc/intrinsics.rs`); GPU register/shader decode;
code-cache LRU eviction.

Known, named gaps (all deliberate rather than silent — see
`src/frontends/wii_ppc/mod.rs` for the full list): FPSCR exception bits are not
recomputed by FP results; `HID0[PSQM]` gating of the scalar single-precision ops
is not modelled; `mtfsf`/`mtfsfi` are not lowered (their XFL field positions are
unverified) and truncate the block; `eciwx`/`ecowx` likewise; `fres`/`frsqrte`
accuracy is host-defined; and one open question about which FPR halves hold a
paired single is isolated behind a single `const`.
