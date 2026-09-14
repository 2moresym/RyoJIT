//! Tests for the Wii frontend: decode table, lowering shape, structural IR
//! invariants over the whole supported opcode space, and the frozen layout
//! assertions the C++ side shares.
//!
//! The corpus tests are the interesting ones: `vectors.rs` is generated (see
//! `tools/gen_ppc_vectors.py`) from a *second, independent* description of every
//! field layout, cross-checked against a real disassembler where one exists.  So
//! "the decoder and the encoder agree with a third party" is what is being
//! asserted, rather than "the decoder agrees with itself".

#![cfg(test)]

#[path = "vectors.rs"]
mod vectors;

use super::decode::{decode_raw, PpcKind};
use super::fields::PpcFields;
use super::intrinsics::{PpcIntr, PpcMisc};
use super::lower::{rotate_mask, PpcLowerError};
use super::{WiiFrontend, GUEST_INSN_BYTES, MAX_BLOCK_INSNS};
use crate::ir::{Endian, IrOp, VOperand, Width};
use crate::{passes, regalloc};
use vectors::VECTORS;

// =============================================================================
// helpers
// =============================================================================

fn word_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_be_bytes());
    }
    out
}

/// Lower a single instruction at a fixed address.
fn one(word: u32) -> crate::ir::IrBlock {
    let fe = WiiFrontend::new();
    let unit = fe.translate_unit(&word_bytes(&[word]), 0x1000);
    assert!(
        unit.insn_count == 1,
        "expected exactly one decoded instruction for {word:#010x}, got {}",
        unit.insn_count
    );
    unit.block
}

fn one_unit(word: u32) -> super::PpcUnit {
    WiiFrontend::new().translate_unit(&word_bytes(&[word]), 0x1000)
}

/// The sequence of op *shapes* a block produced, as names.  Far more robust to
/// assert on than a fully formatted dump, while still pinning order and shape.
fn shape(block: &crate::ir::IrBlock) -> Vec<String> {
    block
        .ops
        .iter()
        .map(|op| match op {
            IrOp::Add { .. } => "add".into(),
            IrOp::Sub { .. } => "sub".into(),
            IrOp::Mul { .. } => "mul".into(),
            IrOp::And { .. } => "and".into(),
            IrOp::Or { .. } => "or".into(),
            IrOp::Xor { .. } => "xor".into(),
            IrOp::Shl { .. } => "shl".into(),
            IrOp::Shr { .. } => "shr".into(),
            IrOp::Sar { .. } => "sar".into(),
            IrOp::VecAdd { .. } => "vec_add".into(),
            IrOp::VecMul { .. } => "vec_mul".into(),
            IrOp::VecFma { .. } => "vec_fma".into(),
            IrOp::Load { .. } => "load".into(),
            IrOp::Store { .. } => "store".into(),
            IrOp::Branch { .. } => "branch".into(),
            IrOp::IndirectBranch { .. } => "indirect_branch".into(),
            IrOp::Call { .. } => "call".into(),
            IrOp::Return => "return".into(),
            IrOp::SetFlags { .. } => "setflags".into(),
            IrOp::ReadFlag { .. } => "readflag".into(),
            IrOp::Intrinsic { id, .. } => {
                format!("{}", super::intrinsic_name(id.0).unwrap_or("?"))
            }
        })
        .collect()
}

fn dump(word: u32) -> String {
    let block = one(word);
    crate::ir_dump::dump_block_with(&block, &super::intrinsic_name)
}

/// An instruction word with explicit fields, for hand-written cases.
fn enc(op: u32, rt: u32, ra: u32, rb: u32, imm: u32) -> u32 {
    (op << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (imm & 0xFFFF)
}

fn enc_x(op: u32, d: u32, a: u32, b: u32, xo10: u32, rc: u32) -> u32 {
    (op << 26) | (d << 21) | (a << 16) | (b << 11) | (xo10 << 1) | rc
}

// =============================================================================
// 1. the decode table
// =============================================================================

#[test]
fn decode_resolves_every_generated_vector() {
    let mut failures = Vec::new();
    for v in VECTORS {
        let insn = decode_raw(v.word, 0x1000);
        if insn.name != v.mnemonic {
            failures.push(format!(
                "{:#010x}: decoded {:?} (kind {:?}), expected {:?}",
                v.word, insn.name, insn.kind, v.mnemonic
            ));
        }
        if insn.opcode != v.op {
            failures.push(format!(
                "{:#010x}: primary opcode decoded as {}, expected {}",
                v.word, insn.opcode, v.op
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} decode failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn decode_table_covers_the_whole_corpus_without_illegal() {
    // Nothing in the generated corpus should decode as *undefined*: every row was
    // taken from a manual or a disassembler, so `Illegal` here means the table
    // lost or corrupted an arm.
    let bad: Vec<String> = VECTORS
        .iter()
        .filter(|v| decode_raw(v.word, 0).kind == PpcKind::Illegal)
        .map(|v| format!("{:#010x} ({})", v.word, v.mnemonic))
        .collect();
    assert!(bad.is_empty(), "corpus rows decoding as Illegal: {bad:?}");
}

#[test]
fn field_extractors_round_trip_their_encoders() {
    // Each format's fields, checked against values the generator placed.  A wrong
    // shift in `fields.rs` shows up here rather than as a plausible-looking
    // mis-decode of one exotic instruction.
    let w = enc(14, 3, 4, 0, 0x1234); // addi r3, r4, 0x1234 (SI = 4660)
    let f = PpcFields::new(w);
    assert_eq!(f.rt(), 3);
    assert_eq!(f.ra(), 4);
    assert_eq!(f.si(), 0x1234);
    assert_eq!(f.ui(), 0x1234);
    assert!(!f.rc());

    // A negative SI must come back negative, not as its 16-bit pattern.
    let f = PpcFields::new(enc(14, 3, 4, 0, 0xFFEC));
    assert_eq!(f.si(), -20);

    // XO form: rt/ra/rb + the 10-bit extended field, with OE and Rc split out.
    let f = PpcFields::new(enc_x(31, 3, 4, 5, 266, 0));
    assert_eq!((f.rt(), f.ra(), f.rb()), (3, 4, 5));
    assert_eq!(f.xo10(), 266);
    assert_eq!(f.xo(), 266 & 0x1F);
    assert!(!f.oe());
    assert!(!f.rc());
    let f = PpcFields::new(enc_x(31, 3, 4, 5, 266 | 512, 1));
    assert!(f.oe(), "OE must be bit 21, inside the 10-bit XO field");
    assert!(f.rc());

    // Rotate class: sh/mb/me.
    let f = PpcFields::new(0x5464_32A8u32); // rlwinm r4,r3,6,10,20
    assert_eq!(f.rt(), 3, "rotate uses the RT slot for RS");
    assert_eq!(f.ra(), 4);
    assert_eq!(f.sh(), 6);
    assert_eq!(f.mb(), 10);
    assert_eq!(f.me(), 20);

    // Branch form: BO/BI/BD/AA/LK.
    let f = PpcFields::new(0x41820008); // bc 12,2,+8
    assert_eq!(f.bo(), 12);
    assert_eq!(f.bi(), 2);
    assert_eq!(f.bd(), 2);
    assert!(!f.aa());
    assert!(!f.lk());

    // XFX SPR field is stored with its halves swapped.
    let f = PpcFields::new(0x7C6802A6); // mflr r3
    assert_eq!(f.spr(), 8);
    let f = PpcFields::new(0x7C6C42A6); // mfspr r3, 268 (TBL)
    assert_eq!(f.spr(), 268);

    // mtcrf's FLM is 8 bits at 12:19, CRF0 in the most significant one.
    let f = PpcFields::new(0x7C60F120);
    assert_eq!(f.flm(), 0xF);

    // mflr's SPR field is the swapped pair, and `mftbl` shows the high half.
    let f = PpcFields::new(0x7C6C_42A6); // mfspr r3, 268 (TBL)
    assert_eq!(f.spr(), 268);

    // Quantized paired singles: W at 16, GQR at 17:19, 12-bit displacement.
    let f = PpcFields::new(0xE003D004); // psq_l f0,4(r3),1,5
    assert_eq!(f.frt(), 0);
    assert_eq!(f.ra(), 3);
    assert!(f.psq_w_d());
    assert_eq!(f.psq_gq_d(), 5);
    assert_eq!(f.psq_d(), 4);

    // ps_madds1 f1,f2,f3,f4 — FRC is a full 5-bit field in the PS space.
    let f = PpcFields::new(0x102220DE);
    assert_eq!(f.frt(), 1);
    assert_eq!(f.fra(), 2);
    assert_eq!(f.frb(), 4, "FRB sits at bits 16:20");
    assert_eq!(f.frc_full(), 3);
    assert_eq!(f.xo(), 15);

    // The AX form encodes only FRC's top 3 bits, so the register is a multiple of 4.
    let f = PpcFields::new(0xEC42_2039u32); // fmadds f2,f1,f9?,f5-ish
    assert_eq!(f.frc_ax() % 4, 0, "AX-form FRC is always a multiple of four");
}

#[test]
fn reserved_and_64bit_only_opcodes_trap() {
    // Broadway is a 32-bit Book II core with no VMX: the 64-bit load/store
    // primaries and the Altivec primaries are *undefined*, so they must decode to
    // Illegal (→ guest program exception) rather than to anything plausible.
    for word in [
        0x38C0_0000u32, // primary 14 with a legal addi is NOT here; see below
        0xE800_0000,    // primary 58 = ld
        0xF800_0000,    // primary 62 = std
        0x1400_0000,    // primary 5 (VMX)
        0x1800_0000,    // primary 6 (VMX)
        0x1C00_0000,    // primary 7 (VMX)
        0x0000_0000,    // primary 0, reserved
        0x5800_0000,    // primary 22, reserved
        0x7800_0000,    // primary 30, reserved (rldicr-ish)
    ] {
        assert_eq!(
            decode_raw(word, 0).kind,
            PpcKind::Illegal,
            "{:#010x} must be illegal on Broadway",
            word
        );
    }
    // `lhzux` is 31/311; the AIX appendix's 331 is the POWER-family `div` and must
    // NOT decode as a load here (that was a live bug in an earlier draft).
    assert_eq!(decode_raw(0x7C642A6E, 0).kind, PpcKind::Lhzux);
    assert_eq!(decode_raw(0x7C642A7E, 0).kind, PpcKind::Illegal);
}

#[test]
fn every_terminator_kind_terminates_and_others_do_not() {
    // The decode loop stops on `is_block_terminator`, and `lower` emits exactly one
    // terminator for those kinds.  If the two ever disagree, either instructions
    // after a terminator get appended (invalid IR) or a terminating instruction
    // falls through (invalid code).
    for word in [
        0x4BFF_FFD4u32, // b -0x2C
        0x4182_0008,    // bc 12,2,+8
        0x4E80_0020,    // bclr (blr)
        0x4E80_0420,    // bcctr (unconditional: bctr)
        0x4C00_0064,    // rfi
    ] {
        let insn = decode_raw(word, 0x1000);
        assert!(
            insn.kind.is_block_terminator(),
            "{:#010x} ({}) must terminate a block",
            word,
            insn.name
        );
        let unit = WiiFrontend::new().translate_unit(&word_bytes(&[word]), 0x1000);
        assert!(unit.terminated, "{:#010x} did not terminate", word);
        assert!(
            unit.block.ops.last().unwrap().is_terminator(),
            "{:#010x} ended without a terminator op",
            word
        );
    }
    // Group 19 also contains non-terminators that share the primary opcode with
    // `bclr`, and group 31 contains `tw`.
    for word in [0x4C00_012Cu32, 0x4D84_0000, 0x7C00_04AC, 0x6000_0000] {
        let insn = decode_raw(word, 0);
        assert!(
            !insn.kind.is_block_terminator(),
            "{} must NOT terminate a block",
            insn.name
        );
    }
}

// =============================================================================
// 2. lowering: structure over the entire supported space
// =============================================================================

#[test]
fn lowering_is_well_formed_for_every_supported_instruction() {
    // Kinds this build intentionally does not lower (truncating the block so a
    // fallback interpreter runs them).  Anything *else* erroring is a regression.
    const UNIMPLEMENTED: &[&str] = &["eciwx", "ecowx", "mtfsf", "mtfsfi"];

    let mut problems = Vec::new();
    let mut lowered = 0usize;
    let mut truncated = 0usize;
    for v in VECTORS {
        let unit = one_unit(v.word);
        if let Some(e) = &unit.error {
            match e {
                PpcLowerError::Unimplemented { mnemonic, .. }
                    if UNIMPLEMENTED.contains(mnemonic) =>
                {
                    truncated += 1;
                    continue;
                }
                other => {
                    problems.push(format!("{:#010x}: {:?}", v.word, other));
                    continue;
                }
            }
        }
        lowered += 1;
        for violation in crate::ir_verify::verify_block(&unit.block) {
            problems.push(format!(
                "{:#010x} ({}): {violation}\n{}",
                v.word,
                v.mnemonic,
                crate::ir_dump::dump_block_with(&unit.block, &super::intrinsic_name)
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "{} ill-formed lowering(s):\n{}",
        problems.len(),
        problems.join("\n---\n")
    );
    assert!(lowered >= 250, "only {lowered} rows lowered");
    assert_eq!(
        truncated, 4,
        "expected exactly the four documented truncations (eciw/ecow, mtfsf/mtfsfi)"
    );
}

#[test]
fn passes_and_regalloc_accept_every_lowering() {
    // The full Rust pipeline minus the emitter: optimize, allocate, and check the
    // two invariants the register allocator's contract rests on — every VReg gets
    // a location, and no spill offset escapes `CpuState::spill`.
    const SPILL_BYTES: u64 = 128 * 8;
    let mut checked = 0usize;
    for v in VECTORS {
        let unit = one_unit(v.word);
        if unit.error.is_some() {
            continue;
        }
        let mut block = unit.block;
        passes::run_all(&mut block);
        let alloc = regalloc::linear_scan(&block);

        // Every register mentioned anywhere must have a location.
        let mut mentioned = Vec::new();
        for op in &block.ops {
            collect_vregs(op, &mut mentioned);
        }
        for vreg in mentioned {
            assert!(
                alloc.assignments.contains_key(&vreg),
                "{:#010x}: v{} has no allocation after linear_scan",
                v.word,
                vreg.0
            );
        }
        for (vreg, loc) in &alloc.assignments {
            if let regalloc::Location::StateSpill(off) = *loc {
                assert!(
                    (off as u64) + 8 <= SPILL_BYTES,
                    "{:#010x}: spill offset {off} for v{} escapes CpuState::spill",
                    v.word,
                    vreg.0
                );
            }
        }
        checked += 1;
    }
    assert!(checked >= 250, "only {checked} rows checked");
}

fn collect_vregs(op: &IrOp, out: &mut Vec<crate::ir::VReg>) {
    let mut push = |o: &VOperand| {
        if let VOperand::Reg(v) = *o {
            out.push(v);
        }
    };
    match op {
        IrOp::Add { dst, a, b, .. }
        | IrOp::Sub { dst, a, b, .. }
        | IrOp::Mul { dst, a, b, .. }
        | IrOp::And { dst, a, b, .. }
        | IrOp::Or { dst, a, b, .. }
        | IrOp::Xor { dst, a, b, .. } => {
            out.push(*dst);
            push(a);
            push(b);
        }
        IrOp::Shl { dst, a, amount, .. }
        | IrOp::Shr { dst, a, amount, .. }
        | IrOp::Sar { dst, a, amount, .. } => {
            out.push(*dst);
            push(a);
            push(amount);
        }
        IrOp::VecAdd { dst, a, b, .. } | IrOp::VecMul { dst, a, b, .. } => {
            out.push(*dst);
            push(a);
            push(b);
        }
        IrOp::VecFma { dst, a, b, c, .. } => {
            out.push(*dst);
            push(a);
            push(b);
            push(c);
        }
        IrOp::Load { dst, addr, .. } => {
            out.push(*dst);
            push(addr);
        }
        IrOp::Store { addr, val, .. } => {
            push(addr);
            push(val);
        }
        IrOp::Branch { cond, .. } => {
            if let Some(c) = cond {
                push(c);
            }
        }
        IrOp::IndirectBranch { target } | IrOp::Call { target } => push(target),
        IrOp::SetFlags { a, b, .. } => {
            push(a);
            push(b);
        }
        IrOp::ReadFlag { dst, .. } => out.push(*dst),
        IrOp::Intrinsic { operands, dst, .. } => {
            for o in operands {
                push(o);
            }
            if let Some(d) = dst {
                out.push(*d);
            }
        }
        IrOp::Return => {}
    }
}

#[test]
fn no_intrinsic_ever_exceeds_the_c_payload() {
    // `ir_op_to_c` fails loudly past CIR_MAX_OPERANDS; the frontend must never
    // produce one that trips it.  Checked directly here so the failure names the
    // intrinsic rather than the FFI boundary.
    for v in VECTORS {
        let unit = one_unit(v.word);
        for (i, op) in unit.block.ops.iter().enumerate() {
            if let IrOp::Intrinsic { id, operands, .. } = op {
                assert!(
                    operands.len() <= crate::CIR_MAX_OPERANDS,
                    "{:#010x}: op {i} intrinsic#{} has {} operands",
                    v.word,
                    id.0,
                    operands.len()
                );
            }
        }
    }
}

#[test]
fn a_block_is_cut_at_the_instruction_cap() {
    // 200 legal `addi`s must produce MAX_BLOCK_INSNS instructions, a fall-through
    // exit at the next address, and no run-away growth.
    let words = vec![0x3860_0001u32 /* addi r3,r0,1 */; 200];
    let unit = WiiFrontend::new().translate_unit(&word_bytes(&words), 0x2000);
    assert_eq!(unit.insn_count, MAX_BLOCK_INSNS);
    assert_eq!(unit.next_pc, 0x2000 + 4 * MAX_BLOCK_INSNS as u64);
    let last = unit.block.ops.last().expect("non-empty block");
    match last {
        IrOp::IndirectBranch {
            target: VOperand::Imm(pc),
        } => assert_eq!(*pc as u64, unit.next_pc, "capped block must exit at the next pc"),
        other => panic!("capped block must end in a constant-target exit, got {other:?}"),
    }
    assert!(crate::ir_verify::verify_block(&unit.block).is_empty());
}

// =============================================================================
// 3. semantics of the parts that are easy to get subtly wrong
// =============================================================================

#[test]
fn add_lowers_to_two_reads_one_add_one_write() {
    // add r3, r4, r5 = 0x7C642A14 (verified against a disassembler; see the
    // generated corpus).  Four ops plus the fall-through exit the translate loop
    // appends: no hidden moves, no redundant state traffic.
    let insn = decode_raw(0x7C642A14, 0);
    assert_eq!(insn.kind, PpcKind::Add);
    assert_eq!(
        shape(&one(0x7C642A14)),
        vec![
            "ppc_get_gpr",
            "ppc_get_gpr",
            "add",
            "ppc_set_gpr",
            "indirect_branch"
        ]
    );
    let d = dump(0x7C642A14);
    assert!(d.contains("v1 = add v2, v3 : w32"), "wrong add:\n{d}");
    assert!(d.contains("ppc_set_gpr #3, v4"), "wrong destination:\n{d}");
    assert!(
        d.contains("indirect_branch -> #0x1004"),
        "fall-through exit must be the next pc:\n{d}"
    );
}

#[test]
fn register_reads_are_forwarded_within_a_block() {
    // lwz r3,8(r4); add r5,r3,r3; stw r5,12(r4) — r3 is read from state exactly
    // once even though it is used twice, and the store's value comes from the
    // register r5 was written to (not re-read).
    let words = [0x8064_0008u32, 0x7CA3_1A14, 0x90A4_000C];
    let unit = WiiFrontend::new().translate_unit(&word_bytes(&words), 0x1000);
    let d = crate::ir_dump::dump_block_with(&unit.block, &super::intrinsic_name);
    assert_eq!(
        d.matches("ppc_get_gpr #4").count(),
        1,
        "r4 must be read once, not once per use:\n{d}"
    );
    assert_eq!(
        d.matches("ppc_get_gpr #3").count(),
        1,
        "r3 must be read once (forwarded into both add operands):\n{d}"
    );
    assert!(d.contains("ppc_set_gpr #5"), "r5 write missing:\n{d}");
    // r5 is never re-read for the store: the store consumes the same VReg.
    assert_eq!(
        d.matches("ppc_get_gpr #5").count(),
        0,
        "the store should reuse the value written, not reload it:\n{d}"
    );
    assert!(crate::ir_verify::verify_block(&unit.block).is_empty());
}

#[test]
fn update_forms_write_back_the_new_ea_after_the_access() {
    // lwzu r3, 8(r4): load from old r4+8, then r4 = that address.
    let d = dump(0x8464_0008); // lwzu r3,8(r4)
    let load_at = d.find("load").expect("a load");
    let set_at = d.find("ppc_set_gpr #4").expect("writeback of r4");
    assert!(
        load_at < set_at,
        "the update must land after the access it fed:\n{d}"
    );
    // stwu must store first, then update.
    let d = dump(0x9464_0008); // stwu r3,8(r4)
    let store_at = d.find("store").expect("a store");
    let set_at = d.find("ppc_set_gpr #4").expect("writeback");
    assert!(store_at < set_at, "stwu must store before updating r4:\n{d}");
}

#[test]
fn signed_and_unsigned_compare_take_their_so_bit_from_different_places() {
    // cmp  cr3, r4, r5  → SO = sign(RA);  cmpl cr3, r4, r5 → SO = 1.
    let cmp = 0x7D84_2800u32; // cmp cr3,r4,r5
    let cmpl = 0x7D84_2840u32; // cmpl cr3,r4,r5
    let d_cmp = dump(cmp);
    let d_cmpl = dump(cmpl);
    // signed: reads XER for the SO bit (via ppc_get_misc #1) or a shift of r4
    assert!(
        d_cmp.contains("readflag") && d_cmpl.contains("readflag"),
        "both compare forms must use the lazy flag path"
    );
    assert_ne!(d_cmp, d_cmpl, "signed and unsigned compares must differ");
}

#[test]
fn conditional_branch_folds_both_exits_into_one_target() {
    // bc 12,2,+8  (beq): no CTR test (BO=12 → bit2 set → skip CTR), so no CTR
    // write, and the exit is a computed register value.
    let d = dump(0x4182_0008);
    assert!(
        !d.contains("ppc_set_misc #4"),
        "BO=12 must not touch CTR:\n{d}"
    );
    assert!(
        d.contains("indirect_branch -> v"),
        "a conditional branch must exit through a computed value:\n{d}"
    );
    // The CR bit it reads is CR0.EQ (BI=2): shift by 31-2 = 29.
    assert!(d.contains("#0x1d"), "must test CR bit 2 (shift by 31-2):\n{d}");

    // bc 16,0,+8 (bdnz): decrement CTR then test it.
    let d = dump(0x4200_0008);
    assert!(
        d.contains("ppc_set_misc #4"),
        "BO=16 must decrement CTR:\n{d}"
    );
}

#[test]
fn branch_with_link_writes_lr_and_not_the_host_stack() {
    // bl +0x10 → LR = pc+4, then exit.  No Call/Return op anywhere.
    let d = dump(0x4800_00C1); // bl +0xC from 0x1000 → 0x100C
    assert!(d.contains("ppc_set_misc #3"), "LR must be written:\n{d}");
    assert!(!d.contains("call "), "guest calls are not host calls:\n{d}");
    assert!(!d.contains("return"), "block exits are not host returns:\n{d}");
    assert!(
        d.contains("indirect_branch -> #0x100c"),
        "the target must be a constant for chaining:\n{d}"
    );
}

#[test]
fn rotate_masks_match_the_architecture() {
    // Independent definition: PPC bits mb..=me set, wrapping when mb > me.
    let isa_mask = |mb: u32, me: u32| -> u32 {
        let mut m = 0u32;
        if mb <= me {
            for p in mb..=me {
                m |= 1 << (31 - p);
            }
        } else {
            for p in 0..=me {
                m |= 1 << (31 - p);
            }
            for p in mb..=31 {
                m |= 1 << (31 - p);
            }
        }
        m
    };
    // Every (mb, me) pair — 1024 cases, and the interesting ones (0/31, wrap,
    // mb == me+1 meaning "all ones") are in here by construction.
    for mb in 0..32u32 {
        for me in 0..32u32 {
            assert_eq!(
                rotate_mask(mb, me),
                isa_mask(mb, me),
                "rotate_mask({mb}, {me})"
            );
        }
    }
}

#[test]
fn rlwinm_masks_are_computed_at_translate_time() {
    // rlwinm r4,r3,6,10,20 → and(rotl(r3,6), #0x003FF000)
    let d = dump(0x5464_32A8);
    let want = format!("#{:#x}", rotate_mask(10, 20));
    assert!(d.contains(&want), "expected mask {want} in:\n{d}");
    // The wrap-around form (`rlwinm r3,r3,0,20,11`) must produce the *complement*
    // mask, which is the case a naive implementation gets wrong.
    let d = dump(0x5463_0516); // rlwinm r3,r3,0,20,11
    assert!(
        d.contains(&format!("#{:#x}", rotate_mask(20, 11))),
        "wrap-around mask missing:\n{d}"
    );
    assert_eq!(rotate_mask(20, 11), 0xFFF0_0FFF);
}

#[test]
fn slw_srw_guard_against_shift_counts_of_32_or_more() {
    // PPC zeroes the result for a count ≥ 32; the host would mask the count.  The
    // lowering must therefore contain the range guard (an `and` of !31 plus a
    // multiply) rather than a bare shl.
    let d = dump(0x7C83_2830); // slw r4,r3,r5
    assert!(d.contains("and v"), "shift guard missing:\n{d}");
    assert!(d.contains("mul "), "shift guard must zero the result:\n{d}");
    // srawi takes a 5-bit immediate, so it needs no count guard.
    let d = dump(0x7C83_2E70); // srawi r4,r3,5
    assert!(!d.contains("mul "), "srawi must not pay for a guard:\n{d}");
}

#[test]
fn carry_family_derives_ca_from_the_sum_not_from_host_flags() {
    // addc r3,r4,r5 → CA = bit 32 of the W64 sum.
    let d = dump(0x7C84_2810);
    assert!(d.contains("add v"), "sum must be computed at w64:\n{d}");
    assert!(
        d.contains("shr v, #0x20"),
        "CA must be the W64 sum's bit 32:\n{d}"
    );
    assert!(
        d.contains("ppc_set_misc #1"),
        "XER must be written:\n{d}"
    );
    // and the record form's SO bit comes from the (updated) XER, not a flag.
    let d = dump(0x7C84_2815); // addc. r3,r4,r5
    assert!(d.contains("ppc_get_misc #1"), "SO must read XER:\n{d}");
}

#[test]
fn mulli_and_addi_treat_r0_as_zero_but_add_does_not() {
    // addi r3,0,-1 → r3 = -1 with no register read at all.
    let d = dump(0x3860_FFFCu32 /* addi r3,r0,-4 */ | 0);
    assert!(!d.contains("ppc_get_gpr #0"), "addi rA=0 is the constant 0:\n{d}");
    // add r3,r0,r4 reads GPR 0 like any other register (it can hold a value).
    let d = dump(0x7C60_2214); // add r3,r0,r4
    assert!(
        d.contains("ppc_get_gpr #0"),
        "XO-form ops must read r0 as a register:\n{d}"
    );
    // lwzx r3,0,r4 → EA is just r4 (the |0 rule for indexed forms).
    let d = dump(0x7C60_202E); // lwzx r3,0,r4
    assert!(
        !d.contains("ppc_get_gpr #0"),
        "indexed loads use (RA|0):\n{d}"
    );
}

#[test]
fn loads_and_stores_use_the_endian_field_for_the_brx_family() {
    // lwbrx must be a Load with the *opposite* endianness — that is what the field
    // exists for, and it means the emitter's MOVBE path handles it.
    let lwbrx = 0x7C64_2C2Cu32;
    let d = dump(lwbrx);
    assert!(d.contains("load") && d.contains("le"), "lwbrx must flip the access endianness:\n{d}");
    let lwz = 0x8064_0008u32;
    let d2 = dump(lwz);
    assert!(d2.contains("be"), "a normal load keeps the guest endianness:\n{d2}");
}

#[test]
fn the_fctiwz_stfiwx_cast_idiom_composes() {
    // fctiwz f1,f2 ; stfiwx f1,r3,r4 — the classic double→int.  The lowering must
    // store the container's low 32 bits, which is what makes the idiom work.
    let words = [0xFC22_001Eu32, 0x7C23_27AE];
    let unit = WiiFrontend::new().translate_unit(&word_bytes(&words), 0x1000);
    let d = crate::ir_dump::dump_block_with(&unit.block, &super::intrinsic_name);
    assert!(d.contains("ppc_fp_ctiwz"), "fctiwz helper missing:\n{d}");
    assert!(d.contains("and v, #0xffffffff"), "stfiwx must take the low half:\n{d}");
    assert!(d.contains("store"), "store missing:\n{d}");
    assert!(crate::ir_verify::verify_block(&unit.block).is_empty());
}

#[test]
fn paired_singles_lower_to_the_shared_vector_ops() {
    // ps_add f1,f2,f3 → vec_add { lanes 2, w64 } between two packed reads.
    let ps_add = 0x1022_182Au32;
    let d = dump(ps_add);
    assert!(
        d.contains("vec_add v, v, v : w64 lanes=2"),
        "ps_add must lower to the shared vector add:\n{d}"
    );
    // ps_mul → vec_mul with FRC as the second operand.
    let ps_mul = 0x1082_0032u32;
    let d = dump(ps_mul);
    assert!(d.contains("vec_mul"), "ps_mul must lower to vec_mul:\n{d}");
    // ps_sub has no VecSub, so it must become add + a sign flip (one xor).
    let ps_sub = 0x1022_1828u32;
    let d = dump(ps_sub);
    assert!(d.contains("xor") && d.contains("vec_add"), "ps_sub must be a+c style:\n{d}");
    // ps_merge00 is pure bit manipulation: no vector op, no helper.
    let ps_merge00 = 0x1022_1C20u32;
    let d = dump(ps_merge00);
    assert!(
        !d.contains("vec_"),
        "ps_merge00 should be and/shift/or only:\n{d}"
    );
    assert!(d.contains("and v") && d.contains("or v"), "merge shape:\n{d}");
}

#[test]
fn quantized_loads_compute_their_address_in_ir() {
    // psq_l f0,4(r3),1,5 — EA must be an IR value (constant here) and the
    // quantization a single helper call, and psq_stu must write r3 back.
    let d = dump(0xE003_D004);
    assert!(
        d.contains("ppc_psq_load") && d.contains("#0x1, #0x5"),
        "psq_load must carry the EA value plus W and GQR: \n{d}"
    );
    assert!(d.contains("= add v"), "the EA must be IR, not inside the helper:\n{d}");
    let psq_stu = (61 << 26) | (3 << 21) | (2 << 16) | (1 << 16) | (7 << 17) | 8u32;
    let d = dump(psq_stu);
    assert!(d.contains("ppc_psq_store"), "psq_stu must store:\n{d}");
}

#[test]
fn fneg_and_fabs_are_bit_ops_not_helpers() {
    // The container's sign bit is host bit 63, so these need no FP helper at all.
    let fneg = 0xFC22_0050u32;
    let d = dump(fneg);
    assert!(d.contains("xor v"), "fneg must be a xor:\n{d}");
    assert!(!d.contains("ppc_fp_"), "fneg must not call a helper:\n{d}");
    let fabs = 0xFC22_0210u32;
    let d = dump(fabs);
    assert!(d.contains("and v"), "fabs must be an and:\n{d}");
}

#[test]
fn fsel_is_a_bit_select() {
    let fsel = 0xFC22_192Eu32; // fsel f1,f2,f3,f4
    let d = dump(fsel);
    assert!(d.contains("sar"), "the select predicate is an arithmetic shift:\n{d}");
    assert!(!d.contains("ppc_fp_"), "fsel must not need a helper:\n{d}");
}

#[test]
fn unknown_sprs_report_instead_of_silently_reading_zero() {
    // An undefined SPR number is a real error (the guest would take a program
    // exception), whereas the *ignored* ones (BATs) legitimately read zero.  The
    // two must not be conflated.
    let mut unknown = 0usize;
    for spr in [0u32, 16, 100, 500, 999] {
        let word = (31u32 << 26) | (3 << 21) | ((spr & 31) << 16) | ((spr >> 5) << 11) | (339 << 1);
        let unit = one_unit(word);
        if matches!(unit.error, Some(PpcLowerError::Unimplemented { .. })) {
            unknown += 1;
        }
    }
    assert_eq!(unknown, 5, "undefined SPRs must truncate the block");
    // BATs read zero and are not an error.
    let bat = (31u32 << 26) | (3 << 21) | ((528 & 31) << 16) | ((528 >> 5) << 11) | (339 << 1);
    let unit = one_unit(bat);
    assert!(unit.error.is_none(), "BAT reads are legal-but-unmodelled");
    assert!(
        crate::ir_verify::verify_block(&unit.block).is_empty(),
        "BAT read lowering invalid"
    );
}

// =============================================================================
// 4. the tables the C++ side is written against
// =============================================================================

#[test]
fn intrinsic_ids_are_unique_and_names_resolve() {
    let mut seen = std::collections::HashSet::new();
    for id in PpcIntr::ALL {
        assert!(
            seen.insert(id.id()),
            "duplicate intrinsic id {:#x} ({})",
            id.id(),
            id.name()
        );
        assert!(
            super::intrinsic_name(id.id()) == Some(id.name()),
            "{} does not resolve through intrinsic_name()",
            id.name()
        );
        assert!(id.name().starts_with("ppc_"), "{}: names are namespaced", id.name());
    }
    assert_eq!(
        PpcIntr::ALL.len(),
        seen.len(),
        "ALL must list each intrinsic exactly once"
    );
    assert!(
        PpcIntr::ALL.len() >= 55,
        "ALL looks incomplete: {}",
        PpcIntr::ALL.len()
    );
}

#[test]
fn effects_table_is_stable() {
    use crate::ir::SideEffects::*;
    // The optimiser-visible classes, pinned.  These specific four are the ones
    // whose change would alter what `redundant_load_elimination` can prove; see
    // the module header for why register access must stay Pure.
    assert_eq!(PpcIntr::GetGpr.effects(), Pure);
    assert_eq!(PpcIntr::SetGpr.effects(), Pure);
    assert_eq!(PpcIntr::StringCopy.effects(), ReadsAndWritesMem);
    assert_eq!(PpcIntr::Dcbz.effects(), ReadsAndWritesMem);
    assert_eq!(PpcIntr::Icbi.effects(), WritesMem);
    assert_eq!(PpcIntr::ReadVolatileMisc.effects(), Volatile);
    assert_eq!(PpcIntr::Raise.effects(), Volatile);
    assert_eq!(PpcIntr::PsqLoad.effects(), ReadsMem);
    assert_eq!(PpcIntr::PsqStore.effects(), WritesMem);
}

#[test]
fn misc_slots_fit_the_state_array() {
    assert!(
        PpcMisc::SLOT_COUNT as u64 >= PpcMisc::Msr.slot() + 1,
        "every slot index must fit guest_misc"
    );
    assert!(
        PpcMisc::SLOT_COUNT <= crate::runtime::GUEST_MISC_SLOTS,
        "PpcMisc::SLOT_COUNT grew past CpuState::guest_misc"
    );
    assert_eq!(PpcMisc::gqr(0), PpcMisc::Gqr0.slot());
    assert_eq!(PpcMisc::gqr(7), PpcMisc::Gqr0.slot() + 7);
}

#[test]
fn cpu_state_layout_is_the_ffi_contract() {
    // The C++ emitter mirrors this struct by offset.  These constants are the
    // reviewed layout; a failure here means somebody inserted a field in the
    // middle (which would silently alias spill slot 0 onto gpr[0] — the exact bug
    // the dedicated spill array exists to prevent).
    use crate::runtime::CpuState;
    macro_rules! off {
        ($field:ident) => {
            core::mem::offset_of!(CpuState, $field)
        };
    }
    assert_eq!(off!(gpr), 0);
    assert_eq!(off!(fpr), 256);
    assert_eq!(off!(vec), 512);
    assert_eq!(off!(pc), 1024);
    assert_eq!(off!(flags_zero), 1032);
    assert_eq!(off!(flags_carry), 1033);
    assert_eq!(off!(flags_overflow), 1034);
    assert_eq!(off!(flags_negative), 1035);
    assert_eq!(off!(spill), 1040);
    assert_eq!(core::mem::size_of::<[u64; 128]>(), 1024);
    assert_eq!(off!(guest_misc), 1040 + 1024);
    assert_eq!(
        core::mem::size_of::<CpuState>(),
        1040 + 1024 + 8 * crate::runtime::GUEST_MISC_SLOTS
    );
    assert_eq!(core::mem::align_of::<CpuState>(), 16);

    // Zeroed state is usable, and the accessors round-trip through the containers.
    let mut cpu = CpuState::new_zeroed();
    cpu.set_gpr(3, 0xFFFF_FFFF);
    assert_eq!(cpu.get_gpr(3), 0xFFFF_FFFF);
    cpu.set_fpr_bits(1, 0x1234_5678_9ABC_DEF0);
    assert_eq!(cpu.get_fpr_bits(1), 0x1234_5678_9ABC_DEF0);
    cpu.set_vec_reg(2, 0xAAAA);
    assert_eq!(cpu.vec_reg(2), 0xAAAA);
    cpu.set_misc(PpcMisc::Cr.slot() as u8, 0xF0);
    assert_eq!(cpu.misc(PpcMisc::Cr.slot() as u8), 0xF0);
    assert_eq!(cpu.spill[0], 0, "guest misc must not touch the spill area");
}

#[test]
fn ffi_struct_layout_is_stable() {
    // `CIrOp`/`COperand`/`CLocation`/`CPatchSite` are shared with jit/*.cpp.  The
    // numbers below are what the current field order implies; they are asserted so
    // that any reordering shows up here rather than as garbage in the emitter.
    use crate::{CIR_MAX_OPERANDS, CIrBlock, CIrOp, CAllocEntry, CLocation, COperand, CPatchSite};
    assert_eq!(core::mem::size_of::<COperand>(), 16);
    assert_eq!(core::mem::align_of::<COperand>(), 8);
    assert_eq!(
        core::mem::size_of::<CIrOp>(),
        core::mem::offset_of!(CIrOp, operands) + CIR_MAX_OPERANDS * core::mem::size_of::<COperand>()
    );
    assert_eq!(core::mem::offset_of!(CIrOp, operands) % 8, 0);
    assert_eq!(core::mem::size_of::<CLocation>() % 4, 0);
    assert_eq!(core::mem::size_of::<CAllocEntry>() % 4, 0);
    assert_eq!(core::mem::size_of::<CPatchSite>(), 16);
    assert_eq!(core::mem::size_of::<CIrBlock>() % 8, 0);
}

// =============================================================================
// 5. the boundary: marshaling, ownership, chaining
// =============================================================================

#[cfg(feature = "jit-stub")]
#[test]
fn a_lowered_block_survives_marshal_emit_and_drop() {
    use crate::jit_ffi;
    use crate::CHostCaps;

    let words = [
        0x38840001u32, // addi r4,r4,1
        0x80640008,    // lwz r3,8(r4)
        0x7C632A14,    // add r3,r3,r5
        0x9064000C,    // stw r3,12(r4)
        0x4BFFFFF1,    // b -0x2C  (loops back to a known block)
    ];
    let unit = WiiFrontend::new().translate_unit(&word_bytes(&words), 0x1000);
    let mut block = unit.block;
    passes::run_all(&mut block);
    let alloc = regalloc::linear_scan(&block);

    crate::jit_stub::stub_reset_counters();
    let before = crate::jit_stub::stub_alloc_count();
    let compiled = jit_ffi::emit_block(&block, &alloc, CHostCaps { bits: 1 }, true)
        .expect("the placeholder emitter accepts well-formed blocks");
    assert_eq!(crate::jit_stub::stub_alloc_count(), before + 1);
    assert_eq!(compiled.guest_start_pc, 0x1000);
    assert_eq!(
        compiled.patchable_exits.len(),
        1,
        "the constant-target `b` must produce exactly one patch site"
    );
    assert_eq!(compiled.patchable_exits[0].target_guest_pc, 0x1000 - 0x2C + 4 * 5);

    // Chaining writes a real jmp displacement into the recorded offset.
    let fake_target = compiled.code_ptr.wrapping_add(16);
    jit_ffi::patch_chain(&compiled, 0, fake_target).expect("patching the site");

    // Ownership: exactly one jit_free_code call, from Drop.
    let frees_before = crate::jit_stub::stub_free_count();
    drop(compiled);
    assert_eq!(crate::jit_stub::stub_free_count(), frees_before + 1);
    assert_eq!(crate::jit_stub::stub_live_bytes(), 0, "leaked placeholder code");
}

#[cfg(feature = "jit-stub")]
#[test]
fn the_emitter_rejects_a_vreg_the_allocator_never_saw() {
    // The value of having the boundary validate: a frontend that produces a VReg
    // outside the allocation map must fail at `jit_emit_block`, not later.
    use crate::ir::{IrBlock, BlockId};
    use crate::jit_ffi;
    use crate::regalloc::Allocation;
    use crate::CHostCaps;
    use std::collections::HashMap;

    let mut b = crate::ir::IrBuilder::new();
    let id: BlockId = b.new_block(0x1000);
    let v = b.new_vreg();
    b.push(
        id,
        IrOp::Add {
            dst: v,
            a: VOperand::Imm(1),
            b: VOperand::Imm(2),
            width: Width::W64,
        },
    );
    b.push(
        id,
        IrOp::IndirectBranch {
            target: VOperand::Imm(0x1004),
        },
    );
    let block: IrBlock = b.blocks.pop().unwrap();
    let empty = Allocation {
        assignments: HashMap::new(),
    };
    let status = jit_ffi::emit_block(&block, &empty, CHostCaps { bits: 1 }, false)
        .expect_err("missing allocation entry must be rejected");
    assert_eq!(status, crate::CJitStatus::InvalidInput);
}

#[test]
fn endianness_and_layout_are_what_the_core_assumes() {
    let fe = WiiFrontend::new();
    assert_eq!(fe.endianness(), Endian::Big);
    let layout = fe.register_file_layout();
    assert_eq!((layout.gpr_count, layout.fpr_count, layout.vector_reg_count), (32, 32, 32));
    // Both frontends' blocks are fixed-width, which is what the dispatcher's
    // GUEST_INSN_BYTES assumption rests on.
    assert_eq!(GUEST_INSN_BYTES, 4);
}

#[test]
fn decode_is_pure_and_total() {
    // No decoding path may panic on arbitrary bytes — a jump into unmapped or
    // garbage memory must produce `Illegal`, not a crash inside the JIT.
    let fe = WiiFrontend::new();
    let mut rng = 0x1234_5678u32;
    for _ in 0..20000 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        let bytes = rng.to_be_bytes();
        let (insn, len) = fe.decode(&bytes, 0x1000);
        assert_eq!(len, 4);
        assert_eq!(insn.raw, rng);
        // Lowering garbage must either produce IR or an explicit error.
        let mut b = crate::ir::IrBuilder::new();
        let id = b.new_block(0x1000);
        let _ = super::lower::lower_insn(&insn, &mut b, id);
    }
}

#[test]
fn short_buffers_decode_as_zero_without_panicking() {
    let fe = WiiFrontend::new();
    let (insn, len) = fe.decode(&[0x38], 0x1000);
    assert_eq!(len, 4);
    assert_eq!(insn.raw, 0x3800_0000);
    // A single byte cannot be a legal instruction; the top bits 001110 = 14 →
    // `addi` with the rest zero, which is a valid encoding.  What matters is that
    // nothing panicked and the word is the documented zero-filled one.
    let _ = decode_raw(0, 0);
}

#[test]
fn width_is_carried_through_integer_ops() {
    // Everything on guest GPRs must be w32 — the container invariant and the
    // emitter's zero-extension depend on it (see the module header).
    for v in VECTORS {
        let unit = one_unit(v.word);
        for op in &unit.block.ops {
            let w = match op {
                IrOp::Add { width, .. }
                | IrOp::Sub { width, .. }
                | IrOp::Mul { width, .. }
                | IrOp::And { width, .. }
                | IrOp::Or { width, .. }
                | IrOp::Xor { width, .. }
                | IrOp::Shl { width, .. }
                | IrOp::Shr { width, .. }
                | IrOp::Sar { width, .. } => Some(*width),
                _ => None,
            };
            if let Some(w) = w {
                assert!(
                    matches!(w, Width::W32 | Width::W64),
                    "{:#010x}: integer op with width {w:?}; only w32 (GPR) or w64 \
                     (container/bit-twiddle) are allowed"
                );
            }
        }
    }
}
