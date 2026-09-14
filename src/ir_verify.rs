//! Structural well-formedness checks for an `ir::IrBlock`.
//!
//! Why this lives next to the IR instead of inside a test module: the frontends
//! are the code that *constructs* IR, and the failure modes of a hand-written
//! lowering are all structural (use-before-def, a terminator in the middle of a
//! block, an intrinsic with more operands than the fixed-size C payload can
//! hold, an immediate too wide for its op).  Each of those turns into *wrong
//! machine code* rather than a crash once it crosses the FFI boundary, so they
//! are worth catching on the Rust side while the IR is still legible.
//!
//! These checks are deliberately cheap and shape-only: no dataflow, no
//! dominance, no liveness.  `passes` and `regalloc` remain the authority on
//! optimisation and placement; this module only rejects IR they cannot
//! represent faithfully.
//!
//! Called from the frontend test suites (and available to a debug assertion in
//! the dispatcher); never called on the execute path.

use crate::ir::{IrBlock, IrOp, VOperand, VReg, Width};
use crate::CIR_MAX_OPERANDS;
use std::collections::HashSet;

/// Width helpers, re-derived here rather than borrowed from `passes`:
/// `passes`' copies are private and its invariants are its own business, and a
/// verifier that shares the optimiser's internals would silently follow it
/// anywhere.
#[inline]
fn width_bits(width: Width) -> Option<u32> {
    match width {
        Width::W8 => Some(8),
        Width::W16 => Some(16),
        Width::W32 => Some(32),
        Width::W64 => Some(64),
        Width::W128 => None,
    }
}

/// An immediate is representable for `width` if it is already zero-extended OR
/// already sign-extended into that width.  Anything else means the frontend
/// produced a value whose high bits the emitter would silently truncate — e.g.
/// forgetting to mask a PPC 32-bit EA computation to `w32`.
#[inline]
fn imm_fits_width(value: i64, width: Width) -> bool {
    let bits = match width_bits(width) {
        Some(64) | None => return true,
        Some(b) => b,
    };
    let mask = (1u64 << bits) - 1;
    let unsigned = (value as u64) & mask;
    let sign_bit = 1u64 << (bits - 1);
    let sign_extended = if unsigned & sign_bit != 0 {
        (unsigned | (!0u64 << bits)) as i64
    } else {
        unsigned as i64
    };
    (value as u64) == unsigned || value == sign_extended
}

fn each_operand(op: &IrOp, f: &mut dyn FnMut(&VOperand)) {
    match op {
        IrOp::Add { a, b, .. }
        | IrOp::Sub { a, b, .. }
        | IrOp::Mul { a, b, .. }
        | IrOp::And { a, b, .. }
        | IrOp::Or { a, b, .. }
        | IrOp::Xor { a, b, .. }
        | IrOp::VecAdd { a, b, .. }
        | IrOp::VecMul { a, b, .. }
        | IrOp::SetFlags { a, b, .. } => {
            f(a);
            f(b);
        }
        IrOp::Shl { a, amount, .. }
        | IrOp::Shr { a, amount, .. }
        | IrOp::Sar { a, amount, .. } => {
            f(a);
            f(amount);
        }
        IrOp::VecFma { a, b, c, .. } => {
            f(a);
            f(b);
            f(c);
        }
        IrOp::Load { addr, .. } => f(addr),
        IrOp::Store { addr, val, .. } => {
            f(addr);
            f(val);
        }
        IrOp::Branch { cond, .. } => {
            if let Some(cond) = cond {
                f(cond);
            }
        }
        IrOp::IndirectBranch { target } | IrOp::Call { target } => f(target),
        IrOp::Intrinsic { operands, .. } => {
            for operand in operands {
                f(operand);
            }
        }
        IrOp::ReadFlag { .. } | IrOp::Return => {}
    }
}

fn def_of(op: &IrOp) -> Option<VReg> {
    match op {
        IrOp::Add { dst, .. }
        | IrOp::Sub { dst, .. }
        | IrOp::Mul { dst, .. }
        | IrOp::And { dst, .. }
        | IrOp::Or { dst, .. }
        | IrOp::Xor { dst, .. }
        | IrOp::Shl { dst, .. }
        | IrOp::Shr { dst, .. }
        | IrOp::Sar { dst, .. }
        | IrOp::VecAdd { dst, .. }
        | IrOp::VecMul { dst, .. }
        | IrOp::VecFma { dst, .. }
        | IrOp::Load { dst, .. }
        | IrOp::ReadFlag { dst, .. } => Some(*dst),
        IrOp::Intrinsic { dst, .. } => *dst,
        _ => None,
    }
}

/// The immediates whose value *must* be representable in the op's width.
///
/// Addresses, branch targets and branch conditions are host-wide 64-bit values
/// (a guest EA lives in a 64-bit container, zero-extended for 32-bit guests),
/// so they are deliberately absent here — the frontend, not the emitter, decides
/// how many bits of them are meaningful.  A value operand of an ALU op or of a
/// store, however, is truncated to its width by the emitter, so an immediate
/// that does not fit must be rejected instead of silently folded.
fn width_checked_operands(op: &IrOp) -> Vec<(VOperand, Width)> {
    match op {
        IrOp::Add { a, b, width }
        | IrOp::Sub { a, b, width }
        | IrOp::Mul { a, b, width }
        | IrOp::And { a, b, width }
        | IrOp::Or { a, b, width }
        | IrOp::Xor { a, b, width }
        | IrOp::VecAdd { a, b, width, .. }
        | IrOp::VecMul { a, b, width, .. }
        | IrOp::SetFlags { a, b, width, .. } => vec![(*a, *width), (*b, *width)],

        IrOp::Shl { a, amount, width }
        | IrOp::Shr { a, amount, width }
        | IrOp::Sar { a, amount, width } => vec![(*a, *width), (*amount, *width)],

        IrOp::Store { val, width, .. } => vec![(*val, *width)],

        _ => Vec::new(),
    }
}

/// Returns one message per violation; empty means the block is structurally
/// sound.
pub fn verify_block(block: &IrBlock) -> Vec<String> {
    let mut out = Vec::new();
    let mut defined: HashSet<VReg> = HashSet::new();

    if block.ops.is_empty() {
        out.push("block has no ops (an empty block cannot be entered)".to_string());
        return out;
    }

    for (index, op) in block.ops.iter().enumerate() {
        // ---- uses must already be defined (SSA, no forward references) ------
        each_operand(op, &mut |operand| {
            if let VOperand::Reg(v) = *operand {
                if !defined.contains(&v) {
                    out.push(format!(
                        "op {index}: uses v{} before its definition",
                        v.0
                    ));
                }
            }
        });

        // ---- definition uniqueness -----------------------------------------
        if let Some(dst) = def_of(op) {
            if !defined.insert(dst) {
                out.push(format!(
                    "op {index}: v{} is defined more than once (IR is SSA-like)",
                    dst.0
                ));
            }
        }

        // ---- terminators ----------------------------------------------------
        if op.is_terminator() && index + 1 != block.ops.len() {
            out.push(format!(
                "op {index}: terminator in the middle of the block (ops after it are unreachable)"
            ));
        }

        match op {
            IrOp::VecAdd { lanes, width, .. } | IrOp::VecMul { lanes, width, .. } => {
                if let Some(e) = vector_element_error(*lanes, *width) {
                    out.push(format!("op {index}: {e}"));
                }
            }
            IrOp::VecFma { lanes, .. } => {
                if !matches!(lanes, 2 | 4) {
                    out.push(format!(
                        "op {index}: vec_fma lanes={lanes} (only 2 (paired singles) or 4 (NEON) are representable)"
                    ));
                }
            }
            IrOp::ReadFlag { .. } => {
                // A ReadFlag with no preceding SetFlags in this block would make
                // the emitter read whatever the host flags happened to hold.
                let producer = block.ops[..index]
                    .iter()
                    .any(|earlier| matches!(earlier, IrOp::SetFlags { .. }));
                if !producer {
                    out.push(format!(
                        "op {index}: readflag with no preceding setflags in this block"
                    ));
                }
            }
            IrOp::Intrinsic { id, operands, .. } => {
                if operands.len() > CIR_MAX_OPERANDS {
                    // Same rule `jit_ffi::ir_op_to_c` enforces when it returns
                    // Err — catching it here keeps the message next to the
                    // lowering bug instead of at the boundary.
                    out.push(format!(
                        "op {index}: intrinsic#{} carries {} operands, the C payload holds {CIR_MAX_OPERANDS}",
                        id.0,
                        operands.len()
                    ));
                }
            }
            _ => {}
        }

        // ---- immediates must fit the op's width ----------------------------
        for (operand, width) in width_checked_operands(op) {
            if let VOperand::Imm(value) = operand {
                if !imm_fits_width(value, width) {
                    out.push(format!(
                        "op {index}: immediate {value:#x} does not fit {width:?} \
                         (the emitter would truncate it silently)"
                    ));
                }
            }
        }

    }

    if !block
        .ops
        .last()
        .map(|last| last.is_terminator())
        .unwrap_or(false)
    {
        out.push("block does not end in a terminator".to_string());
    }

    out
}

#[inline]
fn vector_element_error(lanes: u8, width: Width) -> Option<&'static str> {
    let bits = match width_bits(width) {
        Some(b) => b,
        None => {
            // W128 has no constant-propagation support, but it is a valid
            // vector width; 128 / lanes must still be a real element size.
            if matches!(lanes, 2 | 4 | 8) {
                return None;
            }
            return Some("vector lanes must be 2, 4 or 8 for a 128-bit op");
        }
    };
    if !matches!(lanes, 2 | 4 | 8) {
        return Some("vector lanes must be 2, 4 or 8");
    }
    if bits % lanes as u32 != 0 {
        return Some("vector width is not divisible by the lane count");
    }
    let elem = bits / lanes as u32;
    if !matches!(elem, 16 | 32 | 64) {
        return Some("vector element size must be 16, 32 or 64 bits");
    }
    None
}

/// Test/debug helper: panic with the full violation list.
pub fn assert_valid(block: &IrBlock) {
    let violations = verify_block(block);
    assert!(
        violations.is_empty(),
        "IR block {:#x} is not well formed:\n{}\n---- dumped ----\n{}",
        block.guest_start_pc,
        violations.join("\n"),
        crate::ir_dump::dump_block(block)
    );
}
