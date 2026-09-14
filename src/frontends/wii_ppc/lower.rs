//! Broadway → universal IR lowering.
//!
//! One guest instruction in, a short run of IR ops out.  Nothing here knows
//! which registers the host has, what the code cache looks like, or what the
//! emitter does — it only ever constructs [`IrOp`]s.  The conventions that make
//! that possible (and that the C++ side depends on) are in the module header of
//! [`super`]; the intrinsic operand shapes are in [`super::intrinsics`].
//!
//! ## Shape of the output, for one guest instruction
//!
//! ```text
//!   v1 = ppc_get_gpr #4                 (forwarded: one read per guest reg per block)
//!   v2 = ppc_get_gpr #5
//!   v3 = add v1, v2 : w32
//!   ppc_set_gpr #3, v3
//!   [ record form: v4 = cr0 bits from v3 ; cr = merge(cr, v4) ]
//!   [ update form: ppc_set_gpr #4, ea ]
//! ```
//!
//! Two properties worth defending, because they are what keeps the emitted code
//! from being a load/store around every operand:
//!
//!   • `Ctx::fwd` maps a guest register to the VReg that currently holds its
//!     value *within this block*.  It is updated exactly where a write happens, so
//!     a read cannot return a stale VReg; and since IR VRegs are
//!     single-assignment, reusing one is a rename, not an optimisation that could
//!     go wrong.
//!   • every guest-visible state write is an explicit `SET_*` intrinsic, so a
//!     block always leaves `CpuState` complete.  There is no "flush at block end"
//!     pass with its own liveness model to get wrong — the allocator's spill slots
//!     and the chaining path both rely on that.
//!
//! ## PPC32 value convention
//!
//! GPRs are 32-bit, stored zero-extended in the 64-bit `CpuState::gpr` slots, and
//! every integer IR op on them uses `Width::W32` — so the emitter's 32-bit
//! destinations zero the upper half for free on x86-64 and the container invariant
//! holds without an `And` after every operation.  Effective addresses are computed
//! at W32 too, which *is* PPC's `mod 2^32` EA arithmetic.  Do not widen those to
//! W64 "to save a mask": the high bits are load-bearing for `cmpl`, for the rotate
//! class, and for the redundant-load pass's address matching.
//!
//! Consequently **every immediate fed to a `w32` op is written as a `u32`
//! pattern** (`Self::um`), never as a sign-extended `i64` mask: `!0xF000_0000i64`
//! has high bits set and no 32-bit op can carry it, and
//! `ir_verify::verify_block` rejects exactly that class of bug ("immediate would
//! be silently truncated by the emitter").  The corpus test in `tests.rs` runs
//! that check over every supported instruction with every interesting register
//! and immediate field, so a new arm cannot acquire a bad mask quietly.
//!
//! Where a 32-bit value must be *summed* without losing a carry (the `addc`/`subfe`
//! family, the division/compare overflow rules), the arithmetic runs at W64 on
//! zero-extended operands and is masked back down afterwards.  That is deliberate,
//! not sloppiness: at W64 the carry out of bit 32 is an ordinary bit, so the
//! guest's CA needs no host flags and cannot be mis-derived from a host carry that
//! means something else.

use super::decode::PpcKind;
use super::fields::PpcFields;
use super::intrinsics::{PpcIntr, PpcMisc, SprAccess};
use super::DecodedPpc;
use crate::ir::{
    BlockId, Endian, FlagKind, FlagOp, IntrinsicId, IrBuilder, IrOp, VOperand, VReg, Width,
};
use std::collections::HashMap;

/// XER bit positions in the host 32-bit container (PPC bits 32, 33, 34).
const XER_SO: u32 = 31;
const XER_OV: u32 = 30;
const XER_CA: u32 = 29;

/// All 32 bits, as a `u32` pattern.
const LO32: u32 = 0xFFFF_FFFF;

/// Guest register classes the forwarding table tracks.  A write to one invalidates
/// exactly the entries listed in [`Ctx::kill`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FwdKey {
    Gpr(u8),
    Fpr(u8),
    /// Packed paired single `ps[n]` — overlaps FPR 2n and 2n+1 (see
    /// `PS_PAIR_SPLIT_OVER_TWO_FPRS`), so a write to either FPR kills it.
    Ps(u8),
    /// One VReg threads the whole CR through a block: every CR write produces a
    /// new value, every CR read consumes the current one.
    Cr,
    Xer,
    Fpscr,
    Misc(u8),
}

/// Why a block could not be translated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PpcLowerError {
    /// The encoding is not an instruction on Broadway.  This is *not* an error for
    /// the emulator — it lowers to a guest program exception — and the variant
    /// exists so `translate_unit` can report why a block was cut.
    Illegal { pc: u64, raw: u32 },
    /// A real instruction this build does not lower.  The caller must not execute
    /// the truncated block as if it were complete: a fallback interpreter runs
    /// this instruction and resumes after it.  Silently skipping would let the
    /// guest continue with corrupt state, which is the one failure mode a JIT must
    /// never have.
    Unimplemented {
        pc: u64,
        raw: u32,
        mnemonic: &'static str,
    },
}

impl core::fmt::Display for PpcLowerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PpcLowerError::Illegal { pc, raw } => {
                write!(f, "pc {pc:#010x}: undefined PowerPC instruction {raw:#010x}")
            }
            PpcLowerError::Unimplemented { pc, raw, mnemonic } => write!(
                f,
                "pc {pc:#010x}: unimplemented instruction `{mnemonic}` ({raw:#010x})"
            ),
        }
    }
}

impl std::error::Error for PpcLowerError {}

/// Lower one decoded instruction into `ir`'s block `block`.
pub fn lower_insn(
    insn: &DecodedPpc,
    ir: &mut IrBuilder,
    block: BlockId,
) -> Result<(), PpcLowerError> {
    let mut ctx = Ctx {
        ir,
        block,
        endian: Endian::Big,
        pc: insn.pc,
        next_pc: insn.pc.wrapping_add(4),
        fwd: HashMap::new(),
        zero: None,
    };
    ctx.lower(insn)
}

struct Ctx<'a> {
    ir: &'a mut IrBuilder,
    block: BlockId,
    endian: Endian,
    pc: u64,
    next_pc: u64,
    fwd: HashMap<FwdKey, VReg>,
    zero: Option<VReg>,
}

// =============================================================================
// IR construction primitives
// =============================================================================
impl Ctx<'_> {
    #[inline]
    fn emit(&mut self, op: IrOp) {
        let block = self.block;
        self.ir.push(block, op);
    }

    #[inline]
    fn def(&mut self) -> VReg {
        self.ir.new_vreg()
    }

    #[inline]
    fn reg(v: VReg) -> VOperand {
        VOperand::Reg(v)
    }

    /// Signed/unsized immediate: for values that legitimately occupy the full
    /// 64-bit container (addresses, packed pairs, FP bit patterns at `w64`).
    #[inline]
    fn imm(v: i64) -> VOperand {
        VOperand::Imm(v)
    }

    /// A `u32` pattern, i.e. an immediate for a `w32` op.  See the module header:
    /// masks must never be built by negating an `i64`.
    #[inline]
    fn um(v: u32) -> VOperand {
        VOperand::Imm(v as i64)
    }

    /// A VReg holding zero.  The IR has no `Const` node, so `Add{#0,#0}` is the
    /// materialisation form (and the constant folder sees straight through it).
    /// Cached per block, because every zeroing path and every `x | 0` wants it.
    fn zero(&mut self) -> VReg {
        if let Some(v) = self.zero {
            return v;
        }
        let d = self.def();
        self.emit(IrOp::Add {
            dst: d,
            a: Self::imm(0),
            b: Self::imm(0),
            width: Width::W64,
        });
        self.zero = Some(d);
        d
    }

    /// Materialise a constant into a fresh VReg.  `Add{value, zero}` — so a block
    /// full of `li rD,k` costs exactly one ALU op each, and folding later sees the
    /// immediate.
    fn immv(&mut self, value: i64, width: Width) -> VReg {
        let value = match width {
            Width::W32 => (value as u32) as i64,
            _ => value,
        };
        if value == 0 {
            return self.zero();
        }
        let d = self.def();
        let z = self.zero();
        self.emit(IrOp::Add {
            dst: d,
            a: Self::imm(value),
            b: Self::reg(z),
            width,
        });
        d
    }

    #[inline]
    fn mask_to_width(&self, value: i64, width: Width) -> i64 {
        match width {
            Width::W8 => (value as u8) as i64,
            Width::W16 => (value as u16) as i64,
            Width::W32 => (value as u32) as i64,
            Width::W64 | Width::W128 => value,
        }
    }

    fn binop(&mut self, a: VOperand, b: VOperand, width: Width, kind: BinKind) -> VReg {
        let d = self.def();
        self.emit(match kind {
            BinKind::Add => IrOp::Add { dst: d, a, b, width },
            BinKind::Sub => IrOp::Sub { dst: d, a, b, width },
            BinKind::Mul => IrOp::Mul { dst: d, a, b, width },
            BinKind::And => IrOp::And { dst: d, a, b, width },
            BinKind::Or => IrOp::Or { dst: d, a, b, width },
            BinKind::Xor => IrOp::Xor { dst: d, a, b, width },
        });
        d
    }

    #[inline]
    fn add(&mut self, a: VOperand, b: VOperand, width: Width) -> VReg {
        self.binop(a, b, width, BinKind::Add)
    }
    #[inline]
    fn sub(&mut self, a: VOperand, b: VOperand, width: Width) -> VReg {
        self.binop(a, b, width, BinKind::Sub)
    }
    #[inline]
    fn and(&mut self, a: VOperand, b: VOperand, width: Width) -> VReg {
        self.binop(a, b, width, BinKind::And)
    }
    #[inline]
    fn or(&mut self, a: VOperand, b: VOperand, width: Width) -> VReg {
        self.binop(a, b, width, BinKind::Or)
    }
    #[inline]
    fn xor(&mut self, a: VOperand, b: VOperand, width: Width) -> VReg {
        self.binop(a, b, width, BinKind::Xor)
    }
    #[inline]
    fn mul(&mut self, a: VOperand, b: VOperand, width: Width) -> VReg {
        self.binop(a, b, width, BinKind::Mul)
    }

    fn shift(&mut self, a: VOperand, amount: VOperand, width: Width, kind: ShiftKind) -> VReg {
        let d = self.def();
        self.emit(match kind {
            ShiftKind::Left => IrOp::Shl { dst: d, a, amount, width },
            ShiftKind::Right => IrOp::Shr { dst: d, a, amount, width },
            ShiftKind::Arith => IrOp::Sar { dst: d, a, amount, width },
        });
        d
    }

    /// Rotate-left of a 32-bit value by a *constant*.  PPC rotates, x86 shifts, and
    /// the IR has no rotate: `rotl32(v,n) = (v<<n) | (v>>(32-n))`.  `n = 0` is
    /// folded to the identity rather than emitting `(v<<0)|(v>>32)` — the latter
    /// would be `v|v` under the host's mod-32 shift counts, which happens to be
    /// right but must not be relied on.
    fn rotl32_imm(&mut self, v: VReg, n: u32) -> VReg {
        let n = n & 31;
        if n == 0 {
            return v;
        }
        let left = self.shift(Self::reg(v), Self::imm(n as i64), Width::W32, ShiftKind::Left);
        let right = self.shift(Self::reg(v), Self::imm((32 - n) as i64), Width::W32, ShiftKind::Right);
        self.or(Self::reg(left), Self::reg(right), Width::W32)
    }

    /// Rotate-left by a *register* amount.  Safe for any count in 0..=31; callers
    /// with a possibly-larger count clamp it first (see [`Ctx::lower_shift_reg`]).
    fn rotl32_reg(&mut self, v: VReg, n: VReg) -> VReg {
        let left = self.shift(Self::reg(v), Self::reg(n), Width::W32, ShiftKind::Left);
        let c32 = self.immv(32, Width::W32);
        let complement = self.sub(Self::reg(c32), Self::reg(n), Width::W32);
        let right = self.shift(Self::reg(v), Self::reg(complement), Width::W32, ShiftKind::Right);
        self.or(Self::reg(left), Self::reg(right), Width::W32)
    }

    /// `0 - cond` → all-ones when `cond` is nonzero, zero otherwise.  The IR's only
    /// "boolean to mask" primitive, used by the branch select below.
    fn mask_of(&mut self, cond: VReg, width: Width) -> VReg {
        let z = self.zero();
        self.sub(Self::reg(z), Self::reg(cond), width)
    }

    /// `a` where `mask` is all-ones, `b` where it is zero.
    fn select(&mut self, mask: VReg, a: VReg, b: VReg, width: Width) -> VReg {
        let diff = self.xor(Self::reg(a), Self::reg(b), width);
        let kept = self.and(Self::reg(mask), Self::reg(diff), width);
        self.xor(Self::reg(b), Self::reg(kept), width)
    }

    /// Emit one intrinsic; the only place `IrOp::Intrinsic` is built here, so the
    /// table in `intrinsics.rs` is enforced rather than assumed.
    fn intr_raw(&mut self, id: PpcIntr, operands: Vec<VOperand>, dst: Option<VReg>) {
        debug_assert_eq!(
            operands.len(),
            id.arity(),
            "{} expects {} operand(s), got {}",
            id.name(),
            id.arity(),
            operands.len()
        );
        debug_assert_eq!(
            dst.is_some(),
            id.has_dst(),
            "{} has_dst() = {} but was emitted with dst = {:?}",
            id.name(),
            id.has_dst(),
            dst
        );
        debug_assert!(
            operands.len() <= crate::CIR_MAX_OPERANDS,
            "{} carries more operands than the fixed-size C payload",
            id.name()
        );
        self.emit(IrOp::Intrinsic {
            id: IntrinsicId(id.id()),
            effects: id.effects(),
            operands,
            dst,
        });
    }

    #[inline]
    fn intr(&mut self, id: PpcIntr, operands: Vec<VOperand>) -> VReg {
        let d = self.def();
        self.intr_raw(id, operands, Some(d));
        d
    }

    // ---- guest register access, with per-block forwarding -------------------

    /// Drop a forwarding entry plus anything that aliases it.  FPR writes kill the
    /// packed paired-single views that cover them and vice versa: forgetting that
    /// is how an `fadds` following a `psq_l` would read a stale pair.
    fn kill(&mut self, key: FwdKey) {
        self.fwd.remove(&key);
        match key {
            FwdKey::Fpr(n) => {
                self.fwd.remove(&FwdKey::Ps(n / 2));
            }
            FwdKey::Ps(n) => {
                if 2 * n < 32 {
                    self.fwd.remove(&FwdKey::Fpr(2 * n));
                }
                if 2 * n + 1 < 32 {
                    self.fwd.remove(&FwdKey::Fpr(2 * n + 1));
                }
            }
            _ => {}
        }
    }

    fn get(&mut self, key: FwdKey, id: PpcIntr, index: Option<u32>) -> VReg {
        if let Some(v) = self.fwd.get(&key).copied() {
            return v;
        }
        let operands = match index {
            Some(i) => vec![Self::imm(i as i64)],
            None => Vec::new(),
        };
        let v = self.intr(id, operands);
        self.fwd.insert(key, v);
        v
    }

    #[inline]
    fn get_gpr(&mut self, n: u8) -> VReg {
        self.get(FwdKey::Gpr(n), PpcIntr::GetGpr, Some(n as u32))
    }

    #[inline]
    fn get_fpr(&mut self, n: u8) -> VReg {
        self.get(FwdKey::Fpr(n), PpcIntr::GetFpr, Some(n as u32))
    }

    #[inline]
    fn get_cr(&mut self) -> VReg {
        self.get(FwdKey::Cr, PpcIntr::GetMisc, Some(PpcMisc::Cr.slot() as u32))
    }

    #[inline]
    fn get_xer(&mut self) -> VReg {
        self.get(FwdKey::Xer, PpcIntr::GetMisc, Some(PpcMisc::Xer.slot() as u32))
    }

    #[inline]
    fn get_fpscr(&mut self) -> VReg {
        self.get(FwdKey::Fpscr, PpcIntr::GetMisc, Some(PpcMisc::Fpscr.slot() as u32))
    }

    fn get_misc(&mut self, slot: PpcMisc) -> VReg {
        self.get(FwdKey::Misc(slot as u8), PpcIntr::GetMisc, Some(slot.slot() as u32))
    }

    /// Volatile misc read (time base, decrementer): never forwarded, because two
    /// reads in one block are allowed to differ — that is the whole point of `mftb`
    /// delay loops.
    fn get_misc_volatile(&mut self, slot: PpcMisc) -> VReg {
        self.intr(PpcIntr::ReadVolatileMisc, vec![Self::imm(slot.slot() as i64)])
    }

    /// Slot-numbered misc access, for the SPR path where the slot is a decode-time
    /// number rather than a named register.
    fn get_misc_slot(&mut self, slot: u64) -> VReg {
        let key = FwdKey::Misc((slot % 256) as u8);
        if let Some(v) = self.fwd.get(&key).copied() {
            return v;
        }
        let v = self.intr(PpcIntr::GetMisc, vec![Self::imm(slot as i64)]);
        self.fwd.insert(key, v);
        v
    }

    fn set_misc_slot(&mut self, slot: u64, value: VReg) {
        self.intr_raw(
            PpcIntr::SetMisc,
            vec![Self::imm(slot as i64), Self::reg(value)],
            None,
        );
        self.fwd.insert(FwdKey::Misc((slot % 256) as u8), value);
    }

    fn set_misc(&mut self, slot: PpcMisc, value: VReg) {
        self.intr_raw(
            PpcIntr::SetMisc,
            vec![Self::imm(slot.slot() as i64), Self::reg(value)],
            None,
        );
        match slot {
            PpcMisc::Cr => {
                self.fwd.insert(FwdKey::Cr, value);
            }
            PpcMisc::Xer => {
                self.fwd.insert(FwdKey::Xer, value);
            }
            // FPSCR is written by FP helpers too, so reads after this point must
            // not reuse a cached value.
            PpcMisc::Fpscr => {
                self.kill(FwdKey::Fpscr);
            }
            other => {
                self.fwd.insert(FwdKey::Misc(other as u8), value);
            }
        }
    }

    /// Store a value that is already a valid guest 32-bit GPR value (the result of
    /// a W32 op, or explicitly masked).
    #[inline]
    fn set_gpr(&mut self, n: u8, value: VReg) {
        self.intr_raw(
            PpcIntr::SetGpr,
            vec![Self::imm(n as i64), Self::reg(value)],
            None,
        );
        self.fwd.insert(FwdKey::Gpr(n), value);
    }

    /// Store after forcing a W64 intermediate into the 32-bit guest form.
    fn set_gpr_masked(&mut self, n: u8, value: VReg) {
        let masked = self.and(Self::reg(value), Self::um(LO32), Width::W64);
        self.set_gpr(n, masked);
    }

    fn set_fpr(&mut self, n: u8, value: VReg) {
        self.intr_raw(
            PpcIntr::SetFpr,
            vec![Self::imm(n as i64), Self::reg(value)],
            None,
        );
        self.fwd.insert(FwdKey::Fpr(n), value);
    }

    /// Paired single `ps[n]` packed into one 64-bit value (ps0 in bits 31:0,
    /// ps1 in 63:32).
    fn get_ps(&mut self, n: u8) -> VReg {
        self.get(FwdKey::Ps(n), PpcIntr::GetPsPair, Some(n as u32))
    }

    fn set_ps(&mut self, n: u8, value: VReg) {
        self.intr_raw(
            PpcIntr::SetPsPair,
            vec![Self::imm(n as i64), Self::reg(value)],
            None,
        );
        self.fwd.insert(FwdKey::Ps(n), value);
    }

    // ---- CR / XER / FPSCR plumbing -----------------------------------------

    fn set_cr(&mut self, value: VReg) {
        self.set_misc(PpcMisc::Cr, value);
    }

    /// CR field `j` as a 0..=15 value.
    fn crf(&mut self, j: u32) -> VReg {
        let cr = self.get_cr();
        let shifted = self.shift(Self::reg(cr), Self::imm((28 - 4 * j) as i64), Width::W32, ShiftKind::Right);
        self.and(Self::reg(shifted), Self::um(0xF), Width::W32)
    }

    /// CR bit `bi` (PPC numbering: 0 = CR0.LT, 2 = CR0.EQ) as 0/1.
    fn cr_bit(&mut self, bi: u32) -> VReg {
        let cr = self.get_cr();
        let shifted = self.shift(Self::reg(cr), Self::imm((31 - bi) as i64), Width::W32, ShiftKind::Right);
        self.and(Self::reg(shifted), Self::um(1), Width::W32)
    }

    /// Replace CR field `j` with `bits`, leaving the other fields alone.
    fn set_crf(&mut self, j: u32, bits: VReg) {
        let shift = 28 - 4 * j;
        let cr = self.get_cr();
        let clear = !(0xFu32 << shift);
        let cleared = self.and(Self::reg(cr), Self::um(clear), Width::W32);
        let moved = self.shift(Self::reg(bits), Self::imm(shift as i64), Width::W32, ShiftKind::Left);
        let merged = self.or(Self::reg(cleared), Self::reg(moved), Width::W32);
        self.set_cr(merged);
    }

    /// Set one CR bit, leaving the rest of the register alone.
    fn set_cr_bit(&mut self, bi: u32, bit: VReg) {
        let pos = 31 - bi;
        let cr = self.get_cr();
        let cleared = self.and(Self::reg(cr), Self::um(!(1u32 << pos)), Width::W32);
        let moved = self.shift(Self::reg(bit), Self::imm(pos as i64), Width::W32, ShiftKind::Left);
        let merged = self.or(Self::reg(cleared), Self::reg(moved), Width::W32);
        self.set_cr(merged);
    }

    /// One XER bit as 0/1.
    fn xer_bit(&mut self, pos: u32) -> VReg {
        let xer = self.get_xer();
        let shifted = self.shift(Self::reg(xer), Self::imm(pos as i64), Width::W32, ShiftKind::Right);
        self.and(Self::reg(shifted), Self::um(1), Width::W32)
    }

    /// Write several XER bits in one read-modify-write.  Batching matters: the
    /// common case sets CA and then needs the *new* XER.SO for CR0, and per-bit
    /// read-modify-writes would serialise that into three.
    fn set_xer_bits(&mut self, bits: &[(u32, VReg)]) {
        if bits.is_empty() {
            return;
        }
        let mut mask: u32 = 0;
        for (pos, _) in bits {
            mask |= 1 << pos;
        }
        let xer = self.get_xer();
        let mut cur = self.and(Self::reg(xer), Self::um(!mask), Width::W32);
        for (pos, value) in bits {
            let moved = self.shift(Self::reg(*value), Self::imm(*pos as i64), Width::W32, ShiftKind::Left);
            cur = self.or(Self::reg(cur), Self::reg(moved), Width::W32);
        }
        self.set_misc(PpcMisc::Xer, cur);
    }

    /// `1` when `v != 0`, else `0`.  This is the one thing the IR genuinely needs
    /// host flags for, and precisely why `dead_flag_elimination` exists: when the
    /// result is unused the whole `SetFlags` disappears.
    fn is_nonzero(&mut self, v: VReg, width: Width) -> VReg {
        let eq = self.is_zero_flag(v, width);
        let one = self.immv(1, Width::W32);
        self.sub(Self::reg(one), Self::reg(eq), Width::W32)
    }

    /// `1` when `v == 0`, else `0`, via the lazy flag pair.
    fn is_zero_flag(&mut self, v: VReg, width: Width) -> VReg {
        self.emit(IrOp::SetFlags {
            op: FlagOp::AndOp,
            a: Self::reg(v),
            b: Self::reg(v),
            width,
        });
        let z = self.def();
        self.emit(IrOp::ReadFlag {
            dst: z,
            flag: FlagKind::Zero,
        });
        z
    }

    /// PPC sign bit of a 32-bit value (host bit 31) as 0/1.
    fn sign32(&mut self, v: VReg) -> VReg {
        self.shift(Self::reg(v), Self::imm(31), Width::W32, ShiftKind::Right)
    }

    fn set_fpscr_bit(&mut self, pos: u32, bit: VReg) {
        let fpscr = self.get_fpscr();
        let cleared = self.and(Self::reg(fpscr), Self::um(!(1u32 << pos)), Width::W32);
        let moved = self.shift(Self::reg(bit), Self::imm(pos as i64), Width::W32, ShiftKind::Left);
        let merged = self.or(Self::reg(cleared), Self::reg(moved), Width::W32);
        self.set_misc(PpcMisc::Fpscr, merged);
    }

    // ---- addresses and memory ------------------------------------------------

    /// D-form `EA = (RA|0) + d`, at W32 (PPC's `mod 2^32`).
    ///
    /// `update` selects the rA==0 rule only (in a plain load/store rA=0 means the
    /// constant 0 and there is no base register to read; in an update form rA=0 is
    /// architecturally illegal and this build treats it as "r0 ← ea", harmless and
    /// louder to diagnose than folding the address to a constant).  It deliberately
    /// does NOT write the EA back: PPC updates rA only after the access succeeds, so
    /// every update-form caller emits `set_gpr(ra, ea)` *after* its load/store.  A
    /// `test update_form_write_back_follows_the_access` guard pins that order.
    fn ea_d(&mut self, f: PpcFields, d: i64, update: bool) -> VReg {
        let ra = f.ra();
        if ra == 0 && !update {
            return self.immv(self.mask_to_width(d, Width::W32), Width::W32);
        }
        let base = self.get_gpr(ra);
        self.add(Self::reg(base), Self::um(self.mask_to_width32(d)), Width::W32)
    }

    #[inline]
    fn mask_to_width32(&self, d: i64) -> u32 {
        d as u32
    }

    /// Indexed `EA = (RA|0) + RB`.  The `|0` matters: for the indexed load/store
    /// forms an RA field of 0 means the *value* zero, not GPR 0's contents — that
    /// is what makes `lwzx rD,0,rX` a plain register-indirect load.
    fn ea_x(&mut self, ra: u8, rb: u8) -> VReg {
        let a = if ra == 0 { self.zero() } else { self.get_gpr(ra) };
        let b = self.get_gpr(rb);
        self.add(Self::reg(a), Self::reg(b), Width::W32)
    }

    fn load(&mut self, addr: VOperand, width: Width, sign_ext: bool) -> VReg {
        let d = self.def();
        self.emit(IrOp::Load {
            dst: d,
            addr,
            width,
            sign_ext,
            endian: self.endian,
        });
        d
    }

    /// Reverse-endian access (the `*brx` family): same address, opposite byte
    /// order, and the `endian` field is the only thing that changes — which is
    /// exactly what that field exists for.
    fn load_rev(&mut self, addr: VOperand, width: Width, sign_ext: bool) -> VReg {
        let d = self.def();
        self.emit(IrOp::Load {
            dst: d,
            addr,
            width,
            sign_ext,
            endian: rev(self.endian),
        });
        d
    }

    fn store(&mut self, addr: VOperand, val: VOperand, width: Width) {
        self.emit(IrOp::Store {
            addr,
            val,
            width,
            endian: self.endian,
        });
    }

    fn store_rev(&mut self, addr: VOperand, val: VOperand, width: Width) {
        self.emit(IrOp::Store {
            addr,
            val,
            width,
            endian: rev(self.endian),
        });
    }

    // ---- block exits ---------------------------------------------------------

    /// Exit to a compile-time-known guest PC.  The emitter records a patch site for
    /// this form, which is how direct block chaining happens.
    #[inline]
    fn exit_imm(&mut self, target_pc: u64) {
        self.emit(IrOp::IndirectBranch {
            target: Self::imm(target_pc as i64),
        });
    }

    #[inline]
    fn exit_reg(&mut self, target: VReg) {
        self.emit(IrOp::IndirectBranch {
            target: Self::reg(target),
        });
    }

    /// `exit = taken ? target : next_pc`, as data.  See the module header of
    /// `super` for why a conditional exit is folded into a value instead of using
    /// `IrOp::Branch`.
    fn select_exit(&mut self, taken: VReg, target: VReg) -> VReg {
        let next = self.immv((self.next_pc & 0xFFFF_FFFF) as i64, Width::W64);
        let delta = self.sub(Self::reg(target), Self::reg(next), Width::W64);
        let mask = self.mask_of(taken, Width::W64);
        let kept = self.and(Self::reg(mask), Self::reg(delta), Width::W64);
        self.add(Self::reg(next), Self::reg(kept), Width::W64)
    }

    fn raise(&mut self, vector: u32, cond: VReg) {
        self.intr_raw(
            PpcIntr::Raise,
            vec![
                Self::imm(vector as i64),
                Self::imm(self.pc as i64),
                Self::reg(cond),
            ],
            None,
        );
    }

    /// Undefined opcode: the guest's own program exception.  The exit target is
    /// the *following* instruction, which is what the dispatcher uses if a debugger
    /// suppresses the exception — the alternative (re-executing the faulting word)
    /// would spin.
    fn trap_illegal(&mut self) {
        let one = self.immv(1, Width::W32);
        self.raise(0x700, one);
        self.exit_imm(self.next_pc);
    }

    // ---- PPC's compare result -------------------------------------------------

    /// Write PPC's 4-bit compare field {LT,GT,EQ,SO} into CR field `bf`.
    ///
    /// EQ comes from a zero test on the difference (lazy flags); LT is the
    /// difference's sign bit, corrected by the subtraction's overflow for signed
    /// compares exactly like x86 `jl` (SF XOR OF) or by the borrow for unsigned
    /// ones (CF *is* "a <u b").  GT is then "neither".  The emitter turns
    /// `SetFlags{SubOp}` + the `ReadFlag`s into one host `cmp` plus `setcc`s, so
    /// the guest's 4-bit field costs a handful of 1-byte ops and no spills.
    fn compare_to_crf(&mut self, bf: u32, a: VReg, b: VReg, signed: bool, width: Width, so: SoBit) {
        let bits = self.cmp_field(a, b, signed, width, so);
        self.set_crf(bf, bits);
    }

    /// The 4-bit field value, without writing it anywhere.
    fn cmp_field(&mut self, a: VReg, b: VReg, signed: bool, width: Width, so: SoBit) -> VReg {
        let top = width_bits(width) - 1;
        let diff = self.sub(Self::reg(a), Self::reg(b), width);
        let eq = self.is_zero_flag(diff, width);
        let neg = self.shift(Self::reg(diff), Self::imm(top as i64), width, ShiftKind::Right);

        // The subtraction's own flags, read once for both LT flavours.
        self.emit(IrOp::SetFlags {
            op: FlagOp::SubOp,
            a: Self::reg(a),
            b: Self::reg(b),
            width,
        });
        let lt = if signed {
            let of = self.def();
            self.emit(IrOp::ReadFlag {
                dst: of,
                flag: FlagKind::Overflow,
            });
            self.xor(Self::reg(neg), Self::reg(of), Width::W32)
        } else {
            let c = self.def();
            self.emit(IrOp::ReadFlag {
                dst: c,
                flag: FlagKind::Carry,
            });
            c
        };

        let one = self.immv(1, Width::W32);
        let not_eq = self.sub(Self::reg(one), Self::reg(eq), Width::W32);
        let not_lt = self.sub(Self::reg(one), Self::reg(lt), Width::W32);
        let gt = self.and(Self::reg(not_eq), Self::reg(not_lt), Width::W32);
        let so_bit = match so {
            SoBit::One => one,
            SoBit::SignOf(v, w) => {
                let pos = width_bits(w) - 1;
                self.shift(Self::reg(v), Self::imm(pos as i64), w, ShiftKind::Right)
            }
            SoBit::XerSo => self.xer_bit(XER_SO),
        };
        let l8 = self.shift(Self::reg(lt), Self::imm(3), Width::W32, ShiftKind::Left);
        let g4 = self.shift(Self::reg(gt), Self::imm(2), Width::W32, ShiftKind::Left);
        let e2 = self.shift(Self::reg(eq), Self::imm(1), Width::W32, ShiftKind::Left);
        let t = self.or(Self::reg(l8), Self::reg(g4), Width::W32);
        let t = self.or(Self::reg(t), Self::reg(e2), Width::W32);
        self.or(Self::reg(t), Self::reg(so_bit), Width::W32)
    }

    /// Record form: CR0 = {LT,GT,EQ} of `result` against zero, SO bit from `so`.
    fn record_cr0(&mut self, result: VReg, so: SoBit) {
        let zero = self.zero();
        self.compare_to_crf(0, result, zero, true, Width::W32, so);
    }

    /// Signed overflow of `x + y + carry_in`, computed from the definition PPC and
    /// x86 share: OF = (carry into bit 31) XOR (carry out of bit 31).
    ///
    /// Doing it from the operands instead of from the host's OF flag is what makes
    /// it correct for `subf` (whose `+1` is a *carry-in* here, and must feed both
    /// carries) and for `adde`/`addme` (where CA does too).  Both sums run at W64 on
    /// zero-extended operands, so the two carries are ordinary bits and no host
    /// flag is involved at all.
    fn sum_overflow(&mut self, x: VReg, y: VReg, carry: Option<VReg>) -> VReg {
        let lo31 = Self::um(0x7FFF_FFFF);
        // carry out of bit 31 == bit 32 of the full 33-bit sum.
        let mut full = self.add(Self::reg(x), Self::reg(y), Width::W64);
        // carry into bit 31 == bit 31 of the sum of the low 31 bits.
        let xa = self.and(Self::reg(x), lo31, Width::W32);
        let ya = self.and(Self::reg(y), lo31, Width::W32);
        let mut low = self.add(Self::reg(xa), Self::reg(ya), Width::W64);
        if let Some(c) = carry {
            full = self.add(Self::reg(full), Self::reg(c), Width::W64);
            low = self.add(Self::reg(low), Self::reg(c), Width::W64);
        }
        let cout = self.shift(Self::reg(full), Self::imm(32), Width::W64, ShiftKind::Right);
        let cout = self.and(Self::reg(cout), Self::um(1), Width::W64);
        let cin = self.shift(Self::reg(low), Self::imm(31), Width::W64, ShiftKind::Right);
        let cin = self.and(Self::reg(cin), Self::um(1), Width::W64);
        self.xor(Self::reg(cin), Self::reg(cout), Width::W64)
    }

    /// Bit 32 of a W64 sum: the PPC CA for a 32-bit add of zero-extended operands.
    fn xer_carry_of(&mut self, sum: VReg) -> VReg {
        let c = self.shift(Self::reg(sum), Self::imm(32), Width::W64, ShiftKind::Right);
        self.and(Self::reg(c), Self::um(1), Width::W64)
    }
}

#[inline]
fn rev(e: Endian) -> Endian {
    match e {
        Endian::Big => Endian::Little,
        Endian::Little => Endian::Big,
    }
}

#[inline]
fn width_bits(w: Width) -> u32 {
    match w {
        Width::W8 => 8,
        Width::W16 => 16,
        Width::W32 => 32,
        Width::W64 => 64,
        Width::W128 => 128,
    }
}

#[derive(Clone, Copy)]
enum BinKind {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy)]
enum ShiftKind {
    Left,
    Right,
    Arith,
}

/// What to store into the compare field's SO bit.
#[derive(Clone, Copy)]
enum SoBit {
    /// Logical (unsigned) compares and the logical/shift record forms: SO = 1.
    One,
    /// Signed compares: SO = the sign bit of the first operand.
    SignOf(VReg, Width),
    /// Record form of an arithmetic instruction: SO = XER.SO.
    XerSo,
}

/// Translate-time PPC mask for the rotate class: PPC bits `mb..=me` set, where
/// `mb > me` means the *complement* range (`rlwinm rD,rS,0,20,11` style, which
/// compilers emit for extract-and-clear).
pub(crate) fn rotate_mask(mb: u32, me: u32) -> u32 {
    if mb <= me {
        let n = me - mb + 1;
        if n >= 32 {
            LO32
        } else {
            (((1u64 << n) - 1) as u32) << (31 - me)
        }
    } else {
        // Complement of the gap: host bits (31-me)..31 OR 0..(31-mb).
        let hi = LO32 << (31 - me);
        let lo = if mb == 0 { 0 } else { LO32 >> mb };
        hi | lo
    }
}

// =============================================================================
// the dispatch: one arm per instruction, grouped by lowering shape
// =============================================================================
impl Ctx<'_> {
    fn lower(&mut self, insn: &DecodedPpc) -> Result<(), PpcLowerError> {
        use PpcKind::*;
        let f = insn.f;

        match insn.kind {
            // ---- undefined / not-lowered ----------------------------------
            Illegal => self.trap_illegal(),
            Unsupported => return Err(self.unimplemented(insn)),

            // ---- ordering, hints, nothing to change -----------------------
            Isync => {
                self.intr_raw(PpcIntr::Isync, Vec::new(), None);
            }
            Sync => {
                self.intr_raw(PpcIntr::Sync, Vec::new(), None);
            }
            Eieio => {
                self.intr_raw(PpcIntr::Eieio, Vec::new(), None);
            }
            Tlbsync => {
                self.intr_raw(PpcIntr::SprSideEffect, vec![Self::imm(307), Self::imm(0)], None);
            }

            // ---- integer add / subtract -----------------------------------
            Add | Addc | Adde | Addze | Addme | Subf | Subfc | Subfe | Subfze | Subfme | Neg => {
                self.lower_addish(insn.kind, f)?
            }
            Subfic => {
                // RT ← SI - RA, i.e. SI + ~RA + 1, and CA is always written.
                self.lower_sum(
                    f,
                    Sum {
                        x: SumKind::Const(f.si() as u32),
                        inv: true,
                        y: SumSecond::Ra,
                        ca: CarryIn::One,
                        ca_out: true,
                        ov: true,
                    },
                );
            }
            Addic | AddicS => {
                // RA + SI, always writing CA.  `addic.` additionally records — there
                // is no Rc bit in its encoding, so the flag is taken from the kind.
                let r = self.lower_sum(
                    f,
                    Sum {
                        x: SumKind::Ra,
                        inv: false,
                        y: SumSecond::Imm(f.si() as u32),
                        ca: CarryIn::Zero,
                        ca_out: true,
                        ov: false,
                    },
                );
                if matches!(insn.kind, AddicS) {
                    self.record_cr0(r, SoBit::XerSo);
                }
            }
            Addi => {
                let ea = self.ea_d(f, f.si(), false);
                self.set_gpr(f.rt(), ea);
            }
            Addis => {
                let ea = self.ea_d(f, f.si() << 16, false);
                self.set_gpr(f.rt(), ea);
            }
            Mulli => {
                let base = if f.ra() == 0 { self.zero() } else { self.get_gpr(f.ra()) };
                let r = self.mul(Self::reg(base), Self::um(f.si() as u32), Width::W32);
                if f.rc() {
                    self.record_cr0(r, SoBit::XerSo);
                }
                self.set_gpr(f.rt(), r);
            }
            Mullw => self.lower_mullw(f),
            Mulhw | Mulhwu => self.lower_mulhigh(matches!(insn.kind, Mulhwu), f),
            Divw | Divwu => self.lower_div(matches!(insn.kind, Divwu), f),

            // ---- logical ---------------------------------------------------
            And | Or | Xor | Nand | Nor | Eqv | Andc | Orc => self.lower_logical(insn.kind, f),
            Ori | Oris | Xori | Xoris => self.lower_logical_imm(f),
            AndiS | AndisS => self.lower_andi(f, matches!(insn.kind, AndisS)),

            // ---- shifts / rotates / extends --------------------------------
            Slw | Srw | Sraw => self.lower_shift_reg(insn.kind, f),
            Srawi => self.lower_srawi(f),
            Rlwinm | Rlwimi | Rlwnm => self.lower_rotate(insn.kind, f),
            Cntlzw => self.lower_cntlzw(f),
            Extsb | Extsh => self.lower_extend(insn.kind, f),
            Popcntb => self.lower_popcntb(f),

            // ---- compare / trap --------------------------------------------
            Cmp | Cmpl => self.lower_cmp(f, matches!(insn.kind, Cmp)),
            Cmpi | Cmpli => self.lower_cmpi(f, matches!(insn.kind, Cmpi)),
            Tw | Twi => self.lower_trap_word(f, matches!(insn.kind, Twi)),
            Sc => {
                let one = self.immv(1, Width::W32);
                self.raise(0xC00, one);
                self.exit_imm(self.next_pc);
            }

            // ---- CR / XER ---------------------------------------------------
            Mfcr => {
                let cr = self.get_cr();
                self.set_gpr(f.rt(), cr);
            }
            Mtcrf => self.lower_mtcrf(f),
            Mcrf => {
                let bits = self.crf(f.bfa());
                self.set_crf(f.bf(), bits);
            }
            Crand | Crandc | Cror | Crorc | Crxor | Crnand | Crnor | Creqv => {
                self.lower_cr_logical(insn.kind, f)
            }
            Mcrxr => self.lower_mcrxr(),

            // ---- SPR / MSR / segments / exception return -------------------
            MfSpr => self.lower_mfspr(f)?,
            MtSpr => self.lower_mtspr(f)?,
            MfMsr => {
                let msr = self.get_misc(PpcMisc::Msr);
                self.set_gpr_masked(f.rt(), msr);
            }
            MtMsr => {
                let v = self.get_gpr(f.rt());
                self.set_misc(PpcMisc::Msr, v);
                // MSR changes (MMU / cache enables) must reach the emulation layer.
                self.intr_raw(PpcIntr::SprSideEffect, vec![Self::imm(0), Self::reg(v)], None);
            }
            MfSr | MtSr | MfSrin | MtSrin => self.lower_seg_reg(insn.kind, f),
            Rfi => self.lower_rfi(),

            // ---- branches ----------------------------------------------------
            B => self.lower_b(f),
            Bc => self.lower_bc(f),
            Bclr => self.lower_bclr(f),
            Bcctr => self.lower_bcctr(f),

            // ---- integer loads / stores ------------------------------------
            Lwz | Lbz | Lhz | Lha => {
                let (width, sext) = int_load_shape(insn.kind);
                let ea = self.ea_d(f, f.si(), false);
                let v = self.load(Self::reg(ea), width, sext);
                self.set_gpr(f.rt(), v);
            }
            Lwzu | Lbzu | Lhzu | Lhau => {
                let (width, sext) = int_load_shape(insn.kind);
                let ea = self.ea_d(f, f.si(), true);
                let v = self.load(Self::reg(ea), width, sext);
                self.set_gpr(f.rt(), v);
                // rA is updated after the access, and with the *new* address, so a
                // trapped load leaves rA untouched just like the hardware.
                self.set_gpr(f.ra(), ea);
            }
            Stw | Stb | Sth | Stbu | Sthu => {
                let width = int_store_width(insn.kind);
                let ea = self.ea_d(f, f.si(), false);
                let v = self.get_gpr(f.rt());
                self.store(Self::reg(ea), Self::reg(v), width);
            }
            Stwu => {
                let ea = self.ea_d(f, f.si(), true);
                let v = self.get_gpr(f.rt());
                self.store(Self::reg(ea), Self::reg(v), Width::W32);
                self.set_gpr(f.ra(), ea);
            }
            Lwzx | Lbzx | Lhzx | Lhax => {
                let (width, sext) = int_load_shape_indexed(insn.kind);
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.load(Self::reg(ea), width, sext);
                self.set_gpr(f.rt(), v);
            }
            Lwzux | Lbzux | Lhzux | Lhaux => {
                let (width, sext) = int_load_shape_indexed(insn.kind);
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.load(Self::reg(ea), width, sext);
                self.set_gpr(f.rt(), v);
                self.set_gpr(f.ra(), ea);
            }
            Stwx | Stbx | Sthx => {
                let width = int_store_width_indexed(insn.kind);
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.get_gpr(f.rt());
                self.store(Self::reg(ea), Self::reg(v), width);
            }
            Stwux | Stbux | Sthux => {
                let width = int_store_width_indexed(insn.kind);
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.get_gpr(f.rt());
                self.store(Self::reg(ea), Self::reg(v), width);
                self.set_gpr(f.ra(), ea);
            }
            Lhbrx | Lwbrx => {
                let width = if matches!(insn.kind, Lhbrx) { Width::W16 } else { Width::W32 };
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.load_rev(Self::reg(ea), width, false);
                self.set_gpr(f.rt(), v);
            }
            Sthbrx | Stwbrx => {
                let width = if matches!(insn.kind, Sthbrx) { Width::W16 } else { Width::W32 };
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.get_gpr(f.rt());
                self.store_rev(Self::reg(ea), Self::reg(v), width);
            }
            Lmw | Stmw => {
                let ea = self.ea_d(f, f.si(), false);
                self.intr_raw(
                    PpcIntr::ListCopy,
                    vec![
                        Self::imm(f.rt() as i64),
                        Self::reg(ea),
                        Self::imm(if matches!(insn.kind, Stmw) { 1 } else { 0 }),
                    ],
                    None,
                );
                if matches!(insn.kind, Lmw) {
                    // Every GPR from rt upward changed; no forwarding entry for a
                    // GPR can survive that.
                    self.fwd.retain(|k, _| !matches!(k, FwdKey::Gpr(_)));
                }
            }
            Lswi | Lswx | Stswi | Stswx => self.lower_string_copy(insn.kind, f),
            Lwarx => {
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.intr(PpcIntr::Lwarx, vec![Self::reg(ea), Self::imm(f.woe() as i64)]);
                self.set_gpr(f.rt(), v);
            }
            Stwcx => {
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.get_gpr(f.rt());
                let bits = self.intr(PpcIntr::Stwcx, vec![Self::reg(ea), Self::reg(v)]);
                self.set_crf(0, bits);
            }

            // ---- cache control ---------------------------------------------
            Dcbf | Dcbi | Dcbst | Dcbt | Dcbtst | Dcbz | Icbi | Icbt => {
                self.lower_cache(insn.kind, f)
            }
            Tlbie => {
                let v = self.get_gpr(f.rb());
                self.intr_raw(PpcIntr::SprSideEffect, vec![Self::imm(306), Self::reg(v)], None);
                // Translation changed: end the block so nothing already chained
                // through the old mapping stays reachable, and let the dispatcher
                // re-enter at the next instruction.
                self.exit_imm(self.next_pc);
            }
            Eciwx | Ecowx => {
                // Device-control accesses: not lowered by this build.  They need
                // the runtime's IO path (ordering plus possible page-side
                // effects), and truncating the block so the interpreter runs them
                // is strictly better than approximating them with a normal load.
                return Err(self.unimplemented(insn));
            }

            // ---- scalar FP loads / stores --------------------------------
            Lfd | Lfdu => {
                let update = matches!(insn.kind, Lfdu);
                let ea = self.ea_d(f, f.si(), update);
                let v = self.load(Self::reg(ea), Width::W64, false);
                self.set_fpr(f.frt(), v);
                if update {
                    self.set_gpr(f.ra(), ea);
                }
            }
            Lfdx | Lfdux => {
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.load(Self::reg(ea), Width::W64, false);
                self.set_fpr(f.frt(), v);
                if matches!(insn.kind, Lfdux) {
                    self.set_gpr(f.ra(), ea);
                }
            }
            Lfs | Lfsu => {
                let update = matches!(insn.kind, Lfsu);
                let ea = self.ea_d(f, f.si(), update);
                self.lower_load_single(ea, f.frt());
                if update {
                    self.set_gpr(f.ra(), ea);
                }
            }
            Lfsx | Lfsux => {
                let ea = self.ea_x(f.ra(), f.rb());
                self.lower_load_single(ea, f.frt());
                if matches!(insn.kind, Lfsux) {
                    self.set_gpr(f.ra(), ea);
                }
            }
            Stfd | Stfdu => {
                let update = matches!(insn.kind, Stfdu);
                let ea = self.ea_d(f, f.si(), update);
                let v = self.get_fpr(f.frt());
                self.store(Self::reg(ea), Self::reg(v), Width::W64);
                if update {
                    self.set_gpr(f.ra(), ea);
                }
            }
            Stfdx | Stfdux => {
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.get_fpr(f.frt());
                self.store(Self::reg(ea), Self::reg(v), Width::W64);
                if matches!(insn.kind, Stfdux) {
                    self.set_gpr(f.ra(), ea);
                }
            }
            Stfs | Stfsu => {
                let update = matches!(insn.kind, Stfsu);
                let ea = self.ea_d(f, f.si(), update);
                self.lower_store_single(ea, f.frt());
                if update {
                    self.set_gpr(f.ra(), ea);
                }
            }
            Stfsx | Stfsux => {
                let ea = self.ea_x(f.ra(), f.rb());
                self.lower_store_single(ea, f.frt());
                if matches!(insn.kind, Stfsux) {
                    self.set_gpr(f.ra(), ea);
                }
            }
            Stfiwx => {
                // The other half of the `fctiwz` cast idiom: store the container's
                // low 32 bits, which is where that instruction puts the integer.
                let ea = self.ea_x(f.ra(), f.rb());
                let v = self.get_fpr(f.frt());
                let low = self.and(Self::reg(v), Self::um(LO32), Width::W64);
                self.store(Self::reg(ea), Self::reg(low), Width::W32);
            }
            Lfiwax | Lfiwzx => {
                let ea = self.ea_x(f.ra(), f.rb());
                let raw = self.load(Self::reg(ea), Width::W32, false);
                let zero_extend = Self::imm(if matches!(insn.kind, Lfiwzx) { 1 } else { 0 });
                let widened = self.intr(PpcIntr::FpFromInt, vec![Self::reg(raw), zero_extend]);
                self.set_fpr(f.frt(), widened);
            }

            // ---- scalar FP arithmetic ---------------------------------------
            Fadd | Fsub | Fmul | Fdiv | Fres | Fmadd | Fmsub | Fnmsub | Fnmadd => {
                self.lower_fp_arith(insn.kind, f)?
            }
            Frsp => {
                let a = self.get_fpr(f.fra());
                let r = self.intr(PpcIntr::FpRoundToSingle, vec![Self::reg(a)]);
                self.set_fpr(f.frt(), r);
            }
            Fctiw | Fctiwz => {
                let a = self.get_fpr(f.fra());
                let id = if matches!(insn.kind, Fctiwz) { PpcIntr::FpCtiwz } else { PpcIntr::FpCtiw };
                let r = self.intr(id, vec![Self::reg(a)]);
                self.set_fpr(f.frt(), r);
            }
            Fmr => {
                let a = self.get_fpr(f.fra());
                if f.frt() != f.fra() {
                    self.set_fpr(f.frt(), a);
                }
            }
            Fneg | Fabs | Fnabs => {
                let a = self.get_fpr(f.fra());
                let sign = 1i64 << 63;
                let r = match insn.kind {
                    Fneg => self.xor(Self::reg(a), Self::imm(sign), Width::W64),
                    Fabs => self.and(Self::reg(a), Self::imm(!sign), Width::W64),
                    _ => self.or(Self::reg(a), Self::imm(sign), Width::W64),
                };
                self.set_fpr(f.frt(), r);
            }
            Fsel => {
                // FRT ← (FRA ≥ 0) ? FRC : FRB.  The container's sign bit is the
                // predicate, so this is a plain bit-select: no FP helper involved.
                let a = self.get_fpr(f.fra());
                let b = self.get_fpr(f.frb());
                let c = self.get_fpr(f.frc_ax());
                let neg = self.shift(Self::reg(a), Self::imm(63), Width::W64, ShiftKind::Right);
                let r = self.select(neg, c, b, Width::W64);
                self.set_fpr(f.frt(), r);
            }
            Fcmpu | Fcmpo => {
                let a = self.get_fpr(f.fra());
                let b = self.get_fpr(f.frb());
                let ordered = Self::imm(if matches!(insn.kind, Fcmpo) { 1 } else { 0 });
                let bits = self.intr(PpcIntr::FpCmp, vec![Self::reg(a), Self::reg(b), ordered]);
                self.set_crf(f.bf(), bits);
            }
            Frsqrte => {
                let a = self.get_fpr(f.fra());
                let r = self.intr(PpcIntr::FpRsqrt, vec![Self::reg(a)]);
                self.set_fpr(f.frt(), r);
            }
            Mcrfs => {
                let bits = self.intr(
                    PpcIntr::FpCrfFromFpscr,
                    vec![Self::imm(f.bf() as i64), Self::imm(f.bfa() as i64)],
                );
                self.set_crf(f.bf(), bits);
            }
            Mffs => {
                // FPSCR → FRT: the 32-bit register lands in the container's low
                // half, which is what "FPR bits 32:63" means.
                let fpscr = self.get_fpscr();
                let low = self.and(Self::reg(fpscr), Self::um(LO32), Width::W64);
                self.set_fpr(f.frt(), low);
            }
            // `mtfsf` / `mtfsfi` are deliberately NOT lowered yet: their FP register
            // operand and field-mask live in the "XFL" form's bit positions, which
            // differ from every other FP encoding (FRA is not in bits 11:15 there),
            // and I could not confirm those positions from a source I trust.
            // Guessing them would silently mis-set FPSCR — rounding mode, or the
            // exception enables — which is exactly the kind of wrong-but-plausible
            // bug this file is written to avoid.  Truncating the block hands them to
            // the fallback interpreter instead.  `mffs`, `mtfsb0`, `mtfsb1` and
            // `mcrfs` (whose fields *are* confirmed — see `fields.rs` and the
            // generated corpus) do work.
            Mtfsf | Mtfsfi => return Err(self.unimplemented(insn)),
            Mtfsb0 | Mtfsb1 => {
                // BT (bits 6:10) names the FPSCR bit; PPC bit n of the 32-bit
                // register is host bit 31-n.
                let bt = (f.rt() & 31) as u32;
                let bit = self.immv(if matches!(insn.kind, Mtfsb1) { 1 } else { 0 }, Width::W32);
                self.set_fpscr_bit(31 - bt, bit);
            }

            // ---- paired singles --------------------------------------------
            PsAdd | PsSub | PsMul | PsDiv | PsMadd | PsMsub | PsNmadd | PsNmsub | PsMadds0
            | PsMadds1 | PsMuls0 | PsMuls1 | PsRes | PsRsqrte | PsSel | PsSum0 | PsSum1 => {
                self.lower_ps_arith(insn.kind, f)?
            }
            PsMerge00 | PsMerge01 | PsMerge10 | PsMerge11 => self.lower_ps_merge(insn.kind, f),
            PsMr | PsNeg | PsAbs | PsNabs => self.lower_ps_unary(insn.kind, f),
            PsCmpu0 | PsCmpu1 | PsCmpo0 | PsCmpo1 => self.lower_ps_cmp(insn.kind, f),
            PsqL | PsqLu | PsqSt | PsqStu | PsqLx | PsqLux | PsqStx | PsqStux => {
                self.lower_psq(insn.kind, f)
            }
        }

        Ok(())
    }

    #[inline]
    fn unimplemented(&self, insn: &DecodedPpc) -> PpcLowerError {
        PpcLowerError::Unimplemented {
            pc: insn.pc,
            raw: insn.raw,
            mnemonic: insn.name,
        }
    }

    // ------------------------------------------------------------------------
    // integer add / subtract
    // ------------------------------------------------------------------------

    /// `add`, `addc`, `adde`, `addze`, `addme`, `subf`, `subfc`, `subfe`,
    /// `subfze`, `subfme`, `neg` — one algorithm, because the ISA defines them all
    /// as a 33-bit adder:
    ///
    /// ```text
    ///   total = X + Y' + K + carry_in          (Y' = ~RA for the subf forms)
    ///   RT    = total & 0xFFFFFFFF
    ///   CA    = total >> 32
    ///   OV    = carry-into-bit-31 XOR carry-out-of-bit-31   (oe forms only)
    ///   SO    = SO ^ OV                                      (oe forms only)
    /// ```
    ///
    /// with the per-instruction choices in [`SumKind`]/[`SumSecond`]/[`CarryIn`].
    /// Deriving CA and OV from the W64 sum's bits (instead of from host flags) is
    /// what makes `adde`/`subfme` right: PPC's CA for those is the carry out of the
    /// *three-input* add, and x86's CF would only be that if the carry-in were added
    /// with `adc`, which the IR cannot express.
    fn lower_addish(&mut self, kind: PpcKind, f: PpcFields) -> Result<(), PpcLowerError> {
        use PpcKind::*;
        let spec = match kind {
            Add => Sum { x: SumKind::Ra, inv: false, y: SumSecond::Rb, ca: CarryIn::Zero, ca_out: false, ov: true },
            Addc => Sum { x: SumKind::Ra, inv: false, y: SumSecond::Rb, ca: CarryIn::Zero, ca_out: true, ov: true },
            Adde => Sum { x: SumKind::Ra, inv: false, y: SumSecond::Rb, ca: CarryIn::Xer, ca_out: true, ov: true },
            // The four "no second register" forms: their RB field is *reserved*, so
            // reading GPR 0 there would be a bug — hence an explicit constant.
            Addze => Sum { x: SumKind::Ra, inv: false, y: SumSecond::Imm(0), ca: CarryIn::Xer, ca_out: true, ov: true },
            Addme => Sum { x: SumKind::Ra, inv: false, y: SumSecond::Imm(LO32), ca: CarryIn::Xer, ca_out: true, ov: true },
            Subf => Sum { x: SumKind::Rb, inv: true, y: SumSecond::Ra, ca: CarryIn::One, ca_out: false, ov: true },
            Subfc => Sum { x: SumKind::Rb, inv: true, y: SumSecond::Ra, ca: CarryIn::One, ca_out: true, ov: true },
            Subfe => Sum { x: SumKind::Rb, inv: true, y: SumSecond::Ra, ca: CarryIn::Xer, ca_out: true, ov: true },
            Subfze => Sum { x: SumKind::Zero, inv: true, y: SumSecond::Ra, ca: CarryIn::Xer, ca_out: true, ov: true },
            Subfme => Sum { x: SumKind::Const(LO32), inv: true, y: SumSecond::Ra, ca: CarryIn::Xer, ca_out: true, ov: true },
            Neg => Sum { x: SumKind::Zero, inv: true, y: SumSecond::Ra, ca: CarryIn::One, ca_out: true, ov: false },
            _ => return Ok(()),
        };
        self.lower_sum(f, spec);
        Ok(())
    }

    /// Shared body of the add/subtract family.  Returns the RT value so the
    /// `addic.`-style callers (no Rc bit) can still record CR0.
    fn lower_sum(&mut self, f: PpcFields, s: Sum) -> VReg {
        let xv = match s.x {
            SumKind::Ra => self.get_gpr(f.ra()),
            SumKind::Rb => self.get_gpr(f.rb()),
            SumKind::Zero => self.zero(),
            SumKind::Const(c) => self.immv(c as i64, Width::W32),
        };
        let yv = match s.y {
            SumSecond::Ra => self.get_gpr(f.ra()),
            SumSecond::Rb => self.get_gpr(f.rb()),
            SumSecond::Imm(c) => self.immv(c as i64, Width::W32),
            // No current call site constructs SumSecond::None (every real
            // "no second register" form here uses an explicit SumSecond::Imm
            // instead, e.g. Addze/Addme above). Treated as the additive
            // identity so the match is exhaustive; this path is currently
            // untested — verify against the ISA before relying on it if a
            // future instruction actually needs it.
            SumSecond::None => self.zero(),
        };
        let yv = if s.inv {
            self.xor(Self::reg(yv), Self::um(LO32), Width::W32)
        } else {
            yv
        };

        // Sum at W64 on zero-extended operands: both carries become ordinary bits.
        let mut total = self.add(Self::reg(xv), Self::reg(yv), Width::W64);
        let carry = match s.ca {
            CarryIn::Zero => None,
            CarryIn::One => Some(self.immv(1, Width::W32)),
            CarryIn::Xer => Some(self.xer_bit(XER_CA)),
        };
        if let Some(c) = carry {
            total = self.add(Self::reg(total), Self::reg(c), Width::W64);
        }
        let result = self.and(Self::reg(total), Self::um(LO32), Width::W64);

        let mut xer_bits: Vec<(u32, VReg)> = Vec::new();
        if s.ca_out {
            let ca = self.xer_carry_of(total);
            xer_bits.push((XER_CA, ca));
        }
        if f.oe() && s.ov {
            let o = self.sum_overflow(xv, yv, carry);
            xer_bits.push((XER_OV, o));
            let so_old = self.xer_bit(XER_SO);
            let so_new = self.xor(Self::reg(so_old), Self::reg(o), Width::W32);
            xer_bits.push((XER_SO, so_new));
        }
        if !xer_bits.is_empty() {
            self.set_xer_bits(&xer_bits);
        }
        if f.rc() {
            self.record_cr0(result, SoBit::XerSo);
        }
        self.set_gpr(f.rt(), result);
        result
    }

    fn lower_mullw(&mut self, f: PpcFields) {
        let a = self.get_gpr(f.ra());
        let b = self.get_gpr(f.rb());
        let r = self.mul(Self::reg(a), Self::reg(b), Width::W32);
        if f.oe() {
            // mullw overflows iff the high word is not the low word's sign
            // extension, i.e. iff mulhw's result != (r >> 31, arithmetically).
            let hi = self.intr(PpcIntr::MulHigh, vec![Self::reg(a), Self::reg(b), Self::imm(0)]);
            let sxt = self.shift(Self::reg(r), Self::imm(31), Width::W32, ShiftKind::Arith);
            let diff = self.xor(Self::reg(hi), Self::reg(sxt), Width::W32);
            let z = self.is_zero_flag(diff, Width::W32);
            let one = self.immv(1, Width::W32);
            let ov = self.sub(Self::reg(one), Self::reg(z), Width::W32);
            let so_new = self.xor_two_xer_so(ov);
            self.set_xer_bits(&[(XER_OV, ov), (XER_SO, so_new)]);
        }
        if f.rc() {
            self.record_cr0(r, SoBit::XerSo);
        }
        self.set_gpr(f.rt(), r);
    }

    /// `SO ← SO XOR OV` as a value (callers pair it with the new OV in one
    /// read-modify-write so XER is touched once).
    fn xor_two_xer_so(&mut self, ov: VReg) -> VReg {
        let so_old = self.xer_bit(XER_SO);
        self.xor(Self::reg(so_old), Self::reg(ov), Width::W32)
    }

    /// `mulhw`/`mulhwu`: no multiply-high in the IR ⇒ one helper.
    fn lower_mulhigh(&mut self, unsigned: bool, f: PpcFields) {
        let a = self.get_gpr(f.ra());
        let b = self.get_gpr(f.rb());
        let r = self.intr(PpcIntr::MulHigh, vec![Self::reg(a), Self::reg(b), Self::imm(unsigned as i64)]);
        if f.rc() {
            self.record_cr0(r, SoBit::XerSo);
        }
        self.set_gpr(f.rt(), r);
    }

    /// `divw`/`divwu`.  PPC divides are *not* exceptions on divide-by-zero: the
    /// result is 0 (signed: -1 for the INT_MIN / -1 overflow), and OV records it.
    fn lower_div(&mut self, unsigned: bool, f: PpcFields) {
        let a = self.get_gpr(f.ra());
        let b = self.get_gpr(f.rb());
        let r = self.intr(PpcIntr::DivWord, vec![Self::reg(a), Self::reg(b), Self::imm(unsigned as i64)]);
        if f.oe() {
            let zero = self.zero();
            let b_zero = self.is_zero_flag(b, Width::W32);
            let mut ov = b_zero;
            if !unsigned {
                // Also set for INT_MIN / -1.
                let int_min = self.immv(0x8000_0000, Width::W32);
                let minus_one = self.immv(LO32 as i64, Width::W32);
                let a_diff = self.sub(Self::reg(a), Self::reg(int_min), Width::W32);
                let b_diff = self.sub(Self::reg(b), Self::reg(minus_one), Width::W32);
                let a_min = self.is_zero_flag(a_diff, Width::W32);
                let b_m1 = self.is_zero_flag(b_diff, Width::W32);
                let both = self.and(Self::reg(a_min), Self::reg(b_m1), Width::W32);
                ov = self.or(Self::reg(ov), Self::reg(both), Width::W32);
            }
            let so_new = self.xor_two_xer_so(ov);
            self.set_xer_bits(&[(XER_OV, ov), (XER_SO, so_new)]);
            let _ = zero;
        }
        if f.rc() {
            self.record_cr0(r, SoBit::XerSo);
        }
        self.set_gpr(f.rt(), r);
    }

    // ------------------------------------------------------------------------
    // logical / shift / rotate
    // ------------------------------------------------------------------------

    /// The X-form logical group: destination is **RA** (bits 11:15), sources are
    /// RS and RB.  Their record form sets CR0's SO bit to 1 (a logical op cannot
    /// overflow), which is `SoBit::One`.
    fn lower_logical(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        let rs = self.get_gpr(f.rt());
        let rb = self.get_gpr(f.rb());
        let r = match kind {
            And => self.and(Self::reg(rs), Self::reg(rb), Width::W32),
            Or => self.or(Self::reg(rs), Self::reg(rb), Width::W32),
            Xor => self.xor(Self::reg(rs), Self::reg(rb), Width::W32),
            Nand => {
                let t = self.and(Self::reg(rs), Self::reg(rb), Width::W32);
                self.xor(Self::reg(t), Self::um(LO32), Width::W32)
            }
            Nor => {
                let t = self.or(Self::reg(rs), Self::reg(rb), Width::W32);
                self.xor(Self::reg(t), Self::um(LO32), Width::W32)
            }
            Eqv => {
                let t = self.xor(Self::reg(rs), Self::reg(rb), Width::W32);
                self.xor(Self::reg(t), Self::um(LO32), Width::W32)
            }
            Andc => {
                let nb = self.xor(Self::reg(rb), Self::um(LO32), Width::W32);
                self.and(Self::reg(rs), Self::reg(nb), Width::W32)
            }
            _ => {
                let nb = self.xor(Self::reg(rb), Self::um(LO32), Width::W32);
                self.or(Self::reg(rs), Self::reg(nb), Width::W32)
            }
        };
        if f.rc() {
            self.record_cr0(r, SoBit::One);
        }
        self.set_gpr(f.ra(), r);
    }

    fn lower_logical_imm(&mut self, f: PpcFields) {
        let rs = self.get_gpr(f.rt());
        let mut imm = f.ui();
        if f.op() == 25 || f.op() == 27 {
            imm <<= 16; // oris / xoris
        }
        let r = if f.op() == 24 || f.op() == 25 {
            self.or(Self::reg(rs), Self::um(imm), Width::W32)
        } else {
            self.xor(Self::reg(rs), Self::um(imm), Width::W32)
        };
        self.set_gpr(f.ra(), r);
    }

    /// `andi.`/`andis.`: always record, and the immediate is *zero*-extended (the
    /// `.`-in-the-mnemonic forms have no Rc bit and no sign extension).
    fn lower_andi(&mut self, f: PpcFields, shifted: bool) {
        let rs = self.get_gpr(f.rt());
        let mut imm = f.ui();
        if shifted {
            imm <<= 16;
        }
        let r = self.and(Self::reg(rs), Self::um(imm), Width::W32);
        self.record_cr0(r, SoBit::One);
        self.set_gpr(f.ra(), r);
    }

    /// `slw`/`srw`/`sraw`: the shift count is a register, and PPC's semantics for a
    /// count ≥ 32 (zero for the logical shifts, sign broadcast for `sraw`) are *not*
    /// the host's (x86 masks the count to 5 bits), so the count is range-checked
    /// here rather than assumed.
    fn lower_shift_reg(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        let rs = self.get_gpr(f.rt());
        let rb = self.get_gpr(f.rb());
        let big = self.and(Self::reg(rb), Self::um(!31), Width::W32);
        let big_nz = self.is_nonzero(big, Width::W32);
        let one = self.immv(1, Width::W32);
        let valid = self.sub(Self::reg(one), Self::reg(big_nz), Width::W32);

        let raw = match kind {
            Slw => {
                let t = self.shift(Self::reg(rs), Self::reg(rb), Width::W32, ShiftKind::Left);
                self.mul(Self::reg(t), Self::reg(valid), Width::W32)
            }
            Srw => {
                let t = self.shift(Self::reg(rs), Self::reg(rb), Width::W32, ShiftKind::Right);
                self.mul(Self::reg(t), Self::reg(valid), Width::W32)
            }
            _ => {
                // sraw: clamping the count to 31 makes an out-of-range count
                // produce the sign broadcast the architecture requires.
                let clamped = self.and(Self::reg(rb), Self::um(31), Width::W32);
                self.shift(Self::reg(rs), Self::reg(clamped), Width::W32, ShiftKind::Arith)
            }
        };

        if matches!(kind, Sraw) {
            self.sraw_carry(rs, rb);
        }
        if f.rc() {
            self.record_cr0(raw, SoBit::One);
        }
        self.set_gpr(f.ra(), raw);
    }

    /// `sraw`/`srawi`'s CA rule: "CA ← 1 if the value is negative and any 1 bit was
    /// shifted out", and CA is never *cleared*.  OR-ing into the bit expresses
    /// "unchanged" exactly.
    fn sraw_carry(&mut self, rs: VReg, count: VReg) {
        let clamped = self.and(Self::reg(count), Self::um(31), Width::W32);
        let one = self.immv(1, Width::W32);
        let shifted_one = self.shift(Self::reg(one), Self::reg(clamped), Width::W32, ShiftKind::Left);
        let low_mask = self.sub(Self::reg(shifted_one), Self::reg(one), Width::W32);
        let lost = self.and(Self::reg(rs), Self::reg(low_mask), Width::W32);
        let any_lost = self.is_nonzero(lost, Width::W32);
        let neg = self.sign32(rs);
        let set = self.and(Self::reg(any_lost), Self::reg(neg), Width::W32);
        let ca_old = self.xer_bit(XER_CA);
        let ca_new = self.or(Self::reg(ca_old), Self::reg(set), Width::W32);
        self.set_xer_bits(&[(XER_CA, ca_new)]);
    }

    fn lower_srawi(&mut self, f: PpcFields) {
        let rs = self.get_gpr(f.rt());
        let sh = f.sh();
        let r = self.shift(Self::reg(rs), Self::imm(sh as i64), Width::W32, ShiftKind::Arith);
        if sh > 0 {
            let count = self.immv(sh as i64, Width::W32);
            self.sraw_carry(rs, count);
        }
        if f.rc() {
            self.record_cr0(r, SoBit::One);
        }
        self.set_gpr(f.ra(), r);
    }

    /// `rlwinm rA,rS,SH,MB,ME`  → rA = rotl(rS,SH) & mask(MB..ME)
    /// `rlwimi rA,rS,SH,MB,ME` → rA = (rotl(rS,SH) & mask) | (rA & ~mask)
    /// `rlwnm  rA,rS,rB`       → SH/MB/ME all read from RB's bit fields
    ///
    /// MB/ME are immediates for the first two, so the mask is computed at translate
    /// time and costs one `And`.  The third needs a runtime mask, which is the only
    /// place in this file that does arithmetic on a mask itself
    /// ([`Ctx::dynamic_mask`]).
    fn lower_rotate(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        let rs = self.get_gpr(f.rt());
        let (rotated, mask_reg) = match kind {
            Rlwnm => {
                let rb = self.get_gpr(f.rb());
                let sh = self.and(Self::reg(rb), Self::um(31), Width::W32);
                let rotated = self.rotl32_reg(rs, sh);
                let (mb, me) = {
                    let mb = self.rb_mb(rb);
                    let me = self.rb_me(rb);
                    (mb, me)
                };
                let mask = self.dynamic_mask(mb, me);
                (rotated, Some(mask))
            }
            _ => (self.rotl32_imm(rs, f.sh()), None),
        };

        let masked = match mask_reg {
            Some(m) => self.and(Self::reg(rotated), Self::reg(m), Width::W32),
            None => self.and(Self::reg(rotated), Self::um(rotate_mask(f.mb(), f.me())), Width::W32),
        };

        let result = if matches!(kind, Rlwimi) {
            let dst = self.get_gpr(f.ra());
            let keep_mask = match mask_reg {
                Some(m) => {
                    let all = self.immv(LO32 as i64, Width::W32);
                    Some(self.xor(Self::reg(m), Self::reg(all), Width::W32))
                }
                None => Some(self.immv(!(rotate_mask(f.mb(), f.me())) as i64, Width::W32)),
            };
            let keep = self.and(Self::reg(dst), Self::reg(keep_mask.unwrap()), Width::W32);
            self.or(Self::reg(masked), Self::reg(keep), Width::W32)
        } else {
            masked
        };

        if f.rc() {
            self.record_cr0(result, SoBit::One);
        }
        self.set_gpr(f.ra(), result);
    }

    #[inline]
    fn rb_mb(&mut self, rb: VReg) -> VReg {
        // RB's PPC bits 21:25 = host bits 10..6.
        let s = self.shift(Self::reg(rb), Self::imm(6), Width::W32, ShiftKind::Right);
        self.and(Self::reg(s), Self::um(31), Width::W32)
    }

    #[inline]
    fn rb_me(&mut self, rb: VReg) -> VReg {
        // RB's PPC bits 26:30 = host bits 5..1.
        let s = self.shift(Self::reg(rb), Self::imm(1), Width::W32, ShiftKind::Right);
        self.and(Self::reg(s), Self::um(31), Width::W32)
    }

    /// Build `mask = PPC bits mb..=me` from two runtime values, branch-free, and
    /// correct for the wrap-around case (`mb > me`) that `rlwinm`-style code relies
    /// on.
    ///
    /// In host terms the wanted set is `bits (31-me)..=(31-mb)`, so with
    /// `A = ~0 << (31-me)` (everything at or above the low end) and
    /// `B = ~0 >> mb` (everything at or below the high end):
    ///   • mb ≤ me  →  mask = A & B
    ///   • mb >  me →  mask = A | B, which is `(A & B) | (A ^ B)`
    /// so one sign-derived all-ones mask selects between them with two extra ops and
    /// no shifts ≥ 32 anywhere (both shift counts are ≤ 31 by construction).
    fn dynamic_mask(&mut self, mb: VReg, me: VReg) -> VReg {
        let all = self.immv(LO32 as i64, Width::W32);
        let c31 = self.immv(31, Width::W32);
        let lo_shift = self.sub(Self::reg(c31), Self::reg(me), Width::W32);
        let a = self.shift(Self::reg(all), Self::reg(lo_shift), Width::W32, ShiftKind::Left);
        let b = self.shift(Self::reg(all), Self::reg(mb), Width::W32, ShiftKind::Right);
        let inter = self.and(Self::reg(a), Self::reg(b), Width::W32);
        let differ = self.xor(Self::reg(a), Self::reg(b), Width::W32);
        // wrap = 1 when mb > me, i.e. when (mb - me - 1) >= 0.
        let me_plus_1 = self.add(Self::reg(me), Self::um(1), Width::W32);
        let gap = self.sub(Self::reg(mb), Self::reg(me_plus_1), Width::W32);
        let neg = self.sign32(gap);
        let one = self.immv(1, Width::W32);
        let wrap_bit = self.sub(Self::reg(one), Self::reg(neg), Width::W32);
        let wrap_mask = self.mask_of(wrap_bit, Width::W32);
        let extra = self.and(Self::reg(differ), Self::reg(wrap_mask), Width::W32);
        self.or(Self::reg(inter), Self::reg(extra), Width::W32)
    }

    /// `cntlzw`: fold the high bits down and count the survivors with the same
    /// helper `popcntb` needs — which is how a 750CL does it too.
    fn lower_cntlzw(&mut self, f: PpcFields) {
        let rs = self.get_gpr(f.rt());
        let mut v = rs;
        for shift in [1u32, 2, 4, 8, 16] {
            let s = self.shift(Self::reg(v), Self::imm(shift as i64), Width::W32, ShiftKind::Right);
            v = self.or(Self::reg(v), Self::reg(s), Width::W32);
        }
        let pop = self.intr(PpcIntr::Popcntb, vec![Self::reg(v)]);
        let c32 = self.immv(32, Width::W32);
        let r = self.sub(Self::reg(c32), Self::reg(pop), Width::W32);
        if f.rc() {
            self.record_cr0(r, SoBit::One);
        }
        self.set_gpr(f.ra(), r);
    }

    fn lower_extend(&mut self, kind: PpcKind, f: PpcFields) {
        let rs = self.get_gpr(f.rt());
        let amount = if matches!(kind, PpcKind::Extsb) { 24 } else { 16 };
        let r = self.shift(Self::reg(rs), Self::imm(amount), Width::W32, ShiftKind::Arith);
        if f.rc() {
            self.record_cr0(r, SoBit::One);
        }
        self.set_gpr(f.ra(), r);
    }

    fn lower_popcntb(&mut self, f: PpcFields) {
        let rs = self.get_gpr(f.rt());
        let r = self.intr(PpcIntr::Popcntb, vec![Self::reg(rs)]);
        if f.rc() {
            self.record_cr0(r, SoBit::One);
        }
        self.set_gpr(f.ra(), r);
    }

    // ------------------------------------------------------------------------
    // compare / trap / CR
    // ------------------------------------------------------------------------

    /// `cmp`/`cmpl`: SO = sign(RA) for the signed form and 1 for the logical one.
    /// The L bit picks 32- or 64-bit operands; Broadway is a 32-bit core, and a W64
    /// compare of the zero-extended containers is the faithful reading of an
    /// encoding no compiler emits for it.
    fn lower_cmp(&mut self, f: PpcFields, signed: bool) {
        let width = if f.l_width() { Width::W64 } else { Width::W32 };
        let a = self.get_gpr(f.ra());
        let b = self.get_gpr(f.rb());
        let so = if signed { SoBit::SignOf(a, width) } else { SoBit::One };
        self.compare_to_crf(f.bf(), a, b, signed, width, so);
    }

    fn lower_cmpi(&mut self, f: PpcFields, signed: bool) {
        let width = if !signed && f.l_width() { Width::W64 } else { Width::W32 };
        let a = self.get_gpr(f.ra());
        let imm = if signed { f.si() } else { f.ui() as i64 };
        let b = self.immv(self.mask_to_width(imm, width), width);
        let so = if signed { SoBit::SignOf(a, width) } else { SoBit::One };
        self.compare_to_crf(f.bf(), a, b, signed, width, so);
    }

    /// `tw`/`twi`: TO is an immediate in both, so the trap condition is a static
    /// OR of the tested sub-conditions, and `Raise` carries the predicate — which
    /// is what lets a *conditional* trap exist without a conditional branch op.
    fn lower_trap_word(&mut self, f: PpcFields, immediate: bool) {
        let a = self.get_gpr(f.ra());
        let b = if immediate { self.immv(f.si(), Width::W32) } else { self.get_gpr(f.rb()) };
        let d = self.sub(Self::reg(a), Self::reg(b), Width::W32);
        let to = f.to();
        let mut parts: Vec<VReg> = Vec::new();
        if to & 0x10 != 0 {
            // LT: signed less-than = sign(d) corrected by overflow of a-b.
            parts.push(self.sign32(d));
        }
        if to & 0x08 != 0 {
            // GT: positive and nonzero.
            let nz = self.is_nonzero(d, Width::W32);
            let neg = self.sign32(d);
            let one = self.immv(1, Width::W32);
            let pos = self.sub(Self::reg(one), Self::reg(neg), Width::W32);
            parts.push(self.and(Self::reg(nz), Self::reg(pos), Width::W32));
        }
        if to & 0x04 != 0 {
            parts.push(self.is_zero_flag(d, Width::W32));
        }
        if to & 0x02 != 0 {
            // SO: the sign bit of the first operand.
            parts.push(self.sign32(a));
        }
        if to & 0x01 != 0 {
            // UN: always.
            parts.push(self.immv(1, Width::W32));
        }
        let cond = if parts.is_empty() {
            self.zero()
        } else {
            let mut acc = parts[0];
            for p in &parts[1..] {
                acc = self.or(Self::reg(acc), Self::reg(*p), Width::W32);
            }
            acc
        };
        self.raise(0x700, cond);
        // A trap ends the block: whatever follows it is dead from the JIT's point
        // of view until the runtime decides where to resume.
        self.exit_imm(self.next_pc);
    }

    /// `mtcrf FLM, RS`: copy the selected CR nibbles.  FLM is 8 bits (PPC 12:19)
    /// with bit 12 = CRF0, i.e. field `j` is selected by `(flm >> (7-j)) & 1` —
    /// verified against `mtcrf 0xf, r2` = 0x7C40F120.
    fn lower_mtcrf(&mut self, f: PpcFields) {
        let v = self.get_gpr(f.rt());
        let flm = f.flm();
        if flm == 0 {
            return;
        }
        let cr = self.get_cr();
        let mut cur = cr;
        for j in 0..8u32 {
            if (flm >> (7 - j)) & 1 == 0 {
                continue;
            }
            let shift = 28 - 4 * j;
            let field = self.shift(Self::reg(v), Self::imm(shift as i64), Width::W32, ShiftKind::Right);
            let field = self.and(Self::reg(field), Self::um(0xF), Width::W32);
            let kept = self.and(Self::reg(cur), Self::um(!(0xFu32 << shift)), Width::W32);
            let moved = self.shift(Self::reg(field), Self::imm(shift as i64), Width::W32, ShiftKind::Left);
            cur = self.or(Self::reg(kept), Self::reg(moved), Width::W32);
        }
        if cur != cr {
            self.set_cr(cur);
        }
    }

    /// The eight CR-logic ops: `BT ← BA op BB`, where BT/BA/BB are *bit* indices
    /// (6:10 / 11:15 / 16:20).
    fn lower_cr_logical(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        let ba = self.cr_bit(f.rt() as u32);
        let bb = self.cr_bit(f.rb() as u32);
        let bit = match kind {
            Crand => self.and(Self::reg(ba), Self::reg(bb), Width::W32),
            Crandc => {
                let one = self.immv(1, Width::W32);
                let n = self.xor(Self::reg(bb), Self::reg(one), Width::W32);
                self.and(Self::reg(ba), Self::reg(n), Width::W32)
            }
            Cror => self.or(Self::reg(ba), Self::reg(bb), Width::W32),
            Crorc => {
                let one = self.immv(1, Width::W32);
                let n = self.xor(Self::reg(bb), Self::reg(one), Width::W32);
                self.or(Self::reg(ba), Self::reg(n), Width::W32)
            }
            Crxor => self.xor(Self::reg(ba), Self::reg(bb), Width::W32),
            Crnand => {
                let t = self.and(Self::reg(ba), Self::reg(bb), Width::W32);
                let one = self.immv(1, Width::W32);
                self.xor(Self::reg(t), Self::reg(one), Width::W32)
            }
            Crnor => {
                let t = self.or(Self::reg(ba), Self::reg(bb), Width::W32);
                let one = self.immv(1, Width::W32);
                self.xor(Self::reg(t), Self::reg(one), Width::W32)
            }
            _ => {
                let t = self.xor(Self::reg(ba), Self::reg(bb), Width::W32);
                let one = self.immv(1, Width::W32);
                self.xor(Self::reg(t), Self::reg(one), Width::W32)
            }
        };
        self.set_cr_bit(f.rt() as u32, bit);
    }

    /// `mcrxr`: CR1 ← {XER.SO, XER.OV, XER.CA, 1}.
    fn lower_mcrxr(&mut self) {
        let so = self.xer_bit(XER_SO);
        let ov = self.xer_bit(XER_OV);
        let ca = self.xer_bit(XER_CA);
        let l8 = self.shift(Self::reg(so), Self::imm(3), Width::W32, ShiftKind::Left);
        let l4 = self.shift(Self::reg(ov), Self::imm(2), Width::W32, ShiftKind::Left);
        let l2 = self.shift(Self::reg(ca), Self::imm(1), Width::W32, ShiftKind::Left);
        let t = self.or(Self::reg(l8), Self::reg(l4), Width::W32);
        let t = self.or(Self::reg(t), Self::reg(l2), Width::W32);
        let one = self.immv(1, Width::W32);
        let t = self.or(Self::reg(t), Self::reg(one), Width::W32);
        self.set_crf(1, t);
    }

    // ------------------------------------------------------------------------
    // SPR / segments / exception return
    // ------------------------------------------------------------------------

    fn lower_mfspr(&mut self, f: PpcFields) -> Result<(), PpcLowerError> {
        let spr = f.spr();
        match PpcMisc::access(spr) {
            SprAccess::Slot(slot, _writable) => {
                // The time base is the emulator's to update between two reads in one
                // block, so it goes through the volatile path.
                let v = if slot == PpcMisc::Tbl.slot() || slot == PpcMisc::Tbu.slot() {
                    let which = if slot == PpcMisc::Tbl.slot() { PpcMisc::Tbl } else { PpcMisc::Tbu };
                    self.get_misc_volatile(which)
                } else {
                    self.get_misc_slot(slot)
                };
                self.set_gpr_masked(f.rt(), v);
                Ok(())
            }
            SprAccess::Dec => {
                let v = self.get_misc_volatile(PpcMisc::Dec);
                self.set_gpr_masked(f.rt(), v);
                Ok(())
            }
            SprAccess::Tb(half) => {
                let which = if half == 0 { PpcMisc::Tbl } else { PpcMisc::Tbu };
                let v = self.get_misc_volatile(which);
                self.set_gpr_masked(f.rt(), v);
                Ok(())
            }
            SprAccess::Ignore => {
                let z = self.zero();
                self.set_gpr(f.rt(), z);
                Ok(())
            }
            SprAccess::Unknown => Err(PpcLowerError::Unimplemented {
                pc: self.pc,
                raw: 0,
                mnemonic: "mfspr:undefined spr",
            }),
        }
    }

    fn lower_mtspr(&mut self, f: PpcFields) -> Result<(), PpcLowerError> {
        let spr = f.spr();
        let v = self.get_gpr(f.rt());
        match PpcMisc::access(spr) {
            SprAccess::Slot(slot, writable) => {
                if !writable {
                    // Real Broadway hardware makes mtspr to a read-only SPR (e.g.
                    // PVR) either a no-op or an illegal instruction depending on
                    // privilege level; treat it as illegal here rather than
                    // silently accepting a write the guest should never see take
                    // effect. Do not collapse this to `_` — the bool is load-
                    // bearing, not decorative (see PpcMisc::access's table).
                    return Err(PpcLowerError::Illegal { pc: self.pc, raw: f.raw });
                }
                self.set_misc_slot(slot, v);
                if slot == PpcMisc::Lr.slot() || slot == PpcMisc::Ctr.slot() {
                    // Branch targets depend on these; nothing to do here beyond the
                    // store, but the dispatcher's inline caches must not see a stale
                    // value, which they would if the store were deferred.  Keeping
                    // it explicit is the invariant "state is complete at every
                    // boundary".
                }
                Ok(())
            }
            SprAccess::Dec => {
                // The runtime has to re-arm its virtual timer, so this is not just
                // a state store.
                self.set_misc(PpcMisc::Dec, v);
                self.intr_raw(PpcIntr::SetDec, vec![Self::reg(v)], None);
                Ok(())
            }
            SprAccess::Tb(_) | SprAccess::Ignore => Ok(()),
            SprAccess::Unknown => Err(PpcLowerError::Unimplemented {
                pc: self.pc,
                raw: 0,
                mnemonic: "mtspr:undefined spr",
            }),
        }
    }

    /// `mtsr`/`mfsr`/`mtsrin`/`mfsrin`.  No MMU in this core: writes are reported to
    /// the runtime (an OS switching segments expects *something* to happen), reads
    /// yield 0.
    fn lower_seg_reg(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        let is_store = matches!(kind, MtSr | MtSrin);
        let number = if matches!(kind, MtSrin | MfSrin) {
            self.get_gpr(f.rb())
        } else {
            self.immv(f.sr() as i64, Width::W32)
        };
        if is_store {
            let v = self.get_gpr(f.rt());
            self.intr_raw(PpcIntr::SegReg, vec![Self::reg(number), Self::reg(v)], None);
        } else {
            let z = self.zero();
            self.set_gpr(f.rt(), z);
        }
    }

    fn lower_rfi(&mut self) {
        let srr1 = self.get_misc(PpcMisc::Srr1);
        self.set_misc(PpcMisc::Msr, srr1);
        self.intr_raw(PpcIntr::SprSideEffect, vec![Self::imm(0x100), Self::imm(0)], None);
        let srr0 = self.get_misc(PpcMisc::Srr0);
        let target = self.and(Self::reg(srr0), Self::um(!3), Width::W32);
        self.exit_reg(target);
    }

    // ------------------------------------------------------------------------
    // branches
    // ------------------------------------------------------------------------

    fn lower_b(&mut self, f: PpcFields) {
        let mut target = ((f.li() << 2) as u32) & LO32;
        if !f.aa() {
            target = target.wrapping_add(self.pc as u32) & LO32;
        }
        if f.lk() {
            self.set_link(self.next_pc);
        }
        self.exit_imm(target as u64);
    }

    fn lower_bc(&mut self, f: PpcFields) {
        let mut target = ((f.bd() << 2) as u32) & LO32;
        if !f.aa() {
            target = target.wrapping_add(self.pc as u32) & LO32;
        }
        if f.lk() {
            self.set_link(self.next_pc);
        }
        let taken = self.branch_taken(f);
        let tv = self.immv(target as i64, Width::W64);
        let sel = self.select_exit(taken, tv);
        self.exit_reg(sel);
    }

    fn lower_bclr(&mut self, f: PpcFields) {
        if f.lk() {
            self.set_link(self.next_pc);
        }
        let taken = self.branch_taken(f);
        let lr = self.get_misc(PpcMisc::Lr);
        let tv = self.and(Self::reg(lr), Self::um(!3), Width::W32);
        let sel = self.select_exit(taken, tv);
        self.exit_reg(sel);
    }

    fn lower_bcctr(&mut self, f: PpcFields) {
        if f.lk() {
            self.set_link(self.next_pc);
        }
        let taken = self.branch_taken(f);
        let ctr = self.get_misc(PpcMisc::Ctr);
        let tv = self.and(Self::reg(ctr), Self::um(!3), Width::W32);
        let sel = self.select_exit(taken, tv);
        self.exit_reg(sel);
    }

    /// `LR ← next instruction address`, low two bits clear (a 32-bit
    /// implementation has no use for them, and `mtlr` of a misaligned value must not
    /// produce a misaligned target).
    fn set_link(&mut self, addr: u64) {
        let v = self.immv((addr & !3) as i64, Width::W64);
        self.set_misc(PpcMisc::Lr, v);
    }

    /// PPC's branch condition, exactly as the ISA states it:
    ///
    /// ```text
    ///   if (BO[2] = 0) and CTR != 0 then CTR <- CTR - 1
    ///   ctr_ok  = BO[2] | ((CTR != 0) XOR BO[3])
    ///   cond_ok = BO[0] | (CR[BI] = BO[1])
    ///   taken   = ctr_ok & cond_ok
    /// ```
    ///
    /// With `bo` as the 5-bit field value read by `PpcFields::bo()` (bit 6 is the
    /// most significant, so `BO[k]` is value bit `4-k`):
    ///   • `BO[0]` (0x10) — set: *skip* the CR test
    ///   • `BO[1]` (0x08) — the CR bit must equal this
    ///   • `BO[2]` (0x04) — set: *skip* the CTR test entirely (and the decrement:
    ///     the same bit gates both, which is why `beq` never disturbs CTR)
    ///   • `BO[3]` (0x02) — CTR test polarity: 0 → „CTR != 0", 1 → „CTR == 0"
    ///   • `BO[4]` (0x01) — branch hint, ignored
    ///
    /// Spot-checks the mapping has to satisfy — and the ones Capstone's own
    /// mnemonic labelling confirms for the *same encodings*:
    ///   `bc 12,2` → `beq`  (test CR0.EQ for 1, leave CTR alone)
    ///   `bc 4,2`  → `bne`  (test CR0.EQ for 0, leave CTR alone)
    ///   `bc 4,0`  → `bge`  (test CR0.LT for 0)
    ///   `bc 16,x` → `bdnz` (decrement CTR, then branch while non-zero, ignore CR)
    ///   `bc 20,x` → unconditional (`blr` / `b`)
    ///
    /// One documented ambiguity: the published wordings of the pseudocode disagree
    /// about the *decrement* for the `bdnzt`/`bdnz-` encodings (bo = 8, 24) —
    /// whether BO[2] gates both test and decrement (what the IBM text this was
    /// transcribed from says, and what is implemented here) or BO[0] gates the
    /// decrement separately.  Both wordings agree on bo ∈ {4, 12, 16, 20}, which
    /// is the complete set a 32-bit PowerPC compiler emits, so the difference is
    /// only reachable from hand-written assembly.  Flagged in the notes.
    ///
    /// The CTR write happens here, before the test, and regardless of whether the
    /// branch is taken — a block that reads CTR after a `bc` must see the
    /// decremented value, which the forwarding table gives it by construction.
    fn branch_taken(&mut self, f: PpcFields) -> VReg {
        let bo = f.bo();
        let mut parts: Vec<VReg> = Vec::new();

        // BO[2] = 0 → the CTR is decremented (when non-zero) and then tested.
        if bo & 0x04 == 0 {
            let ctr = self.get_misc(PpcMisc::Ctr);
            let nz_before = self.is_nonzero(ctr, Width::W32);
            // "decrement if CTR != 0" without a branch: subtract the non-zero flag,
            // which is 1 exactly when a decrement is due.
            let after = self.sub(Self::reg(ctr), Self::reg(nz_before), Width::W32);
            self.set_misc(PpcMisc::Ctr, after);
            let nz = self.is_nonzero(after, Width::W32);
            // BO[3] = 1 means "branch when CTR == 0".
            let ctr_ok = if bo & 0x02 != 0 {
                let one = self.immv(1, Width::W32);
                self.xor(Self::reg(nz), Self::reg(one), Width::W32)
            } else {
                nz
            };
            parts.push(ctr_ok);
        }

        // BO[0] = 0 → the CR bit is tested; BO[1] is the value it must equal.
        if bo & 0x10 == 0 {
            let bit = self.cr_bit(f.bi());
            let want = if bo & 0x08 != 0 { 1 } else { 0 };
            let want = self.immv(want, Width::W32);
            parts.push(self.xor(Self::reg(bit), Self::reg(want), Width::W32));
        }

        match parts.len() {
            // Nothing tested at all: unconditional (blr, bctr, and the hint-only
            // encodings the assembler emits for them).
            0 => self.immv(1, Width::W32),
            1 => parts[0],
            _ => {
                let mut acc = parts[0];
                for p in &parts[1..] {
                    acc = self.and(Self::reg(acc), Self::reg(*p), Width::W32);
                }
                acc
            }
        }
    }

    // ------------------------------------------------------------------------
    // list / string memory, cache control
    // ------------------------------------------------------------------------

    fn lower_string_copy(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        let is_store = matches!(kind, Stswi | Stswx);
        let indexed = matches!(kind, Lswx | Stswx);
        let ea = if indexed {
            self.ea_x(f.ra(), f.rb())
        } else {
            // lswi/stswi: EA = (RA|0), with the count as an immediate field.
            self.ea_d(f, 0, false)
        };
        // For the indexed forms the count is (RB & 31) with 0 meaning 32 — a value
        // the helper interprets, so it stays symbolic here.  For the immediate
        // forms the field is passed as-is (0 again meaning 32, same rule).
        let count = if indexed {
            let rb = self.get_gpr(f.rb());
            self.and(Self::reg(rb), Self::um(31), Width::W32)
        } else {
            self.immv(f.nb() as i64, Width::W32)
        };
        let flags = (is_store as i64) | ((indexed as i64) << 1);
        self.intr_raw(
            PpcIntr::StringCopy,
            vec![Self::reg(ea), Self::reg(count), Self::imm(flags), Self::imm(f.rt() as i64)],
            None,
        );
        if !is_store {
            self.fwd.retain(|k, _| !matches!(k, FwdKey::Gpr(_)));
        }
    }

    fn lower_cache(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        let id = match kind {
            Dcbf => PpcIntr::Dcbf,
            Dcbi => PpcIntr::Dcbi,
            Dcbst => PpcIntr::Dcbst,
            Dcbt => PpcIntr::Dcbt,
            Dcbtst => PpcIntr::Dcbtst,
            Dcbz => PpcIntr::Dcbz,
            Icbi => PpcIntr::Icbi,
            _ => PpcIntr::Icbt,
        };
        // All of them take EA = (RA|0) + RB.  The result is discarded, but the
        // address must still be computed, because the runtime hooks it: `dcbz`
        // zeroes guest memory, and `icbi` has to reach the SMC invalidator.
        let ea = self.ea_x(f.ra(), f.rb());
        let hint = match kind {
            Dcbt | Dcbtst => f.dcrn() as i64,
            Icbt => f.icbt_l() as i64,
            _ => 0,
        };
        self.intr_raw(id, vec![Self::reg(ea), Self::imm(hint)], None);
    }

    // ------------------------------------------------------------------------
    // FP
    // ------------------------------------------------------------------------

    fn lower_load_single(&mut self, ea: VReg, frt: u8) {
        let raw = self.load(Self::reg(ea), Width::W32, false);
        let widened = self.intr(PpcIntr::FpWidenF32, vec![Self::reg(raw)]);
        self.set_fpr(frt, widened);
    }

    fn lower_store_single(&mut self, ea: VReg, frt: u8) {
        let v = self.get_fpr(frt);
        let narrow = self.intr(PpcIntr::FpNarrowF32, vec![Self::reg(v)]);
        let low = self.and(Self::reg(narrow), Self::um(LO32), Width::W64);
        self.store(Self::reg(ea), Self::reg(low), Width::W32);
    }

    /// The FP arithmetic group: identical shape, differing in the helper id, the
    /// operand list, and whether the *single*-precision flag comes from the primary
    /// opcode (59 = single, 63 = double).
    fn lower_fp_arith(&mut self, kind: PpcKind, f: PpcFields) -> Result<(), PpcLowerError> {
        use PpcIntr as I;
        use PpcKind::*;
        let prec = Self::imm((f.op() == 59) as i64);
        let fra = self.get_fpr(f.fra());
        let frb = self.get_fpr(f.frb());
        let frc = self.get_fpr(f.frc_ax());
        let r = match kind {
            // fmadd FRT,FRA,FRC,FRB = FRA*FRC + FRB; the helper's contract is
            // [a, b, c, prec] with dst = a*b + c.
            Fmadd => self.intr(I::FpMadd, vec![Self::reg(fra), Self::reg(frc), Self::reg(frb), prec]),
            Fmsub => self.intr(I::FpMsub, vec![Self::reg(fra), Self::reg(frc), Self::reg(frb), prec]),
            Fnmsub => self.intr(I::FpNmsub, vec![Self::reg(fra), Self::reg(frc), Self::reg(frb), prec]),
            Fnmadd => self.intr(I::FpNmadd, vec![Self::reg(fra), Self::reg(frc), Self::reg(frb), prec]),
            // fmul FRT,FRA,FRC — note FRB's field is reserved *here*, and the
            // third operand is FRC; a two-input multiply, not an fma.
            Fmul => self.intr(I::FpMul, vec![Self::reg(fra), Self::reg(frc), prec]),
            Fadd => self.intr(I::FpAdd, vec![Self::reg(fra), Self::reg(frb), prec]),
            Fsub => self.intr(I::FpSub, vec![Self::reg(fra), Self::reg(frb), prec]),
            Fdiv => self.intr(I::FpDiv, vec![Self::reg(fra), Self::reg(frb), prec]),
            Fres => self.intr(I::FpRes, vec![Self::reg(fra), prec]),
            _ => return Err(self.unimplemented_dummy("fp")),
        };
        self.set_fpr(f.frt(), r);
        // Rc=1 on an FP op updates FPSCR (FX/FPRF/…).  Not modelled — see super's
        // module header — so nothing is emitted for it rather than something
        // plausible.
        let _ = f.rc();
        Ok(())
    }

    #[inline]
    fn unimplemented_dummy(&self, name: &'static str) -> PpcLowerError {
        PpcLowerError::Unimplemented { pc: self.pc, raw: 0, mnemonic: name }
    }

    // ------------------------------------------------------------------------
    // paired singles
    // ------------------------------------------------------------------------

    /// Broadcast ps0 of a packed pair into both lanes.
    fn bcast_lo(&mut self, packed: VReg) -> VReg {
        let lo = self.and(Self::reg(packed), Self::um(LO32), Width::W64);
        let hi = self.shift(Self::reg(lo), Self::imm(32), Width::W64, ShiftKind::Left);
        self.or(Self::reg(lo), Self::reg(hi), Width::W64)
    }

    /// Broadcast ps1 of a packed pair into both lanes.
    fn bcast_hi(&mut self, packed: VReg) -> VReg {
        let lo = self.shift(Self::reg(packed), Self::imm(32), Width::W64, ShiftKind::Right);
        let hi = self.shift(Self::reg(lo), Self::imm(32), Width::W64, ShiftKind::Left);
        self.or(Self::reg(lo), Self::reg(hi), Width::W64)
    }

    /// Flip both lanes' sign bits: negating a packed pair without any FP op
    /// (`-x` is a sign-bit flip for every f32 pattern, NaNs and zeros included).
    fn ps_negate(&mut self, packed: VReg) -> VReg {
        let both = (0x8000_0000u32 as u64) | ((0x8000_0000u64) << 32);
        self.xor(Self::reg(packed), Self::imm(both as i64), Width::W64)
    }

    fn vec_add(&mut self, a: VReg, b: VReg) -> VReg {
        let d = self.def();
        self.emit(IrOp::VecAdd {
            dst: d,
            a: Self::reg(a),
            b: Self::reg(b),
            lanes: 2,
            width: Width::W64,
        });
        d
    }

    fn vec_mul(&mut self, a: VReg, b: VReg) -> VReg {
        let d = self.def();
        self.emit(IrOp::VecMul {
            dst: d,
            a: Self::reg(a),
            b: Self::reg(b),
            lanes: 2,
            width: Width::W64,
        });
        d
    }

    /// The one place `VecFma`'s three inputs get wired up, so the sign handling of
    /// the madd/msub/nmadd/nmsub family stays in a single function: `dst = a*b ± c`.
    fn vec_fma(&mut self, a: VReg, b: VReg, c: VReg, subtract_c: bool) -> VReg {
        let d = self.def();
        let c = if subtract_c { self.ps_negate(c) } else { c };
        self.emit(IrOp::VecFma {
            dst: d,
            a: Self::reg(a),
            b: Self::reg(b),
            c: Self::reg(c),
            lanes: 2,
        });
        d
    }

    /// Paired-single arithmetic.  Every op here is either a shared vector op or a
    /// bit-twiddle on the packed pair — which is the design point: the same
    /// `VecAdd`/`VecMul`/`VecFma` the ARM64 frontend uses for NEON.
    ///
    /// `ps_sub` has no `VecSub` in the IR, so it is `A + (-B)`; the msub/nmsub
    /// family is `fma` with a negated addend; `muls0/1` and `madds0/1` broadcast one
    /// scalar lane and reuse the same vector ops.  Only the ops that need real
    /// arithmetic the IR has no shape for (divide, the two estimates, per-lane
    /// select) become helpers.
    fn lower_ps_arith(&mut self, kind: PpcKind, f: PpcFields) -> Result<(), PpcLowerError> {
        use PpcIntr as I;
        use PpcKind::*;
        let fd = f.frt();
        let a = self.get_ps(f.fra());
        let b = self.get_ps(f.frb());

        let r = match kind {
            PsAdd => self.vec_add(a, b),
            PsSub => {
                let nb = self.ps_negate(b);
                self.vec_add(a, nb)
            }
            PsMul => {
                let c = self.get_ps(f.frc_full());
                self.vec_mul(a, c)
            }
            PsMadd => {
                let c = self.get_ps(f.frc_full());
                self.vec_fma(a, c, b, false)
            }
            PsMsub => {
                let c = self.get_ps(f.frc_full());
                self.vec_fma(a, c, b, true)
            }
            PsNmadd => {
                let c = self.get_ps(f.frc_full());
                let t = self.vec_fma(a, c, b, false);
                self.ps_negate(t)
            }
            PsNmsub => {
                let c = self.get_ps(f.frc_full());
                let t = self.vec_fma(a, c, b, true);
                self.ps_negate(t)
            }
            PsMuls0 => {
                let c = self.get_ps(f.frc_full());
                let bc = self.bcast_lo(c);
                self.vec_mul(a, bc)
            }
            PsMuls1 => {
                let c = self.get_ps(f.frc_full());
                let bc = self.bcast_hi(c);
                self.vec_mul(a, bc)
            }
            PsMadds0 => {
                let c = self.get_ps(f.frc_full());
                let bc = self.bcast_lo(c);
                self.vec_fma(a, bc, b, false)
            }
            PsMadds1 => {
                let c = self.get_ps(f.frc_full());
                let bc = self.bcast_hi(c);
                self.vec_fma(a, bc, b, false)
            }
            PsSum0 => {
                // D.ps0 = A.ps0 + B.ps1 ; D.ps1 = C.ps1
                let a_lo = self.and(Self::reg(a), Self::um(LO32), Width::W64);
                let b_hi = self.shift(Self::reg(b), Self::imm(32), Width::W64, ShiftKind::Right);
                let sum = self.vec_add(a_lo, b_hi);
                let sum_lo = self.and(Self::reg(sum), Self::um(LO32), Width::W64);
                let c = self.get_ps(f.frc_full());
                let c_hi = self.and(Self::reg(c), Self::imm(((LO32 as u64) << 32) as i64), Width::W64);
                self.or(Self::reg(sum_lo), Self::reg(c_hi), Width::W64)
            }
            PsSum1 => {
                // D.ps0 = C.ps0 ; D.ps1 = A.ps0 + B.ps1
                let a_lo = self.and(Self::reg(a), Self::um(LO32), Width::W64);
                let b_hi = self.shift(Self::reg(b), Self::imm(32), Width::W64, ShiftKind::Right);
                let sum = self.vec_add(a_lo, b_hi);
                let sum_hi = self.shift(Self::reg(sum), Self::imm(32), Width::W64, ShiftKind::Left);
                let c = self.get_ps(f.frc_full());
                let c_lo = self.and(Self::reg(c), Self::um(LO32), Width::W64);
                self.or(Self::reg(c_lo), Self::reg(sum_hi), Width::W64)
            }
            PsDiv => {
                let c = self.get_ps(f.frc_full());
                // ps_div is `FRA / FRB` (no third operand); `c` is unused and only
                // read because ps_* share one field layout.
                let _ = c;
                self.intr(I::PsDiv, vec![Self::reg(a), Self::reg(b)])
            }
            PsRes => self.intr(I::PsRes, vec![Self::reg(a)]),
            PsRsqrte => self.intr(I::PsRsqrt, vec![Self::reg(a)]),
            PsSel => {
                let c = self.get_ps(f.frc_full());
                self.intr(I::PsSel, vec![Self::reg(a), Self::reg(b), Self::reg(c)])
            }
            _ => return Err(self.unimplemented_dummy("ps")),
        };
        self.set_ps(fd, r);
        Ok(())
    }

    /// `ps_mergeXY`: take one lane from each source and repack.  Pure scalar bit
    /// ops — no vector op, no helper.
    fn lower_ps_merge(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        // ps0 of the result comes from A's half `ha`, ps1 from B's half `hb`.
        let (ha, hb) = match kind {
            PsMerge00 => (0, 0),
            PsMerge01 => (0, 1),
            PsMerge10 => (1, 0),
            _ => (1, 1),
        };
        let a = self.get_ps(f.fra());
        let b = self.get_ps(f.frb());
        // ps0 slot: take A's half and move it down to bits 31:0.
        let lo = if ha == 1 {
            let hi = self.and(Self::reg(a), Self::imm(((LO32 as u64) << 32) as i64), Width::W64);
            self.shift(Self::reg(hi), Self::imm(32), Width::W64, ShiftKind::Right)
        } else {
            self.and(Self::reg(a), Self::um(LO32), Width::W64)
        };
        // ps1 slot: take B's half and move it up to bits 63:32.
        let hi = if hb == 1 {
            self.and(Self::reg(b), Self::imm(((LO32 as u64) << 32) as i64), Width::W64)
        } else {
            self.shift(Self::reg(b), Self::imm(32), Width::W64, ShiftKind::Left)
        };
        let r = self.or(Self::reg(lo), Self::reg(hi), Width::W64);
        self.set_ps(f.frt(), r);
    }

    /// `ps_mr`/`ps_neg`/`ps_abs`/`ps_nabs`: a copy, or bit ops on the pair.
    fn lower_ps_unary(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        let b = self.get_ps(f.frb());
        let r = match kind {
            PsMr => b,
            PsNeg => self.ps_negate(b),
            PsAbs => {
                let clear = !((1u64 << 31) | (1u64 << 63));
                self.and(Self::reg(b), Self::imm(clear as i64), Width::W64)
            }
            _ => {
                let set = ((1u64 << 31) | (1u64 << 63)) as i64;
                self.or(Self::reg(b), Self::imm(set), Width::W64)
            }
        };
        self.set_ps(f.frt(), r);
    }

    fn lower_ps_cmp(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        let lane = match kind {
            PsCmpu1 | PsCmpo1 => 1,
            _ => 0,
        };
        let ordered = match kind {
            PsCmpo0 | PsCmpo1 => 1,
            _ => 0,
        };
        let a = self.get_ps(f.fra());
        let b = self.get_ps(f.frb());
        let bits = self.intr(
            PpcIntr::PsCmp,
            vec![Self::reg(a), Self::reg(b), Self::imm(lane), Self::imm(ordered)],
        );
        self.set_crf(f.bf(), bits);
    }

    /// Quantized paired-single load/store.  The *address* stays IR (so the
    /// redundant-load pass and the emitter's addressing folding still see it), while
    /// the quantization itself is the helper's: it needs `GQR[i]`, which is guest
    /// state the IR cannot name.
    fn lower_psq(&mut self, kind: PpcKind, f: PpcFields) {
        use PpcKind::*;
        let (indexed, is_store, update) = match kind {
            PsqL => (false, false, false),
            PsqLu => (false, false, true),
            PsqSt => (false, true, false),
            PsqStu => (false, true, true),
            PsqLx => (true, false, false),
            PsqLux => (true, false, true),
            PsqStx => (true, true, false),
            _ => (true, true, true),
        };
        let (w, gq) = if indexed {
            (f.psq_w_x() as i64, f.psq_gq_x() as i64)
        } else {
            (f.psq_w_d() as i64, f.psq_gq_d() as i64)
        };
        let ea = if indexed {
            self.ea_x(f.ra(), f.rb())
        } else {
            self.ea_d(f, f.psq_d(), update)
        };
        if is_store {
            let v = self.get_ps(f.frt());
            self.intr_raw(
                PpcIntr::PsqStore,
                vec![Self::reg(ea), Self::reg(v), Self::imm(w), Self::imm(gq)],
                None,
            );
        } else {
            let v = self.intr(PpcIntr::PsqLoad, vec![Self::reg(ea), Self::imm(w), Self::imm(gq)]);
            self.set_ps(f.frt(), v);
        }
        if update {
            self.set_gpr(f.ra(), ea);
        }
    }
}

// =============================================================================
// operand-shape helpers
// =============================================================================

/// `(width, sign_extend)` for the D-form integer loads.
fn int_load_shape(kind: PpcKind) -> (Width, bool) {
    use PpcKind::*;
    match kind {
        Lwz | Lwzu => (Width::W32, false),
        Lbz | Lbzu => (Width::W8, false),
        Lhz | Lhzu => (Width::W16, false),
        _ => (Width::W16, true), // lha / lhau
    }
}

/// `(width, sign_extend)` for the indexed integer loads, including the update forms.
fn int_load_shape_indexed(kind: PpcKind) -> (Width, bool) {
    use PpcKind::*;
    match kind {
        Lwzx | Lwzux => (Width::W32, false),
        Lbzx | Lbzux => (Width::W8, false),
        Lhzx | Lhzux => (Width::W16, false),
        _ => (Width::W16, true), // lhax / lhaux
    }
}

fn int_store_width(kind: PpcKind) -> Width {
    use PpcKind::*;
    match kind {
        Stw | Stwu => Width::W32,
        Stb | Stbu => Width::W8,
        _ => Width::W16,
    }
}

fn int_store_width_indexed(kind: PpcKind) -> Width {
    use PpcKind::*;
    match kind {
        Stwx | Stwux => Width::W32,
        Stbx | Stbux => Width::W8,
        _ => Width::W16,
    }
}

/// The add/subtract family's operand choices, as data, so the single lowering body
/// stays readable and every member is visibly covered.  `subfic`/`addic` build a
/// `Sum` inline at their dispatch arm instead, because their second operand is an
/// `SI` immediate rather than a register.
#[derive(Clone, Copy)]
enum SumKind {
    Ra,
    Rb,
    Zero,
    Const(u32),
}

#[derive(Clone, Copy)]
enum SumSecond {
    Ra,
    Rb,
    Imm(u32),
    None,
}

#[derive(Clone, Copy)]
enum CarryIn {
    Zero,
    One,
    /// XER.CA as it stood *before* this instruction (the ISA reads the old value).
    Xer,
}

/// One row of the add/subtract family.
#[derive(Clone, Copy)]
struct Sum {
    /// First addend.
    x: SumKind,
    /// XOR the second addend with `0xFFFFFFFF` before adding (`~`).
    inv: bool,
    /// Second addend.
    y: SumSecond,
    /// Extra 0/1 addend.
    ca: CarryIn,
    /// Whether the instruction writes XER.CA.
    ca_out: bool,
    /// Whether the `o` forms can overflow for this instruction.
    ov: bool,
}

