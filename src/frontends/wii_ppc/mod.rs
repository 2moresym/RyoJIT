//! # Wii frontend — PowerPC "Broadway" (750CL-derived, big-endian)
//!
//! This module is the *only* PPC-specific code in the project apart from the GPU
//! translators.  It implements [`Frontend`](crate::frontend::Frontend): decode
//! one fixed-width guest word, lower it into the universal IR, say whether it
//! terminates a block, and report the guest endianness.  Everything downstream
//! (passes → regalloc → C++ emitter → cache → chaining) treats the result
//! exactly like Switch/ARM64 IR.
//!
//! ## The three contracts this module had to settle
//!
//! The shared IR is deliberately small: 9 scalar ALU ops, 3 vector ops,
//! `Load`/`Store`, 4 control-flow ops, 2 flag ops and `Intrinsic`.  It has no
//! `Mov`, no `Const`, no FP arithmetic, no way to name a *guest register*, and
//! no address-space concept.  Filling in the opcode table therefore required
//! three decisions, each recorded here because the C++ side depends on them.
//! None of them changed a single byte of `CIrOp`, of `CpuState`'s existing
//! fields, or of the spill semantics.
//!
//! ### 1. Guest register access = `Intrinsic`, never `Load`/`Store`
//!
//! `Load`/`Store` addresses are **guest effective addresses** (the C++ emitter
//! adds the reserved guest-mapping base).  There is no way to express "read
//! `CpuState::gpr[7]`" as an `IrOp::Load`, and guest state must not be sneaked
//! into the guest address space.  So every register-file touch is an
//! `Intrinsic` with an id from [`intrinsics::PpcIntr`] — the escape hatch the
//! architecture reserves for precisely this.  The two hot ones (`GET_GPR` /
//! `SET_GPR`) cost the emitter one `mov` each, so nothing is lost versus
//! first-class ops, and the *forwarding table* in [`lower::Ctx`] makes an
//! unchanged guest register be read from state exactly once per block: in
//! `lwz r3,.. / add r3,r3,r4 / stw r3,..` the value stays in a host register and
//! only the load and store touch memory.
//!
//! `SideEffects` on those intrinsics describes **guest memory** only, because
//! that is the one thing the passes consult
//! (`passes::redundant_load_elimination` clears its cached memory facts on
//! `WritesMem`/`ReadsAndWritesMem`/`Volatile`).  Register-file access is
//! therefore marked `Pure`.  That is both sound today (nothing in this pipeline
//! reorders or CSEs intrinsics) and *load-bearing*: marking them `WritesMem`
//! would flush the redundant-load cache on every guest register write and kill
//! the pass on ordinary load/use/store code.  If a future pass ever treats
//! `Pure` as "safe to CSE/DCE", `PpcIntr::effects` is the single place to split
//! out a "touches guest state" class, and `tests::effects_table_is_stable`
//! pins today's table so the change cannot be accidental.
//!
//! ### 2. Block exits are data, not `Branch` targets
//!
//! `CIrOp` has no field for a *target guest PC*; `target_block` is a `BlockId`
//! inside the current translation unit, and a PPC block contains no other
//! blocks.  So the frontend terminates every block with an
//! `IrOp::IndirectBranch` whose operand is either
//!   • `Imm(pc)` — a statically known exit (unconditional branch, or a
//!     conditional branch whose two exits were folded into one value, below).
//!     The emitter treats a *constant*-target `IndirectBranch` as a **direct**
//!     branch: it emits a patchable jump and records a `CPatchSite` carrying
//!     that PC, so block chaining works exactly as designed.
//!   • `Reg(v)` — a computed exit (`blr`, `bctr`, …), which is what the
//!     dispatcher's inline cache resolves later.
//!
//! `Branch`/`Call`/`Return` are never emitted by this frontend: `Branch` cannot
//! carry both exit PCs, and `Call`/`Return` would imply a host stack-frame
//! convention the block prologue/epilogue does not have.  Guest `bl` is just a
//! write of `CpuState::lr` followed by an exit — the guest's own ABI stays
//! inside guest state, which is also what keeps chaining and SMC invalidation
//! identical for both consoles.
//!
//! Conditional branches fold into one exit value with pure ALU ops:
//!
//! ```text
//!   taken = (cond != 0)                       // 1 or 0, from lazy flags
//!   mask  = 0 - taken                         // 0 or !0
//!   exit  = fallthrough + (mask & (target - fallthrough))
//! ```
//!
//! ~5 cheap ops, no new IR shape — at the cost of the taken/not-taken edge not
//! being two separate patch sites.  The C++ notes in [`intrinsics`] describe the
//! idiom match that recovers a real `jcc` + two patch sites if a profile ever
//! says it is worth it.
//!
//! ### 3. `CpuState::fpr[n]` is a raw 64-bit container; PS fields are its halves
//!
//! Scalar FP ops interpret the container as an IEEE-754 double, so `lfd`/`stfd`
//! are plain 64-bit `Load`/`Store`s and `fneg`/`fabs`/`fnabs` are one scalar
//! `Xor`/`And` against bit 63 (no intrinsic, no FP helper at all).
//! Paired singles live in the low and high 32-bit halves of the pair, and
//! `GetPsPair`/`SetPsPair` pack them into one 64-bit VReg so that
//! `ps_add`/`ps_mul`/`ps_madds0` lower to the shared
//! `VecAdd`/`VecMul`/`VecFma { lanes: 2, width: W64 }` — the vector IR the
//! architecture was designed around, and the reason no PS-specific vector op had
//! to be added.  See [`PpcIntr::GetPsPair`] for the layout and the one open
//! question (the 750CL splits a pair over two FPRs, Gekko packs both halves into
//! one); it is a single `const`, so the answer only has to be found once.
//!
//! ## Not modelled yet (explicit, never silent)
//!
//! * No VMX/AltiVec — Broadway has none.  Encodings in the VMX opcode space
//!   outside the paired-single set decode as illegal.
//! * `HID0[PSQM]` gating of the *scalar* single-precision ops (`fadds`, `fsubs`,
//!   `fmuls`, `fdivs`, `fmadds`, `fmsubs`, `fnmadds`, `fnmsubs`, `fres`, `frsp`
//!   and `lfs`/`stfs`) is not modelled: they always use the 750CL scalar
//!   semantics (round via double).  The dedicated `ps_*` instructions always
//!   work.
//! * Guest exceptions: `tw`/`twi` and undefined opcodes lower to a `TRAP`
//!   intrinsic, so the guest takes its own 0x200/0x700 exception instead of the
//!   JIT dying.  The guest MMU, the segment registers, `mtmsr` side effects and
//!   the *actions* behind the cache instructions belong to the CPU/emulation
//!   layer: the IR stores/loads those SPRs and calls the matching intrinsic, it
//!   does not implement them.
//! * FPSCR exception bits (FX/FEX/VX/OX/UX/ZX/XX/…) are stored but not
//!   recomputed from FP results.  `mtfsf`/`mtfsfi`/`mtfsb0`/`mtfsb1`/`mffs` are
//!   correct as *state moves*; `fcmpu`/`fcmpo`/`mcrfs` take their results from
//!   the C++ helper (see [`intrinsics`]).  `fres`/`frsqrte` return whatever the
//!   host `rcpps`/`rsqrtps` produce — bit-exactness against Broadway's
//!   implementation-defined estimates is a known accuracy gap for any title that
//!   depends on it.

mod decode;
mod fields;
mod intrinsics;
mod lower;

#[cfg(test)]
mod tests;

pub use self::decode::PpcKind;
pub use self::fields::PpcFields;
pub use self::intrinsics::{intrinsic_name, PpcIntr, PpcMisc};
pub use self::lower::lower_insn;

use crate::frontend::{Frontend, RegisterLayout};
use crate::ir::{BlockId, Endian, IrBlock, IrBuilder, IrOp, VOperand};

/// Decoded guest instruction, as handed around through the [`Frontend`] trait.
///
/// `pc` is part of the decoded form because `lower_to_ir` has no other way to
/// know where the instruction lives, and `bl`/`bcl.`/`bcctr` need it to compute
/// the link value while a conditional branch needs the fall-through PC.
#[derive(Clone, Copy, Debug)]
pub struct DecodedPpc {
    /// The 32-bit instruction word, exactly as fetched from guest memory.
    pub raw: u32,
    /// Primary opcode, bits 0:5.
    pub opcode: u8,
    /// Guest address this word was fetched from.
    pub pc: u64,
    /// Resolved instruction (see [`decode`]).  `PpcKind::Illegal` is a *decoded
    /// result*, not a panic: it lowers to a trap intrinsic so the guest takes
    /// its own illegal-instruction exception instead of taking the JIT down.
    pub kind: PpcKind,
    /// Mnemonic of the resolved instruction, produced by the same table arm as
    /// `kind` — so a diagnostic, a test golden and the decode can never
    /// disagree with each other.
    pub name: &'static str,
    /// Raw field views.
    pub f: PpcFields,
}

/// One translated guest basic block, produced by [`WiiFrontend::translate_unit`].
pub struct PpcUnit {
    pub block: IrBlock,
    /// Guest PC of the first instruction lowered.
    pub start_pc: u64,
    /// Guest PC immediately after the last instruction lowered.  When the loop
    /// stopped at the instruction cap rather than on a branch, this is also the
    /// target of the synthesised exit.
    pub next_pc: u64,
    /// Number of guest instructions lowered.
    pub insn_count: usize,
    /// True when the last instruction was a block terminator.
    pub terminated: bool,
    /// Set when a translation error (an encoding that cannot be represented at
    /// all) occurred.  The block is then *truncated* at that instruction rather
    /// than discarded, so a fallback interpreter can resume exactly where the
    /// JIT gave up instead of re-executing the whole block.
    pub error: Option<lower::PpcLowerError>,
}

/// The Wii frontend.  A unit struct on purpose: `GuestSystem { frontend:
/// WiiFrontend, .. }` is built once per thread and the translator must stay
/// stateless, so there is nowhere for cross-block state to hide (and no lock
/// needed).
#[derive(Default, Clone, Copy, Debug)]
pub struct WiiFrontend;

/// Instruction-count ceiling for one translated block.  Kept here, and used by
/// [`WiiFrontend::translate_unit`], so the frontend's tests exercise the same
/// cut-off the dispatcher applies; `runtime::translate_block_at` uses the same
/// value (64 also matches the small block size the regalloc module's spill
/// budget assumes).
pub const MAX_BLOCK_INSNS: usize = 64;

/// PPC is fixed-width.
pub const GUEST_INSN_BYTES: usize = 4;

impl WiiFrontend {
    #[inline]
    pub fn new() -> Self {
        WiiFrontend
    }

    /// Identify one word without lowering it — the debugging/tracing entry point
    /// (`ryojit demo`, a disassembly-style log, a breakpoint UI).  It is the same
    /// `resolve` the frontend itself uses, never a second table.
    #[inline]
    pub fn identify(&self, word: u32) -> (PpcKind, &'static str) {
        let insn = decode::decode_raw(word, 0);
        (insn.kind, insn.name)
    }

    /// Decode + lower `mem` (guest bytes whose first byte is at `start_pc`)
    /// until a terminator or the instruction cap, then close the block.
    ///
    /// This is the same step-1 loop `runtime::translate_block_at` runs, exposed
    /// for tests and for the future fallback path.  It mirrors that loop's
    /// 6-line skeleton; the semantics are shared because both call `decode`,
    /// `lower_insn` and `is_block_terminator`.  Once guest memory access
    /// (todo #3) lands, the dispatcher can call this directly with a slice.
    pub fn translate_unit(&self, mem: &[u8], start_pc: u64) -> PpcUnit {
        let mut builder = IrBuilder::new();
        let block: BlockId = builder.new_block(start_pc);

        let mut cursor = 0usize;
        let mut insn_count = 0usize;
        let mut terminated = false;
        let mut error = None;

        while insn_count < MAX_BLOCK_INSNS && cursor + GUEST_INSN_BYTES <= mem.len() {
            let raw = u32::from_be_bytes([
                mem[cursor],
                mem[cursor + 1],
                mem[cursor + 2],
                mem[cursor + 3],
            ]);
            let insn = decode::decode_raw(raw, start_pc.wrapping_add(cursor as u64));
            insn_count += 1;
            cursor += GUEST_INSN_BYTES;

            match lower_insn(&insn, &mut builder, block) {
                Ok(()) => {}
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }

            terminated = insn.kind.is_block_terminator();
            if terminated {
                break;
            }
        }

        let mut block_obj = builder
            .blocks
            .pop()
            .expect("new_block always pushes exactly one block");

        // Every block must end in a terminator, or the emitter runs off the end
        // of it (the same rule the dispatcher applies to instruction-capped
        // blocks).  A block that ended in a trap-with-error has no valid exit
        // either, but the caller is not allowed to execute it at all, so the
        // synthesised fall-through exit is the right shape there too.
        if !crate::ir::is_terminator_op(block_obj.ops.last()) {
            block_obj.ops.push(IrOp::IndirectBranch {
                target: VOperand::Imm(start_pc.wrapping_add(cursor as u64) as i64),
            });
            terminated = true;
        }

        PpcUnit {
            block: block_obj,
            start_pc,
            next_pc: start_pc.wrapping_add(cursor as u64),
            insn_count,
            terminated,
            error,
        }
    }
}

impl Frontend for WiiFrontend {
    type DecodedInsn = DecodedPpc;

    /// Field extraction + opcode resolution.  See [`decode`](self::decode).
    fn decode(&self, bytes: &[u8], pc: u64) -> (DecodedPpc, usize) {
        let raw = if bytes.len() >= 4 {
            u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            // Short read.  Zero-fill: the all-zero word is a reserved primary
            // opcode, which lowers to a trap instead of mis-executing garbage.
            let mut buf = [0u8; 4];
            let n = bytes.len().min(4);
            buf[..n].copy_from_slice(&bytes[..n]);
            u32::from_be_bytes(buf)
        };
        (decode::decode_raw(raw, pc), GUEST_INSN_BYTES)
    }

    fn lower_to_ir(&self, insn: &DecodedPpc, ir: &mut IrBuilder, block: BlockId) {
        // `lower_insn` reports the one case where a block has to be truncated
        // (an encoding that cannot be represented at all).  The trait has no
        // error channel, so this follows the dispatcher's existing contract for
        // "this block cannot be translated": report loudly, and let whatever
        // owns the session decide between fatal dialog and interpreter
        // fallback.  `translate_unit` above is the path that gets the error
        // value instead.
        if let Err(e) = lower_insn(insn, ir, block) {
            panic!("RyoJIT PPC lowering failed: {e}");
        }
    }

    fn register_file_layout(&self) -> RegisterLayout {
        // Broadway: 32 GPRs, 32 FPRs, and 32 vector slots that exist in the
        // state layout but have no VMX instructions behind them (see the module
        // header).  Paired-single values live inside the FPRs, not in `vec`.
        RegisterLayout {
            gpr_count: 32,
            fpr_count: 32,
            vector_reg_count: 32,
        }
    }

    /// Precise rather than primary-opcode based: the group-19 arms need the
    /// extended opcode, so this asks the resolved kind.
    fn is_block_terminator(&self, insn: &DecodedPpc) -> bool {
        insn.kind.is_block_terminator()
    }

    fn endianness(&self) -> Endian {
        Endian::Big
    }
}
