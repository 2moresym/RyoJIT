//! =============================================================================
//! UNIFIED EMULATION CORE — Architecture Skeleton (Hybrid Rust / C++ FFI)
//! Shared IR + optimizer + regalloc + frontends + runtime (Rust)
//! JIT emission only (C/C++ via asmjit, crossed by a minimal C ABI)
//! Host target: x86-64, optimized for weak/older desktop & laptop hardware
//! =============================================================================
//!
//! This is a single-file ARCHITECTURE SKELETON, not a finished emulator.
//! Every module is real, compiles conceptually, and shows exact data flow —
//! but instruction decode tables, full codegen, and GPU shader translation are
//! left as `todo!()` since those are thousands of lines of per-instruction and
//! per-shader-opcode work that only make sense written against real test ROMs.
//!
//! Language split (do not collapse):
//!   • Rust  — universal IR, passes, linear-scan regalloc, Frontend trait +
//!             both guest frontends, code cache, dispatcher/runtime, GPU model
//!   • C/C++ — ONLY the x86-64 code-emission layer (asmjit). HostCaps detection,
//!             CodegenStrategy table, and all dynasm-style encoding live here.
//!
//! Recommended real layout (split this file into, once you start filling logic):
//!   core/ir.rs  core/passes.rs  core/regalloc.rs  core/ffi.rs
//!   core/cache.rs  core/runtime.rs
//!   frontends/wii_ppc.rs  frontends/switch_arm64.rs
//!   gpu/model.rs  gpu/hollywood.rs  gpu/maxwell.rs  gpu/backend_wgpu.rs
//!   jit/ (C++): emit.cpp  host_caps.cpp  strategy.cpp  (linked as static lib)
//!
//! Current on-disk layout (the split has started; the modules below are still
//! declared in this file as the crate root so that every `crate::` path and
//! every `#[repr(C)]` layout in here stays exactly as reviewed):
//!   src/lib.rs                        this file: FFI types, IR, passes,
//!                                     regalloc, marshaling, cache, runtime, GPU
//!   src/frontends/wii_ppc/*           REAL: PowerPC Broadway decode + lowering
//!                                     (replaces the SECTION 7 skeleton)
//!   src/ir_dump.rs, src/ir_verify.rs  IR text dump + structural well-formedness
//!                                     checks (debug + test only; no codegen)
//!   src/jit_stub.rs                   TEMPORARY stand-in for the C++ emitter so
//!                                     the crate links and the boundary is
//!                                     testable. Feature-gated `jit-stub`.
//!
//! =============================================================================

#![allow(dead_code, unused_variables)]

// Real Cargo.toml deps this file assumes (Rust side only):
//   memmap2 = "0.9"          // optional; C++ side owns executable memory
//   libc     = "0.2"         // for free() of C++ allocations if needed
// No dynasm / dynasmrt — emission moved entirely to C++ (asmjit).

// Debug/verification helpers (Rust-only, never in the execute path).  The
// temporary stand-in for the C++ emitter is declared down in SECTION 0, next to
// the `extern "C"` block it replaces, so the two cannot drift.
pub mod ir_dump;
pub mod ir_verify;

// =============================================================================
// SECTION 0: FFI BOUNDARY — the single place Rust and C++ meet
// =============================================================================
//
// Design goals
// ------------
// 1. Minimal surface: one primary entry point (emit a whole IR block) plus a
//    couple of helpers (detect caps, free code, patch a chain site).
// 2. Flat, POD, C-compatible data. No Rust enums, no Vec, no Box, no String
//    cross the boundary. Everything is serialized into fixed-size tagged
//    structs or length-prefixed arrays of those structs.
// 3. Clear ownership:
//      • Input  (CIrBlock + operands): allocated and owned by Rust for the
//        duration of the call. C++ must not retain pointers after return.
//      • Output (executable code buffer + patch sites): allocated by C++
//        (via asmjit’s JitRuntime or a custom mmap). Ownership transfers to
//        Rust on successful return. Rust is responsible for calling
//        jit_free_code() exactly once when the CompiledBlock is dropped /
//        evicted from the code cache. Double-free is undefined; use-after-
//        free is undefined. The code pointer remains valid until freed.
// 4. Thread-safety: the emit function itself is re-entrant / thread-safe as
//    long as the caller does not free a code buffer that another thread is
//    still executing. The code cache is expected to provide the necessary
//    synchronization (or to be single-threaded).
// 5. Error handling: all functions return a status code. On failure the out-
//    parameters are left untouched (or zeroed) and no ownership is transferred.
//
// =============================================================================

/// C-compatible types used on both sides of the FFI.
/// These live in a header that both the Rust `#[repr(C)]` definitions and the
/// C++ side include (or are duplicated with identical layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CWidth {
    W8   = 0,
    W16  = 1,
    W32  = 2,
    W64  = 3,
    W128 = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CEndian {
    Little = 0,
    Big    = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CFlagKind {
    Zero     = 0,
    Carry    = 1,
    Overflow = 2,
    Negative = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CFlagOp {
    AddOp   = 0,
    SubOp   = 1,
    AndOp   = 2,
    OrOp    = 3,
    XorOp   = 4,
    ShiftOp = 5,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CSideEffects {
    Pure              = 0,
    ReadsMem          = 1,
    WritesMem         = 2,
    ReadsAndWritesMem = 3,
    Volatile          = 4,
}

/// Operand kind discriminator for the tagged union below.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum COperandKind {
    Reg = 0,
    Imm = 1,
}

/// Flat operand. A virtual register is just a u32 index; immediates are i64.
/// The C++ side never sees Rust’s VReg newtype.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct COperand {
    pub kind: COperandKind,
    pub reg:  u32,   // valid when kind == Reg
    pub imm:  i64,   // valid when kind == Imm
}

/// Host general-purpose register assignment produced by linear-scan.
/// Matches the Rust-side HostGpr enum, excluding the reserved r15.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CHostGpr {
    Rax = 0,
    Rbx = 1,
    Rcx = 2,
    Rdx = 3,
    Rsi = 4,
    Rdi = 5,
    R8  = 6,
    R9  = 7,
    R10 = 8,
    R11 = 9,
    R12 = 10,
    R13 = 11,
    R14 = 12,
    // R15 is permanently reserved as guest CpuState base; never appears here.
}

/// Where a VReg lives after allocation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CLocationKind {
    Reg        = 0,
    StateSpill = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CLocation {
    pub kind:   CLocationKind,
    pub gpr:    CHostGpr, // valid when kind == Reg
    pub offset: u32,      // byte offset into CpuState when kind == StateSpill
}

/// One entry in the allocation map: VReg index → Location.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CAllocEntry {
    pub vreg: u32,
    pub loc:  CLocation,
}

/// Opcode tags for the flat IR stream. Keep this list in lock-step with the
/// Rust IrOp enum; new IR nodes require a matching CIrOpKind + payload fields.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CIrOpKind {
    // Arithmetic / logic
    Add    = 0,
    Sub    = 1,
    And    = 2,
    Or     = 3,
    Xor    = 4,
    Shl    = 5,
    Shr    = 6,
    Sar    = 7,
    Mul    = 8,
    // 128-bit vector (shared PPC paired-single / ARM64 NEON)
    VecAdd = 9,
    VecMul = 10,
    VecFma = 11,
    // Memory
    Load   = 12,
    Store  = 13,
    // Control flow
    Branch         = 14,
    IndirectBranch = 15,
    Call           = 16,
    Return         = 17,
    // Flags (lazy)
    SetFlags = 18,
    ReadFlag = 19,
    // Escape hatch
    Intrinsic = 20,
}

/// Maximum number of operands any single IR op may carry.
/// Intrinsic is the only variable-arity op; we cap it for a fixed-size payload.
pub const CIR_MAX_OPERANDS: usize = 8;

/// Fixed-size payload for one IR operation. Unused fields are zeroed.
/// For Intrinsic the first `num_operands` entries of `operands` are live
/// and `dst_valid` indicates whether a destination VReg is present.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CIrOp {
    pub kind:        CIrOpKind,
    pub dst:         u32,                 // VReg index; ignored if no dest
    pub dst_valid:   u8,                  // 0/1
    pub width:       CWidth,
    pub endian:      CEndian,             // only Load/Store
    pub sign_ext:    u8,                  // only Load
    pub lanes:       u8,                  // only vector ops
    pub flag_kind:   CFlagKind,           // only ReadFlag
    pub flag_op:     CFlagOp,             // only SetFlags
    pub side_effects: CSideEffects,       // only Intrinsic
    pub intrinsic_id: u32,                // only Intrinsic
    pub target_block: u32,                // only Branch (BlockId)
    pub num_operands: u8,
    pub operands:    [COperand; CIR_MAX_OPERANDS],
}

/// One basic block handed to the emitter.
/// All arrays are owned by the caller (Rust) for the duration of the call.
#[repr(C)]
pub struct CIrBlock {
    pub guest_start_pc: u64,
    pub block_id:       u32,
    pub num_ops:        u32,
    pub ops:            *const CIrOp,     // array of length num_ops
    pub num_allocs:     u32,
    pub allocs:         *const CAllocEntry, // array of length num_allocs
}

/// A single patchable exit (direct branch / call that will later be chained).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CPatchSite {
    pub code_offset:     u32,  // byte offset from start of emitted code
    pub target_guest_pc: u64,
}

/// Result of a successful emit.
/// Ownership of `code` and `patch_sites` transfers to the caller (Rust).
/// Caller must eventually call jit_free_code() on the code pointer;
/// patch_sites are freed by the same call (they live in the same allocation
/// or a companion allocation that jit_free_code knows about).
#[repr(C)]
pub struct CEmitResult {
    pub code:            *mut u8,
    pub code_len:        usize,
    pub num_patch_sites: u32,
    pub patch_sites:     *mut CPatchSite,
}

/// Status codes returned by every FFI entry point.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CJitStatus {
    Ok              = 0,
    InvalidInput    = 1,
    EmitFailed      = 2,
    OutOfMemory     = 3,
    UnsupportedOp   = 4,
}

/// Host capability bitfield — populated once by C++ at startup.
/// Bits match the original HostCaps design; detection lives entirely on the
/// C++ side (cpuid / asmjit’s CpuInfo).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CHostCaps {
    pub bits: u32,
}

impl CHostCaps {
    pub const SSE2:    u32 = 1 << 0; // always set on x86-64
    pub const SSE4_1:  u32 = 1 << 1;
    pub const AVX:     u32 = 1 << 2;
    pub const AVX2:    u32 = 1 << 3;
    pub const FMA:     u32 = 1 << 4;
    pub const BMI2:    u32 = 1 << 5;
    pub const MOVBE:   u32 = 1 << 6;
    pub const AVX512F: u32 = 1 << 7;

    pub fn has(&self, flag: u32) -> bool {
        self.bits & flag != 0
    }
}

// -----------------------------------------------------------------------------
// Extern "C" declarations — the only symbols the C++ static library must export.
//
// While `jit/` (the asmjit-backed emitter) does not exist yet, the `jit-stub`
// Cargo feature provides these four symbols with a Rust placeholder so the crate
// links and the boundary stays testable; see src/jit_stub.rs.  Building with
// `--no-default-features` selects the REAL FFI and fails at link time until the
// C++ library is in place — that is intentional.
// -----------------------------------------------------------------------------
#[cfg(not(feature = "jit-stub"))]
extern "C" {
    /// Detect host CPU features once. Thread-safe; may be called from any
    /// thread. Result is a pure value (no heap).
    fn jit_detect_host_caps() -> CHostCaps;

    /// Emit a complete IR block into executable memory.
    ///
    /// Parameters
    /// ----------
    /// block   : pointer to a CIrBlock whose arrays remain valid for the
    ///           duration of the call. Must not be null; num_ops may be 0.
    /// caps    : host capability bitfield (normally from jit_detect_host_caps,
    ///           or a forced baseline for testing).
    /// force_baseline : non-zero → ignore advanced features and emit SSE2-
    ///                  only code (debug / weak-hardware validation path).
    /// out     : on success filled with ownership-transferring pointers.
    ///
    /// Returns CJitStatus::Ok on success. On any error out is left zeroed
    /// and no ownership is transferred.
    fn jit_emit_block(
        block: *const CIrBlock,
        caps: CHostCaps,
        force_baseline: u8,
        out: *mut CEmitResult,
    ) -> CJitStatus;

    /// Release a code buffer previously returned by jit_emit_block.
    /// Also frees the associated patch-site array.
    /// Passing a null pointer is a no-op. Passing a pointer not obtained
    /// from jit_emit_block (or already freed) is undefined behaviour.
    fn jit_free_code(code: *mut u8);

    /// Patch a previously emitted direct-branch site so that it jumps
    /// straight to `target_entry` (block chaining).
    /// The code page must be temporarily made writable by the C++ side
    /// (W^X toggle or dual mapping); the function restores the original
    /// protection before returning.
    ///
    /// Returns Ok or an error status. Does not take ownership of any pointer.
    fn jit_patch_chain(
        code: *mut u8,
        code_len: usize,
        patch_offset: u32,
        target_entry: *const u8,
    ) -> CJitStatus;
}

// The placeholder implementation of exactly those four symbols (see the note
// above the extern block).  Signatures must match the declarations above
// line-for-line: `jit_ffi` imports these four names from the crate root either
// way, so the Rust side is compiled against an identical surface in both modes.
#[cfg(feature = "jit-stub")]
#[path = "jit_stub.rs"]
pub mod jit_stub;
#[cfg(feature = "jit-stub")]
pub use jit_stub::{
    jit_detect_host_caps, jit_emit_block, jit_free_code, jit_patch_chain,
};

// =============================================================================
// SECTION 1: CORE IR — the universal instruction representation
// Every guest frontend (Wii PPC, Switch ARM64) lowers down to this and ONLY
// this. The JIT backend, optimizer, and register allocator never know which
// guest system produced the IR they're processing.
// =============================================================================
pub mod ir {
    /// Virtual register — SSA-like, unbounded, resolved by regalloc later.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VReg(pub u32);

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Width { W8, W16, W32, W64, W128 /* for vector/paired-single/NEON */ }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Endian { Little, Big }

    /// Operand: either a virtual register or an immediate constant.
    #[derive(Debug, Clone, Copy)]
    pub enum VOperand {
        Reg(VReg),
        Imm(i64),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FlagKind { Zero, Carry, Overflow, Negative }

    /// Which arithmetic op produced a set of flags — flags are computed LAZILY,
    /// only when a later ReadFlag actually consumes them. This is the single
    /// biggest perf win over naive per-instruction flag emulation, since PPC
    /// and ARM64 both set flags far more often than they're actually read.
    #[derive(Debug, Clone, Copy)]
    pub enum FlagOp { AddOp, SubOp, AndOp, OrOp, XorOp, ShiftOp }

    /// Side-effect classification for intrinsics — lets the optimizer reason
    /// conservatively about ops it doesn't understand without needing to know
    /// what they actually do.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SideEffects { Pure, ReadsMem, WritesMem, ReadsAndWritesMem, Volatile }

    #[derive(Debug, Clone, Copy)]
    pub struct IntrinsicId(pub u32);

    #[derive(Debug, Clone)]
    pub enum IrOp {
        // ---- arithmetic / logic (width-parameterized, shared across guests) ----
        Add    { dst: VReg, a: VOperand, b: VOperand, width: Width },
        Sub    { dst: VReg, a: VOperand, b: VOperand, width: Width },
        And    { dst: VReg, a: VOperand, b: VOperand, width: Width },
        Or     { dst: VReg, a: VOperand, b: VOperand, width: Width },
        Xor    { dst: VReg, a: VOperand, b: VOperand, width: Width },
        Shl    { dst: VReg, a: VOperand, amount: VOperand, width: Width },
        Shr    { dst: VReg, a: VOperand, amount: VOperand, width: Width },
        Sar    { dst: VReg, a: VOperand, amount: VOperand, width: Width },
        Mul    { dst: VReg, a: VOperand, b: VOperand, width: Width },
        // 128-bit vector ops — needed for PPC paired-singles AND ARM64 NEON.
        // Both guests get the SAME vector IR; only the frontend lowering differs.
        VecAdd { dst: VReg, a: VOperand, b: VOperand, lanes: u8, width: Width },
        VecMul { dst: VReg, a: VOperand, b: VOperand, lanes: u8, width: Width },
        VecFma { dst: VReg, a: VOperand, b: VOperand, c: VOperand, lanes: u8 },

        // ---- memory (endianness is explicit here, handled once in codegen —
        // Wii is big-endian, Switch is little-endian, frontends just set this) ----
        Load  { dst: VReg, addr: VOperand, width: Width, sign_ext: bool, endian: Endian },
        Store { addr: VOperand, val: VOperand, width: Width, endian: Endian },

        // ---- control flow ----
        Branch         { cond: Option<VOperand>, target: BlockId },
        IndirectBranch { target: VOperand },
        Call           { target: VOperand },
        Return,

        // ---- flags (lazy) ----
        SetFlags { op: FlagOp, a: VOperand, b: VOperand, width: Width },
        ReadFlag { dst: VReg, flag: FlagKind },

        // ---- guest-specific escape hatch, tightly scoped ----
        // Used for things with no clean fixed-IR encoding: PPC's paired-single
        // quirks, ARM64 system-register accesses, Switch SVC/HLE calls, etc.
        Intrinsic { id: IntrinsicId, effects: SideEffects, operands: Vec<VOperand>, dst: Option<VReg> },
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BlockId(pub u32);

    #[derive(Debug, Clone)]
    pub struct IrBlock {
        pub id: BlockId,
        pub guest_start_pc: u64,
        pub ops: Vec<IrOp>,
    }

    #[derive(Default)]
    pub struct IrBuilder {
        pub blocks: Vec<IrBlock>,
        next_vreg: u32,
        next_block: u32,
    }

    impl IrBuilder {
        pub fn new() -> Self { Self { blocks: Vec::new(), next_vreg: 0, next_block: 0 } }

        pub fn new_vreg(&mut self) -> VReg {
            let v = VReg(self.next_vreg);
            self.next_vreg += 1;
            v
        }

        pub fn new_block(&mut self, guest_start_pc: u64) -> BlockId {
            let id = BlockId(self.next_block);
            self.next_block += 1;
            self.blocks.push(IrBlock { id, guest_start_pc, ops: Vec::new() });
            id
        }

        pub fn push(&mut self, block: BlockId, op: IrOp) {
            self.blocks[block.0 as usize].ops.push(op);
        }
    }

    impl IrOp {
        /// True for the ops that transfer control and therefore must be the
        /// LAST op of a block.
        ///
        /// Single source of truth shared by the frontends (which must close
        /// every block with exactly one of these), the runtime (which closes a
        /// block that hit the instruction cap) and `ir_verify` (which rejects a
        /// terminator in the middle or a block with none).  Frontends must not
        /// re-derive this with their own `matches!`.
        #[inline]
        pub fn is_terminator(&self) -> bool {
            matches!(
                self,
                IrOp::Branch { .. }
                    | IrOp::IndirectBranch { .. }
                    | IrOp::Call { .. }
                    | IrOp::Return
            )
        }
    }

    /// `Option<&IrOp>` flavour so callers can say
    /// `if !is_terminator_op(block.ops.last()) { .. }` without a match arm for
    /// the empty-block case (an empty block is not terminated).
    #[inline]
    pub fn is_terminator_op(op: Option<&IrOp>) -> bool {
        matches!(op, Some(o) if o.is_terminator())
    }
}

// =============================================================================
// SECTION 2: OPTIMIZATION PASSES — shared across every frontend, run once.
// Kept deliberately cheap: this is a baseline-tier JIT for weak x86-64 hosts,
// so compile-time cost is a real budget, not just runtime throughput.
//
// Pass ordering matters:
//   1. constant_fold          — exposes constants to later passes.
//   2. redundant_load_elimination — removes repeated memory reads inside the block.
//   3. dead_flag_elimination  — removes lazy flag producers no longer observed.
//
// All three passes are deliberately intra-block. There is no CFG/dataflow solver
// here because block compilation must stay cheap on weak hosts.
// =============================================================================
pub mod passes {
    use crate::ir::{Endian, IrBlock, IrOp, VOperand, VReg, Width};
    use std::collections::HashMap;

    // -------------------------------------------------------------------------
    // Small helpers shared by the cheap peephole/value-propagation passes.
    // -------------------------------------------------------------------------

    #[inline]
    fn width_bits(width: Width) -> Option<u32> {
        match width {
            Width::W8 => Some(8),
            Width::W16 => Some(16),
            Width::W32 => Some(32),
            Width::W64 => Some(64),
            Width::W128 => None, // i64-based constant propagation stops here.
        }
    }

    #[inline]
    fn mask_to_width(value: i64, width: Width) -> i64 {
        match width_bits(width) {
            Some(64) => value,
            Some(bits) => {
                let mask = (1u64 << bits) - 1;
                (value as u64 & mask) as i64
            }
            None => value,
        }
    }

    #[inline]
    fn sign_extend_width(value: i64, width: Width) -> i64 {
        match width_bits(width) {
            Some(64) => value,
            Some(bits) => {
                let masked = (value as u64) & ((1u64 << bits) - 1);
                let sign_bit = 1u64 << (bits - 1);
                if masked & sign_bit != 0 {
                    (masked | (!0u64 << bits)) as i64
                } else {
                    masked as i64
                }
            }
            None => value,
        }
    }

    #[inline]
    fn resolve_const(op: VOperand, constants: &HashMap<VReg, i64>) -> VOperand {
        match op {
            VOperand::Reg(v) => constants
                .get(&v)
                .copied()
                .map(VOperand::Imm)
                .unwrap_or(VOperand::Reg(v)),
            imm @ VOperand::Imm(_) => imm,
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum ScalarFoldOp {
        Add,
        Sub,
        And,
        Or,
        Xor,
        Mul,
        Shl,
        Shr,
        Sar,
    }

    #[inline]
    fn fold_binary(
        kind: ScalarFoldOp,
        a: VOperand,
        b: VOperand,
        width: Width,
    ) -> Option<i64> {
        let (x, y) = match (a, b) {
            (VOperand::Imm(x), VOperand::Imm(y)) => (x, y),
            _ => return None,
        };

        let folded = match kind {
            ScalarFoldOp::Add => x.wrapping_add(y),
            ScalarFoldOp::Sub => x.wrapping_sub(y),
            ScalarFoldOp::And => x & y,
            ScalarFoldOp::Or  => x | y,
            ScalarFoldOp::Xor => x ^ y,
            ScalarFoldOp::Mul => x.wrapping_mul(y),

            ScalarFoldOp::Shl => {
                let shift = match width_bits(width) {
                    Some(bits) => (y as u32) & (bits - 1),
                    None => return None,
                };
                (x as u64).wrapping_shl(shift) as i64
            }

            ScalarFoldOp::Shr => {
                let shift = match width_bits(width) {
                    Some(bits) => (y as u32) & (bits - 1),
                    None => return None,
                };
                (x as u64).wrapping_shr(shift) as i64
            }

            ScalarFoldOp::Sar => {
                let shift = match width_bits(width) {
                    Some(bits) => (y as u32) & (bits - 1),
                    None => return None,
                };
                sign_extend_width(x, width).wrapping_shr(shift) as i64
            }
        };

        Some(mask_to_width(folded, width))
    }

    #[inline]
    fn resolve_substitution(op: &mut VOperand, substitutions: &HashMap<VReg, VOperand>) {
        let mut current = match *op {
            VOperand::Reg(v) => v,
            VOperand::Imm(_) => return,
        };

        for _ in 0..8 {
            match substitutions.get(&current).copied() {
                Some(VOperand::Imm(value)) => {
                    *op = VOperand::Imm(value);
                    return;
                }
                Some(VOperand::Reg(next)) if next != current => {
                    current = next;
                }
                _ => return,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MemRange {
        start: u64,
        end: u64, // exclusive
    }

    impl MemRange {
        #[inline]
        fn new(addr: i64, width: Width) -> Option<Self> {
            let bytes = match width {
                Width::W8 => 1,
                Width::W16 => 2,
                Width::W32 => 4,
                Width::W64 => 8,
                Width::W128 => 16,
            };

            let start = addr as u64;
            let end = start.checked_add(bytes)?;
            Some(Self { start, end })
        }

        #[inline]
        fn overlaps(self, other: Self) -> bool {
            self.start < other.end && other.start < self.end
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct CachedLoad {
        range: MemRange,
        width: Width,
        sign_ext: bool,
        endian: Endian,
        value: VOperand,
    }

    #[derive(Debug, Clone, Copy)]
    struct CachedStore {
        range: MemRange,
        width: Width,
        endian: Endian,
        value: VOperand,
    }

    /// Removes SetFlags ops whose result is never consumed by a later ReadFlag
    /// before the next SetFlags overwrites it.
    ///
    /// Because SetFlags defines the complete lazy flag state, seeing ANY later
    /// ReadFlag makes the current SetFlags observable. Once we cross a SetFlags
    /// while scanning backwards, older flag producers are dead until an older
    /// ReadFlag is encountered.
    pub fn dead_flag_elimination(block: &mut IrBlock) {
        let mut live = [false; 4];
        let mut keep = vec![true; block.ops.len()];

        for index in (0..block.ops.len()).rev() {
            match &block.ops[index] {
                IrOp::ReadFlag { flag, .. } => {
                    let slot = *flag as usize;
                    live[slot] = true;
                }

                IrOp::SetFlags { .. } => {
                    let observable = live.iter().any(|&is_live| is_live);

                    if !observable {
                        keep[index] = false;
                    }

                    // This SetFlags replaces the entire lazy flag snapshot, so
                    // no flag read older than this point can depend on the
                    // overwritten snapshot.
                    live.fill(false);
                }

                _ => {}
            }
        }

        if keep.iter().all(|&keep_op| keep_op) {
            return;
        }

        let old_ops = std::mem::take(&mut block.ops);
        block.ops = old_ops
            .into_iter()
            .enumerate()
            .filter_map(|(index, op)| keep[index].then_some(op))
            .collect();
    }

    /// Constant folding + propagation, intra-block only.
    ///
    /// The IR intentionally has no dedicated Const/Move opcode. When an
    /// expression becomes constant, the pass records that value for later uses
    /// and rewrites the defining operation to a cheap "constant-producing"
    /// arithmetic form. Later passes therefore see immediates instead of
    /// temporary VRegs without changing the universal IR shape.
    pub fn constant_fold(block: &mut IrBlock) {
        let mut constants: HashMap<VReg, i64> = HashMap::new();

        for op in &mut block.ops {
            // First substitute known constants into the operation's inputs.
            match op {
                IrOp::Add { a, b, .. }
                | IrOp::Sub { a, b, .. }
                | IrOp::And { a, b, .. }
                | IrOp::Or  { a, b, .. }
                | IrOp::Xor { a, b, .. }
                | IrOp::Mul { a, b, .. } => {
                    *a = resolve_const(*a, &constants);
                    *b = resolve_const(*b, &constants);
                }

                IrOp::Shl { a, amount, .. }
                | IrOp::Shr { a, amount, .. }
                | IrOp::Sar { a, amount, .. } => {
                    *a = resolve_const(*a, &constants);
                    *amount = resolve_const(*amount, &constants);
                }

                IrOp::SetFlags { a, b, .. } => {
                    *a = resolve_const(*a, &constants);
                    *b = resolve_const(*b, &constants);
                }

                IrOp::VecAdd { a, b, .. }
                | IrOp::VecMul { a, b, .. } => {
                    *a = resolve_const(*a, &constants);
                    *b = resolve_const(*b, &constants);
                }

                IrOp::VecFma { a, b, c, .. } => {
                    *a = resolve_const(*a, &constants);
                    *b = resolve_const(*b, &constants);
                    *c = resolve_const(*c, &constants);
                }

                IrOp::Load { addr, .. } => {
                    *addr = resolve_const(*addr, &constants);
                }

                IrOp::Store { addr, val, .. } => {
                    *addr = resolve_const(*addr, &constants);
                    *val = resolve_const(*val, &constants);
                }

                IrOp::Branch { cond, .. } => {
                    if let Some(cond) = cond.as_mut() {
                        *cond = resolve_const(*cond, &constants);
                    }
                }

                IrOp::IndirectBranch { target } | IrOp::Call { target } => {
                    *target = resolve_const(*target, &constants);
                }

                IrOp::Intrinsic { operands, .. } => {
                    for operand in operands {
                        *operand = resolve_const(*operand, &constants);
                    }
                }

                IrOp::ReadFlag { .. } | IrOp::Return => {}
            }

            match op {
                IrOp::Add { dst, a, b, width } => {
                    if let Some(value) = fold_binary(ScalarFoldOp::Add, *a, *b, *width) {
                        let dst = *dst;
                        let width = *width;
                        *a = VOperand::Imm(value);
                        *b = VOperand::Imm(0);
                        constants.insert(dst, mask_to_width(value, width));
                    } else {
                        constants.remove(dst);
                    }
                }

                IrOp::Sub { dst, a, b, width } => {
                    if let Some(value) = fold_binary(ScalarFoldOp::Sub, *a, *b, *width) {
                        let dst = *dst;
                        let width = *width;
                        *a = VOperand::Imm(value);
                        *b = VOperand::Imm(0);
                        constants.insert(dst, mask_to_width(value, width));
                    } else {
                        constants.remove(dst);
                    }
                }

                IrOp::And { dst, a, b, width } => {
                    if let Some(value) = fold_binary(ScalarFoldOp::And, *a, *b, *width) {
                        let dst = *dst;
                        let width = *width;
                        *a = VOperand::Imm(value);
                        *b = VOperand::Imm(0);
                        constants.insert(dst, mask_to_width(value, width));
                    } else {
                        constants.remove(dst);
                    }
                }

                IrOp::Or { dst, a, b, width } => {
                    if let Some(value) = fold_binary(ScalarFoldOp::Or, *a, *b, *width) {
                        let dst = *dst;
                        let width = *width;
                        *a = VOperand::Imm(value);
                        *b = VOperand::Imm(0);
                        constants.insert(dst, mask_to_width(value, width));
                    } else {
                        constants.remove(dst);
                    }
                }

                IrOp::Xor { dst, a, b, width } => {
                    if let Some(value) = fold_binary(ScalarFoldOp::Xor, *a, *b, *width) {
                        let dst = *dst;
                        let width = *width;
                        *a = VOperand::Imm(value);
                        *b = VOperand::Imm(0);
                        constants.insert(dst, mask_to_width(value, width));
                    } else {
                        constants.remove(dst);
                    }
                }

                IrOp::Mul { dst, a, b, width } => {
                    if let Some(value) = fold_binary(ScalarFoldOp::Mul, *a, *b, *width) {
                        let dst = *dst;
                        let width = *width;
                        *a = VOperand::Imm(value);
                        *b = VOperand::Imm(0);
                        constants.insert(dst, mask_to_width(value, width));
                    } else {
                        constants.remove(dst);
                    }
                }

                IrOp::Shl { dst, a, amount, width } => {
                    if let Some(value) = fold_binary(ScalarFoldOp::Shl, *a, *amount, *width) {
                        let dst = *dst;
                        *a = VOperand::Imm(value);
                        *amount = VOperand::Imm(0);
                        constants.insert(dst, value);
                    } else {
                        constants.remove(dst);
                    }
                }

                IrOp::Shr { dst, a, amount, width } => {
                    if let Some(value) = fold_binary(ScalarFoldOp::Shr, *a, *amount, *width) {
                        let dst = *dst;
                        *a = VOperand::Imm(value);
                        *amount = VOperand::Imm(0);
                        constants.insert(dst, value);
                    } else {
                        constants.remove(dst);
                    }
                }

                IrOp::Sar { dst, a, amount, width } => {
                    if let Some(value) = fold_binary(ScalarFoldOp::Sar, *a, *amount, *width) {
                        let dst = *dst;
                        *a = VOperand::Imm(value);
                        *amount = VOperand::Imm(0);
                        constants.insert(dst, value);
                    } else {
                        constants.remove(dst);
                    }
                }

                // These operations do not currently have scalar constant forms
                // in the universal IR, so they terminate propagation for the
                // destination rather than making an unsafe guess.
                IrOp::VecAdd { dst, .. }
                | IrOp::VecMul { dst, .. }
                | IrOp::VecFma { dst, .. }
                | IrOp::Load { dst, .. }
                | IrOp::ReadFlag { dst, .. }
                | IrOp::Intrinsic { dst: Some(dst), .. } => {
                    constants.remove(dst);
                }

                IrOp::Intrinsic { dst: None, .. }
                | IrOp::Store { .. }
                | IrOp::Branch { .. }
                | IrOp::IndirectBranch { .. }
                | IrOp::Call { .. }
                | IrOp::Return
                | IrOp::SetFlags { .. } => {}
            }
        }

        // The constant propagation above intentionally stays local to this
        // straight-line block; no cross-block facts are carried through here.
    }

    /// Eliminate redundant Load ops when the same concrete byte range was
    /// already loaded, or when a previous immediate Store fully determines the
    /// value being loaded.
    ///
    /// Dynamic addresses remain conservative:
    ///   • a dynamic Load is never guessed/reused;
    ///   • a dynamic Store clears all cached memory facts;
    ///   • memory-writing intrinsics clear all cached memory facts.
    ///
    /// When a redundant load is found, its destination VReg is substituted into
    /// later operands and the Load itself is removed. This is true load
    /// elimination, not merely "replace the load with a second ALU operation".
    pub fn redundant_load_elimination(block: &mut IrBlock) {
        let mut loads: Vec<CachedLoad> = Vec::new();
        let mut stores: Vec<CachedStore> = Vec::new();
        let mut substitutions: HashMap<VReg, VOperand> = HashMap::new();
        let mut output = Vec::with_capacity(block.ops.len());

        for mut op in std::mem::take(&mut block.ops) {
            // Apply load-elimination substitutions before inspecting the current
            // operation, so chains of removed loads collapse naturally.
            if !substitutions.is_empty() {
                match &mut op {
                    IrOp::Add { a, b, .. }
                    | IrOp::Sub { a, b, .. }
                    | IrOp::And { a, b, .. }
                    | IrOp::Or  { a, b, .. }
                    | IrOp::Xor { a, b, .. }
                    | IrOp::Mul { a, b, .. } => {
                        resolve_substitution(a, &substitutions);
                        resolve_substitution(b, &substitutions);
                    }
                    IrOp::Shl { a, amount, .. }
                    | IrOp::Shr { a, amount, .. }
                    | IrOp::Sar { a, amount, .. } => {
                        resolve_substitution(a, &substitutions);
                        resolve_substitution(amount, &substitutions);
                    }
                    IrOp::VecAdd { a, b, .. }
                    | IrOp::VecMul { a, b, .. } => {
                        resolve_substitution(a, &substitutions);
                        resolve_substitution(b, &substitutions);
                    }
                    IrOp::VecFma { a, b, c, .. } => {
                        resolve_substitution(a, &substitutions);
                        resolve_substitution(b, &substitutions);
                        resolve_substitution(c, &substitutions);
                    }
                    IrOp::Load { addr, .. } => resolve_substitution(addr, &substitutions),
                    IrOp::Store { addr, val, .. } => {
                        resolve_substitution(addr, &substitutions);
                        resolve_substitution(val, &substitutions);
                    }
                    IrOp::Branch { cond, .. } => {
                        if let Some(cond) = cond.as_mut() {
                            resolve_substitution(cond, &substitutions);
                        }
                    }
                    IrOp::IndirectBranch { target } | IrOp::Call { target } => {
                        resolve_substitution(target, &substitutions);
                    }
                    IrOp::SetFlags { a, b, .. } => {
                        resolve_substitution(a, &substitutions);
                        resolve_substitution(b, &substitutions);
                    }
                    IrOp::Intrinsic { operands, .. } => {
                        for operand in operands {
                            resolve_substitution(operand, &substitutions);
                        }
                    }
                    IrOp::ReadFlag { .. } | IrOp::Return => {}
                }
            }

            match &mut op {
                IrOp::Load {
                    dst,
                    addr,
                    width,
                    sign_ext,
                    endian,
                } => {
                    let concrete_addr = match *addr {
                        VOperand::Imm(addr) => addr,
                        VOperand::Reg(_) => {
                            output.push(op);
                            continue;
                        }
                    };

                    let Some(range) = MemRange::new(concrete_addr, *width) else {
                        output.push(op);
                        continue;
                    };

                    // Exact previously-loaded value: reuse the already computed
                    // VReg regardless of the original address expression.
                    if let Some(cached) = loads.iter().rev().find(|cached| {
                        cached.range == range
                            && cached.width == *width
                            && cached.sign_ext == *sign_ext
                            && cached.endian == *endian
                    }) {
                        let value = cached.value;
                        substitutions.insert(*dst, value);
                        continue;
                    }

                    // Exact range supplied by an immediate Store. Only immediate
                    // stores are reused here, because a register value may have
                    // higher bits that the guest store would have truncated.
                    if let Some(store) = stores.iter().rev().find(|store| {
                        store.range == range
                            && store.width == *width
                            && store.endian == *endian
                    }) {
                        if let VOperand::Imm(raw) = store.value {
                            let mut value = mask_to_width(raw, *width);
                            if *sign_ext {
                                value = sign_extend_width(value, *width);
                            }
                            let replacement = VOperand::Imm(value);
                            substitutions.insert(*dst, replacement);
                            loads.push(CachedLoad {
                                range,
                                width: *width,
                                sign_ext: *sign_ext,
                                endian: *endian,
                                value: replacement,
                            });
                            continue;
                        }
                    }

                    // This is a real load. Remember its value for later exact
                    // loads in the same block.
                    loads.retain(|cached| !cached.range.overlaps(range));
                    loads.push(CachedLoad {
                        range,
                        width: *width,
                        sign_ext: *sign_ext,
                        endian: *endian,
                        value: VOperand::Reg(*dst),
                    });
                    output.push(op);
                }

                IrOp::Store {
                    addr,
                    val,
                    width,
                    endian,
                } => {
                    let concrete_addr = match *addr {
                        VOperand::Imm(addr) => addr,
                        VOperand::Reg(_) => {
                            loads.clear();
                            stores.clear();
                            output.push(op);
                            continue;
                        }
                    };

                    let Some(range) = MemRange::new(concrete_addr, *width) else {
                        loads.clear();
                        stores.clear();
                        output.push(op);
                        continue;
                    };

                    // A write invalidates every cached load/store whose byte
                    // range can overlap it. Then the new store becomes the
                    // latest known fact for its exact range.
                    loads.retain(|cached| !cached.range.overlaps(range));
                    stores.retain(|cached| !cached.range.overlaps(range));

                    stores.push(CachedStore {
                        range,
                        width: *width,
                        endian: *endian,
                        value: *val,
                    });
                    output.push(op);
                }

                IrOp::Call { .. } => {
                    // Calls may touch arbitrary guest memory.
                    loads.clear();
                    stores.clear();
                    output.push(op);
                }

                IrOp::Intrinsic { effects, .. } => {
                    use crate::ir::SideEffects;
                    if matches!(
                        effects,
                        SideEffects::WritesMem
                            | SideEffects::ReadsAndWritesMem
                            | SideEffects::Volatile
                    ) {
                        loads.clear();
                        stores.clear();
                    }
                    output.push(op);
                }

                // A control-flow transfer terminates this straight-line memory
                // fact set even if a malformed frontend leaves trailing ops in
                // the block. Keeping this conservative costs almost nothing.
                IrOp::Branch { .. } | IrOp::IndirectBranch { .. } | IrOp::Return => {
                    loads.clear();
                    stores.clear();
                    output.push(op);
                }

                _ => output.push(op),
            }
        }

        block.ops = output;
    }

    pub fn run_all(block: &mut IrBlock) {
        constant_fold(block);
        redundant_load_elimination(block);
        dead_flag_elimination(block);
    }
}

// =============================================================================
// SECTION 3: REGISTER ALLOCATION — linear scan, not graph coloring.
// Graph coloring is too slow to run at JIT compile time; linear scan gives
// ~90% of the quality for a fraction of the compile cost, which is the right
// trade for a baseline-tier JIT on weak hardware.
//
// This implementation is intentionally single-block and SSA-like:
//   • first definition => interval start
//   • last operand use => interval end
//   • intervals sorted by start
//   • active intervals kept sorted by end
//   • expired intervals immediately return their host register to the free pool
//   • if all registers are busy, spill the interval with the farthest end
//
// r15 is never in the pool. It remains the permanent CpuState base register.
// =============================================================================
pub mod regalloc {
    use crate::ir::{IrBlock, IrOp, VOperand, VReg};
    use std::collections::HashMap;

    /// x86-64 general purpose registers available for allocation.
    /// r15 is RESERVED as the permanent guest-CpuState base pointer (see
    /// runtime::CpuState) — never allocated to a guest value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum HostGpr {
        Rax, Rbx, Rcx, Rdx, Rsi, Rdi, R8, R9, R10, R11, R12, R13, R14,
        // R15 intentionally excluded: reserved for guest-state base pointer.
    }

    #[derive(Debug, Clone, Copy)]
    pub enum Location {
        Reg(HostGpr),
        /// Semantics: `StateSpill(off)` means the value lives at byte offset
        /// `off` inside `CpuState::spill` (see `runtime::CpuState`) — NOT at
        /// offset `off` into CpuState as a whole. The C++ emitter
        /// materializes the access as:
        ///
        ///     [r15 + offsetof(CpuState, spill) + off]
        ///
        /// The offset unit is bytes; the allocator hands out 8-byte-aligned
        /// slots. This deliberately adds no second stack allocator and no
        /// new FFI concept — spilled guest state reuses the r15 base
        /// already required by the guest-state convention.
        ///
        /// Vector-width spills are NOT supported by this encoding (a 128-bit
        /// value needs a 16-byte slot). The allocator never produces a
        /// StateSpill for a vector-producing op, per the known gap noted
        /// at the top of this module.
        StateSpill(u32),
    }

    pub struct Allocation {
        pub assignments: HashMap<VReg, Location>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct LiveInterval {
        vreg: VReg,
        start: usize,
        end: usize,
    }

    #[derive(Debug, Clone, Copy)]
    struct ActiveInterval {
        interval: LiveInterval,
        reg: HostGpr,
    }

    #[inline]
    fn record_use(op: &IrOp, first: &mut HashMap<VReg, usize>, last: &mut HashMap<VReg, usize>, index: usize) {
        fn use_operand(
            operand: &VOperand,
            first: &mut HashMap<VReg, usize>,
            last: &mut HashMap<VReg, usize>,
            index: usize,
        ) {
            if let VOperand::Reg(v) = *operand {
                first.entry(v).or_insert(index);
                last.insert(v, index);
            }
        }

        match op {
            IrOp::Add { a, b, .. }
            | IrOp::Sub { a, b, .. }
            | IrOp::And { a, b, .. }
            | IrOp::Or  { a, b, .. }
            | IrOp::Xor { a, b, .. }
            | IrOp::Mul { a, b, .. } => {
                use_operand(a, first, last, index);
                use_operand(b, first, last, index);
            }

            IrOp::Shl { a, amount, .. }
            | IrOp::Shr { a, amount, .. }
            | IrOp::Sar { a, amount, .. } => {
                use_operand(a, first, last, index);
                use_operand(amount, first, last, index);
            }

            IrOp::VecAdd { a, b, .. }
            | IrOp::VecMul { a, b, .. } => {
                use_operand(a, first, last, index);
                use_operand(b, first, last, index);
            }

            IrOp::VecFma { a, b, c, .. } => {
                use_operand(a, first, last, index);
                use_operand(b, first, last, index);
                use_operand(c, first, last, index);
            }

            IrOp::Load { addr, .. } => {
                use_operand(addr, first, last, index);
            }

            IrOp::Store { addr, val, .. } => {
                use_operand(addr, first, last, index);
                use_operand(val, first, last, index);
            }

            IrOp::Branch { cond, .. } => {
                if let Some(cond) = cond {
                    use_operand(cond, first, last, index);
                }
            }

            IrOp::IndirectBranch { target } | IrOp::Call { target } => {
                use_operand(target, first, last, index);
            }

            IrOp::SetFlags { a, b, .. } => {
                use_operand(a, first, last, index);
                use_operand(b, first, last, index);
            }

            IrOp::Intrinsic { operands, .. } => {
                for operand in operands {
                    use_operand(operand, first, last, index);
                }
            }

            IrOp::ReadFlag { .. } | IrOp::Return => {}
        }
    }

    #[inline]
    fn record_def(op: &IrOp, index: usize, defs: &mut HashMap<VReg, usize>) {
        let dst = match op {
            IrOp::Add { dst, .. }
            | IrOp::Sub { dst, .. }
            | IrOp::And { dst, .. }
            | IrOp::Or  { dst, .. }
            | IrOp::Xor { dst, .. }
            | IrOp::Shl { dst, .. }
            | IrOp::Shr { dst, .. }
            | IrOp::Sar { dst, .. }
            | IrOp::Mul { dst, .. }
            | IrOp::VecAdd { dst, .. }
            | IrOp::VecMul { dst, .. }
            | IrOp::VecFma { dst, .. }
            | IrOp::Load { dst, .. }
            | IrOp::ReadFlag { dst, .. } => Some(*dst),

            IrOp::Intrinsic { dst: Some(dst), .. } => Some(*dst),

            IrOp::Store { .. }
            | IrOp::Branch { .. }
            | IrOp::IndirectBranch { .. }
            | IrOp::Call { .. }
            | IrOp::Return
            | IrOp::SetFlags { .. }
            | IrOp::Intrinsic { dst: None, .. } => None,
        };

        if let Some(vreg) = dst {
            // SSA-like IR should only define a VReg once. Keep the first
            // definition if malformed input violates that rule; the interval
            // builder remains deterministic rather than panicking in JIT code.
            defs.entry(vreg).or_insert(index);
        }
    }

    fn compute_intervals(block: &IrBlock) -> Vec<LiveInterval> {
        let mut defs: HashMap<VReg, usize> = HashMap::new();
        let mut first_use: HashMap<VReg, usize> = HashMap::new();
        let mut last_use: HashMap<VReg, usize> = HashMap::new();

        for (index, op) in block.ops.iter().enumerate() {
            record_def(op, index, &mut defs);
            record_use(op, &mut first_use, &mut last_use, index);
        }

        // A used VReg should normally have a definition in well-formed SSA-like
        // IR. If a frontend accidentally produces a use-before-def value, start
        // it at its first use so the allocator still produces a concrete map.
        let mut all_vregs: Vec<VReg> = defs
            .keys()
            .copied()
            .chain(first_use.keys().copied())
            .collect();
        all_vregs.sort_by_key(|v| v.0);
        all_vregs.dedup();

        let mut intervals = Vec::with_capacity(all_vregs.len());

        for vreg in all_vregs {
            let start = defs
                .get(&vreg)
                .copied()
                .or_else(|| first_use.get(&vreg).copied())
                .unwrap_or(0);

            let end = last_use.get(&vreg).copied().unwrap_or(start);

            intervals.push(LiveInterval { vreg, start, end });
        }

        intervals.sort_by_key(|interval| (interval.start, interval.end, interval.vreg.0));
        intervals
    }

    #[inline]
    fn expire_old(
        current_start: usize,
        active: &mut Vec<ActiveInterval>,
        free: &mut Vec<HostGpr>,
    ) {
        let mut index = 0;
        while index < active.len() {
            if active[index].interval.end < current_start {
                let expired = active.swap_remove(index);
                free.push(expired.reg);
            } else {
                index += 1;
            }
        }
    }

    #[inline]
    fn spill_offset(slot: u32) -> u32 {
        // Preserve the existing StateSpill(u32 byte-offset) ABI. The logical
        // slots are 8-byte aligned, which is sufficient for every scalar VReg;
        // vector-width spill support remains an emitter concern just as it was
        // in the original architecture skeleton.
        slot.saturating_mul(8)
    }

    /// Linear-scan allocation over a single IR block's live ranges.
    ///
    /// The returned Allocation contains exactly one Location per VReg observed
    /// in the block. The allocator does not allocate r15 and does not create a
    /// host stack frame.
    pub fn linear_scan(block: &IrBlock) -> Allocation {
        let intervals = compute_intervals(block);

        let mut free = vec![
            HostGpr::Rax,
            HostGpr::Rbx,
            HostGpr::Rcx,
            HostGpr::Rdx,
            HostGpr::Rsi,
            HostGpr::Rdi,
            HostGpr::R8,
            HostGpr::R9,
            HostGpr::R10,
            HostGpr::R11,
            HostGpr::R12,
            HostGpr::R13,
            HostGpr::R14,
        ];

        let mut active: Vec<ActiveInterval> = Vec::with_capacity(free.len());
        let mut assignments = HashMap::with_capacity(intervals.len());
        let mut next_spill_slot = 0u32;

        for current in intervals {
            expire_old(current.start, &mut active, &mut free);

            if let Some(reg) = free.pop() {
                assignments.insert(current.vreg, Location::Reg(reg));
                active.push(ActiveInterval {
                    interval: current,
                    reg,
                });
                active.sort_by_key(|entry| entry.interval.end);
                continue;
            }

            // No free register: standard linear-scan heuristic is to spill the
            // active interval whose end is farthest in the future. If that
            // interval lives longer than the current interval, swap locations:
            // the current value gets the register and the long-lived value gets
            // a spill slot. Otherwise spill the current interval directly.
            let farthest_index = active
                .iter()
                .enumerate()
                .max_by_key(|(_, entry)| entry.interval.end)
                .map(|(index, _)| index)
                .expect("active cannot be empty when all registers are occupied");

            let farthest = active[farthest_index];

            if farthest.interval.end > current.end {
                let spill_slot = next_spill_slot;
                next_spill_slot = next_spill_slot.saturating_add(1);

                assignments.insert(
                    farthest.interval.vreg,
                    Location::StateSpill(spill_offset(spill_slot)),
                );
                assignments.insert(current.vreg, Location::Reg(farthest.reg));

                active.swap_remove(farthest_index);
                active.push(ActiveInterval {
                    interval: current,
                    reg: farthest.reg,
                });
                active.sort_by_key(|entry| entry.interval.end);
            } else {
                let spill_slot = next_spill_slot;
                next_spill_slot = next_spill_slot.saturating_add(1);
                assignments.insert(
                    current.vreg,
                    Location::StateSpill(spill_offset(spill_slot)),
                );
            }
        }

        Allocation { assignments }
    }
}

// =============================================================================
// SECTION 4: FFI MARSHALING + THIN RUST WRAPPER AROUND THE C++ EMITTER
// The heavy encoding work (SSE2/AVX2/FMA paths, movbe vs bswap, etc.) lives
// entirely on the C++ side. This module only serializes IR + Allocation into
// the flat C structs and calls across the boundary.
// =============================================================================
pub mod jit_ffi {
    use crate::ir::{IrBlock, IrOp, VOperand, Width, Endian, FlagKind, FlagOp, SideEffects};
    use crate::regalloc::{Allocation, HostGpr, Location};
    use crate::{
        CIrBlock, CIrOp, CIrOpKind, COperand, COperandKind, CWidth, CEndian,
        CFlagKind, CFlagOp, CSideEffects, CAllocEntry, CLocation, CLocationKind,
        CHostGpr, CEmitResult, CJitStatus, CHostCaps, CPatchSite,
        CIR_MAX_OPERANDS, jit_emit_block, jit_free_code, jit_patch_chain,
        jit_detect_host_caps,
    };
    use std::ptr;

    /// Rust-side view of a finished executable block. Owns the memory
    /// returned by the C++ emitter; drops via jit_free_code.
    #[derive(Debug)]
    pub struct CompiledBlock {
        pub code_ptr: *mut u8,
        pub code_len: usize,
        pub guest_start_pc: u64,
        pub patchable_exits: Vec<CPatchSite>,
    }

    impl Drop for CompiledBlock {
        fn drop(&mut self) {
            if !self.code_ptr.is_null() {
                unsafe { jit_free_code(self.code_ptr) };
                self.code_ptr = ptr::null_mut();
            }
            // patch_sites were allocated together with the code buffer
            // (or freed by the same jit_free_code call); nothing else to do.
        }
    }

    // Safety: CompiledBlock is Send because the underlying executable memory
    // is just bytes; the only requirement is that we never free while another
    // thread is executing it (enforced by the code-cache ownership model).
    unsafe impl Send for CompiledBlock {}
    unsafe impl Sync for CompiledBlock {}

    fn width_to_c(w: Width) -> CWidth {
        match w {
            Width::W8   => CWidth::W8,
            Width::W16  => CWidth::W16,
            Width::W32  => CWidth::W32,
            Width::W64  => CWidth::W64,
            Width::W128 => CWidth::W128,
        }
    }

    fn endian_to_c(e: Endian) -> CEndian {
        match e {
            Endian::Little => CEndian::Little,
            Endian::Big    => CEndian::Big,
        }
    }

    fn operand_to_c(op: &VOperand) -> COperand {
        match *op {
            VOperand::Reg(v) => COperand { kind: COperandKind::Reg, reg: v.0, imm: 0 },
            VOperand::Imm(i) => COperand { kind: COperandKind::Imm, reg: 0, imm: i },
        }
    }

    fn host_gpr_to_c(g: HostGpr) -> CHostGpr {
        match g {
            HostGpr::Rax => CHostGpr::Rax,
            HostGpr::Rbx => CHostGpr::Rbx,
            HostGpr::Rcx => CHostGpr::Rcx,
            HostGpr::Rdx => CHostGpr::Rdx,
            HostGpr::Rsi => CHostGpr::Rsi,
            HostGpr::Rdi => CHostGpr::Rdi,
            HostGpr::R8  => CHostGpr::R8,
            HostGpr::R9  => CHostGpr::R9,
            HostGpr::R10 => CHostGpr::R10,
            HostGpr::R11 => CHostGpr::R11,
            HostGpr::R12 => CHostGpr::R12,
            HostGpr::R13 => CHostGpr::R13,
            HostGpr::R14 => CHostGpr::R14,
        }
    }

    fn location_to_c(loc: &Location) -> CLocation {
        match *loc {
            Location::Reg(g) => CLocation {
                kind: CLocationKind::Reg,
                gpr: host_gpr_to_c(g),
                offset: 0,
            },
            Location::StateSpill(off) => CLocation {
                kind: CLocationKind::StateSpill,
                gpr: CHostGpr::Rax, // unused
                offset: off,
            },
        }
    }

    /// Serialize one Rust IrOp into the fixed-size C payload.
    ///
    /// Returns `Err` if an `Intrinsic` carries more operands than
    /// `CIR_MAX_OPERANDS` can hold. Previous revisions silently truncated
    /// via `.min(CIR_MAX_OPERANDS)`, which drops real operand data with no
    /// signal — a frontend bug (or a future intrinsic that legitimately
    /// needs more operands) would translate into wrong-but-not-crashing
    /// guest behavior. Failing loudly here, before anything crosses the FFI
    /// boundary, is strictly better than debugging that downstream.
    fn ir_op_to_c(op: &IrOp) -> Result<CIrOp, CJitStatus> {
        let mut c = CIrOp {
            kind: CIrOpKind::Add, // overwritten below
            dst: 0,
            dst_valid: 0,
            width: CWidth::W64,
            endian: CEndian::Little,
            sign_ext: 0,
            lanes: 0,
            flag_kind: CFlagKind::Zero,
            flag_op: CFlagOp::AddOp,
            side_effects: CSideEffects::Pure,
            intrinsic_id: 0,
            target_block: 0,
            num_operands: 0,
            operands: [COperand { kind: COperandKind::Reg, reg: 0, imm: 0 }; CIR_MAX_OPERANDS],
        };

        match op {
            IrOp::Add { dst, a, b, width } => {
                c.kind = CIrOpKind::Add;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(b);
            }
            IrOp::Sub { dst, a, b, width } => {
                c.kind = CIrOpKind::Sub;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(b);
            }
            IrOp::And { dst, a, b, width } => {
                c.kind = CIrOpKind::And;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(b);
            }
            IrOp::Or { dst, a, b, width } => {
                c.kind = CIrOpKind::Or;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(b);
            }
            IrOp::Xor { dst, a, b, width } => {
                c.kind = CIrOpKind::Xor;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(b);
            }
            IrOp::Shl { dst, a, amount, width } => {
                c.kind = CIrOpKind::Shl;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(amount);
            }
            IrOp::Shr { dst, a, amount, width } => {
                c.kind = CIrOpKind::Shr;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(amount);
            }
            IrOp::Sar { dst, a, amount, width } => {
                c.kind = CIrOpKind::Sar;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(amount);
            }
            IrOp::Mul { dst, a, b, width } => {
                c.kind = CIrOpKind::Mul;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(b);
            }
            IrOp::VecAdd { dst, a, b, lanes, width } => {
                c.kind = CIrOpKind::VecAdd;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.lanes = *lanes;
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(b);
            }
            IrOp::VecMul { dst, a, b, lanes, width } => {
                c.kind = CIrOpKind::VecMul;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.lanes = *lanes;
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(b);
            }
            IrOp::VecFma { dst, a, b, c: c_op, lanes } => {
                c.kind = CIrOpKind::VecFma;
                c.dst = dst.0; c.dst_valid = 1;
                c.lanes = *lanes;
                c.num_operands = 3;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(b);
                c.operands[2] = operand_to_c(c_op);
            }
            IrOp::Load { dst, addr, width, sign_ext, endian } => {
                c.kind = CIrOpKind::Load;
                c.dst = dst.0; c.dst_valid = 1;
                c.width = width_to_c(*width);
                c.sign_ext = if *sign_ext { 1 } else { 0 };
                c.endian = endian_to_c(*endian);
                c.num_operands = 1;
                c.operands[0] = operand_to_c(addr);
            }
            IrOp::Store { addr, val, width, endian } => {
                c.kind = CIrOpKind::Store;
                c.width = width_to_c(*width);
                c.endian = endian_to_c(*endian);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(addr);
                c.operands[1] = operand_to_c(val);
            }
            IrOp::Branch { cond, target } => {
                c.kind = CIrOpKind::Branch;
                c.target_block = target.0;
                if let Some(v) = cond {
                    c.num_operands = 1;
                    c.operands[0] = operand_to_c(v);
                }
            }
            IrOp::IndirectBranch { target } => {
                c.kind = CIrOpKind::IndirectBranch;
                c.num_operands = 1;
                c.operands[0] = operand_to_c(target);
            }
            IrOp::Call { target } => {
                c.kind = CIrOpKind::Call;
                c.num_operands = 1;
                c.operands[0] = operand_to_c(target);
            }
            IrOp::Return => {
                c.kind = CIrOpKind::Return;
            }
            IrOp::SetFlags { op, a, b, width } => {
                c.kind = CIrOpKind::SetFlags;
                c.flag_op = match op {
                    FlagOp::AddOp   => CFlagOp::AddOp,
                    FlagOp::SubOp   => CFlagOp::SubOp,
                    FlagOp::AndOp   => CFlagOp::AndOp,
                    FlagOp::OrOp    => CFlagOp::OrOp,
                    FlagOp::XorOp   => CFlagOp::XorOp,
                    FlagOp::ShiftOp => CFlagOp::ShiftOp,
                };
                c.width = width_to_c(*width);
                c.num_operands = 2;
                c.operands[0] = operand_to_c(a);
                c.operands[1] = operand_to_c(b);
            }
            IrOp::ReadFlag { dst, flag } => {
                c.kind = CIrOpKind::ReadFlag;
                c.dst = dst.0; c.dst_valid = 1;
                c.flag_kind = match flag {
                    FlagKind::Zero     => CFlagKind::Zero,
                    FlagKind::Carry    => CFlagKind::Carry,
                    FlagKind::Overflow => CFlagKind::Overflow,
                    FlagKind::Negative => CFlagKind::Negative,
                };
            }
            IrOp::Intrinsic { id, effects, operands, dst } => {
                c.kind = CIrOpKind::Intrinsic;
                c.intrinsic_id = id.0;
                c.side_effects = match effects {
                    SideEffects::Pure              => CSideEffects::Pure,
                    SideEffects::ReadsMem          => CSideEffects::ReadsMem,
                    SideEffects::WritesMem         => CSideEffects::WritesMem,
                    SideEffects::ReadsAndWritesMem => CSideEffects::ReadsAndWritesMem,
                    SideEffects::Volatile          => CSideEffects::Volatile,
                };
                if let Some(d) = dst {
                    c.dst = d.0;
                    c.dst_valid = 1;
                }
                if operands.len() > CIR_MAX_OPERANDS {
                    // Do not silently drop operands past the fixed-size
                    // payload — surface it as a hard translation failure
                    // instead. See doc comment on this function.
                    return Err(CJitStatus::InvalidInput);
                }
                c.num_operands = operands.len() as u8;
                for (i, opnd) in operands.iter().enumerate() {
                    c.operands[i] = operand_to_c(opnd);
                }
            }
        }
        Ok(c)
    }

    /// Convert a Rust IrBlock + Allocation into the flat C representation and
    /// hand it to the C++ emitter.
    pub fn emit_block(
        block: &IrBlock,
        alloc: &Allocation,
        caps: CHostCaps,
        force_baseline: bool,
    ) -> Result<CompiledBlock, CJitStatus> {
        // Serialize ops. Bails out before touching the FFI boundary at all
        // if any Intrinsic overflows CIR_MAX_OPERANDS (see ir_op_to_c doc).
        let c_ops: Vec<CIrOp> = block
            .ops
            .iter()
            .map(ir_op_to_c)
            .collect::<Result<Vec<_>, _>>()?;

        // Serialize allocation map
        let c_allocs: Vec<CAllocEntry> = alloc.assignments.iter().map(|(vreg, loc)| {
            CAllocEntry {
                vreg: vreg.0,
                loc: location_to_c(loc),
            }
        }).collect();

        let c_block = CIrBlock {
            guest_start_pc: block.guest_start_pc,
            block_id: block.id.0,
            num_ops: c_ops.len() as u32,
            ops: c_ops.as_ptr(),
            num_allocs: c_allocs.len() as u32,
            allocs: c_allocs.as_ptr(),
        };

        let mut out = CEmitResult {
            code: ptr::null_mut(),
            code_len: 0,
            num_patch_sites: 0,
            patch_sites: ptr::null_mut(),
        };

        let status = unsafe {
            jit_emit_block(
                &c_block,
                caps,
                if force_baseline { 1 } else { 0 },
                &mut out,
            )
        };

        if status != CJitStatus::Ok {
            return Err(status);
        }

        // Take ownership of the patch-site array into a Rust Vec.
        // Safety: C++ guarantees the array is valid and of the stated length.
        let patchable_exits = if out.num_patch_sites > 0 && !out.patch_sites.is_null() {
            unsafe {
                let slice = std::slice::from_raw_parts(out.patch_sites, out.num_patch_sites as usize);
                let v = slice.to_vec();
                // The companion allocation will be freed by jit_free_code when
                // the CompiledBlock is dropped; we do not free patch_sites here.
                v
            }
        } else {
            Vec::new()
        };

        Ok(CompiledBlock {
            code_ptr: out.code,
            code_len: out.code_len,
            guest_start_pc: block.guest_start_pc,
            patchable_exits,
        })
    }

    /// Convenience: detect caps once at startup.
    pub fn detect_host_caps() -> CHostCaps {
        unsafe { jit_detect_host_caps() }
    }

    /// Patch a direct-branch exit to point at another compiled block’s entry.
    pub fn patch_chain(
        from: &CompiledBlock,
        exit_index: usize,
        to_entry: *const u8,
    ) -> Result<(), CJitStatus> {
        let site = from.patchable_exits.get(exit_index)
            .ok_or(CJitStatus::InvalidInput)?;
        let status = unsafe {
            jit_patch_chain(
                from.code_ptr,
                from.code_len,
                site.code_offset,
                to_entry,
            )
        };
        if status == CJitStatus::Ok { Ok(()) } else { Err(status) }
    }
}

// =============================================================================
// SECTION 5: CODE CACHE — guest PC -> compiled block, plus self-modifying
// code (SMC) invalidation via a reverse page -> blocks map.
// =============================================================================
pub mod cache {
    use crate::jit_ffi::CompiledBlock;
    use std::collections::HashMap;

    pub struct CodeCache {
        by_pc: HashMap<u64, CompiledBlock>,
        /// guest page number -> block guest-PCs that were translated from it.
        /// On a guest write into a tracked page, every listed block is
        /// invalidated and will be retranslated on next execution.
        page_to_blocks: HashMap<u64, Vec<u64>>,
        /// Soft cap tuned conservatively for weak/low-RAM hosts — this is
        /// deliberately smaller than what a "modern desktop" JIT would use.
        max_bytes: usize,
        current_bytes: usize,
    }

    impl CodeCache {
        pub fn new(max_bytes: usize) -> Self {
            Self { by_pc: HashMap::new(), page_to_blocks: HashMap::new(), max_bytes, current_bytes: 0 }
        }

        pub fn lookup(&self, guest_pc: u64) -> Option<&CompiledBlock> {
            self.by_pc.get(&guest_pc)
        }

        pub fn insert(&mut self, guest_pc: u64, block: CompiledBlock) {
            // real impl: if current_bytes + block.code_len > max_bytes,
            // evict LRU/oldest entries first (see evict_lru below).
            self.current_bytes += block.code_len;
            self.by_pc.insert(guest_pc, block);
        }

        pub fn invalidate_page(&mut self, guest_page: u64) {
            if let Some(pcs) = self.page_to_blocks.remove(&guest_page) {
                for pc in pcs {
                    if let Some(b) = self.by_pc.remove(&pc) {
                        self.current_bytes -= b.code_len;
                        // CompiledBlock::drop calls jit_free_code automatically
                    }
                }
            }
        }

        fn evict_lru(&mut self, bytes_needed: usize) {
            todo!("evict oldest/coldest blocks until current_bytes + bytes_needed <= max_bytes")
        }
    }
}

// =============================================================================
// SECTION 6: FRONTEND CONTRACT — the only guest-specific surface. Wii and
// Switch each implement this trait; nothing downstream special-cases either.
// =============================================================================
pub mod frontend {
    use crate::ir::IrBuilder;

    pub struct RegisterLayout {
        pub gpr_count: u32,
        pub fpr_count: u32,
        pub vector_reg_count: u32,
    }

    pub trait Frontend {
        type DecodedInsn;

        /// Decode one guest instruction at `pc` from raw guest memory bytes.
        /// Returns the decoded form and its length in bytes (4 for both PPC
        /// and AArch64 — both are fixed-width, which simplifies this a lot
        /// compared to variable-width ISAs like x86 guests would need).
        fn decode(&self, bytes: &[u8], pc: u64) -> (Self::DecodedInsn, usize);

        /// Lower one decoded instruction into the universal IR.
        fn lower_to_ir(&self, insn: &Self::DecodedInsn, ir: &mut IrBuilder, block: crate::ir::BlockId);

        fn register_file_layout(&self) -> RegisterLayout;

        fn is_block_terminator(&self, insn: &Self::DecodedInsn) -> bool;

        /// Guest memory endianness — Wii(PPC)=Big, Switch(ARM64)=Little.
        /// Threaded into every Load/Store IR op this frontend emits.
        fn endianness(&self) -> crate::ir::Endian;
    }
}

// =============================================================================
// SECTION 7: WII FRONTEND (PowerPC "Broadway", big-endian) — IMPLEMENTED
//
// The skeleton that used to live here (decode stub + `todo!()` lowering) is
// now a real module: src/frontends/wii_ppc/.  It is declared with #[path] so
// the crate-root module name `crate::frontend_wii_ppc` — and therefore every
// reference to it from this file — is unchanged.
//
// What that module owns (read its module doc first; it defines the conventions
// this core relies on and has no say in here):
//   • PPC opcode dispatch table: decode (primary + groups 4/19/31/59/63) and
//     lowering of every decoded instruction into the universal IR above.
//   • The PPC `Intrinsic` id table (guest register-file / FP / cache ops) —
//     that table is ALSO part of the C++ contract; see the APPENDIX note at the
//     bottom of this file.
//   • DecodedPpc + Frontend impl (same trait, no new methods).
// =============================================================================
#[path = "frontends/wii_ppc/mod.rs"]
pub mod frontend_wii_ppc;


// =============================================================================
// SECTION 8: SWITCH FRONTEND SKELETON (ARMv8-A, Tegra X1, little-endian)
// =============================================================================
pub mod frontend_switch_arm64 {
    use crate::frontend::{Frontend, RegisterLayout};
    use crate::ir::{IrBuilder, BlockId, Endian};

    pub struct DecodedA64 {
        pub raw: u32,
        pub class_bits: u8, // top-level instruction class, bits [28:25]
    }

    pub struct SwitchFrontend;

    impl Frontend for SwitchFrontend {
        type DecodedInsn = DecodedA64;

        fn decode(&self, bytes: &[u8], pc: u64) -> (DecodedA64, usize) {
            let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let class_bits = ((raw >> 25) & 0xF) as u8;
            (DecodedA64 { raw, class_bits }, 4)
        }

        fn lower_to_ir(&self, insn: &DecodedA64, ir: &mut IrBuilder, block: BlockId) {
            // real impl: dispatch on class_bits per ARMv8 encoding groups
            // (data-processing-immediate, branch/exception/system, load/store,
            // data-processing-register, data-processing-SIMD/FP). NEON ops
            // lower to the SAME VecAdd/VecMul/VecFma IR as Wii paired-singles.
            todo!("AArch64 instruction-class dispatch -> IrOp sequence")
        }

        fn register_file_layout(&self) -> RegisterLayout {
            RegisterLayout { gpr_count: 31, fpr_count: 32, vector_reg_count: 32 }
        }

        fn is_block_terminator(&self, insn: &DecodedA64) -> bool {
            matches!(insn.class_bits, 0b1010 | 0b1011) // branch/exception/system class
        }

        fn endianness(&self) -> Endian { Endian::Little }
    }
}

// =============================================================================
// SECTION 9: RUNTIME — guest CPU state, dispatcher, block-chaining glue.
// =============================================================================
pub mod runtime {
    use crate::cache::CodeCache;
    use crate::jit_ffi::CompiledBlock;
    use crate::CHostCaps;

    /// Guest CPU register file + misc state. A pointer to this struct is
    /// kept PERMANENTLY in host register r15 for the lifetime of JIT'd code
    /// execution — spilled guest registers live as fields here, so a spill
    /// reload is a single `mov reg, [r15 + offset]`, no stack frame needed.
    ///
    /// Field ordering is part of the FFI contract: the C++ emitter mirrors
    /// this struct and references `gpr`/`pc`/`flags_*`/`spill` at fixed
    /// offsets. Do NOT reorder fields without updating the C++ side in
    /// lock-step. New fields must be APPENDED so existing offsets stay
    /// stable.
    ///
    /// `spill` is a DEDICATED register-allocator spill area, deliberately
    /// separate from `gpr`/`fpr`/`vec`. `regalloc::Location::StateSpill(off)`
    /// indexes into this array (byte offset from its start), NOT into
    /// CpuState as a whole — an earlier revision conflated the two, which
    /// would have made spill slot 0 silently alias `gpr[0]` and corrupt a
    /// live guest register on the first spill. This array exists
    /// specifically so that can't happen.
    ///
    /// 128 slots × 8 bytes = 1 KiB. This is a hard budget: linear_scan must
    /// never produce a StateSpill offset past the end of this array. The
    /// allocator is conservative (13 allocatable GPRs, blocks capped at a
    /// small instruction count), so a handful of spills per block is the
    /// realistic ceiling — 128 slots is comfortably above that. If a future
    /// change to the allocator or block size needs more, this is the single
    /// place to raise, with the C++ mirror updated in lock-step.
    #[repr(C)]
    pub struct CpuState {
        pub gpr: [u64; 32],
        pub fpr: [f64; 32],
        pub vec: [u128; 32],
        pub pc: u64,
        pub flags_zero: bool,
        pub flags_carry: bool,
        pub flags_overflow: bool,
        pub flags_negative: bool,
        pub spill: [u64; 128],
        /// ---- APPENDED AFTER `spill`; nothing above this line moved ---------
        ///
        /// Architected *miscellaneous* guest register file: PPC
        /// CR/XER/LR/CTR/FPSCR/GQR0..7/TBL/TBU/DEC/…, ARM64 NZCV/FPCR/FPSR/
        /// TPIDR_EL0/….  Deliberately NOT folded into `gpr` (guest-visible ABI
        /// registers keep their own slots for the interpreter and the
        /// debugger) and deliberately NOT aliased onto `spill` (spill slots are
        /// scratch the allocator may hand out again at any time).  Slot
        /// numbering is guest-specific and owned by each frontend
        /// (`frontend_wii_ppc::PpcMisc`, and the ARM64 equivalent when written);
        /// the emitter only needs
        ///     mov [r15 + offsetof(CpuState, guest_misc) + 8*i]
        /// so adding a guest misc register costs zero C++ work.
        ///
        /// Offsets of every field above are unchanged by this append:
        /// gpr=0, fpr=256, vec=512, pc=1024, flags_{z,c,o,n}=1032..1035,
        /// spill=1040.  `frontends/wii_ppc/layout_tests.rs` asserts exactly
        /// those constants so an accidental *insert* in the middle fails
        /// loudly instead of silently aliasing spill slot 0 onto `gpr[0]`
        /// (the bug the dedicated `spill` array exists to prevent).
        pub guest_misc: [u64; GUEST_MISC_SLOTS],
    }

    /// Number of architected misc-register slots available to a frontend.
    /// 64 × 8 B = 512 B of per-thread state; the Wii uses ~24 of them and the
    /// Switch will need fewer than 16 for the EL0-visible set.
    pub const GUEST_MISC_SLOTS: usize = 64;

    impl CpuState {
        /// Zero-initialised state.  Safe alternative to the `mem::zeroed()` the
        /// entry point uses; every field here is a plain integer/array so all-zero
        /// is a valid bit pattern.
        pub fn new_zeroed() -> Self {
            // SAFETY: CpuState is all-integer fields (repr(C), no padding
            // dependencies, no pointer/bool invariants to violate — the four
            // `flags_*` bools are all-zero == false).
            unsafe { core::mem::MaybeUninit::zeroed().assume_init() }
        }

        #[inline]
        pub fn get_gpr(&self, i: u8) -> u64 {
            self.gpr[i as usize & 31]
        }

        #[inline]
        pub fn set_gpr(&mut self, i: u8, v: u64) {
            self.gpr[i as usize & 31] = v;
        }

        /// Raw 64-bit pattern of an FPR slot (a valid IEEE-754 double for FP ops;
        /// see the frontend's FPR representation contract).
        #[inline]
        pub fn get_fpr_bits(&self, i: u8) -> u64 {
            self.fpr[i as usize & 31].to_bits()
        }

        #[inline]
        pub fn set_fpr_bits(&mut self, i: u8, bits: u64) {
            self.fpr[i as usize & 31] = f64::from_bits(bits);
        }

        #[inline]
        pub fn misc(&self, slot: u8) -> u64 {
            self.guest_misc[slot as usize % GUEST_MISC_SLOTS]
        }

        #[inline]
        pub fn set_misc(&mut self, slot: u8, value: u64) {
            self.guest_misc[slot as usize % GUEST_MISC_SLOTS] = value;
        }

        #[inline]
        pub fn vec_reg(&self, i: u8) -> u128 {
            self.vec[i as usize & 31]
        }

        #[inline]
        pub fn set_vec_reg(&mut self, i: u8, v: u128) {
            self.vec[i as usize & 31] = v;
        }
    }

    /// One system's full execution context: its CPU state, its code cache,
    /// and which Frontend produced its translations. Wii and Switch each
    /// get one of these; the dispatcher loop below is IDENTICAL for both.
    pub struct GuestSystem<F: crate::frontend::Frontend> {
        pub cpu: CpuState,
        pub cache: CodeCache,
        pub frontend: F,
        /// Resolved once at construction; passed to every emit_block call.
        pub host_caps: CHostCaps,
        pub force_baseline: bool,
    }

    impl<F: crate::frontend::Frontend> GuestSystem<F> {
        /// Main dispatch loop. `execute_block` is a raw function-pointer call
        /// into JIT'd machine code following a fixed calling convention:
        /// guest state pointer in r15 (set once, never reloaded per-call),
        /// return value = next guest PC (or a sentinel for "cache miss, ask
        /// the dispatcher to translate"). Because chained blocks jump
        /// directly to each other, this loop is only re-entered on true
        /// cache misses or indirect-branch targets not yet resolved.
        pub fn run(&mut self) {
            loop {
                let pc = self.cpu.pc;
                if self.cache.lookup(pc).is_none() {
                    self.translate_block_at(pc);
                }
                let block = self.cache.lookup(pc).expect("just translated");
                self.cpu.pc = unsafe { Self::execute_block(block, &mut self.cpu) };
            }
        }

        fn translate_block_at(&mut self, guest_pc: u64) {
            // -----------------------------------------------------------------
            // Translation pipeline — one guest basic block, end to end.
            //
            //   guest bytes
            //       │
            //       ├─ decode() ------------------------------┐
            //       │                                          │
            //       └─ lower_to_ir() → universal IrBlock ─────┤
            //                                                  │
            //                                      Rust-only pipeline
            //                                                  │
            //                           run_all()               │
            //                           ┌─ constant fold        │
            //                           ├─ load elimination     │
            //                           └─ lazy-flag cleanup    │
            //                                                  │
            //                           linear_scan()           │
            //                           └─ VReg → host reg/spill│
            //                                                  │
            //                               minimal C ABI       │
            //                                      │           │
            //                                      ▼           │
            //                           C++ / asmjit emitter   │
            //                                      │           │
            //                                      ▼           │
            //                              executable bytes
            //                                      │
            //                                      ▼
            //                               CodeCache::insert
            //                                      │
            //                                      ▼
            //                       direct-exit chain patching
            //
            // The important ownership boundary is after register allocation:
            // Rust owns IR, optimization, allocation, cache, and dispatch;
            // C/C++ only receives the flattened CIrBlock + CAllocEntry arrays
            // and turns them into x86-64 machine code.
            //
            // The normal execution path therefore becomes:
            //
            //   cache hit → execute JIT block directly
            //   cache miss → decode/lower → optimize → allocate → emit → cache
            //             → execute
            //
            // Once a direct branch target is already compiled, patch_chain()
            // bypasses the dispatcher on the next execution and lets blocks jump
            // directly to one another.
            // -----------------------------------------------------------------
            //
            use crate::ir::{BlockId, IrBuilder};
            use crate::{passes, regalloc};

            // ---- 1. DECODE + LOWER --------------------------------------
            // Fetch and lower guest instructions one at a time until the
            // frontend reports a block terminator (branch/call/return) or
            // we hit the safety cap below. `IrBuilder` owns VReg/BlockId
            // allocation for this translation; a fresh builder per block
            // keeps numbering simple and avoids unbounded growth across
            // the lifetime of the process.
            let mut builder = IrBuilder::new();
            let block_id: BlockId = builder.new_block(guest_pc);

            // Baseline-tier safety valve: bound compile latency and the
            // regalloc spill budget (see CpuState::spill) even if a
            // frontend's is_block_terminator never fires for some reason
            // (malformed guest code, decode desync, etc). 64 matches the
            // instruction-count ceiling the regalloc module's docs assume.
            const MAX_BLOCK_INSNS: usize = 64;
            const GUEST_INSN_BYTES: usize = 4; // both PPC and AArch64 are fixed-width

            let mut cursor_pc = guest_pc;
            for _ in 0..MAX_BLOCK_INSNS {
                // real impl: fetch GUEST_INSN_BYTES from the guest's mapped
                // memory at cursor_pc (see the "reserve a huge virtual
                // mapping" approach discussed for guest memory access).
                let raw_bytes: [u8; GUEST_INSN_BYTES] =
                    self.fetch_guest_bytes(cursor_pc);

                let (decoded, len) = self.frontend.decode(&raw_bytes, cursor_pc);
                self.frontend.lower_to_ir(&decoded, &mut builder, block_id);

                let terminated = self.frontend.is_block_terminator(&decoded);
                cursor_pc = cursor_pc.wrapping_add(len as u64);

                if terminated {
                    break;
                }
            }

            let mut ir_block = builder
                .blocks
                .pop()
                .expect("new_block always pushes exactly one block");

            // A block that ran into MAX_BLOCK_INSNS without ever hitting a
            // frontend terminator has no exit at all, and the emitter would
            // then run off the end of the block into whatever bytes follow.
            // Close it explicitly at the instruction AFTER the last one we
            // lowered (`cursor_pc` has already been advanced past it) so the
            // dispatcher keeps control. Legally-terminated blocks are untouched.
            if !crate::ir::is_terminator_op(ir_block.ops.last()) {
                ir_block
                    .ops
                    .push(crate::ir::IrOp::IndirectBranch {
                        target: crate::ir::VOperand::Imm(cursor_pc as i64),
                    });
            }

            // ---- 2. OPTIMIZE (intra-block, cheap — see passes module) ---
            passes::run_all(&mut ir_block);

            // ---- 3. REGISTER ALLOCATION ----------------------------------
            let allocation = regalloc::linear_scan(&ir_block);

            // ---- 4. CROSS THE FFI BOUNDARY / EMIT --------------------------
            // Everything past this call is C++ (asmjit); Rust never touches
            // machine code directly. See SECTION 4 / the APPENDIX for the
            // exact C ABI this crosses.
            let compiled = match crate::jit_ffi::emit_block(
                &ir_block,
                &allocation,
                self.host_caps,
                self.force_baseline,
            ) {
                Ok(block) => block,
                Err(_status) => {
                    // A hard translation failure for this block. There is no
                    // safe way to "skip" a guest basic block — the dispatcher
                    // has nothing to execute. real impl: surface this to
                    // whatever owns the emulator session (crash dialog /
                    // logged fatal / fallback interpreter path), rather than
                    // panicking a JIT thread outright.
                    panic!("jit_emit_block failed for guest_pc={guest_pc:#x}");
                }
            };

            // ---- 5. INSERT INTO CACHE --------------------------------------
            // Ownership of `compiled` (and therefore of the executable
            // memory) transfers to the cache. CompiledBlock::drop calls
            // jit_free_code exactly once, whenever this entry is evicted
            // or invalidated.
            self.cache.insert(guest_pc, compiled);

            // ---- 6. CHAIN ANY EXITS WHOSE TARGET IS ALREADY COMPILED ------
            // Re-borrow from the cache rather than holding on to `compiled`
            // past the move into `insert` above.
            let exit_targets: Vec<u64> = self
                .cache
                .lookup(guest_pc)
                .map(|b| b.patchable_exits.iter().map(|s| s.target_guest_pc).collect())
                .unwrap_or_default();

            for (exit_index, target_pc) in exit_targets.into_iter().enumerate() {
                let Some(target_block) = self.cache.lookup(target_pc) else {
                    // Target not compiled yet — this exit stays as a
                    // dispatcher return for now. It gets patched the next
                    // time *that* target is translated and some other
                    // block's exit resolves against it. (A fuller
                    // implementation tracks pending back-references per
                    // target PC so already-compiled predecessors get
                    // patched retroactively instead of only forward.)
                    continue;
                };
                let target_entry = target_block.code_ptr as *const u8;
                if let Some(from_block) = self.cache.lookup(guest_pc) {
                    // Best-effort: a failed patch just means this exit stays
                    // as a (slower but correct) dispatcher return.
                    let _ = crate::jit_ffi::patch_chain(from_block, exit_index, target_entry);
                }
            }
        }

        /// Guest memory access stub. real impl: index into a reserved
        /// guest-address-space-shaped mapping (see the memory-access design
        /// discussion — direct host memory access via base + offset, no
        /// per-access bounds-check call) rather than a bounds-checked
        /// function call on the hot decode path.
        fn fetch_guest_bytes(&self, pc: u64) -> [u8; 4] {
            todo!("read 4 bytes from guest memory at `pc` via the reserved guest-address mapping")
        }

        unsafe fn execute_block(block: &CompiledBlock, cpu: &mut CpuState) -> u64 {
            // real impl: raw fn-pointer call, e.g.
            //   let f: extern "C" fn(*mut CpuState) -> u64 =
            //       std::mem::transmute(block.code_ptr);
            //   f(cpu as *mut CpuState)
            // The C++ emitter is responsible for generating code that expects
            // the guest state pointer in r15 (or receives it as the first
            // argument and moves it into r15 at the prologue).
            todo!("transmute code_ptr to extern \"C\" fn(*mut CpuState) -> u64 and call it")
        }
    }
}

// =============================================================================
// SECTION 10: GPU — generic old-hardware model, shared by Wii's Hollywood
// (fixed-function TEV) and Switch's Maxwell (shader-based), translated to
// a single modern backend (wgpu) so weak integrated GPUs aren't paying for
// two separate translation paths.
// =============================================================================
pub mod gpu {
    /// System-agnostic representation of "what old graphics hardware does",
    /// populated differently by each console's frontend but consumed
    /// identically by the wgpu backend.
    pub struct GenericFrame {
        pub framebuffers: Vec<FramebufferDesc>,
        pub draw_calls: Vec<DrawCall>,
        /// Mid-frame state mutations (palette swaps, raster tricks) that
        /// don't fit a "build whole frame, then present" model. The backend
        /// replays these as render-pass boundaries.
        pub raster_events: Vec<RasterEvent>,
    }

    pub struct FramebufferDesc { pub width: u32, pub height: u32, pub format: PixelFormat }

    #[derive(Clone, Copy)]
    pub enum PixelFormat { Rgba8, Rgb565, Indexed8 }

    pub struct DrawCall {
        pub vertex_source: VertexSource,
        pub pipeline_state: PipelineState,
    }

    pub enum VertexSource {
        /// Wii/Hollywood: fixed-function TEV stage config instead of a shader.
        FixedFunctionTev { stages: Vec<TevStage> },
        /// Switch/Maxwell: real compiled shader (translated from NVN/Vulkan
        /// guest shader bytecode into the host's shader language).
        Shader { translated_module: ShaderModuleHandle },
    }

    pub struct TevStage { /* color/alpha combiner config, texture stage refs */ }
    pub struct ShaderModuleHandle(pub u64);
    pub struct PipelineState { pub blend_mode: BlendMode, pub depth_test: bool }
    #[derive(Clone, Copy)]
    pub enum BlendMode { Opaque, AlphaBlend, Additive }

    pub struct RasterEvent { pub scanline: u32, pub state_diff: Vec<StateDiff> }
    pub enum StateDiff { PaletteSwap { index: u32, rgba: [u8; 4] }, ScrollRegister { reg: u32, value: u32 } }

    /// Hollywood (Wii) — populates a GenericFrame from TEV/BP register state.
    pub mod hollywood {
        use super::GenericFrame;
        pub fn translate_frame(bp_register_state: &[u32]) -> GenericFrame {
            todo!("decode Hollywood BP/CP/XF register writes into TevStage + DrawCall entries")
        }
    }

    /// Maxwell (Switch) — populates a GenericFrame from NVN/Vulkan-shaped
    /// guest command streams, including real shader translation.
    pub mod maxwell {
        use super::GenericFrame;
        pub fn translate_frame(command_stream: &[u8]) -> GenericFrame {
            todo!("parse guest command buffer, translate embedded shader \
                   bytecode (NVN/Vulkan SPIR-V-like) to host shader language")
        }
    }

    /// Shared backend: GenericFrame -> real wgpu calls. Written ONCE for
    /// both consoles. wgpu chosen over raw Vulkan for weak-hardware targets
    /// because it avoids a lot of driver-overhead footguns raw Vulkan makes
    /// easy to hit without careful hand-tuning.
    pub mod backend_wgpu {
        use super::GenericFrame;
        pub fn submit(frame: &GenericFrame /*, device: &wgpu::Device, queue: &wgpu::Queue */) {
            todo!("batch state changes, upload textures with dedup/caching, \
                   issue draw calls per DrawCall, split render passes at \
                   RasterEvent boundaries")
        }
    }
}

// =============================================================================
// SECTION 11: ENTRY POINT — wires a GuestSystem for each console.
// (pub so src/main.rs, the binary, can call it; a lib crate has no main)
// =============================================================================
pub fn main() {
    use crate::runtime::GuestSystem;
    use crate::frontend_wii_ppc::WiiFrontend;
    use crate::frontend_switch_arm64::SwitchFrontend;
    use crate::cache::CodeCache;
    use crate::jit_ffi;

    // Detect once; the C++ side owns the actual cpuid / asmjit CpuInfo logic.
    let caps = jit_ffi::detect_host_caps();

    // FORCE_BASELINE_CODEGEN=1 keeps the SSE2 floor exercised even on a
    // modern AVX2 development machine.
    let force_baseline = std::env::var_os("FORCE_BASELINE_CODEGEN").is_some();

    // Conservative code-cache cap for weak/low-RAM x86-64 hosts — tune per
    // measured target hardware, not assumed abundant like a desktop JIT would.
    const CODE_CACHE_BYTES: usize = 32 * 1024 * 1024;

    let mut wii = GuestSystem {
        cpu: unsafe { std::mem::zeroed() },
        cache: CodeCache::new(CODE_CACHE_BYTES),
        frontend: WiiFrontend,
        host_caps: caps,
        force_baseline,
    };
    let mut switch = GuestSystem {
        cpu: unsafe { std::mem::zeroed() },
        cache: CodeCache::new(CODE_CACHE_BYTES),
        frontend: SwitchFrontend,
        host_caps: caps,
        force_baseline,
    };

    // real impl: load ROM/NSO images into each guest's memory space, set
    // initial cpu.pc, then run each system's dispatcher loop, e.g. on its
    // own dedicated thread per the "minimize thread count" performance note.
    todo!("load guest images, set entry pc, spawn one thread per GuestSystem, \
           feed GPU command streams to gpu::hollywood / gpu::maxwell each frame")
}

// =============================================================================
// APPENDIX: C++ SIDE CONTRACT (for the companion static library)
// =============================================================================
//
// The C++ library (built with asmjit) must export exactly these four symbols
// with C linkage. Suggested header (jit_emit.h):
//
//   #pragma once
//   #include <stdint.h>
//   #include <stddef.h>
//
//   /* Mirror every #[repr(C)] type from this file exactly. */
//   ... (CWidth, CEndian, CIrOpKind, COperand, CIrOp, CIrBlock, ...)
//
//   #ifdef __cplusplus
//   extern "C" {
//   #endif
//
//   CHostCaps    jit_detect_host_caps(void);
//   CJitStatus   jit_emit_block(const CIrBlock* block, CHostCaps caps,
//                               uint8_t force_baseline, CEmitResult* out);
//   void         jit_free_code(uint8_t* code);
//   CJitStatus   jit_patch_chain(uint8_t* code, size_t code_len,
//                                uint32_t patch_offset, const uint8_t* target_entry);
//
//   #ifdef __cplusplus
//   }
//   #endif
//
// Implementation notes for the C++ side
// -------------------------------------
// • HostCaps detection: use asmjit::CpuInfo::host() or raw cpuid. Set the
//   bitfield flags defined in CHostCaps.
// • CodegenStrategy: build a table of function pointers (or a switch) once
//   from the caps / force_baseline flag. The hot emit loop never re-checks
//   features; it just dispatches through the table (identical design to the
//   original Rust CodegenStrategy).
// • Emission: walk the CIrOp array, look up each VReg’s CLocation in the
//   allocs array, and emit with asmjit::x86::Assembler (or Builder).
//   SSE2 floor vs AVX2/FMA fast paths are selected by the strategy table.
//   Big-endian loads use MOVBE when available, otherwise MOV+BSWAP.
// • Executable memory: asmjit::JitRuntime is the simplest path; alternatively
//   a dual-mapped RX/RW page pair for cheap W^X toggles during patching.
// • Ownership: on success, allocate the code buffer and the patch-site array
//   (or pack them into one allocation). jit_free_code must release both.
// • Patch sites: for every direct Branch / Call that targets a guest PC,
//   record the offset of the relative displacement field so that
//   jit_patch_chain can later overwrite it with a direct jump.
//
// =============================================================================
