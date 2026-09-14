//! ============================================================================
//! PLACEHOLDER EMITTER — **DO NOT SHIP THIS FILE**
//! ============================================================================
//!
//! This module exists for exactly one reason: the real code emitter is C++
//! (asmjit) and lives behind the four `extern "C"` symbols declared in
//! `lib.rs` SECTION 0, which do not exist yet (that is item #4 of the plan of
//! record).  Without *something* providing those symbols the crate does not
//! link, which would make the Rust half of the pipeline — decode → lower →
//! passes → regalloc → marshal → cache → `CompiledBlock::drop` — untestable.
//!
//! So this file is a stand-in with three properties, and deliberately no
//! fourth one (it does NOT generate code):
//!
//!   1. Same C ABI, same signatures, same ownership rules as the real thing,
//!      so the `extern "C"` contract cannot silently drift.
//!   2. It VALIDATES the marshaled `CIrBlock` the way the real emitter will
//!      have to: pointer/length consistency, operand-count bounds, and — the
//!      valuable one — *every* VReg referenced by an operand or destination
//!      must have a `CAllocEntry`.  A frontend that emits a VReg the
//!      allocator never saw therefore fails here, at the boundary, instead of
//!      becoming wrong-but-silent machine code later.
//!   3. It hands back a heap buffer that is freed *exactly once* through
//!      `jit_free_code`, with a patch-site array in the same allocation, so
//!      `CompiledBlock`'s `Drop` impl and the block-chaining patch path are
//!      exercised by the test suite.
//!
//! The returned "code" is `jmp rel32` placeholders (one per chainable exit)
//! followed by `ret`: never executed by any test, and *never* a substitute for
//! the real emitter.  `warn_once()` shouts about that at startup.
//!
//! Delete this file and the `jit-stub` Cargo feature once `jit/` is built and
//! linked (`--no-default-features` already refuses to link without it).
//! ============================================================================

use crate::{
    CAllocEntry, CEmitResult, COperandKind, CHostCaps, CIrBlock, CIrOp, CIrOpKind,
    CJitStatus, CPatchSite, CIR_MAX_OPERANDS,
};
use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Every allocation is `[Header][code][patch sites]`; `jit_free_code` gets the
/// code pointer back, so the header sits immediately before it. 64 bytes keeps
/// the code pointer 16-byte aligned, which is what the real emitter needs for
/// `movaps`-style state loads.
const HEADER_BYTES: usize = 64;
const STUB_ALIGN: usize = 16;
const MAGIC: usize = 0x5259_4f4a_5354_5542; // "RYOJSTUB"

#[repr(C)]
struct StubHeader {
    magic: usize,
    total: usize,
    num_patch_sites: u32,
    /// Byte offset from the code pointer to the placeholder patch-site array.
    patch_offset: u32,
    frees_seen: usize,
}

/// Allocation bookkeeping so the tests can assert the ownership contract:
/// `jit_free_code` must run exactly once per successful `jit_emit_block`.
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static FREE_COUNT: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static WARNED: AtomicBool = AtomicBool::new(false);

/// Test-only observability (this module is `pub` only under `jit-stub`).
#[doc(hidden)]
pub fn stub_alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

#[doc(hidden)]
pub fn stub_free_count() -> usize {
    FREE_COUNT.load(Ordering::Relaxed)
}

#[doc(hidden)]
pub fn stub_live_bytes() -> usize {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// Zero the counters. Tests that assert exact free counts must call this in a
/// serialised section, since the counters are process-wide.
#[doc(hidden)]
pub fn stub_reset_counters() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    FREE_COUNT.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
}

fn warn_once() {
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "RyoJIT: PLACEHOLDER jit emitter active (no machine code is \
             generated). Build jit/*.cpp and drop the `jit-stub` feature to \
             run guest code."
        );
    }
}

// -----------------------------------------------------------------------------
// The four exported symbols
// -----------------------------------------------------------------------------

/// The real detection lives on the C++ side (cpuid / asmjit::CpuInfo).  The
/// placeholder reports the SSE2 floor only, which is also the most useful thing
/// it *can* report: it keeps every test on the baseline path, which is the path
/// the "works on a 2011 laptop" design goal is about.
///
/// `FORCE_BASELINE_CODEGEN` is handled by the caller (see `GuestSystem`), so
/// the stub does not need to look at the environment.
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn jit_detect_host_caps() -> CHostCaps {
    CHostCaps {
        bits: CHostCaps::SSE2,
    }
}

/// Validate + "emit".  See the module header: this checks the marshaled block
/// and returns a heap buffer with patch sites, but generates no semantics.
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn jit_emit_block(
    block: *const CIrBlock,
    _caps: CHostCaps,
    _force_baseline: u8,
    out: *mut CEmitResult,
) -> CJitStatus {
    warn_once();

    if block.is_null() || out.is_null() {
        return CJitStatus::InvalidInput;
    }
    // On failure `out` must be left zeroed and no ownership transfers.
    let c_block = &*block;
    if (c_block.num_ops > 0 && c_block.ops.is_null())
        || (c_block.num_allocs > 0 && c_block.allocs.is_null())
    {
        return CJitStatus::InvalidInput;
    }

    let ops = if c_block.num_ops == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(c_block.ops, c_block.num_ops as usize)
    };
    let allocs = if c_block.num_allocs == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(c_block.allocs, c_block.num_allocs as usize)
    };

    if let Err(status) = validate(ops, allocs) {
        return status;
    }

    // ---- patch sites: every exit whose target is a compile-time constant ----
    // This mirrors the rule the real emitter documents in the APPENDIX: for a
    // direct Branch/Call/IndirectBranch with an immediate target, record the
    // offset of the relative displacement field so jit_patch_chain can rewrite
    // it. Non-constant (inline-cached) targets get no patch site.
    let mut sites: Vec<CPatchSite> = Vec::new();
    for op in ops {
        let target = match op.kind {
            CIrOpKind::Branch => Some(op.target_block as u64),
            CIrOpKind::Call | CIrOpKind::IndirectBranch => {
                if op.num_operands >= 1 && op.operands[0].kind == COperandKind::Imm {
                    Some(op.operands[0].imm as u64)
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(target) = target {
            sites.push(CPatchSite {
                // Filled in below once the layout is known; 5 bytes per exit
                // (E9 rel32) then a trailing C3, so site i sits at 5*i + 1.
                code_offset: 1 + 5 * sites.len() as u32,
                target_guest_pc: target,
            });
        }
    }

    let code_len = 5 * sites.len() + 1;
    let total = round_up(HEADER_BYTES + code_len, STUB_ALIGN) + 16 * sites.len();
    let layout = match Layout::from_size_align(total, STUB_ALIGN) {
        Ok(l) => l,
        Err(_) => return CJitStatus::OutOfMemory,
    };
    let base = alloc(layout);
    if base.is_null() {
        return CJitStatus::OutOfMemory;
    }

    let code = base.add(HEADER_BYTES);
    // jmp rel32 placeholders (target patched later) + ret.
    for i in 0..sites.len() {
        *code.add(5 * i) = 0xE9;
        for b in 1..5 {
            *code.add(5 * i + b) = 0;
        }
    }
    *code.add(5 * sites.len()) = 0xC3;

    let patch_ptr = if sites.is_empty() {
        std::ptr::null_mut()
    } else {
        code.add(round_up(code_len, STUB_ALIGN)) as *mut CPatchSite
    };
    if !patch_ptr.is_null() {
        for (i, site) in sites.iter().enumerate() {
            *patch_ptr.add(i) = *site;
        }
    }

    let header = &mut *(base as *mut StubHeader);
    header.magic = MAGIC;
    header.total = total;
    header.num_patch_sites = sites.len() as u32;
    header.patch_offset = round_up(code_len, STUB_ALIGN) as u32;
    header.frees_seen = 0;

    ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    LIVE_BYTES.fetch_add(total, Ordering::Relaxed);

    *out = CEmitResult {
        code,
        code_len,
        num_patch_sites: sites.len() as u32,
        patch_sites: patch_ptr,
    };
    CJitStatus::Ok
}

/// Free the buffer `jit_emit_block` handed back.  A foreign or already-freed
/// pointer is rejected by the magic check rather than double-freed — the real
/// emitter cannot offer that (it goes through asmjit's JitRuntime), which is
/// precisely why the *exact once* discipline is tested here instead.
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn jit_free_code(code: *mut u8) {
    if code.is_null() {
        return;
    }
    let base = code.sub(HEADER_BYTES);
    let header = &*(base as *const StubHeader);
    if header.magic != MAGIC {
        debug_assert!(
            false,
            "jit_free_code called with a pointer that did not come from jit_emit_block"
        );
        return;
    }
    // Read every header field into locals BEFORE poisoning: the header lives at
    // the start of the block being freed, so touching `header` after the
    // write_bytes below would read freed/poisoned memory.
    let total = header.total;
    let header_ptr = base as *const StubHeader;
    if total == 0 || (header_ptr as usize) % std::mem::align_of::<StubHeader>() != 0 {
        return;
    }
    let layout = match Layout::from_size_align(total, STUB_ALIGN) {
        Ok(l) => l,
        Err(_) => return,
    };
    // Poison so a use-after-free in a test shows up as garbage rather than the
    // old code bytes.
    std::ptr::write_bytes(base, 0xCC, total);
    dealloc(base, layout);

    FREE_COUNT.fetch_add(1, Ordering::Relaxed);
    LIVE_BYTES.fetch_sub(total, Ordering::Relaxed);
}

/// Rewrite a 5-byte `jmp rel32` placeholder.  The real emitter must toggle the
/// page to writable first (dual mapping or mprotect); the placeholder buffer is
/// ordinary heap memory, so it just writes.
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn jit_patch_chain(
    code: *mut u8,
    code_len: usize,
    patch_offset: u32,
    target_entry: *const u8,
) -> CJitStatus {
    if code.is_null() || target_entry.is_null() {
        return CJitStatus::InvalidInput;
    }
    let off = patch_offset as usize;
    if off.checked_add(5).map_or(true, |end| end > code_len) {
        return CJitStatus::InvalidInput;
    }
    if *code.add(off) != 0xE9 {
        // Double-patching or a stale offset: refuse rather than scribble over
        // already-emitted instructions.
        return CJitStatus::InvalidInput;
    }
    let from = code.add(off + 5) as usize;
    let rel = target_entry as usize as i64 - from as i64;
    if rel.unsigned_abs() > i32::MAX as u64 {
        return CJitStatus::UnsupportedOp;
    }
    let disp = (rel as i32).to_le_bytes();
    for i in 0..4 {
        *code.add(off + 1 + i) = disp[i];
    }
    CJitStatus::Ok
}

// -----------------------------------------------------------------------------
// Validation of the marshaled block (the part of this stub that earns its keep)
// -----------------------------------------------------------------------------

/// `E9`-only patch offsets must stay inside the buffer; a block with an absurd
/// number of exits means the frontend/marshaling disagree.
fn validate(ops: &[CIrOp], allocs: &[CAllocEntry]) -> Result<(), CJitStatus> {
    // VReg lookup: linear scan is fine here (blocks are capped at
    // MAX_BLOCK_INSNS instructions, and this never runs at execution time).
    let has_vreg = |v: u32| allocs.iter().any(|a| a.vreg == v);

    for op in ops {
        if op.num_operands as usize > CIR_MAX_OPERANDS {
            return Err(CJitStatus::InvalidInput);
        }
        if op.dst_valid > 1 || op.sign_ext > 1 {
            return Err(CJitStatus::InvalidInput);
        }
        // Vector ops are the only ones allowed to carry a lane count, and only
        // in {2,4,8} (paired singles = 2 x f32, NEON = 4 x f32 / 8 x f16 ...).
        if op.lanes != 0 && !matches!(op.lanes, 2 | 4 | 8 | 16) {
            return Err(CJitStatus::InvalidInput);
        }
        if !matches!(
            op.kind,
            CIrOpKind::Add
                | CIrOpKind::Sub
                | CIrOpKind::And
                | CIrOpKind::Or
                | CIrOpKind::Xor
                | CIrOpKind::Shl
                | CIrOpKind::Shr
                | CIrOpKind::Sar
                | CIrOpKind::Mul
                | CIrOpKind::VecAdd
                | CIrOpKind::VecMul
                | CIrOpKind::VecFma
                | CIrOpKind::Load
                | CIrOpKind::Store
                | CIrOpKind::Branch
                | CIrOpKind::IndirectBranch
                | CIrOpKind::Call
                | CIrOpKind::Return
                | CIrOpKind::SetFlags
                | CIrOpKind::ReadFlag
                | CIrOpKind::Intrinsic
        ) {
            return Err(CJitStatus::UnsupportedOp);
        }

        for i in 0..op.num_operands as usize {
            let operand = &op.operands[i];
            if operand.kind == COperandKind::Reg && !has_vreg(operand.reg) {
                // The emitter resolves every Reg operand through the alloc
                // table; a missing entry is a frontend/allocator bug, not
                // something codegen can paper over.
                return Err(CJitStatus::InvalidInput);
            }
        }
        if op.dst_valid == 1 && !has_vreg(op.dst) {
            return Err(CJitStatus::InvalidInput);
        }

        // Ops must agree with their operand count; the marshaling in
        // `ir_op_to_c` fixes these, so a mismatch means the Rust-side op shape
        // changed without the C-side payload rules changing with it.
        let expected = match op.kind {
            CIrOpKind::Load | CIrOpKind::IndirectBranch | CIrOpKind::Call => Some(1),
            CIrOpKind::Return | CIrOpKind::ReadFlag => Some(0),
            CIrOpKind::VecFma => Some(3),
            CIrOpKind::Branch => None, // 0 (uncond) or 1 (cond)
            CIrOpKind::Intrinsic => None,
            _ => Some(2),
        };
        if let Some(expected) = expected {
            if op.num_operands as usize != expected {
                return Err(CJitStatus::InvalidInput);
            }
        }
        match op.kind {
            CIrOpKind::Branch => {
                if op.num_operands > 1 {
                    return Err(CJitStatus::InvalidInput);
                }
            }
            CIrOpKind::Intrinsic => {
                if op.num_operands > CIR_MAX_OPERANDS as u8 {
                    return Err(CJitStatus::InvalidInput);
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[inline]
fn round_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}
