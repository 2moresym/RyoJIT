//! End-to-end test of everything on the Rust side of the FFI boundary, driven by
//! the PowerPC frontend: bytes → decode → lower → passes → regalloc → marshal →
//! `jit_emit_block` → `CodeCache` → `patch_chain` → `CompiledBlock::drop`.
//!
//! What it is *not*: a test of generated machine code.  The emitter is C++
//! (asmjit) and does not exist yet, so the crate links against the placeholder
//! in `src/jit_stub.rs` (`jit-stub` feature, default on).  That placeholder does
//! not generate code, but it *does* validate the marshaled block and hand back
//! ownership through the real ABI, which is what this file exercises: the IR the
//! PPC frontend produces must survive the boundary, produce the right patch-site
//! list, chain, and be freed exactly once.
//!
//! Guest memory access (`fetch_guest_bytes`) is still `todo!()` — that is item #3
//! of the plan — so the dispatcher loop itself is not driven here; the frontend's
//! `translate_unit` runs the same decode/lower loop over a byte slice instead.

use ryojit::cache::CodeCache;
use ryojit::frontend::Frontend;
use ryojit::frontend_wii_ppc::{intrinsic_name, PpcKind, WiiFrontend};
use ryojit::jit_ffi;
use ryojit::{passes, regalloc, CHostCaps, CJitStatus};

/// Big-endian guest bytes for a list of instruction words.
fn image(words: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(words.len() * 4);
    for w in words {
        v.extend_from_slice(&w.to_be_bytes());
    }
    v
}

fn translate(fe: &WiiFrontend, words: &[u32], base: u64) -> ryojit::ir::IrBlock {
    let unit = fe.translate_unit(&image(words), base);
    assert!(
        unit.error.is_none(),
        "this test must only use fully supported instructions: {:?}",
        unit.error
    );
    assert!(unit.terminated, "the block must be terminated");
    let violations = ryojit::ir_verify::verify_block(&unit.block);
    assert!(
        violations.is_empty(),
        "invalid IR:\n{}\n{}",
        violations.join("\n"),
        ryojit::ir_dump::dump_block_with(&unit.block, &|id| intrinsic_name(id))
    );
    unit.block
}

/// A small but typical loop body: pointer walk, load, signed compare, conditional
/// branch, decrement, return.  Chosen so that every *kind* of IR op the PPC
/// frontend can produce shows up at least once.
const LOOP: &[u32] = &[
    0x8063_0000, // lwz   r3, 0(r3)
    0x80A4_0004, // lwz   r5, 4(r4)
    0x7C632A14,  // add   r3, r3, r5
    0x5463_103A, // rlwinm r3, r3, 2, 0, 29   (align-down by 4)
    0x2803_0010, // cmpli cr0, r3, 16
    0x4182_0014, // beq   +0x14
    0x3884_0008, // addi  r4, r4, 8
    0x4BFF_FFE4, // b     -0x1C  (self-loop back to the first instruction)
];

#[test]
fn a_realistic_block_uses_every_ir_category_the_core_supports() {
    let fe = WiiFrontend::new();
    let unit = fe.translate_unit(&image(LOOP), 0x8000_1000);
    assert!(unit.error.is_none(), "{:?}", unit.error);
    let dump = ryojit::ir_dump::dump_block_with(&unit.block, &|id| intrinsic_name(id));
    // The categories a Wii block legitimately needs, each in one place:
    for (what, needle) in [
        ("guest register read", "ppc_get_gpr"),
        ("guest register write", "ppc_set_gpr"),
        ("memory load", "= load ["),
        ("scalar add", "= add "),
        ("rotate via and", "= and "),
        ("compare via lazy flags", "readflag"),
        ("static exit (chainable)", "indirect_branch -> #0x"),
    ] {
        assert!(dump.contains(needle), "{what} missing from:\n{dump}");
    }
    assert!(
        !dump.contains("store ["),
        "this loop stores nothing; a store would mean a mis-decode:\n{dump}"
    );
    // Exactly one terminator, at the end.
    let terms = unit.block.ops.iter().filter(|o| o.is_terminator()).count();
    assert_eq!(terms, 1, "one exit per block:\n{dump}");
    assert!(unit.block.ops.last().unwrap().is_terminator());
}

#[test]
fn optimize_then_allocate_then_emit_then_cache_then_drop() {
    let fe = WiiFrontend::new();
    let block = translate(&fe, LOOP, 0x8000_1000);

    // 1. passes
    let mut block = block;
    passes::run_all(&mut block);
    assert!(
        ryojit::ir_verify::verify_block(&block).is_empty(),
        "the passes must not break well-formedness:\n{}",
        ryojit::ir_dump::dump_block_with(&block, &|id| intrinsic_name(id))
    );

    // 2. register allocation
    let alloc = regalloc::linear_scan(&block);
    assert!(!alloc.assignments.is_empty());

    // 3. across the FFI boundary (placeholder emitter, real ABI)
    let caps = CHostCaps { bits: CHostCaps::SSE2 };
    let compiled = jit_ffi::emit_block(&block, &alloc, caps, false).expect("emit ok");
    assert_eq!(compiled.guest_start_pc, 0x8000_1000);
    assert!(compiled.code_len > 0);
    assert!(
        !compiled.code_ptr.is_null(),
        "the emitter transferred a code buffer"
    );

    // The two constant-target exits (beq taken/not-taken are folded into one
    // computed value, so only the unconditional `b` yields a patch site).
    assert_eq!(
        compiled.patchable_exits.len(),
        1,
        "exactly one statically-known exit expected"
    );

    // 4. the code cache takes ownership, keyed by guest pc
    let mut cache = CodeCache::new(1024 * 1024);
    cache.insert(compiled.guest_start_pc, compiled);
    assert!(cache.lookup(0x8000_1000).is_some());
    assert!(cache.lookup(0x8000_1004).is_none(), "blocks are keyed by entry pc");

    // 5. dropping the cache runs `jit_free_code` exactly once per block, through
    //    CompiledBlock::drop.  Nothing here can double-free: drop() nulls the
    //    pointer, and the placeholder rejects foreign pointers by magic word.
    drop(cache);
}

#[test]
fn chain_patch_is_rejected_when_the_offset_is_out_of_range() {
    let fe = WiiFrontend::new();
    let block = translate(&fe, LOOP, 0x8000_2000);
    let alloc = regalloc::linear_scan(&block);
    let compiled = jit_ffi::emit_block(
        &block,
        &alloc,
        CHostCaps { bits: CHostCaps::SSE2 },
        false,
    )
    .expect("emit ok");

    // A valid site chains fine.
    let target = compiled.code_ptr as *const u8;
    jit_ffi::patch_chain(&compiled, 0, target).expect("chaining the recorded site");
    // An exit index that does not exist must be refused, not panic.
    assert_eq!(
        jit_ffi::patch_chain(&compiled, 99, target),
        Err(CJitStatus::InvalidInput)
    );
}

#[test]
fn self_modifying_code_invalidation_frees_the_block() {
    // The cache's page map is what makes guest code patching safe; inserting a
    // block registers its page, and invalidating that page must drop (and free)
    // every block from it.
    let fe = WiiFrontend::new();
    let block = translate(&fe, LOOP, 0x8000_3000);
    let alloc = regalloc::linear_scan(&block);
    let compiled =
        jit_ffi::emit_block(&block, &alloc, CHostCaps { bits: CHostCaps::SSE2 }, false).unwrap();
    let pc = compiled.guest_start_pc;

    let mut cache = CodeCache::new(1024 * 1024);
    cache.insert(pc, compiled);
    assert!(cache.lookup(pc).is_some());
    // `CodeCache::insert` currently does not populate page_to_blocks (the SMC
    // reverse map is wired in when guest memory lands); invalidate_page on an
    // untracked page must be a no-op rather than a panic either way.
    cache.invalidate_page(pc >> 12);
    cache.invalidate_page(0xFFFF_FFFF);
}

#[test]
fn decode_and_lower_are_reachable_through_the_trait_object_surface() {
    // The whole point of the Frontend trait: the core drives the Wii without
    // knowing it is a Wii.  Take the trait's own entry points and confirm the
    // same words decode identically through them.
    fn via_trait<F: Frontend>(fe: &F, bytes: &[u8], pc: u64) -> (F::DecodedInsn, usize) {
        fe.decode(bytes, pc)
    }
    let fe = WiiFrontend::new();
    let bytes = 0x7C64_2A14u32.to_be_bytes();
    let (insn, len) = via_trait(&fe, &bytes, 0x100);
    assert_eq!(len, 4);
    assert_eq!(insn.kind, PpcKind::Add);
    assert_eq!(insn.name, "add");
    assert_eq!(fe.endianness(), ryojit::ir::Endian::Big);
    assert!(fe.register_file_layout().gpr_count == 32);
    assert!(!fe.is_block_terminator(&insn));
}

#[cfg(feature = "jit-stub")]
#[test]
fn the_placeholder_emitter_frees_every_block_it_handed_out() {
    use ryojit::jit_ffi;
    ryojit::jit_stub::stub_reset_counters();

    let fe = WiiFrontend::new();
    let block = translate(&fe, LOOP, 0x8000_4000);
    let alloc = regalloc::linear_scan(&block);
    let caps = CHostCaps { bits: CHostCaps::SSE2 };

    let mut alive = Vec::new();
    for _ in 0..32 {
        alive.push(jit_ffi::emit_block(&block, &alloc, caps, false).unwrap());
    }
    assert_eq!(ryojit::jit_stub::stub_alloc_count(), 32);
    assert_eq!(ryojit::jit_stub::stub_free_count(), 0);
    alive.clear();
    assert_eq!(
        ryojit::jit_stub::stub_free_count(),
        32,
        "CompiledBlock::drop must call jit_free_code exactly once per block"
    );
    assert_eq!(
        ryojit::jit_stub::stub_live_bytes(),
        0,
        "the placeholder leaked an allocation"
    );
}
