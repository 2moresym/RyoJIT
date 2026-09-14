//! RyoJIT binary.
//!
//! `ryojit` with no arguments runs the library entry point (ROM loading is still
//! `todo!()`).  `ryojit demo <hex…>` is the useful one today: it pushes raw
//! instruction words through everything that actually exists — decode → lower →
//! passes → register allocation → FFI marshal → (placeholder) emit — and prints
//! each stage, so the PPC frontend can be eyeballed or diffed without a ROM, a
//! C++ toolchain, or executable memory.
//!
//! ```sh
//! cargo run --release -- demo 7c642a14 80640008 41820008
//! ```
fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("demo") => demo(args.collect()),
        Some("help" | "-h" | "--help") => usage(0),
        Some(other) => {
            eprintln!("ryojit: unknown subcommand `{other}`\n");
            usage(2);
        }
        None => ryojit::main(),
    }
}

fn usage(code: i32) -> ! {
    eprintln!(
        "usage:\n  \
         ryojit                       run the emulator entry point (todo: ROM load)\n  \
         ryojit demo <hex word>…      translate PowerPC words and dump every stage\n\n\
         words are big-endian 32-bit hex, e.g. 7c642a14 = add r3,r4,r5"
    );
    std::process::exit(code)
}

fn demo(words: Vec<String>) -> ! {
    use ryojit::frontend_wii_ppc::{WiiFrontend, intrinsic_name};
    use ryojit::{passes, regalloc};

    let mut code: Vec<u8> = Vec::new();
    for w in &words {
        let v = u32::from_str_radix(w.trim().trim_start_matches("0x"), 16).map_err(|e| e.to_string()).unwrap_or_else(|_| {
            eprintln!("ryojit demo: {w:?} is not a 32-bit hex instruction word");
            std::process::exit(1)
        });
        code.extend_from_slice(&v.to_be_bytes());
    }
    if code.is_empty() {
        eprintln!("ryojit demo: no instruction words given");
        usage(1);
    }

    let fe = WiiFrontend::new();
    let ops_before: Vec<String> = Vec::new();
    let _ = ops_before;
    let unit = fe.translate_unit(&code, 0x1000);

    println!(
        "guest block @ {:#x}: {} instruction(s), terminated={}, error={:?}",
        unit.start_pc,
        unit.insn_count,
        unit.terminated,
        unit.error.as_ref().map(|e| e.to_string())
    );
    println!();
    println!("── decoded ─────────────────────────────────────────────────────");
    let ops_before;
    for i in 0..unit.insn_count {
        let word = u32::from_be_bytes([
            code[i * 4],
            code[i * 4 + 1],
            code[i * 4 + 2],
            code[i * 4 + 3],
        ]);
        let (kind, name) = fe.identify(word);
        println!(
            "  {:#010x}: {:>8}  {:<12} {:?}",
            unit.start_pc + (i * 4) as u64,
            format!("{word:08x}"),
            name,
            kind
        );
    }
    ops_before = unit.block.ops.len();
    println!();
    println!("── lowered IR ────────────────────────────────────────────────────");
    print!("{}", ryojit::ir_dump::dump_block_with(&unit.block, &|id| intrinsic_name(id)));

    let mut block = unit.block;
    passes::run_all(&mut block);
    println!();
    println!("── after passes ──────────────────────────────────────────────────");
    println!(
        "  {} ops, down from {} (constant folding, load elimination, dead flags)",
        block.ops.len(),
        ops_before
    );
    print!("{}", ryojit::ir_dump::dump_block_with(&block, &|id| intrinsic_name(id)));

    let alloc = regalloc::linear_scan(&block);
    println!();
    println!("── register allocation ───────────────────────────────────────────");
    let mut regs = 0usize;
    let mut spills = 0usize;
    for (_v, loc) in &alloc.assignments {
        match loc {
            regalloc::Location::Reg(_) => regs += 1,
            regalloc::Location::StateSpill(_) => spills += 1,
        }
    }
    println!("  {} live values: {regs} in host registers, {spills} spilled", alloc.assignments.len());

    println!();
    println!("── emit ──────────────────────────────────────────────────────────");
    let caps = ryojit::jit_ffi::detect_host_caps();
    println!("  host caps bits = {:#x} (SSE2 floor = {:#x})", caps.bits, ryojit::CHostCaps::SSE2);
    match ryojit::jit_ffi::emit_block(
        &block,
        &alloc,
        caps,
        std::env::var_os("FORCE_BASELINE_CODEGEN").is_some(),
    ) {
        Ok(compiled) => {
            println!(
                "  emitted {} bytes, {} patchable exit(s) [{}] — {} code buffer",
                compiled.code_len,
                compiled.patchable_exits.len(),
                compiled
                    .patchable_exits
                    .iter()
                    .map(|s| format!("{:#x}@{}", s.target_guest_pc, s.code_offset))
                    .collect::<Vec<_>>()
                    .join(", "),
                if cfg!(feature = "jit-stub") {
                    "PLACEHOLDER (no machine code; jit/*.cpp is todo)"
                } else {
                    "from the C++ emitter"
                }
            );
        }
        Err(status) => println!("  emit refused: {status:?}"),
    }
    std::process::exit(0)
}
