//! The PPC `Intrinsic` escape hatch: id table, effects, and what the emitter owes.
//!
//! [`IrOp::Intrinsic`](crate::ir::IrOp::Intrinsic) is the only place in this
//! frontend where something other than the universal IR appears, so this file is
//! deliberately the single source of truth for three things at once:
//!
//! 1. the numeric id the C++ emitter switches on,
//! 2. the `SideEffects` class the optimiser reasons with,
//! 3. the semantic contract (operand order, widths, what is read and written).
//!
//! Each is one `match` over `PpcIntr`, so id, effect class, arity and signature
//! cannot drift apart.  `lower::Ctx::intr` debug-asserts the operand count
//! against [`PpcIntr::arity`] and the destination against
//! [`PpcIntr::has_dst`], so a new call site that disagrees with the table fails
//! at the first test run rather than at the FFI boundary.
//!
//! ## Numbering rule
//!
//! Ids are grouped with room to grow (0x000 state, 0x100 floating point,
//! 0x200 integer helpers, 0x300 paired singles, 0x400 cache/MMU/exception).
//! **Never renumber or reuse an existing id**: this is a wire format shared with
//! `jit/*.cpp`, and a reused number becomes silently wrong machine code.
//! Append inside a group.
//!
//! ## Emitter contract (companion to `lib.rs`'s APPENDIX)
//!
//! * Every intrinsic is **inline-only**: the emitter expands it to host
//!   instructions, with no host call, no stack frame, and no clobber of any
//!   register the allocator may hand out (the 13 in `CHostGpr`, plus xmm0..7).
//!   That is what keeps the register allocator's model valid across intrinsics.
//!   An intrinsic that needs to call a C++ helper must first grow a declared
//!   clobber list *and* teach `regalloc` about it — never do it quietly.
//! * `operands[i]` is read-only; the destination is written exactly once.
//! * An *immediate* operand is a compile-time constant (register index, GQR
//!   number, mask, flag) the emitter may fold into an addressing mode; a
//!   *register* operand is a `CLocation` lookup like any other value.
//! * `jit_emit_block` returns `CJitStatus::UnsupportedOp` for an id it does not
//!   implement.  Never emit something approximate: the Rust side turns that
//!   status into a hard translation failure for the block, which is the
//!   recoverable direction.
//!
//! ## Where Broadway's real behaviour lives
//!
//! The `Fp*`/`Ps*` helpers are where the parts a fixed IR cannot express go:
//! rounding mode from `FPSCR.FPRF`, single-precision ops rounding to f32 and then
//! re-widening, `fres`/`frsqrte`'s implementation-defined estimates, NaN and
//! signed-zero edge cases, and `fctiwz`'s `2^52`-biased container bits (the
//! `fctiwz` + `stfiwx` integer-cast idiom only works if the helper produces
//! exactly those bits, not a plain integer).

use crate::ir::SideEffects;

/// Guest-misc register slots (`CpuState::guest_misc`), passed as index
/// immediates to `GetMisc`/`SetMisc`.
///
/// Slot numbers are wire constants — the C++ mirror and the debugger read them
/// by number — so they are explicit and append-only.  All slots are `u64`; the
/// guest-meaningful width is stated per slot and truncated on write by the
/// emitter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpcMisc {
    /// Condition register, 32 bits, PPC bit `n` at host bit `31-n`, so CR0 is
    /// the low nibble: `crf(j) = (cr >> (28 - 4*j)) & 0xF`.
    Cr = 0,
    /// Fixed-point exception register, 32 bits: SO = bit 31, OV = bit 30,
    /// CA = bit 29 (i.e. PPC bits 32, 33, 34 of the architected numbering).
    Xer = 1,
    /// Floating-point status & control register, 32 bits, same bit order as CR:
    /// FX = bit 31, FPRF = bits 22:19, SN = bit 12.
    Fpscr = 2,
    /// Link register (SPR 8).  32 bits on Broadway, stored zero-extended.
    Lr = 3,
    /// Count register (SPR 9).
    Ctr = 4,
    /// Processor scratch register (SPR 1008).  A general OS scratch register
    /// (nothing to do with the paired-single *operand* space), kept so a context
    /// switch cannot lose it.
    Pspr = 5,
    /// Time base low (SPR 268 / legacy 1004).  Only ever read through `GetTb`
    /// (volatile); this slot exists so the emulator can publish a value for
    /// debugging and for the fallback interpreter.
    Tbl = 6,
    /// Time base high (SPR 269 / legacy 1005).
    Tbu = 7,
    /// Decrementer (SPR 22).  Written via `SetDec` so the runtime can re-arm its
    /// virtual timer; read via `GetMisc` is *not* enough — use `GetTb`'s dec
    /// mode, which is volatile.
    Dec = 8,
    Sdr1 = 9,
    Srr0 = 10,
    Srr1 = 11,
    Dar = 12,
    Dsisr = 13,
    Hid0 = 14,
    Hid1 = 15,
    L2Cr = 16,
    Iabr = 17,
    Dabr = 18,
    Dccr = 19,
    Iccr = 20,
    /// Broadway/Gekko HID2 (SPR 920) — where Gekko keeps its PS-enable bits.
    Hid2 = 21,
    Sprg0 = 22,
    Sprg1 = 23,
    Sprg2 = 24,
    Sprg3 = 25,
    /// GQR0..GQR7 (SPR 896..903 on the 750CL family) — the quantization
    /// registers behind every `psq_*`.  Indexed as `Gqr0 + i`.
    Gqr0 = 26,
    /// Processor version register (SPR 287), read-only.
    Pvr = 34,
    /// Machine state register.  Stored so `mfmsr`/`mtmsr` round-trip and so the
    /// emulation layer can watch the MMU/cache enable bits; the JIT itself never
    /// interprets it.
    Msr = 35,
}

impl PpcMisc {
    /// Must stay ≥ highest slot + 1 and ≤ `crate::runtime::GUEST_MISC_SLOTS`.
    pub const SLOT_COUNT: usize = 40;

    /// SPR number of GQR0.  A named constant because it is the one number in
    /// this file worth re-verifying against a real ROM (every `psq_*` result
    /// depends on reading the right quantization register).
    pub const GQR_SPR_BASE: u32 = 896;

    #[inline]
    pub const fn slot(self) -> u64 {
        self as u64
    }

    /// Slot for GQR index `i` (0..=7).
    #[inline]
    pub const fn gqr(i: u32) -> u64 {
        Self::Gqr0.slot() + (i & 7) as u64
    }

    /// How an `mfspr`/`mtspr` on this number is handled.
    pub const fn access(spr: u32) -> SprAccess {
        use PpcMisc as M;
        match spr {
            1 => SprAccess::Slot(M::Xer.slot(), true),
            8 => SprAccess::Slot(M::Lr.slot(), true),
            9 => SprAccess::Slot(M::Ctr.slot(), true),
            18 => SprAccess::Slot(M::Dsisr.slot(), true),
            19 => SprAccess::Slot(M::Dar.slot(), true),
            22 => SprAccess::Dec,
            25 => SprAccess::Slot(M::Sdr1.slot(), true),
            26 => SprAccess::Slot(M::Srr0.slot(), true),
            27 => SprAccess::Slot(M::Srr1.slot(), true),
            268 | 1004 => SprAccess::Tb(0),
            269 | 1005 => SprAccess::Tb(1),
            272 => SprAccess::Slot(M::Sprg0.slot(), true),
            273 => SprAccess::Slot(M::Sprg1.slot(), true),
            274 => SprAccess::Slot(M::Sprg2.slot(), true),
            275 => SprAccess::Slot(M::Sprg3.slot(), true),
            287 => SprAccess::Slot(M::Pvr.slot(), false),
            // 750 block address tables.  Not modelled: the emulator maps the
            // whole guest address space, so what these program is a no-op by
            // construction.  Reads give 0, writes are consumed.
            528..=535 | 560..=567 => SprAccess::Ignore,
            896..=903 => SprAccess::Slot(
                M::Gqr0.slot() + (spr - M::GQR_SPR_BASE) as u64,
                true,
            ),
            920 => SprAccess::Slot(M::Hid2.slot(), true),
            1000 => SprAccess::Slot(M::Hid0.slot(), true),
            1008 => SprAccess::Slot(M::Pspr.slot(), true),
            1009 => SprAccess::Slot(M::Hid1.slot(), true),
            1010 => SprAccess::Slot(M::Iabr.slot(), true),
            1013 => SprAccess::Slot(M::Dabr.slot(), true),
            1017 => SprAccess::Slot(M::L2Cr.slot(), true),
            1018 => SprAccess::Slot(M::Dccr.slot(), true),
            1019 => SprAccess::Slot(M::Iccr.slot(), true),
            // Legacy aliases of the SPRGs.
            1020..=1023 => SprAccess::Slot(M::Sprg0.slot() + (spr - 1020) as u64, true),
            // 1001..1007 / 1011..1016 / 1021..: 750-family implementation-defined
            // registers (ICUCR, DPUCR, L2IHT, BAT-ish leftovers…).  OS code writes
            // them on boot; trapping there would stop a Wii title before it ever
            // reaches user code, so they are stored nowhere and read as 0.
            1001..=1007 | 1011..=1016 => SprAccess::Ignore,
            _ => SprAccess::Unknown,
        }
    }
}

/// How an SPR access is handled.  The distinction matters because an emulator
/// that ignores the MMU must still not lose OS state, and must not trap on a
/// register it has consciously decided not to model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SprAccess {
    /// Read/write a `guest_misc` slot.  `writable = false` marks a read-only
    /// register (PVR, TBL aliases): a write to one takes a program exception on
    /// real hardware; lowering treats it as a no-op so a title that pokes it
    /// during boot still runs (and the gap is documented here rather than in a
    /// bug report).
    Slot(u64, bool),
    /// Decrementer (SPR 22): reads are volatile, writes must reach the runtime.
    Dec,
    /// Time base.  0 = TBL, 1 = TBU.  Volatile read, writes are illegal.
    Tb(u32),
    /// Legal but unmodelled: reads 0, writes consumed.
    Ignore,
    /// Not a defined SPR on Broadway → guest illegal-instruction exception,
    /// which is what the guest does on real hardware too.
    Unknown,
}

/// `true` = the 750CL definition of the paired-single operand space: ps0 = the
/// low 32 bits of FPR `2n`, ps1 = the low 32 bits of FPR `2n+1`.  `false` =
/// Gekko's documentation, which puts both halves inside FPR `n`.
///
/// Both are implemented by `GetPsPair`/`SetPsPair` in the emitter; the frontend's
/// IR is identical either way, which is the point of the indirection — one
/// constant decides it for the whole frontend, and a ROM test only has to flip
/// one bit.  See the module header of `super` for why this is the remaining open
/// question in the FP path.
pub const PS_PAIR_SPLIT_OVER_TWO_FPRS: bool = true;

/// The PPC intrinsic table: every escape hatch this frontend can emit.
///
/// Operand lists are normative (the emitter parses them in this order):
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpcIntr {
    // ---- 0x000: guest state access ------------------------------------------
    /// `v = CpuState::gpr[i]` — operands: [index].
    GetGpr = 0x000,
    /// `CpuState::gpr[i] = r` — operands: [index, value].
    SetGpr = 0x001,
    /// `v = raw bits of CpuState::fpr[i]` — operands: [index].
    GetFpr = 0x002,
    /// `CpuState::fpr[i] = raw bits` — operands: [index, value].
    SetFpr = 0x003,
    /// `v = CpuState::guest_misc[slot]` — operands: [slot].
    GetMisc = 0x004,
    /// `CpuState::guest_misc[slot] = r`, truncated to the slot's guest width —
    /// operands: [slot, value].
    SetMisc = 0x005,
    /// `v = packed paired single ps[slot]`: ps0 in bits 31:0, ps1 in 63:32 —
    /// operands: [ps index 0..=31].  See [`PS_PAIR_SPLIT_OVER_TWO_FPRS`].
    GetPsPair = 0x006,
    /// `ps[index] = packed` — operands: [ps index, value].
    SetPsPair = 0x007,

    // ---- 0x100: floating point ----------------------------------------------
    /// `dst = fp_add(a, b, prec)` — operands: [a, b, prec].  Inputs/outputs are
    /// raw container bits.  `prec`: 0 = double, 1 = single (the 750CL definition
    /// of `fadds`: round both operands to f32, add in f32, re-widen).
    FpAdd = 0x100,
    FpSub = 0x101,
    FpMul = 0x102,
    /// `dst = a / b` — operands: [a, b, prec].
    FpDiv = 0x103,
    /// `dst = a*b + c` — operands: [a, b, c, prec].
    FpMadd = 0x105,
    /// `dst = a*b - c` — operands: [a, b, c, prec].
    FpMsub = 0x106,
    /// `dst = -(a*b + c)` — operands: [a, b, c, prec].
    FpNmadd = 0x107,
    /// `dst = -(a*b - c)` — operands: [a, b, c, prec].
    FpNmsub = 0x108,
    /// `dst = 1/a`, `fres` — operands: [a, prec].  Precision is
    /// implementation-defined; see the module header of `super`.
    FpRes = 0x109,
    /// `dst = 1/sqrt(a)`, `frsqrte` — operands: [a].
    FpRsqrt = 0x10A,
    /// `dst = bits(double(f32_round(a)))`, `frsp` — operands: [a].
    FpRoundToSingle = 0x10B,
    /// `dst = bits(double(round_toward_zero(a)))`, `fctiwz` — operands: [a].
    /// Must reproduce the hardware's `2^52`-biased representation so `stfiwx`
    /// on the result yields the integer.
    FpCtiwz = 0x10C,
    /// `dst = bits(double(round_per_fpscr(a)))`, `fctiw` — operands: [a].
    FpCtiw = 0x10D,
    /// `dst = bits(double(int))` — operands: [value, zero_extend].  Used by
    /// `lfiwax` / `lfiwzx`.
    FpFromInt = 0x10E,
    /// `dst = bits(double(f32_pattern))` — operands: [raw32].  `lfs`'s widening.
    FpWidenF32 = 0x10F,
    /// `dst = f32 bits (in the low 32) of container a, rounded per FPSCR.FPRF`
    /// — operands: [a].  `stfs`'s narrowing.
    FpNarrowF32 = 0x110,
    /// `dst = 4-bit CR field from comparing a with b` — operands:
    /// [a, b, ordered].  The emitter owns the FPSCR exception bits here.
    FpCmp = 0x111,
    /// `dst = 4-bit CR field derived from FPSCR`, `mcrfs` — operands: [bf, bfa].
    /// A helper rather than IR because the "sticky exception" fold is cheaper
    /// and less error-prone next to the FPSCR it reads.
    FpCrfFromFpscr = 0x112,
    /// FPSCR ← masked write of an FPR's low 32 bits (`mtfsf`), including the
    /// "clear FX and the exception bits whose field was written" rule —
    /// operands: [fm_mask, value].
    FpMtfsfMasked = 0x113,

    // ---- 0x200: integer helpers with no universal-IR encoding ---------------
    /// `dst = high 32 bits of a*b` — operands: [a, b, signed].  `mulhw`/`mulhwu`:
    /// there is no multiply-high in the IR, and widening first would need an
    /// extend pair plus a shift, which is strictly more work for the emitter.
    MulHigh = 0x200,
    /// `dst = a / b` — operands: [a, b, signed].  Broadway semantics: division by
    /// zero returns 0 (and -1 for the signed-min/-1 overflow case) and never
    /// traps.
    DivWord = 0x201,
    /// `dst = a % b` — operands: [a, b, signed].  750CL `modsw`/`moduw`.
    ModWord = 0x202,
    /// `dst = popcount(a's low 32 bits)`.  Id 0x203 is deliberately *this* and
    /// not something else: the 750CL's `popcntb` needs it, and `cntlzw` is
    /// lowered as `32 - popcntb(or-fold(v))`, which is what a 750CL does with its
    /// own popcntb.  operands: [value].
    Popcntb = 0x203,

    // ---- 0x300: paired singles ----------------------------------------------
    /// `dst = ps_div(a, b)` — operands: [a, b].  Two f32 lanes.
    PsDiv = 0x300,
    /// `dst = ps_reciprocal(a)`, `ps_res` — operands: [a].
    PsRes = 0x301,
    /// `dst = ps_rsqrt(a)`, `ps_rsqrte` — operands: [a].
    PsRsqrt = 0x302,
    /// `dst = per-lane select on a's sign bit`, `ps_sel` — operands: [a, b, c]:
    /// lane takes b where a < 0, c where a >= 0.
    PsSel = 0x303,
    /// `dst = 4-bit CR field from comparing one lane of a with b` — operands:
    /// [a, b, lane, ordered].
    PsCmp = 0x304,
    /// `dst = psq_load(ea, w, gqr)` — operands: [ea, w, gqr].  Returns the packed
    /// pair.  The read happens *inside* the helper, so the redundant-load pass
    /// cannot see it: that is the price of quantization+scaling being one host
    /// sequence rather than three IR ops.
    PsqLoad = 0x305,
    /// `psq_store(ea, val, w, gqr)` — operands: [ea, val, w, gqr].
    PsqStore = 0x306,

    // ---- 0x400: cache / MMU / exceptions / sync -----------------------------
    /// Guest exception.  operands: [vector, pc, cond] where vector is 0x200
    /// (trap), 0x700 (program) or 0xC00 (system call), `pc` is the faulting
    /// instruction's address, and `cond` is a 0/1 value: the emitter tests it and
    /// only raises when non-zero.  That is what makes `tw`/`twi` expressible
    /// without a conditional-branch IR op.  The emitter returns to the dispatcher
    /// with a raise marker and the runtime builds the exception frame
    /// (SRR0/SRR1/MSR) exactly as the interpreter would.
    Raise = 0x400,
    /// Cache-block operations, operands: [ea, hint] each.  One id apiece because their
    /// side effects genuinely differ: `dcbz` writes 32 bytes of guest memory,
    /// `dcbf`/`dcbst` may write back, `dcbt*` are hints, and `icbi`/`icbt` must
    /// reach the self-modifying-code invalidator.
    Dcbf = 0x401,
    Dcbi = 0x402,
    Dcbst = 0x403,
    Dcbt = 0x404,
    Dcbtst = 0x405,
    Dcbz = 0x406,
    Icbi = 0x407,
    Icbt = 0x408,
    /// `lwarx` — operands: [ea, woe].  Reads memory and takes the reservation.
    Lwarx = 0x409,
    /// `stwcx.` — operands: [ea, value]; `dst` = the 4-bit CR0 field value.
    Stwcx = 0x40A,
    /// `sync` / `eieio` / `isync` — no operands, no result.  `Sync` is a host
    /// fence; the other two are ordered no-ops on x86-64 but keep their own ids so
    /// device-timing hooks have somewhere to hang later.
    Sync = 0x40B,
    Eieio = 0x40C,
    Isync = 0x40D,
    /// SPR whose *semantics* the emulation layer owns (`mtmsr`, `tlbie`, …) —
    /// operands: [spr, value].
    SprSideEffect = 0x40E,
    /// `lswi`/`lswx`/`stswi`/`stswx`: the byte count *and* the set of GPRs
    /// touched are runtime values, which no fixed IR shape expresses.
    /// operands: [ea, count, flags, start_gpr] where flags packs
    /// (is_store | is_indexed | nb_is_zero).
    StringCopy = 0x40F,
    /// `lmw`/`stmw` — operands: [start_gpr, ea, is_store].  Count is
    /// `32 - start_gpr`, so no operand carries it.
    ListCopy = 0x410,
    /// Read a `guest_misc` slot whose value the *emulator* updates out from under
    /// the guest (time base halves, decrementer).  Same shape as `GetMisc` but
    /// `Volatile`, so two reads in one block are never merged and a read is never
    /// hoisted.  operands: [slot].
    ReadVolatileMisc = 0x411,
    /// `mtdec` — operand: [value].  The runtime re-arms its timer.
    SetDec = 0x412,
    /// Segment registers (`mtsr`/`mfsr`/`mtsrin`/`mfsrin`) — operands:
    /// [sr_number, value].  Not in `guest_misc`: 16 of them, and reads must not
    /// alias a guest load.
    SegReg = 0x413,
}

impl PpcIntr {
    /// Every variant, in id order.  Used by the dumper and by the consistency
    /// tests, so `ALL` is the one list that must be updated when a helper is
    /// added (`tests::intrinsic_table_is_consistent` checks `has_dst` and
    /// `arity` agree with the tables below for all of them).
    pub const ALL: &'static [PpcIntr] = &[
        PpcIntr::GetGpr,
        PpcIntr::SetGpr,
        PpcIntr::GetFpr,
        PpcIntr::SetFpr,
        PpcIntr::GetMisc,
        PpcIntr::SetMisc,
        PpcIntr::GetPsPair,
        PpcIntr::SetPsPair,
        PpcIntr::FpAdd,
        PpcIntr::FpSub,
        PpcIntr::FpMul,
        PpcIntr::FpDiv,
        PpcIntr::FpMadd,
        PpcIntr::FpMsub,
        PpcIntr::FpNmadd,
        PpcIntr::FpNmsub,
        PpcIntr::FpRes,
        PpcIntr::FpRsqrt,
        PpcIntr::FpRoundToSingle,
        PpcIntr::FpCtiwz,
        PpcIntr::FpCtiw,
        PpcIntr::FpFromInt,
        PpcIntr::FpWidenF32,
        PpcIntr::FpNarrowF32,
        PpcIntr::FpCmp,
        PpcIntr::FpCrfFromFpscr,
        PpcIntr::FpMtfsfMasked,
        PpcIntr::MulHigh,
        PpcIntr::DivWord,
        PpcIntr::ModWord,
        PpcIntr::Popcntb,
        PpcIntr::PsDiv,
        PpcIntr::PsRes,
        PpcIntr::PsRsqrt,
        PpcIntr::PsSel,
        PpcIntr::PsCmp,
        PpcIntr::PsqLoad,
        PpcIntr::PsqStore,
        PpcIntr::Raise,
        PpcIntr::Dcbf,
        PpcIntr::Dcbi,
        PpcIntr::Dcbst,
        PpcIntr::Dcbt,
        PpcIntr::Dcbtst,
        PpcIntr::Dcbz,
        PpcIntr::Icbi,
        PpcIntr::Icbt,
        PpcIntr::Lwarx,
        PpcIntr::Stwcx,
        PpcIntr::Sync,
        PpcIntr::Eieio,
        PpcIntr::Isync,
        PpcIntr::SprSideEffect,
        PpcIntr::StringCopy,
        PpcIntr::ListCopy,
        PpcIntr::ReadVolatileMisc,
        PpcIntr::SetDec,
        PpcIntr::SegReg,
    ];

    /// The wire id.
    #[inline]
    pub const fn id(self) -> u32 {
        self as u32
    }

    /// Name for dumps, panics and emitter traces.
    #[inline]
    pub const fn name(self) -> &'static str {
        use PpcIntr::*;
        match self {
            GetGpr => "ppc_get_gpr",
            SetGpr => "ppc_set_gpr",
            GetFpr => "ppc_get_fpr",
            SetFpr => "ppc_set_fpr",
            GetMisc => "ppc_get_misc",
            SetMisc => "ppc_set_misc",
            GetPsPair => "ppc_get_ps_pair",
            SetPsPair => "ppc_set_ps_pair",
            FpAdd => "ppc_fp_add",
            FpSub => "ppc_fp_sub",
            FpMul => "ppc_fp_mul",
            FpDiv => "ppc_fp_div",
            FpMadd => "ppc_fp_madd",
            FpMsub => "ppc_fp_msub",
            FpNmadd => "ppc_fp_nmadd",
            FpNmsub => "ppc_fp_nmsub",
            FpRes => "ppc_fp_res",
            FpRsqrt => "ppc_fp_rsqrt",
            FpRoundToSingle => "ppc_fp_round_to_single",
            FpCtiwz => "ppc_fp_ctiwz",
            FpCtiw => "ppc_fp_ctiw",
            FpFromInt => "ppc_fp_from_int",
            FpWidenF32 => "ppc_fp_widen_f32",
            FpNarrowF32 => "ppc_fp_narrow_f32",
            FpCmp => "ppc_fp_cmp",
            FpCrfFromFpscr => "ppc_fp_crf_from_fpscr",
            FpMtfsfMasked => "ppc_fp_mtfsf_masked",
            MulHigh => "ppc_mul_high",
            DivWord => "ppc_div_word",
            ModWord => "ppc_mod_word",
            Popcntb => "ppc_popcntb",
            PsDiv => "ppc_ps_div",
            PsRes => "ppc_ps_res",
            PsRsqrt => "ppc_ps_rsqrt",
            PsSel => "ppc_ps_sel",
            PsCmp => "ppc_ps_cmp",
            PsqLoad => "ppc_psq_load",
            PsqStore => "ppc_psq_store",
            Raise => "ppc_raise",
            Dcbf => "ppc_dcbf",
            Dcbi => "ppc_dcbi",
            Dcbst => "ppc_dcbst",
            Dcbt => "ppc_dcbt",
            Dcbtst => "ppc_dcbtst",
            Dcbz => "ppc_dcbz",
            Icbi => "ppc_icbi",
            Icbt => "ppc_icbt",
            Lwarx => "ppc_lwarx",
            Stwcx => "ppc_stwcx",
            Sync => "ppc_sync",
            Eieio => "ppc_eieio",
            Isync => "ppc_isync",
            SprSideEffect => "ppc_spr_side_effect",
            StringCopy => "ppc_string_copy",
            ListCopy => "ppc_list_copy",
            ReadVolatileMisc => "ppc_read_volatile_misc",
            SetDec => "ppc_set_dec",
            SegReg => "ppc_seg_reg",
        }
    }

    /// Side effects, as the optimiser sees them: **guest memory only**, since
    /// that is all `redundant_load_elimination` consults.  `Pure` therefore means
    /// "generates no guest-memory traffic", *not* "has no effect": the
    /// register-file helpers obviously mutate guest state, and program order
    /// (which this pipeline never changes) is what keeps them correct.
    #[inline]
    pub const fn effects(self) -> SideEffects {
        use PpcIntr::*;
        use SideEffects::*;
        match self {
            GetGpr | SetGpr | GetFpr | SetFpr | GetMisc | SetMisc | GetPsPair | SetPsPair => Pure,
            FpAdd
            | FpSub
            | FpMul
            | FpDiv
            | FpMadd
            | FpMsub
            | FpNmadd
            | FpNmsub
            | FpRes
            | FpRsqrt
            | FpRoundToSingle
            | FpCtiwz
            | FpCtiw
            | FpFromInt
            | FpWidenF32
            | FpNarrowF32
            | FpCmp
            | MulHigh
            | DivWord
            | ModWord
            | Popcntb
            | PsDiv
            | PsRes
            | PsRsqrt
            | PsSel
            | PsCmp => Pure,

            // Consume/produce FPSCR (and CR): no *memory* traffic, but marked so
            // that a future pass which reasons about state reads can find them
            // without re-deriving which helpers touch FPSCR.
            FpCrfFromFpscr | FpMtfsfMasked => ReadsMem,

            PsqLoad | Lwarx | Dcbt | Dcbtst => ReadsMem,
            PsqStore | Dcbf | Dcbi | Dcbst | Icbi | Icbt => WritesMem,
            Dcbz | Stwcx | StringCopy | ListCopy => ReadsAndWritesMem,

            // Never reorder, never merge two reads of: fences, exceptions, the
            // time base and the decrementer.
            Sync | Eieio | Isync | Raise | SprSideEffect | SegReg | SetDec
            | ReadVolatileMisc => Volatile,
        }
    }

    /// Fixed operand count, exactly as documented on each variant.
    #[inline]
    pub const fn arity(self) -> usize {
        use PpcIntr::*;
        match self {
            Sync | Eieio | Isync => 0,
            GetGpr | GetFpr | GetMisc | GetPsPair | FpRoundToSingle | FpCtiwz | FpCtiw | FpWidenF32
            | FpNarrowF32 | FpRsqrt | PsRes | PsRsqrt | ReadVolatileMisc | SetDec
            | Popcntb => 1,
            SetGpr | SetFpr | SetMisc | SetPsPair | FpFromInt | FpRes | FpCrfFromFpscr
            | FpMtfsfMasked | Lwarx | Stwcx | PsDiv | SprSideEffect | SegReg => 2,
            // Cache-block ops carry the hint field alongside the EA so a later
            // timing model can use it without another IR change.
            Dcbf | Dcbi | Dcbst | Dcbt | Dcbtst | Dcbz | Icbi | Icbt => 2,
            // MulHigh/DivWord/ModWord carry their signedness flag as an operand,
            // and PsqLoad carries W plus the GQR index — see each variant's doc.
            FpAdd | FpSub | FpMul | FpDiv | PsSel | FpCmp | Raise | ListCopy | PsqLoad => 3,
            // MulHigh/DivWord/ModWord carry their signedness flag as an operand.
            MulHigh | DivWord | ModWord => 3,
            FpMadd | FpMsub | FpNmadd | FpNmsub | PsCmp | PsqStore | StringCopy => 4,
        }
    }

    /// Does this intrinsic define a value?
    #[inline]
    pub const fn has_dst(self) -> bool {
        use PpcIntr::*;
        !matches!(
            self,
            SetGpr
                | SetFpr
                | SetMisc
                | SetPsPair
                | PsqStore
                | Dcbf
                | Dcbi
                | Dcbst
                | Dcbt
                | Dcbtst
                | Dcbz
                | Icbi
                | Icbt
                | Sync
                | Eieio
                | Isync
                | Raise
                | SprSideEffect
                | SegReg
                | SetDec
                | StringCopy
                | ListCopy
        )
    }
}

/// Resolve an intrinsic id for the IR dumper.
pub fn intrinsic_name(id: u32) -> Option<&'static str> {
    PpcIntr::ALL.iter().find(|i| i.id() == id).map(|i| i.name())
}
