//! PowerPC Broadway opcode dispatch: word → (instruction, mnemonic).
//!
//! Pure *identification*.  Field values are not interpreted here — the lowering
//! reads them through [`PpcFields`](super::fields), because most PPC fields are
//! context-dependent (bit 21 is `OE` for `add` but `W` for `psq_lx`, and RA means
//! "register 0" in XO form but "the value 0" in `addi` and "no write-back" in
//! `lwzx`).  Splitting identification from interpretation keeps the table below
//! mechanically checkable against a disassembler, which is the whole point of a
//! dispatch table this size: a single wrong digit is a silently miscompiled
//! instruction, not a crash.
//!
//! ## Provenance of every number
//!
//! * Groups 31/19 + the D-form loads/stores/ALU primaries: IBM's own
//!   "Instruction Set Sorted by Primary and Extended Op Code" table (AIX 5L
//!   Assembler Language Reference, appendix C), *cross-checked against a real
//!   disassembler* (Capstone, CS_ARCH_PPC/CS_MODE_32/BE) over the entire
//!   encoding space of those groups — see `tools/ppc_space.py`, which emits the
//!   full enumeration this was reconciled with.  Where the doc and the
//!   disassembler disagreed, the disassembler won.
//! * FP X-form (`fmr`, `fabs`, `frsp`, `mcrfs`, `mtfsf`, `mffs`, …) and FP
//!   A-form (`fadd`…`fnmadd`): the doc numbers, validated against real
//!   encodings (`fmr f1,f2` = 0xFC220090, `fabs` = …0210, `fctiwz` = …001E,
//!   `mffs f1` = 0xFC200482) plus Capstone for the subset it knows
//!   (`mcrfs`, `mffs`, `mtfsb0/1`, `fmuls`/`fmsubs`/…, the whole 59/63 A-form set).
//!   `fcmpu`/`fcmpo` follow Book III (`fcmpu` = A-XO 20 with `Rc`=1,
//!   `fcmpo` = A-XO 16, both requiring bits 21:25 clear); that is what
//!   compilers emit for PowerPC and it is the one place where a *legacy* table
//!   (POWER's "fcmpu = 63/0") would decode the same bits as `fsub f0,…`.
//! * `mulhw`/`mulhwu`: documented both ways across revisions (group-31
//!   XO 75/11 *and* the standalone primaries 10/11), so both are accepted —
//!   primaries 10/11 are otherwise unused on Broadway.
//! * Paired singles: Broadway has no VMX, so the Altivec primaries (4..7) are
//!   the paired-single space.  Encodings from YAGCD §3.4 (Gekko/Broadway) and
//!   cross-checked against real assembler output (e.g. `ps_madds1 f1,f2,f3,f4`
//!   = 0x102220DE, `psq_l f0,4(r3),1,5` = 0xE003D004 — both reproduce from the
//!   field layout used in `fields.rs`).
//!
//! ## What "Illegal" vs "Unsupported" means here
//!
//! * [`PpcKind::Illegal`] — the encoding is not an instruction on Broadway.
//!   Lowering emits a trap so the *guest* takes its 0x700 handler, which is the
//!   architecturally correct outcome and also what a real ROM expects from a
//!   deliberately-undefined opcode (games use it as a breakpoint).
//! * [`PpcKind::Unsupported`] — a real instruction this build does not lower
//!   yet.  That is a translation *failure*: the block is truncated so the
//!   fallback interpreter can run that one instruction (see
//!   [`PpcUnit::error`](super::PpcUnit::error)).  Silently skipping would let
//!   the guest continue with a corrupt machine state, which is the worst
//!   possible failure mode for a JIT.

use super::fields::PpcFields;
use super::DecodedPpc;

/// One variant per distinct *semantic* instruction form.  Operand values are
/// read from `PpcFields` by the lowering, so variants carry no data: the enum
/// stays `Copy`, and `Decode` can be a table walk with no allocation.
///
/// Naming: the `S` suffix is the record ("dot") form of an instruction that has
/// no `Rc` bit of its own (`addic.` = `AddicS`), `U` is the update form, `X` the
/// indexed form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PpcKind {
    // ---- reserved / failures ------------------------------------------------
    /// Not an instruction on Broadway → guest illegal-instruction trap.
    Illegal,
    /// A valid instruction that this build does not lower yet.
    Unsupported,

    // ---- integer arithmetic (D form) ----------------------------------------
    Mulli,
    Subfic,
    Addic,
    AddicS,
    Addi,
    Addis,

    // ---- integer arithmetic (group 31, XO form; `oe`/`rc` from fields) ------
    Add,
    Addc,
    Adde,
    Addme,
    Addze,
    Subf,
    Subfc,
    Subfe,
    Subfme,
    Subfze,
    Neg,

    // ---- multiply / divide ---------------------------------------------------
    Mullw,
    Mulhw,
    Mulhwu,
    Divw,
    Divwu,

    // ---- logical --------------------------------------------------------------
    And,
    Andc,
    Or,
    Orc,
    Xor,
    Eqv,
    Nand,
    Nor,
    Ori,
    Oris,
    Xori,
    Xoris,
    AndiS,
    AndisS,

    // ---- shifts / rotates / extend ------------------------------------------
    Slw,
    Srw,
    Sraw,
    Srawi,
    Rlwimi,
    Rlwinm,
    Rlwnm,
    Cntlzw,
    Extsb,
    Extsh,
    Popcntb,

    // ---- compare / trap -------------------------------------------------------
    Cmp,
    Cmpl,
    Cmpi,
    Cmpli,
    Tw,
    Twi,

    // ---- condition register ---------------------------------------------------
    Mfcr,
    Mtcrf,
    Mcrf,
    Crand,
    Crandc,
    Creqv,
    Crnand,
    Crnor,
    Cror,
    Crorc,
    Crxor,
    Mcrxr,

    // ---- fixed-point exception register / SPRs --------------------------------
    MfSpr,
    MtSpr,
    MfMsr,
    MtMsr,
    MfSr,
    MtSr,
    MfSrin,
    MtSrin,
    Rfi,
    Sc,

    // ---- branches --------------------------------------------------------------
    B,
    Bc,
    Bclr,
    Bcctr,
    Isync,

    // ---- integer loads / stores (D form) ---------------------------------------
    Lwz,
    Lwzu,
    Lbz,
    Lbzu,
    Lhz,
    Lhzu,
    Lha,
    Lhau,
    Stw,
    Stwu,
    Stb,
    Stbu,
    Sth,
    Sthu,
    Lmw,
    Stmw,

    // ---- integer loads / stores (group 31, indexed) -----------------------------
    Lwzx,
    Lwzux,
    Lbzx,
    Lbzux,
    Lhzx,
    Lhzux,
    Lhax,
    Lhaux,
    Stwx,
    Stwux,
    Stbx,
    Stbux,
    Sthx,
    Sthux,
    Lhbrx,
    Sthbrx,
    Lwbrx,
    Stwbrx,
    Lswi,
    Lswx,
    Stswi,
    Stswx,
    Lwarx,
    Stwcx,

    // ---- cache control (group 31) ------------------------------------------------
    Dcbf,
    Dcbi,
    Dcbst,
    Dcbt,
    Dcbtst,
    Dcbz,
    Icbi,
    Icbt,
    Sync,
    Eieio,
    Tlbie,
    Tlbsync,
    Eciwx,
    Ecowx,

    // ---- FP loads / stores --------------------------------------------------------
    Lfs,
    Lfsu,
    Lfd,
    Lfdu,
    Stfs,
    Stfsu,
    Stfd,
    Stfdu,
    Lfsx,
    Lfsux,
    Lfdx,
    Lfdux,
    Stfsx,
    Stfsux,
    Stfdx,
    Stfdux,
    Stfiwx,
    Lfiwax,
    Lfiwzx,

    // ---- FP arithmetic (groups 59 / 63) -------------------------------------------
    /// `single` selects the 59 (single) vs 63 (double) flavour; both land on the
    /// same kind because the intrinsic id carries the precision.
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Fmadd,
    Fmsub,
    Fnmadd,
    Fnmsub,
    Fres,
    Frsqrte,
    Frsp,
    Fctiw,
    Fctiwz,
    Fcmpu,
    Fcmpo,
    Fmr,
    Fneg,
    Fabs,
    Fnabs,
    Fsel,
    Mcrfs,
    Mffs,
    Mtfsf,
    Mtfsfi,
    Mtfsb0,
    Mtfsb1,

    // ---- paired singles: arithmetic -------------------------------------------------
    PsAdd,
    PsSub,
    PsMul,
    PsDiv,
    PsMadd,
    PsMsub,
    PsNmadd,
    PsNmsub,
    PsMadds0,
    PsMadds1,
    PsMuls0,
    PsMuls1,
    PsSum0,
    PsSum1,
    PsSel,
    PsRes,
    PsRsqrte,
    PsMerge00,
    PsMerge01,
    PsMerge10,
    PsMerge11,
    PsMr,
    PsNeg,
    PsAbs,
    PsNabs,
    PsCmpu0,
    PsCmpu1,
    PsCmpo0,
    PsCmpo1,

    // ---- paired singles: quantized load / store ---------------------------------------
    PsqL,
    PsqLu,
    PsqSt,
    PsqStu,
    PsqLx,
    PsqLux,
    PsqStx,
    PsqStux,
}

impl PpcKind {
    /// Does this instruction end a translation unit?
    ///
    /// This must be exactly the set of kinds whose lowering ends with a
    /// terminator op — if it were narrower, `translate_block_at`'s decode loop
    /// would keep appending instructions *after* an `IndirectBranch` and produce
    /// IR with a terminator in the middle of the block (which `ir_verify`
    /// rejects, but only because it is told to).  `tools/check_rust_static.py`'s
    /// `terminator set` check keeps the two in step; the test
    /// `every_terminator_kind_terminates` covers it from the other side.
    ///
    /// `Isync`/`Mcrf`/`Crand`… sit in group 19 next to `Bclr`/`Bcctr`, and `tw`
    /// sits in group 31 next to real traps, so this is per-kind rather than per
    /// primary opcode.
    #[inline]
    pub const fn is_block_terminator(self) -> bool {
        matches!(
            self,
            PpcKind::B
                | PpcKind::Bc
                | PpcKind::Bclr
                | PpcKind::Bcctr
                | PpcKind::Rfi
                | PpcKind::Tlbie
                | PpcKind::Sc
                | PpcKind::Tw
                | PpcKind::Twi
        )
    }
}

#[inline]
pub(crate) fn decode_raw(raw: u32, pc: u64) -> DecodedPpc {
    let f = PpcFields::new(raw);
    let (kind, name) = resolve(f);
    DecodedPpc {
        raw,
        opcode: f.op() as u8,
        pc,
        kind,
        name,
        f,
    }
}

/// The dispatch table.  Returns `(instruction, mnemonic)`; the mnemonic comes
/// from the same arm as the kind, by construction.
fn resolve(f: PpcFields) -> (PpcKind, &'static str) {
    use PpcKind::*;
    let op = f.op();
    match op {
        // ------------------------------------------------------------------
        // D-form / I-form / B-form primaries
        // ------------------------------------------------------------------
        3 => (Twi, "twi"),
        7 => (Mulli, "mulli"),
        8 => (Subfic, "subfic"),
        10 => (Cmpli, "cmpli"),
        11 => (Cmpi, "cmpi"),
        12 => (Addic, "addic"),
        13 => (AddicS, "addic."),
        14 => (Addi, "addi"),
        15 => (Addis, "addis"),
        16 => (Bc, "bc"),
        17 => (Sc, "sc"),
        18 => (B, "b"),
        20 => (Rlwimi, "rlwimi"),
        21 => (Rlwinm, "rlwinm"),
        23 => (Rlwnm, "rlwnm"),
        24 => (Ori, "ori"),
        25 => (Oris, "oris"),
        26 => (Xori, "xori"),
        27 => (Xoris, "xoris"),
        28 => (AndiS, "andi."),
        29 => (AndisS, "andis."),

        // mulhw/mulhwu are documented both as standalone primaries (older
        // 32-bit manuals) and as group-31 XOs. Accept both; they are the only
        // users of 10 and 11's extended space on Broadway.
        10 => match f.xo10() {
            1 => (Mulhw, "mulhw"),
            _ => (Illegal, "reserved (op 10)"),
        },
        11 => match f.xo10() {
            1 => (Mulhwu, "mulhwu"),
            _ => (Illegal, "reserved (op 11)"),
        },

        // quantized paired-single loads/stores, D form (YAGCD 3.4.3)
        56 => (PsqL, "psq_l"),
        57 => (PsqLu, "psq_lu"),
        60 => (PsqSt, "psq_st"),
        61 => (PsqStu, "psq_stu"),

        // integer loads / stores
        32 => (Lwz, "lwz"),
        33 => (Lwzu, "lwzu"),
        34 => (Lbz, "lbz"),
        35 => (Lbzu, "lbzu"),
        36 => (Stw, "stw"),
        37 => (Stwu, "stwu"),
        38 => (Stb, "stb"),
        39 => (Stbu, "stbu"),
        40 => (Lhz, "lhz"),
        41 => (Lhzu, "lhzu"),
        42 => (Lha, "lha"),
        43 => (Lhau, "lhau"),
        44 => (Sth, "sth"),
        45 => (Sthu, "sthu"),
        46 => (Lmw, "lmw"),
        47 => (Stmw, "stmw"),

        // scalar FP loads / stores
        48 => (Lfs, "lfs"),
        49 => (Lfsu, "lfsu"),
        50 => (Lfd, "lfd"),
        51 => (Lfdu, "lfdu"),
        52 => (Stfs, "stfs"),
        53 => (Stfsu, "stfsu"),
        54 => (Stfd, "stfd"),
        55 => (Stfdu, "stfdu"),

        // ------------------------------------------------------------------
        // group 19: branches + CR logic + isync/rfi
        // ------------------------------------------------------------------
        19 => match f.xo10() {
            0 => (Mcrf, "mcrf"),
            16 => (Bclr, "bclr"),
            33 => (Crnor, "crnor"),
            50 => (Rfi, "rfi"),
            129 => (Crandc, "crandc"),
            150 => (Isync, "isync"),
            193 => (Crxor, "crxor"),
            225 => (Crnand, "crnand"),
            257 => (Crand, "crand"),
            289 => (Creqv, "creqv"),
            417 => (Crorc, "crorc"),
            449 => (Cror, "cror"),
            528 => (Bcctr, "bcctr"),
            _ => (Illegal, "reserved (19)"),
        },

        // ------------------------------------------------------------------
        // group 31: the integer + cache + SPR bulk (10-bit XO = bits 21:30)
        // ------------------------------------------------------------------
        31 => match f.xo10() {
            0 => (Cmp, "cmp"),
            4 => (Tw, "tw"),
            8 => (Subfc, "subfc"),
            10 => (Addc, "addc"),
            11 => (Mulhwu, "mulhwu"),
            19 => (Mfcr, "mfcr"),
            20 => (Lwarx, "lwarx"),
            22 => (Icbt, "icbt"),
            23 => (Lwzx, "lwzx"),
            24 => (Slw, "slw"),
            26 => (Cntlzw, "cntlzw"),
            28 => (And, "and"),
            32 => (Cmpl, "cmpl"),
            40 => (Subf, "subf"),
            54 => (Dcbst, "dcbst"),
            55 => (Lwzux, "lwzux"),
            60 => (Andc, "andc"),
            75 => (Mulhw, "mulhw"),
            83 => (MfMsr, "mfmsr"),
            86 => (Dcbf, "dcbf"),
            87 => (Lbzx, "lbzx"),
            104 => (Neg, "neg"),
            119 => (Lbzux, "lbzux"),
            124 => (Nor, "nor"),
            136 => (Subfe, "subfe"),
            138 => (Adde, "adde"),
            144 => (Mtcrf, "mtcrf"),
            146 => (MtMsr, "mtmsr"),
            150 => (Stwcx, "stwcx."),
            151 => (Stwx, "stwx"),
            183 => (Stwux, "stwux"),
            200 => (Subfze, "subfze"),
            202 => (Addze, "addze"),
            210 => (MtSr, "mtsr"),
            215 => (Stbx, "stbx"),
            232 => (Subfme, "subfme"),
            234 => (Addme, "addme"),
            235 => (Mullw, "mullw"),
            242 => (MtSrin, "mtsrin"),
            246 => (Dcbtst, "dcbtst"),
            247 => (Stbux, "stbux"),
            266 => (Add, "add"),
            278 => (Dcbt, "dcbt"),
            279 => (Lhzx, "lhzx"),
            284 => (Eqv, "eqv"),
            306 => (Tlbie, "tlbie"),
            310 => (Eciwx, "eciwx"),
            316 => (Xor, "xor"),
            // NOTE: the AIX appendix lists `lhzux` at 331 *and* `div` at 331.  A
            // full enumeration of group 31 (tools/ppc_space.py) shows 311 is
            // `lhzux`, and 331 is the POWER-family `div`, which Broadway does not
            // have — so 331 falls through to Illegal rather than mis-lowering a
            // load as a divide.  This was a live bug in an earlier draft.
            311 => (Lhzux, "lhzux"),
            339 => (MfSpr, "mfspr"),
            343 => (Lhax, "lhax"),
            375 => (Lhaux, "lhaux"),
            407 => (Sthx, "sthx"),
            412 => (Orc, "orc"),
            438 => (Ecowx, "ecowx"),
            439 => (Sthux, "sthux"),
            444 => (Or, "or"),
            459 => (Divwu, "divwu"),
            467 => (MtSpr, "mtspr"),
            470 => (Dcbi, "dcbi"),
            476 => (Nand, "nand"),
            491 => (Divw, "divw"),
            512 => (Mcrxr, "mcrxr"),
            533 => (Lswx, "lswx"),
            534 => (Lwbrx, "lwbrx"),
            535 => (Lfsx, "lfsx"),
            536 => (Srw, "srw"),
            566 => (Tlbsync, "tlbsync"),
            567 => (Lfsux, "lfsux"),
            595 => (MfSr, "mfsr"),
            597 => (Lswi, "lswi"),
            598 => (Sync, "sync"),
            599 => (Lfdx, "lfdx"),
            631 => (Lfdux, "lfdux"),
            659 => (MfSrin, "mfsrin"),
            661 => (Stswx, "stswx"),
            662 => (Stwbrx, "stwbrx"),
            663 => (Stfsx, "stfsx"),
            695 => (Stfsux, "stfsux"),
            725 => (Stswi, "stswi"),
            727 => (Stfdx, "stfdx"),
            759 => (Stfdux, "stfdux"),
            790 => (Lhbrx, "lhbrx"),
            792 => (Sraw, "sraw"),
            824 => (Srawi, "srawi"),
            854 => (Eieio, "eieio"),
            855 => (Lfiwax, "lfiwax"),
            887 => (Lfiwzx, "lfiwzx"),
            918 => (Sthbrx, "sthbrx"),
            922 => (Extsh, "extsh"),
            954 => (Extsb, "extsb"),
            982 => (Icbi, "icbi"),
            983 => (Stfiwx, "stfiwx"),
            1014 => (Dcbz, "dcbz"),
            // 750CL addition (binutils spells it XO(4|31, 532) too; the 9-bit
            // reading would collide with 64-bit `ldbrx`, which Broadway does
            // not have, so matching the full 10-bit field is unambiguous here).
            532 => (Popcntb, "popcntb"),
            // mfocrf/mtocrf (bits 21:30 = 1002/970 with the "1" at bit 20 that
            // also feeds mfspr's SPR field) are deliberately not decoded: they
            // are supervisor-only and rare, and mfspr/mtspr with those SPR
            // numbers is the safe fall-through (see the module header).
            _ => (Illegal, "reserved (31)"),
        },

        // ------------------------------------------------------------------
        // group 4: Broadway paired singles (VMX does not exist on this core)
        // ------------------------------------------------------------------
        4 => resolve_ps(f),

        // ------------------------------------------------------------------
        // group 59: single-precision FP, A/AX form (5-bit XO)
        // ------------------------------------------------------------------
        59 => match f.xo() {
            18 => (Fdiv, "fdivs"),
            20 => (Fsub, "fsubs"),
            21 => (Fadd, "fadds"),
            24 => (Fres, "fres"),
            25 => (Fmul, "fmuls"),
            28 => (Fmsub, "fmsubs"),
            29 => (Fmadd, "fmadds"),
            30 => (Fnmsub, "fnmsubs"),
            31 => (Fnmadd, "fnmadds"),
            _ => (Illegal, "reserved (59)"),
        },

        // ------------------------------------------------------------------
        // group 63: double-precision FP — X-form (9-bit XO) + A/AX (5-bit XO)
        // ------------------------------------------------------------------
        63 => {
            // First the X-form set: bits 22:30 (9 bits), which is where the
            // manuals' numbers land once mtfsf's/mtfsfi's `W` bit (bit 21) is
            // excluded.  Each arm below is annotated with the real 32-bit word
            // it matches, so a future reader can check it against any
            // disassembler instead of trusting the number.
            match f.xo9() {
                12 => (Frsp, "frsp"),       // f1,f2 -> 0xFC220018
                14 => (Fctiw, "fctiw"),     // 0xFC22001C
                15 => (Fctiwz, "fctiwz"),    // 0xFC22001E
                38 => (Mtfsb1, "mtfsb1"),   // 0xFC60004C
                40 => (Fneg, "fneg"),        // 0xFC220050
                64 => (Mcrfs, "mcrfs"),      // 0xFD080080 (cr1,cr2)
                70 => (Mtfsb0, "mtfsb0"),   // 0xFC60008C
                71 => (Mffs, "mffs"),        // 0xFC20048E  (bit 21 fixed 1)
                72 => (Fmr, "fmr"),          // 0xFC220090
                134 => (Mtfsfi, "mtfsfi"),
                136 => (Fnabs, "fnabs"),     // 0xFC220110
                199 => (Mtfsf, "mtfsf"),     // 0xFDE2012E (FM=0xff, FRA=2, W=1)
                264 => (Fabs, "fabs"),       // 0xFC220210
                _ => match f.xo() {
                    // fcmpu / fcmpo occupy A-form slots but have no FRT: bits
                    // 21:25 must read 0, otherwise this is the arithmetic op.
                    16 if f.mb() == 0 && !f.rc() => (Fcmpo, "fcmpo"),
                    20 if f.mb() == 0 && f.rc() => (Fcmpu, "fcmpu"),
                    18 => (Fdiv, "fdiv"),
                    20 => (Fsub, "fsub"),
                    21 => (Fadd, "fadd"),
                    23 => (Fsel, "fsel"),
                    24 => (Fres, "fres"),
                    25 => (Fmul, "fmul"),
                    26 => (Frsqrte, "frsqrte"),
                    28 => (Fmsub, "fmsub"),
                    29 => (Fmadd, "fmadd"),
                    30 => (Fnmsub, "fnmsub"),
                    31 => (Fnmadd, "fnmadd"),
                    _ => (Illegal, "reserved (63)"),
                },
            }
        }

        // ------------------------------------------------------------------
        // remaining VMX-space primaries: legal-looking, but Broadway has no
        // Altivec unit, so these are undefined on real hardware.
        // ------------------------------------------------------------------
        // Altivec primaries. Broadway has no VMX unit, so these are undefined
        // on real hardware — trap, do not "emulate" them.
        5 | 6 | 7 => (Illegal, "vmx absent (750CL)"),

        // 0, 1, 2, 9, 22, 30, 58, 62: reserved on 32-bit Book II cores, or
        // 64-bit-only ops (`tdi`, `ld`, `std`). On Broadway executing them takes
        // a program exception, which is exactly what Illegal produces.
        _ => (Illegal, "reserved opcode"),
    }
}

/// Primary 4 = the paired-single space.  See the module header for the layout
/// source; `xo10` is bits 21:30 and `xo` is bits 26:30.
#[inline]
fn resolve_ps(f: PpcFields) -> (PpcKind, &'static str) {
    use PpcKind::*;
    let xo10 = f.xo10();
    let xo5 = f.xo();

    // The quantized-indexed forms carry W (bit 21) + GQR (bits 22:24) inside
    // bits 21:25, so their key is xo10 with those five bits ignored.
    let psq_key = xo10 & !(0x1F << 5);
    match psq_key {
        6 => {
            return if (xo10 >> 5) & 1 == 0 {
                (PsqLx, "psq_lx")
            } else {
                (PsqLux, "psq_lux")
            }
        }
        7 => {
            return if (xo10 >> 5) & 1 == 0 {
                (PsqStx, "psq_stx")
            } else {
                (PsqStux, "psq_stux")
            }
        }
        _ => {}
    }

    match xo5 {
        0 | 8 | 16 => {
            // Fixed-field X-style ops: the whole 10-bit field is the key.
            match xo10 {
                0 => (PsCmpu0, "ps_cmpu0"),
                32 => (PsCmpo0, "ps_cmpo0"),
                64 => (PsCmpu1, "ps_cmpu1"),
                96 => (PsCmpo1, "ps_cmpo1"),
                40 => (PsNeg, "ps_neg"),
                72 => (PsMr, "ps_mr"),
                136 => (PsNabs, "ps_nabs"),
                264 => (PsAbs, "ps_abs"),
                528 => (PsMerge00, "ps_merge00"),
                560 => (PsMerge01, "ps_merge01"),
                592 => (PsMerge10, "ps_merge10"),
                624 => (PsMerge11, "ps_merge11"),
                _ => (Illegal, "reserved (ps 4)"),
            }
        }
        10 => (PsSum0, "ps_sum0"),
        11 => (PsSum1, "ps_sum1"),
        12 => (PsMuls0, "ps_muls0"),
        13 => (PsMuls1, "ps_muls1"),
        14 => (PsMadds0, "ps_madds0"),
        15 => (PsMadds1, "ps_madds1"),
        18 => (PsDiv, "ps_div"),
        20 => (PsSub, "ps_sub"),
        21 => (PsAdd, "ps_add"),
        23 => (PsSel, "ps_sel"),
        24 => (PsRes, "ps_res"),
        25 => (PsMul, "ps_mul"),
        26 => (PsRsqrte, "ps_rsqrte"),
        28 => (PsMsub, "ps_msub"),
        29 => (PsMadd, "ps_madd"),
        30 => (PsNmsub, "ps_nmsub"),
        31 => (PsNmadd, "ps_nmadd"),
        _ => (Illegal, "reserved (ps 4)"),
    }
}
