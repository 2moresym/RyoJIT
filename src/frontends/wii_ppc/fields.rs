//! Raw instruction-field views for PowerPC words.
//!
//! PowerPC numbers bits from the *most significant* end (bit 0 = sign bit of the
//! word).  Every accessor here takes/returns PPC bit numbers in the comments and
//! extracts with one shared helper, so the field tables in `decode.rs` /
//! `lower.rs` can be read side by side with the MPC750CL user manual.  The
//! alternative (hand-written `>> n` per field) is how these decoders usually
//! acquire an off-by-one that survives for weeks.

/// Extract PPC bits `first..=last` (inclusive, 0 = MSB) as a right-aligned u32.
#[inline]
pub(crate) fn bits(raw: u32, first: u32, last: u32) -> u32 {
    let width = last - first + 1;
    let shift = 31 - last;
    // u64 intermediate so a 32-bit field does not overflow the mask.
    ((raw as u64 >> shift) & ((1u64 << width) - 1)) as u32
}

/// Same, sign-extended from the field width to i64 (PPC's `SI`/`BD`/`LI`/`D`
/// fields are signed).
#[inline]
pub(crate) fn bits_se(raw: u32, first: u32, last: u32) -> i64 {
    let width = last - first + 1;
    let unsigned = bits(raw, first, last) as u64;
    let sign = 1u64 << (width - 1);
    ((unsigned ^ sign).wrapping_sub(sign)) as i64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PpcFields {
    pub raw: u32,
}

impl PpcFields {
    #[inline]
    pub const fn new(raw: u32) -> Self {
        PpcFields { raw }
    }

    // ---- primary opcode + the three extended-opcode widths ------------------
    //
    // The user manual uses a different extended-field width per format, and the
    // tables this file was transcribed from mix them, so it is stated once here:
    //   5-bit  XO = bits 26:30  — X / XO / M / A (FP) / B forms: `lwzx`, `ori`,
    //                            `fadd`, `crand`, …
    //   9-bit  XO = bits 22:30  — FP X-form: `fmr`(72), `fabs`(264), `frsp`(12),
    //                            `mcrfs`(64), `mtfsf`(711), `mffs`(583), …
    //   10-bit XO = bits 21:30  — the integer group-31 "XO" format (so the OE
    //                            bit is part of the number: `add` = 266,
    //                            `addo` = 778), group 19 branches (`bclr` = 16),
    //                            the rotate class, `popcntb` (532), and the
    //                            Broadway paired-single ops (`ps_merge00` = 528).
    // All three are right-aligned by `bits()`, i.e. the *last* listed bit is the
    // field's LSB — which is also how the numbers above read in the manuals.
    #[inline]
    pub fn op(self) -> u32 {
        bits(self.raw, 0, 5)
    }
    #[inline]
    pub fn xo(self) -> u32 {
        bits(self.raw, 26, 30)
    }
    #[inline]
    pub fn xo9(self) -> u32 {
        bits(self.raw, 22, 30)
    }
    #[inline]
    pub fn xo10(self) -> u32 {
        bits(self.raw, 21, 30)
    }
    /// Record bit (bit 31) — the "." forms.
    #[inline]
    pub fn rc(self) -> bool {
        bits(self.raw, 31, 31) != 0
    }
    /// Overflow-enable bit (bit 21) — the "o" forms.
    #[inline]
    pub fn oe(self) -> bool {
        bits(self.raw, 21, 21) != 0
    }

    // ---- integer registers --------------------------------------------------
    /// RT / FRt, bits 6:10.
    #[inline]
    pub fn rt(self) -> u8 {
        bits(self.raw, 6, 10) as u8
    }
    /// RA, bits 11:15.
    #[inline]
    pub fn ra(self) -> u8 {
        bits(self.raw, 11, 15) as u8
    }
    /// RB / SH, bits 16:20.
    #[inline]
    pub fn rb(self) -> u8 {
        bits(self.raw, 16, 20) as u8
    }
    /// Shift amount, same field as RB.
    #[inline]
    pub fn sh(self) -> u32 {
        bits(self.raw, 16, 20)
    }
    /// Mask begin, bits 21:25.
    #[inline]
    pub fn mb(self) -> u32 {
        bits(self.raw, 21, 25)
    }
    /// Mask end, bits 26:30.
    #[inline]
    pub fn me(self) -> u32 {
        bits(self.raw, 26, 30)
    }

    // ---- immediates ---------------------------------------------------------
    /// SI, bits 16:31, sign-extended.
    #[inline]
    pub fn si(self) -> i64 {
        bits_se(self.raw, 16, 31)
    }
    /// UI, bits 16:31, zero-extended.
    #[inline]
    pub fn ui(self) -> u32 {
        bits(self.raw, 16, 31)
    }
    /// BD, bits 16:29 (branch displacement, *not* yet scaled by 4).
    #[inline]
    pub fn bd(self) -> i64 {
        bits_se(self.raw, 16, 29)
    }
    /// LI, bits 6:29 (absolute branch, *not* yet scaled by 4).
    #[inline]
    pub fn li(self) -> i64 {
        bits_se(self.raw, 6, 29)
    }
    #[inline]
    pub fn aa(self) -> bool {
        bits(self.raw, 30, 30) != 0
    }
    #[inline]
    pub fn lk(self) -> bool {
        bits(self.raw, 31, 31) != 0
    }

    // ---- branch / condition -------------------------------------------------
    #[inline]
    pub fn bo(self) -> u32 {
        bits(self.raw, 6, 10)
    }
    #[inline]
    pub fn bi(self) -> u32 {
        bits(self.raw, 11, 15)
    }
    /// CR field written by cmp/cmpx/crf ops, bits 6:8.
    #[inline]
    pub fn bf(self) -> u32 {
        bits(self.raw, 6, 8)
    }
    /// CR field read by mcrf/mcrfs, bits 11:13.
    #[inline]
    pub fn bfa(self) -> u32 {
        bits(self.raw, 11, 13)
    }
    /// mtcrf field-load mask, bits 12:19 — 8 bits, one per CR field, with
    /// bit 12 = CRF0 (so CRF `j` is selected by `(flm >> (7 - j)) & 1`).
    /// Capstone confirms this placement: `mtcrf 0xf, r2` = 0x7C40F120.
    #[inline]
    pub fn flm(self) -> u32 {
        bits(self.raw, 12, 19)
    }

    // ---- floating point -----------------------------------------------------
    #[inline]
    pub fn frt(self) -> u8 {
        self.rt()
    }
    #[inline]
    pub fn fra(self) -> u8 {
        self.ra()
    }
    #[inline]
    pub fn frb(self) -> u8 {
        self.rb()
    }
    /// FRC for the AX form (fmadd/fmsub/fnmadd/fnmsub): only bits 21:23 are
    /// encoded, the low two index bits are implied zero — hence the register
    /// number is always a multiple of 4.
    #[inline]
    pub fn frc_ax(self) -> u8 {
        ((bits(self.raw, 21, 23) << 2) & 0x1F) as u8
    }
    /// FRC as a full 5-bit field.  Used by the Broadway/Gekko *paired-single*
    /// AX ops (ps_madd, ps_sel, ps_sum*, …), which encode all five bits —
    /// YAGCD 3.4.2 writes them as `DDDDD AAAAA BBBBB CCCCC …`.
    #[inline]
    pub fn frc_full(self) -> u8 {
        self.mb() as u8
    }
    /// mtfsf field mask, bits 12:19 (bit 12 = FX) — same field position and
    /// order as `mtcrf`'s FLM.
    #[inline]
    pub fn fm(self) -> u32 {
        bits(self.raw, 12, 19)
    }
    /// mtfsf "clear FX" bit, bit 20.
    #[inline]
    pub fn fxm(self) -> bool {
        bits(self.raw, 20, 20) != 0
    }
    /// mtfsf/mtfsfi "full precision" bit, bit 21.
    #[inline]
    pub fn w_fp(self) -> bool {
        bits(self.raw, 21, 21) != 0
    }
    /// mtfsfi update value, bits 15:18.
    #[inline]
    pub fn mtfsfi_u(self) -> u32 {
        bits(self.raw, 15, 18)
    }
    /// mtfsfi writes the *FPSCR* field whose index is at bits 9:11 — note this
    /// is not the `BF` position used by `mcrfs`/`mtfsb0`; mtfsfi's field index
    /// sits one 3-bit group lower in the word.
    #[inline]
    pub fn mtfsfi_bf(self) -> u32 {
        bits(self.raw, 9, 11)
    }

    // ---- SPR / cache / string ops ------------------------------------------
    /// XFX `SPR` field, bits 11:20 — stored *swapped* (low 5 bits in 16:20).
    #[inline]
    pub fn spr(self) -> u32 {
        (bits(self.raw, 16, 20) << 5) | bits(self.raw, 11, 15)
    }
    /// `l*arx`/`st*cx.` "write-through / order-essential" hint, bit 21.
    #[inline]
    pub fn woe(self) -> bool {
        bits(self.raw, 21, 21) != 0
    }
    /// `dcbt`/`dcbtst` stream hint, bits 21:25.
    #[inline]
    pub fn dcrn(self) -> u32 {
        bits(self.raw, 21, 25)
    }
    /// `tw`/`twi` "to" test mask, bits 6:10.
    #[inline]
    pub fn to(self) -> u32 {
        bits(self.raw, 6, 10)
    }
    /// `lswi`/`stswi` byte count, bits 16:20 (0 means 32).
    #[inline]
    pub fn nb(self) -> u32 {
        bits(self.raw, 16, 20)
    }
    /// `mtsr`/`mfsr` segment number, bits 16:19.
    #[inline]
    pub fn sr(self) -> u32 {
        bits(self.raw, 16, 19)
    }
    /// `tlbie` TLB selector, bits 16:20 (reserved on 750CL, must be 0/1).
    #[inline]
    pub fn tlb_sel(self) -> u32 {
        bits(self.raw, 16, 20)
    }
    /// `cmp`/`cmpli` L bit (0 = 32-bit compare, 1 = 64-bit), bit 10.
    #[inline]
    pub fn l_width(self) -> bool {
        bits(self.raw, 10, 10) != 0
    }
    /// `icbt` L (lock) bit, bit 21.
    #[inline]
    pub fn icbt_l(self) -> bool {
        bits(self.raw, 21, 21) != 0
    }

    // ---- paired singles / quantized loads ----------------------------------
    /// psq_* "single word" flag.  D-form (psq_l/lu/st/stu) keeps it at bit 16;
    /// the indexed forms (psq_*x, primary 4) keep it at bit 21.
    #[inline]
    pub fn psq_w_d(self) -> bool {
        bits(self.raw, 16, 16) != 0
    }
    #[inline]
    pub fn psq_w_x(self) -> bool {
        bits(self.raw, 21, 21) != 0
    }
    /// Quantization register selector, D-form bits 17:19.
    #[inline]
    pub fn psq_gq_d(self) -> u32 {
        bits(self.raw, 17, 19)
    }
    /// Quantization register selector, indexed bits 22:24.
    #[inline]
    pub fn psq_gq_x(self) -> u32 {
        bits(self.raw, 22, 24)
    }
    /// psq_* displacement, bits 20:31, sign-extended, byte offset (the low two
    /// bits are ignored by the hardware for the word-aligned part).
    #[inline]
    pub fn psq_d(self) -> i64 {
        bits_se(self.raw, 20, 31)
    }

    /// All-zero word: not a valid instruction, but worth naming because a jump
    /// into an unpopulated guest page produces exactly this, and the error
    /// message should say "zeroed memory" rather than "reserved opcode 0".
    #[inline]
    pub fn is_all_zero(self) -> bool {
        self.raw == 0
    }
}
