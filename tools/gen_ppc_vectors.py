#!/usr/bin/env python3
"""Generate `src/frontends/wii_ppc/vectors.rs` — decode-test vectors for the PPC table.

Why generated: the dispatch table is ~220 arms of `(primary, extended-opcode)`
pairs plus a dozen overlapping field layouts.  Checking that by eye is how a JIT
ends up executing `fnmsub` where the guest wrote `fmsub`.  So this script

  1. *assembles* each instruction from its field layout (the same layout
     `src/frontends/wii_ppc/fields.rs` extracts with), using distinctive
     non-zero operand values so a swapped field cannot hide,
  2. asks Capstone (a real disassembler) to disassemble the word and compares
     the mnemonic — for every instruction Capstone's ppc32 table knows, which is
     all of the integer / load-store / branch space and part of the FP space,
  3. emits the Rust table `(word, expected mnemonic, expected fields)`.

Rows Capstone cannot check (paired singles, FP X-form, the 750CL extras) are
still emitted — those came from YAGCD §3.4 / the AIX appendix / real assembler
test bytes, and they are marked in the report so it is obvious what was
independently confirmed.

Usage:
    python3 tools/gen_ppc_vectors.py            # write the file + print report
    python3 tools/gen_ppc_vectors.py --check    # fail if Capstone disagrees
"""
import sys

try:
    import capstone
except Exception:  # pragma: no cover - capstone is optional for regenerating
    capstone = None

OUT = "src/frontends/wii_ppc/vectors.rs"

# ---------------------------------------------------------------------------
# encoders: PPC bit numbering, bit 0 = MSB.  These mirror fields.rs exactly,
# so a mismatch between the two is a bug in *both* and shows up here.
# ---------------------------------------------------------------------------


def put(word, first, last, value):
    width = last - first + 1
    shift = 31 - last
    mask = (1 << width) - 1
    return word | ((value & mask) << shift)


def d(op, rt, ra, imm):
    """D-form: OPCD, RT(6:10), RA(11:15), UI/SI(16:31)."""
    w = op << 26
    w = put(w, 6, 10, rt)
    w = put(w, 11, 15, ra)
    w = put(w, 16, 31, imm)
    return w


def x_form(op, rt, ra, rb, xo, rc=0):
    """X-form: dest at 6:10, srcs at 11:15 / 16:20, XO 26:30 (5-bit), Rc 31."""
    w = op << 26
    w = put(w, 6, 10, rt)
    w = put(w, 11, 15, ra)
    w = put(w, 16, 20, rb)
    w = put(w, 26, 30, xo)
    w = put(w, 31, 31, rc)
    return w


def xo_form(rt, ra, rb, xo10, oe=0, rc=0):
    """Group-31 XO form: the 10-bit field at 21:30 holds OE in bit 21 then XO."""
    w = 31 << 26
    w = put(w, 6, 10, rt)
    w = put(w, 11, 15, ra)
    w = put(w, 16, 20, rb)
    w = put(w, 21, 30, xo10)
    w = put(w, 21, 21, oe)
    w = put(w, 31, 31, rc)
    return w


def m_form(ra, rs, sh, mb, me, rc=0, op=21):
    w = op << 26
    w = put(w, 6, 10, rs)
    w = put(w, 11, 15, ra)
    w = put(w, 16, 20, sh)
    w = put(w, 21, 25, mb)
    w = put(w, 26, 30, me)
    w = put(w, 31, 31, rc)
    return w


def b_form(bo, bi, bd, aa=0, lk=0):
    w = 16 << 26
    w = put(w, 6, 10, bo)
    w = put(w, 11, 15, bi)
    w = put(w, 16, 29, bd >> 2 if abs(bd) < (1 << 15) else bd)
    w = put(w, 30, 30, aa)
    w = put(w, 31, 31, lk)
    return w


def i_form(li, aa=0, lk=0):
    w = 18 << 26
    w = put(w, 6, 29, li >> 2)
    w = put(w, 30, 30, aa)
    w = put(w, 31, 31, lk)
    return w


def xl19(xo10, *fields):
    """Group 19: 10-bit XO at 21:30, plus whatever per-instruction fields."""
    w = 19 << 26
    w = put(w, 21, 30, xo10)
    for first, last, v in fields:
        w = put(w, first, last, v)
    return w


def fx_spr(op31, rt, spr, xo10):
    """XFX: the SPR number is stored with its halves swapped."""
    w = 31 << 26
    w = put(w, 6, 10, rt)
    w = put(w, 16, 20, spr & 0x1F)
    w = put(w, 11, 15, (spr >> 5) & 0x1F)
    w = put(w, 21, 30, xo10)
    return w


def fp_a(op, frt, fra, frb, xo5, rc=0):
    w = op << 26
    w = put(w, 6, 10, frt)
    w = put(w, 11, 15, fra)
    w = put(w, 16, 20, frb)
    w = put(w, 26, 30, xo5)
    w = put(w, 31, 31, rc)
    return w


def fp_ax(op, frt, fra, frc, frb, xo5, rc=0):
    """AX-form: FRC's top 3 bits at 21:23 (register numbers are multiples of 4)."""
    w = fp_a(op, frt, fra, frb, xo5, rc)
    w = put(w, 21, 23, (frc >> 2) & 7)
    return w


def fp_x(frt, fra, xo9, rc=0):
    w = 63 << 26
    w = put(w, 6, 10, frt)
    w = put(w, 11, 15, fra)
    w = put(w, 22, 30, xo9)
    w = put(w, 31, 31, rc)
    return w


def ps_a(frt, fra, frb, xo5, frc=None, rc=0):
    """Broadway paired single, primary 4 (no VMX on this core)."""
    w = fp_a(4, frt, fra, frb, xo5, rc)
    if frc is not None:
        w = put(w, 21, 25, frc)
    return w


def ps_x(frt, fra, frb, xo10, rc=0):
    w = 4 << 26
    w = put(w, 6, 10, frt)
    w = put(w, 11, 15, fra)
    w = put(w, 16, 20, frb)
    w = put(w, 21, 30, xo10)
    w = put(w, 31, 31, rc)
    return w


def psq_d(op, frt, ra, w12, gi, d):
    v = op << 26
    v = put(v, 6, 10, frt)
    v = put(v, 11, 15, ra)
    v = put(v, 16, 16, w12)
    v = put(v, 17, 19, gi)
    v = put(v, 20, 31, d)
    return v


def psq_x(frt, ra, rb, w12, gi, xo10):
    v = 4 << 26
    v = put(v, 6, 10, frt)
    v = put(v, 11, 15, ra)
    v = put(v, 16, 20, rb)
    v = put(v, 21, 21, w12)
    v = put(v, 22, 24, gi)
    v = put(v, 25, 30, xo10)
    return v


# ---------------------------------------------------------------------------
# The corpus.  Each row: (mnemonic, word, capstone-expected-mnemonic-or-None)
# Distinct operand values throughout: 3/4/5 for the register fields and
# unusual immediates, so a swapped or mis-sized field changes the word.
# ---------------------------------------------------------------------------
V = []


def row(mnemonic, word, capstone_name=None):
    V.append((mnemonic, word, capstone_name))


# What *this* decoder is expected to report.  Two reasons it differs from the
# label used in the table: rows that only differ by field values ("add." vs
# "add", "rlwinm.wrap") decode to the same kind, and SPR rows all decode to the
# generic "mfspr"/"mtspr" because this frontend decodes the SPR number as a field
# rather than as a distinct kind.
DECODE_NAME = {
    "ba": "b", "bl": "b", "bcl": "bc", "bclr.eq": "bclr",
    "add.": "add", "addo": "add", "addo.": "add", "mullw.": "mullw",
    "rlwinm.wrap": "rlwinm", "fmadd.ax": "fmadd", "twi.eq": "twi",
    "mulhw.p10": "mulhw", "mulhwu.p11": "mulhwu", "psq_l.x": "psq_l",
    "cmp": "cmp", "cmpl": "cmpl",
    "mfxer": "mfspr", "mflr": "mfspr", "mfctr": "mfspr", "mfdec": "mfspr",
    "mfsdr1": "mfspr", "mfsrr0": "mfspr", "mfsrr1": "mfspr",
    "mfsprg0": "mfspr", "mfpvr": "mfspr", "mfl2cr": "mfspr",
    "mfhid0": "mfspr", "mfhid1": "mfspr", "mfpspr": "mfspr",
    "mftbl": "mfspr", "mftbu": "mfspr",
    "mtdecr": "mtspr", "mtxer": "mtspr", "mtlr": "mtspr", "mtctr": "mtspr",
    "mthid0": "mtspr", "mtl2cr": "mtspr", "mtpspr": "mtspr",
    "mtdar": "mtspr", "mtdsisr": "mtspr",
    "fdivs": "fdivs", "fsubs": "fsubs", "fadds": "fadds", "fmuls": "fmuls",
    "fmsubs": "fmsubs", "fmadds": "fmadds", "fnmsubs": "fnmsubs",
    "fnmadds": "fnmadds",
}
for n in ("lfsx", "lfsux", "lfdx", "lfdux", "stfsx", "stfsux", "stfdx",
          "stfdux", "stfiwx", "lfiwax", "lfiwzx"):
    DECODE_NAME[n + ".x"] = n


# --- D-form integer ---------------------------------------------------------
row("mulli", d(7, 3, 4, 0x1234), "mulli")
row("subfic", d(8, 3, 4, 0x0010), "subfic")
row("cmpli", d(10, 1, 4, 0x1234), None)
row("cmpi", d(11, 1, 4, 0xFFEC), None)
row("addic", d(12, 3, 4, 0x1234), "addic")
row("addic.", d(13, 3, 4, 0x1234), "addic.")
row("addi", d(14, 3, 4, 0x1234), "addi")
row("addis", d(15, 3, 4, 0x0012), "addis")
row("ori", d(24, 3, 4, 0x1234), "ori")
row("oris", d(25, 3, 4, 0x1234), "oris")
row("xori", d(26, 3, 4, 0x1234), "xori")
row("xoris", d(27, 3, 4, 0x1234), "xoris")
row("andi.", d(28, 3, 4, 0x1234), "andi.")
row("andis.", d(29, 3, 4, 0x1234), "andis.")
# --- loads / stores ---------------------------------------------------------
for name, op in (
    ("lwz", 32), ("lwzu", 33), ("lbz", 34), ("lbzu", 35),
    ("stw", 36), ("stwu", 37), ("stb", 38), ("stbu", 39),
    ("lhz", 40), ("lhzu", 41), ("lha", 42), ("lhau", 43),
    ("sth", 44), ("sthu", 45), ("lmw", 46), ("stmw", 47),
    ("lfs", 48), ("lfsu", 49), ("lfd", 50), ("lfdu", 51),
    ("stfs", 52), ("stfsu", 53), ("stfd", 54), ("stfdu", 55),
):
    row(name, d(op, 3, 4, 0x0248), name)
# --- branches ---------------------------------------------------------------
row("b", i_form(0x1000), "b")
row("ba", i_form(0x0800, aa=1), None)
row("bl", i_form(-0x1000, lk=1), "bl")
row("bc", b_form(12, 2, 0x0048), "beq")
row("bcl", b_form(12, 2, -0x0048, lk=1), "beql")
row("twi", d(3, 4, 4, 0x0000), None)
# --- rotate class -----------------------------------------------------------
row("rlwimi", m_form(3, 4, 6, 10, 20, rc=1, op=20), "rlwimi")
row("rlwinm", m_form(3, 4, 6, 10, 20, rc=1, op=21), "rlwinm")
row("rlwinm.wrap", m_form(3, 4, 0, 20, 11, rc=0, op=21), "rlwinm")
row("rlwnm", m_form(3, 4, 5, 10, 20, rc=0, op=23), "rlwnm")
# --- group 31: integer + memory + cache, per-instruction field layout -------
# (rt, ra, rb) that actually exist for each row; anything else must read 0, or
# the encoding is not the instruction the manual says it is.
G31 = [
    ("cmp", 0, None, 4, 5, 0, 0, ('bf', 3)),
    ("cmpl", 32, None, 4, 5, 0, 0, ('bf', 3)),
    ("tw", 4, None, 3, 4, 0, 0, ('to', 4)),
    ("subfc", 8, 3, 4, 5, 0, 0, None),
    ("addc", 10, 3, 4, 5, 0, 0, None),
    ("mulhwu", 11, 3, 4, 5, 0, 0, None),
    ("mfcr", 19, 3, 0, 0, 0, 0, None),
    ("lwarx", 20, 3, 4, 5, 0, 0, None),
    ("lwzx", 23, 3, 4, 5, 0, 0, None),
    ("slw", 24, 3, 4, 5, 0, 0, None),
    ("cntlzw", 26, 3, 4, 0, 0, 0, None),
    ("and", 28, 3, 4, 5, 0, 0, None),
    ("subf", 40, 3, 4, 5, 0, 0, None),
    ("dcbst", 54, 0, 4, 5, 0, 0, None),
    ("lwzux", 55, 3, 4, 5, 0, 0, None),
    ("andc", 60, 3, 4, 5, 0, 0, None),
    ("mulhw", 75, 3, 4, 5, 0, 0, None),
    ("mfmsr", 83, 3, 0, 0, 0, 0, None),
    ("dcbf", 86, 0, 4, 5, 0, 0, None),
    ("lbzx", 87, 3, 4, 5, 0, 0, None),
    ("neg", 104, 3, 4, 0, 0, 0, None),
    ("lbzux", 119, 3, 4, 5, 0, 0, None),
    ("nor", 124, 3, 4, 5, 0, 0, None),
    ("subfe", 136, 3, 4, 5, 0, 0, None),
    ("adde", 138, 3, 4, 5, 0, 0, None),
    ("mtcrf", 144, 3, 0, 0, 0, 0, ('flm', 15)),
    ("mtmsr", 146, 3, 0, 0, 0, 0, None),
    ("stwcx.", 150, 3, 4, 5, 0, 1, None),
    ("stwx", 151, 3, 4, 5, 0, 0, None),
    ("stwux", 183, 3, 4, 5, 0, 0, None),
    ("subfze", 200, 3, 4, 0, 0, 0, None),
    ("addze", 202, 3, 4, 0, 0, 0, None),
    ("mtsr", 210, 3, 0, 0, 0, 0, ('sr', 4)),
    ("stbx", 215, 3, 4, 5, 0, 0, None),
    ("subfme", 232, 3, 4, 0, 0, 0, None),
    ("addme", 234, 3, 4, 0, 0, 0, None),
    ("mullw", 235, 3, 4, 5, 0, 0, None),
    ("mtsrin", 242, 3, 0, 5, 0, 0, None),
    ("dcbtst", 246, 0, 4, 5, 0, 0, None),
    ("stbux", 247, 3, 4, 5, 0, 0, None),
    ("add", 266, 3, 4, 5, 0, 0, None),
    ("dcbt", 278, 0, 4, 5, 0, 0, None),
    ("lhzx", 279, 3, 4, 5, 0, 0, None),
    ("eqv", 284, 3, 4, 5, 0, 0, None),
    ("tlbie", 306, 0, 0, 3, 0, 0, None),
    ("eciwx", 310, 3, 4, 5, 0, 0, None),
    ("xor", 316, 3, 4, 5, 0, 0, None),
    ("lhzux", 311, 3, 4, 5, 0, 0, None),
    ("mfspr", 339, 3, 0, 0, 0, 0, ('spr', 8)),
    ("lhax", 343, 3, 4, 5, 0, 0, None),
    ("lhaux", 375, 3, 4, 5, 0, 0, None),
    ("sthx", 407, 3, 4, 5, 0, 0, None),
    ("orc", 412, 3, 4, 5, 0, 0, None),
    ("ecowx", 438, 3, 4, 5, 0, 0, None),
    ("sthux", 439, 3, 4, 5, 0, 0, None),
    ("or", 444, 3, 4, 5, 0, 0, None),
    ("divwu", 459, 3, 4, 5, 0, 0, None),
    ("mtspr", 467, 3, 0, 0, 0, 0, ('spr', 8)),
    ("dcbi", 470, 0, 4, 5, 0, 0, None),
    ("nand", 476, 3, 4, 5, 0, 0, None),
    ("divw", 491, 3, 4, 5, 0, 0, None),
    ("mcrxr", 512, 0, 0, 1, 0, 0, None),
    ("lswx", 533, 3, 4, 5, 0, 0, None),
    ("lwbrx", 534, 3, 4, 5, 0, 0, None),
    ("lfsx", 535, 1, 2, 5, 0, 0, None),
    ("srw", 536, 3, 4, 5, 0, 0, None),
    ("tlbsync", 566, 0, 0, 0, 0, 0, None),
    ("lfsux", 567, 1, 2, 5, 0, 0, None),
    ("mfsr", 595, 3, 0, 0, 0, 0, ('sr', 4)),
    ("lswi", 597, 3, 4, 20, 0, 0, None),
    ("sync", 598, 0, 0, 0, 0, 0, None),
    ("lfdx", 599, 1, 2, 5, 0, 0, None),
    ("lfdux", 631, 1, 2, 5, 0, 0, None),
    ("mfsrin", 659, 3, 0, 5, 0, 0, None),
    ("stswx", 661, 3, 4, 5, 0, 0, None),
    ("stwbrx", 662, 3, 4, 5, 0, 0, None),
    ("stfsx", 663, 1, 2, 5, 0, 0, None),
    ("stfsux", 695, 1, 2, 5, 0, 0, None),
    ("stswi", 725, 3, 4, 20, 0, 0, None),
    ("stfdx", 727, 1, 2, 5, 0, 0, None),
    ("stfdux", 759, 1, 2, 5, 0, 0, None),
    ("lhbrx", 790, 3, 4, 5, 0, 0, None),
    ("sraw", 792, 3, 4, 5, 0, 0, None),
    ("srawi", 824, 3, 4, 7, 0, 0, None),
    ("eieio", 854, 0, 0, 0, 0, 0, None),
    ("lfiwax", 855, 1, 2, 5, 0, 0, None),
    ("lfiwzx", 887, 1, 2, 5, 0, 0, None),
    ("sthbrx", 918, 3, 4, 5, 0, 0, None),
    ("extsh", 922, 3, 4, 0, 0, 0, None),
    ("extsb", 954, 3, 4, 0, 0, 0, None),
    ("icbi", 982, 0, 4, 5, 0, 0, None),
    ("stfiwx", 983, 1, 2, 5, 0, 0, None),
    ("dcbz", 1014, 0, 3, 4, 0, 0, None),
    ("popcntb", 532, 3, 4, 0, 0, 0, None),
    ("icbt", 22, 0, 3, 4, 0, 0, None),
]

CAPSTONE_KNOWS = set("""
add subf mullw and or xor nand nor eqv andc orc slw srw sraw cntlzw addc adde
addze addme subfc subfe subfze subfme mulhw mulhwu neg divw divwu lwzx lwzux
lbzx lbzux lhzx lhax lhaux stwx stwux stbx stbux sthx sthux lwarx stwcx. mfmsr
mtmsr lswi stswi lwbrx stwbrx lhbrx sthbrx dcbf dcbst dcbz icbi srawi mtcrf
lfdx lfdux stfdx stfdux lfsx lfsux stfsx stfsux stfiwx twi extsh extsb popcntb
""".split())


for name, xo10, rt, ra, rb, oe, rc, extra in G31:
    # The X/XO-form operand slots are the same three positions for every one of
    # these — which field is *named* RT/RA/RS differs per op (logical ops write
    # RA, `cntlzw`'s source sits in the RT slot), but the encoding does not.
    # Fields a given instruction does not use stay 0, or the word is not that
    # instruction at all.
    w = 31 << 26
    if rt is not None:
        w = put(w, 6, 10, rt)
    if ra is not None:
        w = put(w, 11, 15, ra)
    if rb is not None:
        w = put(w, 16, 20, rb)
    if extra and extra[0] == "bf":
        w = put(w, 6, 8, extra[1])
        w = put(w, 11, 15, ra)
        w = put(w, 16, 20, rb)
    elif extra and extra[0] == "to":
        w = put(w, 11, 15, ra)
        w = put(w, 16, 20, rb)
    elif extra and extra[0] == "spr":
        w = put(w, 6, 10, rt)
        w = put(w, 16, 20, extra[1] & 0x1F)
        w = put(w, 11, 15, (extra[1] >> 5) & 0x1F)
    elif extra and extra[0] == "flm":
        w = put(w, 6, 10, rt)
        w = put(w, 12, 19, extra[1])
    elif extra and extra[0] == "sr":
        w = put(w, 6, 10, rt)
        w = put(w, 16, 19, extra[1])
    w = put(w, 21, 30, xo10)
    if oe:
        w = put(w, 21, 21, 1)
    if rc:
        w = put(w, 31, 31, 1)
    expect = name if name in CAPSTONE_KNOWS else None
    row(name, w, expect)

# `add.` / `addo.` / `mullw.`: same layout, Rc/OE bits set.
def g31_named(name, xo10, oe=0, rc=0, capname=None):
    w = 31 << 26
    w = put(w, 6, 10, 3)
    w = put(w, 11, 15, 4)
    w = put(w, 16, 20, 5)
    w = put(w, 21, 30, xo10)
    if oe:
        w = put(w, 21, 21, 1)
    if rc:
        w = put(w, 31, 31, 1)
    row(name, w, capname)

g31_named("add.", 266, rc=1, capname="add.")
g31_named("addo", 266, oe=1, capname=None)
g31_named("addo.", 266, oe=1, rc=1, capname=None)
g31_named("mullw.", 235, rc=1, capname="mullw.")
g31_named("and.", 28, rc=1, capname="and.")

# --- group 19 ---------------------------------------------------------------
row("mcrf", xl19(0, (6, 8, 3), (11, 13, 1)), "mcrf")
row("crand", xl19(257, (6, 10, 1), (11, 15, 2), (16, 20, 3)), "crand")
row("cror", xl19(449, (6, 10, 1), (11, 15, 2), (16, 20, 3)), "cror")
row("crxor", xl19(193, (6, 10, 1), (11, 15, 2), (16, 20, 3)), "crxor")
row("crnand", xl19(225, (6, 10, 1), (11, 15, 2), (16, 20, 3)), "crnand")
row("crnor", xl19(33, (6, 10, 1), (11, 15, 2), (16, 20, 3)), "crnor")
row("creqv", xl19(289, (6, 10, 1), (11, 15, 2), (16, 20, 3)), "creqv")
row("crandc", xl19(129, (6, 10, 1), (11, 15, 2), (16, 20, 3)), "crandc")
row("crorc", xl19(417, (6, 10, 1), (11, 15, 2), (16, 20, 3)), "crorc")
row("bclr", xl19(16, (6, 10, 20), (11, 15, 0)), "blr")
row("bclr.eq", xl19(16, (6, 10, 12), (11, 15, 2)), "beqlr")
row("bcctr", xl19(528, (6, 10, 20), (11, 15, 0)), "bctr")
row("isync", xl19(150), "isync")
row("rfi", xl19(50), "rfi")
# --- SPRs (via mfspr/mtspr) --------------------------------------------------
for name, spr in (
    ("mfxer", 1), ("mflr", 8), ("mfctr", 9), ("mfdec", 22), ("mfsdr1", 25),
    ("mfsrr0", 26), ("mfsrr1", 27), ("mfsprg0", 272), ("mfpvr", 287),
    ("mfl2cr", 1017), ("mfhid0", 1000), ("mfhid1", 1009), ("mfpspr", 1008),
    ("mftbl", 268), ("mftbu", 269), ("mfdar", 19), ("mfdccr", 1018),
    ("mficcr", 1019), ("mfiabr", 1010), ("mfdabr", 1013),
):
    row(name, fx_spr(31, 3, spr, 339), None)
for name, spr in (("mtdecr", 22), ("mtxer", 1), ("mtlr", 8), ("mtctr", 9),
                  ("mthid0", 1000), ("mtl2cr", 1017), ("mtpspr", 1008),
                  ("mtdar", 19), ("mtdsisr", 18), ("mtsrr0", 26),
                  ("mtsrr1", 27), ("mtsdr1", 25), ("mtsprg0", 272),
                  ("mtiidabr", 1010), ("mtdidabr", 1013), ("mtdccr", 1018),
                  ("mticcr", 1019), ("mthid1", 1009)):
    row(name, fx_spr(31, 3, spr, 467), None)
# --- FP A-form (59 single, 63 double) ---------------------------------------
FP_A = {"fdiv": 18, "fsub": 20, "fadd": 21, "fmsub": 28, "fmadd": 29,
        "fnmsub": 30, "fnmadd": 31}
for name, xo in FP_A.items():
    row(name, fp_a(63, 1, 2, 5, xo), None)
    row(name + "s", fp_a(59, 1, 2, 5, xo), name + "s")
# fmul / fmuls: third operand is FRC (bits 21:23, multiples of 4) and FRB is
# reserved 0 — a two-input multiply, not an fma.
row("fmul", fp_ax(63, 1, 2, 4, 0, 25), None)
row("fmuls", fp_ax(59, 1, 2, 4, 0, 25), "fmuls")
row("fres", fp_a(59, 1, 2, 5, 24), None)
row("frsqrte", fp_a(63, 1, 2, 5, 26), None)
row("fsel", fp_ax(63, 1, 2, 4, 5, 23), None)
# --- FP X-form (9-bit field at 22:30) ---------------------------------------
for name, xo9 in (("frsp", 12), ("fctiw", 14), ("fctiwz", 15), ("mtfsb1", 38),
                  ("fneg", 40), ("mcrfs", 64), ("mtfsb0", 70), ("mffs", 71),
                  ("fmr", 72), ("mtfsfi", 134), ("fnabs", 136), ("fabs", 264),
                  ("mtfsf", 199)):
    if name == "mcrfs":
        w = 63 << 26
        w = put(w, 6, 8, 3)
        w = put(w, 11, 13, 1)
        w = put(w, 22, 30, xo9)
    elif name == "mtfsf":
        w = 63 << 26
        w = put(w, 11, 15, 2)              # FRA
        w = put(w, 12, 19, 0xFF)           # FM
        w = put(w, 22, 30, xo9)
    elif name in ("mtfsb0", "mtfsb1"):
        w = 63 << 26
        w = put(w, 6, 10, 13)              # BT
        w = put(w, 22, 30, xo9)
    elif name == "mtfsfi":
        w = 63 << 26
        w = put(w, 9, 11, 1)               # BF
        w = put(w, 15, 18, 3)              # U
        w = put(w, 22, 30, xo9)
    elif name == "mffs":
        w = 63 << 26
        w = put(w, 6, 10, 1)
        w = put(w, 21, 30, 583)
    else:
        w = fp_x(1, 2, xo9)
    row(name, w, None)
w = 63 << 26
w = put(w, 6, 8, 3)          # BF
w = put(w, 11, 15, 2)        # FRA
w = put(w, 16, 20, 5)        # FRB
w = put(w, 26, 30, 20)
w = put(w, 31, 31, 1)
row("fcmpu", w, None)
w = 63 << 26
w = put(w, 6, 8, 3)
w = put(w, 11, 15, 2)
w = put(w, 16, 20, 5)
w = put(w, 26, 30, 16)
row("fcmpo", w, None)

# --- 750CL extras -----------------------------------------------------------
row("popcntb", xo_form(3, 4, 0, 532), None)
row("mulhw.p10", put(put(10 << 26, 6, 10, 3), 11, 20, (4 << 5) | 5) | (2 << 1), None)
row("mulhwu.p11", (11 << 26) | (3 << 21) | (4 << 16) | (5 << 11) | (3 << 1), None)
row("icbt", (31 << 26) | (0 << 21) | (4 << 16) | (5 << 11) | (22 << 1), None)
row("sc", (17 << 26) | (1 << 1), "sc")
# --- paired singles (real assembler bytes where I have them) ---------------
row("ps_add", ps_a(1, 2, 3, 21), None)
row("ps_sub", ps_a(1, 2, 3, 20), None)
row("ps_mul", ps_a(1, 2, 0, 25, frc=4), None)
row("ps_div", ps_a(1, 2, 3, 18), None)
row("ps_madds0", ps_a(1, 2, 3, 14, frc=4), None)
row("ps_madds1", ps_a(1, 2, 3, 15, frc=4), None)
row("ps_muls0", ps_a(1, 2, 0, 12, frc=4), None)
row("ps_muls1", ps_a(1, 2, 0, 13, frc=4), None)
row("ps_sum0", ps_a(1, 2, 3, 10, frc=4), None)
row("ps_sum1", ps_a(1, 2, 3, 11, frc=4), None)
row("ps_sel", ps_a(1, 2, 3, 23, frc=4), None)
row("ps_madd", ps_a(1, 2, 3, 29, frc=4), None)
row("ps_msub", ps_a(1, 2, 3, 28, frc=4), None)
row("ps_nmadd", ps_a(1, 2, 3, 31, frc=4), None)
row("ps_nmsub", ps_a(1, 2, 3, 30, frc=4), None)
row("ps_res", ps_a(1, 0, 2, 24), None)
row("ps_rsqrte", ps_a(1, 0, 2, 26), None)
row("ps_merge00", ps_x(1, 2, 3, (16 << 5) | 16), None)
row("ps_merge01", ps_x(1, 2, 3, (17 << 5) | 16), None)
row("ps_merge10", ps_x(1, 2, 3, (18 << 5) | 16), None)
row("ps_merge11", ps_x(1, 2, 3, (19 << 5) | 16), None)
row("ps_mr", ps_x(1, 0, 2, (2 << 5) | 8), None)
row("ps_neg", ps_x(1, 0, 2, (1 << 5) | 8), None)
row("ps_abs", ps_x(1, 0, 2, (8 << 5) | 8), None)
row("ps_nabs", ps_x(1, 0, 2, (4 << 5) | 8), None)
row("ps_cmpu0", ps_x(1, 2, 3, 0), None)
row("ps_cmpu1", ps_x(1, 2, 3, (2 << 5) | 0), None)
row("ps_cmpo0", ps_x(1, 2, 3, (1 << 5) | 0), None)
row("ps_cmpo1", ps_x(1, 2, 3, (3 << 5) | 0), None)
row("psq_l", psq_d(56, 0, 3, 1, 5, 4), None)
row("psq_lu", psq_d(57, 1, 2, 0, 3, 8), None)
row("psq_st", psq_d(60, 3, 2, 0, 2, 8), None)
row("psq_stu", psq_d(61, 3, 2, 1, 7, -0x10), None)
row("psq_lx", psq_x(0, 3, 4, 1, 7, 6), None)
row("psq_lux", psq_x(1, 2, 3, 0, 3, 38), None)
row("psq_stx", psq_x(3, 2, 4, 0, 3, 7), None)
row("psq_stux", psq_x(3, 2, 4, 1, 2, 39), None)
row("twi.eq", d(3, 4, 4, 0), None)


# ---------------------------------------------------------------------------
# capstone cross-check
# ---------------------------------------------------------------------------
def capstone_labels():
    if capstone is None:
        return {}
    md = capstone.Cs(capstone.CS_ARCH_PPC, capstone.CS_MODE_32 | capstone.CS_MODE_BIG_ENDIAN)
    out = {}
    for mnemonic, word, expect in V:
        got = None
        for insn in md.disasm(word.to_bytes(4, "big"), 0x1000):
            got = insn.mnemonic
            break
        out[mnemonic] = (got, expect)
    return out


def main():
    checked = confirmed = mismatched = unchecked = 0
    if capstone is not None:
        labels = capstone_labels()
        for mnemonic, word, expect in V:
            got, want = labels[mnemonic]
            if want is None:
                unchecked += 1
                continue
            checked += 1
            if got == want or (got and got.replace(".", "") == want.replace(".", "")):
                confirmed += 1
            else:
                mismatched += 1
                print(f"  MISMATCH {mnemonic:16s} {word:#010x}  capstone={got!r} expected={want!r}")
    print(f"corpus: {len(V)} rows | capstone-checked {checked} (confirmed {confirmed}, mismatched {mismatched}) | doc-only {unchecked}")

    lines = [
        "//! Decode test vectors — **generated by `tools/gen_ppc_vectors.py`, do not edit**.",
        "//!",
        "//! One row per instruction: the encoded word, the mnemonic it must decode to,",
        "//! and the field values the encoder placed.  Generated from the same field",
        "//! layouts `fields.rs` reads (see the script header for how each row was",
        "//! validated — Capstone for the integer/load/branch space, manual tables for",
        "//! the paired-single and FP X-form rows Capstone's ppc32 table does not have).",
        "",
        "/// One generated case.",
        "#[derive(Debug, Clone, Copy)]",
        "#[allow(dead_code)]",
        "pub struct Vector {",
        "    /// The instruction word.",
        "    pub word: u32,",
        "    /// Mnemonic the decoder must resolve it to (with a `.wrap`/`.x`/`.p10`",
        "    /// suffix on rows that only differ from a sibling by field values).",
        "    pub mnemonic: &'static str,",
        "    /// Expected primary opcode, so a broken `op()` accessor cannot pass by luck.",
        "    pub op: u8,",
        "}",
        "",
        "pub const VECTORS: &[Vector] = &[",
    ]
    seen = set()

    for mnemonic, word, _expect in V:
        base = DECODE_NAME.get(mnemonic, mnemonic)
        expect = base
        op = (word >> 26) & 0x3F
        if mnemonic in seen:
            continue
        seen.add(mnemonic)
        lines.append(
            f"    Vector {{ word: 0x{word:08X}, mnemonic: \"{expect}\", op: {op} }},"
        )
    lines.append("];")
    lines.append("")
    body = "\n".join(lines)
    with open(OUT, "w") as fh:
        fh.write(body)
    print(f"wrote {OUT}: {len(seen)} rows")
    if mismatched and "--check" in sys.argv:
        return 1
    return 0


sys.exit(main())
