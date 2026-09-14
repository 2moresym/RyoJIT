#!/usr/bin/env python3
"""Enumerate the PPC32 encoding space and have Capstone label it.

Why this exists: the Wii frontend's decode table is thousands of (primary,
extended-opcode, field-layout) triples.  Getting one digit wrong silently
miscompiles a guest instruction, which is the worst class of bug in a JIT.
Instead of trusting memory (or a single doc page), we ask a real disassembler
(Capstone, CS_ARCH_PPC / CS_MODE_32 / big-endian) to label the *whole* space,
then transcribe only what we lower.

Usage:  python3 tools/ppc_space.py > tools/ppc_encoding_space.txt

Output is a flat list of   group/idx -> mnemonic operands
that also gets diffed against the Rust decode table by tools/check_decode.py.
"""
import sys

import capstone

md = capstone.Cs(capstone.CS_ARCH_PPC, capstone.CS_MODE_32 | capstone.CS_MODE_BIG_ENDIAN)
md.detail = False


def dis(word: int):
    """Return (mnemonic, op_str) for one 32-bit instruction, or None."""
    for insn in md.disasm(word.to_bytes(4, "big"), 0x1000):
        return insn.mnemonic, insn.op_str
    return None


# Non-zero register indices so that capstone's pretty-printing (which folds
# li/lis/mr/etc. and which treats GPR0 specially in some forms) does not hide
# the real field placement.
RT, RA, RB, RS = 3, 4, 5, 6


def emit(label: str, word: int):
    d = dis(word & 0xFFFFFFFF)
    if d and d[0] not in ("", ".long", ".byte"):
        print("%-28s %08x  %-10s %s" % (label, word & 0xFFFFFFFF, d[0], d[1]))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "primary"):
        print("== primary opcodes 0..63 (generic D-form-ish field fill) ==")
        for op in range(64):
            # fill RT/RA/RB/SH/MB/ME-ish bits with distinct nonzero values
            for variant, extra in (
                ("D", (RT << 21) | (RA << 16) | 0x1234),
                ("X", (RS << 21) | (RA << 16) | (RB << 11)),
            ):
                emit("op=%d/%s" % (op, variant), (op << 26) | extra)

    if which in ("all", "g31"):
        print("== group 31, XO bits 21:30 == 0..1023 ==")
        for xo in range(1024):
            emit("g31.xo=%d" % xo, (31 << 26) | (RT << 21) | (RA << 16) | (RB << 11) | (xo << 1))

    if which in ("all", "g19"):
        print("== group 19, XO bits 21:30 == 0..1023 ==")
        for xo in range(1024):
            emit("g19.xo=%d" % xo, (19 << 26) | (xo << 1) | (1 << 16) | (2 << 11))

    if which in ("all", "g59"):
        print("== group 59 (FP, XO bits 26:30) == 0..31, rc=0/1 ==")
        for xo in range(32):
            for rc in (0, 1):
                emit("g59.xo=%d.rc=%d" % (xo, rc),
                     (59 << 26) | (RT << 21) | (RA << 16) | (RB << 11) | (xo << 1) | rc)

    if which in ("all", "g63"):
        print("== group 63 (FP, XO bits 26:30) == 0..31, rc=0/1, plus FRT-as-index variants ==")
        for xo in range(32):
            for rc in (0, 1):
                emit("g63.xo=%d.rc=%d" % (xo, rc),
                     (63 << 26) | (RT << 21) | (RA << 16) | (RB << 11) | (xo << 1) | rc)
            # 63-group entries whose bits 21:25 are NOT a float register index
            # (fmadd/fmsub/fnmsub/fnmadd use FC bits 21:23; psq_* use W bit 21)
            emit("g63.xo=%d.fc3" % xo,
                 (63 << 26) | (RT << 21) | (RA << 16) | (RB << 11) | (xo << 1) | (3 << 8)) # PPC bits 21:23 (fmadd FC / psq W) = raw bits 8:10

    if which in ("all", "spr"):
        print("== mfspr/mtspr SPR number space (bits 11:20 swapped field) ==")
        for spr in list(range(0, 1024)):
            sprx = ((spr & 0x1F) << 5) | ((spr >> 5) & 0x1F)  # encode: bits16:20 = spr&31, bits11:15 = spr>>5
            emit("mfspr spr=%d" % spr, (31 << 26) | (RT << 21) | (sprx << 11) | (339 << 1))
            if spr in (1, 8, 22, 25, 26, 272, 268, 269, 287, 638, 1008, 1009, 1010, 1011, 1013, 1017, 1019, 1020, 1021, 1022, 920, 921):
                emit(" mtspr spr=%d" % spr, (31 << 26) | (RS << 21) | (sprx << 11) | (467 << 1))


main()
