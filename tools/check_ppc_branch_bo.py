#!/usr/bin/env python3
"""Truth-table check for the `bc`/`bclr`/`bcctr` condition logic in lower.rs.

`branch_taken` implements PowerPC's

    if (BO[2] = 0) and CTR != 0 then CTR <- CTR - 1
    ctr_ok  = BO[2] | ((CTR != 0) XOR BO[3])
    cond_ok = BO[0] | (CR[BI] = BO[1])

The whole hazard here is the bit-to-rule mapping: `bo` is a 5-bit field whose
*most significant* bit is BO[0], and getting one of the four rules wired to the
wrong bit changes when every conditional branch in every Wii game is taken — a
bug that looks like a gameplay glitch, not a crash.  An earlier draft of
`branch_taken` had exactly that (decrement keyed off BO[0] instead of BO[2]).

So: `ISA` below is transcribed straight from the pseudocode, `LOWERING` mirrors
the bit masks in `branch_taken`, and they are compared over the complete input
space.  Re-run this after touching `branch_taken`.
"""
for line in ():
    pass

def isa(bo, crbit, ctr):
    """Reference: the pseudocode above, read literally."""
    ctr_new = ctr
    if ((bo >> 2) & 1) == 0 and ctr != 0:
        ctr_new = (ctr - 1) & 0xFFFF_FFFF
    ctr_ok = ((bo >> 2) & 1) == 1 or (((ctr_new != 0) ^ ((bo >> 1) & 1)) == 1)
    cond_ok = ((bo >> 4) & 1) == 1 or (crbit == ((bo >> 3) & 1))
    return (1 if ctr_ok and cond_ok else 0), ctr_new


def lowering(bo, crbit, ctr):
    """Mirror of src/frontends/wii_ppc/lower.rs::branch_taken."""
    ctr_new = ctr
    ctr_ok = True
    if bo & 0x04 == 0:                       # BO[2]: CTR is tested
        if ctr != 0:
            ctr_new = (ctr - 1) & 0xFFFF_FFFF  # "decrement if non-zero"
        nz = 1 if ctr_new != 0 else 0
        ctr_ok = (nz ^ (1 if bo & 0x02 else 0)) == 1  # BO[3]: polarity
    if bo & 0x10 == 0:                        # BO[0]: CR is tested
        want = 1 if bo & 0x08 else 0          # BO[1]: value it must equal
        if (crbit ^ want) != 0:
            return 0, ctr_new
    if not ctr_ok:
        return 0, ctr_new
    return 1, ctr_new


def main():
    bad = []
    for bo in range(32):
        for crbit in (0, 1):
            for ctr in (0, 1, 2, 0xFFFF_FFFF, 0x8000_0000):
                if isa(bo, crbit, ctr) != lowering(bo, crbit, ctr):
                    bad.append((bo, crbit, ctr, isa(bo, crbit, ctr), lowering(bo, crbit, ctr)))
    named = {
        "beq (bc 12,2)": (12, 1, 7, 1),
        "beq not taken": (12, 0, 7, 0),
        "bdnz from 1": (16, 0, 1, 0),      # 1 -> 0, then CTR==0 -> not taken
        "bdnz from 2": (16, 0, 2, 1),
        "bdnz from 0": (16, 0, 0, 0),
        # bo=24 is "decrement, branch if CTR != 0, skip CR, plus the taken hint" —
        # i.e. the *other* bdnz spelling; it must NOT be taken when CTR hits zero.
        "bo 24 from 1": (24, 0, 1, 0),
        "blr unconditional": (20, 0, 0, 1),
        "bclr ignores CR": (20, 0, 7, 1),
    }
    for label, (bo, crbit, ctr, want) in named.items():
        got = lowering(bo, crbit, ctr)[0]
        if got != want:
            bad.append((label, bo, crbit, ctr, got, want))
    if bad:
        print(f"FAIL: {len(bad)} disagreement(s)")
        for b in bad[:12]:
            print("   ", b)
        return 1
    print("branch_taken's BO mapping matches the ISA pseudocode over all 32 BO values")
    print(f"          (plus {len(named)} named-mnemonic spot checks)")
    return 0


raise SystemExit(main())
