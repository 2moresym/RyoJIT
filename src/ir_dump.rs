//! Deterministic textual dump of an `ir::IrBlock`.
//!
//! Two reasons this exists:
//!   • the frontends' tests assert on lowering results, and comparing a rendered
//!     string is far more readable (and far easier to review in a diff) than
//!     comparing `IrOp` variants structurally;
//!   • when a JIT'd block misbehaves, this is what you print.  It is a debug
//!     path only — nothing in the execute path calls into here.
//!
//! Format is stable and line-oriented on purpose.  An intrinsic's numeric id is
//! printed even when a name is available, so a golden never hides an id-table
//! renumbering behind a pretty name.

use crate::ir::{Endian, FlagKind, FlagOp, IrBlock, IrOp, SideEffects, VOperand, Width};

/// Renders one operand: `v12` for a virtual register, `#0x10` / `#-0x10` for an
/// immediate (hex keeps masks readable, which is most of what PPC lowering
/// produces).
fn operand_str(op: &VOperand) -> String {
    match *op {
        VOperand::Reg(v) => format!("v{}", v.0),
        VOperand::Imm(i) => format!("#{:#x}", i),
    }
}

fn width_str(w: Width) -> &'static str {
    match w {
        Width::W8 => "w8",
        Width::W16 => "w16",
        Width::W32 => "w32",
        Width::W64 => "w64",
        Width::W128 => "w128",
    }
}

fn endian_str(e: Endian) -> &'static str {
    match e {
        Endian::Little => "le",
        Endian::Big => "be",
    }
}

fn flag_kind_str(f: FlagKind) -> &'static str {
    match f {
        FlagKind::Zero => "z",
        FlagKind::Carry => "c",
        FlagKind::Overflow => "v",
        FlagKind::Negative => "s",
    }
}

fn flag_op_str(o: FlagOp) -> &'static str {
    match o {
        FlagOp::AddOp => "add",
        FlagOp::SubOp => "sub",
        FlagOp::AndOp => "and",
        FlagOp::OrOp => "or",
        FlagOp::XorOp => "xor",
        FlagOp::ShiftOp => "shl",
    }
}

fn effects_str(e: SideEffects) -> &'static str {
    match e {
        SideEffects::Pure => "pure",
        SideEffects::ReadsMem => "reads",
        SideEffects::WritesMem => "writes",
        SideEffects::ReadsAndWritesMem => "rw",
        SideEffects::Volatile => "volatile",
    }
}

/// Render one op, resolving intrinsic ids through `namer` (which returns
/// `None` for ids it does not know, so a frontend can supply just its own
/// namespace).
pub fn op_str(op: &IrOp, namer: &dyn Fn(u32) -> Option<&'static str>) -> String {
    match op {
        IrOp::Add { dst, a, b, width } => {
            format!("v{} = add {}, {} : {}", dst.0, operand_str(a), operand_str(b), width_str(*width))
        }
        IrOp::Sub { dst, a, b, width } => {
            format!("v{} = sub {}, {} : {}", dst.0, operand_str(a), operand_str(b), width_str(*width))
        }
        IrOp::Mul { dst, a, b, width } => {
            format!("v{} = mul {}, {} : {}", dst.0, operand_str(a), operand_str(b), width_str(*width))
        }
        IrOp::And { dst, a, b, width } => {
            format!("v{} = and {}, {} : {}", dst.0, operand_str(a), operand_str(b), width_str(*width))
        }
        IrOp::Or { dst, a, b, width } => {
            format!("v{} = or {}, {} : {}", dst.0, operand_str(a), operand_str(b), width_str(*width))
        }
        IrOp::Xor { dst, a, b, width } => {
            format!("v{} = xor {}, {} : {}", dst.0, operand_str(a), operand_str(b), width_str(*width))
        }
        IrOp::Shl { dst, a, amount, width } => {
            format!("v{} = shl {}, {} : {}", dst.0, operand_str(a), operand_str(amount), width_str(*width))
        }
        IrOp::Shr { dst, a, amount, width } => {
            format!("v{} = shr {}, {} : {}", dst.0, operand_str(a), operand_str(amount), width_str(*width))
        }
        IrOp::Sar { dst, a, amount, width } => {
            format!("v{} = sar {}, {} : {}", dst.0, operand_str(a), operand_str(amount), width_str(*width))
        }
        IrOp::VecAdd { dst, a, b, lanes, width } => {
            format!(
                "v{} = vec_add {}, {} : {} lanes={}",
                dst.0, operand_str(a), operand_str(b), width_str(*width), lanes
            )
        }
        IrOp::VecMul { dst, a, b, lanes, width } => {
            format!(
                "v{} = vec_mul {}, {} : {} lanes={}",
                dst.0, operand_str(a), operand_str(b), width_str(*width), lanes
            )
        }
        IrOp::VecFma { dst, a, b, c, lanes } => {
            format!(
                "v{} = vec_fma {}, {}, {} : lanes={}",
                dst.0, operand_str(a), operand_str(b), operand_str(c), lanes
            )
        }
        IrOp::Load { dst, addr, width, sign_ext, endian } => {
            format!(
                "v{} = load [{}] : {} {} {}",
                dst.0,
                operand_str(addr),
                width_str(*width),
                endian_str(*endian),
                if *sign_ext { "sext" } else { "zext" }
            )
        }
        IrOp::Store { addr, val, width, endian } => {
            format!(
                "store [{}] = {} : {} {}",
                operand_str(addr),
                operand_str(val),
                width_str(*width),
                endian_str(*endian)
            )
        }
        IrOp::Branch { cond, target } => match cond {
            Some(c) => format!("branch {} -> bb{}", operand_str(c), target.0),
            None => format!("branch -> bb{}", target.0),
        },
        IrOp::IndirectBranch { target } => {
            format!("indirect_branch -> {}", operand_str(target))
        }
        IrOp::Call { target } => format!("call {}", operand_str(target)),
        IrOp::Return => "return".to_string(),
        IrOp::SetFlags { op, a, b, width } => {
            format!(
                "setflags {} {}, {} : {}",
                flag_op_str(*op),
                operand_str(a),
                operand_str(b),
                width_str(*width)
            )
        }
        IrOp::ReadFlag { dst, flag } => {
            format!("v{} = readflag {}", dst.0, flag_kind_str(*flag))
        }
        IrOp::Intrinsic { id, effects, operands, dst } => {
            let name = namer(id.0).map(|n| n.to_string()).unwrap_or_else(|| "?".to_string());
            let args: Vec<String> = operands.iter().map(operand_str).collect();
            match dst {
                Some(d) => format!(
                    "v{} = intrinsic {}#{} [{}] ({})",
                    d.0,
                    name,
                    id.0,
                    args.join(", "),
                    effects_str(*effects)
                ),
                None => format!(
                    "intrinsic {}#{} [{}] ({})",
                    name,
                    id.0,
                    args.join(", "),
                    effects_str(*effects)
                ),
            }
        }
    }
}

/// Full block dump.  `namer` resolves intrinsic ids (see
/// `frontend_wii_ppc::intrinsic_name`); pass a closure returning `None` if you
/// do not care about names.
pub fn dump_block_with(block: &IrBlock, namer: &dyn Fn(u32) -> Option<&'static str>) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "bb{} @ {:#x}: {} ops\n",
        block.id.0, block.guest_start_pc, block.ops.len()
    ));
    for op in &block.ops {
        out.push_str("  ");
        out.push_str(&op_str(op, namer));
        out.push('\n');
    }
    out
}

/// Convenience dump without an intrinsic namer.
pub fn dump_block(block: &IrBlock) -> String {
    dump_block_with(block, &|_id| None)
}
