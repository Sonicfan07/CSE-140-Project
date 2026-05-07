#!/usr/bin/env python3
"""
CSE 140 Project: 5-Stage Pipelined RISC-V CPU Simulator

Supported instructions:
  Part 1: lw, sw, add, addi, sub, and, andi, or, ori, beq
  Part 2: jal, jalr
  Extra:  lb, sb, bne, sll, srl

Pipeline stages:
  IF -> ID -> EX -> MEM -> WB

Pipeline registers:
  if_id, id_ex, ex_mem, mem_wb

Main pipeline choices:
  - No forwarding unit
  - No branch predictor
  - RAW hazards stall the PC and IF/ID register
  - Taken branches/jumps flush the wrong-path instructions
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ABI names are used for the Section 2 sample output.
ABI_NAMES = {
     0: "zero",  1: "ra",   2: "sp",   3: "gp",
     4: "tp",    5: "t0",   6: "t1",   7: "t2",
     8: "s0",    9: "s1",  10: "a0",  11: "a1",
    12: "a2",   13: "a3",  14: "a4",  15: "a5",
    16: "a6",   17: "a7",  18: "s2",  19: "s3",
    20: "s4",   21: "s5",  22: "s6",  23: "s7",
    24: "s8",   25: "s9",  26: "s10", 27: "s11",
    28: "t3",   29: "t4",  30: "t5",  31: "t6",
}

# Section 1 prints x3/x5/etc.; Section 2 prints ra/a0/t5/etc.
USE_ABI_NAMES = False


# -------------------------
# Global CPU state
# -------------------------

pc = 0                         # current fetch address
total_clock_cycles = 0          # increments once per pipeline clock cycle
rf = [0] * 32                   # register file: x0-x31
d_mem = [0] * 64                # data memory, each entry is one 32-bit word


# -------------------------
# Pipeline register classes
# -------------------------

@dataclass
class IF_ID:
    valid: bool = False
    pc: int = 0
    next_pc: int = 0
    instr_word: int = 0


@dataclass
class CtrlSignals:
    # These are the control signals created in Decode and carried through the pipeline.
    RegWrite: int = 0
    branch_type: int = 0    # 0 = no branch, 1 = beq, 2 = bne
    MemRead: int = 0
    MemWrite: int = 0
    MemtoReg: int = 0       # 0 = ALU result, 1 = memory data, 2 = PC+4
    ALUSrc: int = 0         # 0 = rs2, 1 = immediate
    Jump: int = 0           # jal
    JumpReg: int = 0        # jalr


@dataclass
class ID_EX:
    valid: bool = False
    ctrl: CtrlSignals = field(default_factory=CtrlSignals)
    pc: int = 0
    next_pc: int = 0
    rs1_val: int = 0
    rs2_val: int = 0
    imm: int = 0
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    alu_ctrl: int = 0
    mem_funct3: int = 0
    mnemonic: str = ""


@dataclass
class EX_MEM:
    valid: bool = False
    ctrl: CtrlSignals = field(default_factory=CtrlSignals)
    next_pc: int = 0
    alu_result: int = 0
    store_data: int = 0
    rd: int = 0
    branch_taken: bool = False
    jump_taken: bool = False
    branch_target: int = 0
    jump_target: int = 0
    mem_funct3: int = 0
    mnemonic: str = ""


@dataclass
class MEM_WB:
    valid: bool = False
    ctrl: CtrlSignals = field(default_factory=CtrlSignals)
    next_pc: int = 0
    alu_result: int = 0
    mem_data: int = 0
    rd: int = 0
    store_address: Optional[int] = None
    store_value: Optional[int] = None
    mnemonic: str = ""


if_id = IF_ID()
id_ex = ID_EX()
ex_mem = EX_MEM()
mem_wb = MEM_WB()


# -------------------------
# Small helper functions
# -------------------------

def extract_bits(bin_str: str, high: int, low: int) -> str:
    """Return bits [high:low] using RISC-V bit numbering."""
    return bin_str[31 - high: 32 - low]


def sign_extend(value: int, bits: int) -> int:
    """Sign-extend a number that originally used `bits` bits."""
    if value & (1 << (bits - 1)):
        return value - (1 << bits)
    return value


def signed32(value: int) -> int:
    """Treat a 32-bit pattern as a signed integer."""
    return value if value < (1 << 31) else value - (1 << 32)


def to_hex32(value: int) -> str:
    """Print values like the project sample output: 0x10, 0x2f, etc."""
    return f"0x{value & 0xFFFFFFFF:x}"


def reg_name(idx: int) -> str:
    """Choose x# names or ABI names depending on the sample being run."""
    if USE_ABI_NAMES:
        return ABI_NAMES.get(idx, f"x{idx}")
    return f"x{idx}"


def parse_instruction_line(line: str) -> int:
    """Accept either binary strings or hex instruction words."""
    text = line.strip()
    if not text:
        raise ValueError("empty line")
    if set(text) <= {"0", "1"} and len(text) <= 32:
        return int(text, 2)
    return int(text.lower().replace("0x", ""), 16)


def instruction_to_bin(instr: int) -> str:
    return format(instr & 0xFFFFFFFF, "032b")


# -------------------------
# Control Unit and ALU Control
# -------------------------

def ALUControl(alu_op: int, funct3: int, funct7: int, opcode: int) -> int:
    """Convert ALUOp + funct fields into the ALU operation code."""

    if alu_op == 0b00:
        return 0b0010       # ADD, used for load/store address and jumps

    if alu_op == 0b01:
        return 0b0110       # SUB, used for beq/bne comparison

    if alu_op == 0b10:
        if opcode == 0x33:  # R-type
            if funct3 == 0x0 and funct7 == 0x00: return 0b0010   # add
            if funct3 == 0x0 and funct7 == 0x20: return 0b0110   # sub
            if funct3 == 0x7 and funct7 == 0x00: return 0b0000   # and
            if funct3 == 0x6 and funct7 == 0x00: return 0b0001   # or
            if funct3 == 0x2 and funct7 == 0x00: return 0b0101   # slt
            if funct3 == 0x1 and funct7 == 0x00: return 0b0011   # sll
            if funct3 == 0x5 and funct7 == 0x00: return 0b0100   # srl

        if opcode == 0x13:  # I-type arithmetic
            if funct3 == 0x0: return 0b0010   # addi
            if funct3 == 0x7: return 0b0000   # andi
            if funct3 == 0x6: return 0b0001   # ori

    raise ValueError(
        f"Unsupported ALUControl: ALUOp={alu_op:02b} opcode=0x{opcode:02x} "
        f"funct3={funct3} funct7={funct7}"
    )


def ControlUnit(opcode: int, funct3: int = 0, funct7: int = 0):
    """Create the control signals for one decoded instruction."""
    ctrl = CtrlSignals()
    alu_op = 0

    if opcode == 0x03:              # lw / lb
        ctrl.RegWrite = 1
        ctrl.MemRead = 1
        ctrl.MemtoReg = 1
        ctrl.ALUSrc = 1
        alu_op = 0b00

    elif opcode == 0x23:            # sw / sb
        ctrl.MemWrite = 1
        ctrl.ALUSrc = 1
        alu_op = 0b00

    elif opcode == 0x63:            # beq / bne
        if funct3 == 0x0:
            ctrl.branch_type = 1
        elif funct3 == 0x1:
            ctrl.branch_type = 2
        else:
            raise ValueError(f"Unsupported branch funct3={funct3}")
        alu_op = 0b01

    elif opcode == 0x33:            # R-type arithmetic/logic
        ctrl.RegWrite = 1
        ctrl.ALUSrc = 0
        alu_op = 0b10

    elif opcode == 0x13:            # addi / andi / ori
        ctrl.RegWrite = 1
        ctrl.ALUSrc = 1
        alu_op = 0b10

    elif opcode == 0x6F:            # jal
        ctrl.Jump = 1
        ctrl.RegWrite = 1
        ctrl.MemtoReg = 2
        alu_op = 0b00

    elif opcode == 0x67:            # jalr
        ctrl.JumpReg = 1
        ctrl.RegWrite = 1
        ctrl.ALUSrc = 1
        ctrl.MemtoReg = 2
        alu_op = 0b00

    else:
        raise ValueError(f"Unsupported opcode: 0x{opcode:02x}")

    alu_ctrl = ALUControl(alu_op, funct3, funct7, opcode)
    return ctrl, alu_ctrl


# -------------------------
# Decode helper
# -------------------------

def decode_fields(instr_word: int) -> dict:
    """Break one 32-bit instruction into opcode, registers, immediate, and controls."""
    b = instruction_to_bin(instr_word)
    opc = int(extract_bits(b, 6, 0), 2)

    rd = rs1 = rs2 = funct3 = funct7 = imm = 0
    mnemonic = "nop"

    if opc == 0x33:                 # R-type
        rd = int(extract_bits(b, 11, 7), 2)
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1 = int(extract_bits(b, 19, 15), 2)
        rs2 = int(extract_bits(b, 24, 20), 2)
        funct7 = int(extract_bits(b, 31, 25), 2)
        mnemonic = {
            (0x0, 0x00): "add",
            (0x0, 0x20): "sub",
            (0x7, 0x00): "and",
            (0x6, 0x00): "or",
            (0x2, 0x00): "slt",
            (0x1, 0x00): "sll",
            (0x5, 0x00): "srl",
        }.get((funct3, funct7), "r-type")

    elif opc == 0x13:               # I-type arithmetic
        rd = int(extract_bits(b, 11, 7), 2)
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1 = int(extract_bits(b, 19, 15), 2)
        imm = sign_extend(int(extract_bits(b, 31, 20), 2), 12)
        mnemonic = {0: "addi", 7: "andi", 6: "ori"}.get(funct3, "i-arith")

    elif opc == 0x03:               # lw / lb
        rd = int(extract_bits(b, 11, 7), 2)
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1 = int(extract_bits(b, 19, 15), 2)
        imm = sign_extend(int(extract_bits(b, 31, 20), 2), 12)
        if funct3 == 0x2:
            mnemonic = "lw"
        elif funct3 == 0x0:
            mnemonic = "lb"
        else:
            raise ValueError(f"Unsupported load funct3={funct3}")

    elif opc == 0x23:               # sw / sb
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1 = int(extract_bits(b, 19, 15), 2)
        rs2 = int(extract_bits(b, 24, 20), 2)
        imm = sign_extend(
            (int(extract_bits(b, 31, 25), 2) << 5) |
            int(extract_bits(b, 11, 7), 2),
            12,
        )
        if funct3 == 0x2:
            mnemonic = "sw"
        elif funct3 == 0x0:
            mnemonic = "sb"
        else:
            raise ValueError(f"Unsupported store funct3={funct3}")

    elif opc == 0x63:               # beq / bne
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1 = int(extract_bits(b, 19, 15), 2)
        rs2 = int(extract_bits(b, 24, 20), 2)
        imm12 = int(extract_bits(b, 31, 31), 2)
        im105 = int(extract_bits(b, 30, 25), 2)
        imm41 = int(extract_bits(b, 11, 8), 2)
        imm11 = int(extract_bits(b, 7, 7), 2)
        imm = sign_extend((imm12 << 11) | (imm11 << 10) | (im105 << 4) | imm41, 12)
        if funct3 == 0x0:
            mnemonic = "beq"
        elif funct3 == 0x1:
            mnemonic = "bne"
        else:
            raise ValueError(f"Unsupported branch funct3={funct3}")

    elif opc == 0x6F:               # jal
        rd = int(extract_bits(b, 11, 7), 2)
        imm20 = int(extract_bits(b, 31, 31), 2)
        im10_1 = int(extract_bits(b, 30, 21), 2)
        imm11 = int(extract_bits(b, 20, 20), 2)
        im19_12 = int(extract_bits(b, 19, 12), 2)
        raw = (imm20 << 19) | (im19_12 << 11) | (imm11 << 10) | im10_1
        imm = sign_extend(raw, 20) << 1
        mnemonic = "jal"

    elif opc == 0x67:               # jalr
        rd = int(extract_bits(b, 11, 7), 2)
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1 = int(extract_bits(b, 19, 15), 2)
        imm = sign_extend(int(extract_bits(b, 31, 20), 2), 12)
        mnemonic = "jalr"

    else:
        raise ValueError(f"Unsupported opcode: 0x{opc:02x}")

    ctrl, alu_ctrl = ControlUnit(opc, funct3, funct7)

    return {
        "opcode": opc,
        "mnemonic": mnemonic,
        "rd": rd,
        "rs1": rs1,
        "rs2": rs2,
        "funct3": funct3,
        "funct7": funct7,
        "imm": imm,
        "ctrl": ctrl,
        "alu_ctrl": alu_ctrl,
        "mem_funct3": funct3,
    }


# -------------------------
# Hazard detection
# -------------------------

def _writes_rd(valid: bool, ctrl: CtrlSignals, rd: int) -> bool:
    return valid and ctrl.RegWrite and rd != 0


def raw_hazard(if_id_l: IF_ID, id_ex_l: ID_EX, ex_mem_l: EX_MEM) -> bool:
    """Return True when the instruction in ID needs a register not written back yet."""
    if not if_id_l.valid:
        return False

    try:
        f = decode_fields(if_id_l.instr_word)
    except Exception:
        return False

    opcode = f["opcode"]
    needed_regs = set()

    if f["rs1"] != 0:
        needed_regs.add(f["rs1"])

    # R-type, stores, and branches also read rs2.
    if opcode in (0x33, 0x23, 0x63) and f["rs2"] != 0:
        needed_regs.add(f["rs2"])

    if not needed_regs:
        return False

    if _writes_rd(id_ex_l.valid, id_ex_l.ctrl, id_ex_l.rd):
        if id_ex_l.rd in needed_regs:
            return True

    if _writes_rd(ex_mem_l.valid, ex_mem_l.ctrl, ex_mem_l.rd):
        if ex_mem_l.rd in needed_regs:
            return True

    return False


# -------------------------
# Pipeline stages
# -------------------------

def stage_IF(program: List[int], pc_in: int) -> tuple:
    """Fetch the instruction at pc and compute pc + 4."""
    index = pc_in // 4

    if index < 0 or index >= len(program):
        return IF_ID(valid=False), pc_in

    instr_word = program[index]
    next_pc = pc_in + 4
    return IF_ID(valid=True, pc=pc_in, next_pc=next_pc, instr_word=instr_word), next_pc


def stage_ID(if_id_in: IF_ID) -> ID_EX:
    """Decode the instruction and read source registers from rf."""
    if not if_id_in.valid:
        return ID_EX(valid=False)

    try:
        f = decode_fields(if_id_in.instr_word)
    except Exception:
        # Keep the original project behavior: unsupported instructions act like bubbles.
        return ID_EX(valid=False)

    return ID_EX(
        valid=True,
        ctrl=f["ctrl"],
        pc=if_id_in.pc,
        next_pc=if_id_in.next_pc,
        rs1_val=rf[f["rs1"]],
        rs2_val=rf[f["rs2"]],
        imm=f["imm"],
        rd=f["rd"],
        rs1=f["rs1"],
        rs2=f["rs2"],
        alu_ctrl=f["alu_ctrl"],
        mem_funct3=f["mem_funct3"],
        mnemonic=f["mnemonic"],
    )


def stage_EX(id_ex_in: ID_EX) -> EX_MEM:
    """Run the ALU and decide branch/jump targets."""
    if not id_ex_in.valid:
        return EX_MEM(valid=False)

    ctrl = id_ex_in.ctrl
    op_a = id_ex_in.rs1_val
    op_b = id_ex_in.imm if ctrl.ALUSrc else id_ex_in.rs2_val

    alu_ctrl = id_ex_in.alu_ctrl
    if alu_ctrl == 0b0000:
        alu_result = op_a & op_b
    elif alu_ctrl == 0b0001:
        alu_result = op_a | op_b
    elif alu_ctrl == 0b0010:
        alu_result = op_a + op_b
    elif alu_ctrl == 0b0011:
        alu_result = (op_a << (op_b & 0x1F)) & 0xFFFFFFFF
    elif alu_ctrl == 0b0100:
        alu_result = (op_a & 0xFFFFFFFF) >> (op_b & 0x1F)
    elif alu_ctrl == 0b0101:
        alu_result = 1 if signed32(op_a) < signed32(op_b) else 0
    elif alu_ctrl == 0b0110:
        alu_result = op_a - op_b
    else:
        alu_result = 0

    alu_zero = (alu_result == 0)

    # Branch target follows the project slide formula: PC+4 + (imm << 1).
    branch_target = id_ex_in.next_pc + (id_ex_in.imm << 1)

    if ctrl.branch_type == 1:       # beq
        branch_taken = bool(alu_zero)
    elif ctrl.branch_type == 2:     # bne
        branch_taken = bool(not alu_zero)
    else:
        branch_taken = False

    jump_target = 0
    if ctrl.Jump:                   # jal
        jump_target = id_ex_in.pc + id_ex_in.imm
    elif ctrl.JumpReg:              # jalr
        jump_target = (id_ex_in.rs1_val + id_ex_in.imm) & ~1

    return EX_MEM(
        valid=True,
        ctrl=ctrl,
        next_pc=id_ex_in.next_pc,
        alu_result=alu_result,
        store_data=id_ex_in.rs2_val,
        rd=id_ex_in.rd,
        branch_taken=branch_taken,
        jump_taken=bool(ctrl.Jump or ctrl.JumpReg),
        branch_target=branch_target,
        jump_target=jump_target,
        mem_funct3=id_ex_in.mem_funct3,
        mnemonic=id_ex_in.mnemonic,
    )


def stage_MEM(ex_mem_in: EX_MEM) -> MEM_WB:
    """Read/write data memory for lw/lb/sw/sb."""
    if not ex_mem_in.valid:
        return MEM_WB(valid=False)

    ctrl = ex_mem_in.ctrl
    mem_data = 0
    store_address = None
    store_value = None

    addr = ex_mem_in.alu_result & 0xFFFFFFFF
    word_idx = addr // 4
    byte_lane = addr % 4

    if ctrl.MemRead:
        if ex_mem_in.mem_funct3 == 0x0:       # lb
            raw_word = d_mem[word_idx]
            raw_byte = (raw_word >> (byte_lane * 8)) & 0xFF
            mem_data = sign_extend(raw_byte, 8)
        else:                                 # lw
            mem_data = d_mem[word_idx]

    if ctrl.MemWrite:
        if ex_mem_in.mem_funct3 == 0x0:       # sb
            raw_word = d_mem[word_idx]
            byte_val = ex_mem_in.store_data & 0xFF
            shift = byte_lane * 8
            mask = (~(0xFF << shift)) & 0xFFFFFFFF
            d_mem[word_idx] = (raw_word & mask) | (byte_val << shift)
        else:                                 # sw
            d_mem[word_idx] = ex_mem_in.store_data & 0xFFFFFFFF

        store_address = addr
        store_value = d_mem[word_idx]

    return MEM_WB(
        valid=True,
        ctrl=ctrl,
        next_pc=ex_mem_in.next_pc,
        alu_result=ex_mem_in.alu_result,
        mem_data=mem_data,
        rd=ex_mem_in.rd,
        store_address=store_address,
        store_value=store_value,
        mnemonic=ex_mem_in.mnemonic,
    )


def stage_WB(mem_wb_in: MEM_WB) -> List[str]:
    """Write the final result to rf and build the output messages."""
    if not mem_wb_in.valid:
        return []

    ctrl = mem_wb_in.ctrl
    messages = []

    if ctrl.RegWrite and mem_wb_in.rd != 0:
        if ctrl.MemtoReg == 1:
            wb_val = mem_wb_in.mem_data
        elif ctrl.MemtoReg == 2:
            wb_val = mem_wb_in.next_pc
        else:
            wb_val = mem_wb_in.alu_result

        rf[mem_wb_in.rd] = wb_val & 0xFFFFFFFF
        messages.append(f"{reg_name(mem_wb_in.rd)} is modified to {to_hex32(rf[mem_wb_in.rd])}")

    if mem_wb_in.store_address is not None:
        messages.append(
            f"memory {to_hex32(mem_wb_in.store_address)} is modified to "
            f"{to_hex32(mem_wb_in.store_value)}"
        )

    rf[0] = 0
    return messages


# -------------------------
# Main pipeline loop
# -------------------------

def run_pipeline(program: List[int]) -> None:
    """Run the program through the five-stage pipeline."""
    global pc, total_clock_cycles, if_id, id_ex, ex_mem, mem_wb

    program_done = False
    next_sequential_pc = pc

    while True:
        # 1. WB happens first so Decode can see the newest register values.
        wb_messages = stage_WB(mem_wb)
        wb_had_valid = mem_wb.valid

        # 2. Check whether the instruction in ID must wait for a register value.
        stall = raw_hazard(if_id, id_ex, ex_mem)

        # 3. Let MEM and EX work using the old pipeline registers.
        new_mem_wb = stage_MEM(ex_mem)
        new_ex_mem = stage_EX(id_ex)

        # 4. Branches/jumps are resolved in EX, so wrong-path instructions get flushed.
        flush = False
        new_pc = pc
        if new_ex_mem.valid and (new_ex_mem.branch_taken or new_ex_mem.jump_taken):
            flush = True
            new_pc = (
                new_ex_mem.jump_target
                if new_ex_mem.jump_taken
                else new_ex_mem.branch_target
            ) & 0xFFFFFFFF

        # 5. Decode/fetch either advance, stall, or flush.
        if stall or flush:
            new_id_ex = ID_EX(valid=False)
        else:
            new_id_ex = stage_ID(if_id)

        if flush:
            new_if_id = IF_ID(valid=False)
        elif stall:
            new_if_id = if_id
        elif program_done:
            new_if_id = IF_ID(valid=False)
        else:
            new_if_id, next_sequential_pc = stage_IF(program, pc)
            if not new_if_id.valid:
                program_done = True

        # 6. Commit the new pipeline registers together, like a clock edge.
        mem_wb = new_mem_wb
        ex_mem = new_ex_mem
        id_ex = new_id_ex
        if_id = new_if_id

        # 7. Update PC according to flush/stall/normal execution.
        if flush:
            pc = new_pc
        elif stall:
            pass
        elif not program_done:
            pc = next_sequential_pc

        total_clock_cycles += 1

        if wb_had_valid:
            print(f"total_clock_cycles {total_clock_cycles} :")
            for msg in wb_messages:
                print(msg)
            print(f"pc is modified to {to_hex32(pc)}")

        all_empty = (
            not if_id.valid and not id_ex.valid and
            not ex_mem.valid and not mem_wb.valid
        )
        if program_done and all_empty:
            break

    print("program terminated:")
    print(f"total execution time is {total_clock_cycles} cycles")


# -------------------------
# Initialization
# -------------------------

def _reset_pipeline() -> None:
    global if_id, id_ex, ex_mem, mem_wb
    if_id = IF_ID()
    id_ex = ID_EX()
    ex_mem = EX_MEM()
    mem_wb = MEM_WB()


def initialize_section1() -> None:
    """Initial state for sample_part1.txt."""
    global pc, total_clock_cycles, rf, d_mem, USE_ABI_NAMES
    USE_ABI_NAMES = False
    pc = 0
    total_clock_cycles = 0
    rf = [0] * 32
    d_mem = [0] * 64

    rf[1] = 0x20
    rf[2] = 0x5
    rf[10] = 0x70
    rf[11] = 0x4
    d_mem[0x70 // 4] = 0x5
    d_mem[0x74 // 4] = 0x10

    _reset_pipeline()


def initialize_section2() -> None:
    """Initial state for sample_part2.txt."""
    global pc, total_clock_cycles, rf, d_mem, USE_ABI_NAMES
    USE_ABI_NAMES = True
    pc = 0
    total_clock_cycles = 0
    rf = [0] * 32
    d_mem = [0] * 64

    rf[8] = 0x20     # s0
    rf[10] = 0x5     # a0
    rf[11] = 0x2     # a1
    rf[12] = 0xa     # a2
    rf[13] = 0xf     # a3

    _reset_pipeline()


def initialize_extra() -> None:
    """Initial state for an optional extra-instruction sample."""
    global pc, total_clock_cycles, rf, d_mem, USE_ABI_NAMES
    USE_ABI_NAMES = False
    pc = 0
    total_clock_cycles = 0
    rf = [0] * 32
    d_mem = [0] * 64

    rf[1] = 0x20
    rf[2] = 0x4
    rf[3] = 0xAB
    rf[4] = 0x3
    rf[5] = 0x7
    rf[6] = 0x3
    d_mem[0x20 // 4] = 0xDEADBEEF

    _reset_pipeline()


# -------------------------
# File loading and main
# -------------------------

def load_program(filename: str) -> List[int]:
    """Load instruction words from a text file."""
    program = []
    with open(filename, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            program.append(parse_instruction_line(line))
    return program


def main() -> None:
    filename = input("Enter the program file name to run:\n").strip()

    if "part2" in filename.lower() or "section2" in filename.lower():
        initialize_section2()
    elif "extra" in filename.lower() or "part3" in filename.lower():
        initialize_extra()
    else:
        initialize_section1()

    program = load_program(filename)
    run_pipeline(program)


if __name__ == "__main__":
    main()
