#!/usr/bin/env python3
"""
CSE 140 Project – Extra Credit: 5-Stage Pipelined RISC-V CPU
=============================================================
Supported instructions (17 total):

  Section 1 (10): lw, sw, add, addi, sub, and, andi, or, ori, beq
  Section 2  (2): jal, jalr
  Extra Duty (5): lb, sb, bne, sll, srl

Pipeline stages : IF → ID → EX → MEM → WB
Stage registers : if_id, id_ex, ex_mem, mem_wb

Hazard handling (no forwarding unit, no branch predictor):
  * RAW hazard  – stall PC + IF/ID; insert bubble into ID/EX
  * Branch/Jump – flush IF/ID + ID/EX after EX resolves target (2-cycle penalty)
  * Flush always overrides stall when both occur in the same cycle

──────────────────────────────────────────────────────────────
HOW TO READ THE SECTION 1 RUBRIC MARKERS IN THIS FILE
──────────────────────────────────────────────────────────────
Every rubric item from Section 1 is marked with a banner like:
  # ╔══ SECTION 1 RUBRIC: <Item Name> ══╗
and ends with:
  # ╚══ END: <Item Name> ══╝

The items and where to find them:
  1. Fetch()       → stage_IF()          (~line 490)
  2. Decode()      → stage_ID() +        (~line 505)
                     decode_fields()     (~line 310)
  3. Execute()     → stage_EX()          (~line 540)
  4. Mem()         → stage_MEM()         (~line 595)
  5. Writeback()   → stage_WB()          (~line 655)
  6. ControlUnit() → ControlUnit() +     (~line 235)
                     ALUControl()        (~line 205)
  7. Register file → global `rf`         (~line  72)
  8. Data memory   → global `d_mem`      (~line  73)
  9. Global pc     → global `pc`         (~line  68)
 10. next_pc       → global / IF latch   (~line  68)
 11. branch_target → EX_MEM latch        (~line 128)
 12. alu_zero      → local in stage_EX   (~line 565)
 13. total_clock_cycles → global         (~line  70)
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ╔══════════════════════════════════════════════════════════════╗
# ║  ABI register name map                                       ║
# ║  Used by Writeback to print human-readable register names    ║
# ║  e.g.  x1 → "ra",  x10 → "a0",  x8 → "s0"                 ║
# ╚══════════════════════════════════════════════════════════════╝
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


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: Global Architectural State               ║
# ║                                                              ║
# ║  The spec requires these named global variables:            ║
# ║    • pc               – current program counter             ║
# ║    • next_pc          – pc + 4 (stored inside IF_ID latch   ║
# ║                         and propagated through the pipeline) ║
# ║    • branch_target    – computed in EX, stored in EX_MEM    ║
# ║    • alu_zero         – 1-bit zero flag, local to stage_EX  ║
# ║    • total_clock_cycles – incremented once per cycle        ║
# ║    • rf[32]           – 32-entry integer register file      ║
# ║    • d_mem[64]        – 64-entry data memory (word-indexed) ║
# ║    • RegWrite, Branch, MemRead, MemWrite, MemtoReg,         ║
# ║      ALUSrc, ALUOp    – carried inside CtrlSignals bundle   ║
# ╚══════════════════════════════════════════════════════════════╝

# Program counter: points to the instruction currently being fetched.
# Initialized to 0; updated every cycle (advance, hold on stall, or redirect on branch/jump).
pc = 0

# Cycle counter: incremented once at the end of every pipeline clock step.
total_clock_cycles = 0

# Register file: 32 general-purpose 32-bit registers.
# x0 (index 0) is hardwired to zero and re-zeroed after every Writeback.
rf = [0] * 32

# Data memory: 64 word-sized slots (each slot = 4 bytes).
# Addressed by byte address; word index = byte_addr // 4.
# e.g. address 0x70 maps to d_mem[0x70 // 4] = d_mem[28].
d_mem = [0] * 64


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: Pipeline Stage Registers (Latches)       ║
# ║                                                              ║
# ║  Required by the Extra Credit spec:                         ║
# ║    if_id, id_ex, ex_mem, mem_wb                             ║
# ║  Each is a dataclass holding every signal that stage        ║
# ║  produces.  `valid=False` marks a bubble (NOP).             ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class IF_ID:
    """
    Latch between Instruction Fetch and Instruction Decode.
    Holds the raw instruction word and the PC values for this instruction.
    """
    valid:      bool = False   # False = bubble (no real instruction here)
    pc:         int  = 0       # PC of this instruction (used by JAL for jump target)
    next_pc:    int  = 0       # PC + 4 (passed forward as potential return address)
    instr_word: int  = 0       # 32-bit instruction fetched from program memory


@dataclass
class CtrlSignals:
    """
    Bundle of all control signals generated by ControlUnit().
    Carried through every pipeline stage so each stage uses the
    signals that belong to ITS instruction, not a later one.

    SECTION 1 RUBRIC: ControlUnit() signals
      RegWrite    – 1 = write result back to rd in the register file
      branch_type – 0=none, 1=BEQ, 2=BNE  (replaces the spec's Boolean Branch)
      MemRead     – 1 = read data memory this cycle (lw, lb)
      MemWrite    – 1 = write data memory this cycle (sw, sb)
      MemtoReg    – selects Writeback source:
                    0 = ALU result  (R-type, I-type arithmetic)
                    1 = memory data (lw, lb)
                    2 = PC+4        (JAL, JALR return address)  [Section 2]
      ALUSrc      – 0 = use rs2_val as ALU operand B
                    1 = use sign-extended immediate as operand B
      Jump        – 1 = JAL instruction   [Section 2]
      JumpReg     – 1 = JALR instruction  [Section 2]
    """
    RegWrite:    int = 0
    branch_type: int = 0   # 0=none, 1=BEQ, 2=BNE
    MemRead:     int = 0
    MemWrite:    int = 0
    MemtoReg:    int = 0   # 0=ALU, 1=Mem, 2=PC+4
    ALUSrc:      int = 0
    Jump:        int = 0   # Section 2: JAL
    JumpReg:     int = 0   # Section 2: JALR


@dataclass
class ID_EX:
    """
    Latch between Instruction Decode and Execute.
    Carries decoded register values, the immediate, and control signals
    into the EX stage so EX can operate without touching global state.
    """
    valid:      bool        = False
    ctrl:       CtrlSignals = field(default_factory=CtrlSignals)
    pc:         int  = 0       # PC of this instruction (JAL needs it)
    next_pc:    int  = 0       # PC+4 (branch target base; also JAL/JALR return addr)
    rs1_val:    int  = 0       # value read from rf[rs1] during Decode
    rs2_val:    int  = 0       # value read from rf[rs2] during Decode
    imm:        int  = 0       # sign-extended immediate from Decode
    rd:         int  = 0       # destination register index
    rs1:        int  = 0       # source register 1 index (kept for hazard detection)
    rs2:        int  = 0       # source register 2 index (kept for hazard detection)
    alu_ctrl:   int  = 0       # 4-bit ALU operation code from ALUControl()
    mem_funct3: int  = 0       # raw funct3 forwarded to MEM for byte/word selection
    mnemonic:   str  = ""      # human-readable name for debugging


@dataclass
class EX_MEM:
    """
    Latch between Execute and Memory Access.

    SECTION 1 RUBRIC: branch_target global variable
      branch_target is stored here after EX computes it.
      The pipeline loop reads branch_taken + branch_target to redirect PC.
    """
    valid:         bool        = False
    ctrl:          CtrlSignals = field(default_factory=CtrlSignals)
    next_pc:       int  = 0       # PC+4, forwarded for JAL/JALR return-address WB
    alu_result:    int  = 0       # result of the ALU operation
    store_data:    int  = 0       # rs2_val, data to write to memory (sw, sb)
    rd:            int  = 0       # destination register index
    branch_taken:  bool = False   # True when beq/bne condition is met
    jump_taken:    bool = False   # True for JAL or JALR (always jumps)
    branch_target: int  = 0       # PC to jump to on a taken branch
    jump_target:   int  = 0       # PC to jump to for JAL/JALR  [Section 2]
    mem_funct3:    int  = 0       # forwarded to MEM stage
    mnemonic:      str  = ""


@dataclass
class MEM_WB:
    """Latch between Memory Access and Write Back."""
    valid:         bool        = False
    ctrl:          CtrlSignals = field(default_factory=CtrlSignals)
    next_pc:       int  = 0       # PC+4, used by WB for JAL/JALR return-address write
    alu_result:    int  = 0       # ALU result (written back for non-memory instructions)
    mem_data:      int  = 0       # data loaded from memory (written back for lw, lb)
    rd:            int  = 0       # destination register index
    store_address: Optional[int] = None   # byte address of sw/sb (None = no store)
    store_value:   Optional[int] = None   # full word value written to d_mem
    mnemonic:      str  = ""


# Global instances of each pipeline stage register.
# All start as bubbles (valid=False) before the first instruction enters the pipeline.
if_id  = IF_ID()
id_ex  = ID_EX()
ex_mem = EX_MEM()
mem_wb = MEM_WB()


# ══════════════════════════════════════════════════════════════
#  Utility helpers
# ══════════════════════════════════════════════════════════════

def extract_bits(bin_str: str, high: int, low: int) -> str:
    """
    Extract bits [high:low] (inclusive, RISC-V bit numbering) from a
    32-character binary string.  Bit 31 is bin_str[0], bit 0 is bin_str[31].
    """
    return bin_str[31 - high : 32 - low]


def sign_extend(value: int, bits: int) -> int:
    """
    Sign-extend a `bits`-wide unsigned integer to a Python (arbitrary-width) int.
    If the MSB of `value` is 1, the number is negative in two's complement.
    """
    if value & (1 << (bits - 1)):
        return value - (1 << bits)
    return value


def signed32(value: int) -> int:
    """Interpret a 32-bit pattern as a signed integer (for SLT comparison)."""
    return value if value < (1 << 31) else value - (1 << 32)


def to_hex32(value: int) -> str:
    """Format any integer as a lowercase 32-bit hex string with 0x prefix."""
    return f"0x{value & 0xFFFFFFFF:x}"


def reg_name(idx: int) -> str:
    """Return the ABI name for a register index (e.g. 1 → 'ra', 10 → 'a0')."""
    return ABI_NAMES.get(idx, f"x{idx}")


def parse_instruction_line(line: str) -> int:
    """
    Parse one line from the program file.
    Accepts:
      • 32-character binary string  ("00000111000001010010000110000011")
      • 8-digit hex without prefix  ("07052183")
      • hex with 0x prefix          ("0x07052183")
    """
    text = line.strip()
    if not text:
        raise ValueError("empty line")
    if set(text) <= {"0", "1"} and len(text) <= 32:
        return int(text, 2)          # binary string
    return int(text.lower().replace("0x", ""), 16)   # hex string


def instruction_to_bin(instr: int) -> str:
    """Convert a 32-bit integer instruction to a zero-padded binary string."""
    return format(instr & 0xFFFFFFFF, "032b")


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: ALU Control                              ║
# ║                                                              ║
# ║  ALUControl() maps (ALUOp, funct3, funct7, opcode) to a     ║
# ║  4-bit alu_ctrl code consumed by stage_EX().                ║
# ║                                                              ║
# ║  alu_ctrl encodings (per lecture slide):                    ║
# ║    0b0000  AND                                               ║
# ║    0b0001  OR                                                ║
# ║    0b0010  ADD                                               ║
# ║    0b0011  SLL  (Extra Duty – shift left logical)           ║
# ║    0b0100  SRL  (Extra Duty – shift right logical)          ║
# ║    0b0101  SLT  (set less than)                             ║
# ║    0b0110  SUB                                               ║
# ╚══════════════════════════════════════════════════════════════╝

def ALUControl(alu_op: int, funct3: int, funct7: int, opcode: int) -> int:
    """
    ALU Control unit: converts the two-bit ALUOp signal (from ControlUnit) and
    the instruction's funct3/funct7 fields into a 4-bit alu_ctrl code.

    ALUOp truth table:
      00 → always ADD  (memory address calculation for lw/sw/lb/sb, JAL, JALR)
      01 → always SUB  (comparison for beq/bne: zero flag tells us equal or not)
      10 → look at funct3/funct7 to pick the operation (R-type and I-type arith)
    """
    # ── ALUOp = 00: memory/jump address calculation always uses ADD ──
    if alu_op == 0b00:
        return 0b0010   # ADD

    # ── ALUOp = 01: branch comparison always uses SUB (check zero flag) ──
    if alu_op == 0b01:
        return 0b0110   # SUB

    # ── ALUOp = 10: use funct3 / funct7 to select operation ──────────
    if alu_op == 0b10:

        if opcode == 0x33:    # R-type instructions (funct7 distinguishes add vs sub)
            if funct3 == 0x0 and funct7 == 0x00: return 0b0010   # add
            if funct3 == 0x0 and funct7 == 0x20: return 0b0110   # sub
            if funct3 == 0x7 and funct7 == 0x00: return 0b0000   # and
            if funct3 == 0x6 and funct7 == 0x00: return 0b0001   # or
            if funct3 == 0x2 and funct7 == 0x00: return 0b0101   # slt
            # ── Extra Duty instructions ───────────────────────────────
            if funct3 == 0x1 and funct7 == 0x00: return 0b0011   # sll
            if funct3 == 0x5 and funct7 == 0x00: return 0b0100   # srl

        elif opcode == 0x13:  # I-type arithmetic (funct7 not used)
            if funct3 == 0x0: return 0b0010   # addi
            if funct3 == 0x7: return 0b0000   # andi
            if funct3 == 0x6: return 0b0001   # ori

    raise ValueError(
        f"Unsupported ALUControl: ALUOp={alu_op:02b} opcode=0x{opcode:02x} "
        f"funct3={funct3} funct7={funct7}"
    )

# ╚══ END: ALU Control ══╝


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: ControlUnit()                            ║
# ║                                                              ║
# ║  Receives the 7-bit opcode (and funct3 for disambiguation). ║
# ║  Sets all 7 control signals as per the lecture truth table. ║
# ║  Then calls ALUControl() to get the 4-bit alu_ctrl code.   ║
# ║                                                              ║
# ║  Returns (CtrlSignals, alu_ctrl) so both are available      ║
# ║  to Decode and passed forward through the pipeline.         ║
# ╚══════════════════════════════════════════════════════════════╝

def ControlUnit(opcode: int, funct3: int = 0, funct7: int = 0):
    """
    Main control unit.  Maps opcode → control signal bundle + alu_ctrl.

    Section 1 instructions and their signal settings:
      lw   (0x03): RegWrite=1, MemRead=1, MemtoReg=1, ALUSrc=1, ALUOp=00
      sw   (0x23): MemWrite=1, ALUSrc=1, ALUOp=00
      beq  (0x63): branch_type=1, ALUOp=01
      add/sub/and/or/slt (0x33): RegWrite=1, ALUSrc=0, ALUOp=10
      addi/andi/ori (0x13): RegWrite=1, ALUSrc=1, ALUOp=10

    Section 2 additions:
      jal  (0x6F): Jump=1, RegWrite=1, MemtoReg=2 (return addr)
      jalr (0x67): JumpReg=1, RegWrite=1, ALUSrc=1, MemtoReg=2

    Extra Duty additions (lb/sb share opcodes with lw/sw; bne/sll/srl extend
    existing opcode groups — no new signals needed beyond branch_type=2):
      lb   (0x03, funct3=0): same signals as lw; MEM stage uses funct3 to read byte
      sb   (0x23, funct3=0): same signals as sw; MEM stage uses funct3 to write byte
      bne  (0x63, funct3=1): branch_type=2 instead of 1; ALUOp=01 (SUB same as BEQ)
      sll  (0x33, funct3=1): same as other R-type; ALUControl picks 0b0011
      srl  (0x33, funct3=5): same as other R-type; ALUControl picks 0b0100
    """
    ctrl   = CtrlSignals()   # start with all signals = 0
    alu_op = 0

    # ── Section 1: Load family ────────────────────────────────────────
    # lw (funct3=2) and lb (funct3=0) both use the same control signals.
    # The MEM stage uses mem_funct3 to decide byte vs word access width.
    if opcode == 0x03:
        ctrl.RegWrite = 1   # result (loaded data) goes back to register file
        ctrl.MemRead  = 1   # read from data memory
        ctrl.MemtoReg = 1   # Writeback mux: select memory data (not ALU result)
        ctrl.ALUSrc   = 1   # ALU operand B = immediate (address offset)
        alu_op = 0b00       # ALUOp=00 → ADD (compute effective address = rs1 + imm)

    # ── Section 1: Store family ───────────────────────────────────────
    # sw (funct3=2) and sb (funct3=0) both use the same control signals.
    elif opcode == 0x23:
        ctrl.MemWrite = 1   # write to data memory
        ctrl.ALUSrc   = 1   # ALU operand B = immediate (address offset)
        alu_op = 0b00       # ALUOp=00 → ADD (compute effective address = rs1 + imm)

    # ── Section 1: Branch family ──────────────────────────────────────
    # beq (funct3=0) and bne (funct3=1) differ only in which zero-flag sense is used.
    elif opcode == 0x63:
        if   funct3 == 0x0: ctrl.branch_type = 1   # BEQ: branch when alu_zero==1
        elif funct3 == 0x1: ctrl.branch_type = 2   # BNE: branch when alu_zero==0
        else: raise ValueError(f"Unsupported branch funct3={funct3}")
        alu_op = 0b01       # ALUOp=01 → SUB (rs1 - rs2; zero flag tells us equality)

    # ── Section 1: R-type (add, sub, and, or, slt, sll, srl) ─────────
    elif opcode == 0x33:
        ctrl.RegWrite = 1   # ALU result goes to register file
        ctrl.ALUSrc   = 0   # ALU operand B = rs2_val (not an immediate)
        alu_op = 0b10       # ALUOp=10 → use funct3/funct7 to pick operation

    # ── Section 1: I-type arithmetic (addi, andi, ori) ───────────────
    elif opcode == 0x13:
        ctrl.RegWrite = 1   # ALU result goes to register file
        ctrl.ALUSrc   = 1   # ALU operand B = sign-extended immediate
        alu_op = 0b10       # ALUOp=10 → use funct3 to pick operation

    # ── Section 2: JAL ───────────────────────────────────────────────
    elif opcode == 0x6F:
        ctrl.Jump     = 1   # tells EX to compute PC+imm as jump target
        ctrl.RegWrite = 1   # rd = PC+4 (return address)
        ctrl.MemtoReg = 2   # Writeback mux: select PC+4
        alu_op = 0b00       # ADD (not actually used for JAL; target computed separately)

    # ── Section 2: JALR ──────────────────────────────────────────────
    elif opcode == 0x67:
        ctrl.JumpReg  = 1   # tells EX to compute (rs1+imm)&~1 as jump target
        ctrl.RegWrite = 1   # rd = PC+4 (return address)
        ctrl.ALUSrc   = 1   # ALU operand B = immediate
        ctrl.MemtoReg = 2   # Writeback mux: select PC+4
        alu_op = 0b00       # ADD (computes rs1+imm, also used as jump target in EX)

    else:
        raise ValueError(f"Unsupported opcode: 0x{opcode:02x}")

    alu_ctrl = ALUControl(alu_op, funct3, funct7, opcode)
    return ctrl, alu_ctrl

# ╚══ END: ControlUnit() ══╝


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: Decode() – instruction field extraction  ║
# ║                                                              ║
# ║  decode_fields() is a pure helper called by both stage_ID   ║
# ║  and the hazard detector. It extracts every field from the  ║
# ║  32-bit instruction word using RISC-V bit positions and     ║
# ║  performs sign extension for immediate-bearing formats.      ║
# ╚══════════════════════════════════════════════════════════════╝

def decode_fields(instr_word: int) -> dict:
    """
    Decode a single 32-bit instruction word into all its constituent fields.

    Returns a dict containing:
      opcode, mnemonic, rd, rs1, rs2, funct3, funct7, imm,
      ctrl (CtrlSignals), alu_ctrl (int), mem_funct3 (int)

    RISC-V instruction formats handled:
      R-type  [funct7|rs2|rs1|funct3|rd|opcode]   → add,sub,and,or,slt,sll,srl
      I-type  [imm[11:0]|rs1|funct3|rd|opcode]    → addi,andi,ori,lw,lb,jalr
      S-type  [imm[11:5]|rs2|rs1|funct3|imm[4:0]|opcode] → sw, sb
      B-type  [imm[12|10:5]|rs2|rs1|funct3|imm[4:1|11]|opcode] → beq, bne
      J-type  [imm[20|10:1|11|19:12]|rd|opcode]   → jal
    """
    b   = instruction_to_bin(instr_word)     # 32-char binary string, bit31 at index 0
    opc = int(extract_bits(b, 6, 0), 2)      # bits[6:0] = opcode

    rd = rs1 = rs2 = funct3 = funct7 = imm = 0
    mnemonic = "nop"

    # ── R-type: add, sub, and, or, slt, sll (Extra), srl (Extra) ────
    if opc == 0x33:
        # Field positions for R-type:
        #   bits[11:7]  = rd       (destination register)
        #   bits[14:12] = funct3   (selects operation within the opcode group)
        #   bits[19:15] = rs1      (source register 1)
        #   bits[24:20] = rs2      (source register 2)
        #   bits[31:25] = funct7   (further distinguishes add vs sub, srl vs sra)
        rd     = int(extract_bits(b, 11,  7), 2)
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1    = int(extract_bits(b, 19, 15), 2)
        rs2    = int(extract_bits(b, 24, 20), 2)
        funct7 = int(extract_bits(b, 31, 25), 2)
        mnemonic = {
            (0x0, 0x00): "add",
            (0x0, 0x20): "sub",
            (0x7, 0x00): "and",
            (0x6, 0x00): "or",
            (0x2, 0x00): "slt",
            (0x1, 0x00): "sll",   # Extra Duty
            (0x5, 0x00): "srl",   # Extra Duty
        }.get((funct3, funct7), "r-type")

    # ── I-type arithmetic: addi, andi, ori ──────────────────────────
    elif opc == 0x13:
        # Field positions for I-type:
        #   bits[11:7]  = rd
        #   bits[14:12] = funct3
        #   bits[19:15] = rs1
        #   bits[31:20] = imm[11:0]  (sign-extended to 32 bits)
        rd     = int(extract_bits(b, 11,  7), 2)
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1    = int(extract_bits(b, 19, 15), 2)
        imm    = sign_extend(int(extract_bits(b, 31, 20), 2), 12)
        mnemonic = {0: "addi", 7: "andi", 6: "ori"}.get(funct3, "i-arith")

    # ── Load family: lw (funct3=2) and lb (funct3=0, Extra Duty) ────
    elif opc == 0x03:
        # Same I-type layout as addi etc; funct3 tells us byte vs word width.
        rd     = int(extract_bits(b, 11,  7), 2)
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1    = int(extract_bits(b, 19, 15), 2)
        imm    = sign_extend(int(extract_bits(b, 31, 20), 2), 12)
        if   funct3 == 0x2: mnemonic = "lw"
        elif funct3 == 0x0: mnemonic = "lb"   # Extra Duty
        else: raise ValueError(f"Unsupported load funct3={funct3}")

    # ── Store family: sw (funct3=2) and sb (funct3=0, Extra Duty) ───
    elif opc == 0x23:
        # S-type splits the immediate across two fields:
        #   imm[11:5] at bits[31:25],  imm[4:0] at bits[11:7]
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1    = int(extract_bits(b, 19, 15), 2)
        rs2    = int(extract_bits(b, 24, 20), 2)
        imm    = sign_extend(
            (int(extract_bits(b, 31, 25), 2) << 5) | int(extract_bits(b, 11, 7), 2), 12
        )
        if   funct3 == 0x2: mnemonic = "sw"
        elif funct3 == 0x0: mnemonic = "sb"   # Extra Duty
        else: raise ValueError(f"Unsupported store funct3={funct3}")

    # ── Branch family: beq (funct3=0) and bne (funct3=1, Extra Duty) ─
    elif opc == 0x63:
        # B-type splits the immediate with bits scattered across the word:
        #   imm[12]    at bit[31]
        #   imm[10:5]  at bits[30:25]
        #   imm[4:1]   at bits[11:8]
        #   imm[11]    at bit[7]
        # Note: imm[0] is always 0 (instructions are 2-byte aligned minimum).
        # The assembled value stored in the field equals offset/2 due to encoding;
        # EX adds it as (imm << 1) on top of next_pc to get the branch target.
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1    = int(extract_bits(b, 19, 15), 2)
        rs2    = int(extract_bits(b, 24, 20), 2)
        imm12  = int(extract_bits(b, 31, 31), 2)
        im105  = int(extract_bits(b, 30, 25), 2)
        imm41  = int(extract_bits(b, 11,  8), 2)
        imm11  = int(extract_bits(b,  7,  7), 2)
        imm    = sign_extend((imm12 << 11) | (imm11 << 10) | (im105 << 4) | imm41, 12)
        if   funct3 == 0x0: mnemonic = "beq"
        elif funct3 == 0x1: mnemonic = "bne"   # Extra Duty
        else: raise ValueError(f"Unsupported branch funct3={funct3}")

    # ── JAL: J-type (Section 2) ──────────────────────────────────────
    elif opc == 0x6F:
        # J-type scatters imm[20:1] across 4 fields; imm[0] always 0.
        #   imm[20]    at bit[31]
        #   imm[10:1]  at bits[30:21]
        #   imm[11]    at bit[20]
        #   imm[19:12] at bits[19:12]
        rd      = int(extract_bits(b, 11,  7), 2)
        imm20   = int(extract_bits(b, 31, 31), 2)
        im10_1  = int(extract_bits(b, 30, 21), 2)   # imm[10:1], already positioned
        imm11   = int(extract_bits(b, 20, 20), 2)
        im19_12 = int(extract_bits(b, 19, 12), 2)
        # Reassemble: place each piece at its correct bit position
        raw     = (imm20 << 19) | (im19_12 << 11) | (imm11 << 10) | im10_1
        imm     = sign_extend(raw, 20) << 1   # <<1 converts imm[20:1] → byte offset
        mnemonic = "jal"

    # ── JALR: I-type (Section 2) ─────────────────────────────────────
    elif opc == 0x67:
        rd     = int(extract_bits(b, 11,  7), 2)
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1    = int(extract_bits(b, 19, 15), 2)
        imm    = sign_extend(int(extract_bits(b, 31, 20), 2), 12)
        mnemonic = "jalr"

    else:
        raise ValueError(f"Unsupported opcode: 0x{opc:02x}")

    ctrl, alu_ctrl = ControlUnit(opc, funct3, funct7)

    return dict(
        opcode     = opc,
        mnemonic   = mnemonic,
        rd         = rd,
        rs1        = rs1,
        rs2        = rs2,
        funct3     = funct3,
        funct7     = funct7,
        imm        = imm,
        ctrl       = ctrl,
        alu_ctrl   = alu_ctrl,
        mem_funct3 = funct3,   # raw funct3 kept for MEM-stage byte/word selection
    )

# ╚══ END: Decode() field extraction ══╝


# ══════════════════════════════════════════════════════════════
#  Hazard detection  (Extra Credit – no forwarding unit)
#
#  With no forwarding, ANY instruction in ID that reads a register
#  that EX or MEM hasn't committed yet must stall.
#
#  Stall sources checked each cycle (using OLD latches before EX runs):
#    1. id_ex (EX stage)  – its result won't be in rf for 2 more cycles.
#    2. ex_mem (MEM stage) – its result reaches WB next cycle, but WB in
#       THIS cycle commits the OLD mem_wb, not ex_mem. So one more stall.
# ══════════════════════════════════════════════════════════════

def _writes_rd(valid: bool, ctrl: CtrlSignals, rd: int) -> bool:
    """True if this stage will write a meaningful (non-zero) destination register."""
    return valid and ctrl.RegWrite and rd != 0


def raw_hazard(if_id_l: IF_ID, id_ex_l: ID_EX, ex_mem_l: EX_MEM) -> bool:
    """
    Detect a Read-After-Write (RAW) data hazard.

    Returns True (→ stall) if the instruction sitting in IF/ID needs to read
    a register that an in-flight instruction (in EX or MEM) will write but
    hasn't committed to rf yet.

    Which registers does each instruction READ?
      R-type, beq, bne, sw, sb  →  rs1 AND rs2
      I-type arith, lw, lb, jalr →  rs1 only
      jal                         →  neither (no register read)
    """
    if not if_id_l.valid:
        return False          # nothing in IF/ID → no hazard possible
    try:
        f = decode_fields(if_id_l.instr_word)
    except Exception:
        return False

    opc   = f["opcode"]
    needs = set()

    # Every instruction that reads a register reads rs1 (if non-zero)
    if f["rs1"] != 0:
        needs.add(f["rs1"])

    # R-type, stores (sw/sb), and branches (beq/bne) also read rs2
    if opc in (0x33, 0x23, 0x63) and f["rs2"] != 0:
        needs.add(f["rs2"])

    if not needs:
        return False   # instruction reads no registers → no hazard

    # Check EX stage (id_ex): result won't be in rf for 2 more cycles
    if _writes_rd(id_ex_l.valid, id_ex_l.ctrl, id_ex_l.rd):
        if id_ex_l.rd in needs:
            return True   # stall

    # Check MEM stage (ex_mem): WB this cycle commits OLD mem_wb, not ex_mem
    if _writes_rd(ex_mem_l.valid, ex_mem_l.ctrl, ex_mem_l.rd):
        if ex_mem_l.rd in needs:
            return True   # stall

    return False


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: Fetch()  →  stage_IF()                  ║
# ║                                                              ║
# ║  Reads the instruction at pc from program memory.           ║
# ║  Computes next_pc = pc + 4.                                 ║
# ║  Returns (IF_ID latch, next sequential pc).                 ║
# ║                                                              ║
# ║  PC redirection (branch/jump) is handled by the main loop   ║
# ║  AFTER EX resolves the target, and is NOT done here.        ║
# ╚══════════════════════════════════════════════════════════════╝

def stage_IF(program: List[int], pc_in: int) -> tuple:
    """
    Instruction Fetch stage.

    'program' is the list of 32-bit instruction words loaded from the input file.
    pc_in // 4 converts the byte address to a word (list) index.

    If pc_in is out of range (past the end of program), returns an invalid
    (bubble) latch so the pipeline can drain naturally.
    """
    index = pc_in // 4   # byte address → word index (each instruction is 4 bytes)

    if index < 0 or index >= len(program):
        # PC is past the last instruction: signal end-of-program with a bubble
        return IF_ID(valid=False), pc_in

    instr_word = program[index]    # fetch the 32-bit instruction word
    next_pc    = pc_in + 4         # compute PC+4 for sequential execution

    # Pack into the IF_ID latch and return it alongside the next sequential PC
    latch = IF_ID(valid=True, pc=pc_in, next_pc=next_pc, instr_word=instr_word)
    return latch, next_pc

# ╚══ END: Fetch() ══╝


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: Decode()  →  stage_ID()                 ║
# ║                                                              ║
# ║  Decodes the instruction from IF_ID, calls ControlUnit(),   ║
# ║  and reads rs1/rs2 values from the register file.           ║
# ║  Populates the ID_EX latch for the Execute stage.           ║
# ╚══════════════════════════════════════════════════════════════╝

def stage_ID(if_id_in: IF_ID) -> ID_EX:
    """
    Instruction Decode stage.

    Steps:
      1. Call decode_fields() to extract opcode, registers, immediate, etc.
      2. decode_fields() internally calls ControlUnit() → returns ctrl + alu_ctrl.
      3. Read rs1 and rs2 values from the global register file (rf).
      4. Pack everything into the ID_EX latch.

    WB runs BEFORE ID within each clock cycle, so rf already holds the most
    recently committed value — no forwarding path needed for WB→ID.
    """
    if not if_id_in.valid:
        return ID_EX(valid=False)   # bubble passes through as bubble

    try:
        f = decode_fields(if_id_in.instr_word)
    except Exception:
        return ID_EX(valid=False)   # unsupported instruction → treat as bubble

    return ID_EX(
        valid      = True,
        ctrl       = f["ctrl"],          # control signals from ControlUnit()
        pc         = if_id_in.pc,        # this instruction's PC (needed by JAL)
        next_pc    = if_id_in.next_pc,   # PC+4 (branch base; also return address)
        rs1_val    = rf[f["rs1"]],       # read rs1 from register file NOW
        rs2_val    = rf[f["rs2"]],       # read rs2 from register file NOW
        imm        = f["imm"],           # sign-extended immediate
        rd         = f["rd"],            # destination register index
        rs1        = f["rs1"],           # source reg indices kept for hazard detection
        rs2        = f["rs2"],
        alu_ctrl   = f["alu_ctrl"],      # 4-bit ALU operation code
        mem_funct3 = f["mem_funct3"],    # funct3 forwarded to MEM for byte/word select
        mnemonic   = f["mnemonic"],      # human-readable name (debug/demo only)
    )

# ╚══ END: Decode() ══╝


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: Execute()  →  stage_EX()                ║
# ║                                                              ║
# ║  Runs the ALU operation specified by alu_ctrl.              ║
# ║  Generates alu_zero (1-bit) for branch decisions.           ║
# ║  Computes branch_target = next_pc + (imm << 1).             ║
# ║  Determines branch_taken based on branch_type and alu_zero. ║
# ║  [Section 2] Computes JAL / JALR jump targets.              ║
# ╚══════════════════════════════════════════════════════════════╝

def stage_EX(id_ex_in: ID_EX) -> EX_MEM:
    """
    Execute stage.

    ALU operands:
      op_a = rs1_val  (always)
      op_b = imm      if ALUSrc == 1  (I-type, loads, stores, JALR)
           = rs2_val  if ALUSrc == 0  (R-type, branches)

    alu_ctrl truth table (4-bit code):
      0b0000 → AND
      0b0001 → OR
      0b0010 → ADD
      0b0011 → SLL  rd = rs1 << (rs2 & 0x1F)         [Extra Duty]
      0b0100 → SRL  rd = unsigned(rs1) >> (rs2 & 0x1F)[Extra Duty]
      0b0101 → SLT  rd = (signed rs1 < signed rs2) ? 1 : 0
      0b0110 → SUB  (also used for beq/bne comparison)

    alu_zero:
      Set to 1 when alu_result == 0.
      BEQ uses alu_zero==1 (subtraction result is 0 → operands are equal).
      BNE uses alu_zero==0 (subtraction result is non-zero → operands differ).

    branch_target:
      Computed as next_pc + (imm << 1).
      The B-type encoding stores imm[12:1]; <<1 appends the implicit bit-0=0.

    jump_target (Section 2):
      JAL:  PC + imm           (offset already byte-addressed from decode)
      JALR: (rs1_val + imm) & ~1  (clear bit 0 per RISC-V spec)
    """
    if not id_ex_in.valid:
        return EX_MEM(valid=False)

    ctrl = id_ex_in.ctrl

    # ── Select ALU operand B ──────────────────────────────────────────
    op_a = id_ex_in.rs1_val
    op_b = id_ex_in.imm if ctrl.ALUSrc else id_ex_in.rs2_val

    # ── Execute ALU operation ─────────────────────────────────────────
    ac = id_ex_in.alu_ctrl
    if   ac == 0b0000: alu_result = op_a & op_b                             # AND
    elif ac == 0b0001: alu_result = op_a | op_b                             # OR
    elif ac == 0b0010: alu_result = op_a + op_b                             # ADD
    elif ac == 0b0011: alu_result = (op_a << (op_b & 0x1F)) & 0xFFFFFFFF   # SLL [Extra]
    elif ac == 0b0100: alu_result = (op_a & 0xFFFFFFFF) >> (op_b & 0x1F)   # SRL [Extra]
    elif ac == 0b0101: alu_result = 1 if signed32(op_a) < signed32(op_b) else 0  # SLT
    elif ac == 0b0110: alu_result = op_a - op_b                             # SUB
    else:              alu_result = 0

    # ── SECTION 1 RUBRIC: alu_zero ───────────────────────────────────
    # 1-bit flag: 1 when the ALU result is exactly zero.
    # Used by BEQ (branch if equal) and BNE (branch if not equal).
    alu_zero = (alu_result == 0)

    # ── SECTION 1 RUBRIC: branch_target ──────────────────────────────
    # Branch target = next_pc (PC+4) + byte offset.
    # The B-type imm field stores offset/2; <<1 recovers the byte offset.
    branch_target = id_ex_in.next_pc + (id_ex_in.imm << 1)

    # Determine whether the branch is taken
    if   ctrl.branch_type == 1: branch_taken = bool(alu_zero)       # BEQ
    elif ctrl.branch_type == 2: branch_taken = bool(not alu_zero)   # BNE [Extra Duty]
    else:                       branch_taken = False                 # not a branch

    # ── Section 2: Jump target computation ───────────────────────────
    jump_target = 0
    if ctrl.Jump:
        # JAL: jump to PC + sign-extended J-type offset (already byte-addressed)
        jump_target = id_ex_in.pc + id_ex_in.imm
    elif ctrl.JumpReg:
        # JALR: jump to (rs1 + imm), with bit 0 cleared per spec
        jump_target = (id_ex_in.rs1_val + id_ex_in.imm) & ~1

    jump_taken = bool(ctrl.Jump or ctrl.JumpReg)

    return EX_MEM(
        valid         = True,
        ctrl          = ctrl,
        next_pc       = id_ex_in.next_pc,   # forwarded for JAL/JALR return-addr WB
        alu_result    = alu_result,
        store_data    = id_ex_in.rs2_val,   # data to write for sw/sb
        rd            = id_ex_in.rd,
        branch_taken  = branch_taken,
        jump_taken    = jump_taken,
        branch_target = branch_target,
        jump_target   = jump_target,
        mem_funct3    = id_ex_in.mem_funct3,
        mnemonic      = id_ex_in.mnemonic,
    )

# ╚══ END: Execute() ══╝


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: Mem()  →  stage_MEM()                   ║
# ║                                                              ║
# ║  Accesses data memory for load and store instructions.      ║
# ║  All other instructions pass through without touching d_mem.║
# ║                                                              ║
# ║  d_mem layout:                                              ║
# ║    Each entry = one 4-byte word.                            ║
# ║    Byte address 0x00 → d_mem[0]                             ║
# ║    Byte address 0x04 → d_mem[1]   …   0x7C → d_mem[31]     ║
# ║                                                              ║
# ║  Extended for Extra Duty (lb/sb byte granularity):          ║
# ║    mem_funct3 == 0x2 → word (lw / sw)                       ║
# ║    mem_funct3 == 0x0 → byte (lb / sb)                       ║
# ╚══════════════════════════════════════════════════════════════╝

def stage_MEM(ex_mem_in: EX_MEM) -> MEM_WB:
    """
    Memory Access stage.

    For lw  : reads the full 32-bit word at the aligned word address.
    For lb  : reads one byte at byte_addr, sign-extends it to 32 bits.  [Extra]
    For sw  : writes the full 32-bit value of rs2 to the word address.
    For sb  : writes only the low 8 bits of rs2 to the byte lane,
              leaving the other 3 bytes of the word unchanged.           [Extra]

    The effective byte address comes from the ALU result (rs1 + imm).
      word_idx  = byte_addr // 4   (which d_mem entry)
      byte_lane = byte_addr  % 4   (which byte within that word; 0 = LSB)
    """
    if not ex_mem_in.valid:
        return MEM_WB(valid=False)

    ctrl          = ex_mem_in.ctrl
    mem_data      = 0             # output to Writeback (for loads)
    store_address = None          # set only if a store actually executes
    store_value   = None          # full word value written (for output message)

    addr      = ex_mem_in.alu_result & 0xFFFFFFFF   # effective byte address
    word_idx  = addr // 4                            # index into d_mem array
    byte_lane = addr  % 4                            # byte offset within the word

    # ── SECTION 1 RUBRIC: Load (MemRead) ─────────────────────────────
    if ctrl.MemRead:
        if ex_mem_in.mem_funct3 == 0x0:
            # lb [Extra Duty]: extract the target byte, then sign-extend to 32 bits.
            # (byte_lane * 8) shifts the desired byte down to bits [7:0].
            raw_word = d_mem[word_idx]
            raw_byte = (raw_word >> (byte_lane * 8)) & 0xFF
            mem_data = sign_extend(raw_byte, 8)
        else:
            # lw: read the entire 32-bit word (byte address must be word-aligned)
            mem_data = d_mem[word_idx]

    # ── SECTION 1 RUBRIC: Store (MemWrite) ───────────────────────────
    if ctrl.MemWrite:
        if ex_mem_in.mem_funct3 == 0x0:
            # sb [Extra Duty]: write only the low 8 bits of rs2 to byte_lane.
            # Build a mask that zeros only the target byte, then OR in the new byte.
            raw_word = d_mem[word_idx]
            byte_val = ex_mem_in.store_data & 0xFF
            shift    = byte_lane * 8
            mask     = (~(0xFF << shift)) & 0xFFFFFFFF   # e.g. 0xFFFFFF00 for lane 0
            d_mem[word_idx] = (raw_word & mask) | (byte_val << shift)
            store_address   = addr
            store_value     = d_mem[word_idx]
        else:
            # sw: overwrite the entire 32-bit word
            d_mem[word_idx] = ex_mem_in.store_data & 0xFFFFFFFF
            store_address   = addr
            store_value     = d_mem[word_idx]

    return MEM_WB(
        valid         = True,
        ctrl          = ctrl,
        next_pc       = ex_mem_in.next_pc,
        alu_result    = ex_mem_in.alu_result,
        mem_data      = mem_data,
        rd            = ex_mem_in.rd,
        store_address = store_address,
        store_value   = store_value,
        mnemonic      = ex_mem_in.mnemonic,
    )

# ╚══ END: Mem() ══╝


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: Writeback()  →  stage_WB()              ║
# ║                                                              ║
# ║  Writes the instruction result back to the register file.   ║
# ║  Also prints the output lines required by the spec.         ║
# ║  Increments total_clock_cycles (done in the main loop).     ║
# ║                                                              ║
# ║  MemtoReg mux selects the write-back source:                ║
# ║    0 → ALU result  (R-type, I-type arithmetic)              ║
# ║    1 → memory data (lw, lb)                                 ║
# ║    2 → PC+4        (JAL, JALR return address) [Section 2]  ║
# ╚══════════════════════════════════════════════════════════════╝

def stage_WB(mem_wb_in: MEM_WB) -> List[str]:
    """
    Write Back stage.

    Runs FIRST within each clock cycle (before Decode reads rf) so that an
    instruction completing WB in cycle N is immediately visible to the
    instruction entering Decode in cycle N — no forwarding path needed.

    Returns a list of human-readable output strings (may be empty for
    instructions that don't modify rf or memory, e.g. beq not taken).
    The main loop prints these together with the cycle number and pc.
    """
    if not mem_wb_in.valid:
        return []   # bubble: nothing to commit

    ctrl     = mem_wb_in.ctrl
    messages = []

    # ── Write to register file (if RegWrite and rd != x0) ────────────
    if ctrl.RegWrite and mem_wb_in.rd != 0:
        # MemtoReg mux: choose what gets written to rd
        if ctrl.MemtoReg == 1:
            wb_val = mem_wb_in.mem_data       # lw / lb: data loaded from memory
        elif ctrl.MemtoReg == 2:
            wb_val = mem_wb_in.next_pc        # JAL / JALR: return address (PC+4)
        else:
            wb_val = mem_wb_in.alu_result     # arithmetic / logical / shift result

        # Write to rf and record the change for output
        rf[mem_wb_in.rd] = wb_val & 0xFFFFFFFF
        messages.append(
            f"{reg_name(mem_wb_in.rd)} is modified to {to_hex32(rf[mem_wb_in.rd])}"
        )

    # ── Report memory write (sw / sb) ────────────────────────────────
    if mem_wb_in.store_address is not None:
        messages.append(
            f"memory {to_hex32(mem_wb_in.store_address)} is modified to "
            f"{to_hex32(mem_wb_in.store_value)}"
        )

    # x0 is hardwired to 0 — enforce this in case anything slipped through
    rf[0] = 0
    return messages

# ╚══ END: Writeback() ══╝


# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1 RUBRIC: Main simulation loop  →  run_pipeline()  ║
# ║                                                              ║
# ║  Ties all five stages together into one clock cycle.        ║
# ║  Handles the PC mux (next_pc vs branch_target vs stall).    ║
# ║  Increments total_clock_cycles once per iteration.          ║
# ║  Prints output when Writeback retires a real instruction.   ║
# ╚══════════════════════════════════════════════════════════════╝

def run_pipeline(program: List[int]) -> None:
    """
    Clock-accurate pipeline simulation.

    CYCLE ORDERING (critical for correctness):
      Within one clock cycle we process stages in this order:
        1. WB  – commits to rf FIRST so Decode reads the up-to-date value
        2. RAW hazard check – uses OLD latches (before this cycle's EX fires)
        3. MEM – reads/writes d_mem
        4. EX  – runs the ALU, computes branch/jump targets
        5. Flush check – if EX resolved a taken branch/jump, flush IF+ID
        6. ID  – decode (produces bubble if stall or flush)
        7. IF  – fetch (flush overrides stall)
        8. Commit all new latch values simultaneously
        9. Update pc
       10. Increment total_clock_cycles
       11. Print output if WB retired a real instruction
       12. Termination check

    HAZARD HANDLING (Extra Credit pipeline):
      Stall  – PC and IF/ID held; bubble inserted into ID/EX.
               Triggered by RAW hazard (EX or MEM stage will write a reg
               that the instruction in ID needs to read).
      Flush  – IF/ID and ID/EX replaced with bubbles; PC redirected to
               branch_target or jump_target.
               Triggered by any taken branch (BEQ/BNE) or any jump (JAL/JALR).
      Priority: flush > stall (if both occur in the same cycle, flush wins
               and the previously-stalled instruction is discarded).
    """
    global pc, total_clock_cycles
    global if_id, id_ex, ex_mem, mem_wb

    program_done       = False   # True once IF has run off the end of the program
    next_sequential_pc = pc      # updated each IF call; holds the next pc to use

    while True:

        # ── Step 1: Writeback ─────────────────────────────────────────
        # WB runs before everything else so rf is up-to-date when ID reads it.
        wb_messages  = stage_WB(mem_wb)
        wb_had_valid = mem_wb.valid   # remember if a real instruction was here

        # ── Step 2: RAW hazard check ──────────────────────────────────
        # Uses the CURRENT (old) id_ex and ex_mem latches — before EX fires.
        stall = raw_hazard(if_id, id_ex, ex_mem)

        # ── Step 3: Memory Access ─────────────────────────────────────
        new_mem_wb = stage_MEM(ex_mem)

        # ── Step 4: Execute ───────────────────────────────────────────
        new_ex_mem = stage_EX(id_ex)

        # ── Step 5: Branch / Jump flush detection ─────────────────────
        # If EX resolved a taken branch or any jump, we must flush the two
        # instructions that have already entered IF and ID (they are wrong-path).
        flush  = False
        new_pc = pc
        if new_ex_mem.valid and (new_ex_mem.branch_taken or new_ex_mem.jump_taken):
            flush  = True
            new_pc = (
                new_ex_mem.jump_target   if new_ex_mem.jump_taken
                else new_ex_mem.branch_target
            ) & 0xFFFFFFFF

        # ── Step 6: Decode ────────────────────────────────────────────
        # If stalling OR flushing, insert a bubble instead of decoding.
        new_id_ex = ID_EX(valid=False) if (stall or flush) else stage_ID(if_id)

        # ── Step 7: Instruction Fetch ─────────────────────────────────
        # Priority: flush > stall > normal fetch.
        # Flush wins even if a stall was also pending (flush discards the stalled instr).
        if flush:
            new_if_id = IF_ID(valid=False)   # flush: discard in-flight instruction
        elif stall:
            new_if_id = if_id                # stall: hold the IF/ID latch unchanged
        elif program_done:
            new_if_id = IF_ID(valid=False)   # no more instructions to fetch
        else:
            new_if_id, next_sequential_pc = stage_IF(program, pc)
            if not new_if_id.valid:
                program_done = True          # just ran off the end of the program

        # ── Step 8: Commit all latches simultaneously ─────────────────
        # All four stage registers are updated at once, modelling a rising clock edge.
        mem_wb = new_mem_wb
        ex_mem = new_ex_mem
        id_ex  = new_id_ex
        if_id  = new_if_id

        # ── Step 9: Update PC ─────────────────────────────────────────
        # SECTION 1 RUBRIC: the PC mux choosing among next_pc, branch_target
        if flush:
            pc = new_pc                  # redirect to branch/jump target
        elif stall:
            pass                         # hold PC (do not advance)
        elif not program_done:
            pc = next_sequential_pc      # normal sequential advance

        # ── Step 10: Increment cycle counter ─────────────────────────
        # SECTION 1 RUBRIC: total_clock_cycles incremented once per cycle
        total_clock_cycles += 1

        # ── Step 11: Print output when WB retires a real instruction ──
        # SECTION 1 RUBRIC: print rf/d_mem changes and pc after each instruction
        if wb_had_valid:
            print(f"total_clock_cycles {total_clock_cycles} :")
            for msg in wb_messages:          # register or memory modification lines
                print(msg)
            print(f"pc is modified to {to_hex32(pc)}")

        # ── Step 12: Termination ──────────────────────────────────────
        # Stop when the program is done fetching AND all four stage registers
        # have drained to bubbles (every in-flight instruction has retired).
        all_empty = (
            not if_id.valid  and not id_ex.valid and
            not ex_mem.valid and not mem_wb.valid
        )
        if program_done and all_empty:
            break

    print("program terminated:")
    print(f"total execution time is {total_clock_cycles} cycles")

# ╚══ END: Main simulation loop ══╝


# ══════════════════════════════════════════════════════════════
#  Initialization helpers
#  Each function sets the register file and data memory to the
#  state specified by the project rubric, then resets all latches.
# ══════════════════════════════════════════════════════════════

def _reset_pipeline() -> None:
    """Reset all four pipeline stage registers to empty bubbles."""
    global if_id, id_ex, ex_mem, mem_wb
    if_id = IF_ID(); id_ex = ID_EX(); ex_mem = EX_MEM(); mem_wb = MEM_WB()


def initialize_section1() -> None:
    """
    Section 1 initial state (used with sample_part1.txt).
    Per project spec:
      x1 = 0x20, x2 = 0x5, x10 = 0x70, x11 = 0x4
      d_mem[0x70//4] = 0x5,  d_mem[0x74//4] = 0x10
    """
    global pc, total_clock_cycles, rf, d_mem
    pc = total_clock_cycles = 0
    rf    = [0] * 32
    d_mem = [0] * 64
    rf[1]  = 0x20    # x1
    rf[2]  = 0x5     # x2
    rf[10] = 0x70    # a0
    rf[11] = 0x4     # a1
    d_mem[0x70 // 4] = 0x5
    d_mem[0x74 // 4] = 0x10
    _reset_pipeline()


def initialize_section2() -> None:
    """
    Section 2 initial state (used with sample_part2.txt).
    Per project spec:
      s0=0x20, a0=0x5, a1=0x2, a2=0xa, a3=0xf
      d_mem all zeros
    """
    global pc, total_clock_cycles, rf, d_mem
    pc = total_clock_cycles = 0
    rf    = [0] * 32
    d_mem = [0] * 64
    rf[8]  = 0x20    # s0
    rf[10] = 0x5     # a0
    rf[11] = 0x2     # a1
    rf[12] = 0xa     # a2
    rf[13] = 0xf     # a3
    _reset_pipeline()


def initialize_extra() -> None:
    """
    Extra Duty initial state (used with sample_extra.txt).
    Registers chosen to produce clear, hand-verifiable results for
    lb, sb, bne, sll, and srl.

      ra (x1)  = 0x20  base address for lb / sb tests
      sp (x2)  = 0x4   shift amount for sll / srl
      gp (x3)  = 0xAB  byte payload for sb; also source for sll/srl
      tp (x4)  = 0x3   bne operand (differs from t0 → branch taken)
      t0 (x5)  = 0x7   bne operand (differs from tp → branch taken)
      t1 (x6)  = 0x3   unused in current sample; available for beq testing

    d_mem[0x20 // 4] = 0xDEADBEEF  so lb can load a known byte value.
    """
    global pc, total_clock_cycles, rf, d_mem
    pc = total_clock_cycles = 0
    rf    = [0] * 32
    d_mem = [0] * 64
    rf[1]  = 0x20
    rf[2]  = 0x4
    rf[3]  = 0xAB
    rf[4]  = 0x3
    rf[5]  = 0x7
    rf[6]  = 0x3
    d_mem[0x20 // 4] = 0xDEADBEEF
    _reset_pipeline()


# ══════════════════════════════════════════════════════════════
#  Program loader
# ══════════════════════════════════════════════════════════════

def load_program(filename: str) -> List[int]:
    """
    Read a program file and return a list of 32-bit instruction words.
    Lines starting with '#' or '//' are treated as comments and skipped.
    Accepts both binary-string and hex formats (see parse_instruction_line).
    """
    program = []
    with open(filename, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            program.append(parse_instruction_line(line))
    return program


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

def main() -> None:
    """
    Ask for the program filename, select the matching initialization state,
    load the program, and run the pipeline simulation.
    """
    filename = input("Enter the program file name to run:\n").strip()

    # Select initialization based on filename convention
    if "part2" in filename.lower() or "section2" in filename.lower():
        initialize_section2()
    elif "extra" in filename.lower() or "part3" in filename.lower():
        initialize_extra()
    else:
        initialize_section1()   # default: Section 1 state

    program = load_program(filename)
    run_pipeline(program)


if __name__ == "__main__":
    main()
