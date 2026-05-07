from dataclasses import dataclass, field
from typing import List, Optional

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

#Section 1: global pc, total_clock_cyckes, rf, d_mem
#points to the instruction currently being fetched.
# Initialized to 0; updated every cycle (advance, hold on stall, or redirect on branch/jump).
pc = 0

# incremented once at the end of every pipeline clock step.
total_clock_cycles = 0

#32-bit registers.
# x0 (index 0) is hardwired to zero and re-zeroed after every Writeback.
rf = [0] * 32

#64 word-sized slots (each slot = 4 bytes).
# Addressed by byte address; word index = byte_addr // 4.
d_mem = [0] * 64


@dataclass
class IF_ID:
    valid:      bool = False   # False = bubble (no real instruction here)
    pc:         int  = 0       # PC of this instruction (used by JAL for jump target)
    next_pc:    int  = 0       # PC + 4 (passed forward as potential return address)
    instr_word: int  = 0       # 32-bit instruction fetched from program memory

#Section 1: RegQrite, Branch, MemRead, MemWrite,  MemtoRef, ALUSrc
#Section 2: Jump, JumpReg (jal/jalr)
@dataclass
class CtrlSignals:
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

#Section 1: branch_target
@dataclass
class EX_MEM:
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

#pull bits from 32 char binary
def extract_bits(bin_str: str, high: int, low: int) -> str:
    return bin_str[31 - high : 32 - low]

def sign_extend(value: int, bits: int) -> int:
    if value & (1 << (bits - 1)):
        return value - (1 << bits)
    return value

def signed32(value: int) -> int:
    return value if value < (1 << 31) else value - (1 << 32)

def to_hex32(value: int) -> str:
    return f"0x{value & 0xFFFFFFFF:x}"

def reg_name(idx: int) -> str:
    return ABI_NAMES.get(idx, f"x{idx}")

def parse_instruction_line(line: str) -> int:
    text = line.strip()
    if not text:
        raise ValueError("empty line")
    if set(text) <= {"0", "1"} and len(text) <= 32:
        return int(text, 2)          # binary string
    return int(text.lower().replace("0x", ""), 16)   # hex string

def instruction_to_bin(instr: int) -> str:
    return format(instr & 0xFFFFFFFF, "032b")

#Section 1: ALUControl, translating ALUOp, funct3, funct7 to 2 bit
def ALUControl(alu_op: int, funct3: int, funct7: int, opcode: int) -> int:
    #ALUOp = 00: memory/jump address calculation always uses ADD
    if alu_op == 0b00:
        return 0b0010   # ADD

    #ALUOp = 01: branch comparison always uses SUB (check zero flag)
    if alu_op == 0b01:
        return 0b0110   # SUB

    #ALUOp = 10: use funct3 / funct7 to select operation
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

#Section 1: ControlUnit, set control signals based on opcode
#Section 2: add jal/jalr: extra duty reuses existing opcodes lb, sb, bne
def ControlUnit(opcode: int, funct3: int = 0, funct7: int = 0):
    ctrl   = CtrlSignals()   # start with all signals = 0
    alu_op = 0

    #Section 1: Load
    # lw (funct3=2) and lb (funct3=0) both use the same control signals.
    # The MEM stage uses mem_funct3 to decide byte vs word access width.
    if opcode == 0x03:
        ctrl.RegWrite = 1   # result (loaded data) goes back to register file
        ctrl.MemRead  = 1   # read from data memory
        ctrl.MemtoReg = 1   # Writeback mux: select memory data (not ALU result)
        ctrl.ALUSrc   = 1   # ALU operand B = immediate (address offset)
        alu_op = 0b00       # ALUOp=00 → ADD (compute effective address = rs1 + imm)

    #Section 1: Store
    # sw (funct3=2) and sb (funct3=0) both use the same control signals.
    elif opcode == 0x23:
        ctrl.MemWrite = 1   # write to data memory
        ctrl.ALUSrc   = 1   # ALU operand B = immediate (address offset)
        alu_op = 0b00       # ALUOp=00 → ADD (compute effective address = rs1 + imm)

    #Section 1: Branch
    # beq (funct3=0) and bne (funct3=1) differ only in which zero-flag sense is used.
    elif opcode == 0x63:
        if   funct3 == 0x0: ctrl.branch_type = 1   # BEQ: branch when alu_zero==1
        elif funct3 == 0x1: ctrl.branch_type = 2   # BNE: branch when alu_zero==0
        else: raise ValueError(f"Unsupported branch funct3={funct3}")
        alu_op = 0b01       # ALUOp=01 → SUB (rs1 - rs2; zero flag tells us equality)

    #Section 1: R-type (add, sub, and, or, slt, sll, srl)
    elif opcode == 0x33:
        ctrl.RegWrite = 1   # ALU result goes to register file
        ctrl.ALUSrc   = 0   # ALU operand B = rs2_val (not an immediate)
        alu_op = 0b10       # ALUOp=10 → use funct3/funct7 to pick operation

    #Section 1: I-type arithmetic (addi, andi, ori)
    elif opcode == 0x13:
        ctrl.RegWrite = 1   # ALU result goes to register file
        ctrl.ALUSrc   = 1   # ALU operand B = sign-extended immediate
        alu_op = 0b10       # ALUOp=10 → use funct3 to pick operation

    #Section 2: JAL
    elif opcode == 0x6F:
        ctrl.Jump     = 1   # tells EX to compute PC+imm as jump target
        ctrl.RegWrite = 1   # rd = PC+4 (return address)
        ctrl.MemtoReg = 2   # Writeback mux: select PC+4
        alu_op = 0b00       # ADD (not actually used for JAL; target computed separately)

    #Section 2: JALR
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

#Section 1: Decode, extract all fields from 32-bit instruction
#Section 1: R, I, S, B
#Section 2: J
def decode_fields(instr_word: int) -> dict:
    b   = instruction_to_bin(instr_word)     # 32-char binary string, bit31 at index 0
    opc = int(extract_bits(b, 6, 0), 2)      # bits[6:0] = opcode

    rd = rs1 = rs2 = funct3 = funct7 = imm = 0
    mnemonic = "nop"

    #R-type: add, sub, and, or, slt, sll (Extra), srl (Extra)
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

    #I-type arithmetic: addi, andi, ori
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

    #Load: lw (funct3=2) and lb (funct3=0, Extra Duty)
    elif opc == 0x03:
        # Same I-type layout as addi etc; funct3 tells us byte vs word width.
        rd     = int(extract_bits(b, 11,  7), 2)
        funct3 = int(extract_bits(b, 14, 12), 2)
        rs1    = int(extract_bits(b, 19, 15), 2)
        imm    = sign_extend(int(extract_bits(b, 31, 20), 2), 12)
        if   funct3 == 0x2: mnemonic = "lw"
        elif funct3 == 0x0: mnemonic = "lb"   # Extra Duty
        else: raise ValueError(f"Unsupported load funct3={funct3}")

    #Store: sw (funct3=2) and sb (funct3=0, Extra Duty)
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

    #Branch: beq (funct3=0) and bne (funct3=1, Extra Duty)
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

    #JAL: J-type (Section 2)
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

    #JALR: I-type (Section 2)
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

#check instruction in IF/ID reads register
def _writes_rd(valid: bool, ctrl: CtrlSignals, rd: int) -> bool:
    return valid and ctrl.RegWrite and rd != 0


def raw_hazard(if_id_l: IF_ID, id_ex_l: ID_EX, ex_mem_l: EX_MEM) -> bool:
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

#Section 1: Fetch, read instruction at pc, compute next_pc = pc + 4
def stage_IF(program: List[int], pc_in: int) -> tuple:
    index = pc_in // 4   # byte address → word index (each instruction is 4 bytes)

    if index < 0 or index >= len(program):
        # PC is past the last instruction: signal end-of-program with a bubble
        return IF_ID(valid=False), pc_in

    instr_word = program[index]    # fetch the 32-bit instruction word
    next_pc    = pc_in + 4         # compute PC+4 for sequential execution

    # Pack into the IF_ID latch and return it alongside the next sequential PC
    latch = IF_ID(valid=True, pc=pc_in, next_pc=next_pc, instr_word=instr_word)
    return latch, next_pc

#Section 1: Decode, decode instruction, call ControlUnit, read rs1/rs2
def stage_ID(if_id_in: IF_ID) -> ID_EX:
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

#Section 1: Execute, runs ALU, sets alu_zero, compute branch_target
#Section 2: compute jal/jalr jump targets
#Extra: sll, srl
def stage_EX(id_ex_in: ID_EX) -> EX_MEM:
    if not id_ex_in.valid:
        return EX_MEM(valid=False)

    ctrl = id_ex_in.ctrl

    #Select ALU operand B
    op_a = id_ex_in.rs1_val
    op_b = id_ex_in.imm if ctrl.ALUSrc else id_ex_in.rs2_val

    #Execute ALU operation
    ac = id_ex_in.alu_ctrl
    if   ac == 0b0000: alu_result = op_a & op_b                             # AND
    elif ac == 0b0001: alu_result = op_a | op_b                             # OR
    elif ac == 0b0010: alu_result = op_a + op_b                             # ADD
    elif ac == 0b0011: alu_result = (op_a << (op_b & 0x1F)) & 0xFFFFFFFF   # SLL [Extra]
    elif ac == 0b0100: alu_result = (op_a & 0xFFFFFFFF) >> (op_b & 0x1F)   # SRL [Extra]
    elif ac == 0b0101: alu_result = 1 if signed32(op_a) < signed32(op_b) else 0  # SLT
    elif ac == 0b0110: alu_result = op_a - op_b                             # SUB
    else:              alu_result = 0

    #SECTION 1 RUBRIC: alu_zero 
    # 1-bit flag: 1 when the ALU result is exactly zero.
    # Used by BEQ (branch if equal) and BNE (branch if not equal).
    alu_zero = (alu_result == 0)

    #SECTION 1 RUBRIC: branch_target
    # Branch target = next_pc (PC+4) + byte offset.
    # The B-type imm field stores offset/2; <<1 recovers the byte offset.
    branch_target = id_ex_in.next_pc + (id_ex_in.imm << 1)

    # Determine whether the branch is taken
    if   ctrl.branch_type == 1: branch_taken = bool(alu_zero)       # BEQ
    elif ctrl.branch_type == 2: branch_taken = bool(not alu_zero)   # BNE [Extra Duty]
    else:                       branch_taken = False                 # not a branch

    #Section 2: Jump target computation
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

#Section 1: Memory access, lw reads word, sw writes word
#Ectra: lb reads 1 byte, sign-extends it, sb writes 1 bytes
def stage_MEM(ex_mem_in: EX_MEM) -> MEM_WB:
    if not ex_mem_in.valid:
        return MEM_WB(valid=False)

    ctrl          = ex_mem_in.ctrl
    mem_data      = 0             # output to Writeback (for loads)
    store_address = None          # set only if a store actually executes
    store_value   = None          # full word value written (for output message)

    addr      = ex_mem_in.alu_result & 0xFFFFFFFF   # effective byte address
    word_idx  = addr // 4                            # index into d_mem array
    byte_lane = addr  % 4                            # byte offset within the word

    #SECTION 1 RUBRIC: Load (MemRead) 
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

    #SECTION 1 RUBRIC: Store (MemWrite)
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

#Section 1: Writeback, result to register file, prints output
#Section 2: MemtoReg selects PC+4 as write value return address for jal, jalr
def stage_WB(mem_wb_in: MEM_WB) -> List[str]:
    if not mem_wb_in.valid:
        return []   # bubble: nothing to commit

    ctrl     = mem_wb_in.ctrl
    messages = []

    #Write to register file (if RegWrite and rd != x0)
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

    #Report memory write (sw / sb)
    if mem_wb_in.store_address is not None:
        messages.append(
            f"memory {to_hex32(mem_wb_in.store_address)} is modified to "
            f"{to_hex32(mem_wb_in.store_value)}"
        )

    # x0 is hardwired to 0 — enforce this in case anything slipped through
    rf[0] = 0
    return messages

def run_pipeline(program: List[int]) -> None:
    global pc, total_clock_cycles
    global if_id, id_ex, ex_mem, mem_wb

    program_done       = False   # True once IF has run off the end of the program
    next_sequential_pc = pc      # updated each IF call; holds the next pc to use

    while True:

        #Step 1: Writeback 
        # WB runs before everything else so rf is up-to-date when ID reads it.
        wb_messages  = stage_WB(mem_wb)
        wb_had_valid = mem_wb.valid   # remember if a real instruction was here

        #Step 2: RAW hazard check 
        # Uses the CURRENT (old) id_ex and ex_mem latches — before EX fires.
        stall = raw_hazard(if_id, id_ex, ex_mem)

        #Step 3: Memory Access 
        new_mem_wb = stage_MEM(ex_mem)

        #Step 4: Execute 
        new_ex_mem = stage_EX(id_ex)

        #Step 5: Branch / Jump flush detection 
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

        #Step 6: Decode 
        # If stalling OR flushing, insert a bubble instead of decoding.
        new_id_ex = ID_EX(valid=False) if (stall or flush) else stage_ID(if_id)

        #Step 7: Instruction Fetch 
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

        #Step 8: Commit all latches simultaneously 
        # All four stage registers are updated at once, modelling a rising clock edge.
        mem_wb = new_mem_wb
        ex_mem = new_ex_mem
        id_ex  = new_id_ex
        if_id  = new_if_id

        #Step 9: Update PC 
        # SECTION 1 RUBRIC: the PC mux choosing among next_pc, branch_target
        if flush:
            pc = new_pc                  # redirect to branch/jump target
        elif stall:
            pass                         # hold PC (do not advance)
        elif not program_done:
            pc = next_sequential_pc      # normal sequential advance

        #Step 10: Increment cycle counter 
        # SECTION 1 RUBRIC: total_clock_cycles incremented once per cycle
        total_clock_cycles += 1

        #Step 11: Print output when WB retires a real instruction 
        # SECTION 1 RUBRIC: print rf/d_mem changes and pc after each instruction
        if wb_had_valid:
            print(f"total_clock_cycles {total_clock_cycles} :")
            for msg in wb_messages:          # register or memory modification lines
                print(msg)
            print(f"pc is modified to {to_hex32(pc)}")

        #Step 12: Termination 
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

def _reset_pipeline() -> None:
    global if_id, id_ex, ex_mem, mem_wb
    if_id = IF_ID(); id_ex = ID_EX(); ex_mem = EX_MEM(); mem_wb = MEM_WB()


def initialize_section1() -> None:
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


def load_program(filename: str) -> List[int]:
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
