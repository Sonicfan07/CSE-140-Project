from dataclasses import dataclass
from typing import List, Optional, Tuple

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

pc = 0
next_pc = 0
branch_target = 0
jump_target = 0          # NEW: computed by Execute() for JAL/JALR
alu_zero = 0
total_clock_cycles = 0

rf    = [0] * 32         # register file  (32 × 32-bit words)
d_mem = [0] * 64         # data memory    (64 × 4-byte words)

RegWrite = 0
Branch   = 0
MemRead  = 0
MemWrite = 0
MemtoReg = 0             # 0 = ALU result, 1 = memory data, 2 = PC+4
ALUSrc   = 0
ALUOp    = 0

#Section 1
Jump    = 0              # 1 for JAL  (PC-relative unconditional jump)
JumpReg = 0              # 1 for JALR (register-relative unconditional jump)

def extract_bits(bin_str: str, high: int, low: int) -> str:
    start = 31 - high
    end   = 31 - low + 1
    return bin_str[start:end]


def sign_extend(value: int, bits: int) -> int:
    if value & (1 << (bits - 1)):
        return value - (1 << bits)
    return value


def signed32(value: int) -> int:
    return value if value < (1 << 31) else value - (1 << 32)


def to_hex32(value: int) -> str:
    return f"0x{value & 0xFFFFFFFF:x}"


def reg_name(idx: int) -> str:
    """Return ABI name (e.g. 'ra') for display, matching expected sample output."""
    return ABI_NAMES.get(idx, f"x{idx}")


def parse_instruction_line(line: str) -> int:
    text = line.strip()
    if not text:
        raise ValueError("empty instruction line")
    if set(text) <= {"0", "1"} and len(text) <= 32:
        return int(text, 2)
    if text.lower().startswith("0x"):
        return int(text, 16)
    return int(text, 16)


def instruction_to_bin(instr: int) -> str:
    return format(instr & 0xFFFFFFFF, "032b")


def reset_control_signals() -> None:
    global RegWrite, Branch, MemRead, MemWrite, MemtoReg, ALUSrc, ALUOp
    global Jump, JumpReg
    RegWrite = 0
    Branch   = 0
    MemRead  = 0
    MemWrite = 0
    MemtoReg = 0
    ALUSrc   = 0
    ALUOp    = 0
    Jump     = 0
    JumpReg  = 0

@dataclass
class DecodedInstruction:
    instr_word: int
    opcode:     int
    mnemonic:   str
    rd:         int = 0
    rs1:        int = 0
    rs2:        int = 0
    funct3:     int = 0
    funct7:     int = 0
    imm:        int = 0
    rs1_val:    int = 0
    rs2_val:    int = 0


@dataclass
class ExecuteResult:
    alu_result:   int
    store_data:   int
    rd:           int
    branch_taken: bool
    jump_taken:   bool           # NEW
    mem_address:  int


@dataclass
class MemResult:
    alu_result:   int
    mem_data:     int
    rd:           int
    wb_pc4:       int            # NEW: next_pc captured for JAL/JALR writeback
    store_address: Optional[int] = None
    store_value:   Optional[int] = None

def ALUControl(alu_op: int, funct3: int, funct7: int, opcode: int) -> int:
    if alu_op == 0b00:
        return 0b0010   # ADD (lw / sw / addi family)
    if alu_op == 0b01:
        return 0b0110   # SUB (beq)
    if alu_op == 0b10:
        if opcode == 0x33:          # R-type
            if funct3 == 0x0 and funct7 == 0x00: return 0b0010   # add
            if funct3 == 0x0 and funct7 == 0x20: return 0b0110   # sub
            if funct3 == 0x7:                    return 0b0000   # and
            if funct3 == 0x6:                    return 0b0001   # or
            if funct3 == 0x2 and funct7 == 0x00: return 0b0101   # slt
        elif opcode == 0x13:        # I-type arith
            if funct3 == 0x0: return 0b0010   # addi
            if funct3 == 0x7: return 0b0000   # andi
            if funct3 == 0x6: return 0b0001   # ori
    raise ValueError(
        f"Unsupported ALU control: ALUOp={alu_op:b}, opcode=0x{opcode:x}, "
        f"funct3={funct3}, funct7={funct7}"
    )

def ControlUnit(opcode: int, funct3: int = 0, funct7: int = 0) -> int:
    global RegWrite, Branch, MemRead, MemWrite, MemtoReg, ALUSrc, ALUOp
    global Jump, JumpReg

    reset_control_signals()

    if opcode == 0x03:       # lw
        RegWrite = 1; MemRead = 1; MemtoReg = 1; ALUSrc = 1; ALUOp = 0b00
    elif opcode == 0x23:     # sw
        MemWrite = 1; ALUSrc = 1; ALUOp = 0b00
    elif opcode == 0x63:     # beq
        Branch = 1; ALUOp = 0b01
    elif opcode == 0x33:     # R-type
        RegWrite = 1; ALUSrc = 0; ALUOp = 0b10
    elif opcode == 0x13:     # I-type arith (addi/andi/ori)
        RegWrite = 1; ALUSrc = 1; ALUOp = 0b10

    #Section 2
    elif opcode == 0x6F:     # JAL  (J-type)
        Jump     = 1
        RegWrite = 1
        MemtoReg = 2         # write PC+4 to rd
        ALUOp    = 0b00      # ADD returned (target computed separately in Execute)

    elif opcode == 0x67:     # JALR (I-type, funct3=0x0)
        JumpReg  = 1
        RegWrite = 1
        ALUSrc   = 1         # use immediate as op_b
        MemtoReg = 2         # write PC+4 to rd
        ALUOp    = 0b00      # ADD: rs1 + imm → jump_target

    else:
        raise ValueError(f"Unsupported opcode: 0x{opcode:x}")

    return ALUControl(ALUOp, funct3, funct7, opcode)

def Fetch(program: List[int]) -> Optional[int]:
    global pc, next_pc
    index = pc // 4
    if index < 0 or index >= len(program):
        return None
    instr_word = program[index]
    next_pc    = pc + 4
    return instr_word


def Decode(instr_word: int) -> Tuple[DecodedInstruction, int]:
    global rf

    bin_str = instruction_to_bin(instr_word)
    opcode  = int(extract_bits(bin_str, 6, 0), 2)

    rd = rs1 = rs2 = funct3 = funct7 = imm = 0
    mnemonic = "unknown"

    if opcode == 0x33:
        rd     = int(extract_bits(bin_str, 11,  7), 2)
        funct3 = int(extract_bits(bin_str, 14, 12), 2)
        rs1    = int(extract_bits(bin_str, 19, 15), 2)
        rs2    = int(extract_bits(bin_str, 24, 20), 2)
        funct7 = int(extract_bits(bin_str, 31, 25), 2)
        if   funct3 == 0x0 and funct7 == 0x00: mnemonic = "add"
        elif funct3 == 0x0 and funct7 == 0x20: mnemonic = "sub"
        elif funct3 == 0x7 and funct7 == 0x00: mnemonic = "and"
        elif funct3 == 0x6 and funct7 == 0x00: mnemonic = "or"
        elif funct3 == 0x2 and funct7 == 0x00: mnemonic = "slt"
        else: raise ValueError("Unsupported R-type instruction")

    elif opcode == 0x13:
        rd     = int(extract_bits(bin_str, 11,  7), 2)
        funct3 = int(extract_bits(bin_str, 14, 12), 2)
        rs1    = int(extract_bits(bin_str, 19, 15), 2)
        imm    = sign_extend(int(extract_bits(bin_str, 31, 20), 2), 12)
        if   funct3 == 0x0: mnemonic = "addi"
        elif funct3 == 0x7: mnemonic = "andi"
        elif funct3 == 0x6: mnemonic = "ori"
        else: raise ValueError("Unsupported I-type arithmetic instruction")

    elif opcode == 0x03:
        rd     = int(extract_bits(bin_str, 11,  7), 2)
        funct3 = int(extract_bits(bin_str, 14, 12), 2)
        rs1    = int(extract_bits(bin_str, 19, 15), 2)
        imm    = sign_extend(int(extract_bits(bin_str, 31, 20), 2), 12)
        if funct3 != 0x2: raise ValueError("Only lw (funct3=2) is supported")
        mnemonic = "lw"

    elif opcode == 0x23:
        funct3   = int(extract_bits(bin_str, 14, 12), 2)
        rs1      = int(extract_bits(bin_str, 19, 15), 2)
        rs2      = int(extract_bits(bin_str, 24, 20), 2)
        imm_high = int(extract_bits(bin_str, 31, 25), 2)
        imm_low  = int(extract_bits(bin_str, 11,  7), 2)
        imm      = sign_extend((imm_high << 5) | imm_low, 12)
        if funct3 != 0x2: raise ValueError("Only sw (funct3=2) is supported")
        mnemonic = "sw"

    elif opcode == 0x63:
        funct3   = int(extract_bits(bin_str, 14, 12), 2)
        rs1      = int(extract_bits(bin_str, 19, 15), 2)
        rs2      = int(extract_bits(bin_str, 24, 20), 2)
        imm12    = int(extract_bits(bin_str, 31, 31), 2)
        imm10_5  = int(extract_bits(bin_str, 30, 25), 2)
        imm4_1   = int(extract_bits(bin_str, 11,  8), 2)
        imm11    = int(extract_bits(bin_str,  7,  7), 2)
        imm = sign_extend(
            (imm12 << 11) | (imm11 << 10) | (imm10_5 << 4) | imm4_1, 12
        )
        if funct3 != 0x0: raise ValueError("Only beq (funct3=0) is supported")
        mnemonic = "beq"

    #JAL (J-type)
    elif opcode == 0x6F:
        # J-type immediate: imm[20|10:1|11|19:12] scattered across the word
        rd      = int(extract_bits(bin_str, 11,  7), 2)
        imm20   = int(extract_bits(bin_str, 31, 31), 2)
        imm10_1 = int(extract_bits(bin_str, 30, 21), 2)
        imm11   = int(extract_bits(bin_str, 20, 20), 2)
        imm19_12= int(extract_bits(bin_str, 19, 12), 2)
        # Reassemble: [20|19:12|11|10:1] then append implicit 0 at LSB
        raw_imm = (imm20 << 19) | (imm19_12 << 11) | (imm11 << 10) | imm10_1
        imm = sign_extend(raw_imm, 20) << 1   # already byte-addressed offset
        mnemonic = "jal"

    #JALR (I-type, opcode 0x67)
    elif opcode == 0x67:
        rd     = int(extract_bits(bin_str, 11,  7), 2)
        funct3 = int(extract_bits(bin_str, 14, 12), 2)
        rs1    = int(extract_bits(bin_str, 19, 15), 2)
        imm    = sign_extend(int(extract_bits(bin_str, 31, 20), 2), 12)
        if funct3 != 0x0: raise ValueError("Only jalr (funct3=0) is supported")
        mnemonic = "jalr"

    else:
        raise ValueError(f"Unsupported opcode: 0x{opcode:x}")

    alu_ctrl = ControlUnit(opcode, funct3, funct7)

    decoded = DecodedInstruction(
        instr_word=instr_word,
        opcode=opcode,
        mnemonic=mnemonic,
        rd=rd, rs1=rs1, rs2=rs2,
        funct3=funct3, funct7=funct7,
        imm=imm,
        rs1_val=rf[rs1],
        rs2_val=rf[rs2],
    )
    return decoded, alu_ctrl

def Execute(decoded: DecodedInstruction, alu_ctrl: int) -> ExecuteResult:
    global alu_zero, branch_target, jump_target, next_pc

    op_a = decoded.rs1_val
    op_b = decoded.imm if ALUSrc else decoded.rs2_val

    # Perform ALU operation
    if   alu_ctrl == 0b0000: alu_result = op_a & op_b          # AND
    elif alu_ctrl == 0b0001: alu_result = op_a | op_b          # OR
    elif alu_ctrl == 0b0010: alu_result = op_a + op_b          # ADD
    elif alu_ctrl == 0b0110: alu_result = op_a - op_b          # SUB
    elif alu_ctrl == 0b0101:                                    # SLT
        alu_result = 1 if signed32(op_a) < signed32(op_b) else 0
    else:
        raise ValueError(f"Unsupported alu_ctrl: {alu_ctrl:04b}")

    alu_zero = 1 if alu_result == 0 else 0

    # Branch target: next_pc + (sign-extended offset << 1)
    # (imm stored in decoded is already the raw 12-bit field; Execute shifts it)
    branch_target = next_pc + (decoded.imm << 1)

    #Jump target calculation
    # JAL:  PC + imm  (imm already holds the full byte offset after Decode)
    # JALR: (rs1 + imm) with LSB cleared (RISC-V spec §2.5)
    if Jump:
        # current pc (before increment) is next_pc - 4
        jump_target = (next_pc - 4) + decoded.imm
    elif JumpReg:
        jump_target = (decoded.rs1_val + decoded.imm) & ~1   # clear LSB

    branch_taken = bool(Branch and alu_zero)
    jump_taken   = bool(Jump or JumpReg)

    return ExecuteResult(
        alu_result=alu_result,
        store_data=decoded.rs2_val,
        rd=decoded.rd,
        branch_taken=branch_taken,
        jump_taken=jump_taken,
        mem_address=alu_result,
    )

def Mem(ex_result: ExecuteResult) -> MemResult:
    global d_mem

    mem_data      = 0
    store_address = None
    store_value   = None

    if MemRead:
        if ex_result.mem_address % 4 != 0:
            raise ValueError(f"Unaligned lw address: {to_hex32(ex_result.mem_address)}")
        idx = ex_result.mem_address // 4
        if not (0 <= idx < len(d_mem)):
            raise IndexError(f"lw address out of range: {to_hex32(ex_result.mem_address)}")
        mem_data = d_mem[idx]

    if MemWrite:
        if ex_result.mem_address % 4 != 0:
            raise ValueError(f"Unaligned sw address: {to_hex32(ex_result.mem_address)}")
        idx = ex_result.mem_address // 4
        if not (0 <= idx < len(d_mem)):
            raise IndexError(f"sw address out of range: {to_hex32(ex_result.mem_address)}")
        d_mem[idx] = ex_result.store_data & 0xFFFFFFFF
        store_address = ex_result.mem_address
        store_value   = d_mem[idx]

    return MemResult(
        alu_result=ex_result.alu_result,
        mem_data=mem_data,
        rd=ex_result.rd,
        wb_pc4=next_pc,          # carry next_pc for JAL/JALR writeback
        store_address=store_address,
        store_value=store_value,
    )

def Writeback(mem_result: MemResult, ex_result: ExecuteResult) -> List[str]:
    global rf, pc, next_pc, branch_target, jump_target, total_clock_cycles

    messages = []

    if RegWrite and mem_result.rd != 0:
        if MemtoReg == 1:
            wb_value = mem_result.mem_data        # lw: data from memory
        elif MemtoReg == 2:
            wb_value = mem_result.wb_pc4          # JAL/JALR: return address (PC+4)
        else:
            wb_value = mem_result.alu_result      # arithmetic / logical result

        rf[mem_result.rd] = wb_value & 0xFFFFFFFF
        messages.append(
            f"{reg_name(mem_result.rd)} is modified to {to_hex32(rf[mem_result.rd])}"
        )

    if mem_result.store_address is not None:
        messages.append(
            f"memory {to_hex32(mem_result.store_address)} is modified to "
            f"{to_hex32(mem_result.store_value)}"
        )

    # PC selection priority: jump > branch > sequential
    if ex_result.jump_taken:
        pc = jump_target & 0xFFFFFFFF
    elif ex_result.branch_taken:
        pc = branch_target & 0xFFFFFFFF
    else:
        pc = next_pc

    total_clock_cycles += 1
    messages.append(f"pc is modified to {to_hex32(pc)}")

    rf[0] = 0   # x0 is hardwired to 0
    return messages

def initialize_arch_state_section1() -> None:
    """Section 1 initial register / memory state."""
    global pc, next_pc, branch_target, jump_target, alu_zero, total_clock_cycles
    global rf, d_mem

    pc = next_pc = branch_target = jump_target = alu_zero = total_clock_cycles = 0
    rf    = [0] * 32
    d_mem = [0] * 64

    rf[1]  = 0x20   # x1
    rf[2]  = 0x5    # x2
    rf[10] = 0x70   # x10 (a0)
    rf[11] = 0x4    # x11 (a1)

    d_mem[0x70 // 4] = 0x5
    d_mem[0x74 // 4] = 0x10


def initialize_arch_state_section2() -> None:
    """Section 2 initial register / memory state (from project PDF)."""
    global pc, next_pc, branch_target, jump_target, alu_zero, total_clock_cycles
    global rf, d_mem

    pc = next_pc = branch_target = jump_target = alu_zero = total_clock_cycles = 0
    rf    = [0] * 32
    d_mem = [0] * 64   # all zeros for Section 2

    # s0=0x20, a0=0x5, a1=0x2, a2=0xa, a3=0xf
    rf[8]  = 0x20   # s0
    rf[10] = 0x5    # a0
    rf[11] = 0x2    # a1
    rf[12] = 0xa    # a2
    rf[13] = 0xf    # a3

def load_program(filename: str) -> List[int]:
    program = []
    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            program.append(parse_instruction_line(line))
    return program

def run_program(program: List[int]) -> None:
    while True:
        instr_word = Fetch(program)
        if instr_word is None:
            break

        decoded, alu_ctrl = Decode(instr_word)
        ex_result          = Execute(decoded, alu_ctrl)
        mem_result         = Mem(ex_result)
        messages           = Writeback(mem_result, ex_result)

        print(f"total_clock_cycles {total_clock_cycles} :")
        for msg in messages:
            print(msg)

    print("program terminated:")
    print(f"total execution time is {total_clock_cycles} cycles")

def main() -> None:
    filename = input("Enter the program file name to run:\n").strip()

    # Auto-select init state based on filename convention
    if "part2" in filename.lower() or "section2" in filename.lower():
        initialize_arch_state_section2()
    else:
        initialize_arch_state_section1()

    program = load_program(filename)
    run_program(program)


if __name__ == "__main__":
    main()
