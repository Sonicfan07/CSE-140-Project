from dataclasses import dataclass
from typing import List, Optional, Tuple

pc = 0
next_pc = 0
branch_target = 0
alu_zero = 0
total_clock_cycles = 0

rf = [0] * 32           # register file

d_mem = [0] * 64        # each entry = 1 word = 4 bytes

# Control signals (globals, as requested by project spec)
RegWrite = 0
Branch = 0
MemRead = 0
MemWrite = 0
MemtoReg = 0
ALUSrc = 0
ALUOp = 0

def pad_to_32_bits(bin_str: str) -> str:
    return bin_str.zfill(32)


def extract_bits(bin_str: str, high: int, low: int) -> str:
    start = 31 - high
    end = 31 - low + 1
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
    return f"x{idx}"


def parse_instruction_line(line: str) -> int:
    text = line.strip()
    if not text:
        raise ValueError("empty instruction line")

    if set(text) <= {"0", "1"} and len(text) <= 32:
        return int(text, 2)

    if text.lower().startswith("0x"):
        return int(text, 16)

    # bare hex like 005101b3
    return int(text, 16)


def instruction_to_bin(instr: int) -> str:
    return format(instr & 0xFFFFFFFF, "032b")


def reset_control_signals() -> None:
    global RegWrite, Branch, MemRead, MemWrite, MemtoReg, ALUSrc, ALUOp
    RegWrite = 0
    Branch = 0
    MemRead = 0
    MemWrite = 0
    MemtoReg = 0
    ALUSrc = 0
    ALUOp = 0

@dataclass
class DecodedInstruction:
    instr_word: int
    opcode: int
    mnemonic: str
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    funct3: int = 0
    funct7: int = 0
    imm: int = 0
    rs1_val: int = 0
    rs2_val: int = 0


@dataclass
class ExecuteResult:
    alu_result: int
    store_data: int
    rd: int
    branch_taken: bool
    mem_address: int


@dataclass
class MemResult:
    alu_result: int
    mem_data: int
    rd: int
    store_address: Optional[int] = None
    store_value: Optional[int] = None

def ALUControl(alu_op: int, funct3: int, funct7: int, opcode: int) -> int:
    # Load/store/addi family => ADD
    if alu_op == 0b00:
        return 0b0010

    # beq => SUB compare
    if alu_op == 0b01:
        return 0b0110

    # R-type / logic-immediate family based on funct fields
    if alu_op == 0b10:
        if opcode == 0x33:  # R-type
            if funct3 == 0x0 and funct7 == 0x00:
                return 0b0010  # add
            if funct3 == 0x0 and funct7 == 0x20:
                return 0b0110  # sub
            if funct3 == 0x7:
                return 0b0000  # and
            if funct3 == 0x6:
                return 0b0001  # or
            if funct3 == 0x2 and funct7 == 0x00:
                return 0b0101  # slt
        elif opcode == 0x13:  # I-type arithmetic in this project: addi/andi/ori
            if funct3 == 0x0:
                return 0b0010  # addi
            if funct3 == 0x7:
                return 0b0000  # andi
            if funct3 == 0x6:
                return 0b0001  # ori

    raise ValueError(
        f"Unsupported ALU control combination: ALUOp={alu_op:b}, opcode=0x{opcode:x}, funct3={funct3}, funct7={funct7}"
    )


def ControlUnit(opcode: int, funct3: int = 0, funct7: int = 0) -> int:
    global RegWrite, Branch, MemRead, MemWrite, MemtoReg, ALUSrc, ALUOp

    reset_control_signals()

    # lw
    if opcode == 0x03:
        RegWrite = 1
        MemRead = 1
        MemtoReg = 1
        ALUSrc = 1
        ALUOp = 0b00

    # sw
    elif opcode == 0x23:
        MemWrite = 1
        ALUSrc = 1
        ALUOp = 0b00

    # beq
    elif opcode == 0x63:
        Branch = 1
        ALUOp = 0b01

    # R-type add/sub/and/or
    elif opcode == 0x33:
        RegWrite = 1
        ALUSrc = 0
        ALUOp = 0b10

    # I-type addi/andi/ori
    elif opcode == 0x13:
        RegWrite = 1
        ALUSrc = 1
        ALUOp = 0b10

    else:
        raise ValueError(f"Unsupported opcode for Section 1: 0x{opcode:x}")

    return ALUControl(ALUOp, funct3, funct7, opcode)

def Fetch(program: List[int]) -> Optional[int]:
    global pc, next_pc

    index = pc // 4
    if index < 0 or index >= len(program):
        return None

    instr_word = program[index]
    next_pc = pc + 4
    return instr_word



def Decode(instr_word: int) -> Tuple[DecodedInstruction, int]:
    global rf

    bin_str = instruction_to_bin(instr_word)
    opcode = int(extract_bits(bin_str, 6, 0), 2)

    rd = rs1 = rs2 = funct3 = funct7 = imm = 0
    mnemonic = "unknown"

    if opcode == 0x33:  # R-type
        rd = int(extract_bits(bin_str, 11, 7), 2)
        funct3 = int(extract_bits(bin_str, 14, 12), 2)
        rs1 = int(extract_bits(bin_str, 19, 15), 2)
        rs2 = int(extract_bits(bin_str, 24, 20), 2)
        funct7 = int(extract_bits(bin_str, 31, 25), 2)

        if funct3 == 0x0 and funct7 == 0x00:
            mnemonic = "add"
        elif funct3 == 0x0 and funct7 == 0x20:
            mnemonic = "sub"
        elif funct3 == 0x7 and funct7 == 0x00:
            mnemonic = "and"
        elif funct3 == 0x6 and funct7 == 0x00:
            mnemonic = "or"
        elif funct3 == 0x2 and funct7 == 0x00:
            mnemonic = "slt"
        else:
            raise ValueError("Unsupported R-type instruction in Section 1")

    elif opcode == 0x13:  # I-type arithmetic
        rd = int(extract_bits(bin_str, 11, 7), 2)
        funct3 = int(extract_bits(bin_str, 14, 12), 2)
        rs1 = int(extract_bits(bin_str, 19, 15), 2)
        imm = sign_extend(int(extract_bits(bin_str, 31, 20), 2), 12)

        if funct3 == 0x0:
            mnemonic = "addi"
        elif funct3 == 0x7:
            mnemonic = "andi"
        elif funct3 == 0x6:
            mnemonic = "ori"
        else:
            raise ValueError("Unsupported I-type arithmetic instruction in Section 1")

    elif opcode == 0x03:  # lw
        rd = int(extract_bits(bin_str, 11, 7), 2)
        funct3 = int(extract_bits(bin_str, 14, 12), 2)
        rs1 = int(extract_bits(bin_str, 19, 15), 2)
        imm = sign_extend(int(extract_bits(bin_str, 31, 20), 2), 12)
        if funct3 != 0x2:
            raise ValueError("Only lw is supported for load opcode in Section 1")
        mnemonic = "lw"

    elif opcode == 0x23:  # sw
        funct3 = int(extract_bits(bin_str, 14, 12), 2)
        rs1 = int(extract_bits(bin_str, 19, 15), 2)
        rs2 = int(extract_bits(bin_str, 24, 20), 2)
        imm_high = int(extract_bits(bin_str, 31, 25), 2)
        imm_low = int(extract_bits(bin_str, 11, 7), 2)
        imm = sign_extend((imm_high << 5) | imm_low, 12)
        if funct3 != 0x2:
            raise ValueError("Only sw is supported for store opcode in Section 1")
        mnemonic = "sw"

    elif opcode == 0x63:  # beq
        funct3 = int(extract_bits(bin_str, 14, 12), 2)
        rs1 = int(extract_bits(bin_str, 19, 15), 2)
        rs2 = int(extract_bits(bin_str, 24, 20), 2)
        imm12 = int(extract_bits(bin_str, 31, 31), 2)
        imm10_5 = int(extract_bits(bin_str, 30, 25), 2)
        imm4_1 = int(extract_bits(bin_str, 11, 8), 2)
        imm11 = int(extract_bits(bin_str, 7, 7), 2)
        # decoded branch offset the datapath's shift-left-1.
        # should shift-left-1 the sign-extended offset and then add it to PC+4.
        imm = sign_extend((imm12 << 11) | (imm11 << 10) | (imm10_5 << 4) | imm4_1, 12)
        if funct3 != 0x0:
            raise ValueError("Only beq is supported for branch opcode in Section 1")
        mnemonic = "beq"

    else:
        raise ValueError(f"Unsupported opcode in Section 1: 0x{opcode:x}")

    alu_ctrl = ControlUnit(opcode, funct3, funct7)

    decoded = DecodedInstruction(
        instr_word=instr_word,
        opcode=opcode,
        mnemonic=mnemonic,
        rd=rd,
        rs1=rs1,
        rs2=rs2,
        funct3=funct3,
        funct7=funct7,
        imm=imm,
        rs1_val=rf[rs1],
        rs2_val=rf[rs2],
    )

    return decoded, alu_ctrl



def Execute(decoded: DecodedInstruction, alu_ctrl: int) -> ExecuteResult:
    global alu_zero, branch_target, next_pc

    op_a = decoded.rs1_val
    op_b = decoded.imm if ALUSrc else decoded.rs2_val

    if alu_ctrl == 0b0000:      # AND
        alu_result = op_a & op_b
    elif alu_ctrl == 0b0001:    # OR
        alu_result = op_a | op_b
    elif alu_ctrl == 0b0010:    # ADD
        alu_result = op_a + op_b
    elif alu_ctrl == 0b0110:    # SUB
        alu_result = op_a - op_b
    elif alu_ctrl == 0b0101:    # SLT
        alu_result = 1 if signed32(op_a) < signed32(op_b) else 0
    else:
        raise ValueError(f"Unsupported alu_ctrl: {alu_ctrl:04b}")

    alu_zero = 1 if alu_result == 0 else 0

    # shift-left-1 offset, add to PC+4.
    branch_target = next_pc + (decoded.imm << 1)
    branch_taken = bool(Branch and alu_zero)

    mem_address = alu_result

    return ExecuteResult(
        alu_result=alu_result,
        store_data=decoded.rs2_val,
        rd=decoded.rd,
        branch_taken=branch_taken,
        mem_address=mem_address,
    )



def Mem(ex_result: ExecuteResult) -> MemResult:
    global d_mem

    mem_data = 0
    store_address = None
    store_value = None

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
        store_value = d_mem[idx]

    return MemResult(
        alu_result=ex_result.alu_result,
        mem_data=mem_data,
        rd=ex_result.rd,
        store_address=store_address,
        store_value=store_value,
    )



def Writeback(mem_result: MemResult, branch_taken: bool) -> List[str]:
    global rf, pc, next_pc, branch_target, total_clock_cycles

    messages = []

    if RegWrite and mem_result.rd != 0:
        wb_value = mem_result.mem_data if MemtoReg else mem_result.alu_result
        rf[mem_result.rd] = wb_value & 0xFFFFFFFF
        messages.append(f"{reg_name(mem_result.rd)} is modified to {to_hex32(rf[mem_result.rd])}")

    if mem_result.store_address is not None:
        messages.append(
            f"memory {to_hex32(mem_result.store_address)} is modified to {to_hex32(mem_result.store_value)}"
        )

    pc = branch_target if branch_taken else next_pc
    total_clock_cycles += 1
    messages.append(f"pc is modified to {to_hex32(pc)}")

    rf[0] = 0
    return messages

def initialize_arch_state() -> None:
    global pc, next_pc, branch_target, alu_zero, total_clock_cycles, rf, d_mem

    pc = 0
    next_pc = 0
    branch_target = 0
    alu_zero = 0
    total_clock_cycles = 0

    rf = [0] * 32
    d_mem = [0] * 64

    rf[1] = 0x20
    rf[2] = 0x5
    rf[10] = 0x70
    rf[11] = 0x4

    d_mem[0x70 // 4] = 0x5
    d_mem[0x74 // 4] = 0x10



def load_program(filename: str) -> List[int]:
    program = []
    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue
            program.append(parse_instruction_line(line))
    return program



def run_program(program: List[int]) -> None:
    while True:
        instr_word = Fetch(program)
        if instr_word is None:
            break

        decoded, alu_ctrl = Decode(instr_word)
        ex_result = Execute(decoded, alu_ctrl)
        mem_result = Mem(ex_result)
        messages = Writeback(mem_result, ex_result.branch_taken)

        print(f"total_clock_cycles {total_clock_cycles} :")
        for msg in messages:
            print(msg)

    print("program terminated:")
    print(f"total execution time is {total_clock_cycles} cycles")



def main() -> None:
    initialize_arch_state()
    filename = input("Enter the program file name to run:\n").strip()
    program = load_program(filename)
    run_program(program)


if __name__ == "__main__":
    main()
