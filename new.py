from dataclasses import dataclass, field
from typing import Optional

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

pc               = 0
total_clock_cycles = 0
rf    = [0] * 32
d_mem = [0] * 64


@dataclass
class IF_ID:
    valid:      bool = False
    pc:         int  = 0
    next_pc:    int  = 0
    instr_word: int  = 0

@dataclass
class CtrlSignals:
    RegWrite:    int = 0
    branch_type: int = 0   # 0=none, 1=BEQ, 2=BNE
    MemRead:     int = 0
    MemWrite:    int = 0
    MemtoReg:    int = 0   # 0=ALU, 1=Mem, 2=PC+4
    ALUSrc:      int = 0
    Jump:        int = 0
    JumpReg:     int = 0

@dataclass
class ID_EX:
    valid:      bool        = False
    ctrl:       CtrlSignals = field(default_factory=CtrlSignals)
    pc:         int  = 0
    next_pc:    int  = 0
    rs1_val:    int  = 0
    rs2_val:    int  = 0
    imm:        int  = 0
    rd:         int  = 0
    rs1:        int  = 0
    rs2:        int  = 0
    alu_ctrl:   int  = 0
    mem_funct3: int  = 0
    mnemonic:   str  = ""

@dataclass
class EX_MEM:
    valid:         bool        = False
    ctrl:          CtrlSignals = field(default_factory=CtrlSignals)
    next_pc:       int  = 0
    alu_result:    int  = 0
    store_data:    int  = 0
    rd:            int  = 0
    branch_taken:  bool = False
    jump_taken:    bool = False
    branch_target: int  = 0
    jump_target:   int  = 0
    mem_funct3:    int  = 0
    mnemonic:      str  = ""

@dataclass
class MEM_WB:
    valid:         bool        = False
    ctrl:          CtrlSignals = field(default_factory=CtrlSignals)
    next_pc:       int  = 0
    alu_result:    int  = 0
    mem_data:      int  = 0
    rd:            int  = 0
    store_address: Optional[int] = None
    store_value:   Optional[int] = None
    mnemonic:      str  = ""


if_id  = IF_ID()
id_ex  = ID_EX()
ex_mem = EX_MEM()
mem_wb = MEM_WB()


def bits(b, hi, lo):
    return int(b[31 - hi : 32 - lo], 2)

def sign_extend(val, n):
    return val - (1 << n) if val & (1 << (n - 1)) else val

def signed32(val):
    return val if val < (1 << 31) else val - (1 << 32)

def to_hex32(val):
    return f"0x{val & 0xFFFFFFFF:x}"

def reg_name(idx):
    return ABI_NAMES.get(idx, f"x{idx}")

def to_bin(instr):
    return format(instr & 0xFFFFFFFF, "032b")

def parse_line(line):
    t = line.strip()
    if not t:
        raise ValueError("empty")
    if set(t) <= {"0", "1"} and len(t) <= 32:
        return int(t, 2)
    return int(t.lower().replace("0x", ""), 16)

# Load, I-arith, and JALR all share the same field positions
def i_fields(b):
    return bits(b, 11, 7), bits(b, 14, 12), bits(b, 19, 15), sign_extend(bits(b, 31, 20), 12)


def ALUControl(alu_op, funct3, funct7, opcode):
    if alu_op == 0b00: return 0b0010
    if alu_op == 0b01: return 0b0110
    if alu_op == 0b10:
        if opcode == 0x33:
            r = {(0,0):2,(0,32):6,(7,0):0,(6,0):1,(2,0):5,(1,0):3,(5,0):4}
            if (funct3, funct7) in r: return r[(funct3, funct7)]
        elif opcode == 0x13:
            i = {0:2, 7:0, 6:1}
            if funct3 in i: return i[funct3]
    raise ValueError(f"Unsupported ALUControl: alu_op={alu_op:02b} opcode=0x{opcode:02x} funct3={funct3} funct7={funct7}")


def ControlUnit(opcode, funct3=0, funct7=0):
    c, alu_op = CtrlSignals(), 0

    if opcode == 0x03:
        c.RegWrite = c.MemRead = c.ALUSrc = 1; c.MemtoReg = 1
    elif opcode == 0x23:
        c.MemWrite = c.ALUSrc = 1
    elif opcode == 0x63:
        c.branch_type = {0: 1, 1: 2}.get(funct3)
        if c.branch_type is None: raise ValueError(f"Unsupported branch funct3={funct3}")
        alu_op = 0b01
    elif opcode == 0x33:
        c.RegWrite = 1; alu_op = 0b10
    elif opcode == 0x13:
        c.RegWrite = c.ALUSrc = 1; alu_op = 0b10
    elif opcode == 0x6F:
        c.Jump = c.RegWrite = 1; c.MemtoReg = 2
    elif opcode == 0x67:
        c.JumpReg = c.RegWrite = c.ALUSrc = 1; c.MemtoReg = 2
    else:
        raise ValueError(f"Unsupported opcode: 0x{opcode:02x}")

    return c, ALUControl(alu_op, funct3, funct7, opcode)


def decode_fields(instr_word):
    b   = to_bin(instr_word)
    opc = bits(b, 6, 0)

    rd = rs1 = rs2 = funct3 = funct7 = imm = 0
    mnemonic = "nop"

    if opc == 0x33:
        rd     = bits(b, 11,  7)
        funct3 = bits(b, 14, 12)
        rs1    = bits(b, 19, 15)
        rs2    = bits(b, 24, 20)
        funct7 = bits(b, 31, 25)
        mnemonic = {
            (0,  0): "add", (0, 32): "sub", (7, 0): "and",
            (6,  0): "or",  (2,  0): "slt", (1,  0): "sll",
            (5,  0): "srl",
        }.get((funct3, funct7), "r-type")

    elif opc == 0x13:
        rd, funct3, rs1, imm = i_fields(b)
        mnemonic = {0: "addi", 7: "andi", 6: "ori"}.get(funct3, "i-arith")

    elif opc == 0x03:
        rd, funct3, rs1, imm = i_fields(b)
        mnemonic = {2: "lw", 0: "lb"}.get(funct3)
        if mnemonic is None: raise ValueError(f"Unsupported load funct3={funct3}")

    elif opc == 0x23:
        funct3 = bits(b, 14, 12)
        rs1    = bits(b, 19, 15)
        rs2    = bits(b, 24, 20)
        imm    = sign_extend((bits(b, 31, 25) << 5) | bits(b, 11, 7), 12)
        mnemonic = {2: "sw", 0: "sb"}.get(funct3)
        if mnemonic is None: raise ValueError(f"Unsupported store funct3={funct3}")

    elif opc == 0x63:
        funct3 = bits(b, 14, 12)
        rs1    = bits(b, 19, 15)
        rs2    = bits(b, 24, 20)
        imm    = sign_extend(
            (bits(b, 31, 31) << 11) | (bits(b, 7, 7) << 10) |
            (bits(b, 30, 25) << 4)  |  bits(b, 11, 8), 12
        )
        mnemonic = {0: "beq", 1: "bne"}.get(funct3)
        if mnemonic is None: raise ValueError(f"Unsupported branch funct3={funct3}")

    elif opc == 0x6F:
        rd  = bits(b, 11, 7)
        raw = (bits(b, 31, 31) << 19) | (bits(b, 19, 12) << 11) | \
              (bits(b, 20, 20) << 10) |  bits(b, 30, 21)
        imm = sign_extend(raw, 20) << 1
        mnemonic = "jal"

    elif opc == 0x67:
        rd, funct3, rs1, imm = i_fields(b)
        mnemonic = "jalr"

    else:
        raise ValueError(f"Unsupported opcode: 0x{opc:02x}")

    ctrl, alu_ctrl = ControlUnit(opc, funct3, funct7)
    return dict(
        opcode=opc, mnemonic=mnemonic, rd=rd, rs1=rs1, rs2=rs2,
        funct3=funct3, funct7=funct7, imm=imm,
        ctrl=ctrl, alu_ctrl=alu_ctrl, mem_funct3=funct3,
    )


def raw_hazard(if_id_l, id_ex_l, ex_mem_l):
    if not if_id_l.valid:
        return False
    try:
        f = decode_fields(if_id_l.instr_word)
    except Exception:
        return False

    needs = {r for r in (f["rs1"], f["rs2"]) if r != 0}
    if f["opcode"] not in (0x33, 0x23, 0x63):
        needs.discard(f["rs2"])
    if not needs:
        return False

    for stage in (id_ex_l, ex_mem_l):
        if stage.valid and stage.ctrl.RegWrite and stage.rd != 0 and stage.rd in needs:
            return True
    return False


def stage_IF(program, pc_in):
    idx = pc_in // 4
    if not (0 <= idx < len(program)):
        return IF_ID(valid=False), pc_in
    return IF_ID(valid=True, pc=pc_in, next_pc=pc_in + 4, instr_word=program[idx]), pc_in + 4


def stage_ID(if_id_in):
    if not if_id_in.valid:
        return ID_EX(valid=False)
    try:
        f = decode_fields(if_id_in.instr_word)
    except Exception:
        return ID_EX(valid=False)
    return ID_EX(
        valid=True, ctrl=f["ctrl"],
        pc=if_id_in.pc, next_pc=if_id_in.next_pc,
        rs1_val=rf[f["rs1"]], rs2_val=rf[f["rs2"]],
        imm=f["imm"], rd=f["rd"], rs1=f["rs1"], rs2=f["rs2"],
        alu_ctrl=f["alu_ctrl"], mem_funct3=f["mem_funct3"],
        mnemonic=f["mnemonic"],
    )


def stage_EX(id_ex_in):
    if not id_ex_in.valid:
        return EX_MEM(valid=False)

    ctrl = id_ex_in.ctrl
    a    = id_ex_in.rs1_val
    b    = id_ex_in.imm if ctrl.ALUSrc else id_ex_in.rs2_val

    ac = id_ex_in.alu_ctrl
    if   ac == 0: result = a & b
    elif ac == 1: result = a | b
    elif ac == 2: result = a + b
    elif ac == 3: result = (a << (b & 0x1F)) & 0xFFFFFFFF
    elif ac == 4: result = (a & 0xFFFFFFFF) >> (b & 0x1F)
    elif ac == 5: result = 1 if signed32(a) < signed32(b) else 0
    elif ac == 6: result = a - b
    else:         result = 0

    branch_taken  = (ctrl.branch_type == 1 and result == 0) or \
                    (ctrl.branch_type == 2 and result != 0)
    branch_target = id_ex_in.next_pc + (id_ex_in.imm << 1)

    if ctrl.Jump:       jump_target = id_ex_in.pc + id_ex_in.imm
    elif ctrl.JumpReg:  jump_target = (id_ex_in.rs1_val + id_ex_in.imm) & ~1
    else:               jump_target = 0

    return EX_MEM(
        valid=True, ctrl=ctrl,
        next_pc=id_ex_in.next_pc,
        alu_result=result, store_data=id_ex_in.rs2_val,
        rd=id_ex_in.rd,
        branch_taken=branch_taken, jump_taken=bool(ctrl.Jump or ctrl.JumpReg),
        branch_target=branch_target, jump_target=jump_target,
        mem_funct3=id_ex_in.mem_funct3, mnemonic=id_ex_in.mnemonic,
    )


def stage_MEM(ex_mem_in):
    if not ex_mem_in.valid:
        return MEM_WB(valid=False)

    ctrl = ex_mem_in.ctrl
    addr = ex_mem_in.alu_result & 0xFFFFFFFF
    wi   = addr // 4
    lane = addr  % 4
    mem_data = store_address = store_value = 0
    store_address = store_value = None

    if ctrl.MemRead:
        if ex_mem_in.mem_funct3 == 0:
            mem_data = sign_extend((d_mem[wi] >> (lane * 8)) & 0xFF, 8)
        else:
            mem_data = d_mem[wi]

    if ctrl.MemWrite:
        if ex_mem_in.mem_funct3 == 0:
            shift = lane * 8
            d_mem[wi] = (d_mem[wi] & ((~(0xFF << shift)) & 0xFFFFFFFF)) | \
                        ((ex_mem_in.store_data & 0xFF) << shift)
        else:
            d_mem[wi] = ex_mem_in.store_data & 0xFFFFFFFF
        store_address = addr
        store_value   = d_mem[wi]

    return MEM_WB(
        valid=True, ctrl=ctrl,
        next_pc=ex_mem_in.next_pc,
        alu_result=ex_mem_in.alu_result, mem_data=mem_data,
        rd=ex_mem_in.rd,
        store_address=store_address, store_value=store_value,
        mnemonic=ex_mem_in.mnemonic,
    )


def stage_WB(mem_wb_in):
    if not mem_wb_in.valid:
        return []

    ctrl, messages = mem_wb_in.ctrl, []

    if ctrl.RegWrite and mem_wb_in.rd != 0:
        if   ctrl.MemtoReg == 1: val = mem_wb_in.mem_data
        elif ctrl.MemtoReg == 2: val = mem_wb_in.next_pc
        else:                    val = mem_wb_in.alu_result
        rf[mem_wb_in.rd] = val & 0xFFFFFFFF
        messages.append(f"{reg_name(mem_wb_in.rd)} is modified to {to_hex32(rf[mem_wb_in.rd])}")

    if mem_wb_in.store_address is not None:
        messages.append(
            f"memory {to_hex32(mem_wb_in.store_address)} is modified to "
            f"{to_hex32(mem_wb_in.store_value)}"
        )

    rf[0] = 0
    return messages


def run_pipeline(program):
    global pc, total_clock_cycles
    global if_id, id_ex, ex_mem, mem_wb

    program_done = False
    next_seq_pc  = pc

    while True:
        wb_messages         = stage_WB(mem_wb)
        wb_had_valid        = mem_wb.valid
        wb_snapshot_next_pc = mem_wb.next_pc

        stall      = raw_hazard(if_id, id_ex, ex_mem)
        new_mem_wb = stage_MEM(ex_mem)
        new_ex_mem = stage_EX(id_ex)

        flush  = False
        new_pc = pc
        if new_ex_mem.valid and (new_ex_mem.branch_taken or new_ex_mem.jump_taken):
            flush  = True
            new_pc = (new_ex_mem.jump_target if new_ex_mem.jump_taken
                      else new_ex_mem.branch_target) & 0xFFFFFFFF

        new_id_ex = ID_EX(valid=False) if (stall or flush) else stage_ID(if_id)

        if flush:
            new_if_id = IF_ID(valid=False)
        elif stall:
            new_if_id = if_id
        elif program_done:
            new_if_id = IF_ID(valid=False)
        else:
            new_if_id, next_seq_pc = stage_IF(program, pc)
            if not new_if_id.valid:
                program_done = True

        mem_wb = new_mem_wb
        ex_mem = new_ex_mem
        id_ex  = new_id_ex
        if_id  = new_if_id

        if flush:
            pc = new_pc
        elif not stall and not program_done:
            pc = next_seq_pc

        if wb_had_valid:
            total_clock_cycles += 1
            print(f"total_clock_cycles {total_clock_cycles} :")

            for msg in wb_messages:
                print(msg)

            wb_pc = wb_snapshot_next_pc

            # If the instruction completing WB was a jump,
            # show its redirected target instead of PC+4
            if mem_wb.ctrl.Jump or mem_wb.ctrl.JumpReg:
                if ex_mem.jump_taken:
                    wb_pc = ex_mem.jump_target

            print(f"pc is modified to {to_hex32(wb_pc)}")

        if program_done and not any([if_id.valid, id_ex.valid, ex_mem.valid, mem_wb.valid]):
            break

    print("program terminated:")
    print(f"total execution time is {total_clock_cycles} cycles")


def _reset():
    global if_id, id_ex, ex_mem, mem_wb
    if_id = IF_ID(); id_ex = ID_EX(); ex_mem = EX_MEM(); mem_wb = MEM_WB()


def initialize(section):
    global pc, total_clock_cycles, rf, d_mem
    pc = total_clock_cycles = 0
    rf = [0] * 32; d_mem = [0] * 64

    if section == 1:
        rf[1] = 0x20; rf[2] = 0x5; rf[10] = 0x70; rf[11] = 0x4
        d_mem[0x70 // 4] = 0x5; d_mem[0x74 // 4] = 0x10
    elif section == 2:
        rf[8] = 0x20; rf[10] = 0x5; rf[11] = 0x2; rf[12] = 0xa; rf[13] = 0xf
    elif section == 3:
        rf[1] = 0x20; rf[2] = 0x4; rf[3] = 0xAB
        rf[4] = 0x3;  rf[5] = 0x7; rf[6]  = 0x3
        d_mem[0x20 // 4] = 0xDEADBEEF

    _reset()


def load_program(filename):
    program = []
    with open(filename, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            program.append(parse_line(line))
    return program


def main():
    filename = input("Enter the program file name to run:\n").strip()
    name = filename.lower()

    if "part2" in name or "section2" in name:
        initialize(2)
    elif "extra" in name or "part3" in name:
        initialize(3)
    else:
        initialize(1)

    run_pipeline(load_program(filename))


if __name__ == "__main__":
    main()
