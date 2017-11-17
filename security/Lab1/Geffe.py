class Geffe:
    def __init__(self, reg1, reg2, reg3):
        self.reg1 = reg1
        self.reg2 = reg2
        self.reg3 = reg3
        self.bin_ans = ""

    def LFSR(self, shift_register, regs):
        shift_register = (((
        (shift_register >> regs[0])
        ^ (shift_register >> regs[1])
        ^ (shift_register >> regs[2])
        ^ (shift_register >> regs[3])
        ^ shift_register) & 0x00000001) << 30) | (shift_register >> 1)

        return shift_register

    def geffe(self, n):
        for i in range(n):
            self.reg1 = self.LFSR(self.reg1, [6, 4, 2, 1])
            self.reg2 = self.LFSR(self.reg2, [2, 7, 3, 10])
            self.reg3 = self.LFSR(self.reg3, [12, 4, 11, 14])

            x1 = self.reg1 & 0x00000001
            x2 = self.reg2 & 0x00000001
            x3 = self.reg3 & 0x00000001

            self.bin_ans += str((x1 & x2) ^ (~x1 & x3))

        return self.bin_ans