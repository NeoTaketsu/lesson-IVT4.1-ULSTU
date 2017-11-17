class Linear:
    def __init__(self):
        self.a = 1103515245
        self.m = 2 ** 32 - 1
        self.c = 12345
        self.x = 2
        self.bin_ans = ""

    def linear_cong(self, x):
        return (self.a * x + self.c) % self.m

    def seed(self, x):
        self.x = x

    def random_bin(self, x, n = 1):
        for i in range(n):
            x = self.linear_cong(x)
            self.bin_ans += bin(x).lstrip('-0b')
        return self.bin_ans

    def random(self):
        self.x = self.linear_cong(self.x)
        return self.x
