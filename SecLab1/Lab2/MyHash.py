class Hash:
    def __init__(self):
        self.size = 2 ** 31 - 1

    def hashing_password(self, password):
        s = 0
        mult = 1
        for i in range(len(password) - 1):
            s += (abs(ord(password[i]) - ord(password[i + 1])) * mult)
            s %= self.size
            mult *= 63
            mult %= self.size

        return s