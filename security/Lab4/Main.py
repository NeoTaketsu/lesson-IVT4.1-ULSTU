from fractions import Fraction

def fact(n):
    for i in range(2, int(n ** (1/3)) + 1):
        if n % i == 0:
            return True
    return False

print(fact(268))