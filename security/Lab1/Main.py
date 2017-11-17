import random as rand
import time

import Linear

from Lab1 import Geffe


def save_file(name, data):
    with open(name + ".txt", "w") as f:
        f.write(data)

is_exit = True
while is_exit:
    key = input("\n1. Системный рандом\n"
          "2. Программный рандом\n"
          "3. Линейный конгруэнтный генератор\n"
          "4. Генератор Геффа\n"
          "0. Выход")

    if key == '0':
        is_exit = False
        break

    size = int(input("Укажите размер последовательности: "))
    out = ""

    if (key == '1'):
        for i in range(size):
            out += bin(rand.SystemRandom().randint(0, 2 ** 31)).lstrip('-0b')
        print(out)
        save_file("system", out)
    elif (key == '2'):
        for i in range(size):
            out += bin(rand.randint(0, 2 ** 31)).lstrip('-0b')
        print(out)
        save_file("programm", out)
    elif (key == '3'):
        startTime = time.time()
        lin = Linear.Linear()
        out = lin.random_bin(rand.randint(0, 2 ** 31), size)
        endTime = time.time()
        fullTime = endTime - startTime
        print(fullTime)
        save_file("linear", out)
    elif (key == '4'):
        startTime = time.time()
        gef = Geffe.Geffe(rand.randint(0, 2 ** 31), rand.randint(0, 2 ** 31), rand.randint(0, 2 ** 31))
        out = gef.geffe(size * 15)
        endTime = time.time()
        fullTime = endTime - startTime
        print(fullTime)
        save_file("geffe", out)
    else:
        is_exit = False
        break