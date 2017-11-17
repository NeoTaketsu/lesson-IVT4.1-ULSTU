import MyHash
import os
import sys
sys.path.insert(0, '../')

import Linear


def crypt(data, key):
    num = int.from_bytes(data, byteorder='little')
    num ^= key

    num_byte = (num).to_bytes(4, byteorder='little')

    return num_byte


file_name_in = input("Введите имя файла для преобразования:\n")
file_name_out = "_" + file_name_in

password = input("Введите пароль:\n")
password = MyHash.Hash().hashing_password(password)

cr = list()
lin = Linear.Linear()
lin.seed(password)

fin = open(file_name_in, "rb")
fout = open(file_name_out, "wb")

sz = os.path.getsize(file_name_in)
cur = 0
curOut = 0

byte = fin.read(4)
while byte:

    key = lin.random()

    byte_encode = crypt(byte, key)
    fout.write(byte_encode)
    byte = fin.read(4)
    #Загрузка
    cur += 1
    work = (cur * 20 // sz) * 5
    if work > curOut:
        curOut = work
        print("\rВыполнено " + str(curOut) + "%", end="")

fin.close()
fout.close()