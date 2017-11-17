import hashlib

SYNC = 100

def computeMD5hash(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()

def del_repeat(key):
    pass_new = ''
    for i in range(len(key)):
        if pass_new.find(key[i]) == -1 and key[i] != ' ':
            pass_new += key[i]

    return pass_new

def crypt(p, k):
    global SYNC
    m = (SYNC + k) % 256
    SYNC = m
    return p ^ m

OFB = lambda p, k: p ^ k

action = input('''1. Шифровать файл
2. Дешифровать файл
0. Выход''')


if (action == '0'):
    exit(0)

file_name_in = input("Введите имя файла для преобразования:\n")
file_name_out = "_" + file_name_in

key = del_repeat(computeMD5hash(input("Введите пароль: ")))
print(key)
SYNC = ord(key[0])
p_i = 0
p_m = len(key)
fin = open(file_name_in, "rb")
fout = open(file_name_out, "wb")

byte = fin.read(1)
while byte:
    num = int.from_bytes(byte, byteorder='little')
    crypt_num = 0
    if(action == '1'):
        crypt_num = crypt(num, ord(key[p_i % p_m]))
        crypt_num = OFB(crypt_num, ord(key[p_i % p_m]))
    elif(action == '2'):
        crypt_num = OFB(num, ord(key[p_i % p_m]))
        crypt_num = crypt(crypt_num, ord(key[p_i % p_m]))

    p_i = p_i + 1

    num_byte = (crypt_num).to_bytes(1, byteorder='little')
    fout.write(num_byte)
    
    byte = fin.read(1)

fin.close()
fout.close()