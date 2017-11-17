import random
import math
import pandas as pd

def generate():
    list = []
    for i in range(500):
        list.append(random.uniform(-3.14, 3.14))
    return list


x = generate()
x.sort()

x1 = [(4 / (1 + math.exp(-x1))) for x1 in x]
x2 = [math.sin(x2) for x2 in x]
y = [(y+y) for y in x]
df = pd.DataFrame({
    'x': x,
    'x1':x1,
    'x2': x2,
    'y': y
})

df = df.sort_values(['x1'])
print(df, '\n')
df.to_csv('data.csv')