import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv', sep=',')
df = df.sort_values(['x1','y'], ascending=[False,True])
plt.subplot(2, 1, 1)
plt.plot(df['x1'], df['y'])
plt.grid(True)

df = df.sort_values(['x2','y'], ascending=[False,True])
plt.subplot(2, 1, 2)
plt.plot(df['x2'], df['y'])
plt.grid(True)

plt.show()

print(dir(plt))