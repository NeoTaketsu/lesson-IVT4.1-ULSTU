import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv', sep=',')
plt.subplot(2, 1, 1)
plt.title(r'$x1 = 4 / (1 + e^-x), x2 = sin(x)$')
plt.plot(df['x1'], df['x'], label = 'x1(x)')
plt.plot(df['x2'], df['x'], label = 'x2(x)')
plt.legend()
plt.ylabel("x")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df['y'], df['x1'], label = 'y(x1)')
plt.plot(df['y'], df['x2'], label = 'y(x2)')
plt.legend()
plt.ylabel("y")
plt.grid(True)

plt.show()

print(dir(plt))