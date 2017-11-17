import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

df = pd.read_csv('data.csv', sep=',')

ax.plot(df['x1'], df['x2'], df['y'], label='parametric curve')
ax.legend()

plt.show()