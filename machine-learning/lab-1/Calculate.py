import pandas as pd

df = pd.read_csv('data.csv', sep=',')
print(df, '\n')

x1_mean = df['x1'].mean()
x2_mean = df['x2'].mean()
y_mean = df['y'].mean()
print('Среднее: x1={}, x2={} y={}\n'.format(x1_mean, x2_mean, y_mean))
print('CSV файл сохранён\n')
df[(df.x1 < x1_mean) | (df.x2 < x2_mean)][['x1', 'x2', 'y']].to_csv('new-data.csv')


x1 = df['x1'].max()
x2 = df['x2'].max()
y = df['y'].max()
print('Max: x1={}, x2={} y={}'.format(x1, x2, y))

x1 = df['x1'].min()
x2 = df['x2'].min()
y = df['y'].min()
print('Min: x1={}, x2={} y={}'.format(x1, x2, y))


x1 = df['x1']
x1 = x1[x1 > x1_mean].sum()
x2 = df['x2']
x2 = x2[x2 > x2_mean].sum()
y = df['y']
y = y[y > y_mean].sum()
print('\nСумма элементов, которые больше среднего: x1={}, x2={} y={}'.format(x1, x2, y))