from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
import matplotlib.pyplot as plt

'''
 learning_rate_init - коэффициента обучения
 alpha - параметра регуляризации
 activation - функции оптимизации
'''

"""MPL fit start"""
def MPL(activation = 'relu', alpha=0.0001, learning_rate_init=0.001):
    global y_test, x_train, y_train
    mlp = MLPClassifier(activation=activation, alpha=alpha, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
           learning_rate_init=learning_rate_init, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)
    mlp.fit(x_train, y_train)
    pred_y = mlp.predict(x_test)
    return r2_score(y_test, pred_y)

def chart(x, y, name, xlabel = 'Alpha', ylabel = 'Variance'):
    plt.plot(x, y, label=name)
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)

    plt.show()


words = ['brickface', 'sky', 'foliage', 'cement', 'window', 'path', 'grass']
df = pd.read_csv('segmentation.data.txt', engine='python', sep=',', index_col=False)

x_train = df.iloc[:, 1:]
y = df.iloc[:, 0]
y_train = [words.index(x.lower()) for x in y]
y_train = pd.DataFrame(y_train)

df = pd.read_csv('segmentation.test.txt', engine='python', sep=',', index_col=False)

x_test = df.iloc[:, 1:]
y = df.iloc[:, 0]
y_test = [words.index(x.lower()) for x in y]
y_test = pd.DataFrame(y_test)

# standardize the data attributes
x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

#Проверка alpha
alpha = [10]
variance = []
for i in range(10):
    variance.append(MPL(alpha = alpha[i]))
    alpha.append(alpha[i] * 0.1)
alpha.pop()
m_i = variance.index(max(variance))
print(variance[m_i], alpha[m_i])
chart(alpha, variance, "MLP Alpha")

#Проверка activation
activation = ['identity', 'logistic', 'tanh', 'relu']
variance.clear()
for i in range(len(activation)):
    variance.append(MPL(activation = activation[i], alpha = 0.001))
    print(activation[i], '=', variance[i])

#Проверка learning_rate_init
learning_rate_init = [1]
variance.clear()
for i in range(6):
    variance.append(MPL(learning_rate_init = learning_rate_init[i], activation = 'tanh', alpha = 0.001))
    learning_rate_init.append(learning_rate_init[i] * 0.1)
learning_rate_init.pop()
print(learning_rate_init, variance)
m_i = variance.index(max(variance))
print(variance[m_i], learning_rate_init[m_i])
chart(learning_rate_init, variance, "MLP learning_rate_init", xlabel='learning_rate_init')



"""Perceptron fit start"""
def perceptron(alpha = 0.0001):
    prc = Perceptron(alpha = alpha, shuffle=False)
    prc.fit(x_train, y_train)
    pred_y = prc.predict(x_test)
    return r2_score(y_test, pred_y)

#Проверка alpha
alpha = [10]
variance.clear()
for i in range(10):
    variance.append(perceptron(alpha = alpha[i]))
    alpha.append(alpha[i] * 0.1)
alpha.pop()
m_i = variance.index(max(variance))
print(variance[m_i], alpha[m_i])
chart(alpha, variance, "Perceptron Alpha")
"""Perceptron fit end"""