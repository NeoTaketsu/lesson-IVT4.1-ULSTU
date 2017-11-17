import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
'''
 learning_rate_init - коэффициента обучения
 alpha - параметра регуляризации
 activation - функции оптимизации
'''
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

"""MPL fit start"""
mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
mlp.fit(x_train, y_train)
pred_y = mlp.predict(x_test)
print("MLP mean squared error: %.2f" % mean_squared_error(y_test, pred_y))
print('MLP variance score: %.2f' % r2_score(y_test,pred_y))
print(classification_report(y_test,pred_y))
columns = ['pred_y']
pred_y = pd.DataFrame(pred_y, columns=columns)
y_test = y_test.reset_index(drop=True)
res = pd.concat([pred_y, y_test], axis=1)
res.to_csv("MLPResult.txt", index = False)


"""MPL fit start"""
alpha = 100
n = 10
err = []
for i in range(n):
    mlp = MLPClassifier(activation='relu', alpha=alpha, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)
    mlp.fit(x_train, y_train)
    pred_y = mlp.predict(x_test)
    print("MLP mean squared error: %.2f" % mean_squared_error(y_test, pred_y))
    err.append(r2_score(y_test,pred_y))
    print('MLP variance score: %.2f' % r2_score(y_test,pred_y))
    print(classification_report(y_test,pred_y))
    columns = ['pred_y']
    pred_y = pd.DataFrame(pred_y, columns=columns)
    y_test = y_test.reset_index(drop=True)
    res = pd.concat([pred_y, y_test], axis=1)
    res.to_csv("Отчёт.txt", index = False)
    alpha *= 0.1


"""Perceptron fit start"""
prc = Perceptron(shuffle=False)
prc.fit(x_train, y_train)
pred_y = prc.predict(x_test)
print("Perceptron mean squared error: %.2f" % mean_squared_error(y_test, pred_y))
print('Perceptron variance score: %.2f' % r2_score(y_test,pred_y))
print(classification_report(y_test,pred_y))
columns = ['pred_y']
pred_y = pd.DataFrame(pred_y, columns=columns)
y_test = y_test.reset_index(drop=True)
res = pd.concat([pred_y, y_test], axis=1)
res.to_csv("PerceptronResult.txt", index = False)
"""Perceptron fit end"""