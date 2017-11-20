from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn import preprocessing


def compare(pred_y, y_test):
    print(type(y_test))
    y_test = y_test.values.tolist()
    count = 0
    for i in range(len(pred_y)):
        if int(pred_y[i]) == int(y_test[i]):
            count += 1
    print(count, len(pred_y))

def linear(x_train, y_train, x_test, y_test):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    print('Coefficients: \n', regr.coef_)
    pred_y = regr.predict(x_test)
    print("Mean squared error: %.2f" % mean_squared_error(y_test, pred_y))
    compare(pred_y.tolist(), y_test)
    pd.DataFrame({
        'y_test': y_test,
        'y_pred': pred_y
    }).to_csv('data_tmp.txt')

    print('Variance score: %.2f' % r2_score(y_test, pred_y))



def mpl(x_train, y_train, x_test, y_test):
    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)

    """MPL fit start"""
    mlp = MLPClassifier(activation='identity', alpha=1, batch_size='auto', beta_1=0.9,
                        beta_2=0.999, early_stopping=False, epsilon=1e-08,
                        hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
                        learning_rate_init=0.01, max_iter=200, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=None,
                        shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.1,
                        verbose=False, warm_start=False)
    mlp.fit(x_train, y_train)
    pred_y = mlp.predict(x_test)
    print("MLP mean squared error: %.2f" % mean_squared_error(y_test, pred_y))
    print('MLP variance score: %.2f' % r2_score(y_test, pred_y))
    print(classification_report(y_test, pred_y))
    columns = ['pred_y']
    pred_y = pd.DataFrame(pred_y, columns=columns)
    y_test = y_test.reset_index(drop=True)
    res = pd.concat([pred_y, y_test], axis=1)
    res.to_csv("MLPResult.txt", index=False)

    pred_y = pred_y['pred_y'].values.tolist()
    compare(pred_y, y_test)

def perceptron(x_train, y_train, x_test, y_test):
    """Perceptron fit start"""
    prc = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False, n_iter=None)
    prc.fit(x_train, y_train)
    pred_y = prc.predict(x_test)
    print("Perceptron mean squared error: %.2f" % mean_squared_error(y_test, pred_y))
    print('Perceptron variance score: %.2f' % r2_score(y_test, pred_y))
    print(classification_report(y_test, pred_y))
    columns = ['pred_y']
    pred_y = pd.DataFrame(pred_y, columns=columns)
    y_test = y_test.reset_index(drop=True)
    res = pd.concat([pred_y, y_test], axis=1)
    res.to_csv("PerceptronResult.txt", index=False)
    compare(pred_y['pred_y'].values.tolist(), y_test)
    """Perceptron fit end"""


df = pd.read_csv('data.csv', engine='python', sep=';', index_col=False)
print(df)
train_count = int(len(df.index) * 0.7)
x_train = df.iloc[:, :6]
x_test = df.iloc[train_count:, :6]
y_train = df.iloc[:, 6]
y_test = df.iloc[train_count:, 6]

linear(x_train, y_train, x_test, y_test)
mpl(x_train, y_train, x_test, y_test)
perceptron(x_train, y_train, x_test, y_test)