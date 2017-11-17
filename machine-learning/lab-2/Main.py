import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, f1_score
import re
from sklearn import linear_model

regex = r"[^0-9\.]+"
columns_name = ["Lever_position", "Ship_speed_(v)_[knots]", "Gas_Turbine_shaft_torque[kN_m]", "4_-_Gas_Turbine_rate_of_revolutions_(GTn)_[rpm]", "5_-_Gas_Generator_rate_of_revolutions_(GGn)_[rpm]", "6_-_Starboard_Propeller_Torque_(Ts)_[kN]", "7_-_Port_Propeller_Torque_(Tp)_[kN]", "8_-_HP_Turbine_exit_temperature_(T48)_[C]", "9_-_GT_Compressor_inlet_air_temperature_(T1)_[C]", "10_-_GT_Compressor_outlet_air_temperature_(T2)_[C]", "11_-_HP_Turbine_exit_pressure_(P48)_[bar]", "12_-_GT_Compressor_inlet_air_pressure_(P1)_[bar]", "13_-_GT_Compressor_outlet_air_pressure_(P2)_[bar]", "14_-_Gas_Turbine_exhaust_gas_pressure_(Pexh)_[bar]", "15_-_Turbine_Injecton_Control_(TIC)_[%]", "16_-_Fuel_flow_(mf)_[kg/s]", "17_-_GT_Compressor_decay_state_coefficient.", "18_-_GT_Turbine_decay_state_coefficient._"]
df = pd.read_csv('data.txt', engine='python', sep='   ', names=columns_name, index_col=False)
df.to_csv("data_out.txt")
df = df.loc[:, (df.dtypes == np.float64) | (df.dtypes == np.int64)]

train_count = int(len(df.index) * 0.7)

x_train_1 = df.iloc[:train_count, :15]
x_train_2 = df.iloc[:train_count, 16:]
x_train = pd.concat([x_train_1, x_train_2], axis=1).as_matrix()

x_test_1 = df.iloc[train_count:, :15]
x_test_2 = df.iloc[train_count:, 16:]
x_test = pd.concat([x_test_1, x_test_2], axis=1).as_matrix()

y_train = df.iloc[:train_count, 15].as_matrix()
y_test = df.iloc[train_count:, 15].as_matrix()
# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
print('Coefficients: \n', regr.coef_)
pred_y = regr.predict(x_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, pred_y))
pd.DataFrame({
    'y_test': y_test,
    'y_pred': pred_y
}).to_csv('data_tmp.txt')
print('Variance score: %.2f' % r2_score(y_test,pred_y))


degrees = [1, 2, 3]
n = len(degrees)
err = np.empty(n)

for i in range(n):

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(x_train, y_train)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, x_train, y_train,
                             scoring="neg_mean_squared_error", cv=10)

    y_pred = pipeline.predict(x_test)
    err[i-1] = -scores.mean()
    print("Degree {}\nMSE = {}(+/- {})".format(
        degrees[i], -scores.mean(), scores.std()))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

fig = plt.figure()
plt.plot(err)
plt.show()

reg = linear_model.Lasso(alpha = 0.001)
reg.fit(x_train, y_train)

pd.DataFrame({
    'y_test': y_test,
    'y_pred': reg.predict(x_test)
}).to_csv('compare_lasso.txt')