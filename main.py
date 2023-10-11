from sklearn.ensemble import RandomForestRegressor
from sklearn.metric import mean_squared_error
import numpy 
from .module_a import polynom_3
from useful_package.module_b import hyperbola

x_train = np.arange(-5,5, 10**3)
y_train = hyperbola(x_train)
x_test = np.arange(-1, 1, 10**3)
y_test = hyperbola(x_test)

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
mse_hyperbola = mean_squared_error(y_true, y_pred)
print(f"{mse_hyperbola=}")


x_train = np.arange(-5,5, 10**3)
y_train = polynom_3(x_train)
x_test = np.arange(-1, 1, 10**3)
y_test = polynom_3(x_test)

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
mse_polynom = mean_squared_error(y_true, y_pred)
print(f"{mse_polynom=}")

