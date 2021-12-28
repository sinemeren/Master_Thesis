import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data_dir = "data/"
file_name_train = 'dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1.xls'
input_size = 4

# read the excel file with pandas
df_train = pd.read_excel(data_dir + file_name_train)

# convert it to numpy
data_train = df_train.to_numpy()

# input and output
x_train, y_train = (data_train[:, :input_size], data_train[:, input_size:])

polynomial_regression = PolynomialFeatures(degree=2)
x_poly = polynomial_regression.fit_transform(x_train)
polynomial_regression.fit(x_poly, y_train)
linear_regression = LinearRegression()
linear_regression.fit(x_poly, y_train)


file_name_test = 'dataWithDepthAndDiameter_TEST_new_1.xls'

# read the excel file with pandas
df_test = pd.read_excel(data_dir + file_name_test)

# convert it to numpy
data_test = df_test.to_numpy()

# input and output
x_test, y_test = (data_test[:, :input_size], data_test[:, input_size:])

y_predicted = linear_regression.predict(
    polynomial_regression.fit_transform(x_test))

print("target", y_test)
print("predictions", y_predicted)

absolute_error = abs(y_predicted - y_test)
print("absolute error", absolute_error)

print("average prediction error", np.mean(absolute_error, axis=0))

x_pos = np.linspace(0, len(y_test)-1, len(y_test))


plt.xlabel('Testing Index')
plt.ylabel('Diameter')
plt.plot(x_pos, y_predicted[:, 0], 'x', label='Predicted Data')
plt.plot(x_pos, y_test[:, 0], '^', label='Experimental Data')
plt.title("Polynomial Regression - Diameter")
plt.legend()
plt.show()

plt.xlabel('Testing Index')
plt.ylabel('Depth')
plt.plot(x_pos, y_predicted[:, 1], 'x', label='Predicted Data')
plt.plot(x_pos, y_test[:, 1], '^', label='Experimental Data')
plt.title("Polynomial Regression - Depth")
plt.legend()
plt.show()
