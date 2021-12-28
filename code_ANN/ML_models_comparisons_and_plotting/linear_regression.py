import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# training data
data_dir = "data/"
file_name_train = 'dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1.xls'
input_size = 4

# read the excel file with pandas
df_train = pd.read_excel(data_dir + file_name_train)

# convert it to numpy
data_train = df_train.to_numpy()

# input and output
x_train, y_train = (data_train[:, :input_size], data_train[:, input_size:])

# linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

file_name_test = 'dataWithDepthAndDiameter_TEST_new_1.xls'

# read the excel file with pandas
df_test = pd.read_excel(data_dir + file_name_test)

# convert it to numpy
data_test = df_test.to_numpy()

# input and output
x_test, y_test = (data_test[:, :input_size], data_test[:, input_size:])

y_predicted = model.predict(x_test)

print("target", y_test)
print("predictions", y_predicted)

abs_error = abs(y_predicted - y_test)
abs_error
print("average prediction error", np.mean(abs_error, axis=0))


mse_error = mean_squared_error(y_test, y_predicted)

x_pos = np.linspace(0, len(y_test)-1, len(y_test))

plt.xlabel('Testing Index')
plt.ylabel('Diameter')
plt.plot(x_pos, y_predicted[:, 0], 'x', label='Predicted Data')
plt.plot(x_pos, y_test[:, 0], '^', label='Experimental Data')
plt.title("Linear Regression - Diameter")
plt.legend()
plt.show()

plt.xlabel('Testing Index')
plt.ylabel('Depth')
plt.plot(x_pos, y_predicted[:, 1], 'x', label='Predicted Data')
plt.plot(x_pos, y_test[:, 1], '^', label='Experimental Data')
plt.title("Linear Regression - Depth")
plt.legend()
plt.show()
