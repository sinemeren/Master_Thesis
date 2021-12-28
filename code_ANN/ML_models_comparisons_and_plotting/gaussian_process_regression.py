import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct, RBF, RationalQuadratic
import matplotlib.pyplot as plt


data_dir = "data/"
file_name_train = 'dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1.xls'
input_size = 4


random_state = 0


# read the excel file with pandas
df_train = pd.read_excel(data_dir + file_name_train)

# convert it to numpy
data_train = df_train.to_numpy()

# input and output
x_train, y_train = (data_train[:, :input_size], data_train[:, input_size:])


# shape your prior belief via the choice of kernel
# Create kernel and define GPR
kernel = 4.0*RBF() + WhiteKernel()
#kernel = DotProduct() + WhiteKernel()
#kernel = 5.0*RationalQuadratic() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=random_state)

# Fit GPR model
gpr.fit(x_train, y_train)

file_name_test = 'dataWithDepthAndDiameter_TEST_new_1.xls'

# read the excel file with pandas
df_test = pd.read_excel(data_dir + file_name_test)

# convert it to numpy
data_test = df_test.to_numpy()

# input and output
x_test, y_test = (data_test[:, :input_size], data_test[:, input_size:])
# predict

y_predicted = gpr.predict(x_test)

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
plt.title("Gaussian Process Regression - Diameter")
plt.legend()
plt.show()

plt.xlabel('Testing Index')
plt.ylabel('Depth')
plt.plot(x_pos, y_predicted[:, 1], 'x', label='Predicted Data')
plt.plot(x_pos, y_test[:, 1], '^', label='Experimental Data')
plt.title("Gaussian Process Regression - Depth")
plt.legend()
plt.show()
