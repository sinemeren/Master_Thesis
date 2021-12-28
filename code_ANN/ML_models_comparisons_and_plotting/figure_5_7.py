import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct, RBF, RationalQuadratic
import matplotlib.pyplot as plt
from scipy import stats
from torch._C import StringType
import NNmodels
import seaborn as sns

# load the training data
data_dir = "data/"
file_name_train = 'dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1.xls'
input_size = 4

# read the excel file with pandas
df_train = pd.read_excel(data_dir + file_name_train)

# convert it to numpy
data_train = df_train.to_numpy()

# input and output
x_train, y_train = (data_train[:, :input_size], data_train[:, input_size:])


# load the test data
file_name_test = 'dataWithDepthAndDiameter_TEST_new_1.xls'

# read the excel file with pandas
df_test = pd.read_excel(data_dir + file_name_test)

# convert it to numpy
data_test = df_test.to_numpy()

# input and output
x_test, y_test = (data_test[:, :input_size], data_test[:, input_size:])

# Models


# 1. Linear Regression Model
model_LinearRegression = LinearRegression()
model_LinearRegression.fit(x_train, y_train)

# 2. Decision Tree Regression Model
max_depth = 5
model_DecisionTreeRegression = tree.DecisionTreeRegressor(max_depth=max_depth)
model_DecisionTreeRegression.fit(x_train, y_train)

# 3. Gaussion Process Regression Model
# Create kernel and define GPR
kernel = 4.0*RBF() + WhiteKernel()
random_state = 0
model_GaussionProcessRegression = GaussianProcessRegressor(
    kernel=kernel, random_state=random_state)
# Fit GPR model
model_GaussionProcessRegression.fit(x_train, y_train)


# 4. Polynomial Regression Model
polynomial_regression = PolynomialFeatures(degree=2)
x_poly = polynomial_regression.fit_transform(x_train)
polynomial_regression.fit(x_poly, y_train)
linear_regression = LinearRegression()
linear_regression.fit(x_poly, y_train)

# Predictions
y_predicted_Linear_Regression = model_LinearRegression.predict(x_test)
y_predicted_Gaussion_Process_Regression = model_GaussionProcessRegression.predict(
    x_test)
y_predicted_Decision_Tree_Regression = model_DecisionTreeRegression.predict(
    x_test)
y_predicted_Polynomial_Regression = linear_regression.predict(
    polynomial_regression.fit_transform(x_test))


y_predicted_383_new_full = NNmodels.predict("383_new_full")


# HL: 19, seed:10
y_predicted_rvfl = np.array([[67.77074683, 41.50598147], [44.26832232, 29.06933324], [117.66775221, 62.39726218], [58.74097228, 43.7977239], [
                             78.54598809, 41.18530487], [32.58012983, 22.36925911], [67.36255973, 42.01421088], [82.27646078, 45.356633]])

models = ['Linear Regr.', 'Gaussian Process Regr.',
          'Decision Tree Regr.', 'Polynomial Regr.', 'ANN Model 383', 'RVFL']

ae_diameter_linear, ae_depth_linear = abs(
    y_test[:, 0] - y_predicted_Linear_Regression[:, 0]), abs(y_test[:, 1] - y_predicted_Linear_Regression[:, 1])
ae_diameter_gaussian, ae_depth_gaussian = abs(
    y_test[:, 0] - y_predicted_Gaussion_Process_Regression[:, 0]), abs(y_test[:, 1] - y_predicted_Gaussion_Process_Regression[:, 1])
ae_diameter_decisionTree, ae_depth_decisionTree = abs(
    y_test[:, 0] - y_predicted_Decision_Tree_Regression[:, 0]), abs(y_test[:, 1] - y_predicted_Decision_Tree_Regression[:, 1])
ae_diameter_polynom, ae_depth_polynom = abs(
    y_test[:, 0] - y_predicted_Polynomial_Regression[:, 0]), abs(y_test[:, 1] - y_predicted_Polynomial_Regression[:, 1])
ae_diameter_predicted383, ae_depth_predicted383 = abs(
    y_test[:, 0] - y_predicted_383_new_full[:, 0]), abs(y_test[:, 1] - y_predicted_383_new_full[:, 1])
ae_diameter_RVFL, ae_depth_RVFL = abs(
    y_test[:, 0] - y_predicted_rvfl[:, 0]), abs(y_test[:, 1] - y_predicted_rvfl[:, 1])


args_diameter = (ae_diameter_linear, ae_diameter_gaussian,
                 ae_diameter_decisionTree, ae_diameter_polynom,
                 ae_diameter_predicted383, ae_diameter_RVFL)

data_diameter = np.concatenate(args_diameter)

args_depth = (ae_depth_linear, ae_depth_gaussian, ae_depth_decisionTree,
              ae_depth_polynom, ae_depth_predicted383, ae_depth_RVFL)

data_depth = np.concatenate(args_depth)

mlmodels = []
j = 0
for i in range((len(models)*len(ae_diameter_linear))):

    if(i % 8 == 0 and i > 6):
        j = j+1
    mlmodels.append(models[j])

df = pd.DataFrame(
    {'Models': mlmodels, 'Diameter': data_diameter, 'Depth': data_depth})
#df = df[['Group', 'Diameter', 'Depth']]

dd = pd.melt(df, id_vars=['Models'], value_vars=[
             'Diameter', 'Depth'], var_name='Predicted values')
s = sns.boxplot(x='Models', y='value', data=dd,
                hue='Predicted values')
plt.ylabel("Absolute Error in µm", fontsize=13)
plt.tick_params(axis='both', labelsize=12)
s.set_xlabel("Models", fontsize=13)
plt.legend(fontsize=13)
# plt.savefig("ml_nn_rvfl_comparison_both_boxplot.png", bbox_inches='tight',
#            pad_inches=0.1)
plt.show()
